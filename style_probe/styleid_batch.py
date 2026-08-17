#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跑**真 StyleID**（作者自己的 diffusers 实现），批量，带断点续跑。

    为什么不用原版 ldm：那是 2022 年的代码，依赖 pytorch-lightning 1.x
    （`pytorch_lightning.utilities.distributed.rank_zero_only` 在 pl 2.x 已删），
    而 pl 1.x 不支持 Python 3.13。修它是无底洞，且修好只为跑一次。

    diffusers 实现是**同一批作者的同一个算法**，四个组件齐全：
      · DDIM inversion（style 和 content 各一次）
      · K/V 注入，content 的 q 与 style 的 q 按 γ 混合
      · attention temperature τ
      · 初始 latent 的 AdaIN
    README 说「量化指标请用原版」，那是**配置差异**（原版 SD1.4/50 步，
    diffusers 版默认 2.1-base/20 步），不是算法差异 —— 把配置对齐即可。

    原脚本一次只跑一对，800 对要起 800 次进程。这里做三件事：

    ① **只缓存真正用到的部分。** 注入用的是 content 的 q + style 的 k/v
       （见 __modify_self_attn_qkv：`q_c, k_s, v_s = attn_features_modify[...]`），
       所以 content 只留 q、style 只留 k/v，省掉三分之一到一半的显存/内存。
    ② 40 次 style 反演 + 20 次 content 反演 + 800 次生成，
       而不是 800×(2 反演 + 1 生成)。
    ③ 输出命名 `{style}__{content}.png`，与 protocol.py 的 tar/ 一致，
       所以 matrix.py / run_artfid.py 不用改一行就能吃。

    ⚠️ 两个消融开关是**在位者自己代码里的**，不是我们加的：
       --without_init_adain / --without_attn_injection。
       这是因果分解，不是相关。

用法：
    # 在它自带的 40 style × 20 content 上跑（可与发表值比的那一套）
    python style_probe/styleid_batch.py --out $SD_OUT/style/styleid/full
    # 消融
    python style_probe/styleid_batch.py --without_init_adain \
        --out $SD_OUT/style/styleid/no_adain
    python style_probe/styleid_batch.py --without_attn_injection \
        --out $SD_OUT/style/styleid/no_inject
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
SID = REPO / "help_code" / "StyleID" / "diffusers_implementation"


def load_sd(model_key, dtype, device="cuda"):
    """照 stable_diffusion.load_stable_diffusion 原样搬，只换 model_key。

    原函数把 '1.5' 硬编码成 `runwayml/stable-diffusion-v1-5` —— 那个 repo
    在 HF 上已经没了。本机缓存里是 `stable-diffusion-v1-5/stable-diffusion-v1-5`。
    """
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    for variant in ("fp16", None):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, torch_dtype=dtype, variant=variant, safety_checker=None)
            break
        except Exception as e:
            last = e
    else:
        raise last
    vae, tok, te, unet = pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet
    for m in (vae, te, unet):
        m.to(device)
    del pipe
    sch = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler",
                                        torch_dtype=dtype)
    return vae, tok, te, unet, sch


def _keep(feats, which, to_cpu):
    """只留需要的那一路：content 留 q（下标 0），style 留 k/v（下标 1,2）。"""
    out = {}
    for layer, per_t in feats.items():
        out[layer] = {}
        for t, qkv in per_t.items():
            if which == "q":
                v = (qkv[0].cpu() if to_cpu else qkv[0],)
            else:
                v = tuple(x.cpu() if to_cpu else x for x in qkv[1:3])
            out[layer][t] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnt", default=str(REPO / "help_code/StyleID/data/cnt"))
    ap.add_argument("--sty", default=str(REPO / "help_code/StyleID/data/sty"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get(
        "SD15_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    ap.add_argument("--sd_version", default="1.5")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--layers", nargs="+", type=int, default=[7, 8, 9, 10, 11])
    ap.add_argument("--without_init_adain", action="store_true")
    ap.add_argument("--without_attn_injection", action="store_true")
    ap.add_argument("--limit_sty", type=int, default=0, help="只跑前 N 个 style（调试）")
    a = ap.parse_args()

    import torch
    import cv2
    import numpy as np
    sys.path.insert(0, str(SID))
    os.chdir(SID)                      # 它的 utils 里有相对路径
    import run_styleid_diffusers as R
    from utils import normalize, denormalize
    from stable_diffusion import encode_latent, get_unet_layers

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    styles = sorted(p for p in Path(a.sty).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    conts = sorted(p for p in Path(a.cnt).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if a.limit_sty:
        styles = styles[:a.limit_sty]
    print(f"{len(styles)} style × {len(conts)} content = "
          f"{len(styles)*len(conts)} 张 → {out}")
    print(f"gamma={a.gamma} T={a.T} layers={a.layers} steps={a.ddim_steps} "
          f"sd={a.sd_version}  init_adain={not a.without_init_adain} "
          f"attn_inject={not a.without_attn_injection}")

    dtype = torch.float16
    vae, tok, te, unet, sch = load_sd(a.model, dtype)
    sch.set_timesteps(a.ddim_steps)

    _, attn = get_unet_layers(unet)
    ok = [i for i in a.layers if i < len(attn) and attn[i] is not None]
    if len(ok) != len(a.layers):
        sys.exit(f"!! 注入层不合法：给了 {a.layers}，可用的只有 "
                 f"{[i for i,x in enumerate(attn) if x is not None]}")
    res = {}
    for i in ok:
        res[i] = "?"
    print(f"  注入 {len(ok)} 层：{ok}（up_blocks[{ok[0]//3}] 起）")

    cfg = SimpleNamespace(gamma=a.gamma, T=a.T, layers=a.layers,
                          ddim_steps=a.ddim_steps, sd_version=a.sd_version,
                          without_init_adain=a.without_init_adain,
                          without_attn_injection=a.without_attn_injection,
                          cnt_fn="", sty_fn="", save_dir=str(out))
    W = R.style_transfer_module(unet, vae, te, tok, sch, cfg,
                                style_transfer_params={"gamma": a.gamma,
                                                       "tau": a.T,
                                                       "injection_layers": a.layers})

    def invert(img_path, text=None):
        im = cv2.imread(str(img_path))[:, :, ::-1]
        lat = encode_latent(normalize(im).to(device=vae.device, dtype=dtype), vae)
        kw = W.get_text_condition(text)
        W.trigger_get_qkv, W.trigger_modify_qkv = True, False
        _, lats = W.invert_process(lat, denoise_kwargs=kw)
        return lats[-1]

    # ── content 只反演一次，缓存 q（放 CPU；20 张 × 20 步大约几 GB）
    t0 = time.time()
    cq, clat = [], []
    for p in conts:
        lat = invert(p)
        cq.append(_keep(W.attn_features, "q", to_cpu=True))
        clat.append(lat)
        W.attn_features = {k: {} for k in W.attn_features}
    print(f"  content 反演完毕 {len(conts)} 张，{time.time()-t0:.0f}s")

    kw_gen = W.get_text_condition(None)
    done = skipped = 0
    for si, sp in enumerate(styles):
        want = [out / f"{sp.stem}__{cp.stem}.png" for cp in conts]
        if all(f.exists() for f in want):
            skipped += len(want); continue
        slat = invert(sp)
        skv = _keep(W.attn_features, "kv", to_cpu=False)
        W.attn_features = {k: {} for k in W.attn_features}

        for ci, cp in enumerate(conts):
            f = out / f"{sp.stem}__{cp.stem}.png"
            if f.exists():
                skipped += 1; continue
            # content q（CPU）搬回 GPU，与 style 的 k/v 组成注入表
            W.attn_features_modify = {}
            for layer in skv:
                W.attn_features_modify[layer] = {}
                for t in skv[layer]:
                    q = cq[ci][layer][t][0].to(slat.device)
                    W.attn_features_modify[layer][t] = (q, *skv[layer][t])
            W.trigger_get_qkv = False
            W.trigger_modify_qkv = not a.without_attn_injection

            cl = clat[ci]
            if a.without_init_adain:
                lat_cs = cl
            else:
                lat_cs = ((cl - cl.mean(dim=(2, 3), keepdim=True))
                          / (cl.std(dim=(2, 3), keepdim=True) + 1e-4)
                          * slat.std(dim=(2, 3), keepdim=True)
                          + slat.mean(dim=(2, 3), keepdim=True))
            imgs, _ = W.reverse_process(lat_cs, denoise_kwargs=kw_gen)
            from PIL import Image
            Image.fromarray(denormalize(imgs[-1])[0]).save(f)
            done += 1
        el = time.time() - t0
        tot = len(styles) * len(conts)
        print(f"  [{si+1}/{len(styles)}] {sp.stem}  已生成 {done}（跳过 {skipped}）"
              f"  {el/60:.0f}m 已用 / "
              f"{(tot-done-skipped)*el/max(done,1)/60:.0f}m 剩余", flush=True)

    print(f"\n生成 {done} 张，跳过 {skipped} 张 → {out}")
    print(f"\n下一步：")
    print(f"  # ArtFID（三边张数要一致，先造对齐目录）")
    print(f"  python style_probe/styleid_batch.py --help  # 见 DESIGN.md 的记录要求")


if __name__ == "__main__":
    main()
