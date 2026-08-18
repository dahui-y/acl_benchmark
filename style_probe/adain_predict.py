#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaIN 的逐对收益/代价，**能不能在生成之前预测出来**。

    这是整个方向的生死判据。oracle 上界 0.97 ArtFID 只有在存在可实现规则时
    才有意义；否则它永远是 oracle。

    预测量不是从一堆候选里挑出来的，而是**这个算子的定义本身**：

        latent_cs = (z_c − μ_c)/σ_c · σ_s + μ_s          (run_styleid_diffusers.py)

    AdaIN 做的就是把 content latent 的逐通道均值/方差对齐到 style latent。
    所以它的作用强度直接等于两个 latent 的统计距离 —— 而那在生成前就能算，
    只需要一次 DDIM 反演（style 40 张、content 20 张，各一次，约 3 分钟），
    之后 800 对的判别是毫秒级的向量运算。

    这与 §七 那次失败的命名尝试有本质区别：那次是在 4 个手造指标 × 3 个尺度
    里找相关，属于钓鱼；这次的预测量是算子公式里的那两个量，没有选择空间。

判据（事先写死）：
    对「AdaIN 逐对收益」或「逐对代价」，至少一个方向 **|r| ≥ 0.5**，
    才存在可实现的规则。低于此 → oracle 无法落地，方向停。

用法：
    python style_probe/adain_predict.py --root $SD_OUT/style/styleid
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SID = REPO / "help_code" / "StyleID" / "diffusers_implementation"


def latent_stats(args, paths):
    """每张图反演一次，返回 (μ, σ)，各 [N, 4]（latent 的 4 个通道）。

    必须用**反演后**的 latent，不是 VAE 编码的 latent —— AdaIN 作用在
    invert_process 的最后一个 latent 上（见 run_styleid_diffusers.py 主流程）。
    """
    import torch
    import cv2
    sys.path.insert(0, str(SID))
    sys.path.insert(0, str(REPO / "style_probe"))
    os.chdir(SID)
    import run_styleid_diffusers as R
    from utils import normalize
    from stable_diffusion import encode_latent
    from styleid_batch import load_sd
    from types import SimpleNamespace

    dtype = torch.float16
    vae, tok, te, unet, sch = load_sd(args.model, dtype)
    sch.set_timesteps(args.ddim_steps)
    cfg = SimpleNamespace(gamma=0.75, T=1.5, layers=[7, 8, 9, 10, 11],
                          ddim_steps=args.ddim_steps, sd_version="1.4",
                          without_init_adain=False, without_attn_injection=False,
                          cnt_fn="", sty_fn="", save_dir="")
    W = R.style_transfer_module(unet, vae, te, tok, sch, cfg,
                                style_transfer_params={"gamma": 0.75, "tau": 1.5,
                                                       "injection_layers": [7, 8, 9, 10, 11]})
    R.tokenizer, R.text_encoder, R.vae = tok, te, vae
    R.unet, R.scheduler, R.device, R.dtype = unet, sch, "cuda", dtype
    R.guidance_scale, R.unet_wrapper, R.ddim_steps = 0., W, args.ddim_steps

    mus, sds = [], []
    for p in paths:
        im = cv2.imread(str(p))[:, :, ::-1]
        z = encode_latent(normalize(im).to(device=vae.device, dtype=dtype), vae)
        kw = W.get_text_condition(None)
        W.trigger_get_qkv, W.trigger_modify_qkv = False, False   # 不用存 qkv，省内存
        _, lats = W.invert_process(z, denoise_kwargs=kw)
        zz = lats[-1].float()
        mus.append(zz.mean(dim=(2, 3)).flatten().cpu().numpy())
        sds.append(zz.std(dim=(2, 3)).flatten().cpu().numpy())
        W.attn_features = {k: {} for k in W.attn_features}
    return np.stack(mus), np.stack(sds)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sty", default=str(REPO / "help_code/StyleID/data/sty"))
    ap.add_argument("--cnt", default=str(REPO / "help_code/StyleID/data/cnt"))
    ap.add_argument("--model", default=os.environ.get(
        "SD14_PATH", "CompVis/stable-diffusion-v1-4"))
    ap.add_argument("--ddim_steps", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.root)

    z = {n: np.load(root / n / "matrix.npz", allow_pickle=True)
         for n in ("full", "no_adain")}
    names_s = [Path(str(x)).stem for x in z["full"]["sty"]]
    names_c = [Path(str(x)).stem for x in z["full"]["cnt"]]
    ben = z["full"]["style_gain"] - z["no_adain"]["style_gain"]     # AdaIN 的收益
    cost = z["full"]["content_norm"] - z["no_adain"]["content_norm"]  # AdaIN 的代价

    def _find(d, stem):
        for ext in (".png", ".jpg", ".jpeg"):
            if (Path(d) / f"{stem}{ext}").exists():
                return Path(d) / f"{stem}{ext}"
        sys.exit(f"!! 找不到 {stem} in {d}")

    sp = [_find(a.sty, s) for s in names_s]
    cp = [_find(a.cnt, c) for c in names_c]
    print(f"反演 {len(sp)} style + {len(cp)} content（各一次）…")
    mu_s, sd_s = latent_stats(a, sp)
    mu_c, sd_c = latent_stats(a, cp)
    print(f"latent 统计：μ {mu_s.shape}  σ {sd_s.shape}")

    ns, nc = len(sp), len(cp)
    # 预测量：全部直接来自 AdaIN 的公式，没有可挑的空间
    P = {
        "d_mu":     np.linalg.norm(mu_s[:, None, :] - mu_c[None, :, :], axis=2),
        "d_sigma":  np.linalg.norm(sd_s[:, None, :] - sd_c[None, :, :], axis=2),
        "log_ratio": np.abs(np.log(sd_s[:, None, :] / sd_c[None, :, :])).mean(2),
        # 合起来：AdaIN 对 latent 的实际改动量（把公式代进去的一阶量）
        "shift":    (np.linalg.norm(mu_s[:, None, :] - mu_c[None, :, :], axis=2)
                     + np.linalg.norm(sd_s[:, None, :] - sd_c[None, :, :], axis=2)),
    }

    print(f"\n{'预测量':<12} {'vs 收益 r':>12} {'vs 代价 r':>12}")
    best = 0.0
    res = {}
    for k, v in P.items():
        rb = np.corrcoef(v.ravel(), ben.ravel())[0, 1]
        rc = np.corrcoef(v.ravel(), cost.ravel())[0, 1]
        res[k] = dict(benefit=float(rb), cost=float(rc))
        print(f"{k:<12} {rb:>12.3f} {rc:>12.3f}")
        best = max(best, abs(rb), abs(rc))

    print(f"\n逐对样本量 n = {ns*nc}；最强 |r| = {best:.3f}")
    if best >= 0.5:
        print("→ **存在可实现的规则。** 下一步：定阈值，跑实测（非 oracle）ArtFID。")
    else:
        print("→ **预测不出来。** oracle 的 0.97 落不了地。")
        print("   按事先写死的判据，这条路到此为止 —— 不要再换预测量硬找。")
    # 预测量矩阵要存下来 —— 下一步定阈值时不该再反演一次
    np.savez(root / "adain_predict.npz",
             benefit=ben, cost=cost,
             mu_s=mu_s, sd_s=sd_s, mu_c=mu_c, sd_c=sd_c,
             sty=names_s, cnt=names_c, **P)
    (root / "adain_predict.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"→ {root/'adain_predict.npz'}")
    print(f"→ {root/'adain_predict.json'}")


if __name__ == "__main__":
    main()
