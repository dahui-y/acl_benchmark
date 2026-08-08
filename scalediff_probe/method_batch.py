"""把 v1 的干预跑到 30 条 prompt 上。

v1 只有一张图。门槛（跑之前定的）是 lone 类 4096² 的 delta 至少减半：
基线 +6.20（5/5 条有幻影）-> ≤3。一张图证不了这个，必须整批。

和 batch_run.py 的唯一区别是装了 BlendCrossAttn；seed、步数、RNG 钉法、
manifest 格式全都一样，所以 count_objects.py --batch <这个目录> 可以直接跑，
两边的表逐行可比。

去主体 prompt 由 subject_phrases.strip_subject 自动生成，不再手写。
empty 那五条 head=None，主体图建不起来，自动退化成原版 —— 它们在表里是
"干预不该动的对照"，delta 应当和基线持平。

    python scalediff_probe/method_batch.py --s 1
    python scalediff_probe/method_batch.py --s 0        # 自检：应与 batch 一致
    python scalediff_probe/method_batch.py --s 1 --only lone
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "help_code" / "ScaleDiff" / "SDXL"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROMPTS, NEGATIVE                       # noqa: E402
from subject_phrases import HEADS, strip_subject            # noqa: E402
from method_v0 import subject_token_ids                     # noqa: E402
from method_v1 import BlendGate, BlendCrossAttn             # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for snap in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots").glob("*"):
        u = snap / "unet"
        if (u / "diffusion_pytorch_model.safetensors").exists():
            return False
        if list(u.glob("*.fp16.safetensors")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, default=1.0, help="混合强度；0 = 原版")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[77])
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restart-ratio", type=float, default=0.4)
    ap.add_argument("--scale-factor", type=float, default=0.125)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--norm", default="rel", choices=["rel", "minmax"],
                    help="rel = 相对均匀分布的倍数（v2）；minmax = v1 的逐图归一化")
    ap.add_argument("--k", type=float, default=1.0,
                    help="rel 模式的界：注意力高于全图平均 k 倍的位置用完整 prompt")
    ap.add_argument("--width", type=float, default=0.5, help="rel 模式过渡带宽度")
    a = ap.parse_args()

    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    out = Path(a.out) if a.out else root / f"method_batch_s{a.s:g}"
    out.mkdir(parents=True, exist_ok=True)

    items = [(i, c, s, p, HEADS[i]) for i, (c, s, p) in enumerate(PROMPTS)
             if a.only is None or c in a.only]
    print(f"{len(items)} 条 x {len(a.seeds)} seed,  strength={a.s:g}  -> {out}\n")

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()

    # 原版的 attn2 处理器，留着做退回。
    # 不能用 set_default_attn_processor()：第一次 pipe() 之后 attn1 上装的是
    # ScaleDiff 的 AttnProcessor2_0_local，diffusers 见到非默认处理器就抛
    # ValueError。而且我们本来也只想还原 attn2，attn1 归 ScaleDiff 管。
    BASE_ATTN2 = {k: v for k, v in pipe.unet.attn_processors.items()
                  if k.endswith("attn2.processor")}

    # noise_pred_step 只包一次；用可变 holder 指向当前 prompt 的 gate
    holder = {"gate": None}
    orig_step = pipe.noise_pred_step

    def patched(latents, t, *args, **kw_):
        g = holder["gate"]
        if g is not None:
            ph = 1 if latents.shape[-1] <= 128 else 2
            if ph == 2 and g.phase == 1:
                g.finalize()
            g.phase = ph
        return orig_step(latents, t, *args, **kw_)

    pipe.noise_pred_step = patched

    manifest_path = out / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            r = json.loads(line)
            done.add((r["idx"], r["seed"]))
        print(f"manifest 里已有 {len(done)} 条，跳过\n")

    print(f"norm={a.norm}  k={a.k}  width={a.width}\n")
    print(f"{'tag':<22}{'head':<13}{'cov':>7}{'过渡带':>8}{'sec':>8}{'GB':>6}")
    t_all = time.time()
    with manifest_path.open("a") as mf:
        for idx, cat, subj, prompt, head in items:
            for seed in a.seeds:
                tag = f"{idx:02d}_{cat}_s{seed}"
                if (idx, seed) in done:
                    continue

                alt_prompt, removed = strip_subject(prompt, head)
                tok_ids = subject_token_ids(pipe, prompt, head) if head else []
                gate = BlendGate(tok_ids, strength=a.s, norm=a.norm,
                                 k=a.k, width=a.width)
                holder["gate"] = gate

                if tok_ids and a.s > 0:
                    pe, npe, _, _ = pipe.encode_prompt(
                        prompt=alt_prompt, device="cuda", num_images_per_prompt=1,
                        do_classifier_free_guidance=True, negative_prompt=NEGATIVE)
                    gate.alt = torch.cat([npe, pe])
                    procs = dict(pipe.unet.attn_processors)
                    for k_ in procs:
                        if k_.endswith("attn2.processor"):
                            procs[k_] = BlendCrossAttn(gate)
                    pipe.unet.set_attn_processor(procs)
                else:
                    # 没有主体词，或 s=0：只把 attn2 还原成原版，attn1 不动
                    procs = dict(pipe.unet.attn_processors)
                    procs.update(BASE_ATTN2)
                    pipe.unet.set_attn_processor(procs)
                    holder["gate"] = None

                # 放大阶段的噪声走全局 RNG（pipeline:547 没传 generator）
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    images = pipe(
                        prompt, negative_prompt=NEGATIVE, height=1024, width=1024,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        num_inference_steps=a.steps, guidance_scale=7.5,
                        restart_ratio=a.restart_ratio, scale_factor=a.scale_factor,
                        upsample_stage=a.stage)
                except torch.cuda.OutOfMemoryError:
                    print(f"{tag}  OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                dt = time.time() - t0

                cov, band = -1.0, -1.0
                if gate is not None and gate.map is not None:
                    cov, _lo, band = gate.stats()

                paths = {}
                for im in images:
                    p = out / f"{tag}_{im.width}.png"
                    im.save(p)
                    paths[im.width] = p.name
                # 把混合权重图存下来。诊断 26_bridge / 27_watch 变差要用它：
                # 多出来的物体落在过渡带上（-> 接缝伪影，该改混合方式）
                # 还是落在远景里（-> 去主体 prompt 泄漏，该改摘除规则）。
                # 两个解释指向完全不同的修法，必须先分开。
                if gate is not None and gate.map is not None:
                    import numpy as _np
                    _np.save(out / f"{tag}_map.npy",
                             gate.map.float().cpu().numpy())
                peak = torch.cuda.max_memory_allocated() / 2**30
                mf.write(json.dumps({
                    "idx": idx, "cat": cat, "subject": subj, "prompt": prompt,
                    "seed": seed, "files": paths, "sec": round(dt, 1),
                    "peak_gb": round(peak, 2), "strength": a.s, "head": head,
                    "alt_prompt": alt_prompt, "removed": removed,
                    "applied": bool(tok_ids and a.s > 0),
                    "cov": round(cov, 4), "band": round(band, 4),
                    "norm": a.norm, "k": a.k, "width": a.width,
                }, ensure_ascii=False) + "\n")
                mf.flush()
                covs = f"{cov:6.1%}" if cov >= 0 else "     -"
                bs = f"{band:7.1%}" if band >= 0 else "      -"
                print(f"{tag:<22}{str(head):<13}{covs}{bs}{dt:8.1f}{peak:6.1f}")

    print(f"\n总计 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"下一步:  python scalediff_probe/count_objects.py --batch {out}")


if __name__ == "__main__":
    main()
