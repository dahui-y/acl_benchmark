"""跑 30 条 prompt，每条存 1024² / 2048² / 4096²，同 seed。

一次 pipe() 调用（upsample_stage=2）就返回全部三档，所以不需要跑三遍。
seed 77 那一张实测 74.7s，30 条约 37 分钟。

断点续跑：已经存在的输出直接跳过，可以随时 Ctrl-C 再接上。

    python scalediff_probe/batch_run.py
    python scalediff_probe/batch_run.py --only lone empty      # 只跑某几类
    python scalediff_probe/batch_run.py --seeds 77 1234        # 多个 seed
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

from prompts import PROMPTS, NEGATIVE          # noqa: E402

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
    ap.add_argument("--out", default=str(Path(os.environ.get("SD_OUT", "./scalediff_out")) / "batch"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[77])
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restart-ratio", type=float, default=0.4)
    ap.add_argument("--scale-factor", type=float, default=0.125)
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些类别")
    a = ap.parse_args()

    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    items = [(i, c, s, p) for i, (c, s, p) in enumerate(PROMPTS)
             if a.only is None or c in a.only]
    print(f"{len(items)} 条 prompt x {len(a.seeds)} seed -> {out}\n")

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()

    manifest_path = out / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            r = json.loads(line)
            done.add((r["idx"], r["seed"]))
        print(f"manifest 里已有 {len(done)} 条，跳过\n")

    t_all = time.time()
    with manifest_path.open("a") as mf:
        for idx, cat, subj, prompt in items:
            for seed in a.seeds:
                tag = f"{idx:02d}_{cat}_s{seed}"
                if (idx, seed) in done:
                    continue
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                try:
                    images = pipe(
                        prompt,
                        negative_prompt=NEGATIVE,
                        height=1024, width=1024,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        num_inference_steps=a.steps,
                        guidance_scale=7.5,
                        restart_ratio=a.restart_ratio,
                        scale_factor=a.scale_factor,
                        upsample_stage=a.stage,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"{tag}  OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                dt = time.time() - t0
                paths = {}
                for im in images:
                    p = out / f"{tag}_{im.width}.png"
                    im.save(p)
                    paths[im.width] = p.name
                mf.write(json.dumps({
                    "idx": idx, "cat": cat, "subject": subj, "prompt": prompt,
                    "seed": seed, "files": paths, "sec": round(dt, 1),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"{tag:<22}{dt:6.1f}s  "
                      f"{torch.cuda.max_memory_allocated() / 2**30:4.1f} GB  {prompt[:52]}")

    print(f"\n总计 {(time.time() - t_all) / 60:.1f} 分钟。manifest: {manifest_path}")


if __name__ == "__main__":
    main()
