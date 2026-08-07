"""第一次跑 ScaleDiff。只回答三个问题：能不能跑通、显存峰值、单图秒数。

不做任何方法上的改动，也不做评测。这一步的产出就是三个数和三张图。

为什么分两次跑（stage=1 再 stage=2）而不是直接 stage=2：
    4096² 在 24 GB 上能不能过是未知数（论文用 A6000 48 GB）。如果直接跑
    stage=2 然后在最后一级 OOM，前面 1024²/2048² 的结果和计时也一起丢了。
    分开跑，2048² 那一档先落袋。

为什么自动加 variant="fp16"：
    ScaleDiff 原代码只传 torch_dtype=torch.float16，没传 variant。我们缓存里的
    SDXL 是 fp16 变体（6.7 GB），不加 variant 会在 from_pretrained 报
    FileNotFoundError。这里先探一下缓存里有什么，再决定传不传。

    python scalediff_probe/run_one.py
    python scalediff_probe/run_one.py --stages 1          # 只跑到 2048²
    python scalediff_probe/run_one.py --offload           # 显存不够时
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
# pipeline_scalediff_sdxl.py 里是 `from attn_scalediff_sdxl import ...`（同目录导入），
# 所以必须把这个目录放进 sys.path，否则 ImportError。
sys.path.insert(0, str(SDXL_DIR))

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

# 一条内容丰富、有大面积背景的 prompt —— 作者自己承认背景区还有 repetitive
# artifacts，第一张图就往那里看。
PROMPT = ("a photograph of a lone hiker standing on a rocky ridge, "
          "vast forested valley and distant snow mountains behind, golden hour")


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    unet = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    for snap in unet.glob("*"):
        u = snap / "unet"
        if (u / "diffusion_pytorch_model.safetensors").exists():
            return False
        if list(u.glob("*.fp16.safetensors")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2],
                    help="upsample_stage 列表；1 -> 2048², 2 -> 4096²")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restart-ratio", type=float, default=0.4)   # 论文 tau=400
    ap.add_argument("--scale-factor", type=float, default=0.125)  # 论文 SDXL 值
    ap.add_argument("--offload", action="store_true",
                    help="用 enable_model_cpu_offload 换显存（会更慢）")
    ap.add_argument("--out", default=os.environ.get("SD_OUT", "./scalediff_out"))
    a = ap.parse_args()

    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    out = Path(a.out) / "run_one"
    out.mkdir(parents=True, exist_ok=True)

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
        print("缓存里只有 fp16 变体 -> 自动加 variant='fp16'（原代码没有这一句）")

    t0 = time.time()
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw)
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    print(f"模型加载 {time.time() - t0:.1f}s\n")

    prev = 0.0
    for st in a.stages:
        res = 1024 * (2 ** st)
        torch.cuda.empty_cache()
        # 见 batch_run.py 里的说明：放大阶段的噪声走全局 RNG
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            images = pipe(
                PROMPT,
                negative_prompt=NEG,
                height=1024, width=1024,
                generator=torch.Generator(device="cuda").manual_seed(a.seed),
                num_inference_steps=a.steps,
                guidance_scale=7.5,
                restart_ratio=a.restart_ratio,
                scale_factor=a.scale_factor,
                upsample_stage=st,
            )
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"stage={st} ({res}²)  OOM，峰值 {peak:.1f} GB。"
                  f"试试 --offload，或只跑 --stages 1")
            break
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30

        # 一次调用返回 [1024², 2048², ...] 全部中间分辨率
        for img in images:
            img.save(out / f"s{a.seed}_stage{st}_{img.width}.png")

        # stage=2 的耗时包含了 stage=1 的部分，做差得到最后一级的增量
        extra = f"  (最后一级增量 ~{dt - prev:.0f}s)" if prev else ""
        print(f"stage={st}  -> {res}²   {dt:6.1f}s   峰值显存 {peak:5.1f} GB   "
              f"输出 {[im.width for im in images]}{extra}")
        prev = dt

    print(f"\n图在 {out}")


if __name__ == "__main__":
    main()
