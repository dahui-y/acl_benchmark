"""在 DemoFusion（CVPR 2024）上跑我们的 30 条 prompt。

DemoFusion 在论文里一件事顶三个用途（骨架 §1.2 / §6 / §3.15）：

  ① 律的一般性 —— 我们的核心主张是"每种让高分辨率算得动的机制都会造成重复"，
     但目前只在 ScaleDiff（注意力窗）上量过。DemoFusion 是完全不同的机制
     （重叠 patch + 渐进上采样），是这条谱系的公共节点（116 引用）。
     空视野->重复的关系在它身上复现，"律"才成立，否则只是 ScaleDiff 缺陷报告。
  ② 尺子的外部验证 —— 领域有已发表的共识：DemoFusion 重复严重
     （AccDiffusion 整篇为修它而写，HiWave 也点名）。我们的计数器若给它打出
     重复分 >= ScaleDiff，尺子就被一个外部、非自证的标签验证了。零标注。
  ③ 门阈值 τ=3/8 的新数据 —— 旧 30 条已被用过五轮，不能再做决定。

预注册的预测（写在跑之前）：
  P1  DemoFusion 的 lone 类 excess >= ScaleDiff 的 5.20
  P2  空视野比例与 |excess| 的正相关复现（trigger.py --batch 本目录）
  P3  empty 类 excess 仍 ~0（无主体 -> 无重复，两个必要条件的印证）

manifest 格式与 batch_run.py 完全一致，count_objects / trigger / compare 全部
直接可用。默认只跑 seed 77（每张 ~13-17 分钟，30 条约 7-8.5 小时，挂过夜）。

DemoFusion 论文自述可在 RTX 3090 24GB 上跑；OOM 时自动降 view_batch_size 重试。

    python scalediff_probe/demofusion_batch.py
    python scalediff_probe/demofusion_batch.py --only lone empty      # 先跑 10 条探路
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "help_code" / "DemoFusion"))
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[77])
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--view-batch", type=int, default=16,
                    help="OOM 时自动减半重试")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    out = Path(a.out) if a.out else root / "demofusion"
    out.mkdir(parents=True, exist_ok=True)

    items = [(i, c, s, p) for i, (c, s, p) in enumerate(PROMPTS)
             if a.only is None or c in a.only]
    print(f"{len(items)} 条 x {len(a.seeds)} seed -> {out}   {a.size}²\n")

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = DemoFusionSDXLPipeline.from_pretrained(CKPT, **kw).to("cuda")

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
                # 全局 RNG 也钉住 —— 和我们其余批次同一纪律；
                # DemoFusion 内部若有未传 generator 的采样，这样才可复现
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                vb = a.view_batch
                images = None
                while images is None:
                    try:
                        # gradio_demo.py 的论文默认参数：
                        # cosine_scale 3/1/1, sigma 0.8(demo 里 1.0), stride 64
                        images = pipe(
                            prompt, negative_prompt=NEGATIVE,
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                            height=a.size, width=a.size,
                            view_batch_size=vb, stride=64,
                            num_inference_steps=a.steps, guidance_scale=7.5,
                            cosine_scale_1=3.0, cosine_scale_2=1.0,
                            cosine_scale_3=1.0, sigma=1.0,
                            multi_decoder=True, show_image=False)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if vb <= 1:
                            print(f"{tag}  OOM 到 view_batch=1 仍失败，跳过")
                            break
                        vb //= 2
                        print(f"{tag}  OOM -> view_batch_size={vb} 重试")
                if images is None:
                    continue
                dt = time.time() - t0

                paths = {}
                for im in images:
                    p = out / f"{tag}_{im.width}.png"
                    im.save(p)
                    paths[im.width] = p.name
                peak = torch.cuda.max_memory_allocated() / 2**30
                mf.write(json.dumps({
                    "idx": idx, "cat": cat, "subject": subj, "prompt": prompt,
                    "seed": seed, "files": paths, "sec": round(dt, 1),
                    "peak_gb": round(peak, 2), "method": "demofusion",
                    "view_batch": vb,
                }, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"{tag:<22}{dt:7.1f}s  {peak:4.1f} GB  "
                      f"{[im.width for im in images]}")

    print(f"\n总计 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"下一步:  python scalediff_probe/count_objects.py --batch {out}")
    print(f"        python scalediff_probe/trigger.py --batch {out}")


if __name__ == "__main__":
    main()
