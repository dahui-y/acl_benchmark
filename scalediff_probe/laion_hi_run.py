"""LAION 上的 4096² 基线批跑 —— **去量问题本身，不再量它的代理**。

为什么这一步排在所有事情前面：

    到目前为止，"重复在 LAION 上罕见"这个结论**完全建立在 evf 上**，
    而 evf 是**预测因子**，不是问题本身。它还有两个已知缺陷，
    **都朝同一个方向压低它**：框系统性过冲（§5.4），以及 27% 的图
    检不出框被判为无定义。所以 2% 是 evf 的性质，不是重复的性质。
    **我们一次都没在 LAION 上量过 delta。**

**测量设计：把检测器的尺度漂移消掉。**

    朴素的 delta = count(4096) − count(基图) 里混着两样东西：
      (1) 真的多出了物体（重复）；
      (2) 检测器在高分辨率上看得更仔细 —— `scale_check` 实测残余漂移
          还有 **2.00** 个物体，tile=width/4 已经把它从 4.25 压下来但压不到 0。
    delta=5 于是既可能是重复，也可能是漂移。解法是让两边落在**同一个
    检测尺度**上：

        delta_content = count₁₀₂₄(把 4096 输出降采样回 1024) − count₁₀₂₄(基图)

    同一分辨率、同一套参数，漂移项被完全消掉，剩下的是"内容上真的多了东西"。
    重复出来的是**整个主体的副本**，降到 1024 照样看得见，
    所以这个量**只会低估不会虚报**。同时仍记 delta_raw（4096 上直接数），
    两者之差就是漂移的实测值 —— 一次跑，两个数，外加一个内部校验。

**抽样：四层，包含仪器盲区。**

    A  evf = 0            主体铺满，按律不该重复 —— 阴性对照
    B  0 < evf ≤ 0.25     轻度空视野
    C  evf > 0.25         重度空视野，按律最该重复
    D  **nbox = 0（无定义）**  检测器盲区。**必须抽** ——
       若盲区里也有重复，说明门会整批漏掉，这是方法的硬伤，不是小注脚。

本脚本只负责**生成**（GPU，长），计数与统计在 `laion_delta.py`（CPU/GPU 短）。
分开是为了断点续跑：4 小时的跑不能因为计数脚本改一行就重来。

    python scalediff_probe/laion_hi_run.py --plan-only     # 先看抽样表，不烧 GPU
    nohup python scalediff_probe/laion_hi_run.py > hi.log 2>&1 &
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                     # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"

STRATA = [
    ("A_evf0",   "evf == 0",           lambda r: r["nbox"] >= 1 and r["evf"] <= 0.0),
    ("B_low",    "0 < evf <= 0.25",    lambda r: r["nbox"] >= 1 and 0 < r["evf"] <= 0.25),
    ("C_high",   "evf > 0.25",         lambda r: r["nbox"] >= 1 and r["evf"] > 0.25),
    ("D_undef",  "nbox == 0（盲区）",   lambda r: r["nbox"] == 0),
]


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for snap in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
                 / "snapshots").glob("*"):
        u = snap / "unet"
        if (u / "diffusion_pytorch_model.safetensors").exists():
            return False
        if list(u.glob("*.fp16.safetensors")):
            return True
    return False


def pick_trigger_file(base, split, tag=None):
    """**按行数挑，不按文件名字母序挑** —— 早期 20 张探路轮留下的旧文件
    字母序会排在全量文件前面（caption_struct 就这么栽过一次）。"""
    if tag:
        p = base / f"trigger_{tag}.jsonl"
        if not p.exists():
            sys.exit(f"没有 {p}")
        return p
    cands = [c for c in base.glob(f"trigger_{split}_*.jsonl") if c.exists()]
    if not cands:
        sys.exit(f"没有 {base}/trigger_{split}_*.jsonl，先跑 trigger_select.py")
    n = {c: sum(1 for _ in c.open()) for c in cands}
    if len(cands) > 1:
        for c in sorted(cands, key=lambda c: -n[c]):
            print(f"  候选 {n[c]:>5} 行  {c.name}")
    return max(cands, key=lambda c: n[c])


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--out", default=str(root / "laion_hi"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--stage", type=int, default=2, help="2 -> 4096²；3 -> 8192²")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restart-ratio", type=float, default=0.4)
    ap.add_argument("--scale-factor", type=float, default=0.125)
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="抽样的随机种子。**与生成 seed 分开**，"
                         "换抽样不影响生成的可复现性")
    ap.add_argument("--plan-only", action="store_true",
                    help="只打印抽样表，不加载模型、不烧 GPU")
    a = ap.parse_args()

    base = Path(a.base)
    src = pick_trigger_file(base, a.split, a.tag)
    rows = [json.loads(l) for l in src.open()]
    byidx = {r["idx"]: r for r in rows}
    print(f"用 {src.name}   {len(rows)} 条")

    rng = random.Random(a.sample_seed)
    plan = []
    print(f"\n分层抽样（每层 {a.per_stratum} 条）：")
    for name, desc, pred in STRATA:
        pool = sorted([r["idx"] for r in rows if pred(r)])
        take = pool if len(pool) <= a.per_stratum else rng.sample(pool, a.per_stratum)
        plan += [(i, name) for i in sorted(take)]
        print(f"  {name:<9}{desc:<22} 池 {len(pool):>4}  取 {len(take):>3}"
              + ("   **池子不足，全取**" if len(pool) <= a.per_stratum else ""))
    print(f"合计 {len(plan)} 条")

    mani = {json.loads(l)["idx"]: json.loads(l)
            for l in (base / "manifest.jsonl").open()}
    missing = [i for i, _ in plan if i not in mani]
    if missing:
        sys.exit(f"这些 idx 没有基图：{missing[:10]}")

    res = 1024 * (2 ** a.stage)
    print(f"\n目标分辨率 {res}²   预计 {len(plan)*113/3600:.1f} 小时"
          f"（按 ScaleDiff 论文 113 s/张估）")
    if a.plan_only:
        print("\n--plan-only：没有加载模型。确认抽样表后去掉这个参数再跑。")
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.jsonl"
    done = set()
    if mpath.exists():
        for l in mpath.open():
            done.add(json.loads(l)["idx"])
        print(f"manifest 里已有 {len(done)} 条，跳过")
    todo = [(i, s) for i, s in plan if i not in done]
    if not todo:
        print("没有要跑的了。下一步：python scalediff_probe/laion_delta.py")
        return 0

    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    t_all = time.time()
    with mpath.open("a") as mf:
        for n, (idx, stratum) in enumerate(todo, 1):
            r = mani[idx]
            # 钉全局 RNG，不是只传 generator：ScaleDiff 放大阶段的
            # `noise = torch.randn_like(latents_LFM)` 没传 generator，
            # 走全局 RNG。不钉住，同 seed 两次跑的 4096² 是两张不同的图。
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                images = pipe(
                    r["prompt"], negative_prompt=NEGATIVE,
                    height=1024, width=1024,
                    generator=torch.Generator(device="cuda").manual_seed(a.seed),
                    num_inference_steps=a.steps, guidance_scale=7.5,
                    restart_ratio=a.restart_ratio, scale_factor=a.scale_factor,
                    upsample_stage=a.stage)
            except torch.cuda.OutOfMemoryError:
                print(f"\n{idx} OOM，跳过")
                torch.cuda.empty_cache()
                continue
            dt = time.time() - t0
            paths = {}
            for im in images:
                p = out / f"{idx:05d}_{im.width}.png"
                im.save(p)
                paths[im.width] = p.name
            mf.write(json.dumps({
                "idx": idx, "stratum": stratum, "prompt": r["prompt"],
                "split": r.get("split"), "seed": a.seed, "files": paths,
                "evf": byidx[idx]["evf"], "nbox": byidx[idx]["nbox"],
                "sec": round(dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            el = time.time() - t_all
            print(f"\r{n}/{len(todo)}  {el/60:.0f} 分钟已用  "
                  f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟  "
                  f"{dt:.0f}s {torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  end="", flush=True)
    print(f"\n完成，用时 {(time.time()-t_all)/60:.1f} 分钟")
    print("\n下一步（计数与统计，不用重跑生成）：\n"
          "    python scalediff_probe/laion_delta.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
