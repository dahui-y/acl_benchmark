"""在 AccDiffusion（ECCV 2024）原生管线上跑我们的触发集 —— **去留闸门**。

为什么这是当前最该烧的 GPU（§9.5a 修正后的 E4）：
我们全部"比 AccDiffusion 强"的话到今天为止**一个正面数字都没有**，
全是推理。而三条路都卡在这一个数上：
    delta ≈ 0        -> 对手已在零，继续把 +0.10 压到 +0.05 是白干，转向；
    delta 明显 > 0   -> 我们在场上，且手握"1 次前向 vs N 次"与
                        "它在窗口注意力家族结构上跑不了"两张牌；
    比 ScaleDiff 还差 -> 卖点变成"它以重复为卖点却没测过，我们测了，
                        没解决" —— 杀伤力最大的一种。

**这是跨骨干比较**（AccDiffusion 建在 DemoFusion 上，原话见
accdiffusion_plus.py L1016；我们建在 ScaleDiff 上）。所以**不比绝对
计数，只比 Δdelta** —— 每一方都与**自己的 1024 基图**相减，量的是
"放大过程新加了多少重复"。这正是 §9.5c 那两张相反排序的表所暴露的
问题（同方法同分辨率，FID 排序完全相反）在结构上绕不过我们。

参数取 Readme.md 的官方配方（不是 argparse 的 default —— 那些开关
argparse 里全是 False，照抄会把他们的方法关掉，等于跑了个假 baseline）：
    --use_progressive_upscaling --use_skip_residual --use_multidiffusion
    --use_dilated_sampling --use_guassian --use_md_prompt --shuffle
其中 **use_md_prompt 就是 patch-content-aware prompt，是他们的核心
贡献本身**，关掉这个跑出来的数不能叫 AccDiffusion。

RNG 纪律：--shuffle 与 DemoFusion 家的窗口 jitter 都走 Python 标准库
random（demo_run.py 上栽过一次，md5 对照当场崩），故三处 RNG 全钉。

落盘格式与 parti_hi 完全一致（idx / prompt / files{"1024","4096"}），
vlm_count.py --delta 直接可吃。

    python scalediff_probe/acc_batch.py --plan          # 只看名单，不烧 GPU
    python scalediff_probe/acc_batch.py --n 12          # 先跑 12 条探路
    nohup python scalediff_probe/acc_batch.py > acc.log 2>&1 &
    python scalediff_probe/vlm_count.py --delta --hi $SD_OUT/parti_acc
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "help_code" / "AccDiffusion"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
# AccDiffusion Readme 的 negative prompt（**不是我们的 NEGATIVE**）——
# 他们的里面就带着 "duplicate"。用我们的会改掉他们的方法，
# 用他们的才是他们论文里的那个系统。这个选择必须写进论文的实验节。
ACC_NEGATIVE = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"


def pin(seed):
    """三处 RNG 全钉：shuffle 与窗口 jitter 走标准库 random。"""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    base = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    for snap in base.glob("*"):
        u = snap / "unet"
        if (u / "diffusion_pytorch_model.safetensors").exists():
            return False
        if list(u.glob("*.fp16.safetensors")):
            return True
    return False


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "parti_hi"),
                    help="读 prompt 名单的目录（与我们两臂同一批 idx）")
    ap.add_argument("--out", default=str(root / "parti_acc"))
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0,
                    help="只跑前 N 条（探路；0 = 全部）")
    ap.add_argument("--seed", type=int, default=77,
                    help="与 parti_hi / parti_v12 同 seed，保证配对")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--view-batch", type=int, default=16)
    ap.add_argument("--c", type=float, default=0.3,
                    help="他们的重复阈值，Readme 默认 0.3")
    ap.add_argument("--lowvram", action="store_true",
                    help="24GB 卡上 4096² 大概率要开")
    ap.add_argument("--our-negative", action="store_true",
                    help="改用我们的 NEGATIVE（默认用他们 Readme 的，"
                         "那才是他们论文里的系统）")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    hi = Path(a.hi)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    if a.idx:
        want = set(a.idx)
        rows = [r for r in rows if r["idx"] in want]
    if a.n:
        rows = rows[:a.n]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.jsonl"
    done = set()
    if mpath.exists():
        for line in mpath.open():
            r = json.loads(line)
            done.add(r["idx"])

    todo = [r for r in rows if r["idx"] not in done]
    neg = ACC_NEGATIVE
    if a.our_negative:
        from prompts import NEGATIVE
        neg = NEGATIVE

    print(f"名单 {len(rows)} 条，已完成 {len(done)}，待跑 {len(todo)}")
    print(f"seed={a.seed}  {a.size}²  steps={a.steps}  c={a.c}  "
          f"lowvram={a.lowvram}")
    print(f"negative = {neg[:60]!r}"
          f"{'（他们 Readme 的）' if not a.our_negative else '（我们的）'}\n")
    for r in todo[:5]:
        print(f"  [{r['idx']:>4}] {r['prompt'][:70]}")
    if len(todo) > 5:
        print(f"  ... 另 {len(todo) - 5} 条")
    if a.plan:
        print("\n--plan：没有加载模型。")
        return 0
    if not todo:
        print("\n没有待跑条目。")
        return 0

    import torch
    from accdiffusion_sdxl import AccDiffusionSDXLPipeline

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = AccDiffusionSDXLPipeline.from_pretrained(CKPT, **kw).to("cuda")

    # 官方 main 里原样传的 cross_attention_kwargs
    cak = {"edit_type": "visualize", "n_self_replace": 0.4,
           "n_cross_replace": {"default_": 1.0, "confetti": 0.8}}

    t_all = time.time()
    with mpath.open("a") as mf:
        for n, r in enumerate(todo, 1):
            idx, prompt = r["idx"], r["prompt"]
            pin(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            vb, images = a.view_batch, None
            while images is None:
                try:
                    images = pipe(
                        prompt, negative_prompt=neg,
                        generator=torch.Generator(
                            device="cuda").manual_seed(a.seed),
                        width=a.size, height=a.size,
                        view_batch_size=vb, stride=64,
                        cross_attention_kwargs=cak,
                        num_inference_steps=a.steps,
                        guidance_scale=7.5, multi_guidance_scale=7.5,
                        cosine_scale_1=3.0, cosine_scale_2=1.0,
                        cosine_scale_3=1.0, sigma=0.8,
                        # 以下七个是 Readme 的官方开关，**关掉就不是
                        # AccDiffusion 了**（argparse 的 default 全是 False）
                        use_guassian=True, use_multidiffusion=True,
                        use_skip_residual=True,
                        use_progressive_upscaling=True,
                        use_dilated_sampling=True, use_md_prompt=True,
                        shuffle=True,
                        multi_decoder=True, upscale_mode="bicubic_latent",
                        result_path=str(out / "_acc_tmp"),
                        seed=a.seed, c=a.c, lowvram=a.lowvram,
                        debug=False, save_attention_map=False)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if vb <= 1:
                        print(f"[{idx}] OOM 到 view_batch=1 仍失败，跳过。"
                              f"{'' if a.lowvram else ' 试试 --lowvram'}")
                        break
                    vb //= 2
                    print(f"[{idx}] OOM -> view_batch_size={vb} 重试")
            if images is None:
                continue
            dt = time.time() - t0

            files = {}
            for im in images:
                p = out / f"{idx:05d}_{im.width}.png"
                im.save(p)
                files[str(im.width)] = p.name
            if "1024" not in files:
                # --delta 拿 files["1024"] 当基图；没有它这一行是废的，
                # 与其静默落盘不如当场喊出来
                print(f"[{idx}] ⚠ 没有 1024 基图，只有 {sorted(files)} —— "
                      f"该行无法参与 Δdelta")
            peak = torch.cuda.max_memory_allocated() / 2**30
            mf.write(json.dumps({
                "idx": idx, "prompt": prompt, "seed": a.seed,
                "files": files, "sec": round(dt, 1),
                "peak_gb": round(peak, 2), "method": "accdiffusion_v1",
                "view_batch": vb, "c": a.c, "negative": neg,
                "stratum": r.get("stratum", "trigger"),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            el = (time.time() - t_all) / 60
            eta = el / n * (len(todo) - n)
            print(f"{n}/{len(todo)} [{idx:>4}] {dt:6.1f}s  {peak:4.1f} GB  "
                  f"{sorted(int(k) for k in files)}  "
                  f"已 {el:.0f} 分钟 剩约 {eta:.0f} 分钟")

    print(f"\n总计 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"\n下一步：\n  python scalediff_probe/vlm_count.py --delta --hi {out}")
    print("判读（预注册，跑之前就写死）：")
    print("  delta ≈ 0        -> 对手已在零，我们这条线判死，转向")
    print("  delta 明显 > 0   -> 我们在场上，主表照打")
    print("  比 ScaleDiff 差  -> 卖点变成'它以重复为卖点却从未测过'")


if __name__ == "__main__":
    sys.exit(main() or 0)
