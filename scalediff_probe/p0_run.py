"""P0：三臂批跑，回答"局部保真度的缺口住在边界还是上下文"。

────────────────────────────────────────────────────────────────────────
三个臂
────────────────────────────────────────────────────────────────────────
    A  npa      ScaleDiff 的评测配置（附录 B.1 那个开关**没开**，与他们一致）
    B  shift    NPA + Query Window Random Shifting（他们写了、没用、没实现）
    C  md       MultiDiffusion 注意力（他们消融里七列全赢、只输时间的那行）

判据（§10.7 修订版，写在跑之前）：令 g = C−A、b = B−A，都取**配对**差值。
    g 方向对且 |g| > 2σ_d  -> 缺口为真，进 P1；
    |g| <= 2σ_d            -> **不确定，不是判死**，按报出的 n 补样本；
    g 反向且 |g| > 2σ_d    -> 方向判死。
再看 b/g：
    b >= 0.6g -> 缺口主要住在**边界伪影**，他们那个没开的开关吃掉大半，
                 天花板低，要重估这个方向值不值得做；
    b <= 0.3g -> 住在**上下文不足**，那才是可攻的机制 —— 要的就是这一格。
闸门只看 KIDp；ISp 是佐证列（实测要 n>1615 才看得见发表级的差，§10.12）。

────────────────────────────────────────────────────────────────────────
两个刻意的设计
────────────────────────────────────────────────────────────────────────
**① 按 prompt 交错，不是一个臂跑完再跑下一个。**
三十几小时的跑挂在共享服务器上，中途死是常态。逐臂跑，死在第 20 小时
就只有一个完整的臂和两个空目录 —— 一无所有。交错跑，死在任何时刻都有
前 K 条的**三臂齐全**数据，配对比较当场可做，只是 n 小一点。

**② prompt 取 tune split 的随机 300 条，不用 laion_hi 那 120 条分层样本。**
那 120 条是按空视野比例分层抽的，为"物体重复"这个**罕见**失效设计的。
现在量的是局部保真度 —— **每张图都有的属性**，要的是无偏随机样本。
`eval` split 一次都不碰（方法冻结前不许回看，§7.1.5）。

────────────────────────────────────────────────────────────────────────
    python scalediff_probe/p0_run.py --plan          # 只看名单，不烧 GPU
    python scalediff_probe/attn_variants.py --selftest   # 先过这个
    nohup python scalediff_probe/p0_run.py > p0.log 2>&1 &
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
from prompts import NEGATIVE                                    # noqa: E402
from std_table import prompt_items                              # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
ARMS = ("npa", "shift", "md")


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for s in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
              / "snapshots").glob("*"):
        u = s / "unet"
        if not (u / "diffusion_pytorch_model.safetensors").exists() \
                and list(u.glob("*.fp16.safetensors")):
            return True
    return False


def pick_prompts(prompts, split, n, seed, out):
    """抽一次、写盘、之后一律读盘 —— 三个臂和每次续跑都必须是同一批。"""
    f = Path(out) / "plan.json"
    if f.exists():
        obj = json.loads(f.read_text())
        print(f"沿用已有名单 {f.name}：{len(obj['idx'])} 条 "
              f"(split={obj['split']}, sample_seed={obj['sample_seed']})")
        return obj["idx"], obj["prompt"]
    items = prompt_items(prompts)
    pool = [i for i, x in enumerate(items) if x.get("split") == split]
    if len(pool) < n:
        print(f"⚠ {split} split 只有 {len(pool)} 条，全取")
    take = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    obj = {"prompts_file": str(prompts), "split": split, "sample_seed": seed,
           "idx": take, "prompt": {str(i): items[i]["prompt"] for i in take}}
    Path(out).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(f"抽样写入 {f}：{len(take)} 条")
    return take, obj["prompt"]


def done_of(d):
    m = Path(d) / "manifest.jsonl"
    return ({json.loads(l)["idx"] for l in m.open() if l.strip()}
            if m.exists() else set())


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "p0"))
    ap.add_argument("--prompts", default=str(root / "eval_prompts.json"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--stage", type=int, default=2, help="2 -> 4096²")
    ap.add_argument("--restart-ratio", type=float, default=0.4)
    ap.add_argument("--scale-factor", type=float, default=0.125)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    idxs, pmap = pick_prompts(a.prompts, a.split, a.n, a.sample_seed, out)
    dirs = {m: out / m for m in a.arms}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    done = {m: done_of(d) for m, d in dirs.items()}

    # 交错任务表：(idx, arm)，按 idx 外层、arm 内层 —— 中途死也三臂齐全
    todo = [(i, m) for i in idxs for m in a.arms if i not in done[m]]
    res = 1024 * (2 ** a.stage)
    est = {"npa": 113, "shift": 118, "md": 239}      # 论文值，仅供估时
    hrs = sum(est[m] for _, m in todo) / 3600
    print(f"\n目标 {res}²   臂 {a.arms}   名单 {len(idxs)} 条")
    for m in a.arms:
        print(f"  {m:<6} 已有 {len(done[m]):>4} / {len(idxs)}  -> {dirs[m]}")
    print(f"待跑 {len(todo)} 个任务，按论文时长估 **{hrs:.1f} 小时**"
          f"（我们的卡不同，真实时长以运行中打印为准）")
    print("\n判据（写在跑之前）：g=C−A、b=B−A，配对差值；"
          "\n  |g|>2σ_d 且方向对 -> 缺口为真；|g|<=2σ_d -> 不确定（补样本）；"
          "反向 -> 判死。"
          "\n  b>=0.6g -> 住在边界（天花板低）；b<=0.3g -> 住在上下文（可攻）。"
          "\n  闸门只看 KIDp，ISp 为佐证。")
    if a.plan:
        print("\n--plan：没有加载模型。")
        print("先跑 `python scalediff_probe/attn_variants.py --selftest`，"
              "六条全过再开。")
        return 0
    if not todo:
        print("\n没有要跑的了。下一步：\n"
              f"  python scalediff_probe/std_table.py --patch-only "
              f"--real-split {a.split} --res {res} --boot 200 \\\n"
              f"      --arms " + " ".join(f'"{m}={dirs[m]}"' for m in ARMS
                                          if m in a.arms))
        return 0

    import torch
    import pipeline_scalediff_sdxl as PS
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    import attn_variants

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    # 管线在 __call__ 内部调用模块级的 register_attention_control。
    # 换臂就是换这个函数 —— 基线文件一个字不动（BASELINE_PATCHES.md）。
    mode = {"m": "npa"}
    shift_gen = {"g": None}

    def _reg(p):
        return attn_variants.register(p, mode["m"], shift_gen["g"])
    PS.register_attention_control = _reg

    mf = {m: (dirs[m] / "manifest.jsonl").open("a") for m in a.arms}
    t_all = time.time()
    per = {m: [] for m in a.arms}
    try:
        for n, (idx, arm) in enumerate(todo, 1):
            mode["m"] = arm
            # shifting 的偏移也要可复现：每 (idx, arm) 一颗确定性种子
            shift_gen["g"] = torch.Generator().manual_seed(
                a.seed * 1000003 + idx)
            # 钉全局 RNG —— 放大阶段的 randn_like 没传 generator（同
            # laion_hi_run 里那条注释），不钉住同 seed 也会出两张不同的图
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                images = pipe(
                    pmap[str(idx)], negative_prompt=NEGATIVE,
                    height=1024, width=1024,
                    generator=torch.Generator(device="cuda").manual_seed(a.seed),
                    num_inference_steps=a.steps, guidance_scale=7.5,
                    restart_ratio=a.restart_ratio,
                    scale_factor=a.scale_factor, upsample_stage=a.stage)
            except torch.cuda.OutOfMemoryError:
                print(f"\n[{arm}] {idx} OOM，跳过")
                torch.cuda.empty_cache()
                continue
            dt = time.time() - t0
            per[arm].append(dt)
            paths = {}
            for im in images:
                p = dirs[arm] / f"{idx:05d}_{im.width}.png"
                im.save(p)
                paths[im.width] = p.name
            mf[arm].write(json.dumps({
                "idx": idx, "arm": arm, "prompt": pmap[str(idx)],
                "split": a.split, "seed": a.seed, "files": paths,
                "sec": round(dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }, ensure_ascii=False) + "\n")
            mf[arm].flush()
            el = time.time() - t_all
            print(f"\r{n}/{len(todo)}  {el/3600:.1f}h 已用  "
                  f"剩约 {el/n*(len(todo)-n)/3600:.1f}h  "
                  f"[{arm}] {idx} {dt:.0f}s "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  end="", flush=True)
    finally:
        for f in mf.values():
            f.close()

    print(f"\n\n完成，用时 {(time.time()-t_all)/3600:.2f} 小时")
    print("各臂实测单张时长（中位数）：")
    for m in a.arms:
        if per[m]:
            v = sorted(per[m])
            print(f"  {m:<6} {v[len(v)//2]:.0f}s   n={len(v)}"
                  f"   论文值 {est[m]}s")
    print(f"""
下一步（打分几分钟，特征会落盘缓存）：
  python scalediff_probe/std_table.py --patch-only --real-split {a.split} \\
      --res {res} --boot 200 \\
      --arms {' '.join(f'"{m}={dirs[m]}"' for m in ARMS if m in a.arms)}

**--arms 第一个必须是 npa** —— 配对表以第一个臂为基准。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
