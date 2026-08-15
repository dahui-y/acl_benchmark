"""P0：三臂批跑，回答"局部保真度的缺口住在边界还是上下文"。

────────────────────────────────────────────────────────────────────────
三个臂
────────────────────────────────────────────────────────────────────────
    A  npa      ScaleDiff 的评测配置（附录 B.1 那个开关**没开**，与他们一致）
    B  shift    NPA + Query Window Random Shifting（他们写了、没用、没实现）
    C  md       ⚠️ **目录名叫 md，但它不是 MultiDiffusion**（订正见
                attn_variants.py 文件头）。实为「MultiDiffusion 的注意力
                模式 + NPA 的其余部分」，即**重叠 query 窗口 + 层内平均**，
                只贵 13%（真 MultiDiffusion 是 2.11×，因为它把整个 UNet
                在每个重叠 patch 上重跑一遍）。报表里标 `ovl-attn`。

**判据（重写版，2026-08-15，写在看数字之前）**
原先"g=C−A 是缺口总量、b/g 是边界占比"那套已作废 —— 它锚在
"C 复现了他们发表的 MultiDiffusion 行"，而该前提是错的。现在是：

  C vs A（配对 ΔKIDp）
    显著更好（Δ < −2σ_d） -> 局部保真缺口真实存在，且**重叠平均**这条
        便宜路子（+13%）就能吃到。方向成立，直接进方法设计，
        **不必再补真 MultiDiffusion 臂**。
    不显著 / 更差 -> 歧义：要么没有缺口，要么非得靠真 MultiDiffusion
        那种"整个 UNet 逐 patch 重算"才吃得到。**这时才补真 MD 臂**
        （约 13 h），用来区分这两种可能。
  B vs A（配对 ΔKIDp）
    独立读：他们那个没启用的随机平移本身有没有效。不受上述错误影响。

    D  ctx      **只把 K/V 窗口放大**（64²->128²），query 切法与基线
                完全一致 —— 唯一的单变量"上下文"探针。加这一臂的理由：
                A/B/C 三个臂**没有一个在测我们真正的论点**。论点是
                "每个 query 看不见足够的上下文"，而 shift 只动窗口位置
                （测边界）、ovl-attn 的上下文反而更少（1x vs 4x，与重叠
                平均混淆）。自注意力只占总时长约 6.5%，放大 4 倍也只到
                ~95s/张。

  **D vs A 的判据（写在 ctx 跑出任何数字之前，2026-08-15）**
    配对 ΔKIDp < −2σ_d          -> **上下文缺失是真病因**，方法沿这条设计；
    n>=300 仍落在噪声内          -> **上下文假设被正面否掉**，退回"分解的
        代价小到测不出"，§10 整条方向按判死处理，转 B 计划（评测方向）。
    配对 ΔKIDp > +2σ_d          -> 上下文更多反而更差，同样判死，但那是个
        本身值得报告的反直觉结果。

闸门只看 KIDp；ISp 是佐证列（功效不足，§10.12/§10.14）。

**另记**：主表（P1）**不需要** MultiDiffusion —— 它是他们消融表 Table 3
的一行，不是 Table 2 的对手。主表对手是 DemoFusion / AccDiffusion /
FreeScale / DiffuseHigh，引用其发表值。

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
ARMS = ("npa", "shift", "md", "ctx")


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
    ap.add_argument("--ctx-units", type=int, default=4,
                    help="ctx 臂的 K/V 边长 = N × (window//2)。"
                         "4 = 2×ws（面积 4x，默认）；显存吃紧退到 3（2.25x）")
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
    # 我们卡上的实测中位数（论文 A6000 上 npa 是 113s）。md 只有 1.13x
    # 而不是论文的 2.11x —— 因为它不是真 MultiDiffusion，见文件头。
    est = {"npa": 75, "shift": 78, "md": 85, "ctx": 95}
    hrs = sum(est[m] for _, m in todo) / 3600
    print(f"\n目标 {res}²   臂 {a.arms}   名单 {len(idxs)} 条")
    for m in a.arms:
        print(f"  {m:<6} 已有 {len(done[m]):>4} / {len(idxs)}  -> {dirs[m]}")
    print(f"待跑 {len(todo)} 个任务，按本机实测中位数估 **{hrs:.1f} 小时**")
    print("\n判据（重写版，写在看数字之前 —— 旧的 b/g 读法已作废，见文件头）："
          "\n  C(ovl-attn) vs A：配对 ΔKIDp < −2σ_d -> 缺口为真且重叠平均"
          "就能吃到，进方法设计；"
          "\n      否则 -> 歧义，届时才补真 MultiDiffusion 臂（约 13h）区分。"
          "\n  B(shift) vs A：独立读，他们那个没启用的开关本身有没有效。"
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
        m = mode["m"]
        return attn_variants.register(
            p, f"ctx{a.ctx_units}" if m == "ctx" else m, shift_gen["g"])
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
