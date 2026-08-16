#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
存在性检查：属性绑定（attribute binding）这个问题在 SDXL 上到底有多大空间。

    这个脚本**只跑基线，不做方法**。它回答两个问题，然后按预注册判据给出
    「继续 / 判死」。上一轮（§10）我把顺序做反了 —— 先设计方法再量问题，
    结果三个探针全部死在「差值小于噪声」。这次先量。

────────────────────────────────────────────────────────────────────────
两个读数
────────────────────────────────────────────────────────────────────────

  读数①  患病率 prevalence
        逐 (prompt, seed) 的 BLIP-VQA 绑定分低于阈值的比例。
        这是 §10.1 那条规矩的直接应用：AccDiffusion 修好了重复仍然输 FID，
        因为重复只在 3% 的图上。**能推动 1000 图平均的问题，患病率必须够高。**

  读数②  余量 headroom
        已发表 SOTA 分 − 我们复现的 SDXL 基线分，对上我们自己 bootstrap
        出来的噪声底。如果余量落在噪声底里，说明打分器已经饱和 ——
        再好的方法也读不出来。

────────────────────────────────────────────────────────────────────────
预注册判据（写在看到任何数字之前，2026-08-16）
────────────────────────────────────────────────────────────────────────

  判死（任一成立即停，不进方法设计）：
    A. 患病率 @0.5 < 10%
    B. 余量 < 3 × σ_boot

  继续：
    A 和 B 都不成立，且余量 ≥ 5 × σ_boot 时算「余量充足」。
    介于 3σ 和 5σ 之间记为「边缘」，要人来拍板，不自动放行。

  这三条阈值在开跑前定死。看到数字之后改阈值 = 移动球门（§10 有过一次
  这个争论：futility stopping 合法，改判据不合法）。

────────────────────────────────────────────────────────────────────────
打分器必须是官方的
────────────────────────────────────────────────────────────────────────

  读数②是一个减法：published_SOTA − our_baseline。这个减法只有在两边
  用**同一个打分器**时才成立。std_table.py 那次的教训是特征塔要锁死；
  这里同理 —— 自己写一版 BLIP-VQA，数字与发表值不可比，读数②直接作废。

  所以：
    · prompts   来自官方 T2I-CompBench 的 examples/dataset/{subset}_val.txt
    · 打分      调官方 BLIP-VQA 脚本，本脚本只消费它吐出的 JSON
    · 本脚本负责：生成（文件名按官方约定）、统计、判据

  官方仓库 = https://github.com/Karine-Huang/T2I-CompBench
  用 --preflight 检查它在不在、命名约定对不对（见 NAMING 的注释）。

────────────────────────────────────────────────────────────────────────
bootstrap 要按 prompt 聚类，不能按图
────────────────────────────────────────────────────────────────────────

  同一个 prompt 的 4 个 seed 是相关的（prompt 难易本身是主要方差来源）。
  按图独立重采样会**低估** σ，让余量看起来比实际显著。
  正确做法是 cluster bootstrap：重采样 prompt，整簇带走它的所有 seed。
  paired_boot 那套（std_table.py）的思路一样，这里换成按簇。

用法：
    source scalediff_probe/env.sh
    python comp_probe/exist_check.py --plan                 # 零 GPU，先看账
    python comp_probe/exist_check.py --preflight            # 检查阻塞项
    python comp_probe/exist_check.py --gen  --subset color  # ~2.5 h
    # …跑官方 BLIP-VQA…
    python comp_probe/exist_check.py --score --subset color
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────

SUBSETS = ("color", "shape", "texture")     # attribute binding 的三个子集
PRIMARY = "color"                            # 预注册的主子集

N_SEEDS = 4          # 每 prompt 的 seed 数。官方协议是 10；存在性检查用 4，
                     # 因为我们只要基线分布和噪声底，不需要可发表精度。
                     # 注意：**这会让我们的均值与发表值有小偏差**，见 --plan 的告警。
STEPS = 50
GUIDANCE = 7.5       # SDXL 默认；发表的 CompBench 基线也是这个
SIZE = 1024

# 患病率阈值。BLIP-VQA 吐的是 [0,1] 的概率，不是二值。
# 主判据用 0.5，同时报整条曲线，避免单点阈值带来的错觉。
PREV_THRESHOLDS = (0.3, 0.5, 0.7)
PREV_PRIMARY = 0.5

# 预注册判据
KILL_PREVALENCE = 0.10      # 患病率低于此 → 判死
KILL_HEADROOM_SIGMA = 3.0   # 余量小于此倍噪声底 → 判死
PASS_HEADROOM_SIGMA = 5.0   # 余量大于此倍 → 余量充足

B_BOOT = 2000
BOOT_SEED = 0

# ── 参考数字 ────────────────────────────────────────────────────────
# 来源：ELDiff (arXiv 2606.20924) Table 2「Comparison results with training-free
# methods on T2I-CompBench」，SDXL 分组。这是一手表格（从 PDF 里逐行抠出来的），
# 不是综述转述。上一版这里填的 0.6369/0.5408/0.5637 是二手数字，**是错的**，
# 已替换。
#
#   Method   Color   Shape   Texture  Spatial  NonSpat  Complex  Latency
#   SDXL     0.6734  0.5064  0.6243   0.2073   0.3166   0.3575   8.07s
#   SynGen   0.7267  0.5249  0.6476   0.2387   0.3172   0.3944   11.62s
#   InitNO   0.6823  0.5229  0.6547   0.2553   0.3147   0.4077   19.42s
#   R2F      0.7135  0.5194  0.6625   0.2447   0.3170   0.4031   10.14s
#   ELDiff   0.7768  0.5663  0.6941   0.2496   0.3261   0.4252   8.07s  ← 要训练，不算
#
# 'sota' 取**训练无关**里的最好值（ELDiff 是 finetune 的，不进这一栏，
# 但它会出现在审稿人看到的表里，见 'trained_bar'）。
REF = {
    "color":   {"sdxl": 0.6734, "sota": 0.7267, "sota_by": "SynGen",
                "trained_bar": 0.7768, "src": "ELDiff arXiv 2606.20924 Table 2"},
    "shape":   {"sdxl": 0.5064, "sota": 0.5249, "sota_by": "SynGen",
                "trained_bar": 0.5663, "src": "同上"},
    "texture": {"sdxl": 0.6243, "sota": 0.6547, "sota_by": "InitNO",
                "trained_bar": 0.6941, "src": "同上"},
}

# 延迟帕累托。我们的机制是「多一次前向、不做优化」→ 3 次前向 vs 基线 2 次
# = 约 1.5×，落在 ~12s。与 SynGen 同档。
#   ⚠️ 这意味着**「比优化法快」这个卖点不成立** —— SynGen 只慢 44% 而且
#   color 最好；真正慢的 InitNO(+141%) 反而 color 更差。
#   所以我们必须**在分数上真的赢 SynGen 的 0.7267**，快不是理由。
LATENCY = {"sdxl": 8.07, "SynGen": 11.62, "R2F": 10.14, "InitNO": 19.42,
           "ours_est": 12.1}

OUT_ROOT = Path(os.environ.get("SD_OUT", "/tmp")) / "comp"
CB_ROOT = Path(os.environ.get("COMPBENCH_ROOT", "")) if os.environ.get("COMPBENCH_ROOT") else None

# 官方文件名约定。T2I-CompBench 的 BLIP-VQA 脚本从**文件名**里解析 prompt，
# 所以命名错了不是「不好看」，是整个打分跑出来是错的。
#   约定：{prompt}_{index:06d}.png，放在一个 samples/ 目录里
# --preflight 会拿官方仓库里的样例目录核对这条；核对不上就别开跑。
NAMING = "{prompt}_{idx:06d}.png"


def prompts_path(subset):
    if CB_ROOT is None:
        return None
    return CB_ROOT / "examples" / "dataset" / f"{subset}_val.txt"


def load_prompts(subset):
    """读官方 prompt 列表。顺序即 idx，不排序、不去重 —— 官方打分脚本按
    文件名排序建立 question_id 映射，我们这边任何重排都会错位。"""
    p = prompts_path(subset)
    if p is None or not p.exists():
        sys.exit(f"!! 找不到 prompt 文件：{p}\n"
                 f"   设 COMPBENCH_ROOT 指向 T2I-CompBench 仓库根目录")
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]


def sanitize(prompt):
    """官方样例里的文件名就是 prompt 原文（含空格）。这里只挡掉真正会
    炸文件系统的字符，**不做任何其它改写** —— 改写会让打分脚本解析出
    另一个 prompt。"""
    return re.sub(r"[/\x00]", "_", prompt)


# ─────────────────────────────────────────────────────────────────────
# --plan：零 GPU，先把账和判据摊开
# ─────────────────────────────────────────────────────────────────────

def cmd_plan(args):
    subset = args.subset
    n_prompt = 300      # 官方 val 集大小；--preflight 会核实
    n_img = n_prompt * N_SEEDS
    sec_per = 5.5       # SDXL 1024² / 50 步 / 4090 的实测量级
    hours = n_img * sec_per / 3600

    print("═" * 68)
    print(f"存在性检查 · 子集 = {subset}")
    print("═" * 68)
    print(f"\n生成账：")
    print(f"  prompts        {n_prompt}")
    print(f"  seeds/prompt   {N_SEEDS}")
    print(f"  图片总数       {n_img}")
    print(f"  单张 ~{sec_per}s  →  约 {hours:.1f} h")
    print(f"  磁盘 ~{n_img * 1.4 / 1024:.1f} GB @1024² png")
    print(f"\n  对照：4096² ScaleDiff 一个臂 n=60 要 7 h。这里 1200 张 2 h。")

    print(f"\n预注册判据（已写死在源码顶部，看数后不得改）：")
    print(f"  判死 A：患病率@{PREV_PRIMARY} < {KILL_PREVALENCE:.0%}")
    print(f"  判死 B：余量 < {KILL_HEADROOM_SIGMA:g} × σ_boot")
    print(f"  放行  ：余量 ≥ {PASS_HEADROOM_SIGMA:g} × σ_boot")
    print(f"  中间带：{KILL_HEADROOM_SIGMA:g}σ–{PASS_HEADROOM_SIGMA:g}σ 记「边缘」，"
          f"不自动放行")

    print(f"\n参考数字（ELDiff 2606.20924 Table 2，SDXL 组）：")
    print(f"  {'子集':8s} {'SDXL':>7s} {'训练无关SOTA':>13s} {'余量':>8s}  {'(要训练的门槛)':>14s}")
    for k, v in REF.items():
        mark = "  ← 本次" if k == subset else ""
        print(f"  {k:8s} {v['sdxl']:7.4f} {v['sota']:8.4f}({v['sota_by']:6s}) "
              f"{v['sota']-v['sdxl']:+8.4f}  {v['trained_bar']:14.4f}{mark}")
    print(f"\n  color 余量最大（+{REF['color']['sota']-REF['color']['sdxl']:.4f}），"
          f"所以主子集选 color 是对的。")

    print(f"\n延迟帕累托（我们的机制 = 多一次前向，无优化）：")
    for k in ("sdxl", "R2F", "SynGen", "ours_est", "InitNO"):
        tag = "  ← 我们的估计" if k == "ours_est" else ""
        print(f"  {k:10s} {LATENCY[k]:6.2f}s{tag}")
    print(f"  ⚠️ 「比优化法快」这个卖点不成立：SynGen 只慢 44% 且 color 最好；"
          f"\n     真正慢的 InitNO(+141%) color 反而更差。"
          f"**必须在分数上真赢 SynGen 0.7267。**")

    print(f"\n!! 两条开跑前的阻塞项：")
    print(f"   1. N_SEEDS={N_SEEDS} 而官方协议是 10。我们的均值会与发表值"
          f"有小偏差；\n"
          f"      读数②做减法时两边不同协议 → **只能当量级参考，不能当"
          f"精确余量**。\n"
          f"      若余量落在边缘带，必须补跑到 10 seeds 再判。")
    print(f"   2. 我们复现的 SDXL 基线必须落在 {REF[subset]['sdxl']} 附近；"
          f"差太远说明协议没对上，\n"
          f"      此时余量读数无效（--score 会自动查这一条）。")
    print()


# ─────────────────────────────────────────────────────────────────────
# --preflight：把能自动查的阻塞项查掉
# ─────────────────────────────────────────────────────────────────────

def cmd_preflight(args):
    ok = True

    def chk(label, good, detail=""):
        nonlocal ok
        print(f"  [{'ok' if good else '!!'}] {label}" + (f"   {detail}" if detail else ""))
        if not good:
            ok = False

    print("═" * 68)
    print("preflight")
    print("═" * 68)

    # ① 官方仓库
    chk("COMPBENCH_ROOT 已设", CB_ROOT is not None, str(CB_ROOT))
    if CB_ROOT is not None:
        chk("仓库目录存在", CB_ROOT.exists(), str(CB_ROOT))
        for s in SUBSETS:
            p = prompts_path(s)
            n = len([l for l in p.read_text().splitlines() if l.strip()]) if p.exists() else 0
            chk(f"prompts {s}", p.exists() and n > 0, f"{n} 条  {p}")

        # ② 命名约定 —— 拿官方样例目录核对，核对不上就别开跑
        samples = list(CB_ROOT.glob("examples/samples/*.png"))[:3]
        if samples:
            print(f"  ·· 官方样例文件名（用来核对 NAMING）：")
            for s in samples:
                print(f"       {s.name}")
            looks_right = all(re.search(r"_\d{6}\.png$", s.name) for s in samples)
            chk("文件名形如 {prompt}_{000000}.png", looks_right)
        else:
            chk("找到官方样例目录 examples/samples/", False,
                "没有样例就无法核对命名约定 —— 这是硬阻塞，命名错了打分全错")

    # ③ 输出盘
    out = OUT_ROOT / args.subset / "samples"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / ".wtest").touch()
        (out / ".wtest").unlink()
        import shutil
        free = shutil.disk_usage(out).free / 2**30
        chk("输出目录可写", True, f"{out}  剩余 {free:.0f} GB")
        chk("空间够（需 ~2 GB）", free > 5, f"{free:.0f} GB")
    except Exception as e:
        chk("输出目录可写", False, f"{out}  {e}")

    # ④ SDXL
    try:
        import torch
        chk("CUDA 可用", torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无卡")
    except Exception as e:
        chk("torch", False, str(e))
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        import diffusers
        chk("diffusers", True, diffusers.__version__)
    except Exception as e:
        chk("diffusers", False, str(e))

    # ⑤ BLIP 权重 —— 服务器是 HF_HUB_OFFLINE=1 钉死的
    hf = Path(os.environ.get("HF_HOME", "")) / "hub"
    blip = list(hf.glob("models--Salesforce--blip*")) if hf.exists() else []
    chk("BLIP 权重已在本地缓存", bool(blip),
        (str(blip[0].name) if blip else
         "离线模式下打分会挂住 —— 用 scalediff_probe/fetch_weights.sh 走 hf-mirror 先拉"))

    # ⑥ REF 是否填完
    r = REF[args.subset]
    chk("REF sota 已填", r["sota"] is not None,
        "还是 None → 判据 B 算不出来" if r["sota"] is None else str(r["sota"]))

    print("\n" + ("preflight 通过，可以开跑" if ok else "!! 有阻塞项，先处理"))
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────
# --gen：只生成基线，不做方法
# ─────────────────────────────────────────────────────────────────────

def cmd_gen(args):
    import torch
    from diffusers import StableDiffusionXLPipeline

    subset = args.subset
    prompts = load_prompts(subset)
    out = OUT_ROOT / subset / "samples"
    out.mkdir(parents=True, exist_ok=True)

    print(f"子集 {subset}：{len(prompts)} prompts × {N_SEEDS} seeds "
          f"= {len(prompts) * N_SEEDS} 张 → {out}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    t0 = time.time()
    done = 0
    # 按 prompt 外层、seed 内层。中断后 partial 数据仍然是「前 k 个 prompt
    # 各 N_SEEDS 张」，簇是完整的 —— cluster bootstrap 才能直接用。
    for i, prompt in enumerate(prompts):
        base = sanitize(prompt)
        paths = [out / NAMING.format(prompt=base, idx=i * N_SEEDS + s)
                 for s in range(N_SEEDS)]
        if all(p.exists() for p in paths):
            done += N_SEEDS
            continue
        for s, path in enumerate(paths):
            if path.exists():
                done += 1
                continue
            g = torch.Generator("cuda").manual_seed(i * 1000 + s)
            img = pipe(prompt=prompt, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE, height=SIZE, width=SIZE,
                       generator=g).images[0]
            img.save(path)
            done += 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            rate = el / max(done, 1)
            left = (len(prompts) * N_SEEDS - done) * rate
            print(f"  {i+1}/{len(prompts)} prompts  {done} 张  "
                  f"{el/60:.0f}m 已用 / {left/60:.0f}m 剩余", flush=True)

    print(f"完成 {done} 张，{(time.time()-t0)/60:.0f} 分钟")
    print(f"\n下一步：跑官方 BLIP-VQA")
    print(f"  cd $COMPBENCH_ROOT/BLIPvqa_eval")
    print(f"  python BLIP_vqa.py --out_dir={out.parent}")
    print(f"  # 产物应为 {out.parent}/annotation_blip/vqa_result.json")


# ─────────────────────────────────────────────────────────────────────
# --score：消费官方 JSON，出两个读数 + 判据
# ─────────────────────────────────────────────────────────────────────

def read_official_scores(subset, n_prompt):
    """读官方 BLIP-VQA 的产物，还原成 [n_prompt, N_SEEDS] 的分数矩阵。

    官方格式：annotation_blip/vqa_result.json = [{"question_id": int,
    "answer": "0.83"}, ...]，question_id 是**按文件名排序后**的序号。
    我们生成时的 idx = i*N_SEEDS+s 已经保证排序序号与 (prompt, seed) 一一
    对应吗？—— **不保证**：排序是按文件名字符串，prompt 在前，所以是按
    prompt 字典序而不是我们的 i。所以这里必须重建映射，不能直接 reshape。
    """
    root = OUT_ROOT / subset
    j = root / "annotation_blip" / "vqa_result.json"
    if not j.exists():
        sys.exit(f"!! 找不到官方打分产物：{j}\n   先跑 BLIPvqa_eval/BLIP_vqa.py")
    recs = json.loads(j.read_text())

    files = sorted((root / "samples").glob("*.png"), key=lambda p: p.name)
    if len(files) != len(recs):
        sys.exit(f"!! 图片数 {len(files)} 与打分条数 {len(recs)} 不一致 —— "
                 f"打分器扫到的目录可能不是 {root/'samples'}")

    # question_id → 文件名 → 我们的 idx → (prompt_i, seed_s)
    by_qid = {int(r["question_id"]): float(r["answer"]) for r in recs}
    mat = np.full((n_prompt, N_SEEDS), np.nan)
    for qid, f in enumerate(files):
        m = re.search(r"_(\d{6})\.png$", f.name)
        if m is None:
            sys.exit(f"!! 文件名不符合约定，无法还原 idx：{f.name}")
        idx = int(m.group(1))
        i, s = divmod(idx, N_SEEDS)
        if i >= n_prompt:
            sys.exit(f"!! idx {idx} 超出 prompt 数 {n_prompt}")
        if qid not in by_qid:
            sys.exit(f"!! question_id {qid} 不在打分结果里")
        mat[i, s] = by_qid[qid]
    if np.isnan(mat).any():
        n = int(np.isnan(mat).sum())
        sys.exit(f"!! 有 {n} 个 (prompt, seed) 没有分数 —— 生成没跑完")
    return mat


def cluster_boot(mat, B, seed):
    """按 prompt 聚类的 bootstrap：重采样行（整簇带走它的所有 seed）。
    按图独立重采样会低估 σ —— 同 prompt 的 seed 是相关的。"""
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    return np.array([mat[rng.integers(0, n, n)].mean() for _ in range(B)])


def cmd_score(args):
    subset = args.subset
    prompts = load_prompts(subset)
    mat = read_official_scores(subset, len(prompts))
    n_prompt, n_seed = mat.shape
    flat = mat.ravel()

    mean = float(flat.mean())
    boots = cluster_boot(mat, B_BOOT, BOOT_SEED)
    sigma = float(boots.std())
    # 对照：按图独立重采样的 σ，用来显示聚类到底重要多少
    rng = np.random.default_rng(BOOT_SEED)
    naive = np.array([flat[rng.integers(0, flat.size, flat.size)].mean()
                      for _ in range(B_BOOT)]).std()

    print("═" * 68)
    print(f"存在性检查 · {subset} · {n_prompt} prompts × {n_seed} seeds")
    print("═" * 68)

    print(f"\n基线")
    print(f"  我们的 SDXL       {mean:.4f}")
    print(f"  σ_boot (聚类)     {sigma:.4f}")
    print(f"  σ_boot (按图，错)  {naive:.4f}   ← 低估 {sigma/max(naive,1e-9):.1f}×，"
          f"判据用聚类那个")
    ref = REF[subset]
    print(f"  发表的 SDXL       {ref['sdxl']}   差 {mean - ref['sdxl']:+.4f}")
    if abs(mean - ref["sdxl"]) > 5 * sigma:
        print(f"  !! 与发表值差超过 5σ —— 复现没对上，先查 seeds/steps/"
              f"guidance/协议，别急着读判据")

    print(f"\n读数① 患病率")
    for t in PREV_THRESHOLDS:
        prev = float((flat < t).mean())
        mark = "  ← 主判据" if t == PREV_PRIMARY else ""
        print(f"  分 < {t:.1f}   {prev:.1%}{mark}")
    prevalence = float((flat < PREV_PRIMARY).mean())
    q = np.percentile(flat, [10, 25, 50, 75, 90])
    print(f"  分位数 p10/p25/p50/p75/p90 = "
          f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f} / {q[4]:.2f}")

    print(f"\n读数② 余量")
    if ref["sota"] is None:
        print(f"  !! REF['{subset}']['sota'] 还是 None —— 判据 B 算不出来。")
        print(f"     先把 SOTA 数字从论文原文填进源码顶部的 REF。")
        headroom = None
    else:
        headroom = ref["sota"] - mean
        print(f"  发表 SOTA         {ref['sota']}")
        print(f"  余量              {headroom:+.4f}")
        print(f"  余量 / σ_boot     {headroom/sigma:.1f}×")

    print(f"\n判据")
    kills = []
    if prevalence < KILL_PREVALENCE:
        kills.append(f"A 患病率 {prevalence:.1%} < {KILL_PREVALENCE:.0%}")
    if headroom is not None and headroom < KILL_HEADROOM_SIGMA * sigma:
        kills.append(f"B 余量 {headroom/sigma:.1f}σ < {KILL_HEADROOM_SIGMA:g}σ")

    if kills:
        print(f"  ❌ 判死：" + "；".join(kills))
        print(f"     不进方法设计。")
    elif headroom is None:
        print(f"  ⏸  A 通过（患病率 {prevalence:.1%}），B 待 REF 填完")
    elif headroom >= PASS_HEADROOM_SIGMA * sigma:
        print(f"  ✅ 通过：患病率 {prevalence:.1%}，余量 {headroom/sigma:.1f}σ")
    else:
        print(f"  ⚠️  边缘：余量 {headroom/sigma:.1f}σ 落在 "
              f"{KILL_HEADROOM_SIGMA:g}–{PASS_HEADROOM_SIGMA:g}σ 之间。"
              f"不自动放行。")
        print(f"     N_SEEDS={N_SEEDS} 而官方协议是 10 —— 边缘带里必须"
              f"补跑到 10 seeds 再判。")

    out = OUT_ROOT / subset / "exist_check.json"
    out.write_text(json.dumps({
        "subset": subset, "n_prompt": n_prompt, "n_seed": n_seed,
        "mean": mean, "sigma_cluster": sigma, "sigma_naive": float(naive),
        "prevalence": {str(t): float((flat < t).mean()) for t in PREV_THRESHOLDS},
        "ref": ref, "headroom": headroom,
        "verdict": "kill" if kills else ("pending" if headroom is None else
                   ("pass" if headroom >= PASS_HEADROOM_SIGMA * sigma else "marginal")),
        "kills": kills,
    }, ensure_ascii=False, indent=2))
    print(f"\n写入 {out}")


# ─────────────────────────────────────────────────────────────────────
# --selftest：两个静默错误的守门人，零 GPU
# ─────────────────────────────────────────────────────────────────────

def cmd_selftest(args):
    import tempfile
    global OUT_ROOT, load_prompts
    fails = []

    def chk(label, good, detail=""):
        print(f"  [{'ok' if good else '!!'}] {label}" + (f"   {detail}" if detail else ""))
        if not good:
            fails.append(label)

    print("═" * 68)
    print("selftest")
    print("═" * 68)

    with tempfile.TemporaryDirectory() as td:
        saved_root, saved_load = OUT_ROOT, load_prompts
        OUT_ROOT = Path(td)
        try:
            # ① question_id ↔ idx 的还原。
            # 官方按**文件名字典序**编 question_id，而我们的 idx 是生成顺序。
            # 两者不同 —— 直接 reshape 会把分数错位，而且完全静默。
            # 这里故意用一组字典序 ≠ 生成序的 prompt。
            prompts = ["zebra in red", "apple green", "midnight blue car"]
            root = OUT_ROOT / "color"
            (root / "samples").mkdir(parents=True, exist_ok=True)
            truth = {}
            for i, p in enumerate(prompts):
                for s in range(N_SEEDS):
                    idx = i * N_SEEDS + s
                    (root / "samples" / NAMING.format(prompt=sanitize(p), idx=idx)).write_bytes(b"")
                    truth[(i, s)] = round(0.1 * i + 0.01 * s, 4)
            files = sorted((root / "samples").glob("*.png"), key=lambda x: x.name)
            recs = []
            for qid, f in enumerate(files):
                i, s = divmod(int(re.search(r"_(\d{6})\.png$", f.name).group(1)), N_SEEDS)
                recs.append({"question_id": qid, "answer": str(truth[(i, s)])})
            (root / "annotation_blip").mkdir(exist_ok=True)
            (root / "annotation_blip" / "vqa_result.json").write_text(json.dumps(recs))

            load_prompts = lambda subset: prompts       # noqa: E731
            mat = read_official_scores("color", len(prompts))
            exp = np.array([[truth[(i, s)] for s in range(N_SEEDS)]
                            for i in range(len(prompts))])
            chk("① 还原矩阵 == 真值", np.allclose(mat, exp))

            naive = np.array([float(r["answer"]) for r in recs]).reshape(len(prompts), N_SEEDS)
            chk("② 偷懒 reshape != 真值（否则①是空测试）", not np.allclose(naive, exp),
                f"字典序首文件 {files[0].name}")
        finally:
            OUT_ROOT, load_prompts = saved_root, saved_load

    # ③ cluster bootstrap 必须比按图 bootstrap 大。簇内完全相关时
    #    理论比值 = sqrt(N_SEEDS)；按图重采样低估 σ 就是这么来的。
    big = np.repeat(np.linspace(0, 1, 200)[:, None], N_SEEDS, axis=1)
    sc = cluster_boot(big, 500, 0).std()
    rng = np.random.default_rng(0)
    fl = big.ravel()
    sn = np.array([fl[rng.integers(0, fl.size, fl.size)].mean() for _ in range(500)]).std()
    ratio = sc / max(sn, 1e-12)
    chk("③ σ_cluster/σ_naive ≈ sqrt(N_SEEDS)", abs(ratio - N_SEEDS ** 0.5) < 0.15,
        f"{ratio:.2f}× vs 期望 {N_SEEDS**0.5:.2f}×")

    print("\n" + ("全部通过" if not fails else f"!! 失败：{fails}"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default=PRIMARY, choices=SUBSETS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="零 GPU，打印账和判据")
    g.add_argument("--selftest", action="store_true", help="零 GPU，查静默错误")
    g.add_argument("--preflight", action="store_true", help="检查阻塞项")
    g.add_argument("--gen", action="store_true", help="生成基线图")
    g.add_argument("--score", action="store_true", help="消费官方打分 → 判据")
    a = ap.parse_args()

    if a.plan:
        cmd_plan(a)
    elif a.selftest:
        sys.exit(cmd_selftest(a))
    elif a.preflight:
        sys.exit(cmd_preflight(a))
    elif a.gen:
        cmd_gen(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
