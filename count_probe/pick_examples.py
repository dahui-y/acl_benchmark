#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
挑出该**用肉眼看**的那几组题，并拼成对照图。零 GPU。

    为什么要挑而不是随便看：画质的中位数变化是 +0，退化只集中在 6~11/60 张尾巴上。
    只看最差的那几张，必然得出"画质很糟"的印象；只看随机几张，又会漏掉真正的问题。
    所以要**分组看**，每组回答一个不同的问题：

      colourless  CountGen 自己就把它变成无色的题
                  → 验证「黑白剪贴画是在位者的失效，不是我们加上去的」
      worse       相对 CountGen 掉得最狠、且**基准本身还有颜色**的题
                  → 这是我们唯一可能真的把图弄坏的地方，必须看
      fixed       我们数对了、CountGen 数错了
                  → 增益长什么样；论文定性图从这里选
      broken      CountGen 数对了、我们数错了
                  → 代价长什么样。不看这一组就是在挑好的看
      random      固定种子的随机抽样
                  → 典型情况。前四组都是尾巴，只看尾巴会严重高估问题

    每组都调 compare.py 拼图，一行里是：原版 SDXL → 各配置的输出 → 布局 mask。

用法：
    python count_probe/pick_examples.py \\
        --ref  $SD_OUT/count/tune_orig_arms \\
        --arm  $SD_OUT/count/tune_both_arms \\
        --runs $SD_OUT/count/tune_orig $SD_OUT/count/tune_fg $SD_OUT/count/tune_both \\
        --out  $SD_OUT/count/look --n 4
"""

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

from yolo_eval import _photo_stats

HERE = Path(__file__).resolve().parent


def _rows(arms):
    p = Path(arms) / "yolo_results.csv"
    if not p.exists():
        raise SystemExit(f"!! 没有 {p}")
    out = []
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        out.append(r)
    return out


def _ok(v):
    if v in ("", None, "None"):
        return None
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    try:
        return bool(int(float(v)))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="参照（原版 CountGen）的 _arms 目录")
    ap.add_argument("--arm", required=True, help="要检查的配置的 _arms 目录")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="拼图要放哪几个跑批目录（**不带** _arms），从左到右")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="每组挑几题")
    ap.add_argument("--grey", type=float, default=15.0)
    ap.add_argument("--drop", type=float, default=10.0,
                    help="worse 组只收掉超过百分之几的题。凑数把 −2%% 的也列进来，"
                         "会让人以为那也算「弄坏了」（argparse 会对 help 做 %% 展开，"
                         "所以这里的百分号要写两遍）")
    ap.add_argument("--seed", type=int, default=0, help="random 组的种子，固定可复现")
    ap.add_argument("--no-figs", action="store_true", help="只打印题目 id，不拼图")
    a = ap.parse_args()

    ra = {r["stem"]: r for r in _rows(a.ref)}
    rb = {r["stem"]: r for r in _rows(a.arm)}
    common = sorted(set(ra) & set(rb))
    if not common:
        raise SystemExit("!! 两个目录没有共同题目")

    # 逐题算两边的 colourfulness（只对共同题）
    stats = {}
    for s in common:
        f = rb[s]["file"]
        pv, pm = Path(a.ref) / "countgen" / f, Path(a.arm) / "countgen" / f
        if pv.exists() and pm.exists():
            stats[s] = (_photo_stats(pv)[0], _photo_stats(pm)[0])

    groups = {}

    # ① CountGen 自己就无色的 —— 验证这是在位者的失效
    groups["colourless"] = sorted(
        (s for s in stats if stats[s][0] < a.grey),
        key=lambda s: stats[s][0])[:a.n]

    # ② 相对 CountGen 掉得最狠，且基准本身还有颜色（小基数上的百分比不可信）
    cand = [(s, 100.0 * (stats[s][1] - stats[s][0]) / stats[s][0])
            for s in stats if stats[s][0] >= a.grey]
    # 只收真的掉下来的。凑数把 −2% 的题也列进来，会让人以为那也是「弄坏了」。
    groups["worse"] = [s for s, d in sorted(cand, key=lambda x: x[1])[:a.n]
                       if d < -a.drop]

    # ③④ 计数上的赢和输 —— 两边都要看，只看赢的就是在挑好的看
    fixed = [s for s in common
             if _ok(rb[s].get("ok_countgen")) and not _ok(ra[s].get("ok_countgen"))]
    broken = [s for s in common
              if _ok(ra[s].get("ok_countgen")) and not _ok(rb[s].get("ok_countgen"))]
    groups["fixed"] = fixed[:a.n]
    groups["broken"] = broken[:a.n]

    # ⑤ 随机抽样 —— 前四组全是尾巴，没有这一组会严重高估问题
    rng = random.Random(a.seed)
    groups["random"] = rng.sample(common, min(a.n, len(common)))

    why = {
        "colourless": f"CountGen 自己的 colourfulness < {a.grey:.0f}"
                      "（看我们有没有比它更糟，而不是看它糟不糟）",
        "worse": f"相对 CountGen 掉超过 {a.drop:.0f}%、且 CountGen 那张本身还有颜色"
                 f"（我们唯一可能真把图弄坏的地方）",
        "fixed": f"我们数对了、CountGen 数错了（共 {len(fixed)} 题）",
        "broken": f"CountGen 数对了、我们数错了（共 {len(broken)} 题）"
                  "—— **必须看**，不看就是在挑好的看",
        "random": f"固定种子随机抽样（中位数变化是 +0，前四组都是尾巴）",
    }

    out = Path(a.out)
    for g, stems in groups.items():
        print(f"\n{'='*68}\n【{g}】{why[g]}")
        if not stems:
            print("  （这一组是空的）"
                  + ("  ← 没有一题掉超过阈值，这本身就是结论" if g == "worse" else ""))
            continue
        for s in stems:
            cv, cm = stats.get(s, (float("nan"), float("nan")))
            d = 100.0 * (cm - cv) / cv if cv == cv and cv > 1e-6 else float("nan")
            print(f"  {s:<42} colour CountGen {cv:>5.1f} → 本配置 {cm:>5.1f} ({d:>+5.0f}%)")
        if a.no_figs:
            continue
        for s in stems:
            subprocess.run([sys.executable, str(HERE / "compare.py"), "--stem", s,
                            "--runs", *a.runs, "--out", str(out / g)],
                           stdout=subprocess.DEVNULL)
        print(f"  → 拼图 {out / g}/cmp_*.png")

    if not a.no_figs:
        print(f"\n每张拼图从左到右：原版 SDXL(vanilla) → "
              + " → ".join(Path(r).name for r in a.runs) + " → 布局 mask")
        print("判读：worse 组要问「这张比 CountGen 那张明显更差吗」，"
              "不是「这张好看吗」；\n      random 组要问「典型情况下两边分得出高下吗」，"
              "分不出才是我们要的结论。")


if __name__ == "__main__":
    main()
