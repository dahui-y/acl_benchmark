#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把「修正失败」这一桶再拆开 —— **零 GPU**，只读 yolo_eval.py 已经落盘的 CSV。

    表三给出的结论是：CountGen 的损失里，"计数器说对但实际错"（结构性、永远
    修不到）只占 9.0%，而"进了修正却没修好"占 **34.1%**。也就是说
    **主要矛盾在修正本身，不在计数器**。换掉计数器最多值 +3.7 分，
    把修正做对值 +34 分。

    那就得知道修正为什么失败。这个脚本按几个**事先就能想到、且与实现直接对应**
    的维度切开，不做钓鱼式的相关性搜索：

    1. **方向**：DBSCAN 数多了还是少了。这两条在它们代码里是**两套完全不同的
       机制**（`extract_mask.py:32-37`）：
         · 多了 → `relayout_overgeneration`：按面积排序，**直接删掉最小的几个**
                  （`relayout.py:62-69`，没有模型，纯启发式）
         · 少了 → `relayout_undergeneration`：用训练好的 ReLayout U-Net 一次补一个
       两者成功率若差得远，就指出该动哪一半。

    2. **幅度** |n_dbscan − N|：要改的量越大是不是越容易失败。

    3. **要求数 N**：难度随 N 的变化。

    4. **修正有没有把本来对的图改坏**：计数器说不匹配、但原版图其实是对的 ——
       这类图会被送进修正，而修正可能把它弄错。这是纯粹的负收益，
       表三里看不出来，因为它被并进了"没修好"。

用法：
    python count_probe/decompose.py --arms $SD_OUT/count/cocoount_arms
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _pct(a, b):
    return f"{100.0*a/b:5.1f}%" if b else "   n/a"


def _mcnemar(b, c):
    """McNemar 的精确二项检验，双尾。不引 scipy —— 只用到组合数。

    H0：在"修正改变了对错"的那些题里，修坏与修好等概率（p=0.5）。
    n=b+c 很小时它检不出任何东西，这正是我们要看见的事实。
    """
    from math import comb
    n = b + c
    if n == 0:
        return "  n/a"
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return f"{min(p, 1.0):7.3f}"


def _row(label, ok, n, extra=""):
    print(f"{label:<26}{n:>6}{_pct(ok, n):>9}  {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    p = Path(a.arms) / "yolo_results.csv"
    if not p.exists():
        raise SystemExit(f"!! 缺 {p} —— 先跑 yolo_eval.py")

    def _i(v):
        return None if v in ("", "None") else int(float(v))

    rows = []
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue                       # N>9，官方整题跳过，单列
        r = {**r, "N": int(r["N"]), "n_dbscan": _i(r["n_dbscan"]),
             "ok_v": _i(r["ok_vanilla"]), "ok_c": _i(r["ok_countgen"]),
             "y_v": _i(r["yolo_vanilla"]), "y_c": _i(r["yolo_countgen"]),
             "match": r["obj_num_match"] in ("True", "true", "1")}
        if r["ok_v"] is None or r["ok_c"] is None or r["n_dbscan"] is None:
            continue
        rows.append(r)

    fixed = [r for r in rows if not r["match"]]          # 真正进了修正的
    print(f"共 {len(rows)} 题（N≤9），其中 {len(fixed)} 题进了修正、"
          f"{len(rows)-len(fixed)} 题被计数器判为已匹配、原样输出\n")

    # ---- 1. 方向 ----
    print("=" * 66)
    print("表四 修正成功率 × 方向（这两条在代码里是两套完全不同的机制）")
    print(f"{'方向':<26}{'题数':>6}{'修正后正确':>9}")
    over = [r for r in fixed if r["n_dbscan"] > r["N"]]
    under = [r for r in fixed if r["n_dbscan"] < r["N"]]
    _row("数多了 → 删最小的 blob", sum(r["ok_c"] for r in over), len(over),
         "relayout_overgeneration，纯启发式、无模型")
    _row("数少了 → ReLayout U-Net", sum(r["ok_c"] for r in under), len(under),
         "relayout_undergeneration，用了那 474MB 权重")
    print(f"\n对照：原版图在这两类上的正确率")
    _row("  数多了（vanilla）", sum(r["ok_v"] for r in over), len(over))
    _row("  数少了（vanilla）", sum(r["ok_v"] for r in under), len(under))

    # ---- 2. 幅度 ----
    print("\n" + "=" * 66)
    print("表五 修正成功率 × 要改动的幅度 |n_dbscan − N|")
    print(f"{'|Δ|':<26}{'题数':>6}{'修正后正确':>9}")
    by_d = defaultdict(list)
    for r in fixed:
        d = abs(r["n_dbscan"] - r["N"])
        by_d[min(d, 5)].append(r)          # ≥5 合并，否则尾部太稀
    for d in sorted(by_d):
        lab = f"{d}" if d < 5 else "≥5"
        _row(lab, sum(r["ok_c"] for r in by_d[d]), len(by_d[d]))

    # ---- 3. N ----
    print("\n" + "=" * 66)
    print("表六 修正成功率 × 要求个数 N")
    print(f"{'N':<26}{'题数':>6}{'修正后正确':>9}{'  同题 vanilla':>14}")
    by_n = defaultdict(list)
    for r in fixed:
        by_n[r["N"]].append(r)
    for n in sorted(by_n):
        rs = by_n[n]
        print(f"{n:<26}{len(rs):>6}{_pct(sum(r['ok_c'] for r in rs), len(rs)):>9}"
              f"{_pct(sum(r['ok_v'] for r in rs), len(rs)):>14}")

    # ---- 3.5 分方向的配对表：净增益到底是不是噪声 ----
    print("\n" + "=" * 66)
    print("表六b 分方向的**配对**变化（准确率之差看不出样本量，配对数才看得出）")
    print(f"{'方向':<22}{'题数':>5}{'错→对':>7}{'对→错':>7}{'净':>6}{'双尾 p':>9}")
    for tag, g in (("数多了（删 blob）", over), ("数少了（U-Net）", under)):
        b = sum(1 for r in g if r["ok_v"] and not r["ok_c"])      # 修坏
        c = sum(1 for r in g if not r["ok_v"] and r["ok_c"])      # 修好
        print(f"{tag:<22}{len(g):>5}{c:>7}{b:>7}{c-b:>6}{_mcnemar(b, c):>9}")
    print("p 是 McNemar 的精确二项检验（H0：修好与修坏等概率）。"
          "b+c 太小时它本来就检不出东西，这一点要老实说。")

    # ---- 4. 修正的净效果 ----
    print("\n" + "=" * 66)
    print("表七 修正的净收益（只看真正进了修正的 %d 题）" % len(fixed))
    gain = [r for r in fixed if not r["ok_v"] and r["ok_c"]]
    harm = [r for r in fixed if r["ok_v"] and not r["ok_c"]]
    keep_ok = [r for r in fixed if r["ok_v"] and r["ok_c"]]
    keep_bad = [r for r in fixed if not r["ok_v"] and not r["ok_c"]]
    n = len(fixed)
    print(f"{'原版错 → 修正后对（净赚）':<30}{len(gain):>5}{_pct(len(gain), n):>9}")
    print(f"{'原版对 → 修正后错（修坏了）':<30}{len(harm):>5}{_pct(len(harm), n):>9}")
    print(f"{'原版对 → 修正后仍对':<30}{len(keep_ok):>5}{_pct(len(keep_ok), n):>9}")
    print(f"{'原版错 → 修正后仍错':<30}{len(keep_bad):>5}{_pct(len(keep_bad), n):>9}")
    print(f"\n净收益 = {len(gain)} − {len(harm)} = {len(gain)-len(harm)} 题"
          f"（占全部 {len(rows)} 题的 {_pct(len(gain)-len(harm), len(rows)).strip()}）")
    if harm:
        print(f"⚠️ 有 {len(harm)} 题是**被修坏的** —— 计数器说它不匹配，其实原版就是对的。"
              f"\n   这一项在表三里被并进了「没修好」，单看那张表看不出来。")

    # ---- 5. 计数器的两类错误 ----
    print("\n" + "=" * 66)
    print("表八 DBSCAN 计数器本身（以 YOLO 在原版图上的读数为参照）")
    same = [r for r in rows if r["n_dbscan"] == r["y_v"]]
    print(f"逐题一致 {_pct(len(same), len(rows))}，"
          f"MAE {sum(abs(r['n_dbscan']-r['y_v']) for r in rows)/len(rows):.2f}")
    hi = sum(1 for r in rows if r["n_dbscan"] > r["y_v"])
    lo = sum(1 for r in rows if r["n_dbscan"] < r["y_v"])
    print(f"数多了 {hi} 题（{_pct(hi, len(rows)).strip()}），"
          f"数少了 {lo} 题（{_pct(lo, len(rows)).strip()}）"
          f" —— 偏向哪边决定了它更容易触发哪套修正机制")
    print("注意：YOLO 自己也会错，这里只是两个数数器互比，当相对读数看。")


if __name__ == "__main__":
    main()
