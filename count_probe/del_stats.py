#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯删除修正器的**环视角**统计：p_del（按 k 分层）、t_edit、改坏/修好通道。零 GPU。

    为什么总分不够用：yolo_eval 的表一只给一个准确率，但修正器只碰
    「v8x 说多了」的题（集合 D），其余题**逐字节等于 vanilla**。总分里
    混着三块来源完全不同的贡献：

        总分 = 不动题(v8x=N) 的自然正确 + 要补题(v8x<N) 的自然正确
               + D 上 p_del × t_edit 的兑现

    只有第三块是方法的功劳。DESIGN.md §4 的生死判据要的正是 p_del，
    而 §5.6 指出的失败机制（密集簇 k 大时框 mask 表达不出意图）预测
    p_del 应随 k 单调下降 —— 所以必须**按 k 分层**，否则一个「D 集恰好
    全是 k=1」的巧合会让调参集看着很美而评测集全泡汤。上一轮方向 a
    就是这么死的（调参集 +6.7 → 评测集 p=1.000）。

    同时做三件防自欺的核对：
      · 各跑批的 vanilla 臂逐题一致（不一致 = 泄漏，先查这个）
      · 未编辑题的方法臂读数必须等于 vanilla 读数（不等 = 管道污染）
      · p_del 是 **v8x 自己说自己修好了**（自评），必须与 t_edit
        （评测器 v9e 的承认率）分开报，绝不合并成一个「成功率」

用法：
    python count_probe/del_stats.py --runs $SD_OUT/count/tune2_del_s10ctx \
        $SD_OUT/count/tune2_del_s10bg $SD_OUT/count/tune2_del_s08ctx \
        $SD_OUT/count/tune2_del_s08bg --cg $SD_OUT/count/tune2_cg
"""

import argparse
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def mcnemar(b, c):
    """McNemar 精确二项检验，双尾。与 summary.py 同一实现。"""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def _i(v):
    return None if v in ("", "None", None) else int(float(v))


def _carved(box, keeps, grid=64):
    """待删框 box 有多大比例被 keeps（保留框，不膨胀）挖掉。

    用 grid×grid 的均匀采样近似面积并集，避免为了矩形并集写扫描线 ——
    这里只需要判「大约挖掉了多少」，0.01 的精度足够分档。
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    hit = 0
    for i in range(grid):
        px = x1 + (i + 0.5) * w / grid
        for j in range(grid):
            py = y1 + (j + 0.5) * h / grid
            if any(k[0] <= px <= k[2] and k[1] <= py <= k[3] for k in keeps):
                hit += 1
    return hit / (grid * grid)


def load_arms(run):
    """{stem: row}，只留计分题（N≤9）。arms 目录约定是 {run}_arms。"""
    p = Path(str(run) + "_arms") / "yolo_results.csv"
    if not p.exists():
        return None
    out = {}
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        y_v, y_m = _i(r["yolo_vanilla"]), _i(r.get("yolo_countgen"))
        if y_v is None or y_m is None:
            continue
        out[r["stem"]] = dict(N=int(r["N"]), y_v=y_v, y_m=y_m,
                              ok_v=int(y_v == int(r["N"])),
                              ok_m=int(y_m == int(r["N"])))
    return out


def load_log(run):
    """{id: rec}。counter_log 里 n_dbscan 对修正器而言是 v8x 初检数。"""
    p = Path(run) / "counter_log.jsonl"
    out = {}
    for l in p.open():
        r = json.loads(l)
        out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="修正器跑批目录")
    ap.add_argument("--cg", default=None, help="同批 CountGen 基线跑批目录")
    a = ap.parse_args()

    runs = [Path(r) for r in a.runs]
    arms = {r.name: load_arms(r) for r in runs}
    logs = {r.name: load_log(r) for r in runs}
    bad = [n for n, v in arms.items() if not v]
    if bad:
        print(f"⚠️ 读不到 yolo_results.csv：{bad}")
        return
    cg = load_arms(Path(a.cg)) if a.cg else None

    # ---- 核对 1：各跑批的 vanilla 臂必须逐题一致 ----
    names = list(arms)
    ref = arms[names[0]]
    print(f"{'='*74}\n核对 1 各跑批 vanilla 臂逐题一致性（同 prompt 同 seed，本应完全相同）")
    all_ok = True
    for n in names[1:] + ([Path(a.cg).name] if cg else []):
        o = arms.get(n) or cg
        common = set(ref) & set(o)
        d = [s for s in common if ref[s]["y_v"] != o[s]["y_v"]]
        all_ok &= not d
        print(f"  {n:<22} 公共 {len(common):>3} 题，vanilla 读数不同 {len(d):>2} 题"
              + ("  ✓" if not d else f"  ⚠️ 例：{d[:3]}"))
    if not all_ok:
        print("  ⚠️ vanilla 对不上 → 配对检验的前提不成立，先查这个再看下面所有数。")

    # ---- 核对 2：未编辑题的方法臂必须等于 vanilla ----
    print(f"\n核对 2 未编辑题（修正器没碰过的）方法臂 == vanilla 臂")
    for n in names:
        L, A = logs[n], arms[n]
        untouched = [s for s in A if not L.get(s, {}).get("corrector", {}).get("want")]
        d = [s for s in untouched if A[s]["y_m"] != A[s]["y_v"]]
        print(f"  {n:<22} 未编辑 {len(untouched):>3} 题，读数不同 {len(d):>2} 题"
              + ("  ✓" if not d else f"  ⚠️ 例：{d[:3]}"))

    # ---- 头寸与总分分解 ----
    print(f"\n{'='*74}\n表一 总分从哪来（v8x 视角三分头寸；只有 D 那一列是方法的功劳）")
    print(f"{'配置':<16}{'总分':>7}{'不动题':>13}{'要补题':>13}{'要删题 D':>15}")
    print(f"{'':16}{'':7}{'对/题(自然)':>13}{'对/题(自然)':>13}{'对/题(方法)':>15}")
    heads = {}
    for n in names:
        L, A = logs[n], arms[n]
        eq, un, D = [], [], []
        for s, r in A.items():
            n0 = L[s]["n_dbscan"]
            (eq if n0 == r["N"] else D if n0 > r["N"] else un).append(s)
        heads[n] = (eq, un, D)
        tot = sum(A[s]["ok_m"] for s in A)
        f = lambda g: f"{sum(A[s]['ok_m'] for s in g):>3}/{len(g):<3}"
        print(f"{n:<16}{100*tot/len(A):>6.1f}%{f(eq):>13}{f(un):>13}{f(D):>15}")
    if cg:
        tot = sum(cg[s]["ok_m"] for s in cg)
        print(f"{Path(a.cg).name:<16}{100*tot/len(cg):>6.1f}%"
              f"{'（CountGen 的头寸按 DBSCAN 划分，与上面不可比，故不拆）':>13}")
    van = 100.0 * sum(ref[s]["ok_v"] for s in ref) / len(ref)
    print(f"{'vanilla':<16}{van:>6.1f}%")

    # ---- p_del 按 k 分层 ----
    print(f"\n{'='*74}\n表二 环内删除成功率 p_del（v8x 自评！）按 k = 初检 − N 分层")
    print(f"{'配置':<16}{'D 题数':>7}{'p_del':>8}   " +
          "  ".join(f"k={k}" for k in (1, 2, 3)) + "   k≥4")
    for n in names:
        L, A = logs[n], arms[n]
        by = defaultdict(lambda: [0, 0])
        for s in heads[n][2]:
            c = L[s]["corrector"]
            k = L[s]["n_dbscan"] - A[s]["N"]
            hit = int(c["n_trail"][-1] == A[s]["N"])
            kk = k if k <= 3 else 4
            by[kk][0] += hit
            by[kk][1] += 1
        tot_h = sum(v[0] for v in by.values())
        tot_n = sum(v[1] for v in by.values())
        cells = "  ".join(f"{by[k][0]}/{by[k][1]:<3}" if by[k][1] else " –  "
                          for k in (1, 2, 3, 4))
        print(f"{n:<16}{tot_n:>7}{100*tot_h/max(tot_n,1):>7.1f}%   {cells}")
    print("  ⚠️ 这是**环内计数器说自己修好了**，即自评。评测器认不认见表三。")

    # ---- t_edit：编辑图上的换算率 ----
    print(f"\n{'='*74}\n表三 t_edit = P(v9e=N | 环内说=N)，编辑图上的换算率")
    print(f"{'配置':<16}{'环内说够':>9}{'v9e 认':>8}{'t_edit':>9}"
          f"{'环内没修成但 v9e 说对':>20}")
    for n in names:
        L, A = logs[n], arms[n]
        say, hit, luck_n, luck = 0, 0, 0, 0
        for s in heads[n][2]:
            c = L[s]["corrector"]
            if c["n_trail"][-1] == A[s]["N"]:
                say += 1
                hit += A[s]["ok_m"]
            else:
                luck_n += 1
                luck += A[s]["ok_m"]
        print(f"{n:<16}{say:>9}{hit:>8}{100*hit/max(say,1):>8.1f}%"
              f"{f'{luck}/{luck_n}':>20}")
    print("  参照：未编辑的 vanilla 上 P(v9e=N | v8x=N) = 85.7%（评测集 n=70）。"
          "\n  编辑图只会更低；低多少就是「重去噪把图改花了」的代价。")

    # ---- 改坏 / 修好通道 ----
    print(f"\n{'='*74}\n表四 编辑通道的净账（只看 D，其余题逐字节没动）")
    print(f"{'配置':<16}{'编辑题':>7}{'修好':>6}{'改坏':>6}{'净':>6}{'McNemar p':>11}")
    for n in names:
        L, A = logs[n], arms[n]
        ed = [s for s in heads[n][2] if L[s]["corrector"]["want"]]
        fix = sum(1 for s in ed if A[s]["ok_m"] and not A[s]["ok_v"])
        brk = sum(1 for s in ed if A[s]["ok_v"] and not A[s]["ok_m"])
        print(f"{n:<16}{len(ed):>7}{fix:>6}{brk:>6}{fix-brk:>+6}"
              f"{mcnemar(brk, fix):>11.3f}")

    # ---- 逐题配对：对 vanilla、对 CountGen ----
    print(f"\n{'='*74}\n表五 逐题配对 McNemar（同 prompt 同 seed）")
    print(f"{'配置':<16}{'vs vanilla':>22}{'vs CountGen':>24}")
    for n in names:
        A = arms[n]
        b = sum(1 for s in A if A[s]["ok_m"] and not A[s]["ok_v"])
        c = sum(1 for s in A if A[s]["ok_v"] and not A[s]["ok_m"])
        cell1 = f"{b}:{c}  p={mcnemar(c, b):.3f}"
        cell2 = "—"
        if cg:
            com = sorted(set(A) & set(cg))
            b2 = sum(1 for s in com if A[s]["ok_m"] and not cg[s]["ok_m"])
            c2 = sum(1 for s in com if cg[s]["ok_m"] and not A[s]["ok_m"])
            cell2 = f"{b2}:{c2}  p={mcnemar(c2, b2):.3f}  (n={len(com)})"
        print(f"{n:<16}{cell1:>22}{cell2:>24}")
    print("\n  提醒：这是**调参集**。四个配置里挑最好的那个，其优势含选择偏倚；"
          "\n  「四个全部为正」比「最好的那个为正」结实得多，判读要以前者为准。")

    # ---- 遮罩取证：被删物体有多少被保留框「挖回去」了 ----
    if any(L[s].get("corrector", {}).get("del_boxes")
           for L in logs.values() for s in L):
        print(f"\n{'='*74}\n表六 遮罩取证：待删框被保留框挖掉的比例（嵌合体的成因）")
        print("  挖回保留框是为了不吃掉邻居，代价是：待删框与保留框重叠时，"
              "\n  被删物体的一部分也被保护下来 → 重去噪只重画上半截 → "
              "模型顺着残留补成别的东西（半个杯子长成狗）。")
        print(f"\n{'配置':<16}{'待删框数':>9}{'被挖过的':>9}{'挖掉>30%':>9}"
              f"{'中位挖掉':>9}{'最大':>7}")
        for n in names:
            L = logs[n]
            fr = []
            for s, r in L.items():
                c = r.get("corrector", {})
                for dele, keep in zip(c.get("del_boxes", []),
                                      c.get("keep_boxes", [])):
                    for b in dele:
                        fr.append(_carved(b, keep))
            if not fr:
                continue
            fr.sort()
            hit = sum(1 for v in fr if v > 0.01)
            big = sum(1 for v in fr if v > 0.30)
            print(f"{n:<16}{len(fr):>9}{hit:>9}{big:>9}"
                  f"{fr[len(fr)//2]:>9.2f}{fr[-1]:>7.2f}")
        print("  判读：'挖掉>30%' 的框基本注定长成嵌合体 —— 一半原物体、"
              "一半新内容。\n  这个数不是超参没调好，是轴对齐矩形框表达不了"
              "「删这个、不碰那个」（DESIGN §6 风险 #3）。")


if __name__ == "__main__":
    main()
