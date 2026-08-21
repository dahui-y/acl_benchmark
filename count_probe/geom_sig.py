#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
个体化假说的几何签名检验（MOTIVATION.md §4 的三条预言）。零 GPU，只读盘。

    要判的事：我们把「布局对了 97/98，最终只有 41.8% 数对」解释成**个体化
    失败**（一区多物 / 两区并一物）。这个解释必须能生出**新的**预言，否则
    就是给数据编故事。

    与之对立的平庸假说 H0：「这些误差只是检测器噪声」。两者在几何上相反：

      P1  裂开 vs 区域面积          个体化：随面积上升   H0：无预言
      P2  空着 vs 与最近邻区距离    个体化：越近越空     H0：无预言
      P3  空着 vs 区域面积          个体化：无强预言     H0：小面积更空

    P1 是主判据（MOTIVATION §4）：Spearman ρ>0 且置换 p<0.05 才算个体化表述
    成立；否则该表述作废，退回较弱的「三个误差通道」诊断式表述。

    ⚠️ 必须控制 N：N 越大区域越小越拥挤，面积与距离都与 N 混杂。所有相关
    都在 N 分层内算秩，再合并（分层 Spearman），置换也在层内做。

用法：
    python count_probe/geom_sig.py --run $SD_OUT/count/cocoount2 \\
        --arms $SD_OUT/count/cocoount2_arms \\
        --boxes $SD_OUT/count/cocoount2_arms/v9e_boxes_countgen.jsonl
    python count_probe/geom_sig.py --self-test
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from blob_split import to_labels, assign, assign_centroid


def blob_geom(lab):
    """→ {blob: (面积格数, 到最近邻 blob 的切比雪夫距离)}。无邻居时距离记 inf。"""
    k = int(lab.max())
    cells = {j: np.argwhere(lab == j) for j in range(1, k + 1)}
    cells = {j: c for j, c in cells.items() if len(c)}
    out = {}
    for j, cj in cells.items():
        best = float("inf")
        for i, ci in cells.items():
            if i == j:
                continue
            d = np.abs(cj[:, None, :] - ci[None, :, :]).max(-1).min()
            best = min(best, float(d))
        out[j] = (len(cj), best)
    return out


def _rank(x):
    """平均秩（处理并列），纯 numpy。"""
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    # 并列取平均秩
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def strat_spearman(groups, n_perm=20000, seed=0):
    """分层 Spearman：层内取秩、层内中心化，再合并求相关；置换也在层内做。

    groups: [(x_array, y_array), ...]，每层一个。层内 <3 个样本的丢弃。
    → (rho, p, 有效层数, 有效样本数)
    """
    xs, ys = [], []
    for x, y in groups:
        if len(x) < 3:
            continue
        rx, ry = _rank(x), _rank(y)
        xs.append(rx - rx.mean())
        ys.append(ry - ry.mean())
    if not xs:
        return float("nan"), float("nan"), 0, 0
    n_lay, n_all = len(xs), sum(len(x) for x in xs)

    def corr(a_list, b_list):
        a = np.concatenate(a_list); b = np.concatenate(b_list)
        da, db = a.std(), b.std()
        return float((a * b).mean() / (da * db)) if da > 0 and db > 0 else 0.0

    rho = corr(xs, ys)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_perm):
        sh = [rng.permutation(y) for y in ys]        # 层内打乱，保持分层结构
        if abs(corr(xs, sh)) >= abs(rho):
            hit += 1
    return rho, (hit + 1) / (n_perm + 1), n_lay, n_all


def collect(run, arms, boxes, img_size, rule):
    """→ 每个 blob 一条记录 (N, 面积, 邻距, 框数)。只取经过布局引导的题。"""
    rows = {r["stem"]: r for r in csv.DictReader((Path(arms) / "yolo_results.csv").open())
            if r["skipped_by_official"] not in ("True", "true", "1")}
    box = {json.loads(l)["stem"]: json.loads(l) for l in Path(boxes).open()}
    out = []
    for stem, r in rows.items():
        p = Path(run) / f"{stem}_masks.npz"
        if not p.exists() or stem not in box:
            continue
        if r["obj_num_match"] in ("True", "true", "1"):
            continue                                  # 原样返回 vanilla，无引导
        lab = to_labels(np.load(p)["postprocess"])
        k = int(lab.max())
        if k == 0:
            continue
        geom = blob_geom(lab)
        cnt = defaultdict(int)
        if rule == "centroid":
            _, adj = assign_centroid(box[stem]["boxes"], lab, (img_size, img_size))
            for hit in adj:                           # 代表点落在框内即算该 blob 有一个
                for j in hit:
                    cnt[j] += 1
        else:
            c, _, _, _ = assign(box[stem]["boxes"], lab, (img_size, img_size),
                                min_cover=0.0)
            cnt.update(c)
        N = int(r["N"])
        for j, (area, dist) in geom.items():
            out.append((N, area, dist, cnt.get(j, 0)))
    return out


def report(recs, tag):
    by_n = defaultdict(list)
    for N, area, dist, n in recs:
        by_n[N].append((area, dist, n))
    print(f"\n{'='*72}\n【归属规则 = {tag}】blob 总数 {len(recs)}，"
          f"N 分层 {sorted(by_n)}")
    print(f"  裂开(≥2 框) {sum(1 for r in recs if r[3] >= 2)}　"
          f"空着(0 框) {sum(1 for r in recs if r[3] == 0)}　"
          f"恰好 1 框 {sum(1 for r in recs if r[3] == 1)}")

    def run(name, xf, yf, pred):
        g = []
        for N, v in by_n.items():
            x = [xf(t) for t in v]
            y = [yf(t) for t in v]
            keep = [i for i, xx in enumerate(x) if np.isfinite(xx)]
            if len(keep) >= 3:
                g.append((np.array([x[i] for i in keep]),
                          np.array([y[i] for i in keep])))
        rho, p, nl, na = strat_spearman(g)
        mark = "✓" if (np.isfinite(rho) and pred(rho) and p < 0.05) else "✗"
        print(f"  {mark} {name:<28} ρ={rho:+.3f}  p={p:.4f}  "
              f"（{nl} 层 / {na} 个 blob）")
        return rho, p

    print("  预言（MOTIVATION §4，见数前写死）：")
    p1 = run("P1 裂开 vs 面积（ρ>0）", lambda t: t[0], lambda t: int(t[2] >= 2),
             lambda r: r > 0)
    p2 = run("P2 空着 vs 邻距（ρ<0）", lambda t: t[1], lambda t: int(t[2] == 0),
             lambda r: r < 0)
    p3 = run("P3 空着 vs 面积（ρ<0=H0）", lambda t: t[0], lambda t: int(t[2] == 0),
             lambda r: r < 0)
    # P4：物体数 vs 面积（密度填充）
    p4 = run("P4 框数 vs 面积（ρ>0）", lambda t: t[0], lambda t: t[2],
             lambda r: r > 0)
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def _self_test():
    ok = True
    # blob_geom：两个 3x3 方块，中间隔 2 格 → 切比雪夫最近距离 3
    lab = np.zeros((16, 16), dtype=int)
    lab[2:5, 2:5] = 1
    lab[2:5, 7:10] = 2
    g = blob_geom(lab)
    ok &= g[1][0] == 9 and g[2][0] == 9
    print(f"  {'✓' if g[1][0]==9 else '✗'} 面积: {g[1][0]}, {g[2][0]}（期望 9, 9）")
    ok &= abs(g[1][1] - 3) < 1e-9
    print(f"  {'✓' if abs(g[1][1]-3)<1e-9 else '✗'} 邻距: {g[1][1]}（期望 3，"
          f"列 4→7）")
    # 单 blob → inf
    lab2 = np.zeros((8, 8), dtype=int); lab2[1:3, 1:3] = 1
    ok &= blob_geom(lab2)[1][1] == float("inf")
    print(f"  {'✓' if blob_geom(lab2)[1][1]==float('inf') else '✗'} 单 blob 邻距 = inf")

    # strat_spearman：层内构造完全正相关 → ρ≈1、p 极小
    rng = np.random.default_rng(0)
    g1 = [(np.arange(10, dtype=float), np.arange(10, dtype=float)) for _ in range(3)]
    rho, p, nl, na = strat_spearman(g1, n_perm=2000)
    ok &= rho > 0.99 and p < 0.01
    print(f"  {'✓' if rho>0.99 and p<0.01 else '✗'} 完全正相关: ρ={rho:.3f} p={p:.4f}")
    # 完全负相关
    g2 = [(np.arange(10, dtype=float), -np.arange(10, dtype=float)) for _ in range(3)]
    rho2, p2, _, _ = strat_spearman(g2, n_perm=2000)
    ok &= rho2 < -0.99 and p2 < 0.01
    print(f"  {'✓' if rho2<-0.99 else '✗'} 完全负相关: ρ={rho2:.3f} p={p2:.4f}")
    # 纯噪声 → 不显著
    g3 = [(rng.normal(size=12), rng.normal(size=12)) for _ in range(4)]
    rho3, p3, _, _ = strat_spearman(g3, n_perm=2000, seed=1)
    ok &= p3 > 0.05
    print(f"  {'✓' if p3>0.05 else '✗'} 纯噪声: ρ={rho3:+.3f} p={p3:.4f}（期望不显著）")
    # ★ 分层必要性：层间有强趋势、层内无趋势 —— 不分层会假阳，分层应当不显著
    g4 = [(np.full(10, s, dtype=float) + rng.normal(0, .01, 10),
           np.full(10, s, dtype=float) + rng.normal(0, .01, 10)) for s in range(1, 5)]
    rho4, p4, _, _ = strat_spearman(g4, n_perm=2000, seed=2)
    print(f"  {'✓' if p4>0.05 else '✗'} 层间趋势/层内无: ρ={rho4:+.3f} p={p4:.4f}"
          f"（期望不显著 —— 这正是控制 N 的意义）")
    ok &= p4 > 0.05
    print("\n全部通过 ✓" if ok else "\n有失败项 ✗")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run"); ap.add_argument("--arms"); ap.add_argument("--boxes")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    if not (a.run and a.arms and a.boxes):
        ap.error("--run / --arms / --boxes 三者都要给（或用 --self-test）")

    res = {}
    for rule in ("cover", "centroid"):
        recs = collect(a.run, a.arms, a.boxes, a.img_size, rule)
        res[rule] = report(recs, rule)

    print(f"\n{'='*72}\n判读（MOTIVATION §4，见数前定死）")
    p1c, p1n = res["cover"]["P1"], res["centroid"]["P1"]
    good = all(np.isfinite(r) and r > 0 and p < 0.05 for r, p in (p1c, p1n))
    print(f"  P1 主判据：cover ρ={p1c[0]:+.3f}(p={p1c[1]:.4f})　"
          f"centroid ρ={p1n[0]:+.3f}(p={p1n[1]:.4f})")
    print("  → " + ("**个体化表述成立**（两条规则均 ρ>0 且 p<0.05）"
                    if good else
                    "**个体化表述作废** —— 退回「三个误差通道」的诊断式表述，"
                    "不得改口"))
    print("  注：只在一条规则上成立不算（MOTIVATION §4 的混杂与边界）。")


if __name__ == "__main__":
    main()
