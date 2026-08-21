#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 CountGen 的计数误差摊派到三个可命名的机制上。只读盘（需先落盘检测框）。

    动机（FINDINGS §7.3）：CountGen 的两个强制机制**都只认前景/背景，
    不认实例数**——
      · `utils/loss_utils.py:5`  foreground_mask = (desired_mask != 0)
        布局的实例编号被二值化丢弃
      · `attention_processors.py:45-61`  只置零「背景 → 所有 blob 的并集」，
        不置零「blob_i → blob_j」（论文 §3.3 公式同：i∈B, j∈F，F 是前景整体）
    所以约束表达的是「长在这里、别长在那里」，从没表达「这里恰好长一个」。
    论文 §7 Limitations 第一句自己承认了后果：
      "results in multiple instances of an object in an area intended for
       just one by the layout"

    本脚本量的就是这个后果占多大。设布局有 k 个 blob，评测器数出 yolo 个框，
    每个框归到重叠最大的 blob，则**恒等式**：

        yolo − k = 多长 − 空blob + 背景漏

      多长   Σ_blob max(cnt−1, 0)  一个 blob 里落了 ≥2 个框 ← 论文承认的失效
      空blob #{blob: cnt == 0}     布局给了位置却没长出来 ← 布局损失该管的
      背景漏 #{框: 不落在任何 blob} 长到了布局之外     ← 自注意力遮罩该管的

    三项互斥且穷尽（每个框至多归一个 blob），所以这是摊派而不是估计。

    ⚠️ 两处会让这张表失真的地方，都已内建对策：
    1. **归属摇摆**。一个框在 A、B 之间摇摆：判给 A（A 已有框）是「多长+1、
       空+1」，判给 B（B 空着）是「0、0」。一次摇摆同时改两项各 1，所以
       「多长」与「空blob」是耦合的，不能各自当独立读数。
       → 对策：`max_matching` 给「多长」求一个**与摇摆无关的下界**（表三）。
    2. **归属过松**。只要有一格重叠就判给该 blob，会把「基本坐在背景上、
       只蹭到 blob 一角」的框算成重复生成，同时虚高多长、虚低背景漏。
       → 对策：`--min-cover` 要求覆盖比例达标，并做敏感性扫描；结论必须在
         整条曲线上都成立才算数。

    其它边界：评测器自身会错（DBSCAN 与 YOLOv9e 一致率只有 64.1%），这里把
    YOLOv9e 当口径是因为主表就是它，但它不是真值；`obj_num_match=True` 的题
    原样返回 vanilla、没经过布局引导，单列不混算。

用法：
    # 1) 先把 CountGen 臂的 YOLOv9e 框落盘（一次检测遍历，约 2 分钟）
    python count_probe/det_counts.py --arms $SD_OUT/count/cocoount_arms \\
        --arm countgen --weights yolov9e.pt \\
        --out $SD_OUT/count/cocoount_arms/v9e_boxes_countgen.jsonl
    # 2) 摊派（默认对 min-cover 做 0 / 0.15 / 0.30 三点扫描）
    python count_probe/blob_split.py --run $SD_OUT/count/cocoount \\
        --arms $SD_OUT/count/cocoount_arms \\
        --boxes $SD_OUT/count/cocoount_arms/v9e_boxes_countgen.jsonl
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def to_labels(m):
    """把 npz 里的布局统一成 (H,W) 的整数标签图，0=背景，1..k=实例。

    countgen_batch 存的 `postprocess` 已经是标签图（run_countgen 用
    `int(object_masks.max())` 当实例数）。但 relayout 的两条分支返回过
    (k, H*W) 的通道形式，所以这里兼容三种形状，避免静默算错。
    """
    m = np.asarray(m)
    if m.ndim == 2 and m.shape[0] == m.shape[1]:
        return m.astype(int)
    if m.ndim == 2:
        k, hw = m.shape
        s = int(round(hw ** 0.5))
        ch = m.reshape(k, s, s)
    elif m.ndim == 3:
        ch = m
    else:
        raise ValueError(f"看不懂的布局形状 {m.shape}")
    lab = np.zeros(ch.shape[1:], dtype=int)
    for i in range(ch.shape[0]):
        lab[ch[i] > 0.5] = i + 1
    return lab


def max_matching(adj, k):
    """框-blob 二分图最大匹配（每个 blob 至多配一个框）。→ 匹配数。

    用途：给「多长」求一个**与归属摇摆无关的下界**。把每个框尽量塞进一个
    还空着的 blob，是最有利于「没有重复生成」的那种分配；这种分配下仍然
    剩下的多长（= 有归属的框数 − 匹配数），任何别的分配都消不掉。
    规模极小（框 ≤ 二十几、blob ≤ 10），朴素增广路足够。
    """
    match = {}

    def aug(b, seen):
        for j in adj[b]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match or aug(match[j], seen):
                match[j] = b
                return True
        return False

    return sum(1 for b in range(len(adj)) if adj[b] and aug(b, set()))


def assign(boxes, lab, img_size, amb_frac=0.30, min_cover=0.0):
    """每个框归到覆盖格数最多的 blob。→ (每 blob 框数, 背景框数, 模糊框数, 邻接表)

    模糊框 = 第二名 blob 的覆盖也达到第一名的 amb_frac 以上，是归属噪声的
    体量指示。min_cover = 落在某 blob 上的格数占框面积的最低比例，不达标的
    blob 不参与归属；全部不达标则该框算背景。
    """
    H, W = lab.shape
    sy, sx = img_size[1] / H, img_size[0] / W
    k = int(lab.max())
    cnt = Counter()
    bg = amb = 0
    adj = []
    for x1, y1, x2, y2 in boxes:
        c0, r0 = int(np.floor(x1 / sx)), int(np.floor(y1 / sy))
        c1, r1 = int(np.ceil(x2 / sx)), int(np.ceil(y2 / sy))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(max(c1, c0 + 1), W), min(max(r1, r0 + 1), H)
        sub = lab[r0:r1, c0:c1]
        area = max(sub.size, 1)
        ov = np.bincount(sub.ravel(), minlength=k + 1).astype(float)
        ov[0] = 0.0
        ov = np.where(ov / area >= min_cover, ov, 0.0)
        if ov.sum() == 0:
            bg += 1
            continue
        cnt[int(ov.argmax())] += 1
        adj.append([j for j in range(1, k + 1) if ov[j] > 0])
        rest = np.sort(ov)[::-1]
        if len(rest) > 1 and rest[1] >= amb_frac * rest[0]:
            amb += 1
    return cnt, bg, amb, adj


def tally(rows, box, run, img_size, amb_frac, min_cover):
    """跑一遍摊派。→ (stat, agg, per)"""
    stat = Counter()
    agg = Counter()
    per = []
    for stem, r in rows.items():
        p = run / f"{stem}_masks.npz"
        if not p.exists():
            stat["缺npz"] += 1
            continue
        if stem not in box:
            stat["缺框"] += 1
            continue
        if r["obj_num_match"] in ("True", "true", "1"):
            stat["未引导"] += 1
            continue
        lab = to_labels(np.load(p)["postprocess"])
        k = int(lab.max())
        if k == 0:
            stat["缺npz"] += 1
            continue
        cnt, bg, amb, adj = assign(box[stem]["boxes"], lab,
                                   (img_size, img_size), amb_frac, min_cover)
        over = sum(max(c - 1, 0) for c in cnt.values())
        empty = sum(1 for j in range(1, k + 1) if cnt.get(j, 0) == 0)
        yolo = len(box[stem]["boxes"])
        assert yolo - k == over - empty + bg, (stem, yolo, k, over, empty, bg)
        N = int(r["N"])
        m = max_matching(adj, k)
        stat["用"] += 1
        agg.update({"题": 1, "blob": k, "多长": over, "空blob": empty,
                    "背景漏": bg, "模糊框": amb,
                    "多长下界": len(adj) - m, "空下界": k - m,
                    "布局≠N": int(k != N), "数对": int(yolo == N)})
        if yolo != N:
            agg.update({"错题": 1, "错·多长": over, "错·空blob": empty,
                        "错·背景漏": bg,
                        "错·有多长": int(over > 0),
                        "错·有空blob": int(empty > 0),
                        "错·有背景漏": int(bg > 0),
                        "错·有多长下界": int(len(adj) - m > 0)})
        per.append((stem, N, k, yolo, over, empty, bg, amb, len(adj) - m))
    return stat, agg, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="含 {stem}_masks.npz 的跑批目录")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--boxes", required=True, help="该臂的检测框 jsonl")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--amb-frac", type=float, default=0.30)
    ap.add_argument("--min-cover", type=float, nargs="+", default=[0.0, 0.15, 0.30],
                    help="框面积中落在该 blob 上的最低比例，低于则算背景。"
                         "给多个值 = 敏感性扫描；结论要在整条曲线上都成立")
    ap.add_argument("--list", type=int, default=0, help="逐题列出前 N 条")
    a = ap.parse_args()

    run, arms = Path(a.run), Path(a.arms)
    rows = {r["stem"]: r for r in csv.DictReader((arms / "yolo_results.csv").open())
            if r["skipped_by_official"] not in ("True", "true", "1")}
    box = {json.loads(l)["stem"]: json.loads(l) for l in Path(a.boxes).open()}

    runs = [(mc, *tally(rows, box, run, a.img_size, a.amb_frac, mc))
            for mc in a.min_cover]
    stat0, agg0, per0 = runs[0][1], runs[0][2], runs[0][3]

    print(f"素材：可用 {stat0['用']} 题 / csv 里 {len(rows)} 题"
          f"　（未经布局引导 {stat0['未引导']}、缺 npz {stat0['缺npz']}、"
          f"缺框 {stat0['缺框']}）")
    if not stat0["用"]:
        print("没有可用样本 —— 确认 --run 目录里有 *_masks.npz，"
              "且 --boxes 是同一臂的检测框")
        return
    cov = stat0["用"] / len(rows)
    acc = 100.0 * agg0["数对"] / agg0["题"]
    print(f"这批的准确率 {acc:.1f}%（{agg0['数对']}/{agg0['题']}）"
          + ("   ⚠️ 覆盖率仅 %.0f%%，若与全集准确率相差明显，这就是**有偏子集**，"
             "\n     底下所有构成比只描述这个子集，不得外推" % (100 * cov)
             if cov < 0.9 else ""))
    print(f"布局本身：{agg0['blob']} 个 blob；布局数 ≠ 要求 N 的题 "
          f"{agg0['布局≠N']}（ReLayout 没改到位，与生成无关）")

    print(f"\n{'='*76}\n表一 误差摊派 × min-cover 敏感性"
          f"（恒等式 yolo − k = 多长 − 空blob + 背景漏）")
    print(f"{'min-cover':>10}{'题数':>6}{'多长':>7}{'空blob':>8}{'背景漏':>8}"
          f"{'多长占比':>10}{'模糊框':>8}")
    for mc, st, ag, _ in runs:
        tot = ag["多长"] + ag["空blob"] + ag["背景漏"]
        print(f"{mc:>10.2f}{ag['题']:>6}{ag['多长']:>7}{ag['空blob']:>8}"
              f"{ag['背景漏']:>8}{100*ag['多长']/max(tot,1):>9.1f}%{ag['模糊框']:>8}")
    print("  读法：三项随 min-cover 的漂移量 = 归属规则带来的不确定度。"
          "\n  「多长占比」在整条曲线上都 ≥30% 才算过门；只在某一点过不算。")

    print(f"\n{'='*76}\n表二 ★ 与归属摇摆无关的下界（把每个框尽量塞进空着的 blob）")
    print(f"{'min-cover':>10}{'多长（点估计）':>16}{'多长下界':>10}"
          f"{'空blob下界':>12}{'错题里有下界多长':>18}")
    for mc, st, ag, _ in runs:
        frac = f"{ag['错·有多长下界']}/{ag['错题']}"
        print(f"{mc:>10.2f}{ag['多长']:>15}{ag['多长下界']:>10}"
              f"{ag['空下界']:>12}{frac:>18}")
    print("  下界的含义：即使按最有利于「没有重复生成」的方式分配每一个框，"
          "\n  仍然剩下这么多「一个 blob 里不止一个物体」。这个数消不掉，"
          "\n  **它才是论文 §7 那条 Limitation 的硬证据**。")

    e = agg0["错题"]
    print(f"\n{'='*76}\n表三 只看数错的 {e} 题（min-cover={a.min_cover[0]:.2f}，"
          f"一题可同时有多种）")
    print(f"{'':<28}{'题数':>6}{'占错题':>9}{'该机制的量':>12}")
    for k_, lab_ in (("多长", "有「一个 blob 长出多个」"),
                     ("空blob", "有「blob 空着」"),
                     ("背景漏", "有「长到布局之外」")):
        print(f"{lab_:<28}{agg0['错·有'+k_]:>6}"
              f"{100*agg0['错·有'+k_]/max(e,1):>8.1f}%{agg0['错·'+k_]:>12}")

    if a.list:
        print(f"\n逐题（按下界多长排序，前 {a.list}）：")
        print(f"{'题':<40}{'N':>3}{'blob':>5}{'yolo':>5}"
              f"{'多长':>5}{'下界':>5}{'空':>4}{'背景':>5}{'模糊':>5}")
        for r in sorted(per0, key=lambda x: -x[8])[:a.list]:
            print(f"{r[0]:<40}{r[1]:>3}{r[2]:>5}{r[3]:>5}"
                  f"{r[4]:>5}{r[8]:>5}{r[5]:>4}{r[6]:>5}{r[7]:>5}")

    print(f"\n判据（见数前写下，不因结果好看而放宽）：以**表二的下界**为准，"
          f"\n「多长下界 / (多长下界 + 空blob下界 + 背景漏)」在整条 min-cover 曲线上"
          f"\n≥30% → 实例级约束值得预登记；<15% → 想法作废；中间地带看错题占比。"
          f"\n另：覆盖率 <90% 时以上全部只描述子集，必须先把 npz 补全再判。")


if __name__ == "__main__":
    main()
