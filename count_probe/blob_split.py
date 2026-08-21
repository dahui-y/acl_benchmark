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

    本脚本量的就是这个后果占多大。设布局有 N 个 blob（ReLayout 已改成要求的
    数量），评测器数出 yolo 个框，每个框归到重叠最大的 blob，则**恒等式**：

        yolo − N = 多长 − 空blob + 背景漏

      多长   Σ_blob max(cnt−1, 0)  一个 blob 里落了 ≥2 个框 ← 论文承认的失效
      空blob #{blob: cnt == 0}     布局给了位置却没长出来 ← 布局损失该管的
      背景漏 #{框: 不落在任何 blob} 长到了布局之外     ← 自注意力遮罩该管的

    三项互斥且穷尽（每个框至多归一个 blob），所以这是摊派而不是估计。

    ⚠️ 判读的边界：
    · 归属靠**框与 blob 的网格重叠**，32×32 的布局对 1024 的图，一格 32 像素。
      密集小物体上归属会有噪声，脚本单列「跨 blob 的模糊框」计数供打折。
    · 评测器自身会错（我们量过：DBSCAN 与 YOLOv9e 一致率 64.1%）。这里把
      YOLOv9e 当口径，因为主表就是它，但它不是真值。
    · obj_num_match=True 的题原样返回 vanilla，没有经过布局引导，单列不混算。

用法：
    # 1) 先把 CountGen 臂的 YOLOv9e 框落盘（一次检测遍历，约 2 分钟）
    python count_probe/det_counts.py --arms $SD_OUT/count/cocoount_arms \\
        --arm countgen --weights yolov9e.pt \\
        --out $SD_OUT/count/cocoount_arms/v9e_boxes_countgen.jsonl
    # 2) 摊派
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
        return m.astype(int)                       # 已是标签图
    if m.ndim == 2:                                # (k, H*W) 通道
        k, hw = m.shape
        s = int(round(hw ** 0.5))
        ch = m.reshape(k, s, s)
    elif m.ndim == 3:                              # (k, H, W) 通道
        ch = m
    else:
        raise ValueError(f"看不懂的布局形状 {m.shape}")
    lab = np.zeros(ch.shape[1:], dtype=int)
    for i in range(ch.shape[0]):                   # 后面的通道覆盖前面的
        lab[ch[i] > 0.5] = i + 1
    return lab


def assign(boxes, lab, img_size, amb_frac=0.30):
    """每个框归到重叠格子最多的 blob。→ (每 blob 的框数, 背景框数, 模糊框数)

    模糊框 = 与第二名 blob 的重叠也达到第一名的 amb_frac 以上。它是归属噪声的
    上界，单独报出来，好让读者知道这张表能信到什么程度。
    """
    H, W = lab.shape
    sy, sx = img_size[1] / H, img_size[0] / W
    k = int(lab.max())
    cnt = Counter()
    bg = amb = 0
    for x1, y1, x2, y2 in boxes:
        c0, r0 = int(np.floor(x1 / sx)), int(np.floor(y1 / sy))
        c1, r1 = int(np.ceil(x2 / sx)), int(np.ceil(y2 / sy))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(max(c1, c0 + 1), W), min(max(r1, r0 + 1), H)
        sub = lab[r0:r1, c0:c1]
        ov = np.bincount(sub.ravel(), minlength=k + 1)
        ov[0] = 0                                  # 背景格不参与归属
        if ov.sum() == 0:
            bg += 1
            continue
        best = int(ov.argmax())
        cnt[best] += 1
        rest = np.sort(ov)[::-1]
        if len(rest) > 1 and rest[1] >= amb_frac * rest[0]:
            amb += 1
    return cnt, bg, amb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="含 {stem}_masks.npz 的跑批目录")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--boxes", required=True, help="该臂的检测框 jsonl")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--amb-frac", type=float, default=0.30)
    ap.add_argument("--list", type=int, default=0, help="逐题列出前 N 条")
    a = ap.parse_args()

    run, arms = Path(a.run), Path(a.arms)
    rows = {r["stem"]: r for r in csv.DictReader((arms / "yolo_results.csv").open())
            if r["skipped_by_official"] not in ("True", "true", "1")}
    box = {json.loads(l)["stem"]: json.loads(l) for l in Path(a.boxes).open()}

    stat = {"用": 0, "缺npz": 0, "缺框": 0, "未引导": 0}
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
            stat["未引导"] += 1        # 原样返回 vanilla，没经过布局引导
            continue
        z = np.load(p)
        lab = to_labels(z["postprocess"])
        k = int(lab.max())
        if k == 0:
            stat["缺npz"] += 1
            continue
        cnt, bg, amb = assign(box[stem]["boxes"], lab,
                              (a.img_size, a.img_size), a.amb_frac)
        over = sum(max(c - 1, 0) for c in cnt.values())
        empty = sum(1 for j in range(1, k + 1) if cnt.get(j, 0) == 0)
        yolo = len(box[stem]["boxes"])
        assert yolo - k == over - empty + bg, (stem, yolo, k, over, empty, bg)
        N = int(r["N"])
        stat["用"] += 1
        agg["题"] += 1
        agg["blob"] += k
        agg["多长"] += over
        agg["空blob"] += empty
        agg["背景漏"] += bg
        agg["模糊框"] += amb
        agg["布局≠N"] += int(k != N)
        ok = yolo == N
        agg["数对"] += int(ok)
        if not ok:
            agg["错题"] += 1
            agg["错·多长"] += over
            agg["错·空blob"] += empty
            agg["错·背景漏"] += bg
            agg["错·有多长"] += int(over > 0)
            agg["错·有空blob"] += int(empty > 0)
            agg["错·有背景漏"] += int(bg > 0)
        per.append((stem, N, k, yolo, over, empty, bg, amb))

    print(f"素材：可用 {stat['用']} 题　（未经布局引导 {stat['未引导']}、"
          f"缺 npz {stat['缺npz']}、缺框 {stat['缺框']}）")
    if not stat["用"]:
        print("没有可用样本 —— 先确认 --run 目录里有 *_masks.npz、"
              "且 --boxes 是同一臂的检测框")
        return

    n = agg["题"]
    print(f"\n{'='*70}\n布局本身：{agg['blob']} 个 blob；"
          f"布局数 ≠ 要求 N 的题 {agg['布局≠N']}（ReLayout 没改到位，不是生成的锅）")
    print(f"生成结果：数对 {agg['数对']}/{n} = {100*agg['数对']/n:.1f}%")

    print(f"\n{'='*70}\n表一 计数误差的摊派（恒等式 yolo − N_blob = 多长 − 空blob + 背景漏）")
    print(f"{'机制':<26}{'总量':>7}{'占误差绝对量':>14}   该由谁负责")
    tot = agg["多长"] + agg["空blob"] + agg["背景漏"]
    for k_, who in (("多长", "论文 §7 Limitations 承认的失效（一个 blob 长出多个）"),
                    ("空blob", "布局损失（recall 那一头）"),
                    ("背景漏", "自注意力遮罩（precision 那一头）")):
        print(f"{k_:<26}{agg[k_]:>7}{100*agg[k_]/max(tot,1):>13.1f}%   {who}")
    print(f"{'（其中跨 blob 的模糊框）':<24}{agg['模糊框']:>7}"
          f"{'':>14}   归属噪声上界，按它给上面打折")

    e = agg["错题"]
    print(f"\n{'='*70}\n表二 只看数错的 {e} 题（一题可同时有多种）")
    print(f"{'':<26}{'题数':>7}{'占错题':>9}{'该机制的量':>12}")
    for k_, lab_ in (("多长", "有「一个 blob 长出多个」"),
                     ("空blob", "有「blob 空着」"),
                     ("背景漏", "有「长到布局之外」")):
        print(f"{lab_:<26}{agg['错·有'+k_]:>7}{100*agg['错·有'+k_]/max(e,1):>8.1f}%"
              f"{agg['错·'+k_]:>12}")

    if a.list:
        print(f"\n逐题（前 {a.list}）：")
        print(f"{'题':<40}{'N':>3}{'blob':>5}{'yolo':>5}"
              f"{'多长':>5}{'空':>4}{'背景':>5}{'模糊':>5}")
        for row in sorted(per, key=lambda x: -x[4])[:a.list]:
            print(f"{row[0]:<40}{row[1]:>3}{row[2]:>5}{row[3]:>5}"
                  f"{row[4]:>5}{row[5]:>4}{row[6]:>5}{row[7]:>5}")

    print(f"\n判读（见数前写下）：若「多长」占误差绝对量 ≥30%，说明「一个 blob 长出"
          f"多个」\n是主要失效之一，实例级约束这个方向值得预登记；<15% 则不值得，"
          f"想法就此作废。\n中间地带按「错题里有多长的比例」再看一眼。")


if __name__ == "__main__":
    main()
