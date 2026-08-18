#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
「补缺失物体」失败的两种病，分辨哪一种。**零生成**，只用已有的图和 mask 数组。

    表六b 的读数：数少了那一支（ReLayout U-Net）净 +2/33，p=0.774 —— 与掷硬币
    无异。但"失败"有两种完全不同的病，对应完全不同的药：

      (i) **位置提议错了**：ReLayout 说"往这儿加一个"，可那个位置本来就不该有
          物体（被遮挡、太小、和已有物体粘连），引导忠实地执行了一个坏布局。
          → 要换的是布局提议。可以做成 training-free，直接对上我们的约束。

      (ii) **位置没错，引导落实不了**：那块地方确实是空的，但扩散过程没能在那儿
          长出一个物体。
          → 要换的是引导机制，即 `utils/loss_utils.py:5` 那句
            `foreground_mask = (desired_mask != 0)` —— 它把逐实例标签压成一张
            二值前景图，只说"物体该出现在这些位置"，没说"该是 N 个分开的东西"。

    判别办法：**新增的那个 blob 的位置上，最终图里到底有没有长出目标类的物体。**

      · 命中率高（≫50%）→ 引导是听话的，病在 (i)：位置给错了，
        或者加上去的物体和邻居粘成了一个，计数仍然不对。
      · 命中率低（≪50%）→ 病在 (ii)：让它在指定位置造一个物体，它做不到。

    "新增的 blob"怎么认：`relayout_undergeneration` 是往通道维后面追加的
    （`relayout.py:36` 的 `out_mask[:, number_of_clusters_founded + i]`），
    而 `from_channels` 把第 i 个通道标成 i+1，所以**标号 > n_dbscan 的就是新增的**。
    脚本同时报告一个与通道记账无关的旁证：`postprocess>0 且 vanilla==0` 的新前景面积。

用法：
    python count_probe/where_added.py --src $SD_OUT/count/cocoount \
        --arms $SD_OUT/count/cocoount_arms
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RES = 32          # mask 的边长；1024² 的图上每格 32 像素


def _blob_box(mask, label):
    ys, xs = np.nonzero(mask == label)
    if len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, len(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="countgen_batch.py 的输出目录")
    ap.add_argument("--arms", required=True, help="make_arms.py 的输出目录")
    ap.add_argument("--weights", default="yolov9e.pt")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--min-det", type=int, default=3,
                    help="YOLO 在该图上检出的目标类总数低于此值就单列 —— "
                         "那种图上「没命中」多半是检测器读不出来，不是引导失败")
    a = ap.parse_args()
    src, arms = Path(a.src), Path(a.arms)

    idx = {r["stem"]: r for r in json.load(open(arms / "index.json"))}
    rows = {r["stem"]: r for r in csv.DictReader((arms / "yolo_results.csv").open())}

    items = []
    for stem, meta in idx.items():
        f = src / f"{stem}_masks.npz"
        if not f.exists():
            continue
        z = np.load(f)
        n_db, N = int(z["n_dbscan"]), int(z["N"])
        if n_db >= N:
            continue                       # 只看"数少了"那一支
        items.append((stem, meta, z, n_db, N))
    if not items:
        raise SystemExit(
            f"!! {src} 里没有「数少了」那一支的 *_masks.npz。\n"
            f"   先： python count_probe/decompose.py --arms {arms} --dump-ids\n"
            f"   再： python count_probe/countgen_batch.py --only-ids {arms}/under_ids.txt "
            f"--out {src}")
    print(f"「数少了」的题：{len(items)} 个（有 mask 数组的）\n")

    from ultralytics import YOLO
    model = YOLO(a.weights)

    tot_blob = hit_blob = 0
    per_item = []
    for stem, meta, z, n_db, N in items:
        post = z["postprocess"].astype(int)
        van = z["vanilla"].astype(int)
        img = arms / "countgen" / meta["file"]
        if not img.exists():
            continue
        kw = {"verbose": False}
        if a.conf is not None:
            kw["conf"] = a.conf
        r = model(str(img), **kw)[0]
        names = r.names
        W, H = r.orig_shape[1], r.orig_shape[0]
        sx, sy = W / RES, H / RES
        boxes = [b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())
                 if names[int(c)] == meta["coco_class"]]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

        added = [l for l in range(n_db + 1, int(post.max()) + 1)]
        nh = 0
        for l in added:
            bb = _blob_box(post, l)
            if bb is None:
                continue
            x0, y0, x1, y1, area = bb
            px0, py0, px1, py1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            if any(px0 <= cx < px1 and py0 <= cy < py1 for cx, cy in centers):
                nh += 1
        n_added = sum(1 for l in added if _blob_box(post, l) is not None)
        tot_blob += n_added
        hit_blob += nh
        new_area = int(((post > 0) & (van == 0)).sum())
        # ★ 原版图的 YOLO 读数是判读的关键列。少了它就分不清两件事：
        #     · 原版本来就超量（计数器低估）——那是计数器的问题
        #     · 原版确实不足、修正却把总数顶过头 ——那是引导的全局副作用
        #   我上一轮就是把"最终图超量"误当成"原版超量"，结论错了。
        yv = rows.get(stem, {}).get("yolo_vanilla")
        per_item.append(dict(stem=stem, N=N, n_dbscan=n_db, n_added=n_added,
                             hit=nh, yolo=len(boxes),
                             y_v=None if yv in (None, "", "None") else int(float(yv)),
                             ok=rows.get(stem, {}).get("ok_countgen"),
                             new_area=new_area))

    print(f"{'题':<32}{'N':>3}{'DBSCAN':>7}{'原版YOLO':>9}{'新增':>5}{'长出':>5}"
          f"{'最终YOLO':>9}{'对':>4}")
    for r in per_item:
        print(f"{r['stem'][:31]:<32}{r['N']:>3}{r['n_dbscan']:>7}"
              f"{str(r['y_v']):>9}{r['n_added']:>5}{r['hit']:>5}"
              f"{r['yolo']:>9}{str(r['ok']):>4}")

    # ---- 计数器低估 vs 修正顶过头 ----
    ok3 = [r for r in per_item if r["y_v"] is not None and r["yolo"] >= a.min_det]
    if ok3:
        under_cnt = [r for r in ok3 if r["y_v"] > r["N"]]      # 原版就已超量 → 计数器低估
        over_shoot = [r for r in ok3 if r["y_v"] <= r["N"] and r["yolo"] > r["N"]]
        stay_low = [r for r in ok3 if r["y_v"] <= r["N"] and r["yolo"] < r["N"]]
        print(f"\n在检测器读得出的 {len(ok3)} 题里：")
        print(f"  原版就已超量（计数器低估，方向本身错）      {len(under_cnt):>3}")
        print(f"  原版不足、修正后**顶过头**（引导的全局副作用）{len(over_shoot):>3}")
        print(f"  原版不足、修正后仍不足                    {len(stay_low):>3}")
        if over_shoot:
            ex = sorted(over_shoot, key=lambda r: -(r["yolo"] - r["N"]))[:4]
            print("  顶过头最厉害的：" + "，".join(
                f"{r['stem'][:16]} N={r['N']} 原版{r['y_v']}→{r['yolo']}" for r in ex))

    print(f"\n{'='*70}")
    print(f"新增 blob 共 {tot_blob} 个，其中位置上真的长出目标类物体的 {hit_blob} 个"
          f"（{100.0*hit_blob/max(tot_blob,1):.1f}%）")

    # ★ 剔除"检测器在这张图上几乎什么都没看见"的题。命中率为 0 在那些图上
    #   不能说明引导失败 —— 更可能是 YOLO 读不出来（例：car 那两题 YOLO 总数 0）。
    good = [r for r in per_item if r["yolo"] >= a.min_det]
    bad = [r for r in per_item if r["yolo"] < a.min_det]
    tb = sum(r["n_added"] for r in good)
    hb = sum(r["hit"] for r in good)
    print(f"剔除 YOLO 总检出 < {a.min_det} 的 {len(bad)} 题后："
          f"{hb}/{tb} = {100.0*hb/max(tb,1):.1f}%")
    if bad:
        print(f"  被剔除的：{', '.join(r['stem'][:20] for r in bad)}")
        print(f"  它们贡献了 {sum(r['n_added']-r['hit'] for r in bad)}/"
              f"{tot_blob-hit_blob} 个未命中 —— 这些是检测器的问题，不是引导的问题。")
    print("\n判读（事先写死）：")
    print("  · 命中率 ≫ 50% → 引导是听话的，病在**位置提议**（i）：")
    print("      要么位置本来就不该有物体，要么加上去的和邻居粘成了一个。")
    print("      → 方法方向：换掉 ReLayout，用 training-free 的布局提议。")
    print("  · 命中率 ≪ 50% → 病在**引导**（ii）：指定位置造不出物体。")
    print("      → 方法方向：动 loss_utils.py:5 那个把实例身份压成二值前景的损失。")
    print("  · 两边都不像（40–60%）→ 两种病都有，得再切一刀，比如按新增 blob 的")
    print("      面积和它与已有 blob 的距离分开看。")

    # 与通道记账无关的旁证：新前景面积
    ar = [r["new_area"] for r in per_item]
    print(f"\n旁证（不依赖通道记账）：postprocess>0 且 vanilla==0 的新前景格数，"
          f"中位数 {int(np.median(ar))}/1024，最小 {min(ar)}，最大 {max(ar)}")
    print("  若中位数接近 0，说明 ReLayout 根本没往空白处加东西，那本身就是结论。")


if __name__ == "__main__":
    main()
