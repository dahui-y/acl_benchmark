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

from edit_share import _perm

RES = 32          # mask 的边长；1024² 的图上每格 32 像素


def _prior_dist(vanilla):
    """每个格子到**最近的 vanilla blob**的距离（格）。vanilla blob 内部为 0。

    这是「模型自己的先验在这儿有没有质量」的代理。真正想要的是 vanilla 那一趟的
    注意力图，但跑批时没存（只存了聚类后的 mask）。vanilla mask 的非零区正是
    模型自己选择放物体的位置，所以「离它多远」是手头能拿到的最接近的量。
    这个代理的粗糙之处要记在账上：它只看聚类**之后**的硬边界，看不到
    次阈值的弱峰 —— 而弱峰恰恰是假设里最有意思的那部分。
    """
    from scipy.ndimage import distance_transform_edt
    fg = vanilla != 0
    if not fg.any():                       # vanilla 一个 blob 都没聚出来
        return np.full(vanilla.shape, np.nan, dtype=float)
    return distance_transform_edt(~fg).astype(float)


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
    ap.add_argument("--rand-trials", type=int, default=20,
                    help="每个新增 blob 随机重放几次做空对照")
    ap.add_argument("--seed", type=int, default=0)
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
    tot_rnd, hit_rnd = [0], [0]
    rng = np.random.default_rng(a.seed)
    per_item = []
    per_blob = []          # 逐个新增 blob：命中与否 + 它离模型自己的先验多远
    rnd_d = []             # 随机撒的框的同一个距离，作零基准
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
        dist = _prior_dist(van)
        nh = 0
        rnd_hit = rnd_n = 0
        for l in added:
            bb = _blob_box(post, l)
            if bb is None:
                continue
            x0, y0, x1, y1, area = bb
            px0, py0, px1, py1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            got = any(px0 <= cx < px1 and py0 <= cy < py1 for cx, cy in centers)
            if got:
                nh += 1
            cells = post == l
            per_blob.append(dict(stem=stem, hit=bool(got), area=int(area),
                                 d_min=float(np.nanmin(dist[cells])),
                                 d_mean=float(np.nanmean(dist[cells]))))
            # ★ 空对照：把同样大小的框**随机扔到画面别处**，看能蒙中多少。
            #   必要性来自肉眼检查：修正后的图常常被物体填满（horse_num=7 那张是
            #   一大群马），那么"在指定位置找到物体"可能只是因为到处都是物体。
            #   没有这个对照，上面那个命中率说明不了引导有没有听话。
            w, h = x1 - x0, y1 - y0
            for t in range(a.rand_trials):
                rx = rng.integers(0, max(RES - w, 1))
                ry = rng.integers(0, max(RES - h, 1))
                qx0, qy0 = rx * sx, ry * sy
                qx1, qy1 = (rx + w) * sx, (ry + h) * sy
                rnd_n += 1
                if any(qx0 <= cx < qx1 and qy0 <= cy < qy1 for cx, cy in centers):
                    rnd_hit += 1
                rnd_d.append(float(np.nanmean(dist[ry:ry + h, rx:rx + w])))
        n_added = sum(1 for l in added if _blob_box(post, l) is not None)
        tot_blob += n_added
        hit_blob += nh
        tot_rnd[0] += rnd_n
        hit_rnd[0] += rnd_hit
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

    # ★ 空对照的读数 —— 没有它，上面那个命中率说明不了任何事
    rr = 100.0 * hit_rnd[0] / max(tot_rnd[0], 1)
    real = 100.0 * hit_blob / max(tot_blob, 1)
    print(f"\n【空对照】同样大小的框随机扔到画面别处（每个 blob 重放 {a.rand_trials} 次，"
          f"共 {tot_rnd[0]} 次）：命中 {rr:.1f}%")
    print(f"  真实位置 {real:.1f}%  vs  随机位置 {rr:.1f}%  →  差 {real-rr:+.1f} 个点")
    if real - rr < 10:
        print("  ⚠️ 两者接近 —— **命中率高只是因为图里到处都是物体**，"
              "它不能证明引导把物体放到了指定位置。前面基于命中率的结论要撤回。")
    else:
        print("  → 真实位置显著高于随机，命中率确实反映了引导的空间服从性。")
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

    # ---- 先验对齐：新增的 blob 离模型自己已经选好的位置有多远 ----
    #
    # 这是目前唯一还没被否掉的假设的判据。前面几轮否掉的三条都是「把引导做强/
    # 做准」：hinge 把目标解满 91% 无收益；编辑区权重份额高 13 倍的支路反而无效；
    # 执行保真度从 44.4% 提到 36.1% 也无收益。剩下的解释是**要求本身**的性质：
    #   · 删多余 = 让模型少做一点它本来就在做的事（新 mask 是原布局的子集）
    #   · 补缺失 = 让模型在它自己没选的地方凭空造一个
    # 若这条成立，那么**离已有物体近的新增 blob 更容易长出来**。
    #
    # 注意这里检验的是「blob 层面」的关系，不是「题目层面」的成败 —— 一道题
    # 计数对不对还牵扯别的因素，blob 长没长出来才是这条假设的直接读数。
    if per_blob:
        # vanilla 一个 blob 都没聚出来的题，距离是 nan（没有先验可对齐），剔掉
        fin = [b for b in per_blob if np.isfinite(b["d_mean"])]
        hit_d = [b["d_mean"] for b in fin if b["hit"]]
        mis_d = [b["d_mean"] for b in fin if not b["hit"]]
        rnd_d = [d for d in rnd_d if np.isfinite(d)]
        if len(fin) < len(per_blob):
            print(f"\n（{len(per_blob)-len(fin)} 个新增 blob 所在的题 vanilla 没聚出"
                  f"任何 blob，无先验可比，已剔除）")
        print(f"\n{'='*70}")
        print("先验对齐检验：新增 blob 到最近的 vanilla blob 的平均距离（格，1 格 = 32 像素）")
        print(f"{'':<24}{'个数':>6}{'距离中位':>10}{'距离均值':>10}")
        for lab, v in (("长出来了", hit_d), ("没长出来", mis_d),
                       ("随机撒的框（零基准）", rnd_d)):
            if v:
                print(f"{lab:<24}{len(v):>6}{np.median(v):>10.2f}{np.mean(v):>10.2f}")
        if len(hit_d) >= 2 and len(mis_d) >= 2:
            p = _perm(hit_d, mis_d)
            print(f"\n  长出 vs 没长出，置换检验（中位数之差，20000 次）p = {p:.3f}")
            if p > 0.05:
                print("  → **假设不成立**。离先验远近不预测新增 blob 能不能长出来。")
                print("     那么「删=先验子集、补=违背先验」这条也就没有支撑，"
                      "\n     「用先验引导放置来替换 ReLayout」这个方法提案不要写。")
            elif np.median(hit_d) < np.median(mis_d):
                print("  → **假设成立且方向对**：离已有物体越近越容易长出来。")
                print("     方法提案有了依据：放置时优先选靠近模型自身先验的位置，"
                      "\n     可以完全 training-free 地替掉那 474MB 的 ReLayout U-Net。")
            else:
                print("  → 显著，但**方向是反的**（离得远反而更容易长出来）。")
                print("     假设按原样不成立；这个反向关系本身要先解释清楚再谈方法。")
        else:
            print("\n  （某一组不足 2 个，检不了。先把「数少了」那一支的题跑全。）")
        if rnd_d:
            print(f"  随机基准的意义：若真实新增 blob 的距离与随机撒的框差不多，"
                  f"\n  说明 ReLayout 的放置本身是**不看先验的**——那本身就是一条结论。")

    # 与通道记账无关的旁证：新前景面积
    ar = [r["new_area"] for r in per_item]
    print(f"\n旁证（不依赖通道记账）：postprocess>0 且 vanilla==0 的新前景格数，"
          f"中位数 {int(np.median(ar))}/1024，最小 {min(ar)}，最大 {max(ar)}")
    print("  若中位数接近 0，说明 ReLayout 根本没往空白处加东西，那本身就是结论。")

    # ---- 前景占比 f 与损失下界 ----
    # CountGen 把 attention map 归一化到 [0,1] 再喂 BCEWithLogits（内部还会过一次
    # sigmoid），所以"预测值"只能落在 [sigmoid(0), sigmoid(1)] = [0.500, 0.731]，
    # 两端都够不到。于是损失有一个**只取决于前景占比 f 的下界**：
    #     L_min(f) = f·10·(−log 0.731) + (1−f)·(−log 0.5) = 0.693 + 2.439·f
    # 阈值是 {0: 1.3, 10: 1.2, 20: 1.15}（pipeline_config.yaml）。
    # f 超过约 25% 时 step0 的阈值在数学上就达不到 —— 这正好解释了我们观测到的
    # 「每次修正都跑满 20 次迭代、从不靠达到阈值退出」。
    fs = np.array([float((z["postprocess"] > 0).mean())
                   for _, _, z, _, _ in items])
    Lmin = 0.6931 + 2.4394 * fs
    print(f"\n前景占比 f（desired_mask 非零的格子/1024）："
          f"中位 {np.median(fs):.1%}，最小 {fs.min():.1%}，最大 {fs.max():.1%}")
    print(f"损失下界 L_min = 0.693 + 2.439·f：中位 {np.median(Lmin):.2f}，"
          f"最大 {Lmin.max():.2f}")
    for t, lab in ((1.3, "step0"), (1.15, "step20")):
        bad = int((Lmin >= t).sum())
        print(f"  阈值 {t}（{lab}）在 {bad}/{len(fs)} 题上**数学上不可达**"
              f"（{100.0*bad/len(fs):.0f}%）")
    print("  → 这些题的 refinement 只能跑满 max_refinement_steps，"
          "把 latent 一路推下去，直到 20 步用完。")


if __name__ == "__main__":
    main()
