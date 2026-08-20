#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐张、配对地查图像质量塌陷。零 GPU，只读已经生成好的图。

    为什么要单独写它：`yolo_eval` / `summary` 里那两列质量数是**整批的平均**。
    平均会把尾巴吃掉 —— 60 张里有 5 张塌成剪贴画、55 张正常，均值几乎不动。
    我们肉眼看到的恰恰是那 5 张。所以"均值没掉"根本不能回答"有没有图塌了"。
    这里做两件均值做不到的事：

      1. **配对**：每张图和它自己那张 vanilla 比（同 prompt 同 seed，唯一的
         差别就是 counting pass），而不是和整批的平均比。
      2. **报尾巴**：给出掉得最狠的前几张的**文件名**，可以直接去看那几张。

    另外加一个专门量"块状伪影"的读数。colourfulness 量不出块状 —— 一张
    颜色鲜艳的拼接图 colourfulness 可以很高。块状是**空间**性质：

      blockiness(P) = 落在 P 像素网格线上的相邻像素差 / 不在网格线上的
                      相邻像素差

    自然照片没有理由在某个固定周期上系统性地更陡，比值≈1。比值明显>1 就说明
    图里有周期为 P 的硬边。挑 P 有讲究，它能指认伪影是**哪一层**造出来的：
      · P=32：1024 图上一个 32×32 注意力 token 的大小。我们的损失就是在
        32×32 的注意力图上把值往 0/1 推的 —— 如果伪影对齐到 32，那它是
        我们的损失直接画出来的。
      · P=8：VAE 的下采样倍率，latent 一格。对齐到 8 说明是解码端的事。
    这个区分不是装饰：它决定该改损失还是该改别的地方。

    注意 blockiness 必须在**原分辨率**上算，缩略图会把网格对齐关系毁掉。

用法：
    python count_probe/quality_probe.py --arms $SD_OUT/count/tune_{orig,fg,inst,inst2,hinge}_arms
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_eval import _photo_stats


def blockiness(path, periods, side_max=1024):
    """→ {P: 网格线上的边强度 / 非网格线上的边强度}。≈1 无结构，>1 有周期性硬边。

    一次读图算所有周期 —— 读图和求差分是这里的大头，按周期重复读会白花几倍时间。
    """
    if isinstance(periods, int):
        periods = [periods]
    im = Image.open(path).convert("L")
    k = 1
    if max(im.size) > side_max:                     # 只缩整数倍，保住网格对齐
        k = max(1, max(im.size) // side_max)
        if k > 1:
            im = im.resize((im.width // k, im.height // k), Image.BOX)
    g = np.asarray(im, dtype=np.float32)
    dc = np.abs(np.diff(g, axis=1)).mean(axis=0)    # 逐列的相邻差
    dr = np.abs(np.diff(g, axis=0)).mean(axis=1)    # 逐行的相邻差

    def _ratio(d, p):
        on = (np.arange(len(d)) + 1) % p == 0
        if on.sum() == 0 or (~on).sum() == 0:
            return float("nan")
        # 分母下限：块内完全纯色时分母是 0（自测里的纯块图）。给 1e-2（0~255 标度）
        # 的地板，读数会很大但有限，不会变成 nan 把整批平均污染掉。
        return float(d[on].mean() / max(d[~on].mean(), 1e-2))

    out = {}
    for p0 in periods:
        p = max(1, p0 // k)
        if p < 2:
            out[p0] = float("nan")
            continue
        v = [x for x in (_ratio(dc, p), _ratio(dr, p)) if x == x]
        out[p0] = sum(v) / len(v) if v else float("nan")
    return out


def _files(d, ref=None):
    """→ [(文件名, 基准图, 本配置的图)]

    基准默认是**本配置自己的 vanilla 臂**（问「相对没有任何干预的原图退化了多少」）。
    给了 --ref 就换成参照配置的 countgen 臂（问「相对在位者的输出退化了多少」）。
    后者才是「我们有没有把 CountGen 弄得更糟」的直接读数 —— 我们是建在它上面的，
    它自己造成的那部分退化不该记在我们头上。两张都是同 prompt、同 seed 的修正图，
    可以逐张配对。
    """
    p = Path(d) / "yolo_results.csv"
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        pm = Path(d) / "countgen" / r["file"]
        pv = (Path(ref) / "countgen" / r["file"]) if ref else (Path(d) / "vanilla" / r["file"])
        if pv.exists() and pm.exists():
            out.append((r["file"], pv, pm))
    return out


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--periods", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--worst", type=int, default=5, help="每个配置列出掉得最狠的几张")
    ap.add_argument("--drop", type=float, default=30.0,
                    help="相对自己那张 vanilla 掉超过百分之几算「塌了」")
    ap.add_argument("--grey", type=float, default=15.0,
                    help="colourfulness 绝对值低于多少算「基本没颜色」")
    ap.add_argument("--ref", default=None,
                    help="把基准从「本配置自己的 vanilla」换成「参照配置的 countgen 臂」，"
                         "例如 --ref $SD_OUT/count/tune_orig_arms。我们是建在 CountGen "
                         "上面的，判据应当是「有没有比它更糟」，而不是「有没有比完全不"
                         "干预更糟」—— 后者把在位者自己造成的退化也算到了我们头上")
    a = ap.parse_args()
    if a.ref and not (Path(a.ref) / "countgen").is_dir():
        raise SystemExit(f"!! --ref {a.ref} 下面没有 countgen/ 目录")
    BASE = "CountGen" if a.ref else "vanilla"

    rows = []
    for d in a.arms:
        d = Path(d)
        if a.ref and d.resolve() == Path(a.ref).resolve():
            print(f"（跳过 {d.name}：它就是参照本身）")
            continue
        fs = _files(d, a.ref)
        if not fs:
            print(f"（跳过 {d.name}：没有可用的成对图）")
            continue
        rec = dict(name=d.name.replace("_arms", ""), n=len(fs), dcol=[], dflat=[],
                   grey_v=0, grey_m=0, blk_v={p: [] for p in a.periods},
                   blk_m={p: [] for p in a.periods}, per_file=[])
        for f, pv, pm in fs:
            cv, fv = _photo_stats(pv)
            cm, fm = _photo_stats(pm)
            dc = 100.0 * (cm - cv) / cv if cv > 1e-6 else float("nan")
            rec["dcol"].append(dc)
            rec["dflat"].append(100.0 * (fm - fv))
            rec["grey_v"] += cv < a.grey
            rec["grey_m"] += cm < a.grey
            bv, bm = blockiness(pv, a.periods), blockiness(pm, a.periods)
            for p in a.periods:
                rec["blk_v"][p].append(bv[p])
                rec["blk_m"][p].append(bm[p])
            bl = {p: (bv[p], bm[p]) for p in a.periods}
            rec["per_file"].append((f, dc, 100.0 * (fm - fv), cv, cm, bl))
        rows.append(rec)

    if not rows:
        raise SystemExit("!! 一个配置都没读到")

    # ---------- 表一：逐张配对的 colourfulness 变化，看分布不看均值 ----------
    lab = (f"参照配置（{Path(a.ref).name}）的同一题输出" if a.ref
           else "它自己那张 vanilla")
    print(f"\n表一 每张图相对**{lab}**的 colourfulness 变化（%）")
    print(f"{'配置':<12}{'题数':>5}{'p10':>8}{'中位':>8}{'p90':>8}"
          f"{'掉>' + str(int(a.drop)) + '%':>10}{'方法无色':>10}{BASE + '无色':>12}")
    for r in rows:
        v = np.array([x for x in r["dcol"] if x == x])
        bad = int((v < -a.drop).sum())
        print(f"{r['name']:<12}{r['n']:>5}{_pct(v,10):>+8.0f}{_pct(v,50):>+8.0f}"
              f"{_pct(v,90):>+8.0f}{bad:>7} 张{r['grey_m']:>8} 张{r['grey_v']:>10} 张")
    print(f"  「方法无色 / {BASE}无色」= colourfulness < {a.grey:.0f} 的张数。"
          + ("两边一样多 = **我们没有比 CountGen 更糟**，它自己造成的那部分退化"
             "不该记在我们头上。" if a.ref else
             "两边一样多说明本来就是那种题（雪地、白底），不是我们弄的。"))
    print("  中位数≈0 而 p10 很负 = **只有尾巴塌了**，这正是均值看不见、肉眼看得见的情形。")
    if not a.ref:
        print("  ⚠️ 这一栏的基准是「完全不干预」，包含了 CountGen 自己造成的退化。"
              "\n     要判「我们有没有把它弄更糟」，加 --ref <原版的 _arms 目录>。")

    # ---------- 表二：块状伪影 ----------
    print(f"\n表二 块状伪影：网格线上的边强度 / 非网格线上的边强度（≈1 = 没有周期性硬边）")
    hdr = "".join(f"{'P=' + str(p):>16}" for p in a.periods)
    print(f"{'配置':<12}{hdr}{'峰值在':>10}   （每格：{BASE} → 本配置）")
    for r in rows:
        cells, ms = "", []
        for p in a.periods:
            bv, bm = np.nanmean(r["blk_v"][p]), np.nanmean(r["blk_m"][p])
            ms.append(bm - bv)
            cells += f"{bv:>7.2f}→{bm:>6.2f}" + (" ⚠️" if bm - bv > 0.05 else "  ")
        pk = a.periods[int(np.argmax(ms))] if max(ms) > 0.05 else None
        print(f"{r['name']:<12}{cells}{('P=' + str(pk)) if pk else '—':>10}")
    print("  怎么读：**看比值在哪个 P 上最高**，不要只看某一个 P 是不是>1。")
    print("    周期 P₀ 的伪影会让 P₀ 的邻居也偏高（P=32 的格线同时也是 P=64 的一半格线），"
          "\n    但比值在真正的 P₀ 上取到峰。自测：只在 32 上加网格的图读数"
          "\n    P8 1.53 / P16 2.18 / P32 3.49（峰在 32）；纯 8×8 块的图读数"
          "\n    P16 15.2 > P32 10.4（峰在更小的 P）—— 所以只看「P=32 是不是>1」会误判。")
    print("  峰在 P=32 → 伪影对齐到注意力 token 的格子（1024 图上一个 32×32 token 就是 32 像素），"
          "\n    是损失把 32×32 的注意力图推成 0/1 直接画出来的 → 改损失（软化/加平滑项）才有用。"
          "\n  峰在 P=8  → 对齐到 VAE latent 一格，解码端的事，与注意力无关 → 改损失没用。"
          "\n  没有峰（全列 ≈vanilla）→ 肉眼看到的「一块一块」不是周期性硬边，"
          "\n    别继续用「块状伪影」描述它，去表三点名的那几张上重新看是什么。")

    # ---------- 表三：点名 ----------
    print(f"\n表三 每个配置掉得最狠的 {a.worst} 张（直接去看这几张，别看均值）")
    for r in rows:
        print(f"\n  {r['name']}")
        for f, dc, df, cv, cm, bl in sorted(r["per_file"], key=lambda x: x[1])[:a.worst]:
            pk = max(bl, key=lambda p: bl[p][1] - bl[p][0])   # 这张图涨得最多的周期
            print(f"    {f[:-4][:34]:<36}colour {cv:>5.1f}→{cm:>5.1f} ({dc:>+5.0f}%)"
                  f"  平涂 {df:>+5.1f}pt  块状峰 P={pk}: {bl[pk][0]:.2f}→{bl[pk][1]:.2f}")


if __name__ == "__main__":
    main()
