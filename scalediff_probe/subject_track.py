"""主体级漏检：4096² 上还认得出基图里那个主体吗？两个 arm 一样吗？

起因（03/1234 我们这版的标注图）：主船清清楚楚地在，GroundingDINO
一个框都没给它；同时检出的两个小框里有一个不是船。
**同一行里漏检和误检方向相反、恰好抵消** —— 靠抵消得到的正确不是正确。

逐行不准这件事本身可以接受（我们报的是 45 行的聚合量）。真正致命的是
另一件：**如果漏检在两个 arm 里频率不同，A/B 差值就有系统偏差** ——
我们这版更容易被漏检，计数就被系统性压低，看起来更好，但那是尺子偏心。

锚点是现成的、而且很硬：**基图 1024² 两个 arm 逐字节相同（90/90 已验证）**，
主体在基图上检得到。所以：

    1. 基图上检出主体框 -> 坐标 x4 得到 4096² 上的锚点（只用最高分那个）；
    2. 每个 arm 的 4096² 里，有没有框覆盖那个锚点？没有 = 主体级漏检；
    3. 漏检的行再分：锚点区域的像素在两个 arm 之间变了没有？
       没变而框没了 -> 检测器漏检；变了 -> 主体真的被改动了。
       阈值同 box_audit：用**命中行**的像素差分布当零假设，不自己拍数。

预注册判据（写在跑之前）：
    ① 两个 arm 的主体命中率之差 <= 2/45 行 -> 无差别偏差，A/B 差值可用；
       差得更多 -> **MAE 必须连同命中率一起报，并讨论方向。**
    ② 我们这版的"主体真被改动"行数 == 0 -> 方法没有删主体；
       > 0 -> 那些行进失败分析，逐个看图。

    python scalediff_probe/subject_track.py
    python scalediff_probe/subject_track.py --cat lone
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_audit import iou, pix_diff, region              # noqa: E402
from count_objects import Detector                       # noqa: E402

HIT_IOU = 0.20          # 锚点命中判据：宽松，我们问的是"认出来没有"不是"框得准"


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def files(rec):
    lo = rec["files"][str(min(int(k) for k in rec["files"]))]
    hi = rec["files"][str(max(int(k) for k in rec["files"]))]
    return lo, hi


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--cat", default=None)
    ap.add_argument("--pctl", type=float, default=95)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))
    if a.cat:
        keys = [k for k in keys if A[k]["cat"] == a.cat]

    det = Detector()
    rows, hit_d, miss = [], [], []
    print(f"{'idx':<5}{'seed':>6}{'cat':<10}{'基图主体':>9}"
          f"{'基线命中':>9}{'我们命中':>9}")
    print("-" * 50)
    for k in keys:
        lo_a, hi_a = files(MA[k])
        _, hi_b = files(MB[k])
        base = Image.open(Path(a.base) / lo_a).convert("RGB")
        b0, s0 = det.detect(base, A[k]["subject"])
        if len(b0) == 0:                      # 基图就没主体（empty 类，或检测失败）
            rows.append((k, None, None, None))
            print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{'无':>9}{'-':>9}{'-':>9}")
            del base
            continue
        j = int(np.argmax(s0))
        scale = Image.open(Path(a.base) / hi_a).width / base.width
        anchor = [float(x) * scale for x in b0[j]]
        del base

        res = {}
        imgs = {}
        for arm, d, fn in (("base", a.base, hi_a), ("ours", a.new, hi_b)):
            im = Image.open(Path(d) / fn).convert("RGB")
            imgs[arm] = im
            bb, _ = det.detect(im, A[k]["subject"])
            res[arm] = any(iou(anchor, list(map(float, x))) >= HIT_IOU for x in bb)

        d_anchor = pix_diff(imgs["base"], imgs["ours"], anchor)
        if res["base"] and res["ours"]:
            hit_d.append(d_anchor)            # 两边都认出来 -> 零假设样本
        else:
            miss.append((k, res["base"], res["ours"], d_anchor))
        rows.append((k, True, res["base"], res["ours"]))
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{'有':>9}"
              f"{'√' if res['base'] else '×':>9}{'√' if res['ours'] else '×':>9}")
        del imgs

    have = [r for r in rows if r[1]]
    hb = sum(1 for r in have if r[2])
    ho = sum(1 for r in have if r[3])
    n = len(have)
    print(f"\n基图有主体的行 {n} 条")
    print(f"  基线 4096² 认出主体 {hb}/{n} = {hb/max(n,1):.0%}")
    print(f"  我们 4096² 认出主体 {ho}/{n} = {ho/max(n,1):.0%}")
    print(f"  判据①  差 {abs(hb-ho)} 行  "
          + ("-> 过：无差别偏差，A/B 差值可用"
             if abs(hb - ho) <= 2 else
             "-> **没过：漏检在两个 arm 上不对称，MAE 必须连同命中率一起报**"))

    if not hit_d:
        print("\n没有两边都命中的行，建不起零假设。")
        return 1
    thr = float(np.percentile(hit_d, a.pctl))
    print(f"\n零假设（两边都命中的行，锚点区域像素差 n={len(hit_d)}）："
          f"中位 {np.median(hit_d):.2f}  {a.pctl:.0f} 分位 {thr:.2f}")

    print(f"\n漏检的 {len(miss)} 行：")
    changed_ours = 0
    for k, rb, ro, d in sorted(miss, key=lambda x: -x[3]):
        who = "基线" if not rb else ""
        who += ("我们" if not ro else "")
        kind = "**主体真的被改动**" if d > thr else "检测器漏检（像素基本没变）"
        if d > thr and not ro:
            changed_ours += 1
        print(f"  {k[0]:02d}_{A[k]['cat']}_s{k[1]:<6} 谁漏={who:<6} "
              f"锚点像素差 {d:6.2f}   {kind}")

    print(f"\n  判据②  我们这版'主体真被改动'的行数 {changed_ours}  "
          + ("-> 过：方法没有删主体" if changed_ours == 0 else
             "-> **这些行进失败分析，逐个看图**"))

    print("""
这张表要进论文的敏感度部分。它回答的不是"检测器准不准"（不准），
而是**"检测器的不准在两个 arm 之间是否对称"** —— 只有对称，A/B 差值才成立。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
