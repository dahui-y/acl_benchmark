"""消失的框，到底是物体被去掉了，还是检测器不叫了？

起因是 03_lone（湖上小船）那张对比图：基线 8 个框里有一个扣在**船的倒影**
上，我们这版没有那个框 —— 可倒影在两张图里都还在。也就是说这一格的
8 -> 1 里，至少有一部分不是"我们删掉了重复物体"，而是"检测器这次没叫"。

这是尺子的问题，不是方法的问题，但它会直接污染主结果（MAE 5.20 -> 1.67）。
必须查清楚，而且不能靠人看。

做法（零标注，全自动）：

    1. 两个 arm 各跑一遍检测，按 IoU 配对；
    2. 基线有、我们没有的框 = **消失的框**；
    3. 对每个消失的框，把**同一坐标**的区域从两张图各裁一块，比像素；
    4. 判据不是我拍的阈值 —— 用**配对上的框**（同一物体、两个 arm 都在）
       的像素差分布当零假设。消失框的差若落在配对框差的 95 分位之内，
       说明那块像素基本没变、框却没了 -> **检测器抖动**；
       超出则说明那块内容真的变了 -> **真删除**。

    输出每张图的 真删除 / 检测器抖动 数量，以及一个总的比例。
    同时把消失框的裁块对存盘，方便抽查。

    python scalediff_probe/box_audit.py                 # 全批
    python scalediff_probe/box_audit.py --idx 3 --seed 2025
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector                    # noqa: E402

MARGIN = 0.25          # 裁块比框大这么多，包含边缘上下文
MIN_SIDE = 48          # 太小的框裁出来没法比，补到这个边长


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def hi_file(rec):
    res = max(int(k) for k in rec["files"])
    return rec["files"][str(res)]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def region(box, W, H):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0 -= w * MARGIN; x1 += w * MARGIN
    y0 -= h * MARGIN; y1 += h * MARGIN
    if x1 - x0 < MIN_SIDE:
        c = (x0 + x1) / 2; x0, x1 = c - MIN_SIDE / 2, c + MIN_SIDE / 2
    if y1 - y0 < MIN_SIDE:
        c = (y0 + y1) / 2; y0, y1 = c - MIN_SIDE / 2, c + MIN_SIDE / 2
    return (int(max(0, x0)), int(max(0, y0)),
            int(min(W, x1)), int(min(H, y1)))


def pix_diff(ia, ib, box):
    """同坐标裁块的平均绝对差（0-255 灰度）。"""
    r = region(box, ia.width, ia.height)
    ca = np.asarray(ia.crop(r).convert("L"), dtype=np.int16)
    cb = np.asarray(ib.crop(r).convert("L"), dtype=np.int16)
    if ca.size == 0:
        return 0.0
    return float(np.abs(ca - cb).mean())


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cat", default=None, help="只看某一类，比如 lone")
    ap.add_argument("--match-iou", type=float, default=0.30)
    ap.add_argument("--pctl", type=float, default=95,
                    help="零假设分位数；配对框差的这个分位以内算检测器抖动")
    ap.add_argument("--dump", type=int, default=12, help="存多少对裁块供抽查")
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))
    if a.idx is not None:
        keys = [k for k in keys if k[0] == a.idx]
    if a.seed is not None:
        keys = [k for k in keys if k[1] == a.seed]
    if a.cat:
        keys = [k for k in keys if A[k]["cat"] == a.cat]
    if not keys:
        sys.exit("没有符合条件的行")

    det = Detector()
    out_dir = root / "box_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    matched_d, gone = [], []          # gone: (diff, key, box, score)
    for k in keys:
        ia = Image.open(Path(a.base) / hi_file(MA[k])).convert("RGB")
        ib = Image.open(Path(a.new) / hi_file(MB[k])).convert("RGB")
        subj = A[k]["subject"]
        ba, sa = det.detect(ia, subj)
        bb, sb = det.detect(ib, subj)

        used = set()
        m, g = [], []
        for i, box in enumerate(ba):
            best, bj = 0.0, -1
            for j, box2 in enumerate(bb):
                if j in used:
                    continue
                v = iou(box, box2)
                if v > best:
                    best, bj = v, j
            if best >= a.match_iou:
                used.add(bj)
                m.append(box)
            else:
                g.append((box, float(sa[i])))
        d_m = [pix_diff(ia, ib, box) for box in m]
        d_g = [pix_diff(ia, ib, box) for box, _ in g]
        matched_d += d_m
        gone += [(d, k, box, s) for d, (box, s) in zip(d_g, g)]
        per_row.append((k, len(ba), len(bb), len(m), len(g), d_m, d_g))
        del ia, ib                     # 4096² 一张 50 MB，别攒着
        print(f"{k[0]:02d}_{A[k]['cat']}_s{k[1]:<6} 基线 {len(ba):>2} 框 -> "
              f"我们 {len(bb):>2} 框   配对 {len(m):>2}   消失 {len(g):>2}")

    if not matched_d:
        print("\n没有配对上的框，无法建立零假设。换更宽的 --match-iou 或更多行。")
        return 1
    thr = float(np.percentile(matched_d, a.pctl))
    print(f"\n零假设（配对框的像素差，n={len(matched_d)}）："
          f"中位 {np.median(matched_d):.2f}   {a.pctl:.0f} 分位 {thr:.2f}")
    print("  —— 这是同一个物体在两个 arm 里的正常差异幅度，不是我定的阈值。")

    real = [x for x in gone if x[0] > thr]
    flick = [x for x in gone if x[0] <= thr]
    n = len(gone)
    print(f"\n消失的框 {n} 个：")
    print(f"  真删除（像素差超出零假设）  {len(real):>3}  {len(real)/max(n,1):.0%}")
    print(f"  检测器抖动（像素基本没变）  {len(flick):>3}  {len(flick)/max(n,1):.0%}")

    print(f"\n{'idx':<5}{'seed':>6}{'cat':<10}{'基线框':>7}{'我们框':>7}"
          f"{'消失':>5}{'真删':>5}{'抖动':>5}")
    print("-" * 52)
    for k, na, nb, nm, ng, d_m, d_g in per_row:
        r = sum(1 for d in d_g if d > thr)
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{na:>7}{nb:>7}"
              f"{ng:>5}{r:>5}{ng - r:>5}")

    # 抽查用的裁块对：抖动的排在前面（那是要盯的）
    flick.sort(key=lambda x: x[0])
    for rank, (d, k, box, s) in enumerate(flick[:a.dump]):
        ia = Image.open(Path(a.base) / hi_file(MA[k])).convert("RGB")
        ib = Image.open(Path(a.new) / hi_file(MB[k])).convert("RGB")
        r = region(box, ia.width, ia.height)
        ca, cb = ia.crop(r), ib.crop(r)
        pair = Image.new("RGB", (ca.width * 2 + 8, ca.height), "white")
        pair.paste(ca, (0, 0)); pair.paste(cb, (ca.width + 8, 0))
        pair.save(out_dir / f"flicker_{rank:02d}_{k[0]:02d}_s{k[1]}_"
                            f"d{d:.1f}_score{s:.2f}.png")
    print(f"\n抖动裁块对（左基线 / 右我们）存于 {out_dir}/flicker_*.png")

    print("""
怎么读这张表：
  抖动占比低  -> 主结果站得住：消失的框确实对应内容被去掉了。
  抖动占比高  -> **MAE 的改善有一部分是尺子在动，不是图在变**，
                 必须在论文里降级处理：要么把计数改成"只统计像素差
                 超出零假设的框"，要么正文写明这一比例。
无论哪种，这个比例都要报 —— 它是尺子的可信度，不是可选项。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
