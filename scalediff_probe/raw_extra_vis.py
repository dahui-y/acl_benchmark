"""把 delta_raw 高的那几张图上"多出来的框"画出来 —— 分辨两个方向相反的解释。

evf_bands 暴露了一个不能用均值糊弄的形态：

    delta_content 各档全平（-0.4 ~ +0.44）
    delta_raw     随 evf 单调上升：0.60 -> 1.67 -> 1.78 -> 3.00 -> 3.29

两列的差（名义上的"漂移"）随 evf 涨了 5 倍。两种解释，含义相反：

  (a) **仪器**：漂移本来就与背景面积相关 —— 空处越多，4096 原生分辨率下
      检测器在纹理上找出的小假框越多。若如此，delta_content 站得住，
      LAION 尾部真的不重复。
  (b) **信号被我自己的刀削掉了**：空处长出的是**小尺寸克隆**（柯基图那种
      1/8 尺度），降采样回 1024 后掉出检测线 —— delta_content 按设计
      抓不到它们，而 delta_raw 抓得到。若如此，**LAION 尾部其实在重复**，
      只是发生在小尺度上，"保守"的尺子把婴儿和洗澡水一起倒了。

数字分不开这两种，**看图能**：把 4096 原生检测框与基图框（×4 放大）配对，
配不上的"多余框"标红。红框落在背景里且框住的是**主体的小副本** -> (b)；
红框框住的是纹理/杂物/半个物体 -> (a)。

    python scalediff_probe/raw_extra_vis.py --n 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_audit import conditioned_text, load_tokenizer   # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "laion_hi"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--min-score", type=float, default=0.50)
    ap.add_argument("--cell", type=int, default=560)
    a = ap.parse_args()

    hi = Path(a.hi)
    deltas = [json.loads(l) for l in (hi / "delta.jsonl").open()]
    mani = {json.loads(l)["idx"]: json.loads(l)
            for l in (hi / "manifest.jsonl").open()}
    rows = sorted(deltas, key=lambda r: -r["delta_raw"])[:a.n]
    print("delta_raw 最高的几条：")
    for r in rows:
        print(f"  [{r['idx']}] raw={r['delta_raw']:+d} dc={r['delta_content']:+d} "
              f"evf={r.get('evf')}  {r['prompt'][:52]}")

    from PIL import Image, ImageDraw
    from count_objects import Detector
    det = Detector(box_thr=a.box_thr)
    tok = load_tokenizer()

    C = a.cell
    cols = 2
    nrow = (len(rows) + cols - 1) // cols
    pad, cap_h = 10, 34
    sheet = Image.new("RGB", (cols * (C + pad) + pad,
                              nrow * (C + cap_h + pad) + pad), "white")
    dr0 = ImageDraw.Draw(sheet)

    for k, r in enumerate(rows):
        m = mani[r["idx"]]
        f4 = m["files"].get("4096") or m["files"].get(4096)
        f1 = m["files"].get("1024") or m["files"].get(1024)
        im4 = Image.open(hi / f4).convert("RGB")
        im1 = Image.open(hi / f1).convert("RGB")
        text, _, _ = conditioned_text(tok, r["prompt"])
        b4, _ = det.detect(im4, text, min_score=a.min_score)
        b1, _ = det.detect(im1, text, min_score=a.min_score)
        s = im4.width / im1.width
        b1s = [[x0 * s, y0 * s, x1 * s, y1 * s] for x0, y0, x1, y1 in b1]

        sc = im4.resize((C, C), Image.LANCZOS)
        d = ImageDraw.Draw(sc)
        f = C / im4.width
        extra = 0
        for bb in b4:
            matched = any(iou(bb, gg) > 0.3 for gg in b1s)
            color = (0, 200, 80) if matched else (255, 40, 40)
            extra += not matched
            d.rectangle([bb[0] * f, bb[1] * f, bb[2] * f, bb[3] * f],
                        outline=color, width=3)
        for gg in b1s:      # 基图框的位置，虚线感用细白框代替
            d.rectangle([gg[0] * f, gg[1] * f, gg[2] * f, gg[3] * f],
                        outline=(255, 255, 255), width=1)
        x = pad + (k % cols) * (C + pad)
        y = pad + (k // cols) * (C + cap_h + pad)
        sheet.paste(sc, (x, y))
        dr0.text((x + 2, y + C + 2),
                 f"[{r['idx']}] raw={r['delta_raw']:+d} dc={r['delta_content']:+d}"
                 f" evf={r.get('evf')}  红框(未配对)={extra}", fill="black")
        dr0.text((x + 2, y + C + 17), r["prompt"][:70], fill=(90, 90, 90))
        print(f"  [{r['idx']}] 4096 检出 {len(b4)}，其中未配对 {extra}")

    p = hi / "raw_extra.jpg"
    sheet.save(p, "JPEG", quality=90)
    print(f"""
写出 {p}
红框 = 4096 上有、与任何基图框（白框，×4）配不上的检测；绿框 = 配上的。
逐张问：红框框住的是**主体的小副本**（(b)：小尺度重复，delta_content
的保守性把它削掉了），还是纹理/杂物/半个物体（(a)：漂移与背景面积相关，
delta_content 站得住）？""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
