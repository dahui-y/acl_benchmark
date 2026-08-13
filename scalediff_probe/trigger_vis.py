"""把空视野比例画出来 —— 在信它或弃它之前，先看它和人眼一致不一致。

现在的处境是：LAION 基图上 evf 几乎恒为 0。有两种完全不同的解释，
而数字本身分不开：

  · **仪器失效** —— 框把不该算的东西也算成了主体；
  · **语料如此** —— aesthetic 子集是商品/美食/室内图，摄影惯例就是主体
    居中占满画面，所以确实没有空视野。

`trigger_debug.py` 已经排除了"整图框是误检"这条：`Cafe interior` 上
0.95 的框是**对的**，那张图的主体本来就是整个室内，evf=0 在语义上正确。

分不开就别用数字分。**画出来看**：基图 + 检测框 + R×R 网格 + 每块判定，
人眼扫一遍就知道 evf 报的是不是那张图真实的构图。

看什么：
  · 框漏了主体（该有框的地方没框）      -> 仪器问题，去调 box_thr / min_score
  · 框住了不该算的（背景、水印、边框）  -> 仪器问题，去调过滤
  · 框都对，就是没有空块              -> **语料如此**，按 §6.9.0a 第三档走
  · 空块判定与人眼不符（擦到边算占用） -> 去调 5% 交集判据

    python scalediff_probe/trigger_vis.py --n 12
    python scalediff_probe/trigger_vis.py --n 12 --min-score 0.30 --tag lo
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import empty_view_fraction        # noqa: E402


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=384)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--min-score", type=float, default=0.50,
                    help="**detect() 在 box_thr 之后还筛一道 min_score。**"
                         "trigger_debug 实测它把 [9][12] 的 3 个框全吃掉、"
                         "evf 被记成 1.00 再被当作'无定义'排除。门要召回，"
                         "这个二次筛正好反着来 —— 所以这里能调，且要并列看两档。")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--idx-file", default=None,
                    help="只画这份名单（parti_profile --write 出的 "
                         "trigger_clean_*.json，键 alive_idx）")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    from count_objects import Detector

    base = Path(a.base)
    rows = [json.loads(l) for l in (base / "manifest.jsonl").open()]
    if a.idx_file:
        want = set(json.loads(Path(a.idx_file).read_text())["alive_idx"])
        rows = [r for r in rows if r["idx"] in want]
        a.n = max(a.n, len(rows))
    rows = rows[:a.n]
    det = Detector(box_thr=a.box_thr)

    C, R = a.cell, a.R
    cols = a.cols
    nrow = (len(rows) + cols - 1) // cols
    pad, cap_h = 8, 34
    sheet = Image.new("RGB", (cols * (C + pad) + pad,
                              nrow * (C + cap_h + pad) + pad), "white")
    dr = ImageDraw.Draw(sheet)

    for k, r in enumerate(rows):
        im = Image.open(base / r["file"]).convert("RGB")
        b, _ = det.detect(im, r["prompt"], min_score=a.min_score)
        evf = empty_view_fraction(b, im.width, im.height, R)
        sc = im.resize((C, C), Image.LANCZOS)
        d = ImageDraw.Draw(sc, "RGBA")
        s = C / im.width
        tw = C / R
        # 先给"空块"铺红底，人眼一眼看到判定结果，而不是自己去数
        for rr in range(R):
            for cc in range(R):
                x0, y0 = cc * tw, rr * tw
                hit = any(
                    max(0.0, min(x0 + tw, bb[2] * s) - max(x0, bb[0] * s)) *
                    max(0.0, min(y0 + tw, bb[3] * s) - max(y0, bb[1] * s))
                    > 0.05 * tw * tw for bb in b)
                if not hit:
                    d.rectangle([x0, y0, x0 + tw, y0 + tw], fill=(255, 0, 0, 60))
                d.rectangle([x0, y0, x0 + tw, y0 + tw], outline=(255, 255, 255, 160))
        for bb in b:
            d.rectangle([bb[0] * s, bb[1] * s, bb[2] * s, bb[3] * s],
                        outline=(0, 200, 255), width=3)
        x = pad + (k % cols) * (C + pad)
        y = pad + (k // cols) * (C + cap_h + pad)
        sheet.paste(sc, (x, y))
        dr.text((x + 2, y + C + 2),
                f"[{r['idx']}] evf={evf:.2f} nbox={len(b)}", fill="black")
        dr.text((x + 2, y + C + 17), r["prompt"][:52], fill=(90, 90, 90))

    tag = a.tag or f"thr{a.box_thr:g}_ms{a.min_score:g}"
    p = Path(a.out) if a.out else base / f"trigger_vis_{tag}.jpg"
    sheet.save(p, "JPEG", quality=92)
    print(f"写出 {p}")
    print("""
红色 = 判为空视野的块；青色 = 检测框。逐张问自己：
  框漏了主体？              -> 仪器问题，调 box_thr / min_score
  框住了背景/水印/整幅边框？ -> 仪器问题，调过滤
  框都对、就是没有红块？     -> **语料如此**，按 §6.9.0a 第三档走，不是尺子的错
  红块判定与人眼不符？       -> 调 empty_view_fraction 的 5% 交集判据""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
