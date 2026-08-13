"""跨臂三列对照：基图 | 基线 4096 | v1 4096 —— 论文方法图的草稿机。

臂内接触表（hi_contact）看不到修复：v1 的 4096 和自己的基图当然
差不多 —— 病是在**基线臂**上发的。修复的证据 = 同一 idx 横着摆：
    基图（1024，两臂共享、逐像素相同）
    基线 4096（病发：331 四尊女神、1059 四辆卡车）
    v1 4096（病除：各一尊/一辆）
标题行同时印两臂的 VLM delta，图和数字互相印证。

    python scalediff_probe/arm_compare.py --idx 331 1059 263 738 1187 1527 583
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_delta(d):
    p = Path(d) / "vlm_delta.jsonl"
    return ({json.loads(l)["idx"]: json.loads(l) for l in p.open()}
            if p.exists() else {})


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(root / "parti_hi"),
                    help="基线臂目录")
    ap.add_argument("--b", default=str(root / "parti_v1"),
                    help="v1 臂目录")
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--cell", type=int, default=900)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "S"),
                    help="病灶区域裁剪：中心 (CX,CY) 与边长 S，全部是 0~1 的"
                         "图幅比例，三列同框。例：--idx 263 --crop 0.18 0.55 0.35"
                         "（整图缩略认不出的克隆，裁出来就是论文图）")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    A, B = Path(a.a), Path(a.b)
    ma = {json.loads(l)["idx"]: json.loads(l)
          for l in (A / "manifest.jsonl").open()}
    mb = {json.loads(l)["idx"]: json.loads(l)
          for l in (B / "manifest.jsonl").open()}
    da, db = load_delta(A), load_delta(B)

    rows = [i for i in a.idx if i in ma and i in mb]
    missing = [i for i in a.idx if i not in ma or i not in mb]
    if missing:
        print(f"跳过（某臂缺图）：{missing}")
    if not rows:
        sys.exit("没有可画的 idx")

    C, pad, cap_h = a.cell, 10, 40
    hdr = 26
    sheet = Image.new("RGB", (3 * (C + pad) + pad,
                              hdr + len(rows) * (C + cap_h + pad) + pad),
                      "white")
    dr = ImageDraw.Draw(sheet)
    for k, title in enumerate(("基图 1024（两臂逐像素相同）",
                               "基线 4096↓（门关）", "v1 4096↓（门开）")):
        dr.text((pad + k * (C + pad) + 4, 6), title, fill="black")

    def prep(im):
        if a.crop:
            cx, cy, s = a.crop
            W, H = im.size
            half = s * W / 2
            x0 = min(max(cx * W - half, 0), W - 2 * half)
            y0 = min(max(cy * H - half, 0), H - 2 * half)
            im = im.crop((int(x0), int(y0),
                          int(x0 + 2 * half), int(y0 + 2 * half)))
        return im.resize((C, C), Image.LANCZOS)

    def img_of(mani_dir, r, res):
        f = r["files"].get(str(res)) or r["files"].get(res)
        return prep(Image.open(Path(mani_dir) / f).convert("RGB")) if f else None

    for n, i in enumerate(rows):
        ra, rb = ma[i], mb[i]
        f_lo = ra["files"].get("1024") or ra["files"].get(1024)
        base = prep(Image.open(A / f_lo).convert("RGB"))
        hi_a = img_of(A, ra, 4096)
        hi_b = img_of(B, rb, 4096)
        y = hdr + pad + n * (C + cap_h + pad)
        for k, im in enumerate((base, hi_a, hi_b)):
            if im:
                sheet.paste(im, (pad + k * (C + pad), y))
        ta = da.get(i, {})
        tb = db.get(i, {})
        dr.text((pad + 4, y + C + 4),
                f"[{i}] {ra['prompt'][:80]}", fill="black")
        dr.text((pad + 4, y + C + 21),
                f"基线 delta={ta.get('delta', '?')}"
                f"（{ta.get('n_base', '?')}->{ta.get('n_hi_dn', '?')}"
                f", subj={ta.get('subject', '?')!r}）   "
                f"v1 delta={tb.get('delta', '?')}"
                f"（{tb.get('n_base', '?')}->{tb.get('n_hi_dn', '?')}）   "
                f"head={mb[i].get('head')!r}",
                fill=(90, 90, 90))
    tag = "_".join(str(i) for i in rows)
    if a.crop:
        tag += f"_crop{a.crop[0]:g}_{a.crop[1]:g}_{a.crop[2]:g}"
    p = Path(a.out) if a.out else B / f"arm_compare_{tag}.jpg"
    sheet.save(p, "JPEG", quality=92)
    print(f"写出 {p}")
    print("左：共享基图；中：基线臂病发；右：v1 臂病除。中列里有、"
          "右列里没有的主体副本 = 门修掉的东西；右列若比左列少了主体 = 误伤。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
