"""多臂对照图：一行一个 idx，一列一个方法，标题印 delta。

表格说服不了眼睛。这个脚本把同一个 (prompt, seed) 在各个方法下的 4096
摆在一起，第一列是基图（真值），后面每列一个方法，标题印该臂的 delta。

    python scalediff_probe/multi_look.py --idx 331 734 1187 1059
    python scalediff_probe/multi_look.py --idx 331 --cell 900     # 单条看大图
    python scalediff_probe/multi_look.py --idx 734 --crop 0.5 0.5 0.4   # 裁病灶

注意：各臂的**基图不是同一张**（管线不同）。所以第一列画的是**每一臂
自己的**基图缩略拼在一起还是只画参照臂的？—— 这里只画参照臂（--ref，
默认 parti_hi）的基图当构图参照，因为 Δdelta 本来就是各减各的，
把四张基图都摆出来只会挤掉正片。要逐臂看基图用 --with-base。
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def load_delta(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return {}
    return {int(json.loads(l)["idx"]): json.loads(l)
            for l in p.open() if l.strip()}


def pick(d, idx, want_base=False):
    """取该臂该 idx 的图：want_base 取 1024，否则取最大分辨率。"""
    d = Path(d)
    if want_base:
        p = d / f"{idx:05d}_1024.png"
        return p if p.exists() else None
    cands = sorted(d.glob(f"{idx:05d}_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))
    cands = [p for p in cands if p.stem.split("_")[1] != "1024"]
    return cands[-1] if cands else None


def cellimg(p, cell, crop=None):
    im = Image.open(p).convert("RGB")
    if crop:
        cx, cy, s = crop
        w, h = im.size
        half = s / 2
        im = im.crop((int((cx - half) * w), int((cy - half) * h),
                      int((cx + half) * w), int((cy + half) * h)))
    return im.resize((cell, cell), Image.LANCZOS)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                   help="名字=目录；默认 基线/v1.2/AccDiffusion")
    ap.add_argument("--ref", default=None, help="画基图的参照臂目录")
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--cell", type=int, default=520)
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "S"))
    ap.add_argument("--with-base", action="store_true",
                    help="每一臂都多画一列自己的基图")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.arms:
        arms = {}
        for s in a.arms:
            k, v = s.split("=", 1)
            arms[k] = Path(v if "/" in v else str(root / v))
    else:
        arms = {"ScaleDiff": root / "parti_hi",
                "Ours (v1.2)": root / "parti_v12",
                "AccDiffusion": root / "parti_acc"}
    ref = Path(a.ref) if a.ref else next(iter(arms.values()))
    deltas = {k: load_delta(v) for k, v in arms.items()}

    cols = []           # [(标题, 取图函数)]
    cols.append(("base 1024 (reference)", lambda i: pick(ref, i, True)))
    for k, d in arms.items():
        if a.with_base:
            cols.append((f"{k} base", lambda i, d=d: pick(d, i, True)))
        cols.append((k, lambda i, d=d: pick(d, i)))

    C, pad = a.cell, 26
    rows = []
    for i in a.idx:
        imgs = []
        for title, fn in cols:
            p = fn(i)
            lab = title
            if title in deltas and i in deltas[title]:
                r = deltas[title][i]
                lab = (f"{title}   delta {r['delta']:+d}"
                       f"   (base {r.get('n_base','?')} -> "
                       f"hi {r.get('n_hi_dn','?')})")
            imgs.append((p, lab))
        if not any(p for p, _ in imgs):
            print(f"[{i}] 一张图都找不到，跳过")
            continue
        strip = Image.new("RGB", (C * len(imgs), C + pad), "white")
        dr = ImageDraw.Draw(strip)
        for j, (p, lab) in enumerate(imgs):
            if p:
                strip.paste(cellimg(p, C, a.crop), (j * C, 0))
            else:
                dr.rectangle([j * C, 0, (j + 1) * C, C], fill="#eeeeee")
                dr.text((j * C + 8, C // 2), "missing", fill="black")
            dr.text((j * C + 6, C + 6), lab, fill="black")
        prm = next((deltas[k][i].get("prompt", "") for k in deltas
                    if i in deltas[k]), "")
        hdr = Image.new("RGB", (strip.width, 20), "white")
        ImageDraw.Draw(hdr).text((6, 4), f"[{i}]  {prm[:150]}", fill="black")
        merged = Image.new("RGB", (strip.width, strip.height + 20), "white")
        merged.paste(hdr, (0, 0))
        merged.paste(strip, (0, 20))
        rows.append(merged)

    if not rows:
        raise SystemExit("没有可出图的 idx")
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)),
                      "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    op = Path(a.out) if a.out else root / "multi_look.jpg"
    sheet.save(op, quality=90)
    print(f"{len(rows)} 行 -> {op}   ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
