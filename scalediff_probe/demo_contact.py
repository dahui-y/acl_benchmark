"""demo_e4 的臂对照表：同 tag 横排各臂，支持原生分辨率裁剪。

    python scalediff_probe/demo_contact.py --res 4096
    python scalediff_probe/demo_contact.py --res 4096 --crop 0.5 0.75 0.5
"""

import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(root / "demo_e4"))
    ap.add_argument("--tags", nargs="*", default=None,
                    help="默认全部 tag")
    ap.add_argument("--arms", nargs="+", default=["base", "v1"])
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--cell", type=int, default=900)
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "S"))
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    d = Path(a.dir)
    rows = {}
    for l in (d / "manifest.jsonl").open():
        r = json.loads(l)
        rows[(r["tag"], r["arm"], r["size"])] = r
    tags = a.tags or sorted({t for t, _, s in rows if s >= a.res})

    def prep(im):
        if a.crop:
            cx, cy, s = a.crop
            W, H = im.size
            half = s * W / 2
            x0 = min(max(cx * W - half, 0), W - 2 * half)
            y0 = min(max(cy * H - half, 0), H - 2 * half)
            im = im.crop((int(x0), int(y0),
                          int(x0 + 2 * half), int(y0 + 2 * half)))
        return im.resize((a.cell, a.cell), Image.LANCZOS)

    C, pad, cap, hdr = a.cell, 8, 24, 24
    sheet = Image.new("RGB", (len(a.arms) * (C + pad) + pad,
                              hdr + len(tags) * (C + cap + pad) + pad),
                      "white")
    dr = ImageDraw.Draw(sheet)
    for k, arm in enumerate(a.arms):
        dr.text((pad + k * (C + pad) + 4, 5),
                f"DemoFusion {arm}（{'门关' if arm == 'base' else '门开'}）",
                fill="black")
    for n, tag in enumerate(tags):
        y = hdr + pad + n * (C + cap + pad)
        for k, arm in enumerate(a.arms):
            r = rows.get((tag, arm, a.res)) or next(
                (v for (t, ar, s), v in rows.items()
                 if t == tag and ar == arm and s >= a.res), None)
            if not r:
                continue
            f = r["files"].get(str(a.res)) or r["files"].get(a.res)
            if not f or not (d / f).exists():
                continue
            sheet.paste(prep(Image.open(d / f).convert("RGB")),
                        (pad + k * (C + pad), y))
        pr = next((v["prompt"] for (t, _, _), v in rows.items() if t == tag), "")
        dr.text((pad + 4, y + C + 4), f"[{tag}] {pr[:100]}", fill="black")
    tagstr = "_".join(tags)[:60]
    p = d / (f"contact_{tagstr}_{a.res}"
             + (f"_crop{a.crop[0]:g}_{a.crop[1]:g}_{a.crop[2]:g}"
                if a.crop else "") + ".jpg")
    sheet.save(p, "JPEG", quality=92)
    print(f"写出 {p}")
    return 0


if __name__ == "__main__":
    main()
