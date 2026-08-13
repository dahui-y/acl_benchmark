"""v2b 眼睛终审：base | v1 | v2b_g1.5 | v2b_g2 四列对照。

P1' 的数字（×34~×160）本身分不出"真纹理"和"没去干净的残余噪声"——
这一判只能眼睛来。--crop 用原生分辨率裁背景区域（不给就整图缩略）。

    python scalediff_probe/v2b_look.py                       # 三行整图
    python scalediff_probe/v2b_look.py --idx 4 --crop 0.25 0.65 0.18
"""

import argparse
import json
import os
import sys
from pathlib import Path

ARMS = ["base", "v1", "v2b_g1.5", "v2b_g2"]


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(root / "method_v2"))
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--cell", type=int, default=700)
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "S"))
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    d = Path(a.dir)
    rows = {}
    for l in (d / "manifest.jsonl").open():
        r = json.loads(l)
        if "arm" in r:
            rows[(r["idx"], r["arm"])] = r

    C, pad, cap, hdr = a.cell, 8, 22, 24
    sheet = Image.new(
        "RGB", (len(a.arms) * (C + pad) + pad,
                hdr + len(a.idx) * (C + cap + pad) + pad), "white")
    dr = ImageDraw.Draw(sheet)
    for k, arm in enumerate(a.arms):
        dr.text((pad + k * (C + pad) + 4, 5), arm, fill="black")

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

    for n, i in enumerate(a.idx):
        y = hdr + pad + n * (C + cap + pad)
        for k, arm in enumerate(a.arms):
            r = rows.get((i, arm))
            if not r:
                continue
            f = r["files"].get(str(a.res)) or r["files"].get(a.res)
            if not f or not (d / f).exists():
                continue
            sheet.paste(prep(Image.open(d / f).convert("RGB")),
                        (pad + k * (C + pad), y))
        pr = rows.get((i, a.arms[0]), {}).get("prompt", "")
        dr.text((pad + 4, y + C + 3), f"[{i}] {pr[:100]}", fill="black")

    tag = "_".join(str(i) for i in a.idx)
    if a.crop:
        tag += f"_crop{a.crop[0]:g}_{a.crop[1]:g}_{a.crop[2]:g}"
    p = d / f"v2b_look_{tag}.jpg"
    sheet.save(p, "JPEG", quality=92)
    print(f"写出 {p}")
    print("判据：背景是**有语义的微纹理**（植被颗粒/岩石肌理）还是"
          "**均匀彩色噪点**。噪点 -> 该 γ 不合格；纹理 -> 接 P2'/armdelta。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
