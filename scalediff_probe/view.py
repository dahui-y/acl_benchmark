"""把一次 run_one 的输出摊开来看。生成的都是能直接截图发出来的尺寸。

三张图，各回答一个问题：

  A_downsample.png   4096² 下采样回 1024²，与基图并排 + 放大后的差异图
                     —— 这就是 P0 的判据本身：合法的细节增强活在基图 Nyquist
                     之上，下采样后应当消失；差异图上还剩下的东西，是模型
                     在基图分辨率上改动的内容。

  B_crops.png        4096² 的【原分辨率】裁块，配上基图同位置放大 4× 的样子
                     —— 看细节是真的多了，还是长出了基图里没有的东西。
                     默认裁背景区，因为作者自己在 Limitations 里点名
                     "repetitive artifacts may still occur in background regions"。

  C_grid.png         4096² 缩略图 + 512 像素网格
                     —— NPA 的 query 块不重叠、无平滑（SDXL 版连 FLUX 那个
                     query_random_jitter 都没有）。从代码算：无论 2048² 还是
                     4096²，query 块在图像空间上都是 512 px 一格。所以若真有
                     接缝，它应当落在 512 的整数倍上。对齐了才算证实。

    python scalediff_probe/view.py                       # 用 SD_OUT/run_one
    python scalediff_probe/view.py --dir <某个目录> --seed 77
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(d, seed, stage, res):
    p = Path(d) / f"s{seed}_stage{stage}_{res}.png"
    if not p.is_file():
        raise SystemExit(f"找不到 {p}\n目录里有：" +
                         "\n  ".join(sorted(x.name for x in Path(d).glob('*.png'))))
    return Image.open(p).convert("RGB")


def label(im, text, h=34):
    out = Image.new("RGB", (im.width, im.height + h), (255, 255, 255))
    out.paste(im, (0, h))
    ImageDraw.Draw(out).text((6, 8), text, fill=(0, 0, 0))
    return out


def hstack(ims, gap=10):
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    H = max(i.height for i in ims)
    s = Image.new("RGB", (W, H), (255, 255, 255))
    x = 0
    for i in ims:
        s.paste(i, (x, 0))
        x += i.width + gap
    return s


def panel_a(base, hi, out, amp):
    """基图 | 4096 下采样回 1024 | 差异 ×amp"""
    down = hi.resize(base.size, Image.LANCZOS)
    b = np.asarray(base, np.float32)
    d = np.asarray(down, np.float32)
    diff = np.clip(np.abs(d - b) * amp, 0, 255).astype(np.uint8)
    strip = hstack([
        label(base, f"base {base.width}²"),
        label(down, f"{hi.width}² -> downsampled to {base.width}²"),
        label(Image.fromarray(diff), f"|diff| x{amp}"),
    ])
    strip.save(out / "A_downsample.png")
    m = np.abs(d - b).mean()
    print(f"A_downsample.png   平均绝对差 {m:.2f}/255   "
          f"(合法细节增强应当接近 0；大的地方是基图分辨率上被改动的内容)")


def panel_b(base, hi, out, boxes, size):
    """4096 原分辨率裁块 vs 基图同位置放大"""
    k = hi.width // base.width
    rows = []
    for (cx, cy) in boxes:
        x, y = int(cx * hi.width) - size // 2, int(cy * hi.height) - size // 2
        x = max(0, min(x, hi.width - size))
        y = max(0, min(y, hi.height - size))
        crop_hi = hi.crop((x, y, x + size, y + size))
        crop_lo = base.crop((x // k, y // k, x // k + size // k, y // k + size // k)) \
                      .resize((size, size), Image.NEAREST)
        rows.append(hstack([label(crop_lo, f"base@({cx:.2f},{cy:.2f}) x{k} nearest"),
                            label(crop_hi, f"{hi.width}² native")]))
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 10 * (len(rows) - 1)
    s = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        s.paste(r, (0, y))
        y += r.height + 10
    s.save(out / "B_crops.png")
    print(f"B_crops.png        {len(boxes)} 个位置，{size}px 原分辨率裁块")


def panel_c(hi, out, grid, view):
    """缩略图 + query 块网格。对齐了才算证实接缝来自 NPA。"""
    sc = view / hi.width
    im = hi.resize((view, int(hi.height * sc)), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    n = 0
    for x in range(grid, hi.width, grid):
        d.line([(x * sc, 0), (x * sc, im.height)], fill=(255, 0, 0), width=1)
        n += 1
    for y in range(grid, hi.height, grid):
        d.line([(0, y * sc), (im.width, y * sc)], fill=(255, 0, 0), width=1)
    label(im, f"{hi.width}²  +  {grid}px NPA query-tile grid").save(out / "C_grid.png")
    print(f"C_grid.png         {grid}px 网格（{n + 1}×{n + 1} 块）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(os.environ.get("SD_OUT", "./scalediff_out")) / "run_one"))
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--amp", type=int, default=4, help="差异图放大倍数")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--grid", type=int, default=512, help="NPA query 块在图像空间的间距")
    ap.add_argument("--view", type=int, default=1400)
    ap.add_argument("--boxes", default="0.5,0.18 0.15,0.35 0.85,0.35 0.5,0.5",
                    help="相对坐标 cx,cy，空格分隔。默认偏向天空/远景等背景区")
    a = ap.parse_args()

    out = Path(a.dir)
    base = load(out, a.seed, a.stage, 1024)
    hi = load(out, a.seed, a.stage, 1024 * 2 ** a.stage)
    print(f"base {base.size}   hi {hi.size}\n")

    boxes = [tuple(float(v) for v in b.split(",")) for b in a.boxes.split()]
    panel_a(base, hi, out, a.amp)
    panel_b(base, hi, out, boxes, a.crop)
    panel_c(hi, out, a.grid, a.view)
    print(f"\n三张图在 {out}")


if __name__ == "__main__":
    main()
