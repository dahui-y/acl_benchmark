"""把"我们的方法修掉了重复"做成一张能看的对比图。

不生成任何东西，只用已经跑完的两个 arm 的输出。三栏：

  ①  4096² 全图，基线 | 我们的        —— 看整体没坏
  ②  检测标注图（count_objects 已存）  —— 看多出来的物体被谁数到了
  ③  **1:1 原像素裁块**，同坐标        —— 看细节，这一栏才是重点：
      裁块位置自动选"两张图差别最大的地方"，也就是被改掉的那块，
      不是我挑的。缩略图上看不出 4096² 的差别，必须原像素看。

  python scalediff_probe/make_figure.py                  # 自动挑改善最大的一条
  python scalediff_probe/make_figure.py --idx 0 --seed 77
  python scalediff_probe/make_figure.py --top 3          # 出三张
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W_COL = 900          # 全图缩略宽
CROP = 768           # 1:1 裁块边长
BAR = 34             # 标题条高


def load(d):
    p = Path(d) / "counts.json"
    if not p.exists():
        sys.exit(f"没有 {p}，先跑 count_objects.py --batch {d}")
    return {(r["idx"], r["seed"]): r for r in json.loads(p.read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def hi_file(rec):
    res = max(int(k) for k in rec["files"])
    return rec["files"][str(res)] if str(res) in rec["files"] else rec["files"][res]


def label(img, text):
    """在图上方加一条标题。"""
    out = Image.new("RGB", (img.width, img.height + BAR), "white")
    out.paste(img, (0, BAR))
    ImageDraw.Draw(out).text((8, 9), text, fill="black")
    return out


def hstack(imgs, gap=16):
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.width + gap
    return out


def vstack(imgs, gap=22):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for i in imgs:
        out.paste(i, ((w - i.width) // 2, y)); y += i.height + gap
    return out


def worst_diff_box(a, b, size, step=256):
    """找两张图差别最大的 size×size 区域。返回左上角坐标。

    在 1/8 缩略图上算差分再放大定位 —— 全分辨率滑窗太慢，而我们只要
    大致位置。用的是绝对差的区域和，不做任何美化挑选。
    """
    s = 8
    ga = np.asarray(a.convert("L").resize((a.width // s, a.height // s))).astype(np.int16)
    gb = np.asarray(b.convert("L").resize((b.width // s, b.height // s))).astype(np.int16)
    d = np.abs(ga - gb)
    k = max(size // s, 1)
    # 积分图求所有 k×k 窗口的和
    ii = np.pad(d, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = d.shape
    best, bxy = -1, (0, 0)
    for y in range(0, H - k + 1, max(step // s, 1)):
        for x in range(0, W - k + 1, max(step // s, 1)):
            v = ii[y + k, x + k] - ii[y, x + k] - ii[y + k, x] + ii[y, x]
            if v > best:
                best, bxy = v, (x * s, y * s)
    x, y = bxy
    return (min(x, a.width - size), min(y, a.height - size))


def build(base_dir, new_dir, key, A, B, MA, MB, out_dir, at=None):
    idx, seed = key
    fa, fb = hi_file(MA[key]), hi_file(MB[key])
    ia = Image.open(Path(base_dir) / fa).convert("RGB")
    ib = Image.open(Path(new_dir) / fb).convert("RGB")
    res = ia.width
    ra, rb = A[key], B[key]
    card = ra.get("card")
    na = ra["counts"][str(res)] if str(res) in ra["counts"] else ra["counts"][res]
    nb = rb["counts"][str(res)] if str(res) in rb["counts"] else rb["counts"][res]

    def th(im):
        return im.resize((W_COL, W_COL * im.height // im.width), Image.LANCZOS)

    row1 = hstack([
        label(th(ia), f"ScaleDiff (baseline)   {res}x{res}   detected {na}"
                      + (f"  /  prompt says {card}" if card is not None else "")),
        label(th(ib), f"+ ours   {res}x{res}   detected {nb}"
                      + (f"  /  prompt says {card}" if card is not None else "")),
    ])

    rows = [row1]

    tag_a = f"{idx:02d}_{ra['cat']}_s{seed}_det_{res}.png"
    pa, pb = Path(base_dir) / tag_a, Path(new_dir) / tag_a
    if pa.exists() and pb.exists():
        rows.append(hstack([
            label(th(Image.open(pa).convert("RGB")), "detector boxes - baseline"),
            label(th(Image.open(pb).convert("RGB")), "detector boxes - ours"),
        ]))

    # 1:1 裁块：默认位置由差分自动选（不是挑的）；--at 可指定，
    # 用于回看 sharpness_map 报出来的具体 tile 坐标。
    x, y = at if at else worst_diff_box(ia, ib, CROP)
    x, y = min(x, ia.width - CROP), min(y, ia.height - CROP)
    ca = ia.crop((x, y, x + CROP, y + CROP))
    cb = ib.crop((x, y, x + CROP, y + CROP))
    # **基图同一位置也放上来。** 没有这一栏就没法区分三种可能：
    #   基图本来就糊     -> 继承，LFM 结构上修不了，唯一的杠杆是全局 τ
    #   基图清楚、4096 糊 -> 外推阶段产生的（SG 低频锁定过强？）
    #   基图那里什么都没有 -> **那不是糊，是没画完的幻影**，属于我们打的失效
    lo = MA[key]["files"][str(min(int(k) for k in MA[key]["files"]))]
    base = Image.open(Path(base_dir) / lo).convert("RGB")
    s = base.width / ia.width
    cbase = base.crop((int(x*s), int(y*s), int((x+CROP)*s), int((y+CROP)*s))) \
                .resize((CROP, CROP), Image.NEAREST)   # NEAREST：不伪造细节
    rows.append(hstack([
        label(cbase, f"base {base.width} @ same spot (NEAREST x{1/s:.0f})"),
        label(ca, f"1:1 pixels @ ({x},{y}) - baseline"),
        label(cb, f"1:1 pixels @ ({x},{y}) - ours"),
    ]))
    del base

    fig = vstack(rows)
    cap = Image.new("RGB", (fig.width, 58), "white")
    d = ImageDraw.Draw(cap)
    d.text((8, 8), f'[{idx:02d} {ra["cat"]} seed {seed}]  "{ra["prompt"][:130]}"',
           fill="black")
    d.text((8, 32), f'excess  baseline {ra["excess"]:+}  ->  ours {rb["excess"]:+}'
                    if ra.get("excess") is not None else "", fill="black")
    fig = vstack([cap, fig], gap=0)

    out = Path(out_dir) / f"fig_{idx:02d}_{ra['cat']}_s{seed}.png"
    fig.save(out)
    return out


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top", type=int, default=1, help="改善最大的前几条")
    ap.add_argument("--at", default=None, metavar="X,Y",
                    help="指定 1:1 裁块左上角，例如 --at 512,1024；"
                         "省略则取两图差分最大处")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))

    if a.idx is not None:
        keys = [k for k in keys if k[0] == a.idx and (a.seed is None or k[1] == a.seed)]
        if not keys:
            sys.exit(f"没有 idx={a.idx} seed={a.seed} 这一条")
    else:
        scored = [(abs(A[k]["excess"]) - abs(B[k]["excess"]), k) for k in keys
                  if A[k].get("excess") is not None and B[k].get("excess") is not None]
        scored.sort(reverse=True)
        keys = [k for _, k in scored[:a.top]]
        print("改善最大的：" + ", ".join(
            f"{k[0]:02d}/{k[1]} ({s:+})" for s, k in scored[:a.top]))

    out_dir = Path(a.out) if a.out else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    at = tuple(int(v) for v in a.at.split(",")) if a.at else None
    for k in keys:
        p = build(a.base, a.new, k, A, B, MA, MB, out_dir, at=at)
        print(f"写出 {p}")
    print("\n第三栏是 1:1 原像素，裁块位置由两图差分自动选定 —— "
          "缩略图上看不出 4096² 的差别，看那一栏。")


if __name__ == "__main__":
    main()
