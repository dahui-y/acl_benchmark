#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""臂 B：把 1024² 基图双三次上采到 4096² —— **完全不添加细节**的平凡基线。

    这一臂在这条线的所有论文里都只出现在**定性图**里（FAM Figure 5 的
    "Direct Upsampling"、AccDiffusion 的对比图），**没有人把它放进表**。
    但它恰恰是「过平滑 / 细节不足」这个问题的零点：

        臂 A  1024² 原生            —— 天花板。FID 把图缩到 299，
                                       所以它衡量的是**全局结构**，看不见细节
        臂 B  A 双三次上采到 4096²   —— 像素数一样，细节一点没多
        臂 C  真的在 4096² 生成      —— 在位者

    两个正交的读数：

        A vs C（整图 FID）  = 4× 还差多少**全局结构**
                              （细节缩到 299 就没了，不可能贡献这个差）
        B vs C（FIDp/裁块） = 相对「直接上采」还多出多少**真细节**

    **B ≈ C 就是一个可命名、可量化、且肉眼可验的失效**：方法在 4× 上
    没有产生比双三次更多的细节。B ≪ C 则说明细节确实被造出来了，
    那「过平滑」这个提法本身要重新定义。

    这一臂零 GPU：只是重采样。

用法：
    python scalediff_probe/bicubic_arm.py --src <1024目录> --out <4096目录>
"""

import argparse
from pathlib import Path

from PIL import Image

EXT = {".png", ".jpg", ".jpeg"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="1024² 基图目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--filter", default="bicubic",
                    choices=["bicubic", "lanczos", "nearest"],
                    help="bicubic 与 FAM Fig.5 的 Direct Upsampling 对齐")
    a = ap.parse_args()
    f = {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
         "nearest": Image.NEAREST}[a.filter]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = sorted(p for p in Path(a.src).iterdir() if p.suffix.lower() in EXT)
    if not src:
        raise SystemExit(f"!! {a.src} 里没有图")
    n = skip = 0
    sizes = set()
    for p in src:
        d = out / f"{p.stem}.png"
        if d.exists():
            skip += 1
            continue
        im = Image.open(p).convert("RGB")
        sizes.add(im.size)
        im.resize((a.size, a.size), f).save(d)
        n += 1
    print(f"上采 {n} 张（跳过 {skip}），{a.filter} → {a.size}² → {out}")
    print(f"源尺寸：{sorted(sizes) if sizes else '(全部跳过)'}")
    if len(sizes) > 1:
        print(f"⚠️ 源目录里尺寸不一致 —— 臂 A 应当全部是同一分辨率，先查清楚")


if __name__ == "__main__":
    main()
