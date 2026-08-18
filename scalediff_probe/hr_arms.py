#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 `{id}_{res}.png` 一锅端的目录拆成按分辨率分开的臂，并核对配对。

    为什么必须用同一次运行的三档，而不是 laion_base 里那 560 张 1024：
    **FID 是集合级的。** 两个臂如果 prompt 集不同，差值里就混进了「选题
    不同」这一项，而那一项无法与「分辨率不同」分开。ScaleDiff 一次
    pipe() 调用（upsample_stage=2）同时返回 1024/2048/4096，所以
    laion_hi 里的三档天然是**同 prompt、同 seed、同一条轨迹**的三个切片。

    脚本只建软链，不复制像素。同时报告：
      · 每档张数与实际尺寸（尺寸不符会当场喊出来，不是静默通过）
      · **三档的 id 集是否完全相同** —— 不同就只保留交集，并说明扔了多少

用法：
    python scalediff_probe/hr_arms.py --src $SD_OUT/laion_hi --out $SD_OUT/hr_arms
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default=r"^(?P<id>.+)_(?P<res>\d+)$",
                    help="文件名（去后缀）的解析式，默认 {id}_{res}")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    rx = re.compile(a.pattern)

    by_res = defaultdict(dict)          # res -> {id: path}
    unparsed = []
    for p in sorted(src.glob("*.png")):
        m = rx.match(p.stem)
        if not m:
            unparsed.append(p.name)
            continue
        by_res[int(m.group("res"))][m.group("id")] = p
    if unparsed:
        print(f"⚠️ {len(unparsed)} 个文件名解析不了，例：{unparsed[:3]}")
    if not by_res:
        raise SystemExit(f"!! {src} 里没有匹配 {a.pattern} 的文件")

    print(f"{src.name}: {sum(len(v) for v in by_res.values())} 张")
    for res in sorted(by_res):
        ids = by_res[res]
        sizes = {Image.open(p).size for p in list(ids.values())[:20]}
        bad = [s for s in sizes if s != (res, res)]
        print(f"  {res:>5}²: {len(ids):>4} 张  实测尺寸 {sorted(sizes)}"
              + (f"  ⚠️ 与文件名不符！" if bad else ""))

    # ★ 交集：三档 id 必须一一对应，否则臂之间的差里混进了「样本不同」
    common = set.intersection(*(set(v) for v in by_res.values()))
    print(f"\n三档共有 id：{len(common)}")
    for res in sorted(by_res):
        miss = len(by_res[res]) - len(common)
        if miss:
            print(f"  {res}² 多出 {miss} 个不在交集里的 id → 丢弃")
    if not common:
        raise SystemExit("!! 交集为空，检查命名")

    for res in sorted(by_res):
        d = out / f"res{res}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for i in sorted(common):
            (d / f"{i}.png").symlink_to(by_res[res][i].resolve())
        print(f"  → {d}  ({len(common)} 张)")

    lo = min(by_res)
    print(f"\n下一步（臂 B，零 GPU）：")
    print(f"  python scalediff_probe/bicubic_arm.py "
          f"--src {out}/res{lo} --out {out}/bicubic{max(by_res)}")
    print(f"\n然后三臂打分：")
    print(f"  python scalediff_probe/std_table.py --arms \\")
    print(f"    native{lo}={out}/res{lo} \\")
    print(f"    bicubic{max(by_res)}={out}/bicubic{max(by_res)} \\")
    print(f"    scalediff{max(by_res)}={out}/res{max(by_res)}")


if __name__ == "__main__":
    main()
