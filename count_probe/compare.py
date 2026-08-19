#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把同一道题在各个跑批目录里的结果拼成一张对照图。零 GPU。

    为什么需要它：「哪张是原版、哪张是我们的」不看文件名，看**目录**。
    每个目录里都有一份同名的 `{stem}_vanilla.png`（原版 SDXL，各目录应当
    逐字节相同）和 `{stem}.png`（该目录配置的输出）。散着看很容易搞混。

    顺带做两件事：
      · 核对各目录的 `_vanilla.png` 是否逐字节相同 —— 我们的改动只作用于
        counting pass，vanilla 臂本该完全不受影响。不同就是泄漏，要当 bug 查。
      · 从各目录的 counter_log.jsonl 读出该题的配置与计数器读数，标在图上。

用法：
    python count_probe/compare.py --stem airplane_num=7_seed=149069 \\
        --runs $SD_OUT/count/tune_orig $SD_OUT/count/tune_inst $SD_OUT/count/tune_inst2 \\
        --out $SD_OUT/count/look
    # 不给 --stem 就列出各目录都有哪些题
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _log(run, stem):
    p = Path(run) / "counter_log.jsonl"
    if not p.exists():
        return {}
    for line in p.open():
        if line.strip():
            r = json.loads(line)
            if r.get("id") == stem:
                return r
    return {}


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:8] if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default=None, help="题目 id，如 dog_num=7_seed=403504")
    ap.add_argument("--runs", nargs="+", required=True, help="若干跑批目录")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=384, help="每格边长")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]

    if not a.stem:
        for r in runs:
            n = len(list(r.glob("*_vanilla.png")))
            cfg = {x.get("loss") for x in
                   (json.loads(l) for l in (r / "counter_log.jsonl").open()
                    if l.strip())} if (r / "counter_log.jsonl").exists() else "?"
            print(f"{r.name:<16} {n:>4} 题   loss={cfg}")
        first = sorted(runs[0].glob("*_vanilla.png"))[:15]
        print("\n前几个题目 id：")
        for f in first:
            print("  " + f.name[:-len("_vanilla.png")])
        return

    # ---- vanilla 一致性检查 ----
    print("各目录的 _vanilla.png（我们的改动不该碰它，md5 必须相同）：")
    md5s = {}
    for r in runs:
        h = _md5(r / f"{a.stem}_vanilla.png")
        md5s[r.name] = h
        print(f"  {r.name:<16} {h}")
    live = [h for h in md5s.values() if h]
    if len(set(live)) > 1:
        print("  ⚠️ **不一致** —— 有东西泄漏到了 vanilla 那一趟，当 bug 查")
    elif live:
        print("  ✓ 一致")

    # ---- 拼图 ----
    panels = []
    v = runs[0] / f"{a.stem}_vanilla.png"
    if v.exists():
        panels.append(("SDXL 原版（vanilla）\n无任何干预", v, _log(runs[0], a.stem)))
    for r in runs:
        f = r / f"{a.stem}.png"
        rec = _log(r, a.stem)
        if f.exists():
            tag = rec.get("loss", "?")
            fg = rec.get("fg_after")
            lab = f"{r.name}\nloss={tag}" + (f"  fg={fg:.0%}" if fg else "")
            panels.append((lab, f, rec))
        else:
            panels.append((f"{r.name}\n（无输出）", None, rec))
    m = runs[0] / f"{a.stem}_masks.png"
    if m.exists():
        panels.append(("布局 mask\n（vanilla / 补齐 / 后处理）", m, {}))

    S, PAD, TXT = a.size, 8, 46
    W = len(panels) * (S + PAD) + PAD
    canvas = Image.new("RGB", (W, S + TXT + 2 * PAD), "white")
    d = ImageDraw.Draw(canvas)
    for i, (lab, f, rec) in enumerate(panels):
        x = PAD + i * (S + PAD)
        if f:
            im = Image.open(f).convert("RGB")
            im.thumbnail((S, S))
            canvas.paste(im, (x + (S - im.width) // 2, PAD + TXT))
        else:
            d.rectangle([x, PAD + TXT, x + S, PAD + TXT + S], outline="gray")
        for j, line in enumerate(lab.split("\n")):
            d.text((x, PAD + j * 14), line[:52], fill="black")
        n = rec.get("n_dbscan")
        if n is not None:
            d.text((x, PAD + 28), f"DBSCAN 数出 {n} / 要求 "
                                  f"{rec.get('requiered_object_num', '?')}", fill="black")

    out = Path(a.out or ".") / f"cmp_{a.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"\n→ {out}")
    print("  从左到右：原版 SDXL → 各目录的输出 → 布局 mask")


if __name__ == "__main__":
    main()
