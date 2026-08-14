"""看门控图：把 mask.pt 叠回基图上，一行四格。

一次运行会为每个 idx 产出四个文件里的三个 + 一张掩码图：
    {idx}_1024.png   基础阶段（1024²）—— **必须与基线逐字节相同**，
                     它是"真值"：4096 里多出来的东西才叫重复
    {idx}_2048.png   中间放大级，只是过程，不参与测量
    {idx}_4096.png   最终输出，测量对象
    {idx}_mask.pt    门控图（64×64，0~1）：1 = 该位置用完整 prompt，
                     0 = 该位置用去主体 prompt（= 门开着，在抑制复制）

四格 contact sheet 的读法：
    [基图 1024] [门控图热力] [门叠在基图上] [4096 缩回 1024]
    热力：红 = 主体（保护）／蓝 = 背景（抑制）／灰白 = 过渡带（半开，坏）

要看的三件事：
    1. 红区是否**盖住主体本身**（不是盖住半张画面，也不是盖偏）；
    2. 灰白是否占满 —— 灰白多 = 门半开 = 什么都没做（v1 的病）；
    3. 第四格相对第一格**多出来的实例**是否落在蓝区里 —— 落在蓝区
       说明门位置对了但强度不够，落在红区说明门把副本当主体保护了。

用法：
    python scalediff_probe/mask_look.py $SD_OUT/parti_v12               # 全部
    python scalediff_probe/mask_look.py $SD_OUT/parti_v12 --idx 263 474
    python scalediff_probe/mask_look.py $SD_OUT/parti_v12 --worst 6     # 只看过渡带最宽的
    python scalediff_probe/mask_look.py $SD_OUT/parti_v12 --band-only   # 不出图，只统计 R1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

CELL = 384


def colorize(m):
    """0 -> 蓝（背景/抑制），0.5 -> 灰白（过渡带），1 -> 红（主体/保护）。"""
    m = np.clip(m, 0, 1)[..., None]
    lo = np.array([60, 110, 230], np.float32)      # 蓝
    mid = np.array([235, 235, 235], np.float32)    # 灰白
    hi = np.array([225, 60, 55], np.float32)       # 红
    t = np.abs(m - 0.5) * 2.0                      # 离 0.5 多远
    end = np.where(m >= 0.5, hi, lo)
    return (mid * (1 - t) + end * t).astype(np.uint8)


def load_mask(p):
    m = torch.load(p, map_location="cpu")
    if hasattr(m, "float"):
        m = m.float()
    m = m.squeeze()
    if m.ndim != 2:
        raise ValueError(f"{p.name} 形状 {tuple(m.shape)}，不是二维门控图")
    return m.numpy()


def up(m, n=CELL):
    """最近邻放大，保住 64×64 的格子感 —— 别让插值把边界糊掉，"
    我们正是要看边界锐不锐。"""
    k = max(1, n // m.shape[0])
    return np.kron(m, np.ones((k, k), m.dtype))[:n, :n]


def band(m):
    """(主体 >0.5, 背景 <0.3, 过渡带 0.3~0.7)"""
    return (float((m > 0.5).mean()),
            float((m < 0.3).mean()),
            float(((m >= 0.3) & (m <= 0.7)).mean()))


def cell(img, label):
    c = Image.new("RGB", (CELL, CELL + 22), "white")
    c.paste(img.resize((CELL, CELL), Image.LANCZOS), (0, 0))
    ImageDraw.Draw(c).text((4, CELL + 5), label, fill="black")
    return c


def row(d, idx, rec):
    f = {n: d / f"{idx:05d}_{n}.png" for n in ("1024", "2048", "4096")}
    mp = d / f"{idx:05d}_mask.pt"
    if not f["1024"].exists() or not f["4096"].exists():
        return None, None
    base = Image.open(f["1024"]).convert("RGB")
    hi = Image.open(f["4096"]).convert("RGB")

    if mp.exists():
        m = load_mask(mp)
        b = band(m)
        mu = up(m)
        heat = Image.fromarray(colorize(mu))
        # 叠加：门控图当色调，基图当明度
        bl = np.asarray(base.resize((CELL, CELL), Image.LANCZOS), np.float32)
        ov = Image.fromarray(
            np.clip(bl * 0.55 + np.asarray(heat, np.float32) * 0.45, 0, 255
                    ).astype(np.uint8))
        htxt = (f"gate  subj {b[0]:.0%}  bg {b[1]:.0%}  band {b[2]:.0%}"
                + ("   <-- HALF-OPEN"
                   if b[2] > 0.5 or b[1] < 0.1 else "   OPEN"))
    else:
        b = None
        heat = Image.new("RGB", (CELL, CELL), "white")
        ov = base
        htxt = "no mask.pt"

    head = (rec or {}).get("head", "?")
    # 图内标签一律 ASCII：PIL 自带位图字体没有 CJK，中文会变成豆腐块
    cells = [cell(base, f"[{idx}] 1024 base = ground truth   head={head}"),
             cell(heat, htxt),
             cell(ov, "gate on base:  red=protect  blue=suppress  grey=half"),
             cell(hi, "4096 down to 1024  = what we measure")]
    sheet = Image.new("RGB", (CELL * 4, CELL + 22), "white")
    for i, c in enumerate(cells):
        sheet.paste(c, (i * CELL, 0))
    return sheet, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--idx", type=int, nargs="*")
    ap.add_argument("--worst", type=int, default=0,
                    help="只出过渡带最宽的 N 行（门最没打开的）")
    ap.add_argument("--band-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = Path(a.run)
    mf = d / "manifest.jsonl"
    recs = {}
    if mf.exists():
        for ln in mf.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if "idx" in r:
                    recs[int(r["idx"])] = r
    ids = sorted(recs) or sorted(
        int(p.name[:5]) for p in d.glob("*_4096.png"))
    if a.idx:
        ids = [i for i in ids if i in set(a.idx)]

    # 先全量算 band（便宜，只读 mask.pt），R1 判据看这张统计
    stats = {}
    for i in ids:
        mp = d / f"{i:05d}_mask.pt"
        if mp.exists():
            stats[i] = band(load_mask(mp))

    if stats:
        bs = np.array([v[2] for v in stats.values()])
        lo = np.array([v[1] for v in stats.values()])
        cv = np.array([v[0] for v in stats.values()])
        open_ = int(((bs < 0.5) & (lo > 0.1)).sum())
        print(f"\n=== 门控图统计（n={len(stats)}） ===")
        print(f"  过渡带   均值 {bs.mean():.0%}   中位 {np.median(bs):.0%}"
              f"   范围 {bs.min():.0%}~{bs.max():.0%}")
        print(f"  背景<0.3 均值 {lo.mean():.0%}   中位 {np.median(lo):.0%}"
              f"   范围 {lo.min():.0%}~{lo.max():.0%}")
        print(f"  主体覆盖 均值 {cv.mean():.0%}   范围 {cv.min():.0%}~{cv.max():.0%}")
        print(f"\n  R1（过渡带<50% 且 背景>10%）："
              f"**{open_}/{len(stats)} 行的门是真开着的**"
              f"（v1 全层平均时是 0/31）")
        half = sorted(stats, key=lambda i: -stats[i][2])
        print("  仍半开的行（过渡带从宽到窄）：",
              ", ".join(f"{i}({stats[i][2]:.0%})"
                        for i in half if stats[i][2] > 0.5) or "无")

    if a.band_only:
        return

    if a.worst:
        ids = sorted(stats, key=lambda i: -stats[i][2])[:a.worst]

    rows = []
    for i in ids:
        s, _ = row(d, i, recs.get(i))
        if s is not None:
            rows.append(s)
    if not rows:
        print("没有可出图的行")
        return
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)),
                      "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    op = Path(a.out) if a.out else d / "mask_look.jpg"
    sheet.save(op, quality=88)
    print(f"\n出图 {len(rows)} 行 -> {op}")


if __name__ == "__main__":
    main()
