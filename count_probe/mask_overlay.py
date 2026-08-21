#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把重去噪遮罩叠回原图，一眼看清「这次删除到底盖住了什么」。零 GPU。

    动机：表六给出 66 个待删框里 54 个被保留框挖过、中位挖掉 53%、最大 100%。
    数字已经足够判案，但要让人（包括审稿人）相信「轴对齐框做不了实例删除」，
    最有力的还是把遮罩画在图上：红色半透明 = 会被重去噪的区域，红框 = 待删
    实例，绿框 = 保留实例。看一眼就知道被删的那只杯子只有上半截进了红区。

    每题一行三格：
        原图 + 框    原图 + 遮罩叠加      修正后的输出
    行标题写该轮每个待删框被挖掉的比例（与 del_stats 表六同一算法）。

    需要跑批时带过 --dump-mask（遮罩落盘）。框来自 counter_log 的
    del_boxes / keep_boxes，是纯诊断字段，不影响生成。

用法：
    python count_probe/mask_overlay.py --run $SD_OUT/count/tune2_del_s10ctx_m \\
        --stem cup_num=3_seed=949361 --out $SD_OUT/count/look2/mask
    # 不给 --stem，给 --worst 3 = 自动挑「被挖比例中位数最高」的三题
"""

import argparse
import json
from pathlib import Path

from del_stats import _carved


def _log(run):
    out = {}
    for l in (Path(run) / "counter_log.jsonl").open():
        r = json.loads(l)
        out[r["id"]] = r
    return out


def _carves(rec):
    """→ [[每个待删框被挖掉的比例], ...] 逐轮。"""
    c = rec.get("corrector", {})
    return [[_carved(b, keep) for b in dele]
            for dele, keep in zip(c.get("del_boxes", []),
                                  c.get("keep_boxes", []))]


def render(run, stem, rec, out_dir, size=384):
    from PIL import Image, ImageDraw
    run = Path(run)
    van = Image.open(run / f"{stem}_vanilla.png").convert("RGB")
    res = Image.open(run / f"{stem}.png").convert("RGB")
    c = rec["corrector"]
    carves = _carves(rec)
    rounds = len(c.get("del_boxes", []))
    if not rounds:
        print(f"  {stem}：没有编辑轮次，跳过")
        return None

    pad, head = 8, 34
    W = size * 3 + pad * 4
    H = head + (size + head) * rounds + pad
    canvas = Image.new("RGB", (W, H), "white")
    d0 = ImageDraw.Draw(canvas)
    d0.text((pad, 8), f"{stem}   N={rec['requiered_object_num']}   "
                      f"环内轨迹={c['n_trail']}   想删={c['want']}", fill="black")

    for r in range(rounds):
        mp = run / f"{stem}_mask{r}.png"
        if not mp.exists():
            print(f"  {stem}：缺 {mp.name}（跑批时要带 --dump-mask）")
            return None
        mask = Image.open(mp).convert("L")
        base = van if r == 0 else res     # 第二轮的输入拿不到，用输出近似并标注

        boxed = base.copy()
        db = ImageDraw.Draw(boxed)
        for b in c["keep_boxes"][r]:
            db.rectangle(b, outline=(0, 220, 0), width=4)
        for b in c["del_boxes"][r]:
            db.rectangle(b, outline=(255, 0, 0), width=4)

        red = Image.new("RGB", base.size, (255, 0, 0))
        over = Image.composite(
            Image.blend(base, red, 0.55), base, mask)

        y = head + (size + head) * r
        src = "原图" if r == 0 else "上一轮输出(以最终输出近似)"
        for i, (im, tag) in enumerate(
                [(boxed, f"第{r+1}轮 {src} 红=待删 绿=保留"),
                 (over, f"红区=重去噪范围 占图 {c['mask_frac'][r]:.1%}"),
                 (res, "最终输出")]):
            x = pad + (size + pad) * i
            canvas.paste(im.resize((size, size), Image.LANCZOS), (x, y + head))
            d0.text((x, y + 8), tag, fill="black")
        cv = "  ".join(f"{v:.0%}" for v in carves[r])
        d0.text((pad + (size + pad) * 1, y + 20),
                f"各待删框被保留框挖掉：{cv}", fill=(180, 0, 0))

    op = Path(out_dir) / f"mask_{stem}.png"
    op.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(op)
    print(f"  → {op}")
    return op


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="带 --dump-mask 跑过的跑批目录")
    ap.add_argument("--stem", nargs="*", default=None)
    ap.add_argument("--worst", type=int, default=0,
                    help="自动挑「被挖比例中位数最高」的前 N 题")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()

    L = _log(a.run)
    stems = list(a.stem or [])
    if a.worst:
        cand = []
        for s, rec in L.items():
            fl = [v for rd in _carves(rec) for v in rd]
            if fl:
                fl.sort()
                cand.append((fl[len(fl) // 2], max(fl), s))
        cand.sort(reverse=True)
        print(f"被挖比例最高的 {a.worst} 题（中位 / 最大）：")
        for m, mx, s in cand[:a.worst]:
            print(f"  {s:40s} 中位 {m:.0%}  最大 {mx:.0%}")
            stems.append(s)
    for s in stems:
        if s not in L:
            print(f"  !! {s} 不在 {a.run} 的 counter_log 里")
            continue
        render(a.run, s, L[s], a.out, a.size)


if __name__ == "__main__":
    main()
