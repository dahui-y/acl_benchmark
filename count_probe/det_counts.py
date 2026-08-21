#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修正环 Phase B：YOLOv8x 逐图检测落盘 + 环视角的头寸表。countgen 环境跑。

    分批架构（环内不交错，两个环境各管各的）：
      Phase A  生成（countgen 环境，已有）
      Phase B  本脚本：检测 → {arm}_v8x.jsonl（逐图框/分数/计数）
      Phase C  按 jsonl 做删/补 + 遮罩重去噪（countgen 环境，待写）
      Phase D  重跑本脚本于修正后的臂 → 决定是否再来一轮

    为什么是 v8x：环内计数器的立项判据是**条件换算率**
    P(评测器=N | 环内计数器=N)，在 167 张 vanilla 上实测：
        v8x 85.7% (n=70)   OWLv2 80.4% (n=56)   双共识 95.6% (n=45)
    ⚠️ 前一版门槛用了**边缘**一致率（v8x 66.5%）并据此误判路线死亡 ——
    修正环只在"它宣布够数"的那一刻需要评测器点头，边缘量把
    "数不清的烂图上两个检测器错得不一样"也算了进去，是错的统计量。
    两次判定的阈值都是见数之前预登记的（80%）。

    循环性红线（预登记）：环内绝不用 YOLOv9e（评测器）。v8x 与它同家族
    不同模型，论文必须披露；最终主对比另加 OWLv2 第三方读数（两臂同测，
    差值有意义）与作者盲数（计数是客观量，盲法+披露即成立）。

    输出的头寸表以环的视角（v8x 计数）重算 P0：修正器决定删几个/补几个
    看的是它，不是评测器。

用法：
    python count_probe/det_counts.py --arms $SD_OUT/count/cocoount_arms
    python count_probe/det_counts.py --arms $SD_OUT/count/xxx_arms --arm countgen
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--arm", default="vanilla")
    ap.add_argument("--weights", default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=None,
                    help="不给 = ultralytics 默认 0.25，与 yolo_eval 语义一致。"
                         "不要为了贴近评测器去调它 —— 那是往评测器上拟合")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO

    arms = Path(a.arms)
    rows = [r for r in csv.DictReader((arms / "yolo_results.csv").open())]
    items = [r for r in rows if (arms / a.arm / r["file"]).exists()]
    print(f"{len(items)} 张 {a.arm} 图")

    model = YOLO(a.weights)
    # 类名表核对：与 yolo_eval 同一张 COCO80。错名不会报错、只会永远数出 0，
    # 所以必须在这里断言，不能靠下游发现。
    from make_arms import COCO80
    bad = [i for i, n in enumerate(COCO80) if model.names.get(i) != n]
    assert not bad, f"!! {a.weights} 类名表与 COCO-80 不符，前几处：{bad[:5]}"

    out_p = Path(a.out) if a.out else arms / f"v8x_boxes_{a.arm}.jsonl"
    recs = []
    with out_p.open("w") as fh:
        for i, r in enumerate(items):
            kw = {"verbose": False}
            if a.conf is not None:
                kw["conf"] = a.conf
            res = model(str(arms / a.arm / r["file"]), **kw)[0]
            sel = [(b, s) for b, s, c in zip(res.boxes.xyxy.tolist(),
                                             res.boxes.conf.tolist(),
                                             res.boxes.cls.tolist())
                   if res.names[int(c)] == r["coco_class"]]
            rec = dict(stem=r["stem"], n=len(sel),
                       boxes=[b for b, _ in sel], scores=[s for _, s in sel])
            recs.append((r, rec))
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  …{i + 1}/{len(items)}")
    print(f"逐图框 → {out_p}")

    # ---- 环视角的头寸表（只对 N≤9 的计分题）----
    main_rows = [(r, rec) for r, rec in recs
                 if r["skipped_by_official"] not in ("True", "true", "1")]
    c = Counter()
    for r, rec in main_rows:
        d = rec["n"] - int(r["N"])
        c["=0" if d == 0 else "+1" if d == 1 else "-1" if d == -1 else
          "+2" if d == 2 else "-2" if d == -2 else "≥+3" if d >= 3 else "≤-3"] += 1
    n = len(main_rows)
    print(f"\n环视角（v8x 计数）与要求 N 的差，N≤9 共 {n} 题：")
    print("  " + "  ".join(f"{k}:{c[k]}({100*c[k]/n:.0f}%)" for k in
                           ("=0", "+1", "-1", "+2", "-2", "≥+3", "≤-3") if c[k]))
    over = sum(v for k, v in c.items() if k.startswith(("+", "≥")))
    under = sum(v for k, v in c.items() if k.startswith(("-", "≤")))
    print(f"  要删的 {over} 题 / 要补的 {under} 题 / 不动的 {c['=0']} 题")

    # 与评测器的两个口径（vanilla 臂才有意义；其它臂 yolo 列是方法输出）
    if a.arm == "vanilla":
        ok9 = [(r, rec) for r, rec in main_rows
               if r.get("yolo_vanilla") not in ("", "None", None)]
        marg = sum(1 for r, rec in ok9
                   if rec["n"] == int(float(r["yolo_vanilla"])))
        say_n = [(r, rec) for r, rec in ok9 if rec["n"] == int(r["N"])]
        hit = sum(1 for r, rec in say_n
                  if int(float(r["yolo_vanilla"])) == int(r["N"]))
        print(f"\n与 YOLOv9e：边缘一致 {100*marg/len(ok9):.1f}%（仅记录用）；"
              f"条件换算率 P(v9e=N|v8x=N) = {100*hit/max(len(say_n),1):.1f}% "
              f"(n={len(say_n)})   ← 立项判据，门槛 80%")


if __name__ == "__main__":
    main()
