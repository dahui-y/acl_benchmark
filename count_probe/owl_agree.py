#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OWLv2（环内检测器）与 YOLOv9e（评测器）的逐题计数一致率 + 阈值标定。

    在 **owl 环境**跑（transformers>=4.35；countgen 环境是 4.29，没有 Owlv2 类）。

    为什么这是修正器路线的地基：修正环按 OWLv2 的读数决定删几个/补几个，
    而成绩单由 YOLOv9e 打。两者每分歧一题，就有一题「环里以为修好了、
    评测说没有」（或反过来）。CountGen 的内部计数器 DBSCAN 与 YOLOv9e 的
    一致率只有 64.1%（MAE 0.89，一手数据）—— 那就是它 9% 的题「说对却错、
    永远修不到」的直接来源。OWLv2 必须明显高于这条线，路线才成立。

    顺带标定两个旋钮（SLD 的 0.15 是在它那 10 个容易类上定的，不能直接搬）：
      · 分数阈值：一次前向、多阈值计数（按分数过滤即可，不用重跑）。
      · NMS IoU：OWLv2 的 post_process 不做 NMS，raw 框里有重复；
        纯 torch 实现，省掉 torchvision 依赖。

    ⚠️ 预登记的判读（跑之前写死）：
      · 最优阈值下一致率 ≥ 80% → 地基成立，进入原型。
      · 70~80% → 勉强，修正环的漏损要在预期收益里扣掉再算头寸。
      · ≤ 64%（不如 DBSCAN）→ 这条路线在这一步就死了，不用写原型。
    一致率同时是「修正收益折算成评测收益」的换算率上限。

用法（owl 环境）：
    python count_probe/owl_agree.py --arms $SD_OUT/count/cocoount_arms \\
        --cache /openbayes/input/input0/Sim2Struct-1000/temp/weights/hf_cache
"""

import argparse
import csv
import json
from pathlib import Path


def nms(boxes, scores, iou_thr):
    """纯 torch NMS，免装 torchvision。boxes: (n,4) xyxy。返回保留索引。"""
    import torch
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest]); yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest]); yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    import torch as _t
    return _t.tensor(keep, dtype=_t.long)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="含 vanilla/ 与 yolo_results.csv 的目录")
    ap.add_argument("--cache", required=True, help="HF 缓存根（含 models--google--owlv2-*）")
    ap.add_argument("--arm", default="vanilla", help="对哪个臂数（默认 vanilla）")
    ap.add_argument("--score-thrs", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--iou-thrs", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--fp16", action="store_true", help="半精度（省显存，读数可能差半分）")
    ap.add_argument("--out", default=None, help="逐题原始框落盘 jsonl（默认 arms 下）")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    arms = Path(a.arms)
    rows = [r for r in csv.DictReader((arms / "yolo_results.csv").open())]
    # N>9 的题 yolo_countgen 为空，但 vanilla 臂照样有图有读数，全部可用
    items = [r for r in rows if (arms / a.arm / r["file"]).exists()
             and r.get("yolo_vanilla") not in ("", None, "None")]
    print(f"{len(items)} 张 {a.arm} 图（csv 共 {len(rows)} 行）")

    proc = Owlv2Processor.from_pretrained(
        "google/owlv2-base-patch16-ensemble", cache_dir=a.cache, local_files_only=True)
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble", cache_dir=a.cache, local_files_only=True)
    if a.fp16:
        model = model.half()
    model = model.to("cuda").eval()

    # 一次前向拿全部候选框（低阈值 0.05），落盘；各 (score, iou) 组合只做过滤
    out_p = Path(a.out) if a.out else arms / f"owl_boxes_{a.arm}.jsonl"
    raw = {}
    with out_p.open("w") as fh:
        for k, r in enumerate(items):
            img = Image.open(arms / a.arm / r["file"]).convert("RGB")
            q = f"a photo of a {r['coco_class']}"
            with torch.no_grad():
                inp = proc(text=[[q]], images=img, return_tensors="pt").to("cuda")
                if a.fp16:
                    inp = {k2: (v.half() if v.dtype == torch.float32 else v)
                           for k2, v in inp.items()}
                res = proc.post_process_object_detection(
                    model(**inp), threshold=0.05,
                    target_sizes=torch.tensor([img.size[::-1]]).to("cuda"))[0]
            rec = dict(stem=r["stem"], boxes=res["boxes"].float().cpu().tolist(),
                       scores=res["scores"].float().cpu().tolist())
            raw[r["stem"]] = rec
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 50 == 0:
                print(f"  …{k + 1}/{len(items)}")
    print(f"逐题原始框 → {out_p}（修正器可直接复用，不必重跑检测）")

    # ---- 扫描 (score, iou)，对齐 YOLOv9e 与要求 N ----
    print(f"\n{'score':>6}{'iou':>5} | {'与YOLO一致':>10}{'MAE':>7} | "
          f"{'与N一致':>8} | 参照：DBSCAN 与 YOLO 一致 64.1%、MAE 0.89")
    best = None
    for st in a.score_thrs:
        for it in a.iou_thrs:
            agree = mae = agn = 0
            for r in items:
                rec = raw[r["stem"]]
                b = torch.tensor(rec["boxes"]); s = torch.tensor(rec["scores"])
                m = s >= st
                keep = nms(b[m], s[m], it)
                n_owl = int(len(keep))
                yv, N = int(float(r["yolo_vanilla"])), int(r["N"])
                agree += n_owl == yv
                mae += abs(n_owl - yv)
                agn += n_owl == N
            n = len(items)
            row = (st, it, 100 * agree / n, mae / n, 100 * agn / n)
            print(f"{st:>6.2f}{it:>5.1f} | {row[2]:>9.1f}%{row[3]:>7.2f} | "
                  f"{row[4]:>7.1f}%")
            if best is None or row[2] > best[2]:
                best = row
    print(f"\n最优（按与 YOLO 一致率）：score={best[0]:.2f} iou={best[1]:.1f} → "
          f"一致 {best[2]:.1f}%、MAE {best[3]:.2f}")
    print("预登记判读：≥80% 地基成立；70~80% 勉强（漏损要计入头寸）；"
          "≤64%（不如 DBSCAN）路线就此死掉。")
    print("注意：一致率是「修正收益 → 评测收益」的换算率上限 —— 环里每修对一题，"
          "评测器只按这个概率承认。")


if __name__ == "__main__":
    main()
