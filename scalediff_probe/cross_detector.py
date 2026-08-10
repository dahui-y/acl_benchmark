"""用已发表的计数协议重测，而不是我们自造的尺子。

依据（两篇都有开源码、都在会议上）：

  GenEval (NeurIPS 2023 D&B)  Mask2Former(COCO) + 置信度 0.3，
      counting 任务判据是"检出数 == 要求数"。
  CountGen / Make It Count (CVPR 2025)  evaluation_script.py 用 YOLOv9e
      (ultralytics 默认阈值)，只统计类名等于目标类的框，
      is_success = (count == expected_count)。

两家是同一套协议：现成检测器 -> 数目标类的实例 -> 和要求的数字比。
**这正是我们在做的事**，所以我们不需要自造依据，直接引用。

我们与他们的两处差别，都要在论文里写明：

  ① 指标粒度。他们报二值正确率，+9 和 +1 都只是"错"。我们报 excess 的
     MAE，因为幅度是我们的主张的一部分。**两个都报**：正确率对齐这条线，
     MAE 说明幅度。
  ② 分辨率。YOLOv9e 默认 imgsz=640。4096² 直接喂进去，60 px 的重复物体
     缩完只剩 9 px —— 这是 FID 在 299² 上失明的同一个机制。所以我们唯一
     的偏离是**分块推理**。本脚本把不分块那一档也跑出来，让这句话有数字：
     **领域现成的计数协议在 4096² 上测不到这个失效。**

零改动的部分：分块/NMS/合并逻辑与 count_objects.py 共用，只换检测后端。
COCO 覆盖情况（已核对）：5 条 lone + 5 条 empty 全在 COCO 类内
（person/boat/bird），主结果与负对照全覆盖；有基数的 20 行里 15 行可测，
eye/hand/building/bridge/staircase 五条在 COCO 外 —— 这也正是我们主用
开放词表 GroundingDINO 的原因。

    pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
    python scalediff_probe/cross_detector.py
    python scalediff_probe/cross_detector.py --no-tile-only   # 只跑"原样"那一档
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import merge_split, nms, suppress_contained   # noqa: E402
from subject_phrases import CARD                                 # noqa: E402

WEIGHTS = "yolov9e.pt"          # CountGen 用的那一个（ultralytics 会自动下载）


class YoloDetector:
    """CountGen 的检测器 + 我们的分块。分块之外的一切都按 ultralytics 默认。"""

    def __init__(self, weights=WEIGHTS, device=0):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device = device
        self.names = {v: k for k, v in self.m.names.items()}

    def _one(self, img, cls_id):
        r = self.m.predict(img, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        keep = (b.cls.cpu().numpy().astype(int) == cls_id)
        return (b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep])

    def detect(self, img, subject, tile=True, min_base_px=8, base_res=1024,
               iou=0.40):
        """tile=False 就是 CountGen 脚本的原样（整图一遍，被缩到 640）。"""
        if subject not in self.names:
            return None                       # 非 COCO 类，这把尺子测不了
        cid = self.names[subject]
        W, H = img.size
        B, S = [], []

        b, s = self._one(img, cid)
        B += [list(map(float, x)) for x in b]
        S += [float(x) for x in s]

        if tile:
            t = max(64, W // 4)
            st = max(32, t // 2)
            xs = sorted(set(list(range(0, max(W - t, 0) + 1, st)) + [max(W - t, 0)]))
            ys = sorted(set(list(range(0, max(H - t, 0) + 1, st)) + [max(H - t, 0)]))
            for y in ys:
                for x in xs:
                    b, s = self._one(img.crop((x, y, x + t, y + t)), cid)
                    for bb, ss in zip(b, s):
                        B.append([float(bb[0]) + x, float(bb[1]) + y,
                                  float(bb[2]) + x, float(bb[3]) + y])
                        S.append(float(ss))

        if not B:
            return np.zeros((0, 4)), np.zeros((0,))
        B, S = np.asarray(B), np.asarray(S)
        k = nms(B, S, iou_thr=iou)
        B, S = merge_split(B[k], S[k])
        B, S = suppress_contained(B, S)
        # 与 GroundingDINO 那把尺子同一条尺度门槛，否则两把尺子不可比：
        # 下采样回基图分辨率后不足 min_base_px 的检测不构成"多出来的物体"。
        if min_base_px and len(B):
            k = img.width / base_res
            side = np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]) / k
            B, S = B[side >= min_base_px], S[side >= min_base_px]
        return B, S


def load_counts(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def hi_file(rec):
    return rec["files"][str(max(int(k) for k in rec["files"]))]


def agg(exc, keys):
    v = [exc[k] for k in keys if exc.get(k) is not None]
    if not v:
        return float("nan"), float("nan")
    mae = sum(abs(x) for x in v) / len(v)
    acc = sum(1 for x in v if x == 0) / len(v)        # CountGen 的判据
    return mae, acc


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--no-tile-only", action="store_true")
    a = ap.parse_args()

    A, B = load_counts(a.base), load_counts(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))

    det = YoloDetector(a.weights)
    modes = [False] if a.no_tile_only else [True, False]

    # gd_* = GroundingDINO 分块（counts.json 里已有的主尺子）
    gd_a = {k: (None if CARD[k[0]] is None else
                A[k]["counts"][str(max(int(x) for x in A[k]["counts"]))] - CARD[k[0]])
            for k in keys}
    gd_b = {k: (None if CARD[k[0]] is None else
                B[k]["counts"][str(max(int(x) for x in B[k]["counts"]))] - CARD[k[0]])
            for k in keys}

    res = {}                       # (mode, arm) -> {key: excess}
    skipped = set()
    for mode in modes:
        for arm, d, M in (("base", a.base, MA), ("ours", a.new, MB)):
            cur = {}
            for k in keys:
                card = CARD[k[0]] if k[0] < len(CARD) else None
                subj = A[k]["subject"]
                if card is None or subj not in det.names:
                    if subj not in det.names:
                        skipped.add(subj)
                    cur[k] = None
                    continue
                img = Image.open(Path(d) / hi_file(M[k])).convert("RGB")
                out = det.detect(img, subj, tile=mode)
                cur[k] = len(out[0]) - card
                del img
            res[(mode, arm)] = cur
            print(f"跑完 YOLOv9e  {'分块' if mode else '原样(不分块)'}  {arm}")

    print(f"\n跳过的非 COCO 主体（这把尺子测不了）: {sorted(skipped)}")

    # **共同子集**：只有两把尺子都测得了的行才能放在一张表里比。
    # 第一版把"有基数的全体"直接对比，分母不同（GDINO 20 行 × 3 seed = 60，
    # YOLO 只有 15 行 × 3 = 45，非 COCO 主体被跳过）—— 那个对比是错的。
    common = [k for k in keys
              if CARD[k[0]] is not None and A[k]["subject"] in det.names]
    print(f"共同子集：{len({k[0] for k in common})} 条 prompt × "
          f"{len({k[1] for k in common})} seed = {len(common)} 行"
          f"（两把尺子都测得了的）")

    groups = [("lone", [k for k in keys if A[k]["cat"] == "lone"]),
              ("empty", [k for k in keys if A[k]["cat"] == "empty"]),
              ("共同子集", common)]

    print(f"\n{'口径':<28}{'组':<14}{'基线 MAE':>10}{'我们 MAE':>10}"
          f"{'降幅':>8}{'基线 acc':>10}{'我们 acc':>10}")
    print("-" * 92)

    def line(name, ea, eb, gname, ks):
        ma, aa = agg(ea, ks)
        mb, ab = agg(eb, ks)
        drop = (1 - mb / ma) * 100 if ma else float("nan")
        print(f"{name:<28}{gname:<14}{ma:>10.2f}{mb:>10.2f}"
              f"{drop:>7.1f}%{aa:>10.0%}{ab:>10.0%}")

    for gname, ks in groups:
        line("GroundingDINO 分块(主)", gd_a, gd_b, gname, ks)
        for mode in modes:
            line(f"YOLOv9e {'分块' if mode else '原样(不分块)'}",
                 res[(mode, "base")], res[(mode, "ours")], gname, ks)
        print()

    # 两把尺子的一致性。**要看的是 A/B 差值，不是绝对计数** ——
    # 我们主张的是"介入让 excess 降了多少"，绝对计数一致与否是次要的。
    if True in modes:
        def rank(v):
            o = np.argsort(np.argsort(v))
            return o.astype(float)

        abs_pairs = ([(gd_a[k], res[(True, "base")][k]) for k in common]
                     + [(gd_b[k], res[(True, "ours")][k]) for k in common])
        x = np.array([p[0] for p in abs_pairs], float)
        y = np.array([p[1] for p in abs_pairs], float)
        print(f"\n一致性（共同子集）")
        print(f"  绝对计数  n={len(x)}  Pearson {np.corrcoef(x, y)[0,1]:+.3f}  "
              f"Spearman {np.corrcoef(rank(x), rank(y))[0,1]:+.3f}  "
              f"平均绝对差 {np.abs(x-y).mean():.2f} 个物体")

        dx = np.array([gd_a[k] - gd_b[k] for k in common], float)
        dy = np.array([res[(True, "base")][k] - res[(True, "ours")][k]
                       for k in common], float)
        print(f"  **A/B 差值** n={len(dx)}  Pearson {np.corrcoef(dx, dy)[0,1]:+.3f}  "
              f"Spearman {np.corrcoef(rank(dx), rank(dy))[0,1]:+.3f}  "
              f"平均绝对差 {np.abs(dx-dy).mean():.2f}")
        same = float(np.mean(np.sign(dx) == np.sign(dy)))
        print(f"  改善方向一致的行：{same:.0%}"
              f"（这是最该报的一个数：两把尺子在多少行上同意"
              f"'介入让它变好/变坏'）")

    print("""
怎么读：
  ① "YOLOv9e 分块" 与 "GroundingDINO 分块" 方向一致、降幅量级相近
     -> 结论不依赖任何单一检测器。这是对"你的 metric 自己造的"最直接的回答。
  ② "YOLOv9e 原样(不分块)" 与它分块那一档的基线 MAE 之比 = 现成协议
     测到了多少。实测 1.27 / 4.60 = 27.6%，**不是全盲，是衰减 3.6 倍**。
     写论文时按"低估"讲，别写成"看不见"—— 方向仍然是对的（-63.2%）。
  ②' empty 那一组是分块的**代价**：不分块 0.00 个假阳性，分块之后
     YOLO 0.13 / GDINO 0.27。可测性不是免费的，这个数要进敏感度表。
  ③ acc 列是 CountGen 的判据（count == expected），用于和那条线对齐；
     MAE 列是我们的，因为 +9 和 +1 在 acc 下没有区别。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
