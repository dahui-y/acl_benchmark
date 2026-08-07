"""数"高分辨率结果比基图多出了几个 prompt 主体"。

为什么需要这个，而不是继续用像素差：
    seed 77 那一对给了定量答案 —— 2048² 平均绝对差 7.68 / 幻影 0 个；
    4096² 平均绝对差 9.08 / 幻影至少 3 个。**标量只涨了 18%，却要区分
    "干净"和"多了三个人"。** 一个连这个都分不开的量不可能当指标。
    所以判据必须是物体级的：数个数，不看纹理漂移。

这【不是】用户研究。它是一段离线程序，同输入同输出。和 FID 依赖
InceptionV3、CLIP score 依赖 CLIP 是同一类东西 —— 区别只在于
InceptionV3 要把 4096² 压到 299²（于是看不见多一个人），而这里在原分辨率
上分块检测（于是看得见）。

分块是必须的，不是优化：
    GroundingDINO 内部把输入缩到 ~800 px。整张 4096² 丢进去，一个 60 px 的
    幻影缩完只剩 12 px，必然漏检。切成 1024² 的块（步长 512，保证任何小于
    1024 px 的物体至少被某一块完整包含）再检测，同一个幻影有 ~47 px，能测到。

    python scalediff_probe/count_objects.py --check      # 阳性对照：seed77 那张
    python scalediff_probe/count_objects.py              # 跑整个 batch
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_ID = "IDEA-Research/grounding-dino-base"


def nms(boxes, scores, iou_thr=0.5):
    """boxes: (N,4) xyxy。分块检测同一个物体会被数两次，必须合并。"""
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class Detector:
    def __init__(self, device="cuda", box_thr=0.30, text_thr=0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
        self.model.eval()
        self.device = device
        self.box_thr, self.text_thr = box_thr, text_thr

    @torch.no_grad()
    def _one(self, img, text):
        # GroundingDINO 的文本要小写、以句点结尾
        inputs = self.proc(images=img, text=text.lower().strip().rstrip(".") + ".",
                           return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=self.box_thr,
                text_threshold=self.text_thr, target_sizes=[img.size[::-1]])[0]
        except TypeError:                       # 老一点的 transformers 用别的参数名
            res = self.proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], box_threshold=self.box_thr,
                text_threshold=self.text_thr, target_sizes=[img.size[::-1]])[0]
        return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()

    def detect(self, img, text, tile=1024, stride=512, iou=0.40, max_aspect=2.0):
        """整图 + 分块两遍，合并后 NMS。

        为什么必须两遍。阳性对照里主角在 4096² 上高 582 px，而 tile=1024 /
        stride=512 意味着任何高于 stride 的物体都会被某条 tile 边界切开：
        实测拿到的是 (0.404,0.471) 214x238 和 (0.405,0.542) 220x345 —— 上半身
        和下半身两个框，y 几乎不重叠，IoU≈0，NMS 合不掉。**每个大物体都被数
        了两次。** 整图那一遍能给出完整的框（主角缩放后仍有 ~114 px，检得到），
        NMS 就能把两个半身框吃掉。分块那一遍负责小幻影（60 px 的人整图检测时
        只剩 12 px，必漏）。两遍各管一头。

        max_aspect 过滤横条：站立的人不可能宽远大于高。实测误检的
        (0.197,0.113) 218x63（宽高比 3.5）和 (0.533,0.075) 72x26（2.8）都在
        天空里，是云。
        """
        W, H = img.size
        B, S = [], []

        # 第一遍：整图。大物体靠它。
        b, s = self._one(img, text)
        B += [list(map(float, x)) for x in b]
        S += [float(x) for x in s]

        # 第二遍：原分辨率分块。小物体靠它。
        if W > tile or H > tile:
            xs = sorted(set(list(range(0, max(W - tile, 0) + 1, stride)) + [max(W - tile, 0)]))
            ys = sorted(set(list(range(0, max(H - tile, 0) + 1, stride)) + [max(H - tile, 0)]))
            for y in ys:
                for x in xs:
                    b, s = self._one(img.crop((x, y, x + tile, y + tile)), text)
                    for bb, ss in zip(b, s):
                        B.append([float(bb[0]) + x, float(bb[1]) + y,
                                  float(bb[2]) + x, float(bb[3]) + y])
                        S.append(float(ss))

        if not B:
            return np.zeros((0, 4)), np.zeros((0,))
        B, S = np.asarray(B), np.asarray(S)

        if max_aspect:
            w, h = B[:, 2] - B[:, 0], np.maximum(B[:, 3] - B[:, 1], 1e-6)
            keep = (w / h) <= max_aspect
            self.dropped_aspect = int((~keep).sum())
            B, S = B[keep], S[keep]
            if len(B) == 0:
                return np.zeros((0, 4)), np.zeros((0,))

        k = nms(B, S, iou_thr=iou)
        return B[k], S[k]


def annotate(img, boxes, scores, view=1400, color=(255, 0, 0)):
    sc = view / img.width
    im = img.resize((view, int(img.height * sc)), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    for b, s in zip(boxes, scores):
        d.rectangle([b[0] * sc, b[1] * sc, b[2] * sc, b[3] * sc], outline=color, width=2)
        d.text((b[0] * sc + 3, b[1] * sc - 12), f"{s:.2f}", fill=color)
    return im


def main():
    ap = argparse.ArgumentParser()
    out_default = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap.add_argument("--batch", default=str(out_default / "batch"))
    ap.add_argument("--check", action="store_true",
                    help="阳性对照：只跑 run_one 的 seed77，看能不能找到那 3 个已确认的人")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--max-aspect", type=float, default=2.0,
                    help="宽/高 超过这个值的框丢掉（站立的人不会是横条）；0 关闭")
    a = ap.parse_args()

    det = Detector(box_thr=a.box_thr, text_thr=a.text_thr)

    if a.check:
        d = out_default / "run_one"
        base = Image.open(d / "s77_stage2_1024.png").convert("RGB")
        hi = Image.open(d / "s77_stage2_4096.png").convert("RGB")
        crops = d / "det_crops"
        crops.mkdir(exist_ok=True)
        for name, im in (("base1024", base), ("hi4096", hi)):
            det.dropped_aspect = 0
            b, s = det.detect(im, "person", a.tile, a.stride, a.iou, a.max_aspect)
            print(f"\n{name}  person x{len(b)}   (按长宽比丢掉 {det.dropped_aspect} 个)")
            for j, (bb, ss) in enumerate(sorted(zip(b, s), key=lambda z: -z[1])):
                cx, cy = (bb[0] + bb[2]) / 2 / im.width, (bb[1] + bb[3]) / 2 / im.height
                w, h = bb[2] - bb[0], bb[3] - bb[1]
                print(f"    #{j:<2} ({cx:.3f}, {cy:.3f})  {w:.0f}x{h:.0f}px  "
                      f"w/h={w/max(h,1):.2f}  {ss:.2f}")
                # 每个框存一张原分辨率裁块 —— 每一个都要能被肉眼核对，
                # 不能靠我在缩略图上猜
                pad = max(w, h)
                box = (max(0, bb[0] - pad), max(0, bb[1] - pad),
                       min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
                im.crop(tuple(map(int, box))).save(
                    crops / f"{name}_{j:02d}_{cx:.3f}_{cy:.3f}_{ss:.2f}.png")
            annotate(im, b, s).save(d / f"D_det_{im.width}.png")

        # 阈值敏感度：三个已确认的幻影分别是 0.91 / 0.89 / 0.83，
        # 所以看提高阈值会不会先杀掉垃圾、后杀掉真幻影
        print("\n阈值敏感度（hi 4096）：")
        det.dropped_aspect = 0
        b_all, s_all = det.detect(hi, "person", a.tile, a.stride, a.iou, a.max_aspect)
        for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            print(f"    thr {t:.2f} -> {int((s_all >= t).sum())} 个")
        print("\n已确认的三个幻影：(0.76,0.40) (0.61,0.93) (0.97,0.62)")
        print("要求：这三个都在，基图上只有主角一个，且每个框的裁块肉眼看得过去。")
        print(f"标注图 {d}/D_det_*.png    逐框裁块 {crops}/")
        return

    batch = Path(a.batch)
    manifest = batch / "manifest.jsonl"
    if not manifest.exists():
        sys.exit(f"没有 {manifest}，先跑 batch_run.py")

    rows = []
    print(f"{'tag':<22}{'cat':<10}{'subject':<10}{'base':>6}{'2048':>6}{'4096':>6}{'delta':>7}")
    print("-" * 70)
    for line in manifest.open():
        r = json.loads(line)
        tag = f"{r['idx']:02d}_{r['cat']}_s{r['seed']}"
        counts = {}
        for res, fn in sorted(r["files"].items(), key=lambda kv: int(kv[0])):
            im = Image.open(batch / fn).convert("RGB")
            b, s = det.detect(im, r["subject"], a.tile, a.stride, a.iou, a.max_aspect)
            counts[int(res)] = len(b)
            if int(res) == max(int(k) for k in r["files"]):
                annotate(im, b, s).save(batch / f"{tag}_det_{res}.png")
        base_n = counts.get(1024, 0)
        hi_n = counts.get(max(counts), 0)
        rows.append({**{k: r[k] for k in ("idx", "cat", "subject", "seed", "prompt")},
                     "counts": counts, "delta": hi_n - base_n})
        print(f"{tag:<22}{r['cat']:<10}{r['subject']:<10}"
              f"{counts.get(1024, 0):>6}{counts.get(2048, 0):>6}{counts.get(4096, 0):>6}"
              f"{hi_n - base_n:>+7}")

    (batch / "counts.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n按类别汇总（delta = 4096 的个数 - 基图的个数）：")
    for cat in sorted({r["cat"] for r in rows}):
        d = [r["delta"] for r in rows if r["cat"] == cat]
        pos = sum(1 for x in d if x > 0)
        print(f"  {cat:<10} n={len(d):<3} 平均 delta {np.mean(d):+.2f}   "
              f"有新增的 {pos}/{len(d)}")
    print(f"\n明细：{batch/'counts.json'}")


if __name__ == "__main__":
    main()
