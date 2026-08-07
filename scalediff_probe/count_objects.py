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

    def detect(self, img, text, tile=1024, stride=512):
        """原分辨率分块检测；小于 tile 的图直接单次前向。"""
        W, H = img.size
        if W <= tile and H <= tile:
            b, s = self._one(img, text)
            k = nms(b, s)
            return b[k], s[k]
        xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
        ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
        if xs[-1] != W - tile:
            xs.append(W - tile)
        if ys[-1] != H - tile:
            ys.append(H - tile)
        B, S = [], []
        for y in ys:
            for x in xs:
                b, s = self._one(img.crop((x, y, x + tile, y + tile)), text)
                for bb, ss in zip(b, s):
                    B.append([bb[0] + x, bb[1] + y, bb[2] + x, bb[3] + y])
                    S.append(float(ss))
        if not B:
            return np.zeros((0, 4)), np.zeros((0,))
        k = nms(B, S)
        return np.asarray(B)[k], np.asarray(S)[k]


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
    a = ap.parse_args()

    det = Detector(box_thr=a.box_thr, text_thr=a.text_thr)

    if a.check:
        d = out_default / "run_one"
        base = Image.open(d / "s77_stage2_1024.png").convert("RGB")
        hi = Image.open(d / "s77_stage2_4096.png").convert("RGB")
        for name, im in (("base 1024", base), ("hi 4096", hi)):
            b, s = det.detect(im, "person", a.tile, a.stride)
            print(f"{name:<12} person x{len(b)}   scores {np.round(s, 2).tolist()}")
            annotate(im, b, s).save(d / f"D_det_{im.width}.png")
            for bb, ss in zip(b, s):
                cx, cy = (bb[0] + bb[2]) / 2 / im.width, (bb[1] + bb[3]) / 2 / im.height
                print(f"    ({cx:.3f}, {cy:.3f})  {bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}px  {ss:.2f}")
        print("\n已确认的三个幻影位置：(0.76,0.40) (0.61,0.93) (0.97,0.62)")
        print("检测器必须把它们都框出来，且基图上【不能】框出它们 —— 这是阳性对照。")
        print(f"标注图：{d}/D_det_*.png")
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
            b, s = det.detect(im, r["subject"], a.tile, a.stride)
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
