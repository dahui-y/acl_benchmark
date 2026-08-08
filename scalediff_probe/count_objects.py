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
    幻影缩完只剩 12 px，必然漏检。切成块再检测，同一个幻影有 ~47 px，能测到。

    tile 取 **图像边长的 1/4**（步长 tile/2），不是固定 1024。固定值会让不同
    分辨率的表观物体尺寸差 4 倍，delta 天然为正 —— 详见 detect() 的注释，
    这是一处已修的结构性偏差。4096² 下 width/4 恰好还是 1024，所以 4096 的
    计数没有变化，只有基图和 2048 被修正。

    python scalediff_probe/count_objects.py --check      # 阳性对照：seed77 那张
    python scalediff_probe/count_objects.py              # 跑整个 batch
    python scalediff_probe/scale_check.py                # 尺度不变性自检
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_phrases import CARD          # noqa: E402

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


def merge_split(boxes, scores, gap=24, overlap=0.6, rounds=3):
    """把被 tile 边界切开的同一个物体接回去。

    NMS 做不到这件事，实测就是证据：主角在 4096² 上高 583 px，拿到的是
    (0.404,0.471) 214x238 和 (0.405,0.542) 220x345 —— 上半身和下半身，
    y 只差 1 px 就接上了，但交集几乎为零，所以 IoU≈0，两个框都留了下来。
    加整图那一遍也救不回来：完整框和下半身框 IoU=0.59 会被后者吃掉，
    而上半身框谁也吃不掉。所以必须专门认"紧邻且另一轴对齐"这个模式。
    """
    B = [list(map(float, b)) for b in boxes]
    S = [float(s) for s in scores]
    for _ in range(rounds):
        merged, used = [], set()
        for i in range(len(B)):
            if i in used:
                continue
            a = B[i]
            for j in range(i + 1, len(B)):
                if j in used:
                    continue
                b = B[j]
                # 纵向紧邻 + 横向重叠够多（或反过来）
                vgap = max(a[1], b[1]) - min(a[3], b[3])
                hov = (min(a[2], b[2]) - max(a[0], b[0])) / max(
                    1e-6, min(a[2] - a[0], b[2] - b[0]))
                hgap = max(a[0], b[0]) - min(a[2], b[2])
                vov = (min(a[3], b[3]) - max(a[1], b[1])) / max(
                    1e-6, min(a[3] - a[1], b[3] - b[1]))
                if (-gap <= vgap <= gap and hov >= overlap) or \
                   (-gap <= hgap <= gap and vov >= overlap):
                    a = [min(a[0], b[0]), min(a[1], b[1]),
                         max(a[2], b[2]), max(a[3], b[3])]
                    S[i] = max(S[i], S[j])
                    used.add(j)
            merged.append((a, S[i]))
            used.add(i)
        if len(merged) == len(B):
            B = [m[0] for m in merged]
            S = [m[1] for m in merged]
            break
        B = [m[0] for m in merged]
        S = [m[1] for m in merged]
    return np.asarray(B) if B else np.zeros((0, 4)), np.asarray(S) if S else np.zeros((0,))


def suppress_contained(boxes, scores, ios_thr=0.75):
    """删掉基本被别的框包住的小框。

    NMS 用 IoU，抓不到包含关系：实测 nosubj 那张的主角给出 226x590 和
    216x242 两个框，小的完全落在大的里面，IoU = 0.39 恰好低于 0.40 的阈值，
    于是主角被数成两个人。这里改用 交集/较小框面积（IoS）。
    """
    if len(boxes) == 0:
        return boxes, scores
    B, S = np.asarray(boxes, float), np.asarray(scores, float)
    area = np.maximum(0, B[:, 2] - B[:, 0]) * np.maximum(0, B[:, 3] - B[:, 1])
    order = area.argsort()[::-1]                 # 从大到小，大的留下
    keep = []
    for i in order:
        drop = False
        for j in keep:
            xx1, yy1 = max(B[i, 0], B[j, 0]), max(B[i, 1], B[j, 1])
            xx2, yy2 = min(B[i, 2], B[j, 2]), min(B[i, 3], B[j, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter / max(min(area[i], area[j]), 1e-9) > ios_thr:
                drop = True
                break
        if not drop:
            keep.append(int(i))
    return B[keep], S[keep]


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

    def detect(self, img, text, tile=None, stride=None, iou=0.40, max_aspect=0.0,
               min_base_px=8, base_res=1024, min_score=0.50, max_area_frac=0.0,
               degenerate_max_score=0.50):
        """整图 + 分块两遍，合并后 NMS。

        tile=None -> 取 img.width/4，stride 取 tile/2。**这不是调参，是修一处
        结构性偏差。** 原来 tile 固定 1024，而分块那一遍有 `if W > tile`，
        于是 1024² 的基图根本不分块，只走整图一遍被缩到 ~800px；4096² 走整图
        一遍 + 49 块原分辨率 tile。同一个相对大小 f 的物体，检测器实际看到：

            4096²：1024 的 tile -> 缩到 800（x0.78）   f x 4096 x 0.78 = f x 3200
            1024²：整图 1024   -> 缩到 800（x0.78）   f x 1024 x 0.78 = f x 800

        **4 倍差。** delta = 高分辨率计数 - 基图计数 于是天然为正 —— 不是模型
        多画了东西，是我们在高分辨率上看得更仔细。"rush hour, many cars" 基图
        数出 0 辆车、"a colony of penguins" 基图 0 只，都是这么来的。

        tile = width/4 之后：1024² 的 tile 是 256，被处理器放大到 800（x3.125），
        f x 1024 x 3.125 = f x 3200 —— 与 4096² 一致。而 4096² 下 width/4 恰好
        就是原来的 1024，**4096 的计数一个字不变**，只修基图和 2048。
        正确性由 scale_check.py 验证：同一张图降采样后计数应当一致。

        为什么必须两遍。阳性对照里主角在 4096² 上高 582 px，而 tile=1024 /
        stride=512 意味着任何高于 stride 的物体都会被某条 tile 边界切开：
        实测拿到的是 (0.404,0.471) 214x238 和 (0.405,0.542) 220x345 —— 上半身
        和下半身两个框，y 几乎不重叠，IoU≈0，NMS 合不掉。**每个大物体都被数
        了两次。** 整图那一遍能给出完整的框（主角缩放后仍有 ~114 px，检得到），
        NMS 就能把两个半身框吃掉。分块那一遍负责小幻影（60 px 的人整图检测时
        只剩 12 px，必漏）。两遍各管一头。

        max_aspect / max_area_frac **默认已关闭**（0）。它们当初是在 lone 这一
        类上为了杀特定误检加的（云的横条、无特征区域的整图框），然后把其余
        类别的主体一起杀了：特写肖像、大教堂立面、俯拍键盘的主体占满画面，
        被 max_area_frac 删光（20/21/22/24/25/28/29 的 4096² 计数全是 0）；
        桥的侧视图又宽又扁，被 max_aspect 删掉（26 的 delta 是 -2）。
        delta 看不见这件事（base 和 4096 都是 0），excess 一眼就露出来。
        它们想杀的东西分别由 min_score 和上面的退化框尺寸规则覆盖。
        需要复现旧行为就传 --max-aspect 2.0 --max-area-frac 0.25。
        """
        W, H = img.size
        if not tile:
            tile = max(64, W // 4)
        if not stride:
            stride = max(32, tile // 2)
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
        B, S = merge_split(B[k], S[k])
        B, S = suppress_contained(B, S)

        # 尺度门槛。判据是"下采样回基图分辨率后多出来的东西"，所以一个
        # 下采样后不足 min_base_px 的检测【不构成证据】—— 基图物理上就画不出
        # 那么小的东西。实测 4096² 上高 17-24 px 的那一批（#6-#12），
        # 折回 1024² 只剩 4-6 px，全是纹理噪点被读成人形。
        # 用"基图像素"表述而不是绝对像素，2048² 和 4096² 的计数才可比。
        if min_base_px and len(B):
            k = img.width / base_res
            side = np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]) / k
            keep = side >= min_base_px
            self.dropped_small = int((~keep).sum())
            B, S = B[keep], S[keep]

        # 退化框。GroundingDINO 拿到一块无特征的图（冰面、沙丘）会吐出一个
        # 覆盖【整块 tile】或整张图的低分框：实测 14_empty 的 4096² 上拿到
        # 1023x1023 / 1022x1019 / 1022x1022 / 2047x2046，分数 0.31-0.42，
        # 而 11_empty 的 1024² 上是 1022x937 —— 就是整幅画面。
        # 一个恰好等于 tile 边界的"人"不是人，是检测器没东西可框。
        self.dropped_degenerate = 0
        self.degenerate_boxes = []
        if len(B):
            w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
            sizes = [tile, 2 * tile, img.width, img.height]
            near = np.zeros(len(B), bool)
            for t in sizes:
                near |= (np.abs(w - t) < 0.03 * t) & (np.abs(h - t) < 0.03 * t)
            # **判据是尺寸【加】低分，不是只看尺寸。** 只看尺寸时，特写肖像、
            # 大教堂立面、俯拍键盘的主体全被删光（20/22/24/25/28/29 的 4096²
            # 计数是 0，逐图核对确认各自恰好一个框被判成退化框）。
            # 依据在上面：无特征区域的整图框分数 0.31-0.42，真主体 0.9 —— 两者
            # 不重叠。低分那一批本来就过不了 min_score，所以这条规则加上分数
            # 条件之后基本是冗余的，留着只为把"尺寸恰好等于 tile"这个模式
            # 显式记下来。
            if degenerate_max_score:
                near &= S < degenerate_max_score
            # max_area_frac 是退化框规则的粗暴版本，而且有害：特写肖像、
            # 大教堂立面、俯拍键盘的主体本来就占满画面，一律被当成退化框删掉。
            # 实测 20/21/22/24/25/28/29 的 4096² 主体计数全是 0，excess 全是 -1。
            # 上面那条"框尺寸落在 tile / 2*tile / 图像尺寸的 3% 以内"已经精确
            # 覆盖了它想杀的东西。0 = 关闭。
            too_big = ((w * h) > max_area_frac * img.width * img.height
                       if max_area_frac else np.zeros(len(B), bool))
            keep = ~(near | too_big)
            self.dropped_degenerate = int((~keep).sum())
            self.degenerate_boxes = [(list(map(float, bb)), float(ss))
                                     for bb, ss in zip(B[~keep], S[~keep])]
            B, S = B[keep], S[keep]

        # 分数下限。阳性对照里三个已确认的幻影是 0.91/0.89/0.83，垃圾框
        # 全在 0.50 以下 —— 两者不重叠。这个阈值是【看着 seed 77 定的】，
        # 是全套参数里唯一一个靠标定而非机制的，写论文时要单独做敏感度分析。
        if min_score and len(B):
            keep = S >= min_score
            self.dropped_score = int((~keep).sum())
            B, S = B[keep], S[keep]
        else:
            self.dropped_score = 0
        return B, S


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
    ap.add_argument("--subject", default="person")
    ap.add_argument("--check", action="store_true",
                    help="阳性对照：只跑 run_one 的 seed77，看能不能找到那 3 个已确认的人")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=0,
                    help="0 = 自动取 width/4（尺度匹配）。给非零值会重现旧的尺度偏差")
    ap.add_argument("--stride", type=int, default=0, help="0 = 自动取 tile/2")
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--image", default=None,
                    help="只查一张图：给路径，逐框存原分辨率裁块")
    ap.add_argument("--max-aspect", type=float, default=0.0,
                    help="宽/高 超过这个值的框丢掉（站立的人不会是横条）；0 关闭")
    ap.add_argument("--min-score", type=float, default=0.50)
    ap.add_argument("--max-area-frac", type=float, default=0.0,
                    help="0 = 关闭（默认）。退化框已由 tile 尺寸规则精确覆盖")
    ap.add_argument("--min-base-px", type=float, default=8.0,
                    help="折算回基图分辨率后短于这个值的检测丢掉；0 关闭")
    a = ap.parse_args()

    det = Detector(box_thr=a.box_thr, text_thr=a.text_thr)

    if a.image:
        # 逐框存裁块。30 条那一批里 empty/texture 类的计数是否可信，只能靠
        # 肉眼核对每一个框 —— 在缩略图上猜是我们已经栽过的做法。
        p = Path(a.image)
        im = Image.open(p).convert("RGB")
        crops = p.parent / f"crops_{p.stem}"
        crops.mkdir(exist_ok=True)
        det.dropped_aspect = det.dropped_small = 0
        b, s_ = det.detect(im, a.subject, a.tile, a.stride, a.iou,
                           a.max_aspect, a.min_base_px, 1024, a.min_score, a.max_area_frac)
        print(f"{p.name}  {im.width}²  {a.subject} x{len(b)}  "
              f"(长宽比 {det.dropped_aspect}，尺度 {det.dropped_small}，退化框 {det.dropped_degenerate}，低分 {det.dropped_score})")
        for bb, ss in det.degenerate_boxes:
            print(f"    [判为退化框] {bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}px  分数 {ss:.2f}")
        for j, (bb, ss) in enumerate(sorted(zip(b, s_), key=lambda z: -z[1])):
            cx, cy = (bb[0]+bb[2])/2/im.width, (bb[1]+bb[3])/2/im.height
            w, h = bb[2]-bb[0], bb[3]-bb[1]
            print(f"    #{j:<2} ({cx:.3f}, {cy:.3f})  {w:.0f}x{h:.0f}px  {ss:.2f}")
            pad = max(w, h)
            im.crop((int(max(0, bb[0]-pad)), int(max(0, bb[1]-pad)),
                     int(min(im.width, bb[2]+pad)), int(min(im.height, bb[3]+pad)))
                    ).save(crops / f"{j:02d}_{cx:.3f}_{cy:.3f}_{ss:.2f}.png")
        annotate(im, b, s_).save(p.parent / f"D_{p.stem}.png")
        print(f"\n逐框裁块 {crops}/    标注图 {p.parent}/D_{p.stem}.png")
        return

    if a.check:
        d = out_default / "run_one"
        base = Image.open(d / "s77_stage2_1024.png").convert("RGB")
        hi = Image.open(d / "s77_stage2_4096.png").convert("RGB")
        crops = d / "det_crops"
        crops.mkdir(exist_ok=True)
        for name, im in (("base1024", base), ("hi4096", hi)):
            det.dropped_aspect = det.dropped_small = 0
            b, s = det.detect(im, "person", a.tile, a.stride, a.iou, a.max_aspect, a.min_base_px, 1024, a.min_score, a.max_area_frac)
            print(f"\n{name}  person x{len(b)}   "
                  f"(长宽比丢掉 {det.dropped_aspect}，尺度门槛丢掉 {det.dropped_small})")
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
        # min_score=0 重新检一遍。原来在已经过了 0.50 筛的分数上扫，
        # 0.50 以下那两行是假的 —— 扫描必须从没设地板的结果出发。
        b_all, s_all = det.detect(hi, "person", a.tile, a.stride, a.iou, a.max_aspect, a.min_base_px, 1024, 0.0, a.max_area_frac)
        for t in (0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
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
    print(f"{'tag':<22}{'cat':<10}{'subject':<10}{'base':>6}{'2048':>6}{'4096':>6}"
          f"{'delta':>7}{'card':>5}{'excess':>7}")
    print("-" * 82)
    for line in manifest.open():
        r = json.loads(line)
        tag = f"{r['idx']:02d}_{r['cat']}_s{r['seed']}"
        counts = {}
        for res, fn in sorted(r["files"].items(), key=lambda kv: int(kv[0])):
            im = Image.open(batch / fn).convert("RGB")
            b, s = det.detect(im, r["subject"], a.tile, a.stride, a.iou, a.max_aspect, a.min_base_px, 1024, a.min_score, a.max_area_frac)
            counts[int(res)] = len(b)
            if int(res) == max(int(k) for k in r["files"]):
                annotate(im, b, s).save(batch / f"{tag}_det_{res}.png")
        base_n = counts.get(1024, 0)
        hi_n = counts.get(max(counts), 0)
        card = CARD[r["idx"]] if r["idx"] < len(CARD) else None
        exc = None if card is None else hi_n - card
        rows.append({**{k: r[k] for k in ("idx", "cat", "subject", "seed", "prompt")},
                     "counts": counts, "delta": hi_n - base_n,
                     "card": card, "excess": exc})
        print(f"{tag:<22}{r['cat']:<10}{r['subject']:<10}"
              f"{counts.get(1024, 0):>6}{counts.get(2048, 0):>6}{counts.get(4096, 0):>6}"
              f"{hi_n - base_n:>+7}"
              f"{'  -' if card is None else f'{card:>5}'}"
              f"{'      -' if exc is None else f'{exc:>+7}'}")

    (batch / "counts.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n按类别汇总（delta = 4096 的个数 - 基图的个数）：")
    for cat in sorted({r["cat"] for r in rows}):
        d = [r["delta"] for r in rows if r["cat"] == cat]
        e = [r["excess"] for r in rows if r["cat"] == cat and r["excess"] is not None]
        pos = sum(1 for x in d if x > 0)
        es = (f"   平均 excess {np.mean(e):+.2f}   超出 prompt 基数的 "
              f"{sum(1 for x in e if x > 0)}/{len(e)}") if e else "   （prompt 未给基数）"
        print(f"  {cat:<10} n={len(d):<3} 平均 delta {np.mean(d):+.2f}   "
              f"有新增的 {pos}/{len(d)}{es}")
    ex = [r["excess"] for r in rows if r["excess"] is not None]
    if ex:
        print(f"\n  主指标（excess = 4096 计数 - prompt 基数，不跨分辨率，无尺度偏差）")
        print(f"    n={len(ex)}   平均 {np.mean(ex):+.2f}   "
              f"超出的 {sum(1 for x in ex if x > 0)}/{len(ex)}")
    print(f"\n明细：{batch/'counts.json'}")


if __name__ == "__main__":
    main()
