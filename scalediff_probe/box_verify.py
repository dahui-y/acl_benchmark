"""框级假阳性核验：检测器画的框，里面真的是那个主体吗？

为什么补这一件（两条来自看图的观察，方向相反，所以不能只查一边）：

  ① 老鹰那张，**两个 arm 里都有一个小框框住的不是鹰**。也就是说我们这版
     真实的 excess 是 0 而不是 +1 —— 假阳性让**我们看起来更差**。
  ② 小船那张，基线有个框扣在**船的倒影**上。倒影不是第二条船，那个框
     本来就不该存在 —— 假阳性让**基线看起来更差**。

box_audit.py 只查"消失的框"，是单边的；它答不了这两条。检测器的假阳性
在两个 arm 里都有，净方向不明，必须单独量。

核验方法（零标注）：把每个框裁出来喂 CLIP，比"这是一个 {subject}"和一组
背景描述谁更像。低于阈值就判为假阳性。

**阈值不是我拍的，用 empty 那一类当零假设。** prompts.py 里 empty 五条
写的就是"没有人的雪原/空房间/…"，subject 却填 person —— 那一类里检测器
画出的**每一个框都是假阳性，按构造如此**。它本来就是为这个目的设的负对照。
取阈值 = empty 类框的核验分的 95 分位，即"放过不超过 5% 的已知假阳性"。

诚实交代：这会改动主结果的数字，所以核验前后两套都要报，不能只报好看的。
用 empty 做负对照是它被设计出来的用途，不是拿测试集调参；但它确实动了
同一批 30 条 prompt，所以按纪律必须写明。

    python scalediff_probe/box_verify.py                    # 两个 arm 都核验
    python scalediff_probe/box_verify.py --pctl 90
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_score import load_clip                      # noqa: E402
from count_objects import Detector                    # noqa: E402
from subject_phrases import CARD                      # noqa: E402

MARGIN = 0.30

# 背景类负例。刻意做成"场景里常见的东西"，这样框住天空/水面/岩石的
# 假阳性会被这些吃掉，而真的主体不会。
NEG = [
    "a photo of the sky", "a photo of clouds", "a photo of water",
    "a photo of rocks", "a photo of trees", "a photo of sand",
    "a photo of snow", "a photo of grass", "a photo of a wall",
    "an empty landscape with nothing in it", "a blurry patch of texture",
    "a reflection on water",                # 小船那一例，直接给它一个去处
]


class Verifier:
    def __init__(self, clip):
        self.c = clip
        self.neg = self._text(NEG)
        self._cache = {}

    @torch.no_grad()
    def _text(self, texts):
        if hasattr(self.c, "tok"):
            e = self.c.m.encode_text(self.c.tok(texts).to("cuda"))
        else:
            inp = self.c.p(text=texts, return_tensors="pt",
                           padding=True, truncation=True).to("cuda")
            e = self.c.m.get_text_features(**inp)
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _img(self, crop):
        if hasattr(self.c, "pre"):
            im = self.c.pre(crop).unsqueeze(0).to("cuda")
            e = self.c.m.encode_image(im)
        else:
            inp = self.c.p(images=crop, return_tensors="pt").to("cuda")
            e = self.c.m.get_image_features(**inp)
        return e / e.norm(dim=-1, keepdim=True)

    def margin(self, img, box, subject):
        """核验分 = sim(裁块, "a photo of a {subject}") - max sim(裁块, 负例)。

        用差值而不是绝对相似度：不同主体词的基线相似度差很多，
        差值把那部分消掉。
        """
        if subject not in self._cache:
            self._cache[subject] = self._text([f"a photo of a {subject}"])
        pos = self._cache[subject]
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        r = (max(0, int(x0 - w * MARGIN)), max(0, int(y0 - h * MARGIN)),
             min(img.width, int(x1 + w * MARGIN)),
             min(img.height, int(y1 + h * MARGIN)))
        if r[2] - r[0] < 8 or r[3] - r[1] < 8:
            return -1.0
        e = self._img(img.crop(r))
        return float((e @ pos.T).max() - (e @ self.neg.T).max())


# 零假设用的额外查询词。empty 五条都是雪原/沙丘/海面/林冠/冰川，
# 里面这些东西一个都没有 —— 所以任何一个词查出来的框，按构造都是假阳性。
# 第一版只查了 person，全部 empty 行加起来才 8 个框，而且两个 arm 在
# empty 上逐字节相同（15/15 已验证），等于同一个框数了两遍 —— 真实样本
# 只有 4 个。n=4 的 95 分位没有意义。这里把词和分辨率都铺开。
NULL_SUBJECTS = ["person", "boat", "bird", "car", "dog", "house", "tree", "animal"]


def load_counts(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def hi_file(rec):
    return rec["files"][str(max(int(k) for k in rec["files"]))]


def scan(d, det, ver, keys, C, M):
    """返回 {key: [(box, score, margin), ...]}（只看最高分辨率那张）。"""
    out = {}
    for k in keys:
        rec = M[k]
        img = Image.open(Path(d) / hi_file(rec)).convert("RGB")
        boxes, scores = det.detect(img, C[k]["subject"])
        out[k] = [(b, float(s), ver.margin(img, b, C[k]["subject"]))
                  for b, s in zip(boxes, scores)]
        del img
    return out


def build_null(d, det, ver, keys, C, M, subjects, verbose=True):
    """零假设：在 empty 五条的所有分辨率上，用一组必定不存在的词查框。

    **只扫一个 arm。** empty 行两个 arm 逐字节相同，扫两遍等于把每个观测
    数两次，会假性缩小分位数的方差。
    """
    vals, seen = [], set()
    for k in keys:
        if C[k]["cat"] != "empty":
            continue
        for res, fn in sorted(M[k]["files"].items(), key=lambda kv: int(kv[0])):
            p = Path(d) / fn
            h = p.stat().st_size, fn
            if h in seen:
                continue
            seen.add(h)
            img = Image.open(p).convert("RGB")
            for subj in subjects:
                boxes, _ = det.detect(img, subj)
                for b in boxes:
                    vals.append(ver.margin(img, b, subj))
            del img
    if verbose:
        print(f"零假设取样：empty 五条 × {len(subjects)} 个必不存在的词 "
              f"× 各分辨率 -> {len(vals)} 个已知假阳性框")
    return vals


def mae(counts, keys):
    v = [abs(counts[k]) for k in keys if counts[k] is not None]
    return sum(v) / len(v) if v else float("nan")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--pctl", type=float, default=95,
                    help="阈值 = empty 类框核验分的这个分位")
    ap.add_argument("--dump", nargs="*", default=None, metavar="IDX:SEED",
                    help="逐框打印分数与核验分，例如 --dump 4:2025 3:1234")
    a = ap.parse_args()

    A, B = load_counts(a.base), load_counts(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))

    det, ver = Detector(), Verifier(load_clip())
    print("扫描基线 arm ...");  SA = scan(a.base, det, ver, keys, A, MA)
    print("扫描我们 arm ...");  SB = scan(a.new, det, ver, keys, A, MB)

    null = build_null(a.base, det, ver, keys, A, MA, NULL_SUBJECTS)
    if len(null) < 20:
        print(f"\n只取到 {len(null)} 个已知假阳性框 —— 样本太小，"
              f"{a.pctl} 分位不稳定。加词或加 empty 行再来。")
        return 1
    thr = float(np.percentile(null, a.pctl))
    print(f"\n零假设：{len(null)} 个框（按构造全是假阳性）")
    print(f"  核验分 中位 {np.median(null):+.4f}  {a.pctl:.0f} 分位 {thr:+.4f}")
    print(f"  -> 阈值 {thr:+.4f}：核验分不高于它的框判为假阳性")

    if a.dump:
        want = {tuple(int(x) for x in s.split(":")) for s in a.dump}
        for k in keys:
            if k not in want:
                continue
            print(f"\n逐框明细 {k[0]:02d}_{A[k]['cat']}_s{k[1]}  "
                  f"subject={A[k]['subject']}  阈值 {thr:+.4f}")
            for arm, S in (("基线", SA), ("我们", SB)):
                for b, s, m in sorted(S[k], key=lambda t: -t[2]):
                    w, h = int(b[2] - b[0]), int(b[3] - b[1])
                    print(f"  {arm}  det {s:.2f}  核验 {m:+.4f}  "
                          f"{w}x{h} @ ({int(b[0])},{int(b[1])})  "
                          f"{'保留' if m > thr else '判为假阳性'}")

    print(f"\n{'idx':<5}{'seed':>6}{'cat':<10}"
          f"{'基线 原/核验':>14}{'我们 原/核验':>14}"
          f"{'card':>5}{'excess 原':>10}{'excess 核验':>12}")
    print("-" * 78)
    e_raw_a, e_raw_b, e_ver_a, e_ver_b = {}, {}, {}, {}
    for k in keys:
        card = CARD[k[0]] if k[0] < len(CARD) else None
        na, nb = len(SA[k]), len(SB[k])
        va = sum(1 for _, _, m in SA[k] if m > thr)
        vb = sum(1 for _, _, m in SB[k] if m > thr)
        e_raw_a[k] = None if card is None else na - card
        e_raw_b[k] = None if card is None else nb - card
        e_ver_a[k] = None if card is None else va - card
        e_ver_b[k] = None if card is None else vb - card
        if card is None:
            continue
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}"
              f"{f'{na} / {va}':>14}{f'{nb} / {vb}':>14}{card:>5}"
              f"{f'{e_raw_a[k]:+} -> {e_raw_b[k]:+}':>10}"
              f"{f'{e_ver_a[k]:+} -> {e_ver_b[k]:+}':>12}")

    for name, ks in (("lone", [k for k in keys if A[k]["cat"] == "lone"]),
                     ("全体", keys)):
        print(f"\n{name} MAE：")
        print(f"  核验前   基线 {mae(e_raw_a, ks):.2f}  ->  我们 {mae(e_raw_b, ks):.2f}")
        print(f"  核验后   基线 {mae(e_ver_a, ks):.2f}  ->  我们 {mae(e_ver_b, ks):.2f}")

    print("""
两套数都要报。核验前后**改善幅度**若一致，说明结论不依赖这把尺子的细节，
这是好事；若差很多，则以核验后为准，并在正文说明核验规则和它的负对照。
注意方向不是单边的：假阳性在基线里（倒影）抬高基线，在我们这边（非鹰的
小框）抬高我们，净效应只能量出来，猜不出来。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
