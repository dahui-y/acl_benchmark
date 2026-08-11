"""结论是否依赖那些"勉强能检到"的小框？

起因：看 09_crowd 的标注图会发现框质量在小物体上很差 —— 有的框住半辆车，
有的一框套两辆，有的整个漏掉。crowd 本身 card=None 不进主指标，但推论
会外溢：**lone 类里我们要数的幻影本身就小**（4096² 上 20-60 px），
主体那只鹰是大的，多出来的九只小鸟不是。

框的毛病对计数的影响其实都是**低估**（半个框仍算 1 个；一框套两个少数 1 个；
漏掉少数 1 个），而且两个 arm 同样低估，对称性已验过。所以真正要查的不是
"框准不准"，而是：

    **把门槛抬高、只数清清楚楚的重复，−68% 还在不在？**

现在的门槛是 min_base_px=8（折回 1024² 后至少 8 px）。8-20 base-px 正是
最脆弱的区间。这里做一次门槛扫描：

    一次检测（min_base_px=0 拿到全部框），事后按尺寸筛，扫多个门槛。
    比逐门槛重跑检测快十几倍，且保证各门槛看到的是同一批框。

预注册判据（写在跑之前）：
    降幅在 8 -> 32 base-px 的整个区间内变化 < 10 个百分点
      -> 结论不依赖边缘小框，主结果稳；
    降幅随门槛显著衰减
      -> **主结果由勉强可检的小框驱动，必须改口径**（例如只报大于某尺寸
         的重复，或把尺寸分层写进正文）。

    python scalediff_probe/size_sweep.py
    python scalediff_probe/size_sweep.py --cat lone portrait structure
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector                     # noqa: E402
from subject_phrases import CARD                       # noqa: E402

THRS = [0, 8, 12, 16, 24, 32, 48, 64]


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def hi_file(rec):
    return rec["files"][str(max(int(k) for k in rec["files"]))]


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--cat", nargs="*", default=["lone"])
    ap.add_argument("--base-res", type=int, default=1024)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = [k for k in sorted(set(A) & set(B) & set(MA) & set(MB))
            if A[k]["cat"] in a.cat and CARD[k[0]] is not None]
    print(f"{len(keys)} 行（{a.cat}，有基数）\n")

    det = Detector()
    # 一次检测拿全部框，事后按尺寸筛 —— 各门槛看到的是同一批框
    sizes = {}
    for arm, d, M in (("a", a.base, MA), ("b", a.new, MB)):
        for k in keys:
            im = Image.open(Path(d) / hi_file(M[k])).convert("RGB")
            bx, _ = det.detect(im, A[k]["subject"], min_base_px=0)
            scale = im.width / a.base_res
            side = (np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]) / scale
                    if len(bx) else np.zeros(0))
            sizes[(arm, k)] = side
            del im
        print(f"检测完 {arm}")

    print(f"\n{'门槛(base px)':<14}{'基线 MAE':>10}{'我们 MAE':>10}{'降幅':>9}"
          f"{'基线均框数':>11}{'我们均框数':>11}")
    print("-" * 66)
    drops = []
    for t in THRS:
        ea, eb, na, nb = [], [], [], []
        for k in keys:
            card = CARD[k[0]]
            ca = int((sizes[("a", k)] >= t).sum())
            cb = int((sizes[("b", k)] >= t).sum())
            ea.append(abs(ca - card)); eb.append(abs(cb - card))
            na.append(ca); nb.append(cb)
        ma, mb = np.mean(ea), np.mean(eb)
        drop = (1 - mb / ma) * 100 if ma else float("nan")
        if t >= 8:
            drops.append(drop)
        print(f"{t:<14}{ma:>10.2f}{mb:>10.2f}{drop:>8.1f}%"
              f"{np.mean(na):>11.2f}{np.mean(nb):>11.2f}")

    span = max(drops) - min(drops)
    print(f"\n8-64 base px 区间内降幅跨度 {span:.1f} 个百分点")
    print("  " + ("-> 过：结论不依赖边缘小框" if span < 10 else
                  "-> **没过：主结果对尺寸门槛敏感，必须把尺寸分层写进正文**"))
    print("""
说明：门槛提高时两个 arm 的 MAE 都会降（漏掉的重复不再被数），
所以要看的是**降幅**这一列是否稳定，不是 MAE 的绝对值。
框质量本身（半个框、一框套两个）不在这里检验 —— 那些毛病都只造成
**低估**，且两个 arm 同样低估，对称性已由 subject_track 验过。""")


if __name__ == "__main__":
    main()
