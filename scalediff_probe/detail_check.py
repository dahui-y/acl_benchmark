"""干预有没有让画面变糊 —— 用频谱能量测，不需要真图参考集。

为什么需要它：CLIP score 在 4096² 上与质量无关，实测三个退化档
    双三次 1024->4096  -0.26%
    高斯模糊 r=8       **+1.18%**（糊了反而涨）
    JPEG q=15          -0.93%
模糊半径 8 px 缩到 320² 只剩 0.6 px，根本没进视觉塔。所以 ±1% 是噪声，
我们的 +0.02% 只能说"语义没变"，细节这件事 CLIP 给不了任何信息。

正规做法是 FID_c（每张图裁 10 个原分辨率 patch 再算），但它要真图参考集，
服务器下不到。频谱能量不需要参考集，而且**正好在这条线的语言里** ——
ScaleDiff 的 LFM 本身就是低通/高通混合，LSRNA 的论据也是频域的。

三个量，都在原分辨率上算，不缩放：
    lapvar   拉普拉斯方差，锐度的常用代理
    hf       径向平均功率谱里 [0.5, 1.0] 归一化频段的能量占比
    hf_hi    同上，[0.75, 1.0]（更靠近奈奎斯特，对平滑化更敏感）

**阳性对照是必须的**：双三次上采样的这三个量必须明显更低，
否则这把尺子和 CLIP 一样废，它给出的"没变糊"就不能信。

    python scalediff_probe/detail_check.py
    python scalediff_probe/detail_check.py --crop 1024   # 只在中心裁块上算，快
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def lapvar(g):
    """拉普拉斯方差。g 是 float32 灰度 [0,1]。"""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    a = np.stack([g[:-2, 1:-1], g[2:, 1:-1], g[1:-1, :-2], g[1:-1, 2:]])
    lap = a.sum(0) - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def hf_ratio(g, lo, hi):
    """径向平均功率谱里 [lo, hi] 归一化频段的能量占比。"""
    F = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())) ** 2)
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    tot = F.sum()
    return float(F[(r >= lo) & (r < hi)].sum() / max(tot, 1e-12))


def metrics(im, crop=0):
    if crop:
        w, h = im.size
        x, y = (w - crop) // 2, (h - crop) // 2
        im = im.crop((x, y, x + crop, y + crop))
    g = np.asarray(im.convert("L"), np.float32) / 255.0
    return lapvar(g), hf_ratio(g, 0.5, 1.0), hf_ratio(g, 0.75, 1.0)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--crop", type=int, default=2048,
                    help="只在中心裁块上算；0 = 整幅 4096²（慢很多）")
    a = ap.parse_args()

    A = {json.loads(l)["idx"]: json.loads(l) for l in (Path(a.base) / "manifest.jsonl").open()}
    B = {json.loads(l)["idx"]: json.loads(l) for l in (Path(a.new) / "manifest.jsonl").open()}
    idxs = sorted(set(A) & set(B))

    rows = {"基线 4096": [], "干预 4096": [], "双三次 1024->4096": []}
    cats, applied = [], []
    for i in idxs:
        fa, fb = A[i]["files"].get("4096"), B[i]["files"].get("4096")
        f1 = A[i]["files"].get("1024")
        if not (fa and fb and f1):
            continue
        rows["基线 4096"].append(metrics(Image.open(Path(a.base) / fa).convert("RGB"), a.crop))
        rows["干预 4096"].append(metrics(Image.open(Path(a.new) / fb).convert("RGB"), a.crop))
        rows["双三次 1024->4096"].append(metrics(
            Image.open(Path(a.base) / f1).convert("RGB").resize((4096, 4096), Image.BICUBIC),
            a.crop))
        cats.append(A[i]["cat"])
        applied.append(bool(B[i].get("applied", True)))

    cats, applied = np.array(cats), np.array(applied)
    M = {k: np.array(v) for k, v in rows.items()}
    n = len(cats)
    names = ["lapvar", "hf[0.50-1.0]", "hf[0.75-1.0]"]

    print(f"n={n}   中心裁块 {a.crop or 4096}²\n")
    print(f"{'':<22}" + "".join(f"{x:>16}" for x in names))
    print("-" * (22 + 16 * 3))
    for k in ("基线 4096", "干预 4096", "双三次 1024->4096"):
        print(f"{k:<20}" + "".join(f"{M[k][:, j].mean():>16.6g}" for j in range(3)))

    print(f"\n{'相对基线':<20}" + "".join(f"{x:>16}" for x in names))
    print("-" * (22 + 16 * 3))
    for k in ("干预 4096", "双三次 1024->4096"):
        print(f"{k:<20}" + "".join(
            f"{100*(M[k][:,j].mean()-M['基线 4096'][:,j].mean())/M['基线 4096'][:,j].mean():>+15.2f}%"
            for j in range(3)))

    # 阳性对照：双三次必须明显更低，否则这把尺子也是废的
    bic = 100 * (M["双三次 1024->4096"][:, 2].mean() - M["基线 4096"][:, 2].mean()) \
        / M["基线 4096"][:, 2].mean()
    ours = 100 * (M["干预 4096"][:, 2].mean() - M["基线 4096"][:, 2].mean()) \
        / M["基线 4096"][:, 2].mean()
    print(f"\n阳性对照（hf[0.75-1.0]）: 双三次 {bic:+.1f}%")
    if bic > -20:
        print("  **双三次没有明显更低 —— 这把尺子没有量程，下面的结论不能信。**")
        print("  （CLIP 那把尺子就是这么废掉的，同一个坑）")
        return
    print(f"  尺子有量程。干预 {ours:+.1f}%，是双三次退化幅度的 "
          f"{abs(ours)/abs(bic)*100:.0f}%")
    if abs(ours) < abs(bic) * 0.1:
        print("  -> 干预没有可测的平滑化")
    else:
        print("  -> **干预有可测的高频损失，要在论文里报**")

    print(f"\n{'按类别（hf[0.75-1.0] 相对基线）':<24}")
    for c in sorted(set(cats)):
        k = cats == c
        d = 100 * (M["干预 4096"][k, 2].mean() - M["基线 4096"][k, 2].mean()) \
            / M["基线 4096"][k, 2].mean()
        print(f"  {c:<12}{d:>+8.2f}%")
    k = applied
    if k.any() and not k.all():
        d = 100 * (M["干预 4096"][k, 2].mean() - M["基线 4096"][k, 2].mean()) \
            / M["基线 4096"][k, 2].mean()
        print(f"  {'仅干预行':<10}{d:>+8.2f}%   (n={int(k.sum())})")
    ke = ~applied
    if ke.any():
        d = 100 * (M["干预 4096"][ke, 2].mean() - M["基线 4096"][ke, 2].mean()) \
            / M["基线 4096"][ke, 2].mean()
        print(f"  {'未干预行':<10}{d:>+8.2f}%   (n={int(ke.sum())})  "
              "<- 应当恰好 0，脚本自检")


if __name__ == "__main__":
    main()
