"""尺度不变性自检：同一张图，换个分辨率，计数应当一样。

这是尺子最基本的一条性质，而我们的判据完全建立在它之上 ——
`delta = 高分辨率计数 - 基图计数`，如果计数本身随分辨率漂移，delta 测的
就是漂移，不是模型多画了东西。

做法：取 4096² 的图，**Lanczos 降采样**到 2048² 和 1024²。内容不变，
只有像素数变了。三档各数一遍，理想情况三个数字相同。

同时跑两种 tile 策略，用来证明修法是对的而不是碰巧：

    fixed   tile 固定 1024（旧行为）。1024² 的图不满足 W > tile，
            于是完全不分块 -> 表观物体尺寸只有 4096² 的 1/4 -> 应当明显漏计
    auto    tile = width/4（新行为）-> 三档表观尺寸一致 -> 应当收敛

判据（跑之前定死）：
    auto 的三档计数在 |1024 档 - 4096 档| 上的平均绝对差
    必须显著小于 fixed 的，且理想情况接近 0。
    这一项不过，count_objects 的 delta 就不能用。

    python scalediff_probe/scale_check.py                    # 默认 batch，全部 30 张
    python scalediff_probe/scale_check.py --n 8              # 只跑前 8 张，快速看
    python scalediff_probe/scale_check.py --batch $SD_OUT/method_batch_s1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector          # noqa: E402


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(root / "batch"))
    ap.add_argument("--n", type=int, default=0, help="只跑前 n 条；0 = 全部")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4096, 2048, 1024])
    ap.add_argument("--min-score", type=float, default=0.50)
    a = ap.parse_args()

    batch = Path(a.batch)
    mp = batch / "manifest.jsonl"
    if not mp.exists():
        sys.exit(f"没有 {mp}")
    rows = [json.loads(l) for l in mp.open()]
    if a.n:
        rows = rows[:a.n]

    det = Detector()
    sizes = sorted(a.sizes, reverse=True)
    src = sizes[0]

    print(f"{len(rows)} 张图，源 {src}²，降采样到 {sizes[1:]}\n")
    hdr = (f"{'tag':<22}{'subject':<10}"
           + "".join(f"{f'fix{s}':>8}" for s in sizes)
           + "  |" + "".join(f"{f'aut{s}':>8}" for s in sizes))
    print(hdr)
    print("-" * len(hdr))

    res = {"fixed": [], "auto": []}
    for r in rows:
        files = r["files"]
        key = str(src) if str(src) in files else str(max(int(k) for k in files))
        im0 = Image.open(batch / files[key]).convert("RGB")
        counts = {"fixed": [], "auto": []}
        for s in sizes:
            im = im0 if im0.width == s else im0.resize((s, s), Image.LANCZOS)
            for mode, tile in (("fixed", 1024), ("auto", 0)):
                b, _ = det.detect(im, r["subject"], tile or None,
                                  None, 0.40, 2.0, 8, 1024, a.min_score, 0.25)
                counts[mode].append(len(b))
        tag = f"{r['idx']:02d}_{r['cat']}_s{r['seed']}"
        print(f"{tag:<22}{r['subject']:<10}"
              + "".join(f"{c:>8}" for c in counts["fixed"])
              + "  |" + "".join(f"{c:>8}" for c in counts["auto"]))
        for m in res:
            res[m].append(counts[m])

    print()
    for m in ("fixed", "auto"):
        A = np.array(res[m])                       # (N, len(sizes))，第 0 列是源分辨率
        drift = np.abs(A - A[:, :1])               # 相对源分辨率的漂移
        agree = (A == A[:, :1]).mean(axis=0)
        print(f"[{m}]")
        for j, s in enumerate(sizes):
            print(f"   {s:>5}²   平均 |计数-源| = {drift[:, j].mean():.2f}   "
                  f"与源一致的比例 {agree[j]:.0%}")
        print(f"   最低分辨率档的平均绝对漂移 = {drift[:, -1].mean():.2f}\n")

    f = np.abs(np.array(res["fixed"]))
    u = np.abs(np.array(res["auto"]))
    fd = np.abs(f - f[:, :1])[:, -1].mean()
    ud = np.abs(u - u[:, :1])[:, -1].mean()
    print(f"判据: auto ({ud:.2f}) 必须显著小于 fixed ({fd:.2f})   "
          + ("-> 过" if ud < fd else "-> **没过**"))
    if ud >= fd:
        print("没过就不要往下走：count_objects 的 delta 还是不可用。")


if __name__ == "__main__":
    main()
