"""局部模糊：整幅图的高频指标看不见一块糊掉的区域。

起因：09_crowd 的 1:1 裁块里有一条明显发糊的斜带（高架匝道那一片），
两个 arm 都有。我们此前测"细节没塌"用的是**整幅图**的高频能量（-1.1%），
但那是个全局标量 —— **一块糊掉的区域会被其余锐利区域稀释掉**。

这和我们自己的论点是同一个错误的镜像：FID 把 4096² 压到 299² 于是
看不见多出来的一个人；全局高频把 16M 像素平均成一个数，于是看不见
局部糊掉的一片。**既然我们靠这条批评领域指标，就不能自己犯同一个错。**

做法：把 4096² 切成 tile，逐块算高频能量（沿用 detail_check 的 hf_ratio），
然后回答三个问题：

  ① **是不是我们造成的？** 逐块比两个 arm。我们系统性更低 -> 是我们的锅；
     基本相等 -> 是 ScaleDiff 管线本身的（那是这条谱系公认的第二个症状：
     ScaleDiff 自己的消融说去掉 LFM 会 "heavily oversmoothed textures"）。
  ② **是不是从基图继承的？** 把基图上采样到同样网格算同一个量。
     糊的块在基图上也糊 -> 继承，不是外推阶段产生的。
  ③ **最糊的块在哪** -> 存坐标，可直接喂 make_figure 看。

判据（预注册）：
  我们比基线低超过 10% 的 tile 占比 < 5%  -> 局部模糊不是我们引入的；
  >= 5% -> **"细节没塌"这句话必须降级，改成分块报告。**

    python scalediff_probe/sharpness_map.py
    python scalediff_probe/sharpness_map.py --idx 9 --seed 77 --dump
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detail_check import hf_ratio                      # noqa: E402

TILE = 256
LO, HI = 0.25, 0.5          # 与 detail_check 同一频带


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def files(rec):
    return (rec["files"][str(min(int(k) for k in rec["files"]))],
            rec["files"][str(max(int(k) for k in rec["files"]))])


def hf_grid(img, tile=TILE):
    """逐 tile 的高频能量占比。返回 (ny, nx) 数组。"""
    g = np.asarray(img.convert("L"), dtype=np.float64)
    H, W = g.shape
    ny, nx = H // tile, W // tile
    out = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            out[i, j] = hf_ratio(g[i*tile:(i+1)*tile, j*tile:(j+1)*tile], LO, HI)
    return out


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--drop", type=float, default=0.10,
                    help="低于基线这个比例才算'明显更糊'")
    ap.add_argument("--dump", action="store_true", help="打印最糊的 tile 坐标")
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))
    if a.idx is not None:
        keys = [k for k in keys if k[0] == a.idx]
    if a.seed is not None:
        keys = [k for k in keys if k[1] == a.seed]

    print(f"tile={a.tile}  频带 [{LO},{HI}]  判据：我们低于基线 {a.drop:.0%} "
          f"以上的 tile 占比 < 5%\n")
    print(f"{'idx':<5}{'seed':>6}{'cat':<10}{'基线 HF':>9}{'我们 HF':>9}"
          f"{'比值':>7}{'更糊 tile':>10}{'基图也糊':>10}")
    print("-" * 68)

    worse_all, tiles_all = 0, 0
    inherit_num, inherit_den = 0, 0
    rows = []
    for k in keys:
        lo_a, hi_a = files(MA[k])
        _, hi_b = files(MB[k])
        ia = Image.open(Path(a.base) / hi_a).convert("RGB")
        ib = Image.open(Path(a.new) / hi_b).convert("RGB")
        base = Image.open(Path(a.base) / lo_a).convert("RGB")
        ga, gb = hf_grid(ia, a.tile), hf_grid(ib, a.tile)
        # 基图放大到同一网格 —— 用 NEAREST 只为对齐坐标，不引入插值细节
        gbase = hf_grid(base.resize(ia.size, Image.BICUBIC), a.tile)

        ratio = gb / np.maximum(ga, 1e-9)
        worse = ratio < (1 - a.drop)
        n = ratio.size
        # "基图也糊"：在我们更糊的那些块里，基图的高频是否也在低分位
        thr_base = np.percentile(gbase, 25)
        inherited = int((worse & (gbase <= thr_base)).sum())

        worse_all += int(worse.sum()); tiles_all += n
        inherit_num += inherited; inherit_den += int(worse.sum())
        rows.append((k, ga, gb, ratio, worse))

        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{ga.mean():>9.4f}"
              f"{gb.mean():>9.4f}{gb.mean()/max(ga.mean(),1e-9):>7.3f}"
              f"{f'{worse.sum()}/{n}':>10}"
              f"{f'{inherited}/{max(int(worse.sum()),1)}':>10}")

        if a.dump:
            flat = np.dstack(np.unravel_index(np.argsort(ratio, axis=None), ratio.shape))[0]
            print("     我们最糊的 5 块（相对基线）：")
            for i, j in flat[:5]:
                print(f"       tile ({j*a.tile},{i*a.tile})  "
                      f"基线 {ga[i,j]:.4f}  我们 {gb[i,j]:.4f}  "
                      f"比值 {ratio[i,j]:.3f}  基图 {gbase[i,j]:.4f}")
            print("     两个 arm 都糊的 5 块（绝对最低，看 ScaleDiff 自身）：")
            s = np.dstack(np.unravel_index(np.argsort(ga, axis=None), ga.shape))[0]
            for i, j in s[:5]:
                print(f"       tile ({j*a.tile},{i*a.tile})  "
                      f"基线 {ga[i,j]:.4f}  我们 {gb[i,j]:.4f}  "
                      f"基图 {gbase[i,j]:.4f}")
        del ia, ib, base

    frac = worse_all / max(tiles_all, 1)
    print(f"\n全体：明显更糊的 tile {worse_all}/{tiles_all} = {frac:.2%}")
    print("  " + ("-> 判据过：局部模糊不是我们引入的" if frac < 0.05 else
                  "-> **判据没过：'细节没塌'必须降级为分块报告**"))
    if inherit_den:
        print(f"  其中落在基图低高频区（继承而来）的占 "
              f"{inherit_num}/{inherit_den} = {inherit_num/inherit_den:.0%}")
    print("""
读法：
  比值 ~1.0 且更糊 tile 占比低  -> 那条发糊的斜带是 ScaleDiff 管线本身的，
     属于这条谱系公认的第二个症状（ScaleDiff 自己的消融：去掉 LFM 会
     "heavily oversmoothed textures"），不是我们造成的 —— 但**要在论文
     里承认它存在**，因为定性图上看得见。
  比值明显 < 1  -> 我们确实在抹细节，"高频 -1.1%" 那句话是被全局平均
     掩盖的，必须改口径。""")


if __name__ == "__main__":
    main()
