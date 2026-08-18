#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐对在 full / no_adain 之间挑一个 —— **oracle 上界**，量 headroom。

    2026-08-18 的因果分解给出：
      · AdaIN 的内容代价 40 个 style 全为正（均值 0.159 归一化 LPIPS）
      · 风格收益高度集中（前 1/4 占 53%），且 style 33/39 为负
      · 两者在 style 之间几乎不相关，**r = 0.090**

    「代价人人付、收益归少数、且两者不相关」直接推出一个改法：
    **逐对决定要不要施加 init AdaIN**。它只是 latent 的均值/方差对齐，
    开销近似为零，完全 training-free。

    但 ArtFID 的风格项是**集合级** FID，逐对更优不一定翻译成表上的分数。
    所以先量上界：用**已经生成好的**两批图，逐对挑一张，看 ArtFID 能到哪。
    不需要再跑一张图。

    ⚠️ 这是 **oracle**：挑选用到了输出本身算出的指标，即测试期信息。
       它给的是**上界，不是方法**。上界不够大就不必去找可实现的规则。

    ⚠️ 必须有**随机对照**：按同样的比例随机挑。如果随机也能拿到同样的
       改进，那改进来自「两批图混合」本身（FID 对集合多样性敏感），
       与我们的判据无关。没有这个对照，整个读数不可信。

用法：
    python style_probe/oracle.py --root $SD_OUT/style/styleid
    # 然后按它打印的循环跑 ArtFID
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

# λ 网格：按 (风格增益 − λ·内容破坏) 逐对选臂。两个量单位不同，
# 所以 λ 没有物理含义，只是扫一条选择曲线 —— 报告时必须说明这一点。
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", nargs=2, default=["full", "no_adain"])
    ap.add_argument("--seed", type=int, default=0, help="随机对照用")
    a = ap.parse_args()
    root = Path(a.root)
    A, B = a.arms

    z = {n: np.load(root / n / "matrix.npz", allow_pickle=True) for n in (A, B)}
    if [str(x) for x in z[A]["sty"]] != [str(x) for x in z[B]["sty"]]:
        raise SystemExit("!! 两臂的 style 顺序不一致")
    names = [Path(str(x)).stem for x in z[A]["sty"]]
    cnames = [Path(str(x)).stem for x in z[A]["cnt"]]
    gS = {n: z[n]["style_gain"] for n in (A, B)}
    cN = {n: z[n]["content_norm"] for n in (A, B)}
    ns, nc = gS[A].shape

    rng = np.random.default_rng(a.seed)
    made = []
    for lam in LAMBDAS:
        # 得分越大越好：风格增益高、内容破坏低
        sA = gS[A] - lam * cN[A]
        sB = gS[B] - lam * cN[B]
        pick = np.where(sA >= sB, 0, 1)          # 0 = A(full), 1 = B(no_adain)
        rate = pick.mean()
        d = root / f"oracle_l{lam}"
        _build(root, d, names, cnames, pick, (A, B))
        print(f"λ={lam:<5} 选 {B} 的比例 {rate*100:5.1f}%  → {d.name}")
        made.append(d)

        # 同比例随机对照：**没有它，任何改进都可能只是混合带来的**
        r = (rng.random((ns, nc)) < rate).astype(int)
        dr = root / f"random_l{lam}"
        _build(root, dr, names, cnames, r, (A, B))
        made.append(dr)

    print(f"\n各臂本身也要报，作为基线。跑：")
    print(f"  S=help_code/StyleID/data/sty; C=help_code/StyleID/data/cnt")
    print(f"  for d in {A} {B} " + " ".join(x.name for x in made) + "; do")
    print(f"    echo \"== $d\"; python style_probe/run_artfid.py "
          f"--tar {root}/$d --sty $S --cnt $C 2>&1 | tail -1")
    print(f"  done")
    print(f"\n判读：**oracle 必须明显好过两个单臂，且明显好过同比例的 random。**")
    print(f"  · oracle ≈ random → 改进来自混合，不是来自判据，作废")
    print(f"  · oracle ≈ 单臂   → 没有 headroom，这条路到此为止")


def _build(root, d, names, cnames, pick, arms):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, s in enumerate(names):
        for j, c in enumerate(cnames):
            fn = f"{s}__{c}.png"
            src = (root / arms[pick[i, j]] / fn).resolve()
            if not src.exists():
                raise SystemExit(f"!! 缺 {src}")
            (d / fn).symlink_to(src)


if __name__ == "__main__":
    main()
