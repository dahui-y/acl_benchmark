#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可实现的规则：**按 latent 统计距离决定要不要施加 init AdaIN**。

    adain_predict.py 的读数（逐对，n=800）：
        d_mu  vs AdaIN 的内容代价  r = 0.610
        shift vs AdaIN 的内容代价  r = 0.639
        d_mu  vs AdaIN 的风格收益  r = 0.416
    代价与收益都随 latent 平移量增长，**但代价涨得更快**。
    所以规则的形状是：**平移量大的对，跳过 AdaIN。**

    与 oracle.py 的区别，这是全部要点：
      · oracle 用输出算出的指标挑臂 —— 测试期信息，是上界不是方法
      · 这里只用 style 图和 content 图的 latent 统计 —— **生成前就能算**，
        代价是每张图一次反演（已在主流程里做过），判别是毫秒级

    **不调参的那一档**：q=0.5，即「对 latent 平移量较小的那一半施加 AdaIN」。
    分位数中位是唯一无需选择的点。如果它就能赢 StyleID，那这个结果不含调参。
    其余 q 只用来看曲线是否平坦 —— 曲线平坦说明结论稳健，
    只有单个 q 好则说明是运气。

    ⚠️ 仍然要有**同比例 random 对照**。理由与 oracle.py 相同：
       FID 是集合级的，混合两批分布不同的图本身就可能降低它。

用法：
    python style_probe/rule.py --root $SD_OUT/style/styleid
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

QS = (0.25, 0.4, 0.5, 0.6, 0.75)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pred", default="shift", choices=["shift", "d_mu", "d_sigma"])
    ap.add_argument("--arms", nargs=2, default=["full", "no_adain"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    A, B = a.arms

    z = np.load(root / "adain_predict.npz", allow_pickle=True)
    pred = z[a.pred]
    names_s = [str(x) for x in z["sty"]]
    names_c = [str(x) for x in z["cnt"]]
    ns, nc = pred.shape
    print(f"预测量 = {a.pred}，{ns}×{nc}，范围 {pred.min():.3f} ~ {pred.max():.3f}")

    rng = np.random.default_rng(a.seed)
    made = []
    for q in QS:
        thr = np.quantile(pred, q)
        # 平移量小 → 施加 AdaIN（用 A=full）；大 → 跳过（用 B=no_adain）
        pick = (pred > thr).astype(int)
        rate = pick.mean()
        tag = f"rule_{a.pred}_q{q}"
        _build(root, root / tag, names_s, names_c, pick, (A, B))
        star = "  ★不调参档" if abs(q - 0.5) < 1e-9 else ""
        print(f"q={q:<5} 阈值 {thr:7.3f}  跳过 AdaIN 的比例 {rate*100:5.1f}%"
              f"  → {tag}{star}")
        made.append(tag)
        r = (rng.random((ns, nc)) < rate).astype(int)
        _build(root, root / f"rand_{a.pred}_q{q}", names_s, names_c, r, (A, B))
        made.append(f"rand_{a.pred}_q{q}")

    print(f"\n跑：")
    print(f"  S=help_code/StyleID/data/sty; C=help_code/StyleID/data/cnt")
    print(f"  for d in {A} {B} " + " ".join(made) + "; do")
    print(f"    printf '%-22s ' $d")
    print(f"    python style_probe/run_artfid.py --tar {root}/$d "
          f"--sty $S --cnt $C 2>&1 | tail -1")
    print(f"  done")
    print(f"\n判读：")
    print(f"  · **q=0.5 这一档**比 {A} 好 —— 那是不含调参的结果，是要写进论文的数")
    print(f"  · 整条 q 曲线都好过 {A} —— 结论稳健；只有单点好 → 是运气")
    print(f"  · 每个 q 都要好过同比例 rand —— 否则改进来自混合，不是来自规则")


def _build(root, d, names_s, names_c, pick, arms):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i, s in enumerate(names_s):
        for j, c in enumerate(names_c):
            fn = f"{s}__{c}.png"
            src = (root / arms[pick[i, j]] / fn).resolve()
            if not src.exists():
                raise SystemExit(f"!! 缺 {src}")
            (d / fn).symlink_to(src)


if __name__ == "__main__":
    main()
