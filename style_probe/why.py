#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把「row 4 / 19 / 36 差，row 34 / 5 好」变成一个**属性**。

    行号不是 problem statement。看图给出的假设是：

      失败的三张 style 参考（淡粉平涂 / 光滑静物 / 纯色条纹）都是
      **低空间频率、无笔触、大面积平色**；
      成功的两张（点彩桌花 / 印象派风景）都是**高频、密集笔触**。

    假设：**K/V 注入搬运的是局部纹理统计**。当一张画的风格身份在于
    全局色彩构成而非笔触时，注入没有东西可搬 —— 于是要么几乎不动
    （row 4，ΔS 是唯一的负值），要么只剩初始 latent 的 AdaIN 把调色板
    糊到整张图上，破坏内容却不产生那张画的样子（row 19 / 36）。

    这个脚本检验它：对 40 张 style 图各算若干**高频/平坦度**指标，
    与每行的 ΔS、内容 LPIPS 求相关。相关强 → 行号变成属性，
    problem statement 有了主语。

    ⚠️ 这是相关不是因果。真正的因果检验是 StyleID 自带的两个开关
    （--without_attn_injection / --without_init_adain）—— 见 DESIGN.md。

用法：
    python style_probe/why.py --seed 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "protocol"


# ★ 判据尺度事先钉死，不扫。
#
# 2026-08-17：第一版全在 512² 像素尺度上算，row 19（橙子静物）成了反例 ——
# 它 grad 0.041 / hf 0.867 全表最高，却在最差组里。回看原图：那张画有明显的
# **画布织纹**。像素尺度的 grad/hf 被织纹主导，而织纹不是画风。
#
# 正确尺度由方法本身给定，不由我挑：StyleID 注入在 decoder 第 7-11 层，
# SD@512 的 latent 是 64×64，那几层跑在 64×64 / 32×32 上。织纹在 64×64
# 上会消失，平色块结构会留下。
#
# 所以 DECIDE_AT = 64。其他尺度只作透明报告，**不参与判定** —— 否则就是
# 在多个尺度里挑一个过线的，那是 p-hacking。
DECIDE_AT = 64
SCALES = (512, 64, 32)


def feats(path, size=DECIDE_AT):
    """几个互相独立的「有多少笔触」的量。全部在灰度图上算，与颜色无关。"""
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    g = np.asarray(im, np.float64) / 255.0

    # ① 梯度能量：最直接的「边多不多」
    gy, gx = np.gradient(g)
    grad = np.hypot(gx, gy)

    # ② 高频占比：FFT 里半径 > 1/8 奈奎斯特的能量份额
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    n = F.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    r = np.hypot(yy, xx) / (n / 2)
    hf = float(F[r > 0.125].sum() / max(F.sum(), 1e-12))

    # ③ 平坦面积占比：局部梯度低于阈值的像素比例 —— 平涂画的直接刻画
    flat = float((grad < 0.02).mean())

    # ④ 拉普拉斯方差：经典的「清晰度 / 纹理量」
    lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0) +
           np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4 * g)
    return dict(grad=float(grad.mean()), hf=hf, flat=flat, lap=float(lap.var()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = OUT / f"seed{a.seed}"
    man = json.loads((d / "manifest.json").read_text())
    z = np.load(d / "matrix.npz", allow_pickle=True)
    dS = z["style_gain"].mean(1)          # 每行的风格增益
    C = z["content_norm"].mean(1)         # 每行的归一化内容破坏

    keys = ["grad", "hf", "flat", "lap"]
    print(f"seed {a.seed}: {len(man['style'])} 张 style 图")
    print(f"判据尺度 = {DECIDE_AT}²（StyleID 注入层的 latent 分辨率，事先钉死）\n")
    allX = {}
    for sz in SCALES:
        rows = [feats(d / "sty" / x["file"], sz) for x in man["style"]]
        X = {k: np.array([r[k] for r in rows]) for k in keys}
        allX[sz] = X
        mark = " ★判据" if sz == DECIDE_AT else " (仅参考)"
        print(f"── 尺度 {sz}²{mark}")
        print(f"   {'指标':<6} {'与 ΔS 的 r':>12} {'与内容破坏的 r':>16}")
        for k in keys:
            print(f"   {k:<6} {np.corrcoef(X[k], dS)[0,1]:>12.3f}"
                  f" {np.corrcoef(X[k], C)[0,1]:>16.3f}")
    X = allX[DECIDE_AT]
    print()

    print(f"\n按 ΔS 排序的两端（看指标是否跟着单调走）：")
    o = np.argsort(dS)
    hdr = f"{'row':>4} {'ΔS':>8} {'内容破坏':>9} " + " ".join(f"{k:>7}" for k in keys)
    print(hdr)
    for i in list(o[:5]) + [None] + list(o[-5:]):
        if i is None:
            print("   " + "-" * (len(hdr) - 3)); continue
        print(f"{i:>4} {dS[i]:>8.4f} {C[i]:>9.4f} "
              + " ".join(f"{X[k][i]:>7.4f}" for k in keys)
              + f"   {man['style'][i]['file']}")

    best = max(keys, key=lambda k: abs(np.corrcoef(X[k], dS)[0, 1]))
    r = np.corrcoef(X[best], dS)[0, 1]
    print(f"\n最强的单指标：{best}，r = {r:+.3f}")
    if abs(r) >= 0.6:
        print("→ **行号可以换成属性了。** problem statement 有主语了。")
    elif abs(r) >= 0.4:
        print("→ 有倾向但不够硬。需要更多 style 图（换 seed 扩样本）或更好的属性。")
    else:
        print("→ **看图得到的假设没有被数据支持。** 别硬套，回去重新看图。")
    json.dump({"decide_at": DECIDE_AT,
               "scales": {str(sz): {k: allX[sz][k].tolist() for k in keys}
                          for sz in SCALES},
               "dS": dS.tolist(), "content": C.tolist()},
              open(d / "why.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
