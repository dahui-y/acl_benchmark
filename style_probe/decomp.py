#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 StyleID 拆成两个部件，逐 style 看各自贡献多少。

    full      = K/V 注入 + 初始 latent AdaIN
    no_adain  = 只有注入
    no_inject = 只有 AdaIN

    于是（在同一格 (i,j) 上，三次运行的 style/content 顺序一致）：

      注入的贡献  = ΔS(full) − ΔS(no_inject)
      AdaIN 的贡献 = ΔS(full) − ΔS(no_adain)

    对内容破坏同理。这是**因果**分解 —— 开关是在位者自己代码里的，
    不是我们造的度量。

    2026-08-18 已知的总体读数（论文配置，SD1.4/50 步/γ=0.75/τ=1.5）：
      full      ArtFID 29.190  FID 18.409  LPIPS 0.5039   （发表值 28.801/18.131/0.5055）
      no_adain  ArtFID 29.358  FID 20.489  LPIPS 0.3662
      no_inject ArtFID 41.510  FID 25.630  LPIPS 0.5587
    → 风格几乎全部来自注入（关掉它 ΔS 均值 0.105→0.004）；
      AdaIN 在 ArtFID 上只值 0.17，比我们的复现误差 0.39 还小。

    这个脚本要回答的是：**那 0.17 是均匀地摊在 40 个 style 上，
    还是集中在少数几个？** 若集中，那几个就是可命名的对象。

用法：
    python style_probe/decomp.py --root $SD_OUT/style/styleid
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(root, name):
    f = Path(root) / name / "matrix.npz"
    if not f.exists():
        raise SystemExit(f"!! 缺 {f} —— 先跑 matrix.py --tar {Path(root)/name}")
    z = np.load(f, allow_pickle=True)
    return z


def report(tag, v, names=None, k=5):
    o = np.argsort(v)
    lo = [(int(i), round(float(v[i]), 4)) for i in o[:k]]
    hi = [(int(i), round(float(v[i]), 4)) for i in o[-k:][::-1]]
    print(f"\n── {tag}")
    print(f"   均值 {v.mean():+.4f}  标准差 {v.std():.4f}  "
          f"最大/最小 {v.max():+.4f}/{v.min():+.4f}")
    print(f"   贡献最大的 {k} 个 style：{hi}")
    print(f"   贡献最小的 {k} 个 style：{lo}")
    # 集中度：前 25% 的 style 占了多少总贡献
    s = np.sort(v)[::-1]
    tot = s.sum()
    if abs(tot) > 1e-9:
        top = s[:max(1, len(s) // 4)].sum() / tot
        print(f"   集中度：贡献最大的 1/4 个 style 占总贡献的 {top*100:.0f}%"
              f"（均匀分布应为 25%）")
    return dict(mean=float(v.mean()), std=float(v.std()),
                top=[int(i) for i in o[-k:][::-1]], bot=[int(i) for i in o[:k]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="含 full/ no_adain/ no_inject/ 的目录")
    a = ap.parse_args()

    Z = {n: load(a.root, n) for n in ("full", "no_adain", "no_inject")}
    # 三次运行用的是同一批 style/content、同样的排序，格子逐一对应。
    # 这一点必须核对，否则整个分解无意义。
    ref = [str(x) for x in Z["full"]["sty"]]
    for n in ("no_adain", "no_inject"):
        if [str(x) for x in Z[n]["sty"]] != ref:
            raise SystemExit(f"!! {n} 的 style 顺序与 full 不一致，格子对不上")
    print(f"三个配置的 style/content 顺序一致，{len(ref)} 个 style")

    gS = {n: Z[n]["style_gain"] for n in Z}          # 风格增益
    cN = {n: Z[n]["content_norm"] for n in Z}        # 归一化内容破坏

    out = {}
    out["inject_style"] = report(
        "注入对**风格增益**的贡献  ΔS(full) − ΔS(no_inject)",
        (gS["full"] - gS["no_inject"]).mean(1))
    out["adain_style"] = report(
        "AdaIN 对**风格增益**的贡献  ΔS(full) − ΔS(no_adain)",
        (gS["full"] - gS["no_adain"]).mean(1))
    out["inject_content"] = report(
        "注入造成的**内容破坏**  Cn(full) − Cn(no_inject)",
        (cN["full"] - cN["no_inject"]).mean(1))
    out["adain_content"] = report(
        "AdaIN 造成的**内容破坏**  Cn(full) − Cn(no_adain)",
        (cN["full"] - cN["no_adain"]).mean(1))

    # ★ 关键问题：两个部件的贡献是不是落在同一批 style 上？
    ds_i = (gS["full"] - gS["no_inject"]).mean(1)
    ds_a = (gS["full"] - gS["no_adain"]).mean(1)
    dc_a = (cN["full"] - cN["no_adain"]).mean(1)
    print(f"\n── 相关性")
    print(f"   注入贡献 vs AdaIN 贡献（风格）：r = "
          f"{np.corrcoef(ds_i, ds_a)[0,1]:+.3f}")
    print(f"   AdaIN 的风格收益 vs 它的内容代价：r = "
          f"{np.corrcoef(ds_a, dc_a)[0,1]:+.3f}")
    print(f"   → 后者接近 +1 说明 AdaIN 对每个 style 都在**同一条权衡曲线上**"
          f"移动（收益与代价同涨同落，换不来净值）；")
    print(f"     明显小于 1 才说明存在「代价小收益大」的 style —— 那才是能改的地方。")

    # AdaIN 的净效果：风格收益减去内容代价，逐 style
    net = ds_a - dc_a
    report("AdaIN 的**净**效果（风格收益 − 内容代价），>0 才是真的赚", net)

    Path(a.root, "decomp.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {Path(a.root)/'decomp.json'}")


if __name__ == "__main__":
    main()
