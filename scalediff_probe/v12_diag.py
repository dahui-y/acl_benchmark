"""v1.2 判后诊断：门开了、delta 没动，到底是"机制无效"还是"没开在该开的行上"。

事实（2026-08-14）：
    v1   门 0/31 开（过渡带 100%、背景 0%）   delta +0.10
    v1.2 门 20/31 开（过渡带 40%、背景 38%）  delta +0.10
    两臂**平均权重都 ≈0.5**（v1 全图挤在 0.5；v1.2 是 38/38/24 的三分），
    即平均介入强度几乎没变，只有空间结构变了 -> 结果没变。

两个互斥解释，本脚本用已落盘的数据分辨（零 GPU）：

  H1 空间结构不起作用。机制的收益来自"平均削弱主体条件"这个**空间均匀**
     的量，逐位置只是装饰。若成立，我们的核心主张（唯一空间自适应的
     低成本实例化）不被自己的数据支持。
     指纹：残留失败**均匀散布**在开门行与半开行之间；v1 与 v1.2 的
     失败行**大体重合**。

  H2 门没开在该开的行上。11 条仍半开的行里恰好压着那些没修掉的样本，
     delta 不动是因为需要它动的地方它没动。
     指纹：残留失败**集中在半开行**；开门行几乎全部清零。

  （H3 地板效应：+0.10 只剩约 3 个单位的余量，n=31 分辨不出 3 与 2 的差别。
    这不是 H1/H2 的替代解释，而是对**任何**结论的置信度上限 —— 一并报出。）

    python scalediff_probe/v12_diag.py
    python scalediff_probe/v12_diag.py --v12 $SD_OUT/parti_v12 --v1 $SD_OUT/parti_v1
"""

import argparse
import json
import os
from pathlib import Path


def load_delta(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return {}
    out = {}
    for ln in p.open():
        r = json.loads(ln)
        if r.get("delta") is not None:
            out[int(r["idx"])] = r
    return out


def load_band(d):
    """逐行门控图统计：(主体>0.5, 背景<0.3, 过渡带)。"""
    import torch
    out = {}
    for p in sorted(Path(d).glob("*_mask.pt")):
        m = torch.load(p, map_location="cpu").float().squeeze()
        out[int(p.name[:5])] = (float((m > 0.5).float().mean()),
                                float((m < 0.3).float().mean()),
                                float(((m >= 0.3) & (m <= 0.7)).float().mean()),
                                float(m.mean()))
    return out


def fisher_2x2(a, b, c, d):
    """双侧 Fisher 精确检验。n=31 这种量级不能用卡方。"""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(x):
        return (comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
                if 0 <= x <= r1 and 0 <= c1 - x <= n - r1 else 0.0)
    p0 = p(a)
    return sum(p(x) for x in range(0, min(r1, c1) + 1) if p(x) <= p0 + 1e-12)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--v12", default=str(root / "parti_v12"))
    ap.add_argument("--v1", default=str(root / "parti_v1"))
    ap.add_argument("--hi", default=str(root / "parti_hi"))
    ap.add_argument("--open-band", type=float, default=0.5)
    ap.add_argument("--open-bg", type=float, default=0.1)
    a = ap.parse_args()

    d12, d1, dhi = load_delta(a.v12), load_delta(a.v1), load_delta(a.hi)
    band = load_band(a.v12)
    ids = sorted(d12)
    if not ids:
        raise SystemExit(f"{a.v12}/vlm_delta.jsonl 里没有可用行")

    # ---------- 逐行总表 ----------
    print(f"{'idx':>5} {'门':>4} {'过渡':>5} {'背景':>5} {'均值':>5} "
          f"{'基线':>5} {'v1':>4} {'v1.2':>5}  prompt")
    n_open = n_open_fail = n_half = n_half_fail = 0
    for i in ids:
        b = band.get(i)
        opened = (b is not None and b[2] < a.open_band and b[1] > a.open_bg)
        fail = d12[i]["delta"] >= 1
        if b is not None:
            if opened:
                n_open += 1
                n_open_fail += fail
            else:
                n_half += 1
                n_half_fail += fail
        print(f"{i:>5} {'开' if opened else '半':>4} "
              f"{b[2]:>5.0%} {b[1]:>5.0%} {b[3]:>5.2f} " if b else
              f"{i:>5} {'?':>4} {'-':>5} {'-':>5} {'-':>5} ", end="")
        print(f"{dhi.get(i,{}).get('delta','-'):>5} "
              f"{d1.get(i,{}).get('delta','-'):>4} "
              f"{d12[i]['delta']:>5}"
              f"{'  ← 未清' if fail else ''}"
              f"{'  ← 倒退' if d12[i]['delta'] <= -1 else ''}"
              f"  {d12[i]['prompt'][:42]}")

    # ---------- H1 vs H2 ----------
    print(f"\n{'='*66}\nH1（空间无用） vs H2（门没开在该开的行上）")
    print(f"{'':>12}{'未清 delta>=1':>14}{'已清':>8}{'失败率':>9}")
    print(f"{'门真开着':>12}{n_open_fail:>14}{n_open-n_open_fail:>8}"
          f"{n_open_fail/max(n_open,1):>9.0%}")
    print(f"{'仍半开':>12}{n_half_fail:>14}{n_half-n_half_fail:>8}"
          f"{n_half_fail/max(n_half,1):>9.0%}")
    p = fisher_2x2(n_open_fail, n_open - n_open_fail,
                   n_half_fail, n_half - n_half_fail)
    print(f"\nFisher 精确检验 双侧 p = {p:.3f}")
    if p < 0.05 and n_half_fail / max(n_half, 1) > n_open_fail / max(n_open, 1):
        print("  -> **失败显著集中在半开行 = H2**：机制是空间的，"
              "只是没开在需要它的行上。下一步把那些行的门打开。")
    else:
        print("  -> 失败**没有**集中在半开行。H2 不被支持，H1 抬头："
              "开门与否与修没修好无关 -> 收益可能来自空间均匀的平均削弱。")
        print("     判决实验：--uniform 消融（同均值、无空间结构）。")

    # ---------- v1 vs v1.2 逐行 ----------
    both = [i for i in ids if i in d1]
    if both:
        same = sum(1 for i in both if d1[i]["delta"] == d12[i]["delta"])
        f1 = {i for i in both if d1[i]["delta"] >= 1}
        f2 = {i for i in both if d12[i]["delta"] >= 1}
        print(f"\n{'='*66}\nv1 vs v1.2 逐行（n={len(both)}）")
        print(f"  逐行完全相同 {same}/{len(both)}")
        print(f"  v1 未清 {sorted(f1)}")
        print(f"  v1.2 未清 {sorted(f2)}")
        print(f"  两臂都未清 {sorted(f1 & f2)}   "
              f"v1.2 新修好 {sorted(f1 - f2)}   v1.2 新弄坏 {sorted(f2 - f1)}")
        if f1 == f2:
            print("  -> **失败行完全一致**：换门没有改变任何一行的结局，"
                  "强烈指向 H1（或该残留与门无关）。")

    # ---------- R3 + 地板效应 ----------
    reg12 = [i for i in ids if d12[i]["delta"] <= -1]
    reg1 = [i for i in both if d1[i]["delta"] <= -1] if both else []
    print(f"\n{'='*66}\nR3 主体误伤：v1.2 有 {len(reg12)} 行 delta<=-1 {reg12}"
          f"（v1 是 {len(reg1)} 行 {reg1}）")
    tot = sum(max(d12[i]["delta"], 0) for i in ids)
    base_tot = sum(max(dhi[i]["delta"], 0) for i in ids if i in dhi)
    print(f"\n地板效应（任何结论的置信上限）：")
    print(f"  基线残留正超额合计 {base_tot} 个实例 / {len(ids)} 行")
    print(f"  v1.2 残留正超额合计 {tot} 个实例 —— 只剩这么多余量。")
    print(f"  n={len(ids)} 时想再分辨 {tot} 与 {max(tot-1,0)} 的差别，"
          f"统计上做不到。**任何后续配置比较都必须扩样本，否则是在噪声里选优。**")


if __name__ == "__main__":
    main()
