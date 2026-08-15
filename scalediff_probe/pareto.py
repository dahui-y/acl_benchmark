"""判决实验：我们那七次"机制改进"，是不是同一个强度旋钮？

**背景（2026-08-14）**：v1 / v1.2 / v2a2 三个配置在 Rep⁺–Dmg⁻ 上完美
单调互换，而**总误差反而随强度上升**：

    v1.2  Rep+ 0.226  Dmg- 0.032   总 0.258   <- 最弱、总误差最低
    v1    Rep+ 0.194  Dmg- 0.097   总 0.291
    v2a2  Rep+ 0.161  Dmg- 0.161   总 0.322   <- 最强、总误差最高

这是**强度旋钮**的指纹，不是三种机制的指纹。若属实，七次尝试（v0 /
v1 / v1.1 / v1.2 / v2 / v2b / v2a）等于同一件事的不同刻度，方法线到顶。

结构性理由：空视野区域里结构引导几乎不提供信息，**文本是唯一的条件
信号**，所以只动文本条件的方法只能在"重复"与"空洞/损伤"之间滑动 ——
这个旋钮只有一个自由度。

**本脚本做的事**：用同一选层（top4）下的强度扫描 s=0.7 / 1.0 / 1.3
画出基准曲线，再看 v2a2（s=1.0 + 推拉）落在曲线上还是下方。

**预注册判据（写在看到数字之前，不许改口）**：
    v2a2 的 Dmg⁻ 比曲线在同一 Rep⁺ 处的插值**低 0.03 以上** -> 机制为真，
        帕累托改进成立，方法线继续；
    落在 ±0.03 带内             -> **同一个旋钮**，方法线当天停掉，转向；
    高于曲线 0.03 以上           -> 推拉比纯调强度更差，同样停掉。
容差 0.03 的由来：31 条上 1 个实例 = 0.032，即**一个样本的分辨力**。
小于这个的差别我们本来就分辨不了（地板效应，见 §3.14d）。

    python scalediff_probe/pareto.py
"""

import argparse
import json
import os
from pathlib import Path

TOL = 0.03          # = 1 个实例 / 31 条，样本分辨力下限


def load(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    rows = [r for r in rows if r.get("delta") is not None]
    if not rows:
        return None
    n = len(rows)
    pos = sum(max(r["delta"], 0) for r in rows)
    neg = sum(max(-r["delta"], 0) for r in rows)
    return {"n": n, "rep": pos / n, "dmg": neg / n,
            "pos": pos, "neg": neg, "ids": {r["idx"] for r in rows}}


def interp(curve, x):
    """在 (rep, dmg) 折线上按 rep 线性插值出 dmg。curve 按 rep 升序。"""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", nargs="*",
                    default=["s=0.7:parti_s07", "s=1.0:parti_v12",
                             "s=1.3:parti_s13"],
                    help="强度曲线的点，格式 名字:目录")
    ap.add_argument("--test", default="v2a2 push:parti_v2a2",
                    help="待判点，格式 名字:目录")
    ap.add_argument("--extra", nargs="*",
                    default=["baseline:parti_hi", "v1 (all-layers):parti_v1",
                             "AccDiffusion:parti_acc"],
                    help="只列出、不参与判决的臂")
    a = ap.parse_args()

    def grab(spec):
        nm, d = spec.split(":", 1)
        return nm, load(d if "/" in d else str(root / d))

    pts, missing = [], []
    for s in a.curve:
        nm, r = grab(s)
        (pts if r else missing).append((nm, r) if r else nm)
    tname, tres = grab(a.test)
    if not tres:
        missing.append(tname)
    if missing:
        raise SystemExit(
            f"缺少 vlm_delta.jsonl：{missing}\n"
            f"先跑：python scalediff_probe/vlm_count.py --delta --hi <目录>")

    # ---------- 样本一致性：曲线各点必须是同一批 idx，否则不可比 ----------
    idsets = [r["ids"] for _, r in pts] + [tres["ids"]]
    common = set.intersection(*idsets)
    if any(len(s) != len(common) for s in idsets):
        print(f"⚠ 各臂 idx 不完全一致（交集 {len(common)}），"
              f"下面的数按各自全量算 —— 严格比较应先对齐\n")

    def row(nm, r):
        tot = r["rep"] + r["dmg"]
        pn = f"{r['pos']}/{r['neg']}"
        print(f"{nm:<20}{r['n']:>4}{r['rep']:>11.3f}{r['dmg']:>11.3f}"
              f"{tot:>9.3f}{pn:>9}")

    print(f"{'配置':<20}{'n':>4}{'Rep+ 重复':>11}{'Dmg- 误伤':>11}"
          f"{'总误差':>9}{'正/负':>9}")
    print("-" * 66)
    for s_ in a.extra:
        nm, r = grab(s_)
        if r:
            row(nm, r)
    print("-" * 66)
    for nm, r in pts:
        row(nm + " (曲线)", r)
    row(tname + " (待判)", tres)

    # ---------- 判决 ----------
    curve = sorted([(r["rep"], r["dmg"]) for _, r in pts])
    exp = interp(curve, tres["rep"])
    diff = tres["dmg"] - exp
    print(f"\n{'='*66}")
    print(f"强度曲线（按 Rep+ 升序）：" +
          "  ".join(f"({x:.3f}, {y:.3f})" for x, y in curve))
    print(f"曲线在 Rep+={tres['rep']:.3f} 处的 Dmg- 插值 = {exp:.3f}")
    print(f"{tname} 实测 Dmg- = {tres['dmg']:.3f}   差 = {diff:+.3f}"
          f"   （容差 ±{TOL}，= 31 条上 1 个实例）")
    print(f"\n预注册判决：")
    if diff < -TOL:
        print(f"  ✅ **落在曲线下方** —— 同样的重复抑制、更少的误伤。")
        print(f"     机制为真，帕累托改进成立。方法线继续，这张图进论文。")
    elif diff > TOL:
        print(f"  ❌ **落在曲线上方** —— 推拉比单纯调强度还差。")
        print(f"     方法线停掉，转向。")
    else:
        print(f"  ❌ **落在曲线上（±{TOL} 带内）** —— 推拉与调强度**不可区分**。")
        print(f"     七次尝试是同一个旋钮的不同刻度，**方法线到顶，今天停掉，转向**。")
        print(f"     这句结论本身可写进新论文的消融：只动文本条件的干预，")
        print(f"     在空视野区域只有一个自由度。")

    # ---------- 附：总误差最优点 ----------
    allp = [(nm, r) for nm, r in pts] + [(tname, tres)]
    best = min(allp, key=lambda t: t[1]["rep"] + t[1]["dmg"])
    print(f"\n总误差最优：{best[0]}  {best[1]['rep']+best[1]['dmg']:.3f}"
          f"（Rep+ {best[1]['rep']:.3f} / Dmg- {best[1]['dmg']:.3f}）")
    print("  若最优点出现在**最弱**的配置上，那是'加强干预净收益为负'的直接证据。")


if __name__ == "__main__":
    main()
