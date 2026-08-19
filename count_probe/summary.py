#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把若干个配置的关键读数拉成**一张**横向对比表。零 GPU，只读已有的 CSV 和图。

    到第三轮已经有五个配置（原版 / v1 / v2 / v3-hinge / 只收 mask），
    逐个翻 yolo_eval + decompose 的九张表没法做决策。这里只留决策要用的那几列：

      · **vanilla 准确率** —— 各配置应当**完全相同**（同 prompt 同 seed，
        我们的改动只碰 counting pass）。不同就是泄漏，先查那个再看别的。
      · **方法准确率** 与相对 vanilla 的增益 —— 主判据。
      · **分两支的修正成功率** —— 原版的增益几乎全来自"删多余"（+40.0，p<0.001），
        新损失不能把它弄坏。只看总分会掩盖"补涨了、删跌了"。
      · **colourfulness / 平涂块占比** —— 计数涨、图塌了不算赢。
        v1 灰底矢量风、v2 白底剪影、v3 块状拼接，都得有个数。

用法：
    python count_probe/summary.py --arms $SD_OUT/count/*_arms
"""

import argparse
import csv
from math import comb
from pathlib import Path

from yolo_eval import _photo_stats


def _i(v):
    return None if v in ("", "None", None) else int(float(v))


def _load(d):
    p = Path(d) / "yolo_results.csv"
    if not p.exists():
        return None
    rows = []
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        r = {**r, "N": int(r["N"]), "n_dbscan": _i(r["n_dbscan"]),
             "ok_v": _i(r["ok_vanilla"]), "ok_m": _i(r.get("ok_countgen")),
             "y_v": _i(r["yolo_vanilla"]), "y_m": _i(r.get("yolo_countgen")),
             "match": r["obj_num_match"] in ("True", "true", "1")}
        if r["ok_v"] is None or r["ok_m"] is None:
            continue
        rows.append(r)
    return rows


def _acc(rs, k):
    return 100.0 * sum(r[k] for r in rs) / len(rs) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--no-quality", action="store_true", help="跳过要读图的那两列")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("A", "B"),
                    help="对这两个配置做**配对**检验。百分比之差看不出样本量："
                         "65.0%% vs 58.3%% 在 60 题上只是差 4 张")
    a = ap.parse_args()

    out = []
    for d in a.arms:
        d = Path(d)
        rows = _load(d)
        if not rows:
            print(f"（跳过 {d.name}：没有 yolo_results.csv 或没有可用行）")
            continue
        fixed = [r for r in rows if not r["match"]]
        over = [r for r in fixed if r["n_dbscan"] is not None and r["n_dbscan"] > r["N"]]
        under = [r for r in fixed if r["n_dbscan"] is not None and r["n_dbscan"] < r["N"]]
        resid = [(r["y_m"] - r["y_v"]) - (r["N"] - r["n_dbscan"])
                 for r in fixed if None not in (r["y_m"], r["y_v"], r["n_dbscan"])]
        rec = dict(name=d.name.replace("_arms", ""), n=len(rows),
                   van=_acc(rows, "ok_v"), met=_acc(rows, "ok_m"),
                   over=_acc(over, "ok_m"), n_over=len(over),
                   under=_acc(under, "ok_m"), n_under=len(under),
                   big=100.0 * sum(1 for x in resid if abs(x) >= 2) / len(resid)
                   if resid else float("nan"))
        if not a.no_quality:
            cv = cm = fv = fm = 0.0
            k = 0
            for r in rows:
                pv, pm = d / "vanilla" / r["file"], d / "countgen" / r["file"]
                if pv.exists() and pm.exists():
                    a1, b1 = _photo_stats(pv)
                    a2, b2 = _photo_stats(pm)
                    cv += a1; cm += a2; fv += b1; fm += b2; k += 1
            if k:
                rec.update(col_v=cv / k, col_m=cm / k, flat_v=fv / k, flat_m=fm / k)
        out.append(rec)

    if not out:
        raise SystemExit("!! 一个都没读到")

    # ---- 一致性：vanilla 必须处处相同 ----
    vs = {round(r["van"], 3) for r in out}
    print(f"vanilla 准确率：{sorted(vs)}"
          + ("   ✓ 各配置一致" if len(vs) == 1 else
             "   ⚠️ **不一致** —— 我们的改动本不该碰 vanilla，先查泄漏再看别的"))

    print(f"\n{'配置':<14}{'题数':>5}{'方法':>8}{'相对vanilla':>11}"
          f"{'删多余(n)':>14}{'补缺失(n)':>14}{'|残差|≥2':>10}")
    for r in sorted(out, key=lambda x: -x["met"]):
        print(f"{r['name']:<14}{r['n']:>5}{r['met']:>7.1f}%{r['met']-r['van']:>+10.1f}"
              f"{r['over']:>9.1f}% ({r['n_over']:>2}){r['under']:>9.1f}% ({r['n_under']:>2})"
              f"{r['big']:>9.1f}%")
    print("\n参照【一手，我们自己跑的】：原版在评测集 167 题上 56.9%（vanilla 40.1%），"
          "\n  删多余 49.2%（p<0.001）/ 补缺失 27.3%（p=0.774），|残差|≥2 44.9%")

    if not a.no_quality and "col_v" in out[0]:
        print(f"\n{'配置':<14}{'colourfulness':>22}{'平涂块占比':>18}")
        print(f"{'':14}{'vanilla → 方法（变化）':>22}{'vanilla → 方法':>18}")
        for r in sorted(out, key=lambda x: -x.get("col_m", 0)):
            d = 100.0 * (r["col_m"] - r["col_v"]) / r["col_v"] if r["col_v"] else 0
            print(f"{r['name']:<14}{r['col_v']:>8.1f} →{r['col_m']:>7.1f}"
                  f"{d:>+6.0f}%{r['flat_v']:>10.1%} →{r['flat_m']:>6.1%}"
                  + ("   ⚠️ 塌了" if d < -20 else ""))
        print("\n  colourfulness 掉得多 = 图从照片塌向平涂/黑白。这两列是代理，"
              "不是标准指标；\n  但计数涨、图塌了不算赢。")

    if a.pair:
        paired(*a.pair)


def _mcnemar(b, c):
    """McNemar 精确二项检验，双尾。b+c 小的时候它检不出东西，这正是要看见的。"""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def paired(da, db):
    """两个配置逐题配对比较。同一批 prompt、同一批 seed，可以直接配对。"""
    na, nb = Path(da).name.replace("_arms", ""), Path(db).name.replace("_arms", "")
    la, lb = _load(da), _load(db)
    if not la or not lb:
        print(f"\n（配对检验跳过：{na if not la else nb} 读不到 yolo_results.csv）")
        return
    ra = {r["file"]: r for r in la}
    rb = {r["file"]: r for r in lb}
    common = sorted(set(ra) & set(rb))
    if len(common) < len(ra) or len(common) < len(rb):
        print(f"\n⚠️ 两边题目对不齐：{na} {len(ra)} 题、{nb} {len(rb)} 题，"
              f"公共 {len(common)} 题。只在公共部分上配对。")
    a_only = sum(1 for f in common if ra[f]["ok_m"] and not rb[f]["ok_m"])
    b_only = sum(1 for f in common if rb[f]["ok_m"] and not ra[f]["ok_m"])
    same = len(common) - a_only - b_only
    print(f"\n{'='*60}\n配对检验：{na}  vs  {nb}（共 {len(common)} 题）")
    print(f"  两者都对或都错          {same:>4}")
    print(f"  只有 {na:<14} 对   {a_only:>4}")
    print(f"  只有 {nb:<14} 对   {b_only:>4}")
    if a_only + b_only == 0:
        print("  两个配置**逐题完全一致**，没有可检验的不一致对。"
              "\n  → 这不是「打平」，是**同一套输出**：先去查两边的改动是不是真的生效了。")
        return
    p = _mcnemar(b_only, a_only)
    print(f"  净差 {a_only - b_only:+d} 张，双尾 p = {p:.3f}")
    if p > 0.05:
        print(f"  → **不显著**。差的这 {abs(a_only-b_only)} 张在这个样本量下"
              f"与掷硬币无法区分，不能当结论用。")
    else:
        print("  → 显著（在这个样本量下能与掷硬币区分开）。")
    print(f"  （注意：这是**调参集**上的数。真正的判据是拿定下来的配置"
          f"去评测集 167 题上跑一遍，那才是没被调过参的。）")


if __name__ == "__main__":
    main()
