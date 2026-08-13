"""Parti 的类别画像 + 触发集定稿（CPU，几秒）。

三件事一次做完：

**(1) 最后一个未判的预注册预测**（写在 `fetch_parti.py` 里、跑之前定死）：
    Outdoor Scenes / World Knowledge 的 evf **高于** Artifacts /
    Food & Beverage（商品图式类别）。
    过 -> "触发构图存在于真实用例、只是不存在于评测语料"有了正面证据；
    不过 -> 该失效只在手工 prompt 里可见，第一根柱子按降档口径写。

**(2) 退化样本筛除。** `p100 = 1.00` 意味着存在"有框、但 16 块全判空"的
    样本 —— 只可能是框小到占不满任何一块的 5%（微小误检）。它们不是触发
    构图，**烧 4096 之前必须剔掉**。判据：最大框面积占比 < 1% 视为退化。

**(3) R 扫描的严重度合成。** 律说重复 ≈ R²×(1−覆盖率)，所以真正该看的
    不是 evf 本身而是 **R² × evf**：它把"视野被切成多少块"和"多少块是空的"
    两个因子乘起来。这一列是第二篇（更高分辨率）立意的定量依据。

    python scalediff_probe/parti_profile.py
    python scalediff_probe/parti_profile.py --tau 0.5 --write   # 写触发集
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "parti_base"))
    ap.add_argument("--items", default=str(root / "parti.json"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.01,
                    help="最大框面积占比低于此 -> 退化样本（微小误检），剔除")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    items = json.loads(Path(a.items).read_text())["items"]
    cands = sorted(Path(a.base).glob("trigger_all_*.jsonl"))
    if not cands:
        sys.exit(f"没有 {a.base}/trigger_all_*.jsonl")
    src = max(cands, key=lambda c: sum(1 for _ in c.open()))
    rows = [json.loads(l) for l in src.open()]
    print(f"用 {src.name}   {len(rows)} 条")

    mean = lambda v: sum(v) / max(len(v), 1)
    for r in rows:
        it = items[r["idx"]] if r["idx"] < len(items) else {}
        r["cat"] = it.get("category", "?")
        bb, wh = r.get("boxes") or [], r.get("wh") or [1024, 1024]
        r["maxarea"] = max(((x[2]-x[0])*(x[3]-x[1])/(wh[0]*wh[1]) for x in bb),
                           default=0.0)
    defined = [r for r in rows if r["nbox"] >= 1]

    # ---- (1) 类别画像 ----
    by = defaultdict(list)
    undef = defaultdict(int)
    for r in rows:
        if r["nbox"] >= 1:
            by[r["cat"]].append(r)
        else:
            undef[r["cat"]] += 1

    print(f"\n{'类别':<20}{'n':>5}{'nbox=0':>8}{'evf均值':>9}"
          f"{f'>{a.tau}':>7}{'占比':>7}")
    print("-" * 58)
    order = sorted(by, key=lambda c: -mean([r["evf"] for r in by[c]]))
    for c in order:
        g = by[c]
        k = sum(1 for r in g if r["evf"] > a.tau)
        print(f"{c:<20}{len(g):>5}{undef[c]:>8}"
              f"{mean([r['evf'] for r in g]):>9.3f}{k:>7}{k/len(g):>7.1%}")

    HI = ["Outdoor Scenes", "World Knowledge"]
    LO = ["Artifacts", "Food & Beverage"]
    hv = [r["evf"] for c in HI for r in by.get(c, [])]
    lv = [r["evf"] for c in LO for r in by.get(c, [])]
    if hv and lv:
        ok = mean(hv) > mean(lv)
        print(f"\n预注册预测：{'/'.join(HI)} ({mean(hv):.3f}) > "
              f"{'/'.join(LO)} ({mean(lv):.3f})  -> "
              + ("**过**" if ok else "**没过**"))
        print("  过 -> 触发构图存在于真实用例、只是不在评测语料里（正面证据）；"
              "\n  没过 -> 该失效只在手工 prompt 里可见，第一根柱子降档写。")

    # ---- (2) 退化样本 ----
    trig = [r for r in defined if r["evf"] > a.tau]
    degen = [r for r in trig if r["maxarea"] < a.min_area]
    clean = [r for r in trig if r["maxarea"] >= a.min_area]
    print(f"\nτ>{a.tau} 的 {len(trig)} 条中：**退化 {len(degen)} 条**"
          f"（最大框面积 < {a.min_area:.0%}，微小误检）-> 剩 {len(clean)} 条可用")
    if degen:
        print("  退化样例：")
        for r in degen[:4]:
            print(f"    evf={r['evf']:.2f} nbox={r['nbox']} "
                  f"maxarea={r['maxarea']:.4f}  {r['prompt'][:52]}")
    print("  可用样例（这些才是触发构图，进 4096 之前仍要看图）：")
    for r in sorted(clean, key=lambda r: -r["evf"])[:8]:
        print(f"    evf={r['evf']:.2f} nbox={r['nbox']} "
              f"maxarea={r['maxarea']:.3f} [{r['cat'][:14]}] {r['prompt'][:44]}")

    # ---- (3) R 扫描 × 严重度 ----
    print(f"\n{'R':>3}{'分辨率':>9}{'evf均值':>9}{'R²×evf':>9}"
          f"{'触发率':>8}   <- 律：重复 ∝ R²×(1−覆盖率)")
    print("-" * 46)
    for R in (2, 4, 8):
        v = ([r["evf"] for r in defined] if R == 4 else
             [r["evf_R"][str(R)] for r in defined if r.get("evf_R")])
        if not v:
            continue
        k = sum(1 for x in v if x > a.tau)
        print(f"{R:>3}{1024*R:>8}²{mean(v):>9.3f}{R*R*mean(v):>9.2f}"
              f"{k/len(v):>8.1%}")
    print("  R²×evf 是严重度的合成量 —— 第二篇（更高分辨率）立意的定量依据。")

    if a.write:
        p = Path(a.base) / f"trigger_clean_tau{a.tau:g}.json"
        p.write_text(json.dumps({
            "base": str(a.base), "tau": a.tau, "min_area": a.min_area,
            "n_defined": len(defined), "n_trigger_raw": len(trig),
            "n_degenerate": len(degen), "n_clean": len(clean),
            "note": "退化 = 最大框面积占比 < min_area（微小误检，非触发构图）",
            "alive_idx": sorted(r["idx"] for r in clean)},
            ensure_ascii=False, indent=1))
        print(f"\n触发集写入 {p}（{len(clean)} 条，键名沿用 alive_idx）")
    else:
        print("\n（加 --write 写出触发集）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
