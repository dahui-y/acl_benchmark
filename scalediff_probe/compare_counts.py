"""把干预前后的两张计数表逐行对起来。

两边的 manifest 是同 seed、同 RNG 钉法、同 prompt 顺序，所以可以逐 idx 配对。

同时做一项完整性核查：empty 那五条 head=None，走的是还原后的原版 attn2，
**应当与基线逐字节相同**。如果不同，说明批跑管线本身引入了漂移，
其余所有差值都不可归因给干预 —— 这一列不过，下面的表就不用看了。

    python scalediff_probe/compare_counts.py \
        --base $SD_OUT/batch --new $SD_OUT/method_batch_s1
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def load(d):
    """键必须是 (idx, seed)。只用 idx 的话，多 seed 时同一个 idx 的三条会互相覆盖，
    **只剩最后一个 seed，而且不报错** —— 静默给出错误结论。踩过。"""
    p = Path(d) / "counts.json"
    if not p.exists():
        raise SystemExit(f"没有 {p}，先跑 count_objects.py --batch {d}")
    return {(r["idx"], r["seed"]): r for r in json.loads(p.read_text())}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def files_by_idx(d):
    out = {}
    mp = Path(d) / "manifest.jsonl"
    if mp.exists():
        for line in mp.open():
            r = json.loads(line)
            out[(r["idx"], r["seed"])] = r
    return out


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    ma, mb = files_by_idx(a.base), files_by_idx(a.new)
    keys = sorted(set(A) & set(B))
    seeds = sorted({k[1] for k in keys})
    if len(keys) < max(len(A), len(B)):
        print(f"注意: 只有 {len(keys)} 条两边都有")
    print(f"seed: {seeds}   共 {len(keys)} 对\n")
    # 逐行明细只打第一个 seed，跨 seed 的结论看后面的汇总
    idxs = [k for k in keys if k[1] == seeds[0]]

    # 基图哈希 —— 决定性的一步。
    # 基图在放大之前生成，两边同 seed 同 RNG，本该逐字节相同。
    #   基图相同而计数不同 -> 两张 counts.json 是不同版本的检测器算的
    #   基图不同           -> phase 1 的数值路径把基图也改了，A/B 不成立
    print("基图（1024²）逐字节核查 —— 决定下面的表能不能读")
    diff_base, same_base = [], []
    for i in keys:
        fa, fb = ma.get(i, {}).get("files", {}), mb.get(i, {}).get("files", {})
        if "1024" not in fa or "1024" not in fb:
            continue
        ha, hb = md5(Path(a.base) / fa["1024"]), md5(Path(a.new) / fb["1024"])
        (same_base if ha == hb else diff_base).append(i)
    print(f"  相同 {len(same_base)} / 不同 {len(diff_base)}")
    if diff_base:
        print(f"  不同的行: {diff_base}")
        print("  -> phase 1 的数值路径改了基图。整个 A/B 不可归因，必须重跑。")
    else:
        print("  -> 基图全同。基图计数若仍有差，那是两张 counts.json 版本不一致，"
              "重新数一遍基线即可。")

    print("\n完整性核查 —— 未干预的行应当逐字节相同")
    bad = same = 0
    for i in keys:
        applied = mb.get(i, {}).get("applied", True)
        if applied:
            continue
        fa, fb = ma.get(i, {}).get("files", {}), mb.get(i, {}).get("files", {})
        k = str(max(int(x) for x in fb)) if fb else None
        ha = md5(Path(a.base) / fa[k]) if k and k in fa else None
        hb = md5(Path(a.new) / fb[k]) if k else None
        ok = ha is not None and ha == hb
        same += ok
        bad += not ok
        if not ok:
            print(f"  [{i:02d}] 不一致  base={ha}  new={hb}")
    print(f"  相同 {same} / 不一致 {bad}"
          + ("   -> 管线无漂移" if bad == 0 else "   -> 下面的差值不可归因"))

    # 主看 excess（相对 prompt 基数，无尺度偏差），delta 只作参考
    hdr = (f"\n[明细：seed {seeds[0]}]"
           f"\n{'idx':<5}{'cat':<10}{'head':<13}{'cov':>7}"
           f"{'base e':>8}{'new e':>7}{'Δe':>5}   {'base d':>7}{'new d':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i in idxs:
        r, q = A[i], B[i]
        m = mb.get(i, {})
        cov = m.get("cov", -1)
        cs = f"{cov:6.1%}" if cov is not None and cov >= 0 else "     -"
        re_, qe = r.get("excess"), q.get("excess")
        mark = ""
        if not m.get("applied", True):
            mark = "  (未干预)"
        elif re_ is None or qe is None:
            mark = "  (无基数)"
        # 目标是 |excess| = 0，不是 excess 最小。
        # 23_hands 的 card=2，excess -1 -> -2 是"两只手都被删了"，
        # 按符号比会被标成好转 —— 那是最严重的回归之一。
        elif abs(qe) < abs(re_):
            mark = "  好转"
        elif abs(qe) > abs(re_):
            mark = "  变差"
        es = ("       -      -    -" if re_ is None or qe is None
              else f"{re_:>+8}{qe:>+7}{qe - re_:>+5}")
        print(f"{i[0]:<5}{r['cat']:<10}{str(m.get('head')):<13}{cs}{es}"
              f"   {r['delta']:>+7}{q['delta']:>+7}{mark}")

    import statistics as st

    def cell(vals):
        return f"{sum(vals)/len(vals):.2f}" if vals else "  -"

    def mae(D, cat, seed):
        v = [abs(D[k]["excess"]) for k in keys
             if k[1] == seed and D[k]["cat"] == cat and D[k].get("excess") is not None]
        return v

    cats = sorted({A[k]["cat"] for k in keys})
    print(f"\n主指标 MAE = 平均 |4096 计数 − prompt 基数|   （每个 seed 一列）")
    hd = f"{'cat':<11}" + "".join(f"{'s'+str(sd):>15}" for sd in seeds) + f"{'mean±std':>18}"
    print(hd); print("-" * len(hd))
    for cat in cats:
        cells, ba_all, ne_all = [], [], []
        for sd in seeds:
            va, vb = mae(A, cat, sd), mae(B, cat, sd)
            if not va:
                cells.append("       -")
                continue
            cells.append(f"{cell(va)}→{cell(vb)}")
            ba_all.append(sum(va)/len(va)); ne_all.append(sum(vb)/len(vb))
        if not ba_all:
            continue
        sa = st.stdev(ba_all) if len(ba_all) > 1 else 0.0
        sb = st.stdev(ne_all) if len(ne_all) > 1 else 0.0
        print(f"{cat:<11}" + "".join(f"{c:>15}" for c in cells)
              + f"   {sum(ba_all)/len(ba_all):.2f}±{sa:.2f}→"
                f"{sum(ne_all)/len(ne_all):.2f}±{sb:.2f}")

    allv = lambda D, sd: [abs(D[k]["excess"]) for k in keys
                          if k[1] == sd and D[k].get("excess") is not None]
    ba_all = [sum(allv(A, sd))/len(allv(A, sd)) for sd in seeds if allv(A, sd)]
    ne_all = [sum(allv(B, sd))/len(allv(B, sd)) for sd in seeds if allv(B, sd)]
    if ba_all:
        sa = st.stdev(ba_all) if len(ba_all) > 1 else 0.0
        sb = st.stdev(ne_all) if len(ne_all) > 1 else 0.0
        print(f"{'全部':<10}" + "".join(
            f"{f'{a_:.2f}→{b_:.2f}':>15}" for a_, b_ in zip(ba_all, ne_all))
            + f"   {sum(ba_all)/len(ba_all):.2f}±{sa:.2f}→"
              f"{sum(ne_all)/len(ne_all):.2f}±{sb:.2f}")

    # 跑之前定死的判据
    print("\n判据（跑前定死）：")
    lone_a = [sum(mae(A, 'lone', sd))/len(mae(A, 'lone', sd))
              for sd in seeds if mae(A, 'lone', sd)]
    lone_b = [sum(mae(B, 'lone', sd))/len(mae(B, 'lone', sd))
              for sd in seeds if mae(B, 'lone', sd)]
    if lone_a:
        ok = all(b < a for a, b in zip(lone_a, lone_b))
        print(f"  ① lone 在【每一个】seed 上都改善: "
              + "  ".join(f"s{sd} {a:.2f}→{b:.2f}"
                          for sd, a, b in zip(seeds, lone_a, lone_b))
              + ("   -> 过，主结果坐实" if ok else "   -> **没过**"))
    for cat in ("portrait", "structure"):
        va = [sum(mae(A, cat, sd))/len(mae(A, cat, sd))
              for sd in seeds if mae(A, cat, sd)]
        vb = [sum(mae(B, cat, sd))/len(mae(B, cat, sd))
              for sd in seeds if mae(B, cat, sd)]
        if not va:
            continue
        worse = sum(1 for a_, b_ in zip(va, vb) if b_ > a_)
        print(f"  ② {cat} 尾部回归出现在 {worse}/{len(va)} 个 seed  "
              + ("-> 真回归，加适用性门" if worse == len(va)
                 else "-> 噪声，v1 直接定稿" if worse <= 1 else "-> 不确定"))

    # 主角有没有被一起抹掉：基图计数应当保持
    lost = [i for i in keys
            if B[i]["counts"].get("1024", B[i]["counts"].get(1024, 0))
            != A[i]["counts"].get("1024", A[i]["counts"].get(1024, 0))]
    print(f"\n基图计数变化的行: {lost if lost else '无'}"
          "   （基图在干预之前生成，本就该完全一致）")


if __name__ == "__main__":
    main()
