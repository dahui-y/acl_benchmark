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
    p = Path(d) / "counts.json"
    if not p.exists():
        raise SystemExit(f"没有 {p}，先跑 count_objects.py --batch {d}")
    return {r["idx"]: r for r in json.loads(p.read_text())}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def files_by_idx(d):
    out = {}
    mp = Path(d) / "manifest.jsonl"
    if mp.exists():
        for line in mp.open():
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    ma, mb = files_by_idx(a.base), files_by_idx(a.new)
    idxs = sorted(set(A) & set(B))
    if len(idxs) < max(len(A), len(B)):
        print(f"注意: 只有 {len(idxs)} 条两边都有\n")

    # 基图哈希 —— 决定性的一步。
    # 基图在放大之前生成，两边同 seed 同 RNG，本该逐字节相同。
    #   基图相同而计数不同 -> 两张 counts.json 是不同版本的检测器算的
    #   基图不同           -> phase 1 的数值路径把基图也改了，A/B 不成立
    print("基图（1024²）逐字节核查 —— 决定下面的表能不能读")
    diff_base, same_base = [], []
    for i in idxs:
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
    for i in idxs:
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
    hdr = (f"\n{'idx':<5}{'cat':<10}{'head':<13}{'cov':>7}"
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
        elif qe < re_:
            mark = "  好转"
        elif qe > re_:
            mark = "  变差"
        es = ("       -      -    -" if re_ is None or qe is None
              else f"{re_:>+8}{qe:>+7}{qe - re_:>+5}")
        print(f"{i:<5}{r['cat']:<10}{str(m.get('head')):<13}{cs}{es}"
              f"   {r['delta']:>+7}{q['delta']:>+7}{mark}")

    def agg(title, key, pick):
        print(f"\n{title}\n{'cat':<12}{'base':>16}{'new':>16}")
        print("-" * 44)
        for cat in sorted({A[i]["cat"] for i in idxs}):
            ii = [i for i in idxs if A[i]["cat"] == cat and pick(i)]
            if not ii:
                continue
            da = [A[i][key] for i in ii]
            db = [B[i][key] for i in ii]
            print(f"{cat:<12}"
                  f"{sum(da)/len(da):>+9.2f} {sum(1 for x in da if x > 0)}/{len(da):<5}"
                  f"{sum(db)/len(db):>+9.2f} {sum(1 for x in db if x > 0)}/{len(db):<5}")
        ii = [i for i in idxs if pick(i)]
        da = [A[i][key] for i in ii]
        db = [B[i][key] for i in ii]
        print(f"{'全部':<11}"
              f"{sum(da)/len(da):>+9.2f} {sum(1 for x in da if x > 0)}/{len(da):<5}"
              f"{sum(db)/len(db):>+9.2f} {sum(1 for x in db if x > 0)}/{len(db):<5}")

    agg("主指标 excess = 4096 计数 - prompt 基数（无尺度偏差）", "excess",
        lambda i: A[i].get("excess") is not None and B[i].get("excess") is not None)
    agg("参考 delta = 4096 计数 - 基图计数（仍带约 +2 的尺度偏移）", "delta",
        lambda i: True)

    # 主角有没有被一起抹掉：基图计数应当保持
    lost = [i for i in idxs
            if B[i]["counts"].get("1024", B[i]["counts"].get(1024, 0))
            != A[i]["counts"].get("1024", A[i]["counts"].get(1024, 0))]
    print(f"\n基图计数变化的行: {lost if lost else '无'}"
          "   （基图在干预之前生成，本就该完全一致）")


if __name__ == "__main__":
    main()
