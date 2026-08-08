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

    # 完整性核查
    print("完整性核查 —— 未干预的行应当逐字节相同")
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

    hdr = f"\n{'idx':<5}{'cat':<10}{'head':<13}{'cov':>7}{'base d':>8}{'new d':>7}{'Δ':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i in idxs:
        r, q = A[i], B[i]
        m = mb.get(i, {})
        cov = m.get("cov", -1)
        cs = f"{cov:6.1%}" if cov is not None and cov >= 0 else "     -"
        mark = ""
        if not m.get("applied", True):
            mark = "  (未干预)"
        elif q["delta"] < r["delta"]:
            mark = "  好转"
        elif q["delta"] > r["delta"]:
            mark = "  变差"
        print(f"{i:<5}{r['cat']:<10}{str(m.get('head')):<13}{cs}"
              f"{r['delta']:>+8}{q['delta']:>+7}{q['delta'] - r['delta']:>+6}{mark}")

    print(f"\n{'cat':<12}{'base':>16}{'new':>16}")
    print("-" * 44)
    for cat in sorted({A[i]["cat"] for i in idxs}):
        ii = [i for i in idxs if A[i]["cat"] == cat]
        da = [A[i]["delta"] for i in ii]
        db = [B[i]["delta"] for i in ii]
        print(f"{cat:<12}"
              f"{sum(da) / len(da):>+9.2f} {sum(1 for x in da if x > 0)}/{len(da):<5}"
              f"{sum(db) / len(db):>+9.2f} {sum(1 for x in db if x > 0)}/{len(db):<5}")
    da = [A[i]["delta"] for i in idxs]
    db = [B[i]["delta"] for i in idxs]
    print(f"{'全部':<11}"
          f"{sum(da) / len(da):>+9.2f} {sum(1 for x in da if x > 0)}/{len(da):<5}"
          f"{sum(db) / len(db):>+9.2f} {sum(1 for x in db if x > 0)}/{len(db):<5}")

    # 主角有没有被一起抹掉：基图计数应当保持
    lost = [i for i in idxs
            if B[i]["counts"].get("1024", B[i]["counts"].get(1024, 0))
            != A[i]["counts"].get("1024", A[i]["counts"].get(1024, 0))]
    print(f"\n基图计数变化的行: {lost if lost else '无'}"
          "   （基图在干预之前生成，本就该完全一致）")


if __name__ == "__main__":
    main()
