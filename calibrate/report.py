"""Turn stored detections into the calibration report.

This is the step that can kill the direction. The benchmark's whole plan is to
let a detector supply the ground truth that we are not paying annotators for, so
the detector's own error rate is the floor under every number the paper would
report. If it cannot count 2-5 large, unoccluded objects in real photographs
whose counts are already known, nothing downstream is worth building.

Four things are reported, in increasing order of how much they matter:

  1. exact-count accuracy and Cohen's kappa   -- comparable to GenEval's 0.823
  2. accuracy by count band                   -- N is the difficulty axis
  3. accuracy by class                        -- which objects are usable
  4. the discrimination the benchmark runs on -- "N vs 1", which is the actual
     decision behind "each ... a balloon" (3) versus "... together" (1)

(4) is the one that decides. A detector that is merely biased -- consistently
reporting 4 where the truth is 5 -- still separates 3 from 1 perfectly, and the
benchmark would survive that. A detector whose counts are noisy AROUND the truth
does not, however well its average looks.

Kappa is reported rather than raw agreement because the cells are not uniform
over counts, and raw agreement on a skewed set is flattering for reasons that
have nothing to do with the detector.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def kappa(pairs, weighted=False):
    """Cohen's kappa over integer categories; quadratic weights if asked.

    Unweighted treats 5-vs-1 as no worse than 5-vs-4, which for counts is wrong;
    the weighted form is the honest one for an ordinal scale. Both are printed
    because the published number we are comparing against is unweighted.
    """
    cats = sorted({v for p in pairs for v in p})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return float("nan")
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    n = len(pairs)
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if not weighted:
            return 0.0 if i == j else 1.0
        return ((cats[i] - cats[j]) / (cats[-1] - cats[0])) ** 2

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k)) / n ** 2
    return 1 - po / pe if pe else float("nan")


def counts_at(rows, thr):
    """(gold, predicted) at a confidence threshold."""
    return [(r["gold"], sum(1 for s in r["scores"] if s >= thr)) for r in rows]


def sweep(rows, thresholds):
    print(f"{'thr':>6}{'exact':>8}{'±1':>8}{'kappa':>9}{'kappa_w':>9}"
          f"{'mean err':>10}")
    best = None
    for t in thresholds:
        pairs = counts_at(rows, t)
        exact = sum(a == b for a, b in pairs) / len(pairs)
        near = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
        bias = sum(b - a for a, b in pairs) / len(pairs)
        print(f"{t:>6.2f}{exact:>8.3f}{near:>8.3f}{kappa(pairs):>9.3f}"
              f"{kappa(pairs, True):>9.3f}{bias:>+10.2f}")
        if best is None or exact > best[1]:
            best = (t, exact)
    return best[0]


def by_band(rows, thr, key, label, min_n=1):
    groups = defaultdict(list)
    for r, (g, p) in zip(rows, counts_at(rows, thr)):
        groups[r[key]].append((g, p))
    print(f"\n{label:<15}{'n':>5}{'exact':>8}{'±1':>8}{'mean pred':>11}")
    for k in sorted(groups, key=lambda x: (isinstance(x, str), x)):
        pairs = groups[k]
        if len(pairs) < min_n:
            continue
        exact = sum(a == b for a, b in pairs) / len(pairs)
        near = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
        mean = sum(b for _, b in pairs) / len(pairs)
        print(f"{str(k):<15}{len(pairs):>5}{exact:>8.3f}{near:>8.3f}{mean:>11.2f}")


def discrimination(rows, thr):
    """The decision the benchmark actually makes: is this image's count N or 1?

    Every distributive item pairs an entailed count of N against a collective
    reading whose entailed count is 1. So what has to be reliable is not the
    absolute count but the separation between the two. Pair each N>=2 cell with
    each N=1 cell of the same class and ask whether the predicted counts order
    them correctly; ties count as failures, because a tie is exactly the case
    where the benchmark cannot tell the readings apart.
    """
    by_class = defaultdict(lambda: defaultdict(list))
    for r, (g, p) in zip(rows, counts_at(rows, thr)):
        by_class[r["class"]][g].append(p)

    print(f"\nN-vs-1 separation at thr={thr:.2f}  "
          f"(ordered / tied / reversed, ties count against us)")
    print(f"{'N':>3}{'pairs':>8}{'ordered':>10}{'tied':>8}{'reversed':>10}")
    totals = defaultdict(lambda: [0, 0, 0])
    for n in range(2, 7):
        ok = tie = rev = 0
        for cls, bands in by_class.items():
            for hi in bands.get(n, []):
                for lo in bands.get(1, []):
                    if hi > lo:
                        ok += 1
                    elif hi == lo:
                        tie += 1
                    else:
                        rev += 1
        tot = ok + tie + rev
        if not tot:
            continue
        totals[n] = [ok, tie, rev]
        print(f"{n:>3}{tot:>8}{ok / tot:>10.3f}{tie / tot:>8.3f}{rev / tot:>10.3f}")
    grand = [sum(v[i] for v in totals.values()) for i in range(3)]
    tot = sum(grand)
    if tot:
        print(f"{'all':>3}{tot:>8}{grand[0] / tot:>10.3f}"
              f"{grand[1] / tot:>8.3f}{grand[2] / tot:>10.3f}")
    return grand[0] / tot if tot else float("nan")


def confusion(rows, thr, cap=6):
    pairs = counts_at(rows, thr)
    print(f"\nconfusion at thr={thr:.2f}  (rows = gold, cols = predicted, "
          f"{cap}+ collapsed)")
    m = defaultdict(lambda: defaultdict(int))
    for g, p in pairs:
        m[min(g, cap)][min(p, cap)] += 1
    cols = sorted({c for row in m.values() for c in row})
    print("      " + "".join(f"{c:>6}" for c in cols))
    for g in sorted(m):
        print(f"{g:>4}: " + "".join(f"{m[g][c]:>6}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--thr", type=float, default=None,
                    help="report at this threshold instead of the sweep's best")
    args = ap.parse_args()

    for path in args.pred:
        rows = [r for r in map(json.loads, path.read_text().splitlines())
                if r.get("status") == "ok"]
        if not rows:
            print(f"\n{path}: no successful rows")
            continue
        print(f"\n{'=' * 66}\n{path.name}  --  {len(rows)} cells, "
              f"{len({r['class'] for r in rows})} classes\n{'=' * 66}")
        best = sweep(rows, args.thresholds)
        thr = args.thr if args.thr is not None else best
        print(f"\noperating threshold: {thr:.2f}"
              + ("" if args.thr is not None else "  (best exact-match)"))
        confusion(rows, thr)
        by_band(rows, thr, "gold", "gold count")
        by_band(rows, thr, "class", "class", min_n=5)
        discrimination(rows, thr)


if __name__ == "__main__":
    main()
