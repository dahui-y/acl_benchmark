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
    """(gold, predicted) at a confidence threshold.

    An MLLM verifier returns the integer itself, so its rows carry `count` and
    the threshold does not apply -- sweeping one over them would report the same
    number ten times and invite the reader to think it had been tuned.
    """
    return [(r["gold"], r["count"] if "count" in r else
             sum(1 for s in r["scores"] if s >= thr)) for r in rows]


def sweep(rows, thresholds):
    """Pick the operating threshold by N-vs-1 separation, not exact match.

    Separation is the declared primary criterion, and the two do not peak at the
    same place: exact match rewards getting the absolute number right, while the
    benchmark only ever needs the distributive reading's count to come out above
    the collective one's. Choosing by exact match would hand the route a worse
    operating point than the one it actually runs at.
    """
    print(f"{'thr':>6}{'exact':>8}{'±1':>8}{'kappa':>9}{'kappa_w':>9}"
          f"{'mean err':>10}{'N-vs-1':>9}")
    best = None
    for t in thresholds:
        pairs = counts_at(rows, t)
        exact = sum(a == b for a, b in pairs) / len(pairs)
        near = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
        bias = sum(b - a for a, b in pairs) / len(pairs)
        sep = discrimination(rows, t, quiet=True)
        print(f"{t:>6.2f}{exact:>8.3f}{near:>8.3f}{kappa(pairs):>9.3f}"
              f"{kappa(pairs, True):>9.3f}{bias:>+10.2f}{sep:>9.3f}")
        if best is None or sep > best[1]:
            best = (t, sep)
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


def discrimination(rows, thr, quiet=False, classes=None, bands=None):
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
        if classes and r["class"] not in classes:
            continue
        by_class[r["class"]][g].append(p)

    if not quiet:
        print(f"\nN-vs-1 separation at thr={thr:.2f}  "
              f"(ordered / tied / reversed, ties count against us)")
        print(f"{'N':>3}{'pairs':>8}{'ordered':>10}{'tied':>8}{'reversed':>10}")
    totals = defaultdict(lambda: [0, 0, 0])
    for n in sorted(bands or range(2, 7)):
        ok = tie = rev = 0
        for counts in by_class.values():
            for hi in counts.get(n, []):
                for lo in counts.get(1, []):
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
        if not quiet:
            print(f"{n:>3}{tot:>8}{ok / tot:>10.3f}{tie / tot:>8.3f}"
                  f"{rev / tot:>10.3f}")
    grand = [sum(v[i] for v in totals.values()) for i in range(3)]
    tot = sum(grand)
    if tot and not quiet:
        print(f"{'all':>3}{tot:>8}{grand[0] / tot:>10.3f}"
              f"{grand[1] / tot:>8.3f}{grand[2] / tot:>10.3f}")
    return grand[0] / tot if tot else float("nan")


def per_class_screen(rows, thr, bands, floor):
    """Which classes separate N from 1 well enough to be usable.

    Post-hoc, and labelled as such wherever it is reported. The pre-registered
    criterion is the pooled number; this only says what a screened suite would
    look like, and a screen chosen on the same data it is evaluated on is
    optimistic by construction. Whether the classes it keeps survive on a fresh
    sample is a separate question that this cannot answer.
    """
    names = sorted({r["class"] for r in rows})
    kept = []
    print(f"\nper-class N-vs-1 at thr={thr:.2f}, N in {sorted(bands)}"
          f"  (POST-HOC screen, floor {floor:.2f})")
    for name in names:
        sep = discrimination(rows, thr, quiet=True, classes={name}, bands=bands)
        if sep == sep:  # not nan
            mark = "keep" if sep >= floor else ""
            print(f"  {name:<15}{sep:>7.3f}  {mark}")
            if sep >= floor:
                kept.append(name)
    pooled = discrimination(rows, thr, quiet=True, classes=set(kept),
                            bands=bands)
    print(f"\n  {len(kept)}/{len(names)} classes kept; pooled separation over "
          f"the kept set: {pooled:.3f}  (POST-HOC)")
    return kept, pooled


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


def plurality(rows, thr, iters=2000, seed=0):
    """The binary question: more than one, or exactly one?

    Exact counting is the wrong question to put to a detector, and asking it was
    my mistake. The contrast the benchmark is built on -- "the three girls are
    each holding a balloon" against "...holding a balloon together" -- is at
    bottom plural against singular. Three is entailed, but what separates the
    two readings is one versus more-than-one, and that is a question detectors
    answer far better than "exactly how many".

    This keeps the property the direction was chosen for: both answers are known
    in advance from the semantics, so the headline stays an absolute rate
    against programmatic truth rather than a difference needing an interval to
    survive. What it gives up is resolution -- a model that draws two balloons
    for three girls passes -- and that belongs in the limitations, not in a
    footnote.

    NOT pre-registered. The criteria in README.md were written for exact counts;
    this is a different measurement and is labelled as such wherever it appears.
    """
    import random  # noqa: PLC0415
    print(f"\nplurality (>1 vs =1) -- NOT the pre-registered measurement")
    print(f"{'thr':>6}{'acc':>8}{'kappa':>9}{'FP 1->many':>12}{'FN many->1':>12}")
    best = None
    for t in thr:
        pr = [(r["gold"] >= 2, sum(1 for s in r["scores"] if s >= t) >= 2)
              for r in rows]
        acc = sum(a == b for a, b in pr) / len(pr)
        ones = [p for p in pr if not p[0]]
        many = [p for p in pr if p[0]]
        fp = sum(b for _, b in ones) / len(ones) if ones else float("nan")
        fn = sum(not b for _, b in many) / len(many) if many else float("nan")
        print(f"{t:>6.2f}{acc:>8.3f}{kappa(pr):>9.3f}{fp:>12.3f}{fn:>12.3f}")
        # Pick the balanced point, not the most accurate one: the two error
        # directions land on different conditions of the suite, so a threshold
        # that trades collective-condition errors for distributive ones would
        # bias the comparison the benchmark exists to make.
        if best is None or abs(fp - fn) < best[1]:
            best = (t, abs(fp - fn))
    t = best[0]
    pr = [(r["gold"] >= 2, sum(1 for s in r["scores"] if s >= t) >= 2)
          for r in rows]
    rng = random.Random(seed)
    bs = sorted(sum(a == b for a, b in d) / len(d) for d in
                ([pr[rng.randrange(len(pr))] for _ in pr] for _ in range(iters)))
    acc = sum(a == b for a, b in pr) / len(pr)
    print(f"\n  balanced threshold {t:.2f}: acc {acc:.3f} "
          f"95% CI [{bs[int(0.025 * iters)]:.3f}, {bs[int(0.975 * iters)]:.3f}], "
          f"kappa {kappa(pr):.3f}")
    print(f"  {'gold N':>8}{'n':>6}{'correct':>10}")
    for n in sorted({r["gold"] for r in rows}):
        sub = [r for r in rows if r["gold"] == n]
        ok = sum((sum(1 for s in r["scores"] if s >= t) >= 2) == (n >= 2)
                 for r in sub)
        print(f"  {n:>8}{len(sub):>6}{ok / len(sub):>10.3f}")
    return t, acc


def bootstrap(rows, thr, bands=None, iters=2000, seed=0):
    """Cluster bootstrap over CELLS for the separation rate and for kappa.

    Resampling pairs would badly understate the uncertainty: each cell enters
    many pairs, so the 624 pairs come from 415 cells and are nowhere near
    independent. Cells are the unit that was sampled from COCO, so cells are the
    unit resampled here.

    Worth the trouble because the pre-registered cut-off sits within a hundredth
    of the point estimate, and a decision that close should not be made on a
    number without an interval around it.
    """
    import random  # noqa: PLC0415
    rng = random.Random(seed)
    sep, kap, exact = [], [], []
    for _ in range(iters):
        draw = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        s = discrimination(draw, thr, quiet=True, bands=bands)
        if s == s:
            sep.append(s)
        pairs = counts_at(draw, thr)
        kap.append(kappa(pairs, True))
        exact.append(sum(a == b for a, b in pairs) / len(pairs))

    def ci(xs):
        xs = sorted(x for x in xs if x == x)
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs))]

    band_txt = f"N in {sorted(bands)}" if bands else "N in 2..5"
    print(f"\nbootstrap over cells at thr={thr:.2f}, {band_txt}, "
          f"{iters} resamples")
    for name, xs, point in (("N-vs-1 separation", sep,
                             discrimination(rows, thr, quiet=True, bands=bands)),
                            ("kappa (weighted)", kap,
                             kappa(counts_at(rows, thr), True)),
                            ("exact match", exact,
                             sum(a == b for a, b in counts_at(rows, thr))
                             / len(rows))):
        lo, hi = ci(xs)
        print(f"  {name:<20}{point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    return ci(sep)


def held_out_screen(rows, thr, bands, floor):
    """Screen classes on one half of the cells, score separation on the other.

    The in-sample screen reports 1.000 and means nothing: with a handful of
    pairs per class, picking the classes that scored well and then scoring those
    same pairs measures only that the selection worked on the data it selected
    from. Splitting first gives a number that can actually be believed -- the
    classes are chosen without ever seeing the cells they are judged on.

    The split alternates within each (class, count) group, so both halves keep
    the same class and count composition and neither ends up with a class's easy
    images only.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["class"], r["gold"])].append(r)
    a, b = [], []
    for key in sorted(groups):
        for i, r in enumerate(sorted(groups[key], key=lambda x: x["image_id"])):
            (a if i % 2 == 0 else b).append(r)

    names = sorted({r["class"] for r in rows})
    kept = [n for n in names
            if (discrimination(a, thr, quiet=True, classes={n}, bands=bands)
                or 0) >= floor]
    scored = discrimination(b, thr, quiet=True, classes=set(kept), bands=bands)
    unscreened = discrimination(b, thr, quiet=True, bands=bands)
    print(f"\nheld-out screen at thr={thr:.2f}, N in {sorted(bands)}, "
          f"floor {floor:.2f}")
    print(f"  classes chosen on half A: {len(kept)}/{len(names)}")
    print(f"  separation on half B, screened  : {scored:.3f}")
    print(f"  separation on half B, unscreened: {unscreened:.3f}")
    return kept, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--thr", type=float, default=None,
                    help="report at this threshold instead of the sweep's best")
    ap.add_argument("--screen", action="store_true",
                    help="add the post-hoc per-class screen")
    ap.add_argument("--screen-bands", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--screen-floor", type=float, default=0.95)
    args = ap.parse_args()

    for path in args.pred:
        all_rows = [r for r in map(json.loads, path.read_text().splitlines())
                    if r]
        rows = [r for r in all_rows if r.get("status") == "ok"]
        failed = len(all_rows) - len(rows)
        if failed:
            print(f"\n{path.name}: {failed} of {len(all_rows)} cells errored "
                  f"and are excluded -- an unparseable count is not a zero")
        if not rows:
            print(f"\n{path}: no successful rows")
            continue
        print(f"\n{'=' * 66}\n{path.name}  --  {len(rows)} cells, "
              f"{len({r['class'] for r in rows})} classes\n{'=' * 66}")
        best = sweep(rows, args.thresholds)
        thr = args.thr if args.thr is not None else best
        print(f"\noperating threshold: {thr:.2f}"
              + ("" if args.thr is not None else "  (best N-vs-1 separation)"))
        confusion(rows, thr)
        by_band(rows, thr, "gold", "gold count")
        by_band(rows, thr, "class", "class", min_n=5)
        discrimination(rows, thr)
        bootstrap(rows, thr)
        bootstrap(rows, thr, bands={3, 4, 5})
        plurality(rows, args.thresholds)
        if args.screen:
            per_class_screen(rows, thr, set(args.screen_bands),
                             args.screen_floor)
            held_out_screen(rows, thr, set(args.screen_bands),
                            args.screen_floor)


if __name__ == "__main__":
    main()
