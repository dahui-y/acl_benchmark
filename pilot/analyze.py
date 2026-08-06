"""The three questions the pilot exists to answer.

**1. Does the model respond to `each` / `together` at all?**
Two ways of failing to find an effect look identical in a count table and are
completely different findings. If the model ignores the adverb, `dist` and
`coll` share a seed and will come out as nearly the same picture -- so pixel
distance between conditions separates "the model read the sentence and got the
number wrong" (publishable) from "the model did not read the sentence"
(a much thinner claim, and one about adverbs rather than about distributivity).
This is measurable only because the conditions share their initial noise.

**2. Do the anchors work?**
`explicit_n` and `explicit_1` state their counts, so they are known positives.
They give the ceiling: on items where the model cannot produce N objects when
told to, a `dist` failure is a counting failure and says nothing about
distributivity. The headline is conditioned on the anchor passing, and the
anchor is defined in advance.

**3. Do two detectors still agree once the images are generated?**
On LVIS photographs agreement was 0.887 and the agreed subset was right 98.1%
of the time. Photographs are not what the benchmark scores. This is the only
number that speaks to the domain shift, and it is why the pilot counts every
image with both detectors instead of just the one that owns each class.

Plus a screening pass over the families whose truth values may not be
single-valued. Those are reported, not scored -- they get cut or kept by eye.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

MAIN = "dist_coll"


def load(paths):
    rows = []
    for p in paths:
        rows += [r for r in map(json.loads, p.read_text().splitlines())
                 if r and r.get("status") == "ok"]
    return rows


def key(r):
    return (r["item_id"], r["condition"], r["seed"], r["noun"])


def by_condition(rows, noun):
    """(item, seed) -> {condition: count} for one noun."""
    out = defaultdict(dict)
    for r in rows:
        if r["noun"] == noun:
            out[(r["item_id"], r["seed"])][r["condition"]] = r["count"]
    return out


def q1_effect(rows, items):
    print("\n" + "=" * 70)
    print("1. does the model respond to `each` / `together`?")
    print("=" * 70)
    per_obj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        item = items[r["item_id"]]
        if item["family"] != MAIN or r["noun"] != item["object_class"]:
            continue
        per_obj[r["condition"]][item["subject_count"]].append(r["count"])

    conds = ["explicit_1", "coll", "bare", "dist", "explicit_n"]
    ns = sorted({n for c in per_obj.values() for n in c})
    print(f"\nmean object count produced   (entailed: explicit_1=1, coll=1, "
          f"dist=N, explicit_n=N)")
    print(f"{'condition':<12}" + "".join(f"{'N=' + str(n):>9}" for n in ns))
    for c in conds:
        if c not in per_obj:
            continue
        cells = []
        for n in ns:
            v = per_obj[c][n]
            cells.append(f"{sum(v) / len(v):>9.2f}" if v else f"{'-':>9}")
        print(f"{c:<12}" + "".join(cells))

    print("\nthe contrast that matters, per N:  mean(dist) - mean(coll)")
    for n in ns:
        d, c = per_obj["dist"][n], per_obj["coll"][n]
        if d and c:
            print(f"  N={n}: {sum(d) / len(d) - sum(c) / len(c):+.2f}"
                  f"   (entailed difference {n - 1:+d})")


def q1b_pixel(items, image_root, model):
    """Did the picture change at all when the adverb changed?

    Same seed across conditions means identical initial noise, so a model that
    ignores `each` returns a near-identical image. A count difference of zero
    with a pixel difference of zero is a different result from a count
    difference of zero with a large pixel difference.
    """
    try:
        import numpy as np              # noqa: PLC0415
        from PIL import Image           # noqa: PLC0415
    except ImportError:
        print("\n(skipping pixel comparison: numpy/PIL not available)")
        return
    root = image_root / model
    print("\n" + "-" * 70)
    print("did the IMAGE change when the adverb changed?  "
          "(shared seed => identical noise)")
    pairs = [("dist", "coll"), ("dist", "bare"), ("dist", "explicit_n")]
    acc = defaultdict(list)
    for item in items.values():
        if item["family"] != MAIN:
            continue
        for seed in item["seeds"]:
            folder = root / f"item{item['item_id']:04d}"
            for a, b in pairs:
                pa, pb = folder / f"{a}__seed{seed}.png", folder / f"{b}__seed{seed}.png"
                if pa.exists() and pb.exists():
                    ia = np.asarray(Image.open(pa).convert("RGB"), dtype=np.int16)
                    ib = np.asarray(Image.open(pb).convert("RGB"), dtype=np.int16)
                    acc[(a, b)].append(float(np.abs(ia - ib).mean()))
    print(f"\n{'pair':<22}{'n':>5}{'mean |Δpixel|':>15}{'identical':>11}")
    for (a, b), v in acc.items():
        ident = sum(1 for x in v if x < 0.5) / len(v)
        print(f"{a + ' vs ' + b:<22}{len(v):>5}{sum(v) / len(v):>15.2f}"
              f"{ident:>11.2f}")
    print("\n  0 means the adverb changed nothing -- a null effect on counts "
          "would then be\n  about the model ignoring the word, not about "
          "distributivity.")


def q2_anchors(rows, items):
    print("\n" + "=" * 70)
    print("2. the anchors -- known positives, so they set the ceiling")
    print("=" * 70)
    hit = defaultdict(lambda: [0, 0])
    for r in rows:
        item = items[r["item_id"]]
        if item["family"] != MAIN or r["entailed"] is None:
            continue
        which = f"{r['condition']}/{r['noun']}"
        hit[which][1] += 1
        hit[which][0] += (r["count"] == r["entailed"])
    print(f"\n{'condition / noun':<26}{'n':>5}{'exact':>9}")
    for k in sorted(hit):
        ok, tot = hit[k]
        print(f"{k:<26}{tot:>5}{ok / tot:>9.3f}")

    # The headline the paper would report: distributive accuracy restricted to
    # items where the model demonstrably CAN produce N objects when told to.
    obj = by_condition(rows, None)
    per = defaultdict(dict)
    for r in rows:
        item = items[r["item_id"]]
        if item["family"] == MAIN and r["noun"] == item["object_class"]:
            per[(r["item_id"], r["seed"])][r["condition"]] = (
                r["count"], r["entailed"])
    passed = [k for k, v in per.items()
              if "explicit_n" in v and v["explicit_n"][0] == v["explicit_n"][1]]
    if not passed:
        print("\n  no item passed explicit_n -- the model cannot produce the "
              "stated count at all,\n  so nothing here can separate a "
              "distributivity failure from a counting failure.")
        return
    for cond in ("dist", "coll"):
        vals = [per[k][cond] for k in passed if cond in per[k]]
        ok = sum(1 for c, e in vals if e is not None and c == e)
        print(f"\n  conditioned on explicit_n passing ({len(passed)}/{len(per)} "
              f"item-seeds):")
        print(f"    {cond}: {ok}/{len(vals)} = {ok / len(vals):.3f} produced "
              f"the entailed count")
    del obj


def q3_cross(paths, items):
    print("\n" + "=" * 70)
    print("3. do the two detectors still agree on GENERATED images?")
    print("=" * 70)
    if len(paths) < 2:
        print("\n  need counts from both detectors; pass two files")
        return
    tables = {}
    for p in paths:
        rows = load([p])
        if rows:
            tables[rows[0]["detector"]] = {key(r): r["count"] for r in rows}
    if len(tables) < 2:
        print("\n  both files came from the same detector")
        return
    (na, a), (nb, b) = tables.items()
    shared = sorted(set(a) & set(b))
    if not shared:
        print("\n  no overlapping (image, noun) cells")
        return
    exact = sum(a[k] == b[k] for k in shared) / len(shared)
    plural = sum((a[k] >= 2) == (b[k] >= 2) for k in shared) / len(shared)
    print(f"\n  {na} vs {nb}, {len(shared)} shared cells")
    print(f"    exact-count agreement : {exact:.3f}")
    print(f"    plurality agreement   : {plural:.3f}")
    print(f"\n  on LVIS photographs plurality agreement was 0.887 and the "
          f"agreed subset\n  was right 98.1% of the time. A large drop here is "
          f"the domain shift showing,\n  and it is the only place we can put a "
          f"number on it.")


def screening(rows, items):
    print("\n" + "=" * 70)
    print("4. screening families -- reported, not scored. cut or keep by eye")
    print("=" * 70)
    fams = defaultdict(list)
    for r in rows:
        item = items[r["item_id"]]
        if item["family"] != MAIN and r["noun"] == item["object_class"]:
            fams[item["family"]].append((item, r))
    for fam, rs in sorted(fams.items()):
        q = rs[0][0].get("screening_question", "")
        print(f"\n  {fam}  --  {q}")
        for item, r in sorted(rs, key=lambda x: (x[0]["item_id"], x[1]["seed"])):
            e = r["entailed"]
            mark = "" if e is None else ("  ok" if r["count"] == e else "  MISS")
            print(f"    item{item['item_id']:04d} seed{r['seed']}  "
                  f"counted {r['count']}, entailed {e}{mark}")
            if r["seed"] == item["seeds"][0]:
                print(f"      {item['conditions'][0]['prompt']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=Path, nargs="+", required=True)
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--images", type=Path,
                    default=Path(__file__).parent / "images")
    ap.add_argument("--model", default="sdxl")
    args = ap.parse_args()

    items = {r["item_id"]: r for r in
             (json.loads(l) for l in args.stimuli.read_text().splitlines() if l)}
    rows = load(args.counts)
    if not rows:
        raise SystemExit("no successful count rows")
    print(f"{len(rows)} counts over {len({r['item_id'] for r in rows})} items "
          f"from {', '.join(sorted({r['detector'] for r in rows}))}")

    q1_effect(rows, items)
    q1b_pixel(items, args.images, args.model)
    q2_anchors(rows, items)
    q3_cross(args.counts, items)
    screening(rows, items)


if __name__ == "__main__":
    main()
