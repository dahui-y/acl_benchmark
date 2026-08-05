"""What can be established about this evaluator with no human labels, and what cannot.

Four numbers, and they are not all the same kind of number:

  1. RELIABILITY   the same video judged twice. Pure judge property.
  2. SPECIFICITY   videos judged against a target state they cannot have
                   reached. The answer is 'no' by construction, so this is a
                   real accuracy measurement -- with no labels, because the
                   ground truth comes from having chosen the wrong verb.
  3. INSTABILITY   control conditions that mean the same thing. NOT a judge
                   check: those are different videos, and the pilot showed one
                   preposition visibly moving the scene, so disagreement may be
                   the judge being right. Once judge noise is subtracted this
                   measures the VIDEO MODEL, and it sets the floor every aspect
                   effect has to clear.
  4. ACTION RATE   the first-level result: is the event rendered at all.

  SENSITIVITY is missing, and cannot be had this way. Nothing in the design
  certifies that a given video does reach its target state, so there is no
  source of known positives. What follows is that the defensible claims are
  about differences between conditions under a judge of known reliability and
  specificity -- not about absolute achievement rates. A manually verified
  subsample is the cheapest way to close that gap; it is a paragraph of work,
  not an annotation effort.

Usage:
    python validate.py --judgments /data/frames/wan2.2-ti2v-5b-480p/judgments.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from criteria import MUST_AGREE, MUST_DIFFER, QUESTIONS  # noqa: E402

QUESTION_IDS = tuple(QUESTIONS)


def load(path, probe="target"):
    """(item, condition, seed, pass) -> {question: answer}, ok rows only."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("probe", "target") != probe:
            continue
        key = (r["item_id"], r["condition"], r["seed"], r["pass"])
        out[key] = {q: r[q]["answer"] for q in QUESTION_IDS if q in r}
    return out


def specificity(negatives):
    """Known negatives: every one of these should answer 'no' on `final`."""
    correct = total = unclear = 0
    for answers in negatives.values():
        if "final" not in answers:
            continue
        total += 1
        correct += answers["final"] == "no"
        unclear += answers["final"] == "unclear"
    return correct, unclear, total


def rate(matches, total):
    return f"{matches}/{total} = {matches / total:.3f}" if total else "n/a"


def kappa(pairs):
    """Cohen's kappa over the (a, b) answer pairs.

    Raw agreement is not reportable on its own here. Several of these questions
    have skewed answer distributions -- most videos are not in the target state
    -- and two raters who both answer "no" nine times in ten agree 82% of the
    time by chance alone. Kappa divides that out; a reviewer will ask for it,
    and rightly."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    expected = sum((sum(a == c for a, _ in pairs) / n) *
                   (sum(b == c for _, b in pairs) / n) for c in labels)
    if expected >= 1.0:            # one label used throughout; kappa undefined
        return None
    return (observed - expected) / (1 - expected)


def fmt(matches, total, pairs):
    k = kappa(pairs)
    return rate(matches, total) + (f"  kappa {k:.3f}" if k is not None
                                   else "  kappa n/a")


def test_retest(judged):
    """Same video, two passes. Anything that differs here is the judge alone."""
    per_q = defaultdict(lambda: [0, 0])
    pairs = defaultdict(list)
    for (item, cond, seed, p) in list(judged):
        if p != 1:
            continue
        other = judged.get((item, cond, seed, 2))
        if other is None:
            continue
        first = judged[(item, cond, seed, 1)]
        for q in QUESTION_IDS:
            if q in first and q in other:
                per_q[q][1] += 1
                per_q[q][0] += first[q] == other[q]
                pairs[q].append((first[q], other[q]))
    return per_q, pairs


def pairwise(judged, cond_a, cond_b, judge_pass=1):
    """Agreement between two conditions of the same item and seed."""
    per_q = defaultdict(lambda: [0, 0])
    pairs = defaultdict(list)
    for (item, cond, seed, p), answers in judged.items():
        if cond != cond_a or p != judge_pass:
            continue
        other = judged.get((item, cond_b, seed, p))
        if other is None:
            continue
        for q in QUESTION_IDS:
            if q in answers and q in other:
                per_q[q][1] += 1
                per_q[q][0] += answers[q] == other[q]
                pairs[q].append((answers[q], other[q]))
    return per_q, pairs


def action_rate(judged, judge_pass=1, condition="prog"):
    """The first-level result: how often the model renders the event at all."""
    yes = total = 0
    for (_item, cond, _seed, p), answers in judged.items():
        if cond != condition or p != judge_pass or "action" not in answers:
            continue
        total += 1
        yes += answers["action"] == "yes"
    return yes, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, required=True)
    args = ap.parse_args()

    judged = load(args.judgments)
    negatives = load(args.judgments, probe="known_negative")
    if not judged:
        raise SystemExit(f"no successful judgments in {args.judgments}")
    passes = sorted({k[3] for k in judged})
    print(f"{len(judged)} judgments, passes {passes}, "
          f"{len(negatives)} known-negative probes\n")

    print("=" * 66)
    print("1. judge noise -- same video judged twice")
    print("=" * 66)
    retest, retest_pairs = test_retest(judged)
    if not retest:
        print("  not measured: run judge.py with --repeat 2")
        noise = {}
    else:
        noise = {}
        for q in QUESTION_IDS:
            m, t = retest[q]
            noise[q] = 1 - m / t if t else None
            print(f"  {q:8s} agreement {fmt(m, t, retest_pairs[q])}")
        print("\n  This is the reliability number human evaluation would "
              "otherwise supply. Report kappa, not raw agreement:\n"
              "  these answers are skewed, and two raters who both say 'no'\n"
              "  nine times in ten agree 82% of the time by chance.")

    print()
    print("=" * 66)
    print("2. control conditions -- must agree, by what the sentences mean")
    print("=" * 66)
    for cond_a, cond_b, why in MUST_AGREE:
        per_q, pairs = pairwise(judged, cond_a, cond_b)
        if not per_q:
            print(f"  {cond_a} vs {cond_b}: no overlapping items")
            continue
        print(f"  {cond_a} vs {cond_b}  ({why})")
        for q in QUESTION_IDS:
            m, t = per_q[q]
            if not t:
                continue
            line = f"    {q:8s} agreement {fmt(m, t, pairs[q])}"
            if noise.get(q) is not None:
                # Disagreement above the judge's own noise is the video model
                # answering a meaning-preserving edit differently.
                instability = (1 - m / t) - noise[q]
                line += f"   -> model instability {max(instability, 0):.3f}"
            print(line)

    print()
    print("=" * 66)
    print("2b. specificity -- videos judged against a state they cannot reach")
    print("=" * 66)
    if not negatives:
        print("  not measured: run judge.py with --known-negative")
    else:
        correct, unclear, total = specificity(negatives)
        print(f"  answered 'no' on final: {rate(correct, total)}"
              f"   ({unclear} 'unclear')")
        print("  Ground truth here is known in advance -- the video shows a "
              "different action.\n  This is the one accuracy number obtainable "
              "without labels. Sensitivity\n  is not: there are no known "
              "positives, so absolute achievement rates stay\n  uncalibrated "
              "and only between-condition differences are safe to claim.")

    print()
    print("=" * 66)
    print("3. discrimination -- must differ, or the judge just says yes")
    print("=" * 66)
    for cond_a, cond_b, question, why in MUST_DIFFER:
        per_q, _ = pairwise(judged, cond_a, cond_b)
        m, t = per_q[question]
        if not t:
            print(f"  {cond_a} vs {cond_b}: no overlapping items")
            continue
        print(f"  {cond_a} vs {cond_b} on {question}: differ {rate(t - m, t)}"
              f"  ({why})")

    print()
    print("=" * 66)
    print("4. first-level result -- does the model render the event at all")
    print("=" * 66)
    yes, total = action_rate(judged)
    print(f"  action executed in `prog`: {rate(yes, total)}")
    print("  Items answered 'no' here carry no information about culmination:")
    print("  with no process there is nothing for an endpoint to be the end of.")


if __name__ == "__main__":
    main()
