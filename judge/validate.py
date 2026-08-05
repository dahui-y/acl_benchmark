"""Establish the judge's reliability without a single human label.

The trick is that some of our conditions have answers that follow from what the
sentences mean. `paraphrase_min` swaps one preposition; `filler` appends a
clause about the footage. Neither says anything different about the object, so
whatever is true of `prog` is true of them. Nobody has to be asked.

That gives an agreement rate -- but a raw agreement rate confounds two things:

    disagreement(prog, paraphrase_min) = judge noise + video-model instability

The pilot already showed the second term is not zero: one preposition visibly
moved the scene. So the judge is run twice over the same videos, and the
test-retest disagreement isolates the first term:

    judge noise            = disagreement(pass 1, pass 2) on identical videos
    model instability      = control disagreement - judge noise

Both numbers are reportable results. The first is the validity section that
human evaluation would otherwise have to provide; the second is a finding about
the models, and it sets the floor every aspect effect has to clear.

`MUST_DIFFER` guards the other direction: a judge that answers "yes" to
everything would score perfectly on agreement. `other_verb` shows a different
action, so the item's target state cannot have been reached.

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


def load(path):
    """(item, condition, seed, pass) -> {question: answer}, ok rows only."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        key = (r["item_id"], r["condition"], r["seed"], r["pass"])
        out[key] = {q: r[q]["answer"] for q in QUESTION_IDS if q in r}
    return out


def rate(matches, total):
    return f"{matches}/{total} = {matches / total:.3f}" if total else "n/a"


def test_retest(judged):
    """Same video, two passes. Anything that differs here is the judge alone."""
    per_q = defaultdict(lambda: [0, 0])
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
    return per_q


def pairwise(judged, cond_a, cond_b, judge_pass=1):
    """Agreement between two conditions of the same item and seed."""
    per_q = defaultdict(lambda: [0, 0])
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
    return per_q


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
    if not judged:
        raise SystemExit(f"no successful judgments in {args.judgments}")
    passes = sorted({k[3] for k in judged})
    print(f"{len(judged)} judgments, passes {passes}\n")

    print("=" * 66)
    print("1. judge noise -- same video judged twice")
    print("=" * 66)
    retest = test_retest(judged)
    if not retest:
        print("  not measured: run judge.py with --repeat 2")
        noise = {}
    else:
        noise = {}
        for q in QUESTION_IDS:
            m, t = retest[q]
            noise[q] = 1 - m / t if t else None
            print(f"  {q:8s} agreement {rate(m, t)}")
        print("\n  This is the reliability number human evaluation would "
              "otherwise supply.")

    print()
    print("=" * 66)
    print("2. control conditions -- must agree, by what the sentences mean")
    print("=" * 66)
    for cond_a, cond_b, why in MUST_AGREE:
        per_q = pairwise(judged, cond_a, cond_b)
        if not per_q:
            print(f"  {cond_a} vs {cond_b}: no overlapping items")
            continue
        print(f"  {cond_a} vs {cond_b}  ({why})")
        for q in QUESTION_IDS:
            m, t = per_q[q]
            if not t:
                continue
            line = f"    {q:8s} agreement {rate(m, t)}"
            if noise.get(q) is not None:
                # Disagreement above the judge's own noise is the video model
                # answering a meaning-preserving edit differently.
                instability = (1 - m / t) - noise[q]
                line += f"   -> model instability {max(instability, 0):.3f}"
            print(line)

    print()
    print("=" * 66)
    print("3. discrimination -- must differ, or the judge just says yes")
    print("=" * 66)
    for cond_a, cond_b, question, why in MUST_DIFFER:
        per_q = pairwise(judged, cond_a, cond_b)
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
