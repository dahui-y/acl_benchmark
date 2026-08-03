"""Build the aspect-controlled stimulus set for the text-encoder probe.

Each item fixes an event (verb, object, subject, scene) and varies only how the
language encodes culmination. Two controls bracket the comparison:

  paraphrase  same meaning, same aspect, different surface form  -> lower bound
  other_verb  different event, same aspect                       -> upper bound

Any cosine distance between aspect conditions has to be read against those two.
A distance no larger than the paraphrase control means aspect is invisible to
the encoder; a distance approaching the other-verb control means it is as
salient as swapping the event itself.

Usage:
    python build_stimuli.py --taxonomy ../action_object_taxonomy --out stimuli.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from lexicon import (
    SCENE_SYNONYMS,
    SCENES,
    SUBJECTS,
    VERB_FORMS,
    indefinite,
    is_usable_object,
    pluralize,
)

# The six aspect conditions. `prog` is the reference cell: it is the form every
# caption corpus is saturated with, and the form OSCBench used exclusively.
ASPECT_CONDITIONS = [
    "prog",         # A man is slicing an apple in the kitchen.
    "perf",         # A man sliced an apple in the kitchen.
    "result",       # A man has sliced an apple in the kitchen.
    "prospective",  # A man is about to slice an apple in the kitchen.
    "failed",       # A man tried to slice an apple in the kitchen but failed.
    "atelic",       # A man is slicing apples in the kitchen.
]

# Two noise floors. `paraphrase_min` is a single-token synonym substitution and is
# the primary reference: a tight floor makes the aspect comparison conservative.
# `paraphrase` fronts the whole scene phrase, moving many tokens; it is reported
# as a secondary, looser reference only.
CONTROL_CONDITIONS = ["paraphrase_min", "paraphrase", "other_verb"]


def realize(condition, subject, gerund, noun, scene):
    return _cap(_realize(condition, subject, gerund, noun, scene))


def _cap(sentence):
    return sentence[0].upper() + sentence[1:]


def _realize(condition, subject, gerund, noun, scene):
    base, past = VERB_FORMS[gerund]
    obj = indefinite(noun)
    if condition == "prog":
        return f"{subject} is {gerund} {obj} {scene}."
    if condition == "perf":
        return f"{subject} {past} {obj} {scene}."
    if condition == "result":
        return f"{subject} has {past} {obj} {scene}."
    if condition == "prospective":
        return f"{subject} is about to {base} {obj} {scene}."
    if condition == "failed":
        return f"{subject} tried to {base} {obj} {scene} but failed."
    if condition == "atelic":
        return f"{subject} is {gerund} {pluralize(noun)} {scene}."
    if condition == "paraphrase_min":
        # One-token synonym substitution. Meaning and aspect untouched.
        return f"{subject} is {gerund} {obj} {SCENE_SYNONYMS[scene]}."
    if condition == "paraphrase":
        # Meaning and aspect held constant; constituent order moves. Looser floor.
        scene_fronted = scene[0].upper() + scene[1:]
        return f"{scene_fronted}, {subject} is {gerund} {obj}."
    raise ValueError(condition)


def load_taxonomy(taxonomy_dir):
    actions = json.loads((taxonomy_dir / "action_category.json").read_text())
    objects = json.loads((taxonomy_dir / "object_category.json").read_text())

    verbs = []
    for category, elements in actions.items():
        for gerund in elements:
            if gerund in VERB_FORMS:
                verbs.append({"gerund": gerund, "action_category": category})

    nouns = []
    for major, subcats in objects.items():
        for sub, elements in subcats.items():
            for noun in elements:
                if is_usable_object(noun):
                    nouns.append({"noun": noun, "object_major": major, "object_sub": sub})
    return verbs, nouns


def build(taxonomy_dir, n_items, seed):
    rng = random.Random(seed)
    verbs, nouns = load_taxonomy(taxonomy_dir)
    if not verbs or not nouns:
        raise SystemExit("taxonomy produced no usable verbs/objects")

    # Sample verb-object pairs without replacement so no event repeats, and keep
    # verbs balanced so the held-out-verb probe split has enough groups.
    pairs = [(v, n) for v in verbs for n in nouns]
    rng.shuffle(pairs)
    seen, chosen = set(), []
    for verb, noun in pairs:
        key = (verb["gerund"], noun["noun"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append((verb, noun))
        if len(chosen) >= n_items:
            break

    records = []
    for idx, (verb, noun) in enumerate(chosen):
        subject = SUBJECTS[idx % len(SUBJECTS)]
        scene = SCENES[(idx // len(SUBJECTS)) % len(SCENES)]
        # The upper-bound control swaps in a different verb from a *different*
        # action category, so it is a genuinely different event rather than a
        # near-synonym (e.g. slicing -> chopping would understate the bound).
        alternatives = [v for v in verbs if v["action_category"] != verb["action_category"]]
        other = rng.choice(alternatives)

        item = {
            "item_id": idx,
            "gerund": verb["gerund"],
            "action_category": verb["action_category"],
            "noun": noun["noun"],
            "object_major": noun["object_major"],
            "object_sub": noun["object_sub"],
            "subject": subject,
            "scene": scene,
            "other_verb_gerund": other["gerund"],
            "texts": {},
        }
        for cond in ASPECT_CONDITIONS + ["paraphrase_min", "paraphrase"]:
            item["texts"][cond] = realize(cond, subject, verb["gerund"], noun["noun"], scene)
        item["texts"]["other_verb"] = realize(
            "prog", subject, other["gerund"], noun["noun"], scene
        )
        records.append(item)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=Path(__file__).parent.parent / "action_object_taxonomy")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = build(args.taxonomy, args.n_items, args.seed)
    with args.out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_texts = len(records) * (len(ASPECT_CONDITIONS) + len(CONTROL_CONDITIONS))
    verbs = {r["gerund"] for r in records}
    print(f"wrote {len(records)} items ({n_texts} texts) to {args.out}")
    print(f"  distinct verbs: {len(verbs)}  distinct objects: {len({r['noun'] for r in records})}")
    print("\nexample item:")
    for cond, text in records[0]["texts"].items():
        print(f"  {cond:12s} {text}")


if __name__ == "__main__":
    main()
