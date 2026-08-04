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
    VERB_ASPECTUAL_CLASS,
    SCENE_SYNONYMS,
    is_compatible,
    SCENES,
    SUBJECTS,
    VERB_FORMS,
    indefinite,
    is_usable_object,
    pluralize,
)

# Three axes of how language encodes whether the event reaches its endpoint.
# `prog` is the reference cell throughout: it is the form caption corpora are
# saturated with, and the only form OSCBench used.

# Axis A -- grammatical aspect.
ASPECT_CONDITIONS = [
    "prog",         # A man is slicing an apple in the kitchen.
    "perf",         # A man sliced an apple in the kitchen.
    "result",       # A man has sliced an apple in the kitchen.
    "prospective",  # A man is about to slice an apple in the kitchen.
    "failed",       # A man tried to slice an apple in the kitchen but failed.
]

# Axis B -- telicity, via the object's quantization. This is the headline axis:
# on umT5-XXL the telic/atelic pair came out statistically indistinguishable
# from a meaning-preserving reword.
#
# Thickened to five conditions so the axis supports a 2x2 rather than a single
# pair. A single cross-telicity contrast cannot rule out the alternative that
# the encoder is simply insensitive to determiners in general, which would make
# the result about morphology rather than telicity. Adding pairs that differ in
# determiner but agree in telicity separates the two:
#
#   cross-telicity pairs   large if telicity is represented
#   within-telicity pairs  small if the effect is really about telicity
#
# If both sit at the floor, the honest conclusion is the broader one -- the
# encoder is blind to determiner and number marking, telicity included.
TELICITY_CONDITIONS = [
    "atelic",           # is slicing apples          bare plural, cumulative  -> atelic
    "atelic_some",      # is slicing some apples     vague quantity           -> atelic
    "telic_plural",     # is slicing the apples      definite plural          -> telic
    "telic_numeral",    # is slicing three apples    numeral                  -> telic
    "telic_partitive",  # is slicing half an apple   measure                  -> telic
]

# Axis C -- phase verbs. Aspectual operators that also carry presuppositions,
# carried over from the presupposition direction (see ../idea_presupposition.md).
PHASE_CONDITIONS = [
    "phase_begin",   # A man began slicing an apple in the kitchen.
    "phase_stop",    # A man stopped slicing an apple in the kitchen.
    "phase_finish",  # A man finished slicing an apple in the kitchen.
    "phase_keep",    # A man kept slicing an apple in the kitchen.
]

TARGET_CONDITIONS = ASPECT_CONDITIONS + TELICITY_CONDITIONS + PHASE_CONDITIONS

# Two noise floors. `paraphrase_min` is a single-token synonym substitution and is
# the primary reference: a tight floor makes the aspect comparison conservative.
# `paraphrase` fronts the whole scene phrase, moving many tokens; it is reported
# as a secondary, looser reference only.
CONTROL_CONDITIONS = ["paraphrase_min", "paraphrase", "filler", "other_verb"]


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
    if condition == "atelic_some":
        return f"{subject} is {gerund} some {pluralize(noun)} {scene}."
    if condition == "telic_plural":
        # Same plural morphology as `atelic`, but definite and therefore quantized.
        return f"{subject} is {gerund} the {pluralize(noun)} {scene}."
    if condition == "telic_numeral":
        return f"{subject} is {gerund} three {pluralize(noun)} {scene}."
    if condition == "telic_partitive":
        return f"{subject} is {gerund} half {obj} {scene}."
    if condition == "phase_begin":
        return f"{subject} began {gerund} {obj} {scene}."
    if condition == "phase_stop":
        return f"{subject} stopped {gerund} {obj} {scene}."
    if condition == "phase_finish":
        return f"{subject} finished {gerund} {obj} {scene}."
    if condition == "phase_keep":
        return f"{subject} kept {gerund} {obj} {scene}."
    if condition == "paraphrase_min":
        # One-token synonym substitution. Meaning and aspect untouched.
        return f"{subject} is {gerund} {obj} {SCENE_SYNONYMS[scene]}."
    if condition == "paraphrase":
        # Meaning and aspect held constant; constituent order moves. Looser floor.
        scene_fronted = scene[0].upper() + scene[1:]
        return f"{scene_fronted}, {subject} is {gerund} {obj}."
    if condition == "filler":
        # Length-matched to the `failed` condition: adds a comparable number of
        # tokens without touching aspect or the event. Running the probe for real
        # showed that raw cosine distance tracks token overlap, so a condition
        # that adds tokens looks "distant" for reasons that have nothing to do
        # with aspect. This control makes that visible.
        return f"{subject} is {gerund} {obj} {scene}, as the footage shows."
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


def build(taxonomy_dir, n_items, n_generate, seed):
    rng = random.Random(seed)
    verbs, nouns = load_taxonomy(taxonomy_dir)
    if not verbs or not nouns:
        raise SystemExit("taxonomy produced no usable verbs/objects")

    # Stratify by verb rather than sampling verb-object pairs freely. Free
    # sampling leaves some verbs with many items and others with almost none,
    # which breaks the held-out-verb probe split and makes the by-aspectual-class
    # analysis rest on unequal cell sizes.
    per_verb = max(1, n_items // len(verbs))
    chosen, dropped = [], []
    for verb in verbs:
        # Only objects the action can plausibly apply to. Skipping this yields
        # items like "melting a scallion", which confound aspect with
        # plausibility and cannot be rendered sensibly by any model.
        usable = [n for n in nouns
                  if is_compatible(verb["action_category"], n["object_sub"],
                                   verb["gerund"], n["noun"])]
        if not usable:
            # A verb whose plausible objects are all mass nouns cannot take part in
            # the telicity alternation. Drop it rather than force bad items.
            dropped.append(verb["gerund"])
            continue
        picks = rng.sample(usable, min(per_verb, len(usable)))
        chosen.extend((verb, noun) for noun in picks)
    rng.shuffle(chosen)
    chosen = chosen[:n_items]

    # The generation subset is what actually gets turned into video, so it is
    # balanced across verbs too -- every verb contributes, no verb dominates.
    gen_per_verb = max(1, n_generate // len(verbs))
    gen_ids, seen_counts = set(), {}
    for idx, (verb, _) in enumerate(chosen):
        g = verb["gerund"]
        if seen_counts.get(g, 0) < gen_per_verb and len(gen_ids) < n_generate:
            gen_ids.add(idx)
            seen_counts[g] = seen_counts.get(g, 0) + 1

    records = []
    for idx, (verb, noun) in enumerate(chosen):
        subject = SUBJECTS[idx % len(SUBJECTS)]
        scene = SCENES[(idx // len(SUBJECTS)) % len(SCENES)]
        # The different-event control swaps in a verb from a *different* action
        # category, so it is a genuinely different event rather than a near
        # synonym (slicing -> chopping would understate it).
        # The control verb must also be plausible for this object, or the
        # "different event" baseline becomes a "nonsense event" baseline.
        alternatives = [v for v in verbs
                        if v["action_category"] != verb["action_category"]
                        and is_compatible(v["action_category"], noun["object_sub"],
                                          v["gerund"], noun["noun"])]
        other = rng.choice(alternatives) if alternatives else rng.choice(verbs)

        item = {
            "item_id": idx,
            "gerund": verb["gerund"],
            "action_category": verb["action_category"],
            "aspectual_class": VERB_ASPECTUAL_CLASS[verb["gerund"]],
            "noun": noun["noun"],
            "object_major": noun["object_major"],
            "object_sub": noun["object_sub"],
            "subject": subject,
            "scene": scene,
            "other_verb_gerund": other["gerund"],
            "in_generation_subset": idx in gen_ids,
            "human_verified": False,   # flipped by the review pass; see --review-out
            "texts": {},
        }
        for cond in TARGET_CONDITIONS + ["paraphrase_min", "paraphrase", "filler"]:
            item["texts"][cond] = realize(cond, subject, verb["gerund"], noun["noun"], scene)
        item["texts"]["other_verb"] = realize(
            "prog", subject, other["gerund"], noun["noun"], scene
        )
        records.append(item)
    if dropped:
        print(f"  dropped verbs (no count-noun objects): {', '.join(sorted(set(dropped)))}")
    return records


def write_review_sheet(records, path, only_generation_subset=True):
    """Export a sheet for the human verification pass.

    Every prompt in an OSCBench-style benchmark goes through human review; this
    is that step. One row per item so a reviewer sees all variants of an event
    together and can judge them against each other, which is the only way to
    catch a variant that is grammatical on its own but not a minimal pair.
    """
    rows = [r for r in records if r["in_generation_subset"] or not only_generation_subset]
    conditions = TARGET_CONDITIONS + CONTROL_CONDITIONS
    with path.open("w") as fh:
        fh.write("\t".join(["item_id", "gerund", "aspectual_class", "noun",
                            *conditions, "natural?", "minimal_pair?", "notes"]) + "\n")
        for rec in rows:
            fh.write("\t".join([
                str(rec["item_id"]), rec["gerund"], rec["aspectual_class"], rec["noun"],
                *[rec["texts"][c] for c in conditions], "", "", "",
            ]) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path,
                    default=Path(__file__).parent.parent / "action_object_taxonomy")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--n-items", type=int, default=200,
                    help="items for the text-side analysis (free)")
    ap.add_argument("--n-generate", type=int, default=60,
                    help="items that get turned into video (the expensive subset)")
    ap.add_argument("--review-out", type=Path, default=None,
                    help="TSV for the human verification pass")
    ap.add_argument("--review-all", action="store_true",
                    help="review every item, not just the generation subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = build(args.taxonomy, args.n_items, args.n_generate, args.seed)
    with args.out.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_conditions = len(TARGET_CONDITIONS) + len(CONTROL_CONDITIONS)
    n_gen = sum(r["in_generation_subset"] for r in records)
    print(f"wrote {len(records)} items x {n_conditions} conditions "
          f"= {len(records) * n_conditions} prompts to {args.out}")
    print(f"  distinct verbs: {len({r['gerund'] for r in records})}  "
          f"distinct objects: {len({r['noun'] for r in records})}")
    print("  aspectual classes: " + ", ".join(
        f"{k}={sum(1 for r in records if r['aspectual_class'] == k)}"
        for k in ("incremental_theme", "degree_achievement", "activity")))
    print(f"  generation subset: {n_gen} items x {n_conditions} = "
          f"{n_gen * n_conditions} prompts  "
          f"(x4 models = {n_gen * n_conditions * 4} videos)")

    if args.review_out:
        n = write_review_sheet(records, args.review_out, not args.review_all)
        print(f"  review sheet: {n} items -> {args.review_out}")

    print("\nexample item:")
    for cond, text in records[0]["texts"].items():
        print(f"  {cond:15s} {text}")


if __name__ == "__main__":
    main()
