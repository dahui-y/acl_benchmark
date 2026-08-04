"""Regression tests pinning the triage decisions into the compatibility rules.

Four mechanisms now interact -- action-category compatibility, verb-level
overrides, a verb-object blocklist, and the mass-noun filter -- and each triage
pass tightened one of them. Without tests, editing any one rule can silently
undo a decision that took a pass to find.

Run: python test_lexicon.py
"""

import json
from pathlib import Path

from lexicon import MASS_OR_GENERIC, VERB_ASPECTUAL_CLASS, VERB_FORMS, is_compatible

TAXONOMY = Path(__file__).parent.parent / "action_object_taxonomy"


def object_subcategory():
    objects = json.loads((TAXONOMY / "object_category.json").read_text())
    return {noun: sub for subs in objects.values() for sub, nouns in subs.items()
            for noun in nouns}


def action_category():
    actions = json.loads((TAXONOMY / "action_category.json").read_text())
    return {g: cat for cat, gerunds in actions.items() for g in gerunds}


SUB = object_subcategory()
CAT = action_category()


def allowed(gerund, noun):
    return is_compatible(CAT[gerund], SUB[noun], gerund, noun)


# (verb, object, why) -- each was produced by an earlier version of the rules and
# removed by a specific triage pass.
MUST_REJECT = [
    ("melting", "oreo", "everything that melts is a mass noun; verb dropped"),
    ("melting", "cake", "same"),
    ("zesting", "carrot", "zest is citrus peel only"),
    ("zesting", "scallion", "same"),
    ("whipping", "coconut", "Mixing is too coarse for whipping"),
    ("mashing", "almond", "Pressing reaches nuts"),
    ("mashing", "coconut", "hard-shelled, blocked per verb"),
    ("squeezing", "peanut", "Pressing reaches nuts"),
    ("squeezing", "banana", "squeezing needs something juicy"),
    ("squeezing", "squash", "Fruiting too broad for this verb"),
    ("frying", "almond", "nuts are cooked in the plural sense"),
    ("roasting", "walnut", "same"),
    ("grilling", "hazelnut", "same"),
    ("rolling", "biscuit", "biscuit is the finished item, not the dough"),
    ("roasting", "cracker", "same bug as rolling a biscuit"),
    ("grating", "capsicum", "grates to pulp, so the target state is undefined"),
    ("shredding", "tomato", "shred implies strands; tomato produces none"),
    ("grating", "scallion", "alliums are not grated"),
    ("roasting", "egg", "eggs are fried or boiled, not dry-roasted"),
    ("grilling", "egg", "same"),
    ("browning", "egg", "same"),
    ("sauteing", "egg", "same"),
]

# Unusual but retained: the criterion is whether the target state is definable,
# not whether the pairing is frequent. The within-item design shares one
# action-object pair across all conditions, so plausibility cancels.
MUST_ALLOW = [
    ("peeling", "radish", "uncommon, but peeled is peeled"),
    ("mincing", "pumpkin", "usually cubed, but minced has a clear end state"),
    ("frying", "cucumber", "a real dish in several cuisines"),
    ("crushing", "cucumber", "smashed cucumber is a real technique"),
    ("slicing", "avocado", "ordinary"),
    ("mashing", "potato", "ordinary"),
    ("zesting", "lemon", "the canonical case"),
    ("whipping", "egg", "the canonical case"),
    ("frying", "egg", "the one Heating verb that does take an egg"),
]

MUST_BE_MASS = ["celery", "chive", "bean", "okra", "dill", "caramel", "butter"]


def main():
    failures = []

    for gerund, noun, why in MUST_REJECT:
        if gerund not in CAT or noun not in SUB:
            continue
        if noun not in MASS_OR_GENERIC and allowed(gerund, noun):
            failures.append(f"should reject {gerund} + {noun} ({why})")

    for gerund, noun, why in MUST_ALLOW:
        if noun in MASS_OR_GENERIC:
            failures.append(f"{noun} should not be mass ({gerund} + {noun}: {why})")
        elif not allowed(gerund, noun):
            failures.append(f"should allow {gerund} + {noun} ({why})")

    for noun in MUST_BE_MASS:
        if noun not in MASS_OR_GENERIC:
            failures.append(f"{noun} should be in MASS_OR_GENERIC")

    missing = set(VERB_FORMS) - set(VERB_ASPECTUAL_CLASS)
    if missing:
        failures.append(f"verbs without an aspectual class: {sorted(missing)}")

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print(f"ok: {len(MUST_REJECT)} rejections, {len(MUST_ALLOW)} retentions, "
          f"{len(MUST_BE_MASS)} mass nouns, all {len(VERB_FORMS)} verbs classed")


if __name__ == "__main__":
    main()
