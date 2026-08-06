"""Build the pilot prompt suite.

Every item is a set of conditions that differ in ONE thing, and each condition
carries the count its meaning entails. Nothing here is annotated: the numbers
below are read off the semantics, which is the whole reason this direction was
chosen over the aspect one.

    dist        A photo of three girls, each holding a balloon.      balloon = 3
    coll        A photo of three girls holding a balloon together.   balloon = 1
    bare        A photo of three girls holding a balloon.            underspecified
    explicit_n  A photo of three girls and three balloons.           balloon = 3
    explicit_1  A photo of three girls and one balloon.              balloon = 1

The word before "balloon" is "a" in the first three. There is no number to copy,
which is what separates this from every counting benchmark: GeckoNum's twelve
templates and T2I-CompBench++'s thousand numeracy prompts all state the target.

`explicit_n` and `explicit_1` are the anchors, and they are known positives --
the count is stated, so a model that fails them has failed at counting rather
than at distributivity. The headline is conditioned on them, and because they
are defined in advance, the informative subset is fixed before any image exists.
That is the thing the aspect pilot could not do: there, only 9 of 37 items
carried information and we found out afterwards.

`bare` has no entailed count on purpose. It is underspecified between the two
readings, and which one a model defaults to is a result, not an error.

The pilot also carries four SCREENING families whose truth values may not be
single-valued. They are here to be looked at and cut, not to be scored: if a
family's correct count is arguable, it does not belong in the suite, and finding
that out costs a dozen images now instead of a rebuttal later.
"""

import argparse
import json
from pathlib import Path

NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# N=2 is deliberately absent. Calibration found it the weakest band under both
# detectors across two rounds -- a margin of one sits inside the counting noise.
COUNTS = [3, 4, 5]

# Objects that are handheld, discrete, and that at least one calibrated detector
# reads well. `balloon` is outside COCO's eighty classes, so it is OWLv2's to
# count; the others are Mask2Former's. Keeping both kinds in the pilot is how we
# find out whether the division of labour survives on generated images.
OBJECTS = ["balloon", "umbrella", "apple", "cup", "book", "teddy bear"]

SUBJECT = "girls"          # detected as `person`; one subject keeps the pilot small
SUBJECT_SINGULAR = "girl"

# The five conditions of a distributive item. `entails` is the count of the
# OBJECT; the subject count is stated in every condition and is checked
# separately, as a read on the model rather than on the phenomenon.
CONDITIONS = [
    ("dist",       "A photo of {n_word} {subject}, each holding {a} {obj}.",      "n"),
    ("coll",       "A photo of {n_word} {subject} holding {a} {obj} together.",   "1"),
    ("bare",       "A photo of {n_word} {subject} holding {a} {obj}.",            None),
    ("explicit_n", "A photo of {n_word} {subject} and {n_word} {obj_plural}.",    "n"),
    ("explicit_1", "A photo of {n_word} {subject} and one {obj}.",                "1"),
]

PLURAL = {"balloon": "balloons", "umbrella": "umbrellas", "apple": "apples",
          "cup": "cups", "book": "books", "teddy bear": "teddy bears"}


def article(noun):
    return "an" if noun[0] in "aeiou" else "a"


def distributive_items(objects, counts, seeds):
    items = []
    for obj in objects:
        for n in counts:
            item_id = len(items)
            conds = []
            for name, template, entail in CONDITIONS:
                prompt = template.format(
                    n_word=NUMBER_WORD[n], subject=SUBJECT, a=article(obj),
                    obj=obj, obj_plural=PLURAL[obj])
                conds.append({
                    "condition": name, "prompt": prompt,
                    # None means underspecified -- not "unknown", not "zero".
                    "entailed": None if entail is None
                                else (n if entail == "n" else 1),
                })
            items.append({"item_id": item_id, "family": "dist_coll",
                          "object": obj, "subject_count": n,
                          "count_class": "person", "object_class": obj,
                          "conditions": conds, "seeds": seeds})
    return items


# Families whose truth value is in question. Scored by eye in the pilot, then
# kept or cut. The `entailed` values written here are the readings we THINK are
# right; the pilot exists to find out whether a competent reader would agree.
SCREENING = [
    ("reciprocal", "shoe",
     [("recip", "A photo of two boys shaking hands with each other.", None),
      ("recip", "A photo of two women hugging each other.", None)],
     "how many handshakes / hands / embraces is the picture obliged to show?"),
    ("pair", "shoe",
     [("pair", "A photo of a pair of shoes on the floor.", 2),
      ("pair", "A photo of a pair of gloves on a table.", 2)],
     "does 'a pair of' reliably fix the count at two?"),
    ("respectively", "balloon",
     [("resp", "A photo of a girl and a boy holding a red balloon and a blue "
               "balloon respectively.", 2),
      ("resp", "A photo of two women wearing a red hat and a green hat "
               "respectively.", 2)],
     "is the count two, and is the colour bound to the right person?"),
    ("part_whole", "apple",
     [("part", "A photo of three apples, two of which have a bite taken out "
               "of them.", 3),
      ("part", "A photo of four cups, one of which is full.", 4)],
     "is the whole-set count unambiguous when a subset is singled out?"),
]


def screening_items(seeds, start_id):
    items = []
    for family, obj_class, prompts, question in SCREENING:
        for cond, prompt, entailed in prompts:
            items.append({
                "item_id": start_id + len(items), "family": family,
                "object": obj_class, "subject_count": None,
                "count_class": obj_class, "object_class": obj_class,
                "screening_question": question,
                "conditions": [{"condition": cond, "prompt": prompt,
                                "entailed": entailed}],
                "seeds": seeds,
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--objects", nargs="+", default=OBJECTS)
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    args = ap.parse_args()

    items = distributive_items(args.objects, args.counts, args.seeds)
    items += screening_items(args.seeds, len(items))

    n_img = sum(len(i["conditions"]) * len(i["seeds"]) for i in items)
    main_img = sum(len(i["conditions"]) * len(i["seeds"])
                   for i in items if i["family"] == "dist_coll")
    print(f"{len(items)} items, {n_img} images "
          f"({main_img} distributive + {n_img - main_img} screening)")
    for fam in dict.fromkeys(i["family"] for i in items):
        k = [i for i in items if i["family"] == fam]
        print(f"  {fam:<14}{len(k):>4} items"
              f"{sum(len(i['conditions']) * len(i['seeds']) for i in k):>6} images")

    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n"
                                for i in items))
    print(f"\nwrote {args.out}")
    print(f"\nexample item:\n" + "\n".join(
        f"  {c['condition']:<11} entails {str(c['entailed']):<4} {c['prompt']}"
        for c in items[0]["conditions"]))


if __name__ == "__main__":
    main()
