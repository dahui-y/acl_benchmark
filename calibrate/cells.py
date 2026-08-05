"""Build the calibration cells from COCO gold annotations.

A *cell* is one (image, class) pair whose gold instance count is known. The
detector is asked "how many <class> are in this image", and its answer is
compared against that count. No new labelling: the counts come out of
`instances_val2017.json`.

Two things about COCO gold counts have to be handled or the calibration number
is meaningless:

**Crowd regions.** `iscrowd=1` marks a blob covering an unspecified number of
instances. An image with one crowd annotation of apples does not have "1 apple".
Any image with a crowd region for the class is dropped, not counted as 1.

**Tiny background instances.** COCO labels every visible instance, including
ones a few pixels across. A generated image for "three apples" contains three
salient apples, so a detector's failure to find a 12-pixel apple in a real photo
says nothing about how it will behave on our images. Cells are therefore
restricted to images where EVERY instance of the class is above an area
threshold -- images with borderline instances are dropped rather than counted
either way, because for those the gold count itself is arguable and we would be
measuring our own threshold rather than the detector.

The result is a deliberately clean regime. That is the point: if the detector
cannot count 2-5 large, unambiguous objects in real photographs, the whole
plan of using it as the verifier is dead, and no amount of care downstream
rescues it. The optimism is intentional -- this step can only kill the
direction, never certify it.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# The object vocabulary the benchmark would actually use: things that are
# discrete, countable, and idiomatic as the object of "the three girls are each
# holding a ___". COCO's 80 classes are the constraint here -- the real suite
# can go beyond them with an open-vocabulary detector, but calibration needs
# gold counts, and gold counts only exist for these.
CANDIDATE_CLASSES = [
    # small handheld things -- the core of the distributive items
    "apple", "orange", "banana", "donut", "cup", "wine glass", "bottle",
    "bowl", "book", "cell phone", "umbrella", "kite", "sports ball",
    "frisbee", "vase", "clock", "teddy bear", "scissors", "spoon", "fork",
    "knife", "toothbrush", "remote", "mouse", "carrot", "broccoli",
    "sandwich", "cake", "pizza", "hot dog",
    # agents and animals -- the subject side ("the three girls", "the two dogs")
    "person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant",
    "zebra", "giraffe",
    # furniture-scale, for the larger-N difficulty band
    "chair", "couch", "potted plant", "bed", "dining table", "tv", "laptop",
]


def build_cells(ann_path, classes, min_area_frac, max_count, drop_borderline=True):
    """Return cells [(image_id, file_name, class_name, gold_count), ...].

    min_area_frac is a fraction of the image area. An instance below it is
    "small". With drop_borderline, an image containing ANY small instance of
    the class is discarded; without it, small instances are simply counted.
    """
    coco = json.loads(Path(ann_path).read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    wanted = {c["id"] for c in coco["categories"] if c["name"] in classes}
    images = {im["id"]: im for im in coco["images"]}

    big = defaultdict(int)        # (img, cat) -> instances above threshold
    small = defaultdict(int)      # (img, cat) -> instances below threshold
    crowd = set()                 # (img, cat) with a crowd region

    for a in coco["annotations"]:
        if a["category_id"] not in wanted:
            continue
        key = (a["image_id"], a["category_id"])
        if a.get("iscrowd"):
            crowd.add(key)
            continue
        im = images[a["image_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        (big if frac >= min_area_frac else small)[key] += 1

    cells = []
    for key in set(big) | set(small):
        if key in crowd:
            continue
        img_id, cat_id = key
        if drop_borderline and small[key]:
            continue
        n = big[key] + (0 if drop_borderline else small[key])
        if not 1 <= n <= max_count:
            continue
        cells.append({
            "image_id": img_id, "file_name": images[img_id]["file_name"],
            "class": cat_name[cat_id], "gold": n,
        })
    return sorted(cells, key=lambda c: (c["class"], c["gold"], c["image_id"]))


def balance(cells, per_n, per_class_n):
    """Take a class-spread, count-balanced subset, deterministically.

    Two reasons not to calibrate on the raw pool. It is 71% N=1, and a metric
    computed over a set that skewed mostly reports the detector's behaviour at
    N=1 -- which is the one band the benchmark barely uses. And `person` alone
    is a quarter of the multi-instance cells, so an unbalanced set would largely
    be measuring person detection.

    Round-robin over classes, so every class contributes before any class
    contributes twice, and the per-class cap binds only the classes that have
    enough images to hit it.
    """
    from itertools import zip_longest  # noqa: PLC0415

    out = []
    for n in sorted({c["gold"] for c in cells}):
        pools = defaultdict(list)
        for c in cells:
            if c["gold"] == n:
                pools[c["class"]].append(c)
        ordered = [pools[k][:per_class_n] for k in sorted(pools)]
        # interleave: one cell per class per lap
        flat = [c for lap in zip_longest(*ordered) for c in lap if c is not None]
        out.extend(flat[:per_n])
    return out


def summarise(cells, max_count):
    by_class = defaultdict(lambda: defaultdict(int))
    by_n = defaultdict(int)
    for c in cells:
        by_class[c["class"]][c["gold"]] += 1
        by_n[c["gold"]] += 1
    print(f"{len(cells)} cells over {len(by_class)} classes\n")
    print("count distribution:")
    for n in range(1, max_count + 1):
        print(f"  N={n}: {by_n[n]:5d}")
    print(f"\n{'class':<15}" + "".join(f"{'N='+str(n):>7}" for n in
                                       range(1, max_count + 1)) + f"{'total':>8}")
    for name in sorted(by_class, key=lambda k: -sum(by_class[k].values())):
        row = by_class[name]
        print(f"{name:<15}" + "".join(f"{row[n]:>7}" for n in
                                      range(1, max_count + 1))
              + f"{sum(row.values()):>8}")
    return by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path,
                    default=Path(__file__).parent / "data" / "annotations"
                    / "instances_val2017.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "cells.jsonl")
    ap.add_argument("--min-area-frac", type=float, default=0.005,
                    help="an instance below this fraction of the image is "
                         "'small'; 0.005 of a 640x480 image is ~1500 px, about "
                         "39x39 -- still small, but unarguably visible")
    ap.add_argument("--max-count", type=int, default=6,
                    help="the benchmark's difficulty bands are N in 2..5; 1 and "
                         "6 are included so the confusion matrix has room on "
                         "both sides")
    ap.add_argument("--keep-borderline", action="store_true",
                    help="count small instances instead of dropping the image; "
                         "reports the harder regime for comparison")
    ap.add_argument("--per-n", type=int, default=0,
                    help="balance to this many cells per count band; 0 = keep "
                         "the raw pool, which is 71%% N=1")
    ap.add_argument("--per-class-n", type=int, default=12,
                    help="cap per (class, N) when balancing, so `person` does "
                         "not become a quarter of the set")
    args = ap.parse_args()

    cells = build_cells(args.annotations, set(CANDIDATE_CLASSES),
                        args.min_area_frac, args.max_count,
                        drop_borderline=not args.keep_borderline)
    if args.per_n:
        cells = balance(cells, args.per_n, args.per_class_n)

    summarise(cells, args.max_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c) + "\n" for c in cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
