"""Build the calibration cells from gold instance annotations.

A *cell* is one (image, class) pair whose gold instance count is known. The
verifier is asked "how many <class> are in this image", and its answer is
compared against that count. No new labelling: someone else already paid for
these counts.

**Why LVIS and not COCO.** The first version of this file used COCO, and the
counts were not trustworthy. COCO image 2149 is annotated with one apple; the
photograph is a bowl holding seven or eight of them. That is not a rare defect
-- it concentrates in exactly the classes a counting benchmark cares about
(apple, banana, broccoli, carrot: small, repeated, touching), and it inflates
the verifier's apparent false-positive rate, because a verifier that correctly
reports eight is scored as wrong. Numbers measured that way understate every
verifier, and understate them worst on the classes that matter most.

LVIS annotates the same photographs and records, per image, which categories are
NOT exhaustively annotated. Dropping those pairs leaves counts that can be
compared against. This is the property the calibration needs and COCO does not
carry, and it costs nothing extra: still no new labels.

Two further filters, both there so the gold count is not itself arguable:

**Tiny background instances.** LVIS labels instances a few pixels across, and no
prompt-driven generation produces those, so a verifier's failure to find one
says nothing about how it behaves on our images. Cells keep only images where
EVERY instance of the class clears an area floor; images with borderline
instances are dropped rather than counted either way.

**Negative categories.** LVIS also records categories verified absent from an
image. Those make clean N=0 cells, which is the one band a detector cannot fake.

The result is a deliberately clean regime, and that is the point: this step can
only kill the direction, never certify it. If a verifier cannot count 2-5 large,
unambiguous objects in real photographs, it will not do better on generated ones.
"""

import argparse
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

# LVIS names differ from COCO's for a good many classes, and a silent miss here
# would drop a class from the calibration without saying so -- build_cells
# raises on anything unmapped rather than quietly skipping it.
LVIS_NAME = {
    "orange": "orange_(fruit)", "wine glass": "wineglass",
    "cell phone": "cellular_telephone", "sports ball": "ball",
    "remote": "remote_control", "mouse": "mouse_(computer_equipment)",
    "potted plant": "flowerpot", "laptop": "laptop_computer",
    "dining table": "dining_table", "teddy bear": "teddy_bear",
    "donut": "doughnut", "couch": "sofa", "wristwatch": "watch",
}

# The object vocabulary the benchmark would actually use: discrete, countable,
# idiomatic as the object of "the three girls are each holding a ___". LVIS's
# 1203 categories reach well past COCO's 80, which is why `balloon` -- the
# canonical example the whole design is written around -- is testable at all.
CANDIDATE_CLASSES = [
    # small handheld things: the core of the distributive items
    "balloon", "apple", "orange", "banana", "donut", "cup", "wine glass",
    "bottle", "bowl", "book", "cell phone", "umbrella", "kite", "sports ball",
    "frisbee", "vase", "clock", "teddy bear", "scissors", "spoon", "fork",
    "knife", "toothbrush", "remote", "mouse", "carrot", "broccoli",
    "sandwich", "cake", "pizza", "candle", "flower_arrangement", "mug",
    "wristwatch", "necklace", "sunglasses", "handbag", "backpack",
    # agents and animals: the subject side ("the three girls", "the two dogs")
    "person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant",
    "zebra", "giraffe", "duck", "goose",
    # furniture-scale, for the larger-N difficulty band
    "chair", "couch", "potted plant", "bed", "dining table", "laptop",
]


def load_source(path):
    """Return (images, annotations, name->id, per-image exhaustive info)."""
    d = json.loads(Path(path).read_text())
    by_name = {c["name"]: c["id"] for c in d["categories"]}
    images = {im["id"]: im for im in d["images"]}
    return d, images, by_name


def build_cells(ann_path, classes, min_area_frac, max_count,
                drop_borderline=True, negatives=0):
    d, images, by_name = load_source(ann_path)

    wanted = {}
    missing = []
    for c in classes:
        name = LVIS_NAME.get(c, c.replace(" ", "_"))
        if name in by_name:
            wanted[by_name[name]] = c
        elif c in by_name:
            wanted[by_name[c]] = c
        else:
            missing.append(c)
    if missing:
        raise SystemExit(
            f"{len(missing)} classes have no category in {Path(ann_path).name}: "
            f"{missing}\n  Add them to LVIS_NAME, or drop them from "
            f"CANDIDATE_CLASSES. They are not skipped silently because a class "
            f"vanishing from the calibration is exactly the kind of thing that "
            f"goes unnoticed.")

    # Pairs LVIS flags as not exhaustively annotated: the count is a lower
    # bound, so scoring a verifier against it would punish it for being right.
    not_exhaustive = {(im["id"], cid) for im in d["images"]
                      for cid in im.get("not_exhaustive_category_ids", [])}

    big = defaultdict(int)
    small = defaultdict(int)
    for a in d["annotations"]:
        if a["category_id"] not in wanted:
            continue
        key = (a["image_id"], a["category_id"])
        im = images[a["image_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        (big if frac >= min_area_frac else small)[key] += 1

    cells, dropped = [], 0
    for key in set(big) | set(small):
        if key in not_exhaustive:
            dropped += 1
            continue
        img_id, cat_id = key
        if drop_borderline and small[key]:
            continue
        n = big[key] + (0 if drop_borderline else small[key])
        if not 1 <= n <= max_count:
            continue
        cells.append(_cell(images[img_id], wanted[cat_id], n))

    # Verified-absent categories give N=0 cells. A verifier that hedges upward
    # -- and both detectors did, at low thresholds -- has nowhere to hide here.
    if negatives:
        neg = [(im, cid) for im in d["images"]
               for cid in im.get("neg_category_ids", []) if cid in wanted]
        per_class = defaultdict(int)
        for im, cid in neg:
            if per_class[cid] < negatives:
                per_class[cid] += 1
                cells.append(_cell(im, wanted[cid], 0))

    print(f"dropped {dropped} (image, class) pairs flagged not-exhaustive")
    return sorted(cells, key=lambda c: (c["class"], c["gold"], c["image_id"]))


def _cell(im, class_name, n):
    return {"image_id": im["id"],
            "file_name": im["coco_url"].rsplit("/", 1)[-1],
            "url": im["coco_url"], "class": class_name, "gold": n}


def balance(cells, per_n, per_class_n):
    """Take a class-spread, count-balanced subset, deterministically.

    The raw pool is dominated by N=1, and a metric computed over a set that
    skewed mostly reports the verifier's behaviour at N=1 -- the one band the
    benchmark barely uses. Round-robin over classes, so every class contributes
    before any class contributes twice.
    """
    out = []
    for n in sorted({c["gold"] for c in cells}):
        pools = defaultdict(list)
        for c in cells:
            if c["gold"] == n:
                pools[c["class"]].append(c)
        ordered = [pools[k][:per_class_n] for k in sorted(pools)]
        flat = [c for lap in zip_longest(*ordered) for c in lap if c is not None]
        out.extend(flat[:per_n])
    return out


def summarise(cells, max_count):
    by_class = defaultdict(lambda: defaultdict(int))
    by_n = defaultdict(int)
    for c in cells:
        by_class[c["class"]][c["gold"]] += 1
        by_n[c["gold"]] += 1
    counts = sorted(by_n)
    print(f"\n{len(cells)} cells over {len(by_class)} classes\n")
    print("count distribution:  " + "  ".join(f"N={n}:{by_n[n]}" for n in counts))
    print(f"\n{'class':<20}" + "".join(f"{'N=' + str(n):>6}" for n in counts)
          + f"{'total':>8}")
    for name in sorted(by_class, key=lambda k: -sum(by_class[k].values())):
        row = by_class[name]
        print(f"{name:<20}" + "".join(f"{row[n]:>6}" for n in counts)
              + f"{sum(row.values()):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path,
                    default=Path(__file__).parent / "data" / "lvis_v1_val.json",
                    help="LVIS v1 val. COCO's instances_val2017.json also "
                         "loads, but its counts are not exhaustive -- see the "
                         "module docstring")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "data" / "cells.jsonl")
    ap.add_argument("--min-area-frac", type=float, default=0.005,
                    help="an instance below this fraction of the image is "
                         "'small'; 0.005 of a 640x480 image is ~39x39 px")
    ap.add_argument("--max-count", type=int, default=5,
                    help="the benchmark's difficulty bands; N=1 is the "
                         "collective reading and is always included")
    ap.add_argument("--keep-borderline", action="store_true")
    ap.add_argument("--negatives", type=int, default=0,
                    help="add up to this many verified-absent (N=0) cells per "
                         "class, from LVIS neg_category_ids")
    ap.add_argument("--per-n", type=int, default=0,
                    help="balance to this many cells per count band")
    ap.add_argument("--per-class-n", type=int, default=12)
    args = ap.parse_args()

    cells = build_cells(args.annotations, CANDIDATE_CLASSES,
                        args.min_area_frac, args.max_count,
                        drop_borderline=not args.keep_borderline,
                        negatives=args.negatives)
    if args.per_n:
        cells = balance(cells, args.per_n, args.per_class_n)

    summarise(cells, args.max_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c) + "\n" for c in cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
