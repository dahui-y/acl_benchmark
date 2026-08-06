"""Contact sheets, so the images get looked at rather than only counted.

Every conclusion in this project that turned out to be wrong was caught by
looking: the aspect pilot's misrendered objects, and the COCO apple bowl that
invalidated a whole round of calibration. A count table cannot show that the
model drew a bunch of bananas where the annotation says one, or that "each
holding a balloon" produced three girls sharing one balloon that the detector
happened to split into three boxes.

Rows are the conditions of one item, in semantic order, so the distributive and
collective readings sit next to each other and share a seed. Any visible
difference is the text's doing.
"""

import argparse
import json
from pathlib import Path

ORDER = ["explicit_1", "coll", "bare", "dist", "explicit_n"]


def sheet(items, root, out, seed, thumb, per_sheet, counts=None, tag=""):
    from PIL import Image, ImageDraw  # noqa: PLC0415

    pad, label_h, left = 6, 22, 150
    made = []
    groups = [items[i:i + per_sheet] for i in range(0, len(items), per_sheet)]
    for gi, group in enumerate(groups):
        conds = ORDER if group[0]["family"] == "dist_coll" else \
            [c["condition"] for c in group[0]["conditions"]]
        w = left + len(conds) * (thumb + pad) + pad
        h = label_h + len(group) * (thumb + label_h + pad) + pad
        canvas = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(canvas)
        for ci, c in enumerate(conds):
            d.text((left + ci * (thumb + pad), 6), c, fill="black")
        for ri, item in enumerate(group):
            y = label_h + ri * (thumb + label_h + pad)
            d.text((6, y + thumb // 2),
                   f"item{item['item_id']:04d}\n{item['object']}"
                   f"\nN={item['subject_count']}", fill="black")
            for ci, c in enumerate(conds):
                p = root / f"item{item['item_id']:04d}" / f"{c}__seed{seed}.png"
                x = left + ci * (thumb + pad)
                if not p.exists():
                    d.rectangle([x, y, x + thumb, y + thumb], outline="red")
                    continue
                canvas.paste(Image.open(p).convert("RGB")
                             .resize((thumb, thumb)), (x, y))
                cap = ""
                if counts:
                    k = (item["item_id"], c, seed, item["object_class"])
                    if k in counts:
                        cap = f"counted {counts[k]}"
                        ent = next((x["entailed"] for x in item["conditions"]
                                    if x["condition"] == c), None)
                        cap += f" / entails {ent}" if ent is not None else " / —"
                d.text((x + 2, y + thumb + 4), cap, fill="black")
        # The family goes in the name: main and screening sheets both start
        # their numbering at zero, so without it the screening sheets overwrite
        # the distributive ones and the overwrite is silent.
        path = out / f"sheet_{tag}_seed{seed}_{gi:02d}.png"
        canvas.save(path)
        made.append(path)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True,
                    help="a model directory under pilot/images/")
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--counts", type=Path, nargs="*", default=[],
                    help="overlay the detector's count under each image")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--thumb", type=int, default=200)
    ap.add_argument("--per-sheet", type=int, default=9)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in args.stimuli.read_text().splitlines() if l]
    if args.family:
        items = [i for i in items if i["family"] == args.family]
    # Screening families have their own condition sets, so they cannot share a
    # sheet with the distributive items.
    main_items = [i for i in items if i["family"] == "dist_coll"]
    other = [i for i in items if i["family"] != "dist_coll"]

    counts = {}
    for p in args.counts:
        for r in map(json.loads, p.read_text().splitlines()):
            if r and r.get("status") == "ok":
                counts[(r["item_id"], r["condition"], r["seed"],
                        r["noun"])] = r["count"]

    out = args.out or args.images / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if main_items:
        made += sheet(main_items, args.images, out, args.seed, args.thumb,
                      args.per_sheet, counts, tag="dist_coll")
    for fam in dict.fromkeys(i["family"] for i in other):
        made += sheet([i for i in other if i["family"] == fam], args.images,
                      out, args.seed, args.thumb, 8, counts, tag=fam)
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
