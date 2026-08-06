"""Count objects in the generated images, with the calibrated detectors.

Reuses `calibrate/detect.py` unchanged, so the verifier running here is exactly
the one whose error rate was measured against LVIS gold. Two detectors, and
which one owns which class is settled by the calibration rather than by
preference: Mask2Former for COCO's eighty, OWLv2 for everything else. Both are
run over everything anyway, because their agreement on generated images is the
number the pilot exists to produce -- on photographs it was 0.887, and whether
it survives the domain shift is the one limitation this project cannot argue
away from an armchair.

The detector is asked "how many <noun>", and never sees the prompt. It cannot
know whether the image came from `dist` or `coll`, so a difference between
conditions cannot leak in from the instrument. That is stronger than the video
judge, which had to be told the verb: here the instrument is structurally
incapable of seeing the sentence.

Both the object and the subject are counted. The object count is the phenomenon;
the subject count is stated in every condition, so it reads how well the model
does at plain counting on this very suite -- a within-pilot version of the
anchor, free.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "calibrate"))
from detect import load_detector  # noqa: E402

# Operating thresholds from the LVIS calibration, chosen for balanced errors --
# calling one object "several" corrupts the collective condition while missing
# objects corrupts the distributive one, and a threshold picked for accuracy
# alone would bias the very comparison this suite makes.
THRESHOLD = {"mask2former": 0.40, "owlv2": 0.20}

# Which detector owns which class, decided by the calibration. Mask2Former does
# not have these in its vocabulary at all; OWLv2 scored 0.909 plurality on them.
OPEN_VOCAB_ONLY = {"balloon", "teddy bear", "glove", "hat", "shoe"}


def load_items(path):
    return {r["item_id"]: r for r in
            (json.loads(l) for l in path.read_text().splitlines() if l)}


def find_images(root):
    out = []
    for p in sorted(root.glob("item*/*.png")):
        condition, _, seed = p.stem.partition("__seed")
        if seed.isdigit():
            out.append({"item_id": int(p.parent.name.removeprefix("item")),
                        "condition": condition, "seed": int(seed), "path": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True,
                    help="a model directory under pilot/images/")
    ap.add_argument("--detector", default="mask2former",
                    choices=["mask2former", "owlv2"])
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    items = load_items(args.stimuli)
    jobs = find_images(args.images)
    if not jobs:
        raise SystemExit(f"no item*/*.png under {args.images}")

    unknown = sorted({j["item_id"] for j in jobs} - set(items))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} image folders have item ids not in "
            f"{args.stimuli.name}: {unknown[:8]}\n"
            f"  The images and the stimuli file come from different builds of "
            f"the suite. This does not error later, it counts the wrong object "
            f"against a plausible-looking question -- regenerate or copy the "
            f"matching stimuli.jsonl.")

    out_path = args.out or args.images / f"counts_{args.detector}.jsonl"
    already = set()
    if out_path.exists():
        already = {(r["item_id"], r["condition"], r["seed"], r["noun"])
                   for r in map(json.loads, out_path.read_text().splitlines())
                   if r}

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = load_detector(args.detector, device)
    thr = THRESHOLD[args.detector]

    todo = []
    for j in jobs:
        item = items[j["item_id"]]
        # The object is always counted; the subject only where one is stated.
        nouns = [item["object_class"]]
        if item.get("subject_count"):
            nouns.append("person")
        for noun in nouns:
            if (j["item_id"], j["condition"], j["seed"], noun) in already:
                continue
            if args.detector == "mask2former" and noun in OPEN_VOCAB_ONLY:
                continue    # not in its vocabulary; OWLv2 owns these
            todo.append({**j, "noun": noun})
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(jobs)} images, {len(todo)} (image, noun) counts to run "
          f"[{args.detector} @ thr {thr} on {device}]")

    t0 = time.time()
    with out_path.open("a") as fh:
        for n, job in enumerate(todo, 1):
            item = items[job["item_id"]]
            cond = next(c for c in item["conditions"]
                        if c["condition"] == job["condition"])
            rec = {"item_id": job["item_id"], "condition": job["condition"],
                   "seed": job["seed"], "noun": job["noun"],
                   "family": item["family"], "object": item["object"],
                   "detector": args.detector, "threshold": thr,
                   # For the subject the entailed count is the stated number;
                   # for the object it is what the semantics gives, which is
                   # None where the sentence is genuinely underspecified.
                   "entailed": (item["subject_count"] if job["noun"] == "person"
                                else cond["entailed"])}
            try:
                image = Image.open(job["path"]).convert("RGB")
                scores, _ = det.scores_for(image, job["noun"])
                rec["count"] = sum(1 for s in scores if s >= thr)
                rec["scores"] = sorted((round(float(s), 3) for s in scores),
                                       reverse=True)[:12]
                rec["status"] = "ok"
            except Exception as exc:      # one bad image must not kill the batch
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"  ERROR item{job['item_id']:04d} {job['condition']} "
                      f"{job['noun']}: {rec['error']}")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 50 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s  "
                      f"eta {(len(todo) - n) * rate / 60:.0f}min", flush=True)
    print(f"done. {out_path}")


if __name__ == "__main__":
    main()
