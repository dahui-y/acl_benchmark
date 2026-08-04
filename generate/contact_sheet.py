"""Lay one item's conditions out as a grid: rows are conditions, columns are time.

This is the OSCBench Figure 4 layout, and it is the only view that shows what
this design is actually asking. Every row shares an item, a seed and therefore
the initial noise, so a difference between two rows can only have come from the
prompt -- reading down a column is reading the effect of the grammar directly.

Two uses, and the same picture serves both: eyeballing whether the prompts
render into something judgeable at all, and the qualitative figure in the paper.

Usage:
    python contact_sheet.py --videos /data/videos/wan2.2-ti2v-5b-480p --item 0
    python contact_sheet.py --videos ... --item 0 --conditions prog perf result
    python contact_sheet.py --videos ... --all-items --out-dir sheets/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "probe"))

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_W = 300          # left gutter holding the condition name and its prompt
HEADER_H = 46
PAD = 4

# The default row set is the story in order: the aspect axis first, then one
# telicity row for contrast, then the verb swap as the upper anchor. The full
# 15 rows are available with --conditions all but are hard to read at once.
DEFAULT_CONDITIONS = ["prog", "perf", "result", "prospective", "failed",
                      "phase_finish", "phase_stop", "telic_plural", "other_verb"]


def sample_frames(video_path, n):
    """Evenly spaced frames including the first and the last -- the same
    linspace rule extract_frames.py and OSCBench use, so what you look at here
    is a subset of what the annotator will see."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    wanted = list(dict.fromkeys(np.linspace(0, total - 1, n, dtype=int).tolist()))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def build_sheet(video_dir, item_id, conditions, seed, prompts, n_frames, cell_w,
                title):
    rows, missing = [], []
    for cond in conditions:
        path = video_dir / f"item{item_id:04d}" / f"{cond}__seed{seed}.mp4"
        if not path.exists():
            missing.append(cond)
            continue
        frames = sample_frames(path, n_frames)
        if not frames:
            missing.append(cond)
            continue
        h, w = frames[0].shape[:2]
        cell_h = int(cell_w * h / w)
        strip = [cv2.resize(f, (cell_w, cell_h)) for f in frames]
        while len(strip) < n_frames:  # a short video still gets a full-width row
            strip.append(np.zeros((cell_h, cell_w, 3), np.uint8))

        gutter = np.full((cell_h, LABEL_W, 3), 32, np.uint8)
        cv2.putText(gutter, cond, (10, 26), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(wrap(prompts.get(cond, ""), 38)[:4]):
            cv2.putText(gutter, line, (10, 52 + i * 20), FONT, 0.42,
                        (170, 170, 170), 1, cv2.LINE_AA)

        cells = [np.pad(c, ((0, PAD), (0, PAD), (0, 0))) for c in [gutter] + strip]
        row = np.hstack(cells)
        rows.append(row)

    if not rows:
        return None, conditions

    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    body = np.vstack(rows)

    header = np.full((HEADER_H, width, 3), 20, np.uint8)
    cv2.putText(header, title, (10, 30), FONT, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, body]), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True,
                    help="a model directory produced by generate.py")
    ap.add_argument("--item", type=int, help="item id; omit with --all-items")
    ap.add_argument("--all-items", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                    help="'all' for every video condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=5, help="columns")
    ap.add_argument("--cell-width", type=int, default=300)
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent.parent / "probe" / "stimuli.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("sheets"))
    args = ap.parse_args()

    items = {r["item_id"]: r for r in
             (json.loads(l) for l in args.stimuli.read_text().splitlines() if l)}

    if args.conditions == ["all"]:
        from generate import VIDEO_CONDITIONS  # noqa: PLC0415
        conditions = VIDEO_CONDITIONS
    else:
        conditions = args.conditions

    if args.all_items:
        found = sorted(int(p.name.removeprefix("item"))
                       for p in args.videos.glob("item*") if p.is_dir())
    elif args.item is not None:
        found = [args.item]
    else:
        ap.error("pass --item N or --all-items")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item_id in found:
        item = items.get(item_id)
        if item is None:
            print(f"item {item_id} is not in {args.stimuli.name}; skipping")
            continue
        title = (f"item {item_id}  |  {item['gerund']} + {item['noun']}  |  "
                 f"{item['aspectual_class']}  |  seed {args.seed}  |  "
                 f"{args.videos.name}")
        sheet, missing = build_sheet(args.videos, item_id, conditions, args.seed,
                                     item["texts"], args.frames, args.cell_width,
                                     title)
        if sheet is None:
            print(f"item {item_id}: no videos found for {missing}")
            continue
        out = args.out_dir / f"item{item_id:04d}_seed{args.seed}.png"
        cv2.imwrite(str(out), sheet)
        note = f"  (missing: {', '.join(missing)})" if missing else ""
        print(f"{out}  {sheet.shape[1]}x{sheet.shape[0]}{note}")


if __name__ == "__main__":
    main()
