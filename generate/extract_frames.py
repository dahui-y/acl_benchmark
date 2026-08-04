"""Sample frames from the generated videos, for annotation and for MLLM eval.

Sampling is `np.linspace(0, total-1, num_frames)`, identical to OSCBench's
`extract_frames.py` at the repo root -- same count, same spacing, so any number
we report next to theirs was computed off comparable evidence. That formula
already includes frame 0 and the final frame, which matters here more than it
did for them: our judgement is anchored on the first frame (has the event
started?) and the last (has the object reached its target state?).

Layout mirrors the video tree so a frame folder is addressable by the same
(model, item, condition, seed) key as the manifest:

    frames/<model>/item0007/prog__seed42/frame_001.jpg ...

Usage:
    python extract_frames.py --videos /data/videos/wan2.2-ti2v-5b \
                             --out /data/frames/wan2.2-ti2v-5b --seeds 42
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def extract(video_path, out_folder, num_frames=20):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    wanted = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())

    out_folder.mkdir(parents=True, exist_ok=True)
    written, frame_id = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in wanted:
            cv2.imwrite(str(out_folder / f"frame_{written + 1:03d}.jpg"), frame)
            written += 1
        frame_id += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True,
                    help="a model directory produced by generate.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="only these seeds; annotation covers one seed, the rest "
                         "go to MLLM eval only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos = sorted(args.videos.rglob("*.mp4"))
    if args.seeds is not None:
        keep = {f"seed{s}" for s in args.seeds}
        videos = [v for v in videos if v.stem.split("__")[-1] in keep]
    if not videos:
        raise SystemExit(f"no videos under {args.videos}")

    index, skipped, short = [], 0, []
    for video in videos:
        rel = video.relative_to(args.videos).with_suffix("")
        folder = args.out / rel
        if folder.exists() and not args.overwrite:
            skipped += 1
            continue
        n = extract(video, folder, args.num_frames)
        if n < args.num_frames:
            short.append((str(rel), n))
        condition, _, seed = video.stem.partition("__seed")
        index.append({"video": str(video), "frames": str(folder),
                      "item_id": int(rel.parts[0].removeprefix("item")),
                      "condition": condition, "seed": int(seed), "n_frames": n})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "frames_index.jsonl").open("a") as fh:
        for rec in index:
            fh.write(json.dumps(rec) + "\n")

    print(f"extracted {len(index)} videos, skipped {skipped} already done")
    if short:
        # A truncated or unreadable file yields fewer frames than asked for and
        # would silently give the annotator less evidence for that cell.
        print(f"  {len(short)} videos yielded fewer than {args.num_frames} frames:")
        for name, n in short[:10]:
            print(f"    {name}: {n}")


if __name__ == "__main__":
    main()
