"""Locate content the high-resolution output invented that the base image does not have.

WHY THIS EXISTS
---------------
Every paper in this line reports that the failure it cares about -- an object
hallucinated into the upscaled image -- is invisible to the metrics.  HiWave
prints the receipt itself: for the wedding photo in its Fig. 9, the version
WITH a hallucinated second couple scores 0.0443 on ImageReward and the clean
version scores 0.0168.  The worse image wins.  So HiWave falls back to a 548-
response user study, Pixelsmith hand-picks figures, and ResDiT (CVPR 2026, a
different architecture and a different paradigm) runs a 20-person study.  Three
papers, two architectures, two paradigms, no automatic detector.

We have no annotator pool, so a user study is not available to us.  That leaves
exactly one option: measure it.  And the two-stage line hands us the ground
truth for free -- the base image says what was supposed to be there.  A
legitimate detail (a pore, a thread) lives above the base image's Nyquist limit
and vanishes when the output is resampled back down.  A hallucinated apple does
not.

WHAT MAKES THIS HARD, AND WHY A PIXEL DIFF WILL NOT DO
------------------------------------------------------
HiWave's own Fig. 11 shows the trap.  Its two failing ablations grow two
floating apples, but they ALSO redraw the curtains and shift the window mullion.
A plain |base - output| is large over that whole window.  So the quantity has to
separate three things that all move pixels:

    (a) new object          -- compact, strong, and structurally unrelated
    (b) global drift        -- diffuse, spread over a large region
    (c) added fine detail   -- gone after resampling to base resolution

We handle (c) by comparing at base resolution, (b) by scoring compactness rather
than total energy, and photometric drift (recolouring, exposure) by normalising
each window before comparing -- z-scored windows make the comparison care about
structure, not about brightness or contrast.

STATUS
------
v0, positive control only.  This file does NOT yet claim to be a metric.  The
one question it has to answer first is: on a failure the incumbent itself
labelled, does the quantity fire where the object is and stay quiet elsewhere?
If it cannot do that on a floating apple, the whole direction is dead and we
learned it without a GPU.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        # Area-average, not bilinear: we are asking what survives a resolution
        # reduction, so the resampler has to be the one that actually band-limits.
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a


def windows(a, win, stride):
    """Sliding windows as a view; a is (H, W) or (H, W, C)."""
    H, W = a.shape[:2]
    ys = range(0, H - win + 1, stride)
    xs = range(0, W - win + 1, stride)
    return ys, xs


def ncc_map(base, out, win=48, stride=8, eps=1e-4):
    """Per-window normalised cross-correlation between base and output.

    Each window is z-scored before the dot product, so a window that merely got
    brighter or more contrasty scores 1.0.  Only a change in *structure* moves
    the number.  Returns (dissimilarity map, y positions, x positions) with
    dissimilarity = 1 - NCC in [0, 2].

    A window whose base content is nearly flat (blank sky, blank wall) has no
    structure to correlate; its NCC is dominated by noise.  Those windows are
    marked with NaN rather than silently reported as "changed" -- the same
    mistake we made once already, when smooth content with nothing to retain was
    scored as content destroyed.
    """
    ys, xs = windows(base, win, stride)
    D = np.full((len(list(ys)), len(list(xs))), np.nan, dtype=np.float32)
    ys, xs = windows(base, win, stride)
    ys, xs = list(ys), list(xs)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            b = base[y:y + win, x:x + win]
            o = out[y:y + win, x:x + win]
            bs, os_ = b.std(), o.std()
            if bs < eps and os_ < eps:
                continue                      # both flat: nothing was invented
            if bs < eps or os_ < eps:
                D[i, j] = 1.0                 # one side gained/lost all structure
                continue
            zb = (b - b.mean()) / bs
            zo = (o - o.mean()) / os_
            D[i, j] = 1.0 - float((zb * zo).mean())
    return D, np.array(ys), np.array(xs)


def blobs(D, thr, min_cells=2):
    """4-connected components of D > thr, as (cells, mean strength, bbox)."""
    M = np.nan_to_num(D, nan=0.0) > thr
    H, W = M.shape
    seen = np.zeros_like(M)
    out = []
    for i in range(H):
        for j in range(W):
            if not M[i, j] or seen[i, j]:
                continue
            stack, comp = [(i, j)], []
            seen[i, j] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and M[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) < min_cells:
                continue
            v = [D[y, x] for y, x in comp]
            yy = [c[0] for c in comp]
            xx = [c[1] for c in comp]
            out.append({"cells": len(comp), "strength": float(np.mean(v)),
                        "bbox": (min(yy), min(xx), max(yy), max(xx))})
    return sorted(out, key=lambda b: -b["cells"] * b["strength"])


def compare(base_path, out_path, win, stride, thr):
    base = load_gray(base_path)
    size = (base.shape[1], base.shape[0])
    out = load_gray(out_path, size=size)
    bg = base.mean(axis=2)
    og = out.mean(axis=2)
    D, ys, xs = ncc_map(bg, og, win, stride)
    valid = ~np.isnan(D)
    stats = {
        "mean": float(np.nanmean(D)),
        "p95": float(np.nanpercentile(D, 95)),
        "frac_over_thr": float((np.nan_to_num(D, nan=0.0) > thr).sum() / max(valid.sum(), 1)),
    }
    return D, ys, xs, stats, blobs(D, thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--save-maps", default=None,
                    help="directory to write dissimilarity heatmaps into")
    a = ap.parse_args()

    print(f"base = {a.base}   win={a.win} stride={a.stride} thr={a.thr}\n")
    hdr = f"{'output':<26}{'meanD':>8}{'p95D':>8}{'>thr%':>8}   top blobs (cells, strength, bbox)"
    print(hdr)
    print("-" * len(hdr))
    for op in a.outs:
        D, ys, xs, s, bl = compare(a.base, op, a.win, a.stride, a.thr)
        top = "  ".join(f"({b['cells']}, {b['strength']:.2f}, {b['bbox']})" for b in bl[:3])
        print(f"{Path(op).stem:<26}{s['mean']:8.3f}{s['p95']:8.3f}"
              f"{100 * s['frac_over_thr']:8.1f}   {top or '-'}")
        if a.save_maps:
            d = Path(a.save_maps); d.mkdir(parents=True, exist_ok=True)
            M = np.nan_to_num(D, nan=0.0)
            M = np.clip(M / max(M.max(), 1e-6), 0, 1)
            base_im = Image.open(a.base).convert("RGB")
            heat = Image.fromarray((M * 255).astype(np.uint8)).resize(
                base_im.size, Image.NEAREST).convert("RGB")
            side = Image.new("RGB", (base_im.width * 3, base_im.height))
            side.paste(base_im, (0, 0))
            side.paste(Image.open(op).convert("RGB").resize(base_im.size), (base_im.width, 0))
            side.paste(heat, (base_im.width * 2, 0))
            side.save(d / f"{Path(op).stem}_map.png")


if __name__ == "__main__":
    main()
