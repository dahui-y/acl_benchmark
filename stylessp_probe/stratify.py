"""Build a spectrum-stratified (content, style) pair list for the StyleSSP probe.

WHY STRATIFY BY SPECTRUM AND NOT AT RANDOM

StyleSSP's whole content-preservation mechanism is one global, per-frequency
scalar applied to the inverted latent z_T. Its paper argues z_T's HIGH
frequencies carry layout, so it keeps them and damps the low ones by alpha=0.7.
The released code does something the paper never states -- it uses the band-pass
variant, whose effective multiplier we computed from the released parameters:

    DC 0.702  |  mid-band 0.898  |  top band 0.823

i.e. it also damps by ~18-20% exactly the band the paper says carries layout.

If content detail and style texture live in the SAME band, a single global
scalar there cannot keep one and drop the other. A random sample of pairs
averages that tension away; stratifying by measured spectral content is what
makes it visible. So every pair here carries two measured numbers -- the
content's high-frequency energy ratio and the style's -- and the pairs are
bucketed on both.

The stratifier is MEASURED, not asserted: buckets come from the images'
own spectra, so this works on whatever content/style folders exist and does not
depend on us guessing which picture counts as "detailed".

    python stratify.py --content DIR --style DIR --out pairs.jsonl
    python stratify.py --content DIR --style DIR --per-bucket 4   # bigger set
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def hf_ratio(path, size=256, cut=0.5):
    """Fraction of spectral energy above a normalised radius.

    Computed on the luminance channel at a fixed size so the number is
    comparable across images of different resolutions. DC is excluded: it is
    mean brightness, carries no structure, and would otherwise dominate and
    compress every image into the same bucket.
    """
    a = np.asarray(Image.open(path).convert("L").resize((size, size),
                                                        Image.LANCZOS),
                   dtype=np.float64) / 255.0
    f = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    p = np.abs(f) ** 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt(((yy - size / 2) / (size / 2)) ** 2
                + ((xx - size / 2) / (size / 2)) ** 2)
    return float(p[r > cut].sum() / max(p.sum(), 1e-12))


def bucket(values, names=("lo", "mid", "hi")):
    """Terciles of the observed distribution, not fixed thresholds.

    Fixed thresholds would put every photo in one bucket for one dataset and
    spread them for another; terciles guarantee the design is balanced whatever
    folders get pointed at, and the actual cut points are reported so the
    stratification stays auditable.
    """
    v = np.asarray(values)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    out = [names[0] if x <= q1 else names[1] if x <= q2 else names[2] for x in v]
    return out, (float(q1), float(q2))


def listdir(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True, type=Path)
    ap.add_argument("--style", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "pairs.jsonl")
    ap.add_argument("--per-bucket", type=int, default=3,
                    help="pairs per (content-bucket x style-bucket) cell")
    ap.add_argument("--cut", type=float, default=0.5,
                    help="normalised radius separating high from low frequency")
    args = ap.parse_args()

    cont = listdir(args.content)
    styl = listdir(args.style)
    if not cont or not styl:
        raise SystemExit(f"found {len(cont)} content and {len(styl)} style "
                         f"images -- check the directories")

    c_hf = [hf_ratio(p, cut=args.cut) for p in cont]
    s_hf = [hf_ratio(p, cut=args.cut) for p in styl]
    c_b, c_cuts = bucket(c_hf)
    s_b, s_cuts = bucket(s_hf)

    print(f"content: {len(cont)} images, HF ratio "
          f"{min(c_hf):.4f}-{max(c_hf):.4f}, terciles at {c_cuts[0]:.4f} / "
          f"{c_cuts[1]:.4f}")
    print(f"style:   {len(styl)} images, HF ratio "
          f"{min(s_hf):.4f}-{max(s_hf):.4f}, terciles at {s_cuts[0]:.4f} / "
          f"{s_cuts[1]:.4f}")

    by_c = {b: [] for b in ("lo", "mid", "hi")}
    for p, h, b in zip(cont, c_hf, c_b):
        by_c[b].append((p, h))
    by_s = {b: [] for b in ("lo", "mid", "hi")}
    for p, h, b in zip(styl, s_hf, s_b):
        by_s[b].append((p, h))

    rows = []
    for cb in ("lo", "mid", "hi"):
        for sb in ("lo", "mid", "hi"):
            # Deterministic: take the first k of each bucket in name order, so
            # the same folders always yield the same suite and a rerun is
            # comparable to the previous one.
            cs, ss = by_c[cb], by_s[sb]
            if not cs or not ss:
                print(f"  WARNING: cell {cb}x{sb} empty "
                      f"({len(cs)} content, {len(ss)} style)")
                continue
            for k in range(args.per_bucket):
                cp, ch = cs[k % len(cs)]
                sp, sh = ss[(k + ("lo", "mid", "hi").index(cb)) % len(ss)]
                rows.append({
                    "id": f"{cb}C_{sb}S_{k}",
                    "content": str(cp), "style": str(sp),
                    "content_hf": round(ch, 5), "style_hf": round(sh, 5),
                    "content_bucket": cb, "style_bucket": sb,
                })

    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\n{len(rows)} pairs -> {args.out}")
    print("\nThe cell that matters is hiC x hiS: fine content and textured "
          "style competing for the\nsame band. loC x loS is the control where "
          "nothing competes. If the failure is real,\nthe gap between those two "
          "cells is where it shows.")
    for cb in ("lo", "mid", "hi"):
        line = "  ".join(
            f"{cb}C x {sb}S: {sum(1 for r in rows if r['content_bucket'] == cb and r['style_bucket'] == sb)}"
            for sb in ("lo", "mid", "hi"))
        print("  " + line)


if __name__ == "__main__":
    main()
