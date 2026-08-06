"""Read the veto: contact sheets first, numbers second.

Deliberately model-free. Every descriptor here is computed from pixels with
numpy alone -- no CLIP, no VGG, no DINO -- because the server this runs on
cannot reach a model hub, and because a veto that depends on a download that
might not arrive is not a veto. The cost is that these are proxies, so the
contact sheets are the primary evidence and the table is corroboration. If the
sheets and the table disagree, the sheets win.

Three quantities per condition, all against the two references produced by the
`none` run at the same seed and the same initial noise:

  S  style distance to the style image      texture Gram + colour histogram.
                                            LOWER means the look came across.
  C  structure similarity to the content    gradient-map correlation.
                                            HIGHER means the content survived.
  L  structure similarity to the style       HIGHER means the style image's
                                            own object leaked in.

A condition only counts as transferring style if S drops AND C stays up AND L
does not rise. Any one of the three alone is easy to fake: a grey mush scores
well on L, and a copy of the style image scores perfectly on S.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prompts import LEAK_OBJECT, PAIRS

HERE = Path(__file__).parent
COLS = ["content", "style", "img", "txt", "both"]


# --------------------------------------------------------------------------
# descriptors


def _gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _oriented(g):
    """Four oriented first-difference responses: -, |, / and \\."""
    return np.stack([
        g - np.roll(g, 1, axis=1),
        g - np.roll(g, 1, axis=0),
        g - np.roll(np.roll(g, 1, axis=0), 1, axis=1),
        g - np.roll(np.roll(g, 1, axis=0), -1, axis=1),
    ])[:, 1:-1, 1:-1]


def style_descriptor(img, levels=3, bins=8):
    """Texture Gram over a 3-level pyramid, plus an RGB histogram.

    The Gram of oriented filter responses is the classical style statistic --
    the same quantity Gatys' style loss uses, minus the learned features. It is
    what makes impasto different from flat woodblock colour independent of what
    is depicted. The colour histogram is kept separate because palette transfer
    and texture transfer fail independently.
    """
    a = np.asarray(img.convert("RGB").resize((256, 256), Image.LANCZOS),
                   dtype=np.float64) / 255.0
    g = _gray(a)
    grams = []
    for _ in range(levels):
        r = _oriented(g)
        flat = r.reshape(4, -1)
        gram = flat @ flat.T / flat.shape[1]
        grams.append(gram.ravel())
        g = 0.25 * (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2])
    tex = np.concatenate(grams)
    tex = tex / (np.linalg.norm(tex) + 1e-12)

    idx = np.clip((a * bins).astype(int), 0, bins - 1)
    hist = np.bincount(
        (((idx[..., 0] * bins + idx[..., 1]) * bins + idx[..., 2])).ravel(),
        minlength=bins ** 3).astype(np.float64)
    hist = hist / (np.linalg.norm(hist) + 1e-12)
    return tex, hist


def style_distance(d1, d2):
    """Cosine distance, texture and colour weighted equally."""
    tex = 1.0 - float(d1[0] @ d2[0])
    col = 1.0 - float(d1[1] @ d2[1])
    return 0.5 * (tex + col)


def _box(m, k=3):
    o = np.zeros_like(m)
    for dy in range(-(k // 2), k // 2 + 1):
        for dx in range(-(k // 2), k // 2 + 1):
            o += np.roll(np.roll(m, dy, 0), dx, 1)
    return o / k ** 2


def structure(img, n=64, k=3):
    """Zero-mean unit-norm gradient magnitude -- correlate two of these to get
    a layout similarity that is largely blind to palette and brush texture.

    n and k are not free choices. At 128px with no blur, adding high-frequency
    texture to an image -- which is exactly what stylisation does -- drags the
    self-similarity of a smooth image down to nothing, so a correctly stylised
    output would score as "content destroyed". Coarsening to 64px and blurring
    the magnitude map fixes that (0.975 -> 0.997 on a texture-added control)
    while still separating two different layouts (~0.03). Going coarser than
    48px starts calling unrelated layouts similar, so this is the floor.
    """
    a = np.asarray(img.convert("RGB").resize((n, n), Image.LANCZOS),
                   dtype=np.float64) / 255.0
    r = _oriented(_gray(a))
    m = _box(np.sqrt(r[0] ** 2 + r[1] ** 2), k)
    m = m - m.mean()
    return m / (np.linalg.norm(m) + 1e-12)


def struct_sim(a, b):
    return float((a * b).sum())


# --------------------------------------------------------------------------
# sheets


def sheet(root, out, seed, tile=320):
    rows = []
    for pid, _, _ in PAIRS:
        row = []
        for c in COLS:
            p = root / f"{pid}__seed{seed}__{c}.png"
            row.append(Image.open(p).convert("RGB").resize((tile, tile))
                       if p.exists() else None)
        rows.append((pid, row))

    pad, head = 26, 30
    W = tile * len(COLS)
    H = head + len(rows) * (tile + pad)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for j, c in enumerate(COLS):
        d.text((j * tile + 6, 8), {"content": "content (baseline)",
                                   "style": "style reference",
                                   "img": "swap IMAGE-stream KV",
                                   "txt": "swap TEXT-stream KV",
                                   "both": "swap both"}[c], fill="black")
    for i, (pid, row) in enumerate(rows):
        y = head + i * (tile + pad)
        for j, im in enumerate(row):
            if im is not None:
                canvas.paste(im, (j * tile, y))
        d.text((6, y + tile + 6),
               f"{pid}   (leak probe: does a {LEAK_OBJECT[pid]} appear?)",
               fill="black")
    canvas.save(out)
    return out


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sd35")
    ap.add_argument("--images", type=Path, default=HERE / "images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    root = args.images / args.model
    if not root.is_dir():
        raise SystemExit(f"no images at {root} -- run run.py first")

    rows = []
    for pid, _, _ in PAIRS:
        for seed in args.seeds:
            paths = {c: root / f"{pid}__seed{seed}__{c}.png" for c in COLS}
            if not all(p.exists() for p in paths.values()):
                missing = [c for c, p in paths.items() if not p.exists()]
                print(f"  skipping {pid} seed{seed}: missing {missing}")
                continue
            imgs = {c: Image.open(p) for c, p in paths.items()}
            dsc = {c: style_descriptor(im) for c, im in imgs.items()}
            st = {c: structure(im) for c, im in imgs.items()}
            for c in ("content", "img", "txt", "both"):
                rows.append({
                    "pair": pid, "seed": seed, "cond": c,
                    "S": style_distance(dsc[c], dsc["style"]),
                    "C": struct_sim(st[c], st["content"]),
                    "L": struct_sim(st[c], st["style"]),
                })

    if not rows:
        raise SystemExit("nothing to analyse")

    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    def agg(cond, key):
        v = [r[key] for r in rows if r["cond"] == cond]
        return float(np.mean(v)), float(np.std(v))

    base_S = agg("content", "S")[0]
    base_L = agg("content", "L")[0]

    print(f"\n{args.model}   n = {len(rows) // 4} (pair x seed)\n")
    print(f"{'condition':<26}{'S (style dist)':>16}{'dS vs base':>12}"
          f"{'C (content)':>14}{'L (leak)':>11}")
    print("-" * 79)
    for c, label in (("content", "baseline (no swap)"),
                     ("img", "swap IMAGE-stream KV"),
                     ("txt", "swap TEXT-stream KV"),
                     ("both", "swap both")):
        s, ss = agg(c, "S")
        cc = agg(c, "C")[0]
        ll = agg(c, "L")[0]
        d = base_S - s
        print(f"{label:<26}{s:>10.4f}±{ss:<5.3f}{d:>+12.4f}"
              f"{cc:>14.3f}{ll:>11.3f}")

    d_img = base_S - agg("img", "S")[0]
    d_txt = base_S - agg("txt", "S")[0]
    c_img, c_txt = agg("img", "C")[0], agg("txt", "C")[0]
    l_img = agg("img", "L")[0] - base_L
    l_txt = agg("txt", "L")[0] - base_L

    print("\nverdict (proxy metrics -- confirm against the contact sheets):")
    print(f"  leakage change vs baseline: img {l_img:+.3f}, txt {l_txt:+.3f} "
          f"(rising = the style image's own object is coming through)")

    # A condition that destroyed the layout did not transfer style, it copied
    # the style image. Say so instead of scoring the copy as a success.
    collapsed = [n for n, c in (("img", c_img), ("txt", c_txt)) if c < 0.5]
    if collapsed:
        print(f"  CAUTION: {collapsed} lost the content layout (C < 0.5). "
              "Their S drop is a copy of the\n  style image, not a transfer. "
              "Discount them and read the sheets.")

    thr, margin = 0.02, 0.25
    best = max(d_img, d_txt)
    if best < thr:
        print("  NEITHER stream moves style. K/V substitution is not the "
              "mechanism in MMDiT at all.\n  That kills the port reading too: "
              "the finding becomes 'the U-Net recipe has no MMDiT\n  form' -- "
              "a result, but not the method this was scouting for.")
    elif abs(d_txt - d_img) < margin * best:
        print(f"  INCONCLUSIVE: both streams move it about equally "
              f"(img {d_img:+.4f}, txt {d_txt:+.4f}).")
        print("  The 2-way test cannot separate them -- most likely the two "
              "streams are entangled\n  through the joint softmax. This is the "
              "one outcome that justifies building the 4-way\n  processor "
              "version. It is NOT a green light on its own.")
    elif d_txt > d_img:
        print(f"  TEXT stream carries it (dS {d_txt:+.4f} vs img {d_img:+.4f}).")
        print("  U-Net has no counterpart -> the mechanism is genuinely new. "
              "PROCEED to the 4-way\n  processor version and a real style "
              "benchmark.")
    else:
        print(f"  IMAGE stream carries it (dS {d_img:+.4f} vs txt "
              f"{d_txt:+.4f}).")
        print("  That is the direct analogue of StyleID's self-attention swap "
              "-> a method built on\n  this is a PORT. Direction dies here, "
              "as agreed.")

    for seed in args.seeds:
        p = sheet(root, root / f"sheet_seed{seed}.png", seed)
        print(f"  sheet -> {p}")
    print("\nLook at the sheets before believing the table.")


if __name__ == "__main__":
    main()
