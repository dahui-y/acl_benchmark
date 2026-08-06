"""Contact sheets + per-bucket detail-retention numbers for the StyleSSP probe.

Reads pairs.jsonl (from stratify.py) plus StyleSSP outputs, and answers one
question per bucket cell: HOW MUCH OF THE CONTENT'S FINE STRUCTURE SURVIVED.

THE MEASUREMENT, AND WHY IT IS A DIFFERENCE AND NOT AN ABSOLUTE

An absolute "detail retention" number is uninterpretable: a stylised image is
SUPPOSED to lose some content detail, that is what stylisation means. The claim
under test is comparative:

    detail loss should grow with the content's high-frequency energy,
    and grow further when the style is also high-frequency,
    because a single global per-frequency scalar on z_T cannot keep the
    content's fine structure while damping the style's.

So the readout is the GAP between bucket cells (hiC x hiS versus loC x loS),
not any single value. And the cleanest control is internal: rerun with
alpha = 1.0, which disables frequency manipulation entirely while leaving every
other component of StyleSSP untouched. If the loss is caused by the filter, it
must shrink at alpha = 1.0. If it does not, the filter is not the cause and the
hypothesis is dead regardless of how bad the numbers look.

Everything here is numpy + PIL. No LPIPS, no CLIP, no download -- the server has
no route to a model hub, and a probe that depends on a download that may not
arrive is not a probe. These are proxies; the sheets are the primary evidence.

    python sheet.py --pairs pairs.jsonl --out-dir /path/to/stylessp/results
    python sheet.py --pairs pairs.jsonl --out-dir RES --ablation RES_alpha1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _lum(path, size):
    return np.asarray(Image.open(path).convert("L").resize((size, size),
                                                           Image.LANCZOS),
                      dtype=np.float64) / 255.0


def band_map(path, size=256, lo=0.5, hi=1.45):
    """Energy map of one frequency band, as a spatial image.

    Band-limited rather than a plain high-pass: we want the band StyleSSP's
    filter actually attenuates, so the measurement and the mechanism refer to
    the same frequencies.
    """
    a = _lum(path, size)
    f = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt(((yy - size / 2) / (size / 2)) ** 2
                + ((xx - size / 2) / (size / 2)) ** 2)
    f[(r <= lo) | (r > hi)] = 0
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))


def retention(out_path, content_path, size=256):
    """Correlation between output and content energy in the attenuated band.

    Zero-meaned and unit-normed, so this is blind to overall contrast: an
    output that keeps the content's fine structure WHERE the content has it
    scores high even if the stylisation changed the palette entirely.
    """
    a, b = band_map(out_path, size), band_map(content_path, size)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b),
                                     1e-12))


def find_output(out_dir, row):
    """StyleSSP names results ours_{content_stem}_{style_stem}.png."""
    c, s = Path(row["content"]).stem, Path(row["style"]).stem
    for cand in (f"ours_{c}_{s}.png", f"{c}_{s}.png", f"{row['id']}.png"):
        p = out_dir / cand
        if p.exists():
            return p
    hits = list(out_dir.glob(f"*{c}*{s}*"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path,
                    default=Path(__file__).parent / "pairs.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="StyleSSP results dir (alpha=0.7, the released value)")
    ap.add_argument("--ablation", type=Path,
                    help="results dir from the alpha=1.0 rerun -- the internal "
                         "control that isolates the frequency manipulation")
    ap.add_argument("--sheets", type=Path,
                    default=Path(__file__).parent / "sheets")
    ap.add_argument("--tile", type=int, default=300)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l]
    args.sheets.mkdir(parents=True, exist_ok=True)

    missing, recs = [], []
    for r in rows:
        out = find_output(args.out_dir, r)
        if out is None:
            missing.append(r["id"])
            continue
        rec = dict(r, out=str(out),
                   ret=retention(out, r["content"]))
        if args.ablation:
            abl = find_output(args.ablation, r)
            rec["ret_a1"] = retention(abl, r["content"]) if abl else None
            rec["out_a1"] = str(abl) if abl else None
        recs.append(rec)

    if missing:
        print(f"missing {len(missing)} output(s): {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
    if not recs:
        raise SystemExit("no outputs found -- check --out-dir")

    (args.sheets / "retention.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs))

    # ------------------------------------------------------------ per-cell
    cells = defaultdict(list)
    for r in recs:
        cells[(r["content_bucket"], r["style_bucket"])].append(r)

    has_abl = args.ablation and any(r.get("ret_a1") is not None for r in recs)
    print(f"\ndetail retention in the band StyleSSP attenuates "
          f"(1.0 = content's fine structure fully kept)\n")
    hdr = f"{'cell':<12}{'n':>3}{'retention':>12}"
    if has_abl:
        hdr += f"{'alpha=1.0':>12}{'delta':>9}"
    print(hdr)
    print("-" * len(hdr))
    grid = {}
    for cb in ("lo", "mid", "hi"):
        for sb in ("lo", "mid", "hi"):
            v = cells.get((cb, sb), [])
            if not v:
                continue
            m = float(np.mean([r["ret"] for r in v]))
            grid[(cb, sb)] = m
            line = f"{cb}C x {sb}S{'':<3}{len(v):>3}{m:>12.3f}"
            if has_abl:
                a = [r["ret_a1"] for r in v if r.get("ret_a1") is not None]
                if a:
                    ma = float(np.mean(a))
                    line += f"{ma:>12.3f}{m - ma:>+9.3f}"
            print(line)

    print("\nverdict:")
    if not has_abl:
        print("  UNATTRIBUTED: no alpha=1.0 control. Retention across cells "
              "is dominated by how much\n  fine structure the CONTENT had in "
              "the first place, so raw cell values cannot separate\n  "
              "'the filter destroyed detail' from 'there was no detail'. "
              "Rerun with alpha=1.0 and\n  pass --ablation before reading "
              "anything into the table above.")
        return

    # The readout is the ablation delta per cell, not a cross-cell comparison
    # of absolute retention. A smooth content image has almost no energy in
    # this band, so its retention is near zero however well the method behaves
    # -- comparing loC against hiC directly measures the inputs, not the
    # method. Cells whose alpha=1.0 retention is itself near zero are dropped
    # for the same reason: with nothing to keep, "was it kept" is undefined.
    FLOOR = 0.2
    per_cell = {}
    for (cb, sb), v in cells.items():
        d = [(r["ret_a1"] - r["ret"], r["ret_a1"]) for r in v
             if r.get("ret_a1") is not None]
        d = [x for x, base in d if base >= FLOOR]
        if d:
            per_cell[(cb, sb)] = float(np.mean(d))
    dropped = len(cells) - len(per_cell)
    if dropped:
        print(f"  ({dropped} cell(s) dropped: alpha=1.0 retention below "
              f"{FLOOR}, nothing to lose there)")
    if not per_cell:
        print("  every cell dropped -- the content set has too little energy "
              "in this band to test on.")
        return

    hot = max(per_cell.values())
    cold = min(per_cell.values())
    worst = max(per_cell, key=per_cell.get)
    print(f"  largest loss from the filter: {worst[0]}C x {worst[1]}S "
          f"({hot:+.3f});  smallest: {cold:+.3f}")

    # Marginals, because a single argmax over nine cells with a handful of
    # pairs each is noise. The prediction is directional -- loss should grow
    # with BOTH factors -- so the marginal trend is the claim, and it says
    # which factor actually drives it rather than just naming a winning cell.
    def marginal(axis):
        out = {}
        for b in ("lo", "mid", "hi"):
            v = [d for (cb, sb), d in per_cell.items()
                 if (cb if axis == 0 else sb) == b]
            if v:
                out[b] = float(np.mean(v))
        return out

    mc, ms = marginal(0), marginal(1)
    print("  marginal loss by content HF: "
          + "  ".join(f"{k} {v:+.3f}" for k, v in mc.items()))
    print("  marginal loss by style   HF: "
          + "  ".join(f"{k} {v:+.3f}" for k, v in ms.items()))
    trend_c = (mc.get("hi", 0) - mc.get("lo", 0)) if len(mc) > 1 else 0.0
    trend_s = (ms.get("hi", 0) - ms.get("lo", 0)) if len(ms) > 1 else 0.0
    print(f"  trend hi-minus-lo:  content {trend_c:+.3f}   style {trend_s:+.3f}"
          f"   (both positive = predicted pattern)")

    if hot < 0.10:
        print("  The frequency manipulation costs almost no fine detail "
              "anywhere. Predicted failure\n  absent -- the lead dies here, "
              "which is the cheap outcome this probe exists to buy.")
    elif hot - cold < 0.10:
        print("  The filter costs detail, but EVENLY across cells. That is a "
              "uniform tax, not the\n  spectral-competition failure predicted: "
              "no evidence that content and style are\n  fighting for the same "
              "band. Lead dies in its interesting form; at most an errata.")
    elif trend_c > 0.05 and trend_s > 0.05:
        print("  Loss grows with BOTH the content's and the style's "
              "high-frequency energy -- the\n  predicted pattern.")
        print("  This supports the claim a single global per-frequency scalar "
              "cannot separate content\n  detail from style texture when they "
              "share a band. Confirm on the sheets, then the next\n  step is "
              "whether a better filter fixes it (errata) or nothing on the "
              "frequency axis can\n  (mechanism limit, and the only version "
              "worth a paper).")
    else:
        driver = "style" if trend_s > trend_c else "content"
        print(f"  Loss is real but only one factor drives it ({driver}); the "
              f"other trend is flat or\n  negative. The spectral-COMPETITION "
              f"story needs both. Something is there, but it is not\n  the "
              f"hypothesis as stated -- diagnose before building anything.")

    # ------------------------------------------------------------- sheets
    cols = ["content", "style", "out"] + (["out_a1"] if has_abl else [])
    labels = {"content": "content", "style": "style",
              "out": "StyleSSP (alpha=0.7)", "out_a1": "alpha=1.0 (no freq)"}
    for cb in ("lo", "mid", "hi"):
        sel = [r for r in recs if r["content_bucket"] == cb]
        if not sel:
            continue
        t, pad, head = args.tile, 30, 30
        canvas = Image.new("RGB", (t * len(cols),
                                   head + len(sel) * (t + pad)), "white")
        d = ImageDraw.Draw(canvas)
        for j, c in enumerate(cols):
            d.text((j * t + 6, 8), labels[c], fill="black")
        for i, r in enumerate(sel):
            y = head + i * (t + pad)
            for j, c in enumerate(cols):
                p = r.get(c)
                if p and Path(p).exists():
                    canvas.paste(Image.open(p).convert("RGB").resize((t, t)),
                                 (j * t, y))
            d.text((6, y + t + 8),
                   f"{r['id']}   content_hf={r['content_hf']:.4f}  "
                   f"style_hf={r['style_hf']:.4f}  retention={r['ret']:.3f}",
                   fill="black")
        out = args.sheets / f"sheet_content_{cb}.png"
        canvas.save(out)
        print(f"  sheet -> {out}")

    print("\nLook at the hi-content sheet first, and specifically at thin "
          "structures -- railings,\nwires, text, hair, dense foliage. The "
          "number is a proxy; those are the evidence.")


if __name__ == "__main__":
    main()
