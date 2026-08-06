"""Step-1 probe: does the MODULATION path carry style off the tradeoff line?

Background, in three sentences. Our sweep (sweep.py) showed K/V substitution in
SD3.5 has no transfer regime: all 24 cells sit on one line from no-effect to
pixel copy (corr(1-C, dS) = 0.874), so style bought through attention is paid
for in content, one for one. The one conditioning path those hooks never touched
is modulation: pooled embedding + timestep -> per-block scale/shift/gate, global
and spatially blind -- exactly the shape of "global appearance". Modulation
Guidance (ICLR 2026) has since shown text-driven shifts in this space change
appearance at the "car style" level, which raises the prior that the channel can
carry style.

WHAT THIS PROBE DOES AND DOES NOT ESTABLISH

It transplants the style branch's modulation outputs into the content branch and
asks one question: do any cells land OFF the K/V tradeoff line (large dS at high
C)? That validates or kills the CHANNEL.

It does NOT establish the image-derived-signal claim. Here the style branch is
text-prompted, so transplanting its modulation is mathematically equivalent to
feeding the style prompt's pooled embedding to the content generation -- a thing
any text method can do. Whether an IMAGE-derived modulation signal adds anything
beyond that is step 3 (real style images), and it only deserves GPU time if this
probe passes. Channel first, signal source second.

MECHANICS

Same recipe as run.py, hooks moved from attention K/V projections to the
modulation linears -- the attention maths and the norm maths stay untouched:

  joint blocks   block.norm1.linear          -> image-stream modulation
                 block.norm1_context.linear  -> text-stream modulation
                 (the final block's norm1_context is AdaLayerNormContinuous;
                  it still exposes .linear, so the same hook works)
  output norm    transformer.norm_out.linear -> final image-stream modulation
  FLUX singles   block.norm.linear           -> single-stream modulation

Both branches share one batch with INDEPENDENT noise (the shared-noise trap is
documented in run.generate). Batch layout [style, content] per chunk, so
out[1::2] = lam*out[0::2] + (1-lam)*out[1::2] with or without CFG. --selftest
proves the direction before anything else runs.

    python mod_probe.py --selftest --model sd35      # ~2 min, run first
    python mod_probe.py --model sd35                 # 60 runs, ~30-40 min
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from analyze import structure, struct_sim, style_descriptor, style_distance
from prompts import LEAK_OBJECT, PAIRS
from run import MODELS, load_pipeline

HERE = Path(__file__).parent

# Sweep round 2 measured the K/V tradeoff line's copy-end at dS = 0.467
# (img__all__allsteps, the strongest cell). The probe's residuals are computed
# against a line anchored there so the two experiments share one ruler. If the
# sweep metrics file is present its value is re-derived instead of trusted.
KV_DS_MAX_FALLBACK = 0.467
OFF_LINE = dict(residual=0.15, C=0.5, dL=0.3)   # same bar as sweep.py

# (name, streams, block-range, skip-fraction, lam)
# 'out' rides with the image stream: norm_out modulates the final image tokens.
SETTINGS = [
    ("mod__all__allsteps",  ("img", "txt", "out"), None,     0.00, 1.0),
    ("mod__all__skip25",    ("img", "txt", "out"), None,     0.25, 1.0),
    ("mod__all__skip50",    ("img", "txt", "out"), None,     0.50, 1.0),
    ("mod__all__half",      ("img", "txt", "out"), None,     0.00, 0.5),
    ("mod__early__allsteps", ("img", "txt"),       (0, 8),   0.00, 1.0),
    ("mod__mid__allsteps",  ("img", "txt"),        (8, 16),  0.00, 1.0),
    ("mod__late__allsteps", ("img", "txt"),        (16, 24), 0.00, 1.0),
    ("modimg__all__allsteps", ("img", "out"),      None,     0.00, 1.0),
    ("modtxt__all__allsteps", ("txt",),            None,     0.00, 1.0),
]


class ModSwapper:
    """Blends the style row's modulation outputs into the content row's.

    lam = 1.0 is a full transplant, lam = 0.5 a midpoint -- the dial exists
    because modulation is a global signal and the interesting question is
    whether PARTIAL strength restyles without the copy-collapse the K/V sweep
    showed at full strength.
    """

    def __init__(self, streams, step_lo, step_hi, lam):
        self.streams = set(streams)
        self.step_lo, self.step_hi, self.lam = step_lo, step_hi, lam
        self.step = 0
        self.fired = 0
        self.handles = []

    def _hook(self, stream):
        def fn(module, inputs, output):
            if stream not in self.streams:
                return output
            if not (self.step_lo <= self.step < self.step_hi):
                return output
            out = output.clone()
            out[1::2] = self.lam * out[0::2] + (1.0 - self.lam) * out[1::2]
            self.fired += 1
            return out
        return fn

    def attach(self, pipe, blocks):
        tf = pipe.transformer
        joint = getattr(tf, "transformer_blocks", [])
        single = getattr(tf, "single_transformer_blocks", [])
        chosen = range(len(joint)) if blocks is None else range(*blocks)

        n = {"img": 0, "txt": 0, "out": 0}
        for i in chosen:
            if i >= len(joint):
                continue
            blk = joint[i]
            for attr, stream in (("norm1", "img"), ("norm1_context", "txt")):
                norm = getattr(blk, attr, None)
                lin = getattr(norm, "linear", None) if norm is not None else None
                if lin is None:
                    continue
                self.handles.append(lin.register_forward_hook(self._hook(stream)))
                n[stream] += 1
        for blk in single:                      # FLUX: one stream, one norm
            lin = getattr(getattr(blk, "norm", None), "linear", None)
            if lin is None:
                break
            self.handles.append(lin.register_forward_hook(self._hook("img")))
            n["img"] += 1
        if blocks is None:
            lin = getattr(getattr(tf, "norm_out", None), "linear", None)
            if lin is not None:
                self.handles.append(lin.register_forward_hook(self._hook("out")))
                n["out"] += 1
        if sum(n.values()) == 0:
            raise SystemExit(
                "attached zero modulation hooks -- this diffusers build does "
                "not expose norm1.linear / norm1_context.linear where expected; "
                "inspect pipe.transformer before trusting anything")
        return n

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def generate(pipe, cfg, style_prompt, content_prompt, seed, setting,
             steps, size):
    """One batch, independent noise per branch -- the round-1 lesson holds."""
    gens = [torch.Generator("cpu").manual_seed(seed),
            torch.Generator("cpu").manual_seed(seed + 10_000)]
    if setting is None:
        swapper, hooks = None, {}
    else:
        name, streams, blocks, frac, lam = setting
        lo = int(round(frac * steps))
        swapper = ModSwapper(streams, lo, 10**9, lam)
        hooks = swapper.attach(pipe, blocks)

    def on_step_end(p, i, t, kw):
        if swapper is not None:
            swapper.step = i + 1
        return kw

    try:
        out = pipe(prompt=[style_prompt, content_prompt],
                   num_inference_steps=steps, guidance_scale=cfg["guidance"],
                   height=size, width=size, generator=gens,
                   callback_on_step_end=on_step_end)
    finally:
        if swapper is not None:
            swapper.detach()
    fired = swapper.fired if swapper is not None else 0
    return out.images, fired, hooks


def selftest(pipe, cfg):
    """Same contract as run.py's: only the content row may move."""
    sp, cp = PAIRS[0][1], PAIRS[0][2]
    full = ("selftest", ("img", "txt", "out"), None, 0.0, 1.0)
    base, _, _ = generate(pipe, cfg, sp, cp, 0, None, 6, 512)
    sw, fired, hooks = generate(pipe, cfg, sp, cp, 0, full, 6, 512)

    def dmax(x, y):
        return int(np.abs(np.asarray(x, dtype=np.int16)
                          - np.asarray(y, dtype=np.int16)).max())

    style_moved, content_moved = dmax(base[0], sw[0]), dmax(base[1], sw[1])
    print(f"\n  hooks: {hooks}   calls fired: {fired}")
    print(f"  style branch   max|diff| = {style_moved:3d}   (must be <= 1)")
    print(f"  content branch max|diff| = {content_moved:3d}   (must be >= 5)")
    ok = fired > 0 and style_moved <= 1 and content_moved >= 5
    if fired == 0:
        print("  FAIL: no hook fired.")
    if style_moved > 1:
        print("  FAIL: style branch changed -- indexing reversed or branches "
              "entangled.")
    if content_moved < 5:
        print("  FAIL: content branch barely moved. Either modulation truly "
              "does nothing here\n        (possible -- Modulation Guidance "
              "reports the pooled path is weak unamplified)\n        or the "
              "hooked linears are not the ones consumed. Check with lam far "
              "above 1\n        before concluding the former.")
    print("  SELFTEST PASSED" if ok else "  SELFTEST FAILED")
    return ok


def kv_line_anchor(images_root, model):
    """Re-derive the K/V line's copy-end from sweep metrics when available."""
    p = images_root / model / "sweep" / "metrics.jsonl"
    if not p.exists():
        return KV_DS_MAX_FALLBACK, "fallback constant (sweep metrics not found)"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    base = np.mean([r["S"] for r in rows if r["setting"] == "baseline"])
    ds = [base - r["S"] for r in rows if r["setting"] != "baseline"]
    return float(max(ds)), f"re-derived from {p}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="sd35")
    ap.add_argument("--path")
    ap.add_argument("--out", type=Path, default=HERE / "images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", type=int)
    ap.add_argument("--offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--sequential-offload", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = dict(MODELS[args.model])
    if args.steps:
        cfg["steps"] = args.steps
    if args.size:
        cfg["size"] = args.size

    root = args.out / args.model / "modprobe"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["pair"], r["seed"], r["setting"]))

    pipe = load_pipeline(cfg, args.path or cfg["path"], args.offload,
                         args.sequential_offload)
    if args.selftest:
        raise SystemExit(0 if selftest(pipe, cfg) else 1)

    grid = SETTINGS[:args.limit] if args.limit else SETTINGS
    total = (len(grid) + 1) * len(PAIRS) * len(args.seeds)
    fh = manifest.open("a")
    t0, n = time.time(), 0

    def run_one(pid, sp, cp, seed, name, setting):
        nonlocal n
        n += 1
        outs = ([root / f"{pid}__seed{seed}__style.png",
                 root / f"{pid}__seed{seed}__content.png"] if setting is None
                else [root / f"{pid}__seed{seed}__{name}.png"])
        if (pid, seed, name) in done and all(p.exists() for p in outs):
            return
        imgs, fired, hooks = generate(pipe, cfg, sp, cp, seed, setting,
                                      cfg["steps"], cfg["size"])
        if setting is None:
            imgs[0].save(outs[0])
            imgs[1].save(outs[1])
        else:
            imgs[1].save(outs[0])
        fh.write(json.dumps({"pair": pid, "seed": seed, "setting": name,
                             "hooks": hooks, "fired": fired}) + "\n")
        fh.flush()
        rate = (time.time() - t0) / n
        print(f"  {n}/{total} {pid} seed{seed} {name:<22} {rate:.0f}s/run "
              f"eta {(total - n) * rate / 60:.0f}min", flush=True)

    for seed in args.seeds:
        for pid, sp, cp in PAIRS:
            run_one(pid, sp, cp, seed, "none", None)
    for setting in grid:
        for seed in args.seeds:
            for pid, sp, cp in PAIRS:
                run_one(pid, sp, cp, seed, setting[0], setting)
    fh.close()

    # ------------------------------------------------------------- report
    rows = []
    for seed in args.seeds:
        for pid, _, _ in PAIRS:
            ref = {k: root / f"{pid}__seed{seed}__{k}.png"
                   for k in ("style", "content")}
            if not all(p.exists() for p in ref.values()):
                continue
            d_sty = style_descriptor(Image.open(ref["style"]))
            s_sty = structure(Image.open(ref["style"]))
            s_con = structure(Image.open(ref["content"]))
            rows.append({"pair": pid, "seed": seed, "setting": "baseline",
                         "S": style_distance(
                             style_descriptor(Image.open(ref["content"])),
                             d_sty),
                         "C": 1.0, "L": struct_sim(s_con, s_sty)})
            for name, *_ in grid:
                p = root / f"{pid}__seed{seed}__{name}.png"
                if not p.exists():
                    continue
                im = Image.open(p)
                rows.append({"pair": pid, "seed": seed, "setting": name,
                             "S": style_distance(style_descriptor(im), d_sty),
                             "C": struct_sim(structure(im), s_con),
                             "L": struct_sim(structure(im), s_sty)})
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    def agg(setting, key):
        v = [r[key] for r in rows if r["setting"] == setting]
        return float(np.mean(v)) if v else float("nan")

    base_S, base_L = agg("baseline", "S"), agg("baseline", "L")
    ds_max, anchor_src = kv_line_anchor(args.out, args.model)
    print(f"\nbaseline: S {base_S:.3f}  L {base_L:.3f}")
    print(f"K/V tradeoff line anchor dS_max = {ds_max:.3f}  ({anchor_src})\n")
    print(f"{'setting':<24}{'S':>8}{'dS':>9}{'C':>8}{'L':>8}{'dL':>9}"
          f"{'resid':>9}{'  off?':>8}")
    print("-" * 84)
    cells = []
    for name, *_ in grid:
        S, C, L = (agg(name, k) for k in "SCL")
        dS, dL = base_S - S, L - base_L
        res = dS - ds_max * (1.0 - C)
        ok = (res > OFF_LINE["residual"] and C > OFF_LINE["C"]
              and dL < OFF_LINE["dL"])
        cells.append((name, dS, C, dL, res, ok))
        print(f"{name:<24}{S:>8.3f}{dS:>+9.3f}{C:>8.3f}{L:>8.3f}{dL:>+9.3f}"
              f"{res:>+9.3f}{'  YES' if ok else '  -':>8}")

    off = sorted((c for c in cells if c[5]), key=lambda c: -c[4])
    print("\nverdict:")
    if off:
        print(f"  {len(off)} cell(s) OFF the K/V tradeoff line:")
        for name, dS, C, dL, res, _ in off:
            print(f"    {name:<24} resid {res:+.3f}  dS {dS:+.3f}  C {C:.3f}")
        print("  The modulation CHANNEL carries style at a cost K/V cannot "
              "reach. This does NOT yet\n  prove the image-signal claim -- "
              "here the transplant equals feeding the style prompt's\n  "
              "pooled embedding, which text methods can do. PROCEED to step 3 "
              "(real style images);\n  the text-equivalence kill criterion is "
              "tested there, not here.")
    else:
        strongest = max(cells, key=lambda c: c[1])
        print(f"  NO cell off the line (strongest: {strongest[0]}, "
              f"dS {strongest[1]:+.3f} at C {strongest[2]:.3f}).")
        print("  The modulation path does not restyle beyond the "
              "style-for-content exchange rate\n  K/V already achieves. Lever "
              "dead; direction dies; back to StyleSSP failure axes\n  3 and 4 "
              "in plan_beyond_stylessp.md.")

    show = [c[0] for c in off[:6]] or [c[0] for c in
                                       sorted(cells, key=lambda c: -c[4])[:6]]
    for seed in args.seeds:
        cols = ["content", "style"] + show
        tile, pad, head = 300, 26, 30
        canvas = Image.new("RGB", (tile * len(cols),
                                   head + len(PAIRS) * (tile + pad)), "white")
        d = ImageDraw.Draw(canvas)
        for j, c in enumerate(cols):
            d.text((j * tile + 6, 8), c, fill="black")
        for i, (pid, _, _) in enumerate(PAIRS):
            y = head + i * (tile + pad)
            for j, c in enumerate(cols):
                p = root / f"{pid}__seed{seed}__{c}.png"
                if p.exists():
                    canvas.paste(Image.open(p).convert("RGB")
                                 .resize((tile, tile)), (j * tile, y))
            d.text((6, y + tile + 6),
                   f"{pid}   (leak probe: does a {LEAK_OBJECT[pid]} appear?)",
                   fill="black")
        outp = root / f"modprobe_seed{seed}.png"
        canvas.save(outp)
        print(f"  sheet -> {outp}")
    print("\nLook at the sheet before believing the table.")


if __name__ == "__main__":
    main()
