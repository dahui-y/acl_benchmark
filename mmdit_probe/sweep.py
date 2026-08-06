"""Round 2: find the injection strength at which style transfers instead of
overwriting, then compare the two streams there.

Round 1 asked the right question at the wrong operating point. It injected into
every block at every timestep, with both branches on one shared latent, and got
back a pixel copy of the style image under all three conditions -- C = 0.14,
L = 1.00. Both streams were saturated, so the comparison between them carried no
information. That is a dial pinned at maximum, not a finding about MMDiT.

Two changes:

  * each branch gets its own noise, so the content branch has a layout of its
    own for a partial injection to preserve (see run.generate);
  * the strength is swept instead of assumed. Every accepted method in this line
    injects into a SUBSET -- StyleID uses late decoder layers and skips the
    early timesteps, because the early steps are where layout is decided. Round
    1 never visited that regime.

The grid is blocks x timestep-range x stream. What we are looking for is a cell
where S drops (the look came across) while C stays up (the layout survived) and
L does not rise (the style image's own object stayed out). If such a cell exists
for the image stream only, the direction is a port and dies. If it exists for the
text stream, there is a mechanism U-Net has no counterpart for. If no cell
exists at all, K/V substitution is simply not how style moves in MMDiT.

    python sweep.py --model sd35 --limit 2   # timing check first
    python sweep.py --model sd35            # 24 settings, ~70 min, one load
    python sweep.py --model sd35 --seeds 0 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analyze import structure, struct_sim, style_descriptor, style_distance
from prompts import LEAK_OBJECT, PAIRS
from run import MODELS, generate, load_pipeline

HERE = Path(__file__).parent

# 24 joint blocks in SD3.5-medium. Thirds, plus "all" as the round-1 control so
# the saturated point stays in the same table as the partial ones.
BLOCK_SETS = [("early", (0, 8)), ("mid", (8, 16)), ("late", (16, 24)),
              ("all", None)]
# Fractions of the schedule. Skipping the first quarter is the StyleID move:
# layout is decided early, so an injection that starts later should restyle
# rather than replace.
STEP_SETS = [("allsteps", 0.0), ("skip25", 0.25), ("skip50", 0.50)]
STREAMS = ["img", "txt"]


def settings(n_blocks, n_steps):
    for bname, brange in BLOCK_SETS:
        blocks = None if brange is None else range(brange[0],
                                                   min(brange[1], n_blocks))
        for sname, frac in STEP_SETS:
            lo = int(round(frac * n_steps))
            for stream in STREAMS:
                yield f"{stream}__{bname}__{sname}", stream, blocks, lo, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sd35")
    ap.add_argument("--path")
    ap.add_argument("--out", type=Path, default=HERE / "images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", type=int)
    ap.add_argument("--offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--sequential-offload", action="store_true")
    ap.add_argument("--limit", type=int, help="first N settings only, for a "
                                              "timing check")
    args = ap.parse_args()

    cfg = dict(MODELS[args.model])
    if args.steps:
        cfg["steps"] = args.steps
    if args.size:
        cfg["size"] = args.size

    root = args.out / args.model / "sweep"
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
    n_blocks = len(pipe.transformer.transformer_blocks)
    grid = list(settings(n_blocks, cfg["steps"]))
    if args.limit:
        grid = grid[:args.limit]
    print(f"{n_blocks} joint blocks, {cfg['steps']} steps -> "
          f"{len(grid)} settings x {len(PAIRS)} pairs x {len(args.seeds)} seeds")

    fh = manifest.open("a")
    t0 = time.time()
    n = 0
    total = (len(grid) + 1) * len(PAIRS) * len(args.seeds)

    def run_one(pid, sp, cp, seed, name, mode, blocks, lo, hi):
        nonlocal n
        n += 1
        outs = ([root / f"{pid}__seed{seed}__style.png",
                 root / f"{pid}__seed{seed}__content.png"] if mode == "none"
                else [root / f"{pid}__seed{seed}__{name}.png"])
        if ((pid, seed, name) in done and all(p.exists() for p in outs)):
            return
        imgs, fired, _ = generate(pipe, cfg, sp, cp, seed, mode, blocks, lo, hi,
                                  cfg["steps"], cfg["size"], share_noise=False)
        if mode == "none":
            imgs[0].save(outs[0])
            imgs[1].save(outs[1])
        else:
            imgs[1].save(outs[0])
        fh.write(json.dumps({"pair": pid, "seed": seed, "setting": name,
                             "mode": mode, "step_lo": lo,
                             "blocks": None if blocks is None
                             else [blocks.start, blocks.stop],
                             "hook_calls": fired}) + "\n")
        fh.flush()
        rate = (time.time() - t0) / max(n, 1)
        print(f"  {n}/{total} {pid} seed{seed} {name:<24} {rate:.0f}s/run "
              f"eta {(total - n) * rate / 60:.0f}min", flush=True)

    for seed in args.seeds:
        for pid, sp, cp in PAIRS:
            run_one(pid, sp, cp, seed, "none", "none", None, 0, 10**9)
    for name, stream, blocks, lo, hi in grid:
        for seed in args.seeds:
            for pid, sp, cp in PAIRS:
                run_one(pid, sp, cp, seed, name, stream, blocks, lo, hi)
    fh.close()

    # ---------------------------------------------------------------- report
    rows = []
    for seed in args.seeds:
        for pid, _, _ in PAIRS:
            ref = {k: root / f"{pid}__seed{seed}__{k}.png"
                   for k in ("style", "content")}
            if not all(p.exists() for p in ref.values()):
                continue
            d_sty = style_descriptor(Image.open(ref["style"]))
            d_con = style_descriptor(Image.open(ref["content"]))
            s_sty = structure(Image.open(ref["style"]))
            s_con = structure(Image.open(ref["content"]))
            rows.append({"pair": pid, "seed": seed, "setting": "baseline",
                         "S": style_distance(d_con, d_sty),
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

    base_S = agg("baseline", "S")
    base_L = agg("baseline", "L")
    print(f"\nbaseline: S {base_S:.3f}  C 1.000  L {base_L:.3f}\n")
    print(f"{'setting':<26}{'S':>8}{'dS':>9}{'C':>8}{'L':>8}{'dL':>9}"
          f"{'  usable?':>10}")
    print("-" * 78)
    best = []
    for name, *_ in grid:
        S, C, L = (agg(name, k) for k in "SCL")
        dS, dL = base_S - S, L - base_L
        # A cell is usable only if all three hold at once. S alone rewards
        # copying the style image; C alone rewards doing nothing.
        ok = dS > 0.05 and C > 0.5 and dL < 0.3
        if ok:
            best.append((dS, C, dL, name))
        print(f"{name:<26}{S:>8.3f}{dS:>+9.3f}{C:>8.3f}{L:>8.3f}{dL:>+9.3f}"
              f"{'  YES' if ok else '  -':>10}")

    print("\nverdict:")
    if not best:
        print("  NO usable cell anywhere in the grid. K/V substitution does not"
              " restyle in MMDiT --\n  it either does nothing or overwrites. "
              "That kills the port reading too: the finding\n  becomes 'the "
              "U-Net recipe has no MMDiT form', which is a result but not the\n"
              "  method this was scouting for.")
    else:
        best.sort(reverse=True)
        img_best = [b for b in best if b[3].startswith("img")]
        txt_best = [b for b in best if b[3].startswith("txt")]
        print(f"  {len(img_best)} usable image-stream cell(s), "
              f"{len(txt_best)} usable text-stream cell(s)")
        for dS, C, dL, name in best[:6]:
            print(f"    {name:<26} dS {dS:+.3f}  C {C:.3f}  dL {dL:+.3f}")
        if txt_best and (not img_best or txt_best[0][0] >= img_best[0][0]):
            print("  TEXT stream works at least as well -> U-Net has no "
                  "counterpart. PROCEED, and\n  build the 4-way processor "
                  "version to separate txt->img from txt->txt.")
        else:
            print("  Only the IMAGE stream works -> that is StyleID's swap "
                  "with different plumbing.\n  PORT. Direction dies here, as "
                  "agreed.")

    # one sheet per seed, columns = baseline refs + the usable cells (or the
    # best six by dS if none passed, so a failure is still inspectable)
    show = [b[3] for b in best[:6]] or [n for n, *_ in grid][:6]
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
        out = root / f"sweep_seed{seed}.png"
        canvas.save(out)
        print(f"  sheet -> {out}")
    print("\nLook at the sheet before believing the table.")


if __name__ == "__main__":
    main()
