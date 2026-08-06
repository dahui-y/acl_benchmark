"""One-vote veto: does style live in MMDiT's image stream or its text stream?

The whole accepted+open training-free style-transfer line (StyleID CVPR 2024,
Cross-Image Attention SIGGRAPH 2024, Ctrl-X NeurIPS 2024, PnP CVPR 2023) works
by substituting keys and values into the U-Net's *self-attention*. MMDiT has no
self-attention layer to substitute into: text and image tokens go through one
joint attention, and -- unlike a U-Net, where the text embedding is frozen
conditioning -- the text representation is rewritten at every block.

So there are two candidate homes for style, and they mean opposite things:

  swap the image-stream K/V   the direct MMDiT analogue of what StyleID does.
                              If style transfers here and nowhere else, a
                              method built on it is a PORT, not a paper.

  swap the text-stream K/V    injecting the style branch's *evolved* text
                              representation. U-Net has no counterpart at all.
                              If style transfers here, there is a mechanism to
                              write about.

This script decides between them and nothing else. It is a veto, not a method.

HOW THE SWAP IS DONE, AND WHY THIS WAY

Not by rewriting the attention processor. Reimplementing FLUX's RoPE, its fused
projections and its single-block layout is exactly the kind of change that
produces a plausible-looking image from wrong maths, and a wrong image here is
worse than no experiment. Instead the two branches are generated in ONE batch,
and a forward hook on the K/V *projection modules* copies the style row over the
content row. The attention maths is untouched.

Each branch gets its own initial noise. Round 1 shared it, which turned out to
be degenerate -- see generate() -- and produced a pixel copy of the style image
under every condition. Use sweep.py, not this script's defaults, to find the
setting at which an injection restyles instead of overwriting.

  double / joint blocks   attn.to_k, attn.to_v          -> image stream
                          attn.add_k_proj, add_v_proj   -> text stream
  FLUX single blocks      attn.to_k, attn.to_v over the concatenated
                          [text; image] sequence -> split by token index

Batch layout is [style, content] per chunk, which holds with and without CFG:
diffusers builds [neg_style, neg_content, pos_style, pos_content], so `out[1::2]
= out[0::2]` is correct in both cases. `--selftest` proves that indexing is the
right way round rather than assuming it.

    python run.py --selftest --model sd35            # 2 min, run this first
    python run.py --model sd35                       # ~15 min
    python run.py --model flux --offload             # ~1.5 h
"""

import argparse
import json
import time
from pathlib import Path

import torch

from prompts import CONDITIONS, PAIRS

HERE = Path(__file__).parent

MODELS = {
    "sd35": {
        "path": "/openbayes/input/input0/Sim2Struct-1000/temp/weights/hf/hub/"
                "models--stabilityai--stable-diffusion-3.5-medium",
        "pipeline": "StableDiffusion3Pipeline",
        "steps": 28, "guidance": 4.5, "size": 1024, "txt_len": 333,
    },
    "flux": {
        "path": "/openbayes/input/input0/Sim2Struct-1000/temp/weights/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "steps": 28, "guidance": 3.5, "size": 1024, "txt_len": 512,
    },
}


def resolve_path(p):
    """Accept either a plain model dir or an HF hub cache dir."""
    p = Path(p)
    if (p / "model_index.json").exists():
        return str(p)
    snaps = sorted((p / "snapshots").glob("*")) if (p / "snapshots").is_dir() else []
    for s in snaps:
        if (s / "model_index.json").exists():
            return str(s)
    raise SystemExit(
        f"no model_index.json under {p}\n"
        f"  looked directly and in snapshots/ ({len(snaps)} found)\n"
        f"  pass the right directory with --path")


# --------------------------------------------------------------------------
# the swap


class Swapper:
    """Copies the style branch's K/V onto the content branch's, under a mode.

    Registered per block, removed after each generation. `step` is advanced by
    the pipeline callback so a run can be restricted to part of the schedule;
    the default is the whole schedule, because a null result on part of it
    would not be a veto.
    """

    def __init__(self, mode, step_lo, step_hi):
        self.mode, self.step_lo, self.step_hi = mode, step_lo, step_hi
        self.step = 0
        self.fired = 0
        self.handles = []

    def _active(self, stream):
        if not (self.step_lo <= self.step < self.step_hi):
            return False
        return self.mode == "both" or self.mode == stream

    def _hook(self, stream, txt_len=None):
        def fn(module, inputs, output):
            if txt_len is None:                       # separate-stream module
                if not self._active(stream):
                    return output
                out = output.clone()
                out[1::2] = out[0::2]
                self.fired += 1
                return out
            # concatenated [text; image] sequence: split by token index
            if not (self._active("txt") or self._active("img")):
                return output
            if output.shape[1] <= txt_len:
                raise RuntimeError(
                    f"expected a concatenated sequence longer than txt_len="
                    f"{txt_len}, got {output.shape[1]}. Pass the right "
                    f"--txt-len for this model.")
            out = output.clone()
            if self._active("txt"):
                out[1::2, :txt_len] = out[0::2, :txt_len]
            if self._active("img"):
                out[1::2, txt_len:] = out[0::2, txt_len:]
            self.fired += 1
            return out
        return fn

    def attach(self, pipe, blocks, txt_len):
        joint = getattr(pipe.transformer, "transformer_blocks", [])
        single = getattr(pipe.transformer, "single_transformer_blocks", [])
        chosen = range(len(joint)) if blocks is None else blocks

        n_joint = n_single = 0
        for i in chosen:
            if i >= len(joint):
                continue
            attn = joint[i].attn
            for name, stream in (("to_k", "img"), ("to_v", "img"),
                                 ("add_k_proj", "txt"), ("add_v_proj", "txt")):
                mod = getattr(attn, name, None)
                if mod is None:          # e.g. a context_pre_only final block
                    continue
                self.handles.append(mod.register_forward_hook(self._hook(stream)))
                n_joint += 1
        # FLUX single blocks run one stream over [text; image]; skipping them
        # would leave two thirds of the model unprobed, so they are handled by
        # token index instead of by module identity.
        for blk in single:
            attn = blk.attn
            miss = [n for n in ("to_k", "to_v") if getattr(attn, n, None) is None]
            if miss:
                print(f"  note: single blocks lack {miss} in this diffusers "
                      f"build (fused qkv?); probing joint blocks only")
                break
            for name in ("to_k", "to_v"):
                self.handles.append(getattr(attn, name).register_forward_hook(
                    self._hook(None, txt_len=txt_len)))
                n_single += 1
        if n_joint == 0:
            raise SystemExit("attached zero hooks -- the block layout is not "
                             "what this script expects; inspect "
                             "pipe.transformer before trusting anything")
        return n_joint, n_single

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------


def load_pipeline(cfg, path, offload, sequential):
    import diffusers

    cls = getattr(diffusers, cfg["pipeline"])
    pipe = cls.from_pretrained(resolve_path(path), torch_dtype=torch.bfloat16)
    if sequential:
        pipe.enable_sequential_cpu_offload()
    elif offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, cfg, style_prompt, content_prompt, seed, mode,
             blocks, step_lo, step_hi, steps, size, share_noise=False):
    """Both branches in one batch, each with its own initial noise by default.

    Round 1 of this probe shared one latent between the branches, reasoning that
    identical noise makes every difference attributable. It does the opposite.
    With a shared latent the two image streams are the SAME tensor at block 0,
    so overwriting the K/V drives the content branch onto the style branch
    exactly: `both` came back as a pixel copy of the style image, which is the
    mathematically required outcome, not a measurement. There is no content to
    preserve, so C measures nothing.

    Independent noise gives the content branch a layout of its own, which is the
    thing a partial injection is supposed to keep. `share_noise` restores the
    degenerate case for anyone who wants to reproduce it on purpose.
    """
    gens = [torch.Generator("cpu").manual_seed(seed),
            torch.Generator("cpu").manual_seed(
                seed if share_noise else seed + 10_000)]
    swapper = Swapper(mode, step_lo, step_hi)
    n_j = n_s = 0
    if mode != "none":
        n_j, n_s = swapper.attach(pipe, blocks, cfg["txt_len"])

    def on_step_end(p, i, t, kw):
        swapper.step = i + 1
        return kw

    try:
        out = pipe(
            prompt=[style_prompt, content_prompt],
            num_inference_steps=steps,
            guidance_scale=cfg["guidance"],
            height=size, width=size,
            generator=gens,
            callback_on_step_end=on_step_end,
        )
    finally:
        swapper.detach()
    return out.images, swapper.fired, (n_j, n_s)


def selftest(pipe, cfg):
    """Prove the batch indexing runs style -> content and not the reverse.

    Only the content row is supposed to be overwritten, so the STYLE image from
    a swapped run must match the style image from the unswapped run while the
    content image must not. If the indexing were backwards the two numbers
    would trade places -- and every contact sheet afterwards would be a
    confident picture of nothing.

    The style branch is allowed a tolerance of 1/255 rather than being required
    to be bit-identical: cloning a tensor changes its memory layout, which can
    change which attention kernel gets picked, and a false failure here would
    stop a run that is actually correct. A reversed index is not a 1-level
    difference, so the test still catches it.
    """
    import numpy as np

    sp, cp = PAIRS[0][1], PAIRS[0][2]
    kw = dict(blocks=None, step_lo=0, step_hi=10**9, steps=6, size=512)
    base, _, _ = generate(pipe, cfg, sp, cp, 0, "none", **kw)
    sw, fired, (n_j, n_s) = generate(pipe, cfg, sp, cp, 0, "both", **kw)

    def dmax(x, y):
        return int(np.abs(np.asarray(x, dtype=np.int16)
                          - np.asarray(y, dtype=np.int16)).max())

    style_moved = dmax(base[0], sw[0])
    content_moved = dmax(base[1], sw[1])

    print(f"\n  hooks: {n_j} on joint blocks, {n_s} on single blocks, "
          f"{fired} calls fired")
    print(f"  style branch   max|diff| = {style_moved:3d}   (must be <= 1)")
    print(f"  content branch max|diff| = {content_moved:3d}   "
          f"(must be >= 5 and >= 10x the style branch)")

    ok = True
    if fired == 0:
        print("  FAIL: no hook ever fired -- the swap is not happening.")
        ok = False
    if style_moved > 1:
        print("  FAIL: the style branch changed. Either the batch indexing is "
              "reversed (out[0::2] =\n        out[1::2] instead of the other "
              "way round) or the two branches are not\n        independent. Do "
              "not run the main experiment until this is fixed.")
        ok = False
    if content_moved < max(5, 10 * style_moved):
        print("  FAIL: the content branch barely moved. The hooks fire but "
              "have little effect --\n        check that to_k/to_v/add_k_proj/"
              "add_v_proj are the modules the attention\n        processor "
              "actually consumes in this diffusers build.")
        ok = False
    print("  SELFTEST PASSED" if ok else "  SELFTEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="sd35")
    ap.add_argument("--path", help="override the weights path")
    ap.add_argument("--out", type=Path, default=HERE / "images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", type=int)
    ap.add_argument("--txt-len", type=int,
                    help="text token count; only used for FLUX single blocks")
    ap.add_argument("--blocks", help="joint-block range 'lo:hi', default all")
    ap.add_argument("--steps-range", help="denoising step range 'lo:hi', "
                                          "default all")
    ap.add_argument("--offload", action="store_true", default=True)
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--sequential-offload", action="store_true")
    ap.add_argument("--share-noise", action="store_true",
                    help="same initial latent for both branches. The "
                         "degenerate case -- see generate() -- kept only so it "
                         "can be reproduced deliberately.")
    ap.add_argument("--tag", default="",
                    help="subdirectory under images/<model>/ so runs at "
                         "different settings do not overwrite each other")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = dict(MODELS[args.model])
    if args.steps:
        cfg["steps"] = args.steps
    if args.size:
        cfg["size"] = args.size
    if args.txt_len:
        cfg["txt_len"] = args.txt_len
    path = args.path or cfg["path"]

    blocks = None
    if args.blocks:
        lo, hi = (int(x) for x in args.blocks.split(":"))
        blocks = range(lo, hi)
    step_lo, step_hi = 0, 10**9
    if args.steps_range:
        step_lo, step_hi = (int(x) for x in args.steps_range.split(":"))

    root = args.out / args.model / args.tag if args.tag else args.out / args.model
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    # Resume is keyed on (pair, seed, condition) AND on the file existing, the
    # same rule as everywhere else here: a manifest line without a file on disk
    # is a crashed run, not a finished one.
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["pair"], r["seed"], r["cond"]))

    todo = []
    for pid, sp, cp in PAIRS:
        for seed in args.seeds:
            for cond in CONDITIONS:
                names = ([f"{pid}__seed{seed}__style.png",
                          f"{pid}__seed{seed}__content.png"] if cond == "none"
                         else [f"{pid}__seed{seed}__{cond}.png"])
                if ((pid, seed, cond) in done
                        and all((root / n).exists() for n in names)):
                    continue
                todo.append((pid, sp, cp, seed, cond))
    n_done = len(PAIRS) * len(args.seeds) * len(CONDITIONS) - len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"{args.model}: {len(PAIRS) * len(args.seeds) * len(CONDITIONS)} runs, "
          f"{n_done} done, {len(todo)} to go   "
          f"[{cfg['steps']} steps @ {cfg['size']}px, cfg {cfg['guidance']}]")

    pipe = load_pipeline(cfg, path, args.offload, args.sequential_offload)

    if args.selftest:
        raise SystemExit(0 if selftest(pipe, cfg) else 1)
    if not todo:
        return

    t0 = time.time()
    with manifest.open("a") as fh:
        for n, (pid, sp, cp, seed, cond) in enumerate(todo, 1):
            imgs, fired, (n_j, n_s) = generate(
                pipe, cfg, sp, cp, seed, cond, blocks, step_lo, step_hi,
                cfg["steps"], cfg["size"], args.share_noise)
            if cond == "none":
                imgs[0].save(root / f"{pid}__seed{seed}__style.png")
                imgs[1].save(root / f"{pid}__seed{seed}__content.png")
            else:
                imgs[1].save(root / f"{pid}__seed{seed}__{cond}.png")
            fh.write(json.dumps({
                "pair": pid, "seed": seed, "cond": cond, "model": args.model,
                "style_prompt": sp, "content_prompt": cp,
                "hooks_joint": n_j, "hooks_single": n_s, "hook_calls": fired,
                "steps": cfg["steps"], "size": cfg["size"],
                "guidance": cfg["guidance"],
                "blocks": args.blocks or "all", "tag": args.tag,
                "share_noise": args.share_noise,
                "step_range": args.steps_range or "all"}) + "\n")
            fh.flush()
            rate = (time.time() - t0) / n
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {n}/{len(todo)}  {pid} seed{seed} {cond:<5} "
                  f"{rate:.0f}s/run  eta {(len(todo) - n) * rate / 60:.0f}min  "
                  f"peak {mem:.1f}GB", flush=True)
    print(f"done -> {root}\nnow: python analyze.py --model {args.model}")


if __name__ == "__main__":
    main()
