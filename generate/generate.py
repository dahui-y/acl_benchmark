"""Generate the video set for the aspect benchmark.

Design, in one paragraph. Each item is one event -- a fixed (verb, object,
subject, scene) -- realised as 18 prompts that differ only in how the language
encodes culmination. The scientific claim is about the *difference between two
conditions of the same item*, so everything other than the prompt has to be held
fixed inside an item, and the one thing that would otherwise dominate is the
initial noise: the seed largely determines who is in the frame, where the camera
sits, what the counter looks like. Two videos from different seeds differ for
two reasons at once and the aspect effect is unrecoverable. So:

    within an item, all 18 conditions share one seed.

Three seeds per item then buy what a single seed cannot: an estimate of how much
of a condition difference is stable rather than one draw from the model's
distribution, error bars, and a variance decomposition (item vs seed as crossed
random effects). It also matters for the telicity axis specifically, where the
prediction is that the videos *do not* differ -- shared noise turns that from a
soft prediction into a hard one, since identical noise plus near-identical
conditioning should give near-identical video.

Iteration order is item -> seed -> condition, so an interrupted run leaves whole
(item, seed) blocks complete. A block is the analysable unit; half a block is
worth nothing.

Usage:
    python generate.py --list                          # what resolves, what it costs
    python generate.py --model wan2.2-ti2v-5b --dry-run
    python generate.py --model wan2.2-ti2v-5b --out-dir /data/videos
    python generate.py --model wan2.2-ti2v-5b --out-dir /data/videos --shard 0/4
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "probe"))

from build_stimuli import CONTROL_CONDITIONS, TARGET_CONDITIONS  # noqa: E402
from models import MODELS, generation_kwargs, resolve, resolved_defaults  # noqa: E402

ALL_CONDITIONS = TARGET_CONDITIONS + CONTROL_CONDITIONS

# Shared across items on purpose. Using per-item seeds would make "item 7 at seed
# 42" incomparable to "item 8 at seed 42" and cost the crossed design its second
# factor for nothing.
DEFAULT_SEEDS = [42, 43, 44]

# The three controls that are not needed on the video side. `paraphrase_min`,
# `paraphrase` and `filler` exist to calibrate the text-encoder distances; on the
# video side only `other_verb` earns its cost, as a check that the model responds
# to the event at all. Generating the other three would add ~17% to the bill for
# no annotation question. Override with --conditions all.
VIDEO_CONDITIONS = TARGET_CONDITIONS + ["other_verb"]


def load_items(stimuli_path, all_items=False):
    items = [json.loads(line) for line in stimuli_path.read_text().splitlines() if line]
    if not all_items:
        items = [it for it in items if it["in_generation_subset"]]
    if not items:
        raise SystemExit(f"no items selected from {stimuli_path}")
    return items


def video_path(out_dir, model, item_id, condition, seed):
    return out_dir / model / f"item{item_id:04d}" / f"{condition}__seed{seed}.mp4"


def plan(items, conditions, seeds, out_dir, model):
    """Job list in item -> seed -> condition order."""
    jobs = []
    for item in items:
        for seed in seeds:
            for cond in conditions:
                if cond not in item["texts"]:
                    raise SystemExit(f"item {item['item_id']} has no condition {cond!r}")
                jobs.append({
                    "model": model,
                    "item_id": item["item_id"],
                    "condition": cond,
                    "seed": seed,
                    "prompt": item["texts"][cond],
                    "gerund": item["gerund"],
                    "noun": item["noun"],
                    "aspectual_class": item["aspectual_class"],
                    "path": str(video_path(out_dir, model, item["item_id"], cond, seed)),
                })
    return jobs


def done_keys(manifest_path):
    """Jobs already finished, by (item, condition, seed). A manifest row only
    counts if the file it points at still exists -- a run killed mid-write leaves
    a truncated mp4 behind."""
    done = set()
    if not manifest_path.exists():
        return done
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok" and Path(rec["path"]).exists():
            done.add((rec["item_id"], rec["condition"], rec["seed"]))
    return done


# --------------------------------------------------------------------------- #
# backend


class DiffusersBackend:
    def __init__(self, cfg, offload=False, device="cuda"):
        import torch  # noqa: PLC0415
        import diffusers  # noqa: PLC0415

        self.torch = torch
        self.cfg = cfg
        self.device = device

        pipe_cls = getattr(diffusers, cfg["pipeline_class"], None)
        if pipe_cls is None:
            candidates = [n for n in dir(diffusers)
                          if n.endswith("Pipeline") and cfg["name"].split("-")[0][:3].lower()
                          in n.lower()]
            raise SystemExit(
                f"diffusers {diffusers.__version__} has no {cfg['pipeline_class']}.\n"
                f"  similar names: {candidates or '(none)'}\n"
                f"  fix the pipeline_class in models.py, or upgrade diffusers."
            )

        dtype = getattr(torch, cfg["dtype"])
        kwargs = {"torch_dtype": dtype}
        if cfg.get("vae_class"):
            vae_cls = getattr(diffusers, cfg["vae_class"])
            kwargs["vae"] = vae_cls.from_pretrained(
                cfg["repo_id"], subfolder="vae",
                torch_dtype=getattr(torch, cfg.get("vae_dtype", "float32")),
            )
        self.pipe = pipe_cls.from_pretrained(cfg["repo_id"], **kwargs)

        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        self.versions = {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(),
        }
        # What the pipeline defaults to for everything models.py leaves unset.
        self.defaults = resolved_defaults(self.pipe)

    def generate(self, prompt, seed, out_path):
        from diffusers.utils import export_to_video  # noqa: PLC0415

        # A fresh generator per call, seeded identically for every condition of
        # an item. Same seed + same latent shape (resolution and frame count are
        # model constants) => bit-identical initial noise across conditions.
        gen = self.torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(prompt=prompt, generator=gen, **generation_kwargs(self.cfg))
        frames = result.frames[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".part.mp4")
        export_to_video(frames, str(tmp), fps=self.cfg["fps"])
        tmp.replace(out_path)  # atomic: a killed run never leaves a half file
        return frames


class DryRunBackend:
    versions = {"backend": "dry-run"}

    def __init__(self, cfg, **_):
        self.cfg = cfg

    def generate(self, prompt, seed, out_path):
        return None


# --------------------------------------------------------------------------- #


def write_settings(path, cfg, backend, seeds, conditions, n_items):
    """Everything the settings table in the paper needs, recorded from what
    actually ran rather than from what we meant to run."""
    path.write_text(json.dumps({
        "model": cfg["name"],
        "repo_id": cfg["repo_id"],
        "pipeline_class": cfg["pipeline_class"],
        "dtype": cfg["dtype"],
        "resolution": f"{cfg['width']}x{cfg['height']}",
        "num_frames": cfg["num_frames"],
        "fps": cfg["fps"],
        "duration_s": round(cfg["num_frames"] / cfg["fps"], 2),
        # Set by us where models.py pins a value, otherwise read back from the
        # pipeline signature so "we used the default" is a recorded number.
        "num_inference_steps": cfg.get("num_inference_steps"),
        "guidance_scale": cfg.get("guidance_scale"),
        "pipeline_defaults": getattr(backend, "defaults", {}),
        "negative_prompt": cfg["negative_prompt"] or "(none)",
        "extra": cfg.get("extra", {}),
        "seeds": seeds,
        "seed_policy": "shared across all conditions within an item",
        "n_items": n_items,
        "n_conditions": len(conditions),
        "conditions": conditions,
        "settings_source": cfg["source"],
        "environment": getattr(backend, "versions", {}),
    }, indent=2, ensure_ascii=False) + "\n")


def determinism_check(backend, job, tolerance=1e-3):
    """Regenerate one job and compare. The identification argument assumes that
    a seed fixes the video given the prompt; non-deterministic attention kernels
    can break that, and if they do, a within-item condition difference is partly
    noise. Cheap to check once, expensive to discover later."""
    import numpy as np  # noqa: PLC0415

    a = backend.generate(job["prompt"], job["seed"], Path(job["path"]))
    b = backend.generate(job["prompt"], job["seed"],
                         Path(job["path"]).with_name("_determinism_check.mp4"))
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    delta = float(np.abs(a - b).max())
    verdict = "deterministic" if delta <= tolerance else "NOT deterministic"
    print(f"  determinism check: max |Δpixel| = {delta:.6f}  -> {verdict}")
    if delta > tolerance:
        print("  a seed no longer pins the video; within-item contrasts carry "
              "sampler noise. Report this, or fix it (deterministic attention "
              "backend, fixed batch size) before the main run.")
    return delta


def cmd_list(args):
    items = load_items(args.stimuli, args.all_items)
    conditions = ALL_CONDITIONS if args.conditions == "all" else VIDEO_CONDITIONS
    n = len(items) * len(conditions) * len(args.seeds)
    print(f"stimuli: {len(items)} items x {len(conditions)} conditions "
          f"x {len(args.seeds)} seeds = {n} videos per model\n")
    try:
        import diffusers
        available = dir(diffusers)
        version = diffusers.__version__
    except ImportError:
        available, version = [], "(not installed)"
    print(f"diffusers {version}")
    for name, cfg in MODELS.items():
        ok = "ok " if cfg["pipeline_class"] in available else "MISSING"
        role = "main" if cfg.get("main_experiment", True) else "----"
        steps = cfg.get("num_inference_steps")
        print(f"  [{ok}] [{role}] {name:20s} {cfg['pipeline_class']:26s} "
              f"{cfg['width']}x{cfg['height']} {cfg['num_frames']}f@{cfg['fps']} "
              f"{f'{steps} steps' if steps else 'default steps'}")
        print(f"                {cfg['source']}")


def cmd_run(args):
    cfg = resolve(args.model)
    items = load_items(args.stimuli, args.all_items)
    conditions = ALL_CONDITIONS if args.conditions == "all" else VIDEO_CONDITIONS

    out_dir = args.out_dir
    jobs = plan(items, conditions, args.seeds, out_dir, args.model)

    if args.shard:
        idx, total = (int(x) for x in args.shard.split("/"))
        # Shard by item, not by job, so every shard still holds whole
        # (item, seed) blocks and can be analysed on its own.
        keep = {it["item_id"] for i, it in enumerate(items) if i % total == idx}
        jobs = [j for j in jobs if j["item_id"] in keep]
        print(f"shard {idx}/{total}: {len(keep)} items, {len(jobs)} videos")

    manifest = out_dir / args.model / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    already = done_keys(manifest)
    todo = [j for j in jobs
            if (j["item_id"], j["condition"], j["seed"]) not in already]
    print(f"{len(jobs)} videos planned, {len(already)} already done, {len(todo)} to go")
    if args.limit:
        todo = todo[:args.limit]
        print(f"  --limit {args.limit}: running {len(todo)}")

    # n_items is the full selection, not this shard's slice, so concurrent shards
    # all write the same settings.json and cannot race to a wrong value.
    if args.dry_run:
        write_settings(out_dir / args.model / "settings.dry-run.json", cfg,
                       DryRunBackend(cfg), args.seeds, conditions, len(items))
        for j in todo[:5]:
            print(f"  item{j['item_id']:04d} seed{j['seed']} {j['condition']:16s} "
                  f"{j['prompt']}")
        if len(todo) > 5:
            print(f"  ... {len(todo) - 5} more")
        # Deliberately no manifest write: a dry run must not leave rows that the
        # resume logic would read as finished work.
        return

    print(f"loading {cfg['repo_id']} ...")
    t0 = time.time()
    backend = DiffusersBackend(cfg, offload=args.offload, device=args.device)
    print(f"loaded in {time.time() - t0:.0f}s  {backend.versions}")

    write_settings(out_dir / args.model / "settings.json", cfg, backend,
                   args.seeds, conditions, len(items))

    if args.determinism_check and todo:
        determinism_check(backend, todo[0])
        todo = todo[1:]

    t_start = time.time()
    with manifest.open("a") as fh:
        for n, job in enumerate(todo, 1):
            path = Path(job["path"])
            t0 = time.time()
            try:
                backend.generate(job["prompt"], job["seed"], path)
                status, error = "ok", None
            except Exception as exc:  # keep going; one bad cell is not a dead run
                status, error = "error", f"{type(exc).__name__}: {exc}"
                print(f"  ERROR {job['condition']} item{job['item_id']}: {error}")
            rec = dict(job, status=status, error=error,
                       seconds=round(time.time() - t0, 1),
                       **generation_kwargs(cfg))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                elapsed = time.time() - t_start
                rate = elapsed / n
                print(f"  {n}/{len(todo)}  {rate:.1f}s/video  "
                      f"eta {(len(todo) - n) * rate / 3600:.1f}h")

    print(f"done. manifest: {manifest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help=f"one of: {', '.join(MODELS)}")
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent.parent / "probe" / "stimuli.jsonl")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(os.environ.get("VIDEO_OUT", "videos")))
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="shared across conditions within an item (default 42 43 44)")
    ap.add_argument("--conditions", choices=["video", "all"], default="video",
                    help="'video' drops the three text-only controls (default)")
    ap.add_argument("--all-items", action="store_true",
                    help="ignore in_generation_subset and render every item")
    ap.add_argument("--shard", help="i/n, split by item across GPUs")
    ap.add_argument("--limit", type=int, help="stop after N videos (smoke test)")
    ap.add_argument("--offload", action="store_true", help="model CPU offload")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--determinism-check", action="store_true",
                    help="generate the first job twice and compare pixels")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no model")
    ap.add_argument("--list", action="store_true",
                    help="show cost and which registry entries resolve")
    args = ap.parse_args()

    if args.list:
        cmd_list(args)
    elif args.model:
        cmd_run(args)
    else:
        ap.error("--model is required (or --list)")


if __name__ == "__main__":
    main()
