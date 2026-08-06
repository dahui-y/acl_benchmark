"""Generate the problem-check videos on HunyuanVideo-1.5.

A deliberately small runner: 24 stratified OSCBench prompts, one seed, one
model. It reuses the registry in `generate/models.py` so the settings that end
up in any later paper table come from the same single place as the aspect
pipeline's did, and it writes an 8-frame strip PNG next to every video so the
outcome can be judged by looking at two dozen images rather than by trusting a
number.

Resume is keyed on the prompt, not the path -- same lesson as everywhere else
in this repo: ids shift when a suite is rebuilt, files on disk keep the old
numbering, and coordinate-matching silently accepts a stale video as done.

    export HF_ENDPOINT=https://hf-mirror.com
    python generate.py --limit 1          # smoke test: VRAM + s/video first
    python generate.py                    # the other 23
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "generate"))
from models import generation_kwargs, resolve, resolved_defaults  # noqa: E402


def load_pipeline(cfg, offload, sequential):
    import torch  # noqa: PLC0415
    import diffusers  # noqa: PLC0415

    cls = getattr(diffusers, cfg["pipeline_class"], None)
    if cls is None:
        raise SystemExit(
            f"diffusers {diffusers.__version__} has no "
            f"{cfg['pipeline_class']}. The class needs diffusers >= 0.36; "
            f"upgrade with: pip install -U diffusers")
    dtype = getattr(torch, cfg["dtype"])
    kwargs = {"torch_dtype": dtype}
    # Wan ships a bespoke VAE that has to be loaded separately and kept in fp32:
    # at the pipeline dtype it silently produces black frames rather than
    # erroring, so this is not an optimisation, it is required for the Wan
    # entries to produce anything at all.
    if cfg.get("vae_class"):
        vae_cls = getattr(diffusers, cfg["vae_class"])
        kwargs["vae"] = vae_cls.from_pretrained(
            cfg["repo_id"], subfolder="vae",
            torch_dtype=getattr(torch, cfg.get("vae_dtype", "float32")))
    pipe = cls.from_pretrained(cfg["repo_id"], **kwargs)
    if sequential:
        pipe.enable_sequential_cpu_offload()
    elif offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def frame_strip(video_path, out_path, n=8):
    """First-to-last frame strip; the judgement is made by looking at these."""
    import imageio.v3 as iio  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    frames = iio.imread(video_path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = [Image.fromarray(frames[i]) for i in idx]
    w, h = tiles[0].size
    strip = Image.new("RGB", (w * n, h))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * w, 0))
    strip.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hunyuanvideo-1.5-480p")
    ap.add_argument("--prompts", type=Path,
                    default=Path(__file__).parent / "osc_prompts.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "videos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offload", action="store_true", default=True,
                    help="model CPU offload -- the Qwen2.5-VL text encoder is "
                         "most of the VRAM, so this defaults to on")
    ap.add_argument("--no-offload", dest="offload", action="store_false")
    ap.add_argument("--sequential-offload", action="store_true",
                    help="slower, smaller: use only if --offload still OOMs")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = resolve(args.model)
    rows = [json.loads(l) for l in args.prompts.read_text().splitlines() if l]
    root = args.out / args.model
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    done = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["id"], r["seed"])] = r["prompt"]

    todo = []
    for r in rows:
        key = (r["id"], args.seed)
        path = root / f"{r['id']}__seed{args.seed}.mp4"
        if key in done and path.exists() and done[key] == r["prompt"]:
            continue
        todo.append(r)
    n_done = len(rows) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} prompts, {n_done} done, {len(todo)} to go   "
          f"[{cfg['repo_id']} @ {cfg['width']}x{cfg['height']} "
          f"{cfg['num_frames']}f seed {args.seed}]")
    if not todo:
        return

    import torch  # noqa: PLC0415
    from diffusers.utils import export_to_video  # noqa: PLC0415

    pipe = load_pipeline(cfg, args.offload, args.sequential_offload)
    kwargs = generation_kwargs(cfg)
    print(f"pipeline defaults in effect: {resolved_defaults(pipe)}")

    t0 = time.time()
    with manifest.open("a") as fh:
        for n, r in enumerate(todo, 1):
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            video = pipe(prompt=r["prompt"], generator=gen, **kwargs).frames[0]
            path = root / f"{r['id']}__seed{args.seed}.mp4"
            export_to_video(video, str(path), fps=cfg["fps"])
            try:
                frame_strip(path, root / f"{r['id']}__seed{args.seed}_strip.png")
            except Exception as exc:  # the strip is a convenience, not the data
                print(f"  (strip failed for {r['id']}: {exc})")
            fh.write(json.dumps({**r, "seed": args.seed, "model": cfg["repo_id"],
                                 "settings": kwargs}) + "\n")
            fh.flush()
            rate = (time.time() - t0) / n
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {n}/{len(todo)}  {r['id']} ({r['band']}/{r['verb']})  "
                  f"{rate:.0f}s/video  eta {(len(todo) - n) * rate / 60:.0f}min  "
                  f"peak {mem:.1f}GB", flush=True)
    print(f"done. videos + strips in {root}")


if __name__ == "__main__":
    main()
