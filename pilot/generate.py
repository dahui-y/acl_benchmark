"""Generate the pilot images.

One rule carries over from the video work and it is the reason the design has
any power: **every condition of an item uses the same seed**. Same seed, same
initial noise, so the only thing that differs between `dist` and `coll` is the
text. On the video side this was verified bit-identical (max pixel delta
0.000000) and the same argument applies here -- without it, a difference between
conditions could just be two different draws.

Layout mirrors the video pipeline so the analysis code reads the same:

    images/<model>/item0007/dist__seed11.png

Resume is keyed on the PROMPT, not on the path. Rebuilding the suite renumbers
items -- adding one object reshuffles every id after it -- and the files on disk
keep the old numbering. Matching on coordinates alone silently accepts a stale
image as done, which on the video side is how a pilot ended up scoring "rolling
a crust" against a video of someone grating a carrot.
"""

import argparse
import json
import time
from pathlib import Path

MODELS = {
    # 3.5 GB in fp16, ~6 s/image at 1024 on a 4090. The right pilot model:
    # fastest of the credible ones, and an effect that shows here is real.
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd35m": "stabilityai/stable-diffusion-3.5-medium",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}


def plan(items, seeds=None):
    jobs = []
    for item in items:
        for cond in item["conditions"]:
            for seed in (seeds or item["seeds"]):
                jobs.append({"item_id": item["item_id"],
                             "condition": cond["condition"],
                             "seed": seed, "prompt": cond["prompt"]})
    return jobs


def done_prompts(manifest_path):
    """(item, condition, seed) -> prompt already generated.

    Storing the prompt is what makes resume safe across a rebuilt suite: a job
    counts as done only when the file exists AND the prompt that produced it
    matches the prompt we are about to send.
    """
    done = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["item_id"], r["condition"], r["seed"])] = r["prompt"]
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sdxl", choices=list(MODELS))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "images")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="omit to use the pipeline's own default, which is what "
                         "the model ships with and what the paper should report")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--offload", action="store_true",
                    help="model CPU offload; not needed for SDXL on 24 GB, "
                         "needed for FLUX")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in args.stimuli.read_text().splitlines() if l]
    jobs = plan(items, args.seeds)
    model_id = args.model_id or MODELS[args.model]
    root = args.out / args.model
    manifest = root / "manifest.jsonl"

    already = done_prompts(manifest)
    todo = []
    stale = 0
    for j in jobs:
        key = (j["item_id"], j["condition"], j["seed"])
        path = root / f"item{j['item_id']:04d}" / f"{j['condition']}__seed{j['seed']}.png"
        if key in already and path.exists():
            if already[key] == j["prompt"]:
                continue
            stale += 1
        todo.append(j)

    n_done = len(jobs) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs)} images planned, {n_done} already done, {len(todo)} to go"
          f"  [{model_id}]")
    if stale:
        print(f"  WARNING: {stale} existing images were generated from a "
              f"DIFFERENT prompt at the same coordinates -- the suite was "
              f"rebuilt and item ids shifted. They will be regenerated.")
    if args.dry_run or not todo:
        for j in todo[:8]:
            print(f"  item{j['item_id']:04d} {j['condition']:<11} "
                  f"seed{j['seed']}  {j['prompt']}")
        return

    import torch  # noqa: PLC0415
    from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    kwargs = {"height": args.size, "width": args.size}
    if args.steps:
        kwargs["num_inference_steps"] = args.steps

    t0 = time.time()
    root.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as fh:
        for n, j in enumerate(todo, 1):
            folder = root / f"item{j['item_id']:04d}"
            folder.mkdir(exist_ok=True)
            path = folder / f"{j['condition']}__seed{j['seed']}.png"
            # A fresh generator per image, seeded identically across the
            # conditions of an item. Reusing one generator would advance its
            # state between conditions and destroy the shared-noise property
            # the whole comparison rests on.
            gen = torch.Generator(device="cuda").manual_seed(j["seed"])
            image = pipe(prompt=j["prompt"], generator=gen, **kwargs).images[0]
            image.save(path)
            fh.write(json.dumps({**j, "path": str(path.relative_to(root)),
                                 "model": model_id}) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  {n}/{len(todo)}  {rate:.1f}s/img  "
                      f"eta {(len(todo) - n) * rate / 60:.0f}min  "
                      f"peak {mem:.1f}GB", flush=True)
    print(f"done. {root}")


if __name__ == "__main__":
    main()
