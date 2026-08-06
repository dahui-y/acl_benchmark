"""Find which of StyleSSP's eight checkpoints already exist on this machine.

Run this BEFORE downloading anything. The full set is ~26 GB, the server has no
route to huggingface.co, and the README's own instruction
(`huggingface-cli download h94/IP-Adapter`) pulls a 40 GB+ repository when the
one file actually needed is 700 MB. So the first question is not "how do we
download" but "what is already here".

Searches the OpenBayes weights tree and any HF cache it can find, matching on
the repo's canonical directory name as well as the local-dir name StyleSSP's
README uses. For anything missing it prints the exact, filtered command to run
ON THE MAC (which can reach HF), plus the ModelScope alternative where one
exists.

    python check_assets.py
    python check_assets.py --roots /openbayes/input /openbayes/home ~/.cache
"""

import argparse
import os
from pathlib import Path

# Assets StyleSSP loads with variant="fp16". A cache holding only the fp32
# safetensors passes every size check and then fails at from_pretrained, twenty
# minutes into a run, with an error that does not say "wrong variant". So the
# presence of *.fp16.safetensors is checked separately from the size.
NEEDS_FP16 = {"stabilityai/stable-diffusion-xl-base-1.0", "TheMistoAI/MistoLine"}

# name, HF repo, what StyleSSP wants it at, approx GB, include-filter, notes
ASSETS = [
    ("SDXL base", "stabilityai/stable-diffusion-xl-base-1.0",
     "config.base_model_path", 6.9, None,
     "loaded three times; fp16 variant required (variant='fp16')"),
    ("BLIP2 flan-t5-xl", "Salesforce/blip2-flan-t5-xl",
     "hardcoded MODEL_ID in infer_style.py", 7.9, None,
     "captions only -- run_batch.py frees it before the pipelines load"),
    ("CLIP ViT-H-14", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
     "hardcoded image_encoder_path", 3.9, None,
     "IP-Adapter's image encoder"),
    ("SDXL VAE fp16 fix", "madebyollin/sdxl-vae-fp16-fix",
     "hardcoded in infer_style.py", 0.2, None, ""),
    ("IP-Adapter", "h94/IP-Adapter",
     "checkpoints/IP-Adapter", 0.7,
     "sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
     "WITHOUT the include filter this repo is 40 GB+"),
    ("IP-Adapter-Instruct", "CiaraRowles/IP-Adapter-Instruct",
     "checkpoints/models/ip-adapter-instruct-sdxl.bin", 1.2,
     "ip-adapter-instruct-sdxl.bin", "single file"),
    ("ControlNet tile SDXL", "xinsir/controlnet-tile-sdxl-1.0",
     "checkpoints/controlnet-tile-sdxl-1.0", 2.5, None, ""),
    ("MistoLine (canny)", "TheMistoAI/MistoLine",
     "checkpoints/MistoLine", 2.5, None,
     "loaded with variant='fp16'; only needed for control_type=tile_canny"),
]

MODELSCOPE = {
    "stabilityai/stable-diffusion-xl-base-1.0": "AI-ModelScope/stable-diffusion-xl-base-1.0",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "h94/IP-Adapter": "AI-ModelScope/IP-Adapter",
    "Salesforce/blip2-flan-t5-xl": "AI-ModelScope/blip2-flan-t5-xl",
}

DEFAULT_ROOTS = [
    "/openbayes/input", "/openbayes/home", "/root/.cache/huggingface",
    "~/.cache/huggingface", "./checkpoints", "../checkpoints",
]


def dir_size_gb(p, cap=200_000):
    total = n = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                if fp.is_file():           # follows symlinks; a dangling link
                    total += fp.stat().st_size   # raises and is counted below
            except OSError:
                n += 1
            if total > cap * 2 ** 30:
                return total / 2 ** 30, n
    return total / 2 ** 30, n


def search(roots, repo):
    """Match the HF cache name, the plain repo name, and the README's dir name."""
    owner, name = repo.split("/")
    needles = {f"models--{owner}--{name}", name}
    hits = []
    for r in roots:
        r = Path(os.path.expanduser(r))
        if not r.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(r):
            # do not descend into matched trees or into git internals
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for d in list(dirnames):
                if d in needles:
                    hits.append(Path(dirpath) / d)
                    dirnames.remove(d)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--min-gb", type=float, default=0.6,
                    help="a hit smaller than this fraction of the expected "
                         "size is treated as an incomplete download")
    args = ap.parse_args()

    roots = [r for r in args.roots if Path(os.path.expanduser(r)).is_dir()]
    print(f"searching: {', '.join(roots) or '(none of the default roots exist)'}\n")

    missing = []
    total_missing = 0.0
    for label, repo, wants, gb, incl, note in ASSETS:
        hits = search(roots, repo)
        status, detail = "MISSING", ""
        if hits:
            sizes = [(h, *dir_size_gb(h)) for h in hits]
            best, sz, bad = max(sizes, key=lambda x: x[1])
            if sz < gb * args.min_gb:
                status = "PARTIAL"
                detail = f"{best}  ({sz:.2f} GB, expected ~{gb} GB)"
            elif repo in NEEDS_FP16 and not any(best.rglob("*.fp16.safetensors")):
                status = "NO-FP16"
                detail = (f"{best}  ({sz:.2f} GB, but no *.fp16.safetensors)\n"
                          f"          StyleSSP loads this with variant='fp16'. "
                          f"Either re-download the fp16\n"
                          f"          variant, or drop variant='fp16' at the "
                          f"call site and accept fp32 VRAM.")
            else:
                status = "  ok  "
                detail = f"{best}  ({sz:.2f} GB)"
            if bad:
                detail += f"  [{bad} unreadable file(s) -- dangling symlinks?]"
        if status != "  ok  ":
            missing.append((label, repo, incl, gb))
            total_missing += gb
        print(f"[{status}] {label:<22}{detail}")
        if note and status != "  ok  ":
            print(f"          note: {note}")

    if not missing:
        print("\nALL PRESENT. Next: point run_batch.py at these paths.")
        return

    print(f"\n{len(missing)} missing, ~{total_missing:.1f} GB total.")
    print("\n--- run these ON THE MAC (it can reach huggingface.co), then "
          "transfer ---")
    for label, repo, incl, gb in missing:
        inc = f' --include "{incl}"' if incl else ""
        print(f"  # {label}  (~{gb} GB)")
        print(f"  huggingface-cli download {repo}{inc} "
              f"--local-dir ckpt/{repo.split('/')[-1]}")
    ms = [(l, MODELSCOPE[r]) for l, r, _, _ in missing if r in MODELSCOPE]
    if ms:
        print("\n--- these have ModelScope mirrors reachable from the server "
              "(no transfer needed) ---")
        print("  pip install modelscope")
        for label, repo in ms:
            print(f"  modelscope download --model {repo} "
                  f"--local_dir ckpt/{repo.split('/')[-1]}   # {label}")
    print("\nDo NOT run the README's unfiltered `huggingface-cli download "
          "h94/IP-Adapter`: that repo\nis 40 GB+ and the single needed file is "
          "700 MB.")


if __name__ == "__main__":
    main()
