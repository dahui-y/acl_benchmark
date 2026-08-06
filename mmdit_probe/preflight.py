"""Answer both questions before spending GPU time: is the env enough, and will
it actually use the weights already on disk?

Loads nothing heavy -- filesystem checks and signature inspection only, a few
seconds. Run it before run.py. The two failures it exists to catch are the ones
that waste the most time:

  * a partially downloaded checkpoint. SD3.5's text_encoder_3 is T5-XXL at ~9GB
    and is very often the component that is missing; from_pretrained will then
    try to fetch it, and on a server with no route to the hub that is a long
    hang followed by an error, not a fast failure.

  * from_pretrained reaching out to the hub even for a fully local path, just
    to revalidate. Same hang. HF_HUB_OFFLINE=1 turns it into an immediate,
    readable error instead.

    python preflight.py
"""

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

from run import MODELS, resolve_path

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def check_env():
    print("environment")
    fine = True
    for mod, need in (("torch", None), ("diffusers", "0.31"),
                      ("transformers", "4.40"), ("accelerate", None),
                      ("numpy", None), ("PIL", None)):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            tag = OK
            if need and tuple(int(x) for x in v.split(".")[:2]) < \
                    tuple(int(x) for x in need.split(".")):
                tag, fine = BAD, False
                v += f"  (need >= {need})"
            print(f"[{tag}] {mod:<14}{v}")
        except ImportError:
            print(f"[{BAD}] {mod:<14}not installed")
            fine = False
    # SD3.5 and FLUX both tokenise with T5, which needs these two. They are a
    # silent import inside transformers, so a missing one surfaces as a
    # confusing tokenizer error much later.
    for mod in ("sentencepiece", "protobuf", "safetensors"):
        try:
            importlib.import_module(mod if mod != "protobuf" else "google.protobuf")
            print(f"[{OK}] {mod:<14}present")
        except ImportError:
            print(f"[{BAD}] {mod:<14}missing -- pip install {mod}")
            fine = False
    return fine


def check_api():
    print("\ndiffusers API")
    import diffusers

    fine = True
    for name in ("StableDiffusion3Pipeline", "FluxPipeline"):
        cls = getattr(diffusers, name, None)
        if cls is None:
            print(f"[{BAD}] {name} absent -- upgrade diffusers")
            fine = False
            continue
        params = inspect.signature(cls.__call__).parameters
        miss = [p for p in ("callback_on_step_end", "generator", "guidance_scale")
                if p not in params]
        if miss:
            print(f"[{BAD}] {name}: __call__ lacks {miss}")
            fine = False
        else:
            print(f"[{OK}] {name}: callback_on_step_end supported")
    return fine


def check_weights():
    print("\nweights")
    fine = True
    for key, cfg in MODELS.items():
        print(f"  {key}  {cfg['path']}")
        try:
            root = Path(resolve_path(cfg["path"]))
        except SystemExit as e:
            print(f"[{BAD}] {e}")
            fine = False
            continue
        print(f"[{OK}] resolved -> {root}")
        idx = json.loads((root / "model_index.json").read_text())
        for comp, spec in sorted(idx.items()):
            if comp.startswith("_") or not isinstance(spec, list):
                continue
            if spec[0] is None:                      # legitimately absent
                continue
            d = root / comp
            if not d.is_dir():
                print(f"[{BAD}]   {comp:<18}directory missing")
                fine = False
                continue
            # Follow symlinks: an HF cache snapshot is links into ../../blobs,
            # and a link whose blob never finished downloading is dangling.
            files = [f for f in d.rglob("*") if f.is_file()]
            dangling = [f for f in d.rglob("*")
                        if f.is_symlink() and not f.exists()]
            size = sum(f.stat().st_size for f in files) / 2**30
            has_w = any(f.suffix in (".safetensors", ".bin", ".pth")
                        for f in files)
            if dangling:
                print(f"[{BAD}]   {comp:<18}{len(dangling)} dangling link(s)"
                      f" -- download incomplete")
                fine = False
            elif not has_w and comp not in ("scheduler", "tokenizer",
                                            "tokenizer_2", "tokenizer_3"):
                print(f"[{BAD}]   {comp:<18}no weight file")
                fine = False
            else:
                print(f"[{OK}]   {comp:<18}{size:6.2f} GB")
    return fine


def check_gpu():
    print("\ngpu")
    import torch

    if not torch.cuda.is_available():
        print(f"[{BAD}] no CUDA device")
        return False
    n = torch.cuda.get_device_properties(0)
    gb = n.total_mem / 2**30 if hasattr(n, "total_mem") else \
        n.total_memory / 2**30
    print(f"[{OK}] {n.name}  {gb:.1f} GB")
    if gb < 22:
        print(f"[{WARN}] under 22 GB: run with --size 768 or "
              f"--sequential-offload")
    return True


def main():
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        print(f"[{WARN}] HF_HUB_OFFLINE is not set. from_pretrained will try "
              f"to reach huggingface.co\n        even for a fully local path. "
              f"On this server that is a hang, not an error.\n"
              f"        export HF_HUB_OFFLINE=1\n")
    # Each check runs even if an earlier one failed: one report listing every
    # problem beats four rounds of fix-one-thing-and-rerun.
    results = []
    for fn in (check_env, check_api, check_weights, check_gpu):
        try:
            results.append(fn())
        except Exception as exc:
            print(f"[{BAD}] {fn.__name__} could not run: "
                  f"{type(exc).__name__}: {exc}")
            results.append(False)
    print()
    if all(results):
        print("ALL CLEAR -> python run.py --selftest --model sd35")
    else:
        print("NOT READY -- fix the FAIL lines above first.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
