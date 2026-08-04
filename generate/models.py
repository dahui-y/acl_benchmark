"""Model registry for the generation run.

One entry per model. Every field here ends up verbatim in `settings.json` and in
the paper's settings table, so this file is the single place where "what we ran"
is defined -- nothing is set from the command line except which models, items,
conditions and seeds to cover.

Two rules that the whole design rests on:

1. **Settings are constant across conditions within a model.** Resolution, frame
   count, step count, guidance and the negative prompt are properties of the
   model, never of the condition. If any of them varied by condition, a
   difference between `prog` and `result` videos would no longer be attributable
   to the prompt.

2. **Values are the model's own defaults**, not tuned by us. Tuning per model
   would make the cross-model comparison ours rather than the models'.

`source` records where each number came from. `verified` marks whether the entry
has been run end-to-end on this machine -- run `python generate.py --list` to
check which entries resolve against the installed diffusers before booking GPU
time.
"""

MODELS = {
    # ---- open-source, full generation subset -------------------------------
    "wan2.2-ti2v-5b": {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_class": "WanPipeline",
        # Wan ships a bespoke VAE that diffusers loads separately and keeps in
        # fp32; loading it at the pipeline dtype produces black frames.
        "vae_class": "AutoencoderKLWan",
        "vae_dtype": "float32",
        "dtype": "bfloat16",
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "source": "Wan2.2 TI2V-5B card defaults (720p24, 5s)",
        "verified": False,
    },
    "wan2.2-t2v-a14b": {
        "repo_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "pipeline_class": "WanPipeline",
        "vae_class": "AutoencoderKLWan",
        "vae_dtype": "float32",
        "dtype": "bfloat16",
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        # A14B is a two-expert MoE: diffusers exposes a second guidance scale for
        # the low-noise expert. Left unset here so the pipeline's own default
        # applies; if you set it, it lands in settings.json like everything else.
        "extra": {},
        "source": "Wan2.2 T2V-A14B card defaults -- CHECK against your checkout",
        "verified": False,
    },
    "hunyuanvideo-1.5": {
        "repo_id": "tencent/HunyuanVideo-1.5",
        "pipeline_class": "HunyuanVideo15Pipeline",
        "dtype": "bfloat16",
        "height": 720,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "source": "HunyuanVideo-1.5 repo defaults -- CHECK, diffusers class name "
                  "and repo id both depend on your diffusers version",
        "verified": False,
    },
    # ---- proprietary reference, subset only --------------------------------
    # Not run through this script: these are API models with no seed control we
    # can share across conditions, so they get their own driver and a smaller
    # subset. Kept here only so the settings table has one row per model.
    # "kling-2.x": {...},
    # "veo-3": {...},
}

# Deliberately empty. Wan's shipped negative prompt penalises "静止不动的画面"
# (a still, motionless picture) among other things -- and several of our
# conditions (`perf`, `result`, `phase_finish`) describe a state rather than an
# ongoing action, so that negative prompt would push exactly the cells we care
# about back towards motion. Using it would build the confound into the data.
# Any model whose default we deviate from this way must say so in the paper.
DEFAULT_NEGATIVE_PROMPT = ""


def resolve(name):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    cfg = dict(MODELS[name])
    cfg.setdefault("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    cfg.setdefault("extra", {})
    cfg["name"] = name
    return cfg


def generation_kwargs(cfg):
    """The arguments actually passed to the pipeline call, in one place so the
    manifest can record precisely what produced each file."""
    kwargs = {
        "height": cfg["height"],
        "width": cfg["width"],
        "num_frames": cfg["num_frames"],
        "num_inference_steps": cfg["num_inference_steps"],
        "guidance_scale": cfg["guidance_scale"],
    }
    if cfg.get("negative_prompt"):
        kwargs["negative_prompt"] = cfg["negative_prompt"]
    kwargs.update(cfg.get("extra", {}))
    return kwargs
