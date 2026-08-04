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

2. **Values are the model's own defaults**, not tuned by us -- the same rule
   OSCBench states in its Appendix B ("we follow the official and default
   implementations"). Tuning per model would make the cross-model comparison
   ours rather than the models'.

Resolution, frame count and FPS are pinned to OSCBench's Table 6 so our numbers
sit next to theirs without a settings caveat. Step count and guidance scale are
deliberately absent: OSCBench does not report them, which means they ran the
pipeline defaults, so we pass nothing and let the pipeline's own defaults apply.
`generate.py` reads back whatever the pipeline actually used and records it, so
the settings table still comes from the run rather than from a guess.

`source` records where each number came from. `verified` marks whether the entry
has been run end-to-end on this machine -- run `python generate.py --list` to
check which entries resolve against the installed diffusers before booking GPU
time.
"""

OSCBENCH_TABLE6 = "OSCBench Table 6"

MODELS = {
    # ---- open-source main experiment ---------------------------------------
    # Both rows reproduce OSCBench Table 6 exactly. Note that their "Wan-2.2"
    # row is 81 frames at 16 FPS, which is the A14B configuration -- TI2V-5B
    # runs 121 frames at 24 FPS. Matching them means A14B.
    "wan2.2-t2v-a14b": {
        "repo_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "pipeline_class": "WanPipeline",
        # Wan ships a bespoke VAE that diffusers loads separately and keeps in
        # fp32; loading it at the pipeline dtype produces black frames.
        "vae_class": "AutoencoderKLWan",
        "vae_dtype": "float32",
        "dtype": "bfloat16",
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        # A14B is a two-expert MoE and diffusers exposes a second guidance scale
        # for the low-noise expert. Left unset with everything else.
        "extra": {},
        "source": f"{OSCBENCH_TABLE6} (Wan-2.2 row: 1280x720, 81f, 16fps)",
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
        "source": f"{OSCBENCH_TABLE6} (HunyuanVideo-1.5 row); CHECK the diffusers "
                  f"class name and repo id against your version",
        "verified": False,
    },
    # ---- fallback, not part of the main comparison --------------------------
    # A14B is a 14B MoE and wants a lot of VRAM. TI2V-5B fits on one card and is
    # useful for the smoke test, the determinism check and timing runs. It is
    # NOT in OSCBench's table, so anything generated with it is ours alone and
    # cannot be reported alongside their Wan-2.2 numbers.
    "wan2.2-ti2v-5b": {
        "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_class": "WanPipeline",
        "vae_class": "AutoencoderKLWan",
        "vae_dtype": "float32",
        "dtype": "bfloat16",
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "source": "Wan2.2 TI2V-5B card defaults -- NOT an OSCBench row, smoke test only",
        "verified": False,
        "main_experiment": False,
    },
    # ---- proprietary reference, subset only --------------------------------
    # Not run through this script: these are API models with no seed control we
    # can share across conditions, so they get their own driver and a smaller
    # subset, mirroring OSCBench's own split (open-source models on every
    # prompt, proprietary models on a 140-prompt subset).
    # "kling-2.5-turbo": {...},
    # "veo-3.1-fast": {...},
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
    manifest can record precisely what produced each file.

    Anything absent from the config is absent here too -- an unset step count
    reaches the pipeline as "not specified" and its own default applies, which
    is what following the official implementation means. Use `resolved_defaults`
    to find out what that default turned out to be."""
    kwargs = {
        "height": cfg["height"],
        "width": cfg["width"],
        "num_frames": cfg["num_frames"],
    }
    for key in ("num_inference_steps", "guidance_scale"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    if cfg.get("negative_prompt"):
        kwargs["negative_prompt"] = cfg["negative_prompt"]
    kwargs.update(cfg.get("extra", {}))
    return kwargs


def resolved_defaults(pipe):
    """What the pipeline's own signature says for the parameters we did not set.

    Without this, "we used the defaults" is unverifiable a year later when the
    library has moved on, and the settings table would have blanks where
    OSCBench's has numbers."""
    import inspect  # noqa: PLC0415

    interesting = ("num_inference_steps", "guidance_scale", "guidance_scale_2",
                   "negative_prompt", "num_frames", "height", "width",
                   "max_sequence_length", "shift")
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return {}
    out = {}
    for name in interesting:
        p = params.get(name)
        if p is not None and p.default is not inspect.Parameter.empty:
            out[name] = p.default
    return out
