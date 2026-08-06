"""Batch-run StyleSSP over a pair list, loading each model exactly once.

WHY NOT JUST LOOP infer_style.py

Their script is one monolithic `__main__` that loads BLIP2, the IP-Adapter-
Instruct pipeline, the inversion pipes and the ControlNet pipeline for a SINGLE
pair. Looping it by subprocess would reload ~26 GB of weights per pair -- two to
three minutes of loading for ~40 s of work, thirty times over.

So this file reuses their functions (import, not copy) and restructures the
schedule. That buys two things beyond speed:

  * VRAM. Their order is load-BLIP2 -> load-instruct -> caption -> invert ->
    del. All three coexist during inversion at roughly 25 GB, which does not fit
    a 24 GB 4090. Here every caption is produced first and BLIP2 is freed before
    anything else loads, and the inversion pipes are freed before the ControlNet
    pipeline is built, so the two heaviest stages are never co-resident.
  * The alpha ablation. --alpha 1.0 disables frequency manipulation while
    leaving every other component untouched. That is the internal control the
    break test needs; without it a detail-loss measurement cannot be attributed
    to the filter (see sheet.py).

Their code is imported, never edited -- the vendored tree stays identical to
upstream so we can always diff against it.

UNTESTED. Written against the code as read, on a machine with no GPU and no
diffusers. Every stage prints what it loaded and every failure is loud. Run
--limit 1 first and read the output before trusting anything.

    python run_batch.py --pairs ../stylessp_probe/pairs.jsonl --limit 1
    python run_batch.py --pairs pairs.jsonl --out results
    python run_batch.py --pairs pairs.jsonl --out results_alpha1 --alpha 1.0
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


def free():
    gc.collect()
    torch.cuda.empty_cache()


def vram():
    return torch.cuda.max_memory_allocated() / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stylessp", type=Path,
                    default=Path(__file__).parent.parent / "help_code/StyleSSP")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="frequency manipulation strength; 1.0 disables it and "
                         "is the ablation control")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    # Their modules resolve `src.*` and the pipeline files relative to the repo
    # root, so we import from inside it rather than copying anything out.
    sys.path.insert(0, str(args.stylessp.resolve()))
    import infer_style as S
    from src.config import RunConfig
    from src.eunms import Model_Type, Scheduler_Type
    from src.frequency_utils import freq_exp
    from src.utils.enums_utils import get_pipes
    from inversion import run as invert
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    from transformers import CLIPVisionModelWithProjection
    from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    from PIL import Image
    import cv2
    import numpy as np

    rows = [json.loads(l) for l in args.pairs.read_text().splitlines() if l]
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in args.out.glob("*.png")}

    cfg = RunConfig(model_type=Model_Type.SDXL,
                    scheduler_type=Scheduler_Type.DDIM,
                    num_inference_steps=args.steps,
                    num_inversion_steps=args.steps,
                    num_renoise_steps=1,
                    perform_noise_correction=False,
                    seed=args.seed)
    print(f"{len(rows)} pairs | alpha={args.alpha} "
          f"({'RELEASED' if args.alpha == 0.7 else 'ABLATION'}) | "
          f"{args.steps} steps @ {cfg.resolution}px")

    # ---------------------------------------------------------------- captions
    # First and alone: BLIP2 is ~8 GB and is never needed again after this.
    print("\n[1/4] BLIP2 -> captions")
    proc = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", device_map="cuda",
        torch_dtype=torch.float16).eval()
    S.processor, S.model = proc, blip          # generate_caption reads globals
    caps = {}
    for r in rows:
        for key in ("content", "style"):
            p = r[key]
            if p not in caps:
                im = Image.open(p).convert("RGB").resize(
                    (cfg.resolution, cfg.resolution))
                caps[p] = S.generate_caption(im)
                print(f"  {Path(p).name}: {caps[p]}")
    del blip, proc, S.model, S.processor
    S.model = S.processor = None
    free()
    print(f"  freed BLIP2; peak so far {vram():.1f} GB")

    # ------------------------------------------------------- inversion + freq
    print("\n[2/4] IP-Adapter-Instruct + inversion pipes")
    ip_instruct = S.init_models(cfg)
    pipe_inv, pipe_inf = get_pipes(Model_Type.SDXL, Scheduler_Type.DDIM,
                                   device=cfg.device,
                                   model_name=cfg.base_model_path)
    latents, embeds = {}, {}
    for i, r in enumerate(rows, 1):
        cp, sp = r["content"], r["style"]
        style_im = Image.open(sp).convert("RGB").resize(
            (cfg.resolution, cfg.resolution))
        cont_im = Image.open(cp).convert("RGB").resize(
            (cfg.resolution, cfg.resolution))
        e = {
            "style": ip_instruct.get_decouple_embeds(
                pil_image=style_im, prompt="", query="use the style from the image"),
            "style_content": ip_instruct.get_decouple_embeds(
                pil_image=style_im, prompt="", query="use the composition from the image"),
            "content": ip_instruct.get_decouple_embeds(
                pil_image=cont_im, prompt="", query="use the composition from the image"),
            "content_style": ip_instruct.get_decouple_embeds(
                pil_image=cont_im, prompt="", query="use the style from the image"),
        }
        _, inv_latent, _, _ = invert(
            cont_im, caps[cp], cfg, pipe_inversion=pipe_inv,
            pipe_inference=pipe_inf, do_reconstruction=False,
            feature_extractor=ip_instruct,
            style_embedding=e["style"], content_embedding=e["content"],
            neg_style_embedding=e["content_style"],
            neg_content_embedding=e["style_content"],
            enable_guidance=False, used_NPI_guidance=True)
        # d_s/d_t/filter_type exactly as released; only alpha is ours to vary.
        _, latent_l, _ = freq_exp(inv_latent, d_s=0.3, d_t=0.9,
                                  alpha=args.alpha, filter_type="gaussian_b")
        latents[r["id"]] = latent_l.to(inv_latent.dtype)
        embeds[r["id"]] = e
        print(f"  {i}/{len(rows)} inverted {r['id']}  peak {vram():.1f} GB",
              flush=True)
    del pipe_inv, pipe_inf
    free()
    print(f"  freed inversion pipes; peak {vram():.1f} GB")

    # ------------------------------------------------------ sampling pipeline
    print("\n[3/4] ControlNets + SDXL inpaint pipeline")
    controlnet = [
        ControlNetModel.from_pretrained(cfg.tile_controlnet_path,
                                        torch_dtype=torch.float16,
                                        use_safetensors=True).to(cfg.device),
        ControlNetModel.from_pretrained(cfg.canny_controlnet_path,
                                        torch_dtype=torch.float16,
                                        variant="fp16").to(cfg.device),
    ]
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=cfg.dtype).to(cfg.device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=cfg.dtype).to(cfg.device)
    pipe = S.StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        cfg.base_model_path, controlnet=controlnet, vae=vae,
        image_encoder=image_encoder, torch_dtype=torch.float16,
        use_safetensors=True, variant="fp16").to(cfg.device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.enable_gradient_checkpointing()
    pipe.load_ip_adapter(cfg.IP_path, subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl_vit-h.safetensors",
                         image_encoder_folder=None)
    pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, 2.5, 0.0]}})
    mask = Image.new("RGB", (cfg.resolution, cfg.resolution), (255, 255, 255))

    print("\n[4/4] sampling")
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        name = f"ours_{Path(r['content']).stem}_{Path(r['style']).stem}.png"
        if name in done:
            print(f"  {i}/{len(rows)} {name} exists, skipping")
            continue
        cp, sp = r["content"], r["style"]
        cont_im = Image.open(cp).convert("RGB").resize(
            (cfg.resolution, cfg.resolution))
        style_im = Image.open(sp).convert("RGB").resize(
            (cfg.resolution, cfg.resolution))
        tile = load_image(cp).resize((cfg.resolution, cfg.resolution))
        canny = S.get_canny_map(np.array(cv2.imread(cp))).resize(
            (cfg.resolution, cfg.resolution))
        e = embeds[r["id"]]
        out = pipe(
            prompt=caps[cp],
            negative_prompt="watermark, lowres, low quality, worst quality, "
                            "deformed, glitch, low contrast, noisy, saturation, "
                            "blurry",
            num_inference_steps=cfg.num_inference_steps, eta=1.0,
            mask_image=mask, image=cont_im, control_image=[tile, canny],
            ip_adapter_image=style_im,
            generator=torch.Generator(device="cpu").manual_seed(cfg.seed),
            latents=latents[r["id"]],
            guidance_scale=cfg.guidance_scale,
            controlnet_conditioning_scale=[0.25, 0.40], npi_interp=0.5,
            style_embeddings_instruct=e["style"],
            content_embeddings_instruct=e["content"],
            style_guidance_scale=cfg.style_guidance_scale,
            content_guidance_scale=cfg.content_guidance_scale,
            ip_instruct_model=ip_instruct, CSD_model=None,
            inv_guidance=cfg.inv_guidance, feature_extractor=ip_instruct,
            do_NPI=False).images[0]
        out.save(args.out / name)
        rate = (time.time() - t0) / i
        print(f"  {i}/{len(rows)} {name}  {rate:.0f}s/pair  "
              f"eta {(len(rows) - i) * rate / 60:.0f}min  peak {vram():.1f} GB",
              flush=True)

    (args.out / "run.json").write_text(json.dumps({
        "alpha": args.alpha, "steps": args.steps, "seed": args.seed,
        "resolution": cfg.resolution, "control": "tile_canny",
        "scales": [0.25, 0.40], "n": len(rows),
        "peak_gb": round(vram(), 2)}, indent=2))
    print(f"\ndone -> {args.out}\nnow: python sheet.py --pairs {args.pairs} "
          f"--out-dir {args.out}")


if __name__ == "__main__":
    main()
