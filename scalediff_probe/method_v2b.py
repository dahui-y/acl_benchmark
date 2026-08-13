"""v2b 首测：空间化 LFM 噪声原料 —— 预注册见 PAPER_SKELETON §3.2vb。

挂点：管线 L547-548
    noise = torch.randn_like(latents_LFM)
    latents = self.scheduler.add_noise(latents_LFM, noise, t_restart)
这勺噪声是放大阶段细节的"原料"，空间均匀。v2b 把它按门控图调制：
    noise' = noise · (1 + (γ−1)(1−m))     背景 m≈0 -> ×γ；主体 m≈1 -> ×1
γ=1 位级恒等（乘 1.0 不改浮点值）。补丁打在 scheduler.add_noise 上，
只在张量空间边 > 128（stage>=2）且 γ>1 且门控图就绪时动手 ——
基础阶段与其他调用一律原样放行。

门控图在 add_noise 被调时还没 finalize（L548 在 stage2 第一次
noise_pred_step 之前）——补丁里主动 finalize，幂等。

    python scalediff_probe/method_v2b.py --gamma 1.5 2.0
（复用 method_v2 目录：base/v1 两臂 manifest 命中自动跳过，
只新跑 v2bonly/v2b × 两个 γ × 3 条 = 12 张，约 16 分钟。）

之后权威计数（P2'）：
    python scalediff_probe/vlm_count.py --armdelta --dir $SD_OUT/method_v2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS, NEGATIVE                        # noqa: E402
from subject_phrases import HEADS, strip_subject             # noqa: E402
from method_v0 import subject_token_ids                      # noqa: E402
from method_v1 import BlendGate, BlendCrossAttn              # noqa: E402
from method_v2 import lap_energy                             # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"


class NoiseGate:
    """补丁 scheduler.add_noise：stage>=2 时按门控图放大背景噪声。"""

    def __init__(self, scheduler):
        self.sch = scheduler
        self.orig = scheduler.add_noise
        self.gate = None
        self.gamma = 1.0
        self.n_mod = 0

    def install(self):
        self_ = self

        def patched(original_samples, noise, timesteps):
            g = self_.gate
            if (self_.gamma > 1.0 and g is not None
                    and original_samples.ndim == 4
                    and original_samples.shape[-1] > 128):
                if g.map is None:
                    g.finalize()          # add_noise 先于 stage2 首步
                if g.map is not None:
                    m = F.interpolate(
                        g.map[None, None].to(device=noise.device,
                                             dtype=noise.dtype),
                        size=original_samples.shape[-2:], mode="bilinear")
                    noise = noise * (1.0 + (self_.gamma - 1.0) * (1.0 - m))
                    self_.n_mod += 1
            return self_.orig(original_samples, noise, timesteps)

        self.sch.add_noise = patched

    def restore(self):
        self.sch.add_noise = self.orig


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--gamma", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--out", default=str(root / "method_v2"),
                    help="故意复用 method_v2 目录：base/v1 臂直接命中跳过")
    a = ap.parse_args()

    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
               / "snapshots").glob("*"):
        if not (s_ / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s_ / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    ng = NoiseGate(pipe.scheduler)
    ng.install()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mani = out / "manifest.jsonl"
    done = set()
    if mani.exists():
        for l in mani.open():
            r = json.loads(l)
            if "arm" in r and "idx" in r:
                done.add((r["idx"], r["arm"]))
        print(f"manifest 已有 {len(done)} 条带 arm 的行（base/v1 命中即复用）")

    arms = [("base", 0.0, 1.0), ("v1", 1.0, 1.0)]
    for g in a.gamma:
        arms += [(f"v2bonly_g{g:g}", 0.0, g), (f"v2b_g{g:g}", 1.0, g)]

    orig_step = pipe.noise_pred_step
    with mani.open("a") as mf:
        for idx in a.idx:
            cat, subj, prompt = PROMPTS[idx]
            head = HEADS[idx]
            nosubj, removed = strip_subject(prompt, head) if head else (prompt, "")
            tok_ids = subject_token_ids(pipe, prompt, head or subj)
            print(f"\n[{idx}] {cat}/{subj}  tok={tok_ids}  摘掉 {removed!r}")

            for arm, s, gamma in arms:
                if (idx, arm) in done:
                    continue
                gate = BlendGate(tok_ids, strength=s)
                if s > 0:
                    pe, npe, _, _ = pipe.encode_prompt(
                        prompt=nosubj, device="cuda", num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=NEGATIVE)
                    gate.alt = torch.cat([npe, pe])
                procs = dict(pipe.unet.attn_processors)
                for k_ in procs:
                    if k_.endswith("attn2.processor"):
                        procs[k_] = BlendCrossAttn(gate)
                pipe.unet.set_attn_processor(procs)
                ng.gate, ng.gamma, ng.n_mod = gate, gamma, 0

                def patched(latents, t, *args, _o=orig_step, _g=gate, **kw2):
                    ph = 1 if latents.shape[-1] <= 128 else 2
                    if ph == 2 and _g.phase == 1:
                        _g.finalize()
                    _g.phase = ph
                    return _o(latents, t, *args, **kw2)
                pipe.noise_pred_step = patched

                torch.manual_seed(a.seed)
                torch.cuda.manual_seed_all(a.seed)
                t0 = time.time()
                imgs = pipe(prompt, negative_prompt=NEGATIVE,
                            height=1024, width=1024,
                            generator=torch.Generator(device="cuda")
                            .manual_seed(a.seed),
                            num_inference_steps=50, guidance_scale=7.5,
                            restart_ratio=0.4, scale_factor=0.125,
                            upsample_stage=a.stage)
                dt = time.time() - t0
                files = {}
                for im in imgs:
                    p = out / f"{idx:02d}_{arm}_{im.width}.png"
                    im.save(p)
                    files[im.width] = p.name
                if gate.map is not None:
                    torch.save(gate.map.cpu(), out / f"{idx:02d}_{arm}_mask.pt")
                mf.write(json.dumps({
                    "idx": idx, "arm": arm, "s": s, "gamma": gamma,
                    "sec": round(dt, 1), "n_noise_mod": ng.n_mod,
                    "files": files,
                    "mask": f"{idx:02d}_{arm}_mask.pt"
                            if gate.map is not None else None,
                    "prompt": prompt}, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"    {arm:<16} {dt:6.1f}s  噪声调制 {ng.n_mod} 次"
                      + ("  <- 应为 0" if gamma <= 1 else
                         f"  <- stage={a.stage} 应为 {a.stage - 0}"))
                pipe.noise_pred_step = orig_step
    ng.restore()

    # ---------- 测量：背景 Laplacian（P1'）----------
    print("\n========== 测量（P1'）==========")
    import numpy as np
    from PIL import Image
    rows = [json.loads(l) for l in mani.open()]
    rows = [r for r in rows if "arm" in r and r.get("files")
            and (r["arm"] in ("base", "v1") or r["arm"].startswith("v2b"))]
    res_hi = 1024 * (2 ** a.stage)
    print(f"\n{'idx':<4}{'arm':<16}{'背景高频':>10}{'sec':>7}")
    stats = {}
    for r in sorted(rows, key=lambda r: (r["idx"], r["arm"])):
        f_hi = r["files"].get(str(res_hi)) or r["files"].get(res_hi)
        if not f_hi:
            continue
        im_hi = Image.open(out / f_hi).convert("RGB")
        mask_bg = None
        if r.get("mask") and (out / r["mask"]).exists():
            m = torch.load(out / r["mask"], map_location="cpu").float().numpy()
            m = np.array(Image.fromarray((m * 255).astype("uint8"))
                         .resize(im_hi.size, Image.BILINEAR)) / 255.0
            mask_bg = m < 0.3
        e = lap_energy(im_hi, mask_bg)
        stats.setdefault(r["idx"], {})[r["arm"]] = e
        print(f"{r['idx']:<4}{r['arm']:<16}{e:>10.1f}{r['sec']:>7.0f}")

    print("\n判读（预注册 §3.2vb）：")
    for g in a.gamma:
        p1 = n = 0
        for idx, d in stats.items():
            if "v1" in d and f"v2b_g{g:g}" in d:
                n += 1
                p1 += (d[f"v2b_g{g:g}"] >= d["v1"] * 1.10)
        print(f"  γ={g:g}   P1' 背景细节 {p1}/{n} 过（>= v1 × 1.10）")
    print("  γ=2 下 P1' 仍 0/3 -> 原料假说死，过平滑章三负收笔；"
          "\n  P1' 过 -> **眼睛必审**：真纹理还是彩色噪点（Laplacian 分不出）；"
          "\n  P2'（不招重复）另跑：python scalediff_probe/vlm_count.py "
          "--armdelta --dir " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
