#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四臂 Lightning 看图局。**这不是一个测量实验，是第 2、3 步：跑 + 看。**

    不算指标、不做表、不做统计、不预注册判据。产出是一批图和一张对照表，
    然后用眼睛看。`survey_slow_turnover.md` 底部那份 playbook 的第 2、3 步，
    我们从来没有正经做过 —— §10 三个探针全部是「先有假设、再测一个数」，
    测量只能确认你已经怀疑的东西，不会让你意外。

────────────────────────────────────────────────────────────────────────
为什么是这个组合
────────────────────────────────────────────────────────────────────────

  能靠「看」发现的失效必须**大**（§10 证明小失效我们赢不了，差值全在
  噪声底以内）。而成熟设定里的大失效早被修了。所以大的、未被修的失效
  只可能住在**比它任何一个部件都新的组合**上。

  少步蒸馏 × HR 脚手架就是这样一个组合，而且算术上就该坏：
      ScaleDiff 的 restart = int(steps * (1 - restart_ratio))
      50 步 / r=0.4 → 第 30 步
      **4 步 / r=0.4 → 第 2 步**        ← 整个 LFM+SG 被压进两步
  DemoFusion 的 skip residual 靠 cosine_scale_* 沿整条轴衰减；4 步里
  这条衰减曲线只有 4 个采样点。

  ⚠️ 本文件不主张「这是个空位」。空位检索是拿错了仪器（同上文档）。
     这里只主张：这是目前进入「跑 + 看」最便宜的入口。

────────────────────────────────────────────────────────────────────────
四个臂（第四个是我上一版漏掉的控制组）
────────────────────────────────────────────────────────────────────────

  direct        Lightning 直出 2048²，不挂任何 HR 方法
                ← **没有它就分不清「HR 方法坏了」和「Lightning 本来
                   就画不了 2048」**
  demofusion    CVPR'24 原型：progressive upscaling + skip residual
  accdiffusion  ECCV'24：与 DemoFusion **同一份代码**（scheduler 处理
                逐行相同，行号只差 2）+ patch-content-aware prompt
  scalediff     NeurIPS'25：独立实现，restart noise τ + LFM + SG

  2 同 1 异，所以三家的差异本身是信息：
    · 三家坏法相同        → 病因在 backbone（蒸馏本身）
    · Demo/Acc 同、Scale 异 → 病因在共享的 DemoFusion 脚手架
    · 三家各坏各的        → 各自脚手架各自的失效，挑最能命名的

────────────────────────────────────────────────────────────────────────
崩溃是数据，不是失败
────────────────────────────────────────────────────────────────────────

  每个 (prompt, arm) 单独 try/except，traceback 写进 errors.jsonl。
  **崩在哪，哪条脚手架假设就是承重的。** 一个臂全崩不影响别的臂 ——
  这也是为什么按 prompt 交错而不是按臂串行（p0_run.py 的教训）。

用法：
    source scalediff_probe/env.sh
    python fewstep_probe/look.py --plan                    # 零 GPU
    python fewstep_probe/look.py --run                     # 4 步，约 1 h
    python fewstep_probe/look.py --run --steps 50 --tag ref  # 50 步参照（可选）
    python fewstep_probe/look.py --sheet                   # 拼对照表 → 看
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "fewstep"

ARMS = ("direct", "demofusion", "accdiffusion", "scalediff")
TARGET = 2048          # 1024 的 2×。选 2048 不选 4096 是因为 4096 就是
                       # §10 里把我们拖成周级的那个分辨率。
BASE = 1024
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"

# 12 条 prompt。选它们**不是为了覆盖分布**，是为了让失效可见 ——
# 这是看图局不是评测。每条后面写清楚它想暴露什么。
PROMPTS = [
    ("a close-up photograph of a peacock feather",              "细密周期纹理：过平滑/摩尔纹最先在这里露"),
    ("a bustling medieval marketplace with many merchants",     "多主体：重复伪影的经典触发"),
    ("an astronaut riding a horse on a beach at sunset",        "单主体 + 大片天空：接缝/色带在平坦区最显眼"),
    ("a dense forest canopy seen from below",                   "高频无结构纹理：细节坍塌"),
    ("a red brick wall with a green wooden door",               "规则网格 + 直线：几何断裂/错位"),
    ("a bowl of ramen with chopsticks on a wooden table",       "中景静物：常见构图，基准参照"),
    ("a snow-covered mountain range under a clear blue sky",    "极大面积低频：能量衰减/发灰"),
    ("a portrait of an elderly man with a detailed beard",      "人脸 + 细发丝：语义关键区的退化最刺眼"),
    ("a city street at night with neon signs and rain",         "高动态 + 小字：文字与光斑"),
    ("a butterfly resting on a sunflower",                      "小主体 + 大背景：主体被上采样稀释"),
    ("an intricate stained glass window in a cathedral",        "强边缘 + 规则分区：分块边界"),
    ("a herd of zebras on the savanna",                         "条纹 + 多实例：重复与纹理同时受压"),
]


def cmd_plan(args):
    n = len(PROMPTS)
    # 4 步 2048²：direct ~2s；三个 HR 方法要跑多块 + 多阶段，估 8-20s
    est = {"direct": 2, "demofusion": 15, "accdiffusion": 18, "scalediff": 10}
    tot = sum(est[a] for a in ARMS) * n
    print("═" * 70)
    print("四臂 Lightning 看图局 —— 这是 playbook 的第 2、3 步（跑 + 看）")
    print("═" * 70)
    print(f"\n这不是测量实验：不算指标、不做表、不做统计、不预注册判据。")
    print(f"产出 = {n * len(ARMS)} 张图 + 一张对照表，然后用眼睛看。\n")

    print(f"臂：")
    for a in ARMS:
        note = "  ← 控制组，没有它分不清是 HR 方法坏了还是 Lightning 画不了 2048" \
               if a == "direct" else ""
        print(f"  {a:14s} ~{est[a]:2d}s/张{note}")
    print(f"\n  {n} prompts × {len(ARMS)} 臂 = {n*len(ARMS)} 张，约 {tot/60:.0f} 分钟")
    print(f"  分辨率 {TARGET}²（不是 4096² —— 那个分辨率就是 §10 里把我们"
          f"拖成周级的元凶）")

    print(f"\n算术上该坏的地方（不是假设，是除法）：")
    for st in (50, args.steps):
        print(f"  {st:2d} 步 / restart_ratio=0.4 → ScaleDiff 在第 "
              f"{int(st*(1-0.4))} 步重启注噪"
              + ("        ← 整个 LFM+SG 压进 2 步" if st <= 4 else ""))

    print(f"\n四种结果，都可接受：")
    print(f"  A 可见、可命名、且三臂坏法**不同**  → 进第 4 步（命名），再第 5 步（查）")
    print(f"  B 就是普遍地糊，没有结构            → 组合只是烂 → 停")
    print(f"  C 什么都没坏                        → 「依赖时间轴」是假的，意外但不成篇")
    print(f"  D **根本跑不起来**                  → 也是数据：崩在哪，"
          f"哪条脚手架假设就是承重的")

    print(f"\n开跑前唯一的阻塞项：")
    print(f"  Lightning 权重 {LIGHTNING_REPO}/{LIGHTNING_CKPT} 要在本地。")
    print(f"  服务器 HF_HUB_OFFLINE=1 钉死，用 scalediff_probe/fetch_weights.sh "
          f"走 hf-mirror 先拉（~5 GB）。")
    print()


def build_pipes(steps, device="cuda"):
    """四个 pipeline 共享同一套权重（unet / vae / text_encoder），
    只有 pipeline 类不同 —— 这样臂间差异只来自脚手架，不来自权重。"""
    import torch
    from diffusers import (StableDiffusionXLPipeline, UNet2DConditionModel,
                           EulerDiscreteScheduler)

    sys.path.insert(0, str(REPO / "help_code" / "DemoFusion"))
    sys.path.insert(0, str(REPO / "help_code" / "AccDiffusion"))
    sys.path.insert(0, str(REPO / "help_code" / "ScaleDiff" / "SDXL"))
    from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
    from accdiffusion_sdxl import AccDiffusionSDXLPipeline
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    base = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

    lightning = steps <= 8
    if lightning:
        # Lightning = 换 UNet 权重 + 换 scheduler 到 Euler/trailing。
        # trailing 不是可选项：Lightning 就是在这个 spacing 上蒸馏的，
        # 用默认 leading 出来的图会明显更差，那是**我们自己引入的伪影**，
        # 不是要找的失效。
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        unet = UNet2DConditionModel.from_config(base.unet.config).to(device, torch.float16)
        unet.load_state_dict(load_file(
            hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT), device=device))
        base.unet = unet
        base.scheduler = EulerDiscreteScheduler.from_config(
            base.scheduler.config, timestep_spacing="trailing")

    kw = dict(vae=base.vae, text_encoder=base.text_encoder,
              text_encoder_2=base.text_encoder_2, tokenizer=base.tokenizer,
              tokenizer_2=base.tokenizer_2, unet=base.unet,
              scheduler=base.scheduler)
    pipes = {
        "direct":       base,
        "demofusion":   DemoFusionSDXLPipeline(**kw),
        "accdiffusion": AccDiffusionSDXLPipeline(**kw),
        "scalediff":    CustomStableDiffusionXLPipeline(**kw),
    }
    for p in pipes.values():
        p.to(device)
        p.set_progress_bar_config(disable=True)
        try:
            p.vae.enable_tiling()
        except Exception:
            pass
    return pipes, lightning


def call_arm(arm, pipe, prompt, gen, steps, cfg):
    """每个臂的调用约定不同，写在一处。
    · ScaleDiff 返回**裸 list**，不是 .images（pipeline 第 588 行 return output_images）
    · ScaleDiff 的 upsample_stage=1 给 2×，从 1024 到 2048
    · DemoFusion / AccDiffusion 返回每个 scale 一张，要取最后一张"""
    common = dict(prompt=prompt, generator=gen, num_inference_steps=steps,
                  guidance_scale=cfg)
    if arm == "direct":
        return pipe(height=TARGET, width=TARGET, **common).images[0]
    if arm in ("demofusion", "accdiffusion"):
        extra = {"c": 0.3} if arm == "accdiffusion" else {}
        r = pipe(height=TARGET, width=TARGET, view_batch_size=8, stride=64,
                 multi_decoder=True, **extra, **common)
        imgs = r if isinstance(r, list) else r.images
        return imgs[-1]                      # 最后一个 scale 才是 2048²
    if arm == "scalediff":
        r = pipe(height=BASE, width=BASE, restart_ratio=0.4, scale_factor=0.125,
                 upsample_stage=1, **common)
        imgs = r if isinstance(r, list) else r.images
        return imgs[-1]
    raise ValueError(arm)


def cmd_run(args):
    import torch
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    errf = out / "errors.jsonl"

    # Lightning 是 CFG-free 蒸馏的：guidance_scale>1 会明显劣化。
    # 但三个 HR pipeline 内部都用 `do_classifier_free_guidance = guidance_scale > 1`
    # 来决定要不要建负分支 —— 传 0 就是把负分支整条关掉。
    # **这本身可能是失效的来源之一，所以要记下来，不要事后才想起。**
    cfg = args.cfg if args.cfg is not None else (0.0 if args.steps <= 8 else 7.5)
    print(f"steps={args.steps}  guidance_scale={cfg}  tag={args.tag}  → {out}")

    pipes, lightning = build_pipes(args.steps)
    (out / "config.json").write_text(json.dumps({
        "steps": args.steps, "cfg": cfg, "lightning": lightning,
        "target": TARGET, "arms": list(ARMS),
        "scalediff_restart_step": int(args.steps * (1 - 0.4)),
    }, indent=2))

    t0 = time.time()
    # 按 prompt 外层、臂内层交错。一个臂全崩也不影响别的臂已完成的部分，
    # 中断后 partial 数据仍然是臂对齐的（p0_run.py 的教训）。
    for i, (prompt, why) in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt}\n      ({why})", flush=True)
        for arm in ARMS:
            path = out / f"{i:02d}_{arm}.png"
            if path.exists():
                print(f"      {arm:14s} skip"); continue
            # 同一个 seed 喂所有臂 —— 图才能并排看
            gen = torch.Generator("cuda").manual_seed(1000 + i)
            t = time.time()
            try:
                img = call_arm(arm, pipes[arm], prompt, gen, args.steps, cfg)
                img.save(path)
                print(f"      {arm:14s} ok   {time.time()-t:5.1f}s  {img.size}")
            except Exception as e:
                tb = traceback.format_exc()
                with errf.open("a") as f:
                    f.write(json.dumps({"i": i, "arm": arm, "prompt": prompt,
                                        "err": repr(e), "tb": tb},
                                       ensure_ascii=False) + "\n")
                # 崩溃是数据。打出最后一行给人看，全文进 errors.jsonl。
                print(f"      {arm:14s} !! {type(e).__name__}: "
                      f"{str(e)[:110]}")

    n_err = sum(1 for _ in errf.open()) if errf.exists() else 0
    print(f"\n{(time.time()-t0)/60:.0f} 分钟。崩溃 {n_err} 次"
          + (f" → {errf}（**这是数据，读它**）" if n_err else ""))
    print(f"下一步：python fewstep_probe/look.py --sheet --tag {args.tag}")


def cmd_sheet(args):
    """拼对照表。同一行是同一个 prompt 的四个臂，同 seed，可以直接并排看。
    看图局的产出就是这张表 —— 不是任何数字。"""
    from PIL import Image, ImageDraw
    out = OUT / args.tag
    cell, pad, hdr = args.cell, 6, 26
    rows = [(i, p) for i, (p, _) in enumerate(PROMPTS)
            if any((out / f"{i:02d}_{a}.png").exists() for a in ARMS)]
    if not rows:
        sys.exit(f"!! {out} 里没有图，先 --run")

    W = pad + len(ARMS) * (cell + pad)
    H = hdr + len(rows) * (cell + hdr + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for c, a in enumerate(ARMS):
        d.text((pad + c * (cell + pad) + 4, 6), a, fill="black")
    y = hdr
    for i, prompt in rows:
        d.text((pad + 4, y + 4), f"[{i:02d}] {prompt[:110]}", fill="black")
        for c, a in enumerate(ARMS):
            x = pad + c * (cell + pad)
            f = out / f"{i:02d}_{a}.png"
            box = (x, y + hdr, x + cell, y + hdr + cell)
            if f.exists():
                im = Image.open(f).convert("RGB")
                im.thumbnail((cell, cell), Image.LANCZOS)
                sheet.paste(im, (x, y + hdr))
            else:
                d.rectangle(box, outline="red")
                d.text((x + 8, y + hdr + cell // 2), "CRASH", fill="red")
        y += cell + hdr + pad

    p = out / f"sheet_{args.tag}.png"
    sheet.save(p)
    print(f"{p}   ({len(rows)} prompts × {len(ARMS)} 臂)")
    print(f"\n把它拉下来看。要找的是**三臂坏法的差异**，不是哪张更好看：")
    print(f"  · direct 也坏 → 病因在 Lightning，与 HR 方法无关")
    print(f"  · demofusion 与 accdiffusion 一样、scalediff 不同 → 病因在共享脚手架")
    print(f"  · 三家各坏各的 → 各自脚手架的失效，挑最能命名的")
    errf = out / "errors.jsonl"
    if errf.exists():
        print(f"\n!! 还有 {sum(1 for _ in errf.open())} 条崩溃在 {errf} —— "
              f"崩在哪，哪条假设就是承重的，别跳过")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="4step", help="输出子目录名")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=None,
                    help="默认：≤8 步用 0.0（Lightning 是 CFG-free 蒸馏的），否则 7.5")
    ap.add_argument("--cell", type=int, default=460, help="对照表每格边长")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    (cmd_plan if a.plan else cmd_run if a.run else cmd_sheet)(a)


if __name__ == "__main__":
    main()
