"""方法 v2 首测：同一张门控图，第二个旋钮 —— Structure Guidance 空间化。

**机制（读 pipeline 源码得来，不是猜的）**：ScaleDiff 每步做

    x_guided = x_pred + scale · (lowpass(x_ref) − lowpass(x_pred))     (refine, 管线 L29)

scale = curr_alpha/base_alpha 随去噪衰减 —— **时间上有日程，空间上是常数**。
背景的新结构（草叶丛、树冠纹理）和主体一样被拉回参考的低频，这就是
窗口家族"过平滑"病灶的所在。而 LFM 的两个分支都不含真正的新细节
（都来自 1024 的上采样），能长出新细节的只有这段受约束的去噪。

**v2**：用 v1 的同一张门控图 m（基阶段 cross-attention，免费），把引导
系数按位置调制：

    x_guided = x_pred + scale · G ⊙ (lowpass(x_ref) − lowpass(x_pred))
    G = g_bg + (1 − g_bg) · m          主体区 G=1 照旧，背景区 G=g_bg < 1

**互锁（这是 v2 的立身之本，必须用对照臂证出来，不能只说）**：
背景放松结构引导 = 打开重复的闸门；只有背景的文本条件已经不再索要主体
（v1 的门）时，放松才安全。所以跑 **四个臂**：

    base     s=0  g_bg=1     原版 ScaleDiff
    v1       s=1  g_bg=1     只有文本门
    v2only   s=0  g_bg<1     只放松引导、无文本门  <- 预测：重复回潮
    v2       s=1  g_bg<1     全套                  <- 预测：细节回来且不回潮

**预注册判据（写在跑之前）**：
    P1 细节   v2 臂背景高频能量（Laplacian 方差，背景掩码内）比 v1 臂
              高 >= 10%；
    P2 互锁   v2only 臂 delta 比 base 臂多 >= 1 个物体（放松确实会招重复），
              且 v2 臂 delta <= v1 臂 + 0.5（门压得住）；
    P3 代价   v2 臂耗时 <= base 臂 × 1.06。
    判死：P1 不过 -> 过平滑主因不在 Structure Guidance，v2 撤，
          回去重读机制（LFM 初始化 / restart 窗口长度是另两个嫌疑）。
          P2 前半不过（v2only 不回潮）-> "互锁"故事不成立，v2 降级为
          独立旋钮，卖点重写。P2 后半不过 -> g_bg 太激进，扫小些。

实现说明：只 monkeypatch 管线模块里的 `refine`。判别两处调用点靠
`scale` 的类型 —— 引导调用传的是张量（alphas_cumprod 切片），LFM 构造
传的是字面量 1（管线 L544/L571）。**这个判别依赖上游写法，升级管线时
必须复查**；每跑一臂打印被调制的调用次数（stage=2 应为 2×30=60）核对。

    python scalediff_probe/method_v2.py --gbg 0.4            # 默认 3 条 lone
    python scalediff_probe/method_v2.py --gbg 0.4 0.7        # 扫两档
    python scalediff_probe/method_v2.py --idx 0 1 2 3 4      # 换 prompt 子集
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
sys.path.insert(0, str(REPO / "help_code" / "ScaleDiff" / "SDXL"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROMPTS, NEGATIVE                        # noqa: E402
from subject_phrases import HEADS, strip_subject             # noqa: E402
from method_v0 import subject_token_ids                      # noqa: E402
from method_v1 import BlendGate, BlendCrossAttn              # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"


class SpatialGuidance:
    """替换管线模块的 refine。gbg=1 或 map 缺席时逐字节等价于原式。"""

    def __init__(self, module):
        self.mod = module
        self.orig = module.refine
        self.gate = None
        self.gbg = 1.0
        self.n_mod = 0

    def install(self):
        mod, orig, self_ = self.mod, self.orig, self

        def spatial_refine(x_ref, x_pred, scale, scale_factor=0.5):
            # 引导调用的 scale 是张量；LFM 构造传字面量 1（见 docstring 警告）
            is_guidance = torch.is_tensor(scale)
            if (not is_guidance or self_.gbg >= 1.0
                    or self_.gate is None or self_.gate.map is None):
                return orig(x_ref, x_pred, scale, scale_factor)
            B, C, H, W = x_pred.shape
            h, w = int(H * scale_factor), int(W * scale_factor)
            ref_ud = F.interpolate(F.interpolate(
                x_ref, size=(h, w), mode="bilinear", antialias=True),
                size=(H, W), mode="bilinear")
            pred_ud = F.interpolate(F.interpolate(
                x_pred, size=(h, w), mode="bilinear", antialias=True),
                size=(H, W), mode="bilinear")
            m = F.interpolate(self_.gate.map[None, None].to(
                device=x_pred.device, dtype=x_pred.dtype),
                size=(H, W), mode="bilinear")
            G = self_.gbg + (1.0 - self_.gbg) * m
            self_.n_mod += 1
            return x_pred + scale * G * (ref_ud - pred_ud)

        mod.refine = spatial_refine

    def restore(self):
        self.mod.refine = self.orig


def lap_energy(img, mask=None):
    """灰度 Laplacian 方差；mask=True 处才计入（背景细节的量尺）。"""
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))[1:-1, 1:-1]
    if mask is not None:
        mk = mask[1:-1, 1:-1]
        if mk.sum() < 100:
            return float("nan")
        lap = lap[mk]
    return float((lap ** 2).mean())


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4],
                    help="PROMPTS 下标；默认 3 条 lone（0-4 是 lone 类）")
    ap.add_argument("--gbg", type=float, nargs="+", default=[0.4])
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--out", default=str(root / "method_v2"))
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--detector", action="store_true",
                    help="额外跑已判死的检测器计数列（默认不跑：它只作方向"
                         "参考，却要在 SDXL 之后再占显存。权威计数走 "
                         "vlm_count.py --armdelta）")
    a = ap.parse_args()

    import pipeline_scalediff_sdxl as sd_mod
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

    sg = SpatialGuidance(sd_mod)
    sg.install()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mani = out / "manifest.jsonl"
    done, foreign = set(), 0
    if mani.exists():
        # **这个目录名早期被别的实验用过**（30 条诊断集上的 v0/v1/v2 选型），
        # 那批行没有 arm 字段。断点续跑只认本脚本自己的 schema；
        # 外来行既不计入 done，也不进测量 —— 否则会拿别的实验的图
        # 冒充某个臂，是最难查的那种污染。
        for l in mani.open():
            r = json.loads(l)
            if "arm" in r and "idx" in r:
                done.add((r["idx"], r["arm"]))
            else:
                foreign += 1
        print(f"manifest 已有 {len(done)} 条本实验的行，跳过")
        if foreign:
            print(f"**忽略 {foreign} 行外来 schema**（无 arm 字段，"
                  f"应是早期实验留在 {out.name}/ 里的）。"
                  f"\n  它们不会进测量。若想彻底隔离，用 --out 换个目录。")

    arms = [("base", 0.0, 1.0), ("v1", 1.0, 1.0)]
    for g in a.gbg:
        arms += [(f"v2only_g{g:g}", 0.0, g), (f"v2_g{g:g}", 1.0, g)]

    orig_step = pipe.noise_pred_step
    with mani.open("a") as mf:
        for idx in a.idx:
            cat, subj, prompt = PROMPTS[idx]
            head = HEADS[idx]
            # strip_subject 返回 (去主体 prompt, 被摘掉的那段) —— 必须解包，
            # 否则整个元组会被当成 prompt 喂进 encode_prompt
            nosubj, removed = strip_subject(prompt, head) if head else (prompt, "")
            tok_ids = subject_token_ids(pipe, prompt, head or subj)
            print(f"\n[{idx}] {cat}/{subj}  主体 token {tok_ids}")
            print(f"    去主体: {nosubj[:70]}")
            print(f"    摘掉了: {removed!r}")
            if not removed:
                print("    **警告：一个词都没摘掉** —— v1/v2 两臂的替代文本"
                      "与原 prompt 相同，门会退化成无操作，这一条的结果无效")

            for arm, s, gbg in arms:
                if (idx, arm) in done:
                    continue
                gate = BlendGate(tok_ids, strength=s)
                if s > 0:
                    pe, npe, _, _ = pipe.encode_prompt(
                        prompt=nosubj, device="cuda", num_images_per_prompt=1,
                        do_classifier_free_guidance=True, negative_prompt=NEGATIVE)
                    gate.alt = torch.cat([npe, pe])
                procs = dict(pipe.unet.attn_processors)
                for k_ in procs:
                    if k_.endswith("attn2.processor"):
                        procs[k_] = BlendCrossAttn(gate)
                pipe.unet.set_attn_processor(procs)
                # v2only 臂 s=0 但仍要 map（引导调制需要它）——所以门控图
                # 永远记录、s 只控制文本混合。BlendGate.weights 在 s<=0 时
                # 返回 None，文本侧自动退回原版，互不干扰。
                sg.gate, sg.gbg, sg.n_mod = gate, gbg, 0

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
                            generator=torch.Generator(device="cuda").manual_seed(a.seed),
                            num_inference_steps=50, guidance_scale=7.5,
                            restart_ratio=0.4, scale_factor=0.125,
                            upsample_stage=a.stage)
                dt = time.time() - t0
                files = {}
                for im in imgs:
                    p = out / f"{idx:02d}_{arm}_{im.width}.png"
                    im.save(p)
                    files[im.width] = p.name
                mpath = out / f"{idx:02d}_{arm}_mask.pt"
                if gate.map is not None:
                    torch.save(gate.map.cpu(), mpath)   # 落盘前离开 GPU
                mf.write(json.dumps({
                    "idx": idx, "arm": arm, "s": s, "gbg": gbg,
                    "sec": round(dt, 1), "n_modulated": sg.n_mod,
                    "files": files, "mask": mpath.name if gate.map is not None else None,
                    "prompt": prompt}, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"    {arm:<14} {dt:6.1f}s  调制 {sg.n_mod:>3} 次"
                      + ("  <- 应为 0" if gbg >= 1 else
                         "  <- 应 >0 且各 gbg<1 臂相等（实测 stage2 = 40）"))
                pipe.noise_pred_step = orig_step
    sg.restore()

    if a.skip_measure:
        print("\n跳过测量。之后单独跑本脚本（生成会因 manifest 全命中而跳过）。")
        return 0

    # ---------- 测量：背景高频（P1）----------
    print("\n========== 测量 ==========")
    from PIL import Image
    import numpy as np
    # 检测器默认不加载：它已降级为方向参考（§8.9a 判死），却要在 SDXL 之后
    # 再占一次显存 —— 白添一个崩溃点，而生成结果已经落盘，不该为它冒险。
    det = tok = None
    if a.detector:
        from count_objects import Detector
        from caption_audit import conditioned_text, load_tokenizer
        det, tok = Detector(), load_tokenizer()
    rows = [json.loads(l) for l in mani.open()]
    rows = [r for r in rows if "arm" in r and "idx" in r and r.get("files")]
    res_hi = 1024 * (2 ** a.stage)
    if not rows:
        print("manifest 里没有本实验的行，无可测量。")
        return 0

    # **计数这一列用的是已判死的检测器**（§8.9a：饱和在 2–3 个，
    # bias ≈ −(card−2)）。P1 是 Laplacian 能量，与它无关，照常判读；
    # **P2a/P2b 是纯计数判据，不以这一列定案** —— 权威读数来自
    #     python scalediff_probe/vlm_count.py --armdelta --dir <本目录>
    # 这里保留该列只作方向参考，且模型已占显存、装不下 VLM，只能分两趟。
    print(f"\n{'idx':<4}{'arm':<14}{'背景高频':>10}{'count1024':>10}"
          f"{'count_hi↓1024':>14}{'delta*':>7}{'sec':>7}")
    print("-" * 66)
    stats = {}
    for r in sorted(rows, key=lambda r: (r["idx"], r["arm"])):
        f_hi = r["files"].get(str(res_hi)) or r["files"].get(res_hi)
        f_lo = r["files"].get("1024") or r["files"].get(1024)
        if not f_hi or not f_lo:
            continue
        im_hi = Image.open(out / f_hi).convert("RGB")
        im_lo = Image.open(out / f_lo).convert("RGB")
        mask_bg = None
        if r.get("mask"):
            # mask 是在 CUDA 上 torch.save 的，load 会原样恢复到 GPU —— 测量
            # 是纯 CPU 步骤，强制落回 CPU（顺便兼容没有 GPU 的机器上重算）
            m = torch.load(out / r["mask"], map_location="cpu").float().numpy()
            m = np.array(Image.fromarray((m * 255).astype("uint8"))
                         .resize(im_hi.size, Image.BILINEAR)) / 255.0
            mask_bg = m < 0.3
        e_bg = lap_energy(im_hi, mask_bg)
        if det is not None:
            from caption_audit import conditioned_text
            text, _, _ = conditioned_text(tok, r["prompt"])
            n_lo = len(det.detect(im_lo, text)[0])
            n_dn = len(det.detect(
                im_hi.resize(im_lo.size, Image.LANCZOS), text)[0])
            d = n_dn - n_lo
        else:
            n_lo = n_dn = "-"
            d = None
        stats.setdefault(r["idx"], {})[r["arm"]] = (e_bg, d)
        print(f"{r['idx']:<4}{r['arm']:<14}{e_bg:>10.1f}{n_lo:>10}{n_dn:>14}"
              f"{('-' if d is None else f'{d:+d}'):>7}{r['sec']:>7.0f}")

    print("\n判读（预注册在 docstring）：")
    for g in a.gbg:
        p1 = p2a = p2b = n = 0
        for idx, arm_d in stats.items():
            if "v1" not in arm_d or f"v2_g{g:g}" not in arm_d:
                continue
            n += 1
            e1, d1 = arm_d["v1"]
            e2, d2 = arm_d[f"v2_g{g:g}"]
            eb, db = arm_d.get("base", (float("nan"), None))
            _, dvo = arm_d.get(f"v2only_g{g:g}", (None, None))
            p1 += (e2 >= e1 * 1.10)
            if None not in (dvo, db):
                p2a += (dvo >= db + 1)
            if None not in (d1, d2):
                p2b += (d2 <= d1 + 0.5)
        print(f"  g_bg={g:g}   **P1 细节 {p1}/{n} 过（这一条现在就算数）**"
              f"   P2a* 互锁(v2only 回潮) {p2a}/{n}"
              f"   P2b* 门压得住 {p2b}/{n}")
    print("  P1 多数不过 -> 过平滑主因不在 Structure Guidance，v2 撤；"
          "\n  P2a 多数不过 -> 互锁故事不成立，v2 降级为独立旋钮；"
          "\n  P2b 多数不过 -> g_bg 太激进，扫小些。"
          "\n  数字之外必看图：背景是长出真细节还是长出噪声/伪影。"
          "\n\n  ***** 带 * 的两列来自已判死的检测器（§8.9a 饱和在 2–3 个），"
          "\n  只作方向参考，P2a/P2b 不以它定案。权威计数分开跑："
          "\n      python scalediff_probe/vlm_count.py --armdelta --dir "
          + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
