"""干预 v1：同一次前向里喂两套文本嵌入，按位置混合。

为什么不是 v0 那样压注意力权重 —— 这是测出来的，不是猜的：

    主体 token [6] "hiker" 吸走的注意力质量 5.52%（77 个 token 里排第 3，
    均匀分布是 1.3%）。压掉它，幻影 5 -> 5，图几乎没变。
    而把 "lone hiker" 从字符串里删掉重新编码，幻影 5 -> 0。

    注意力质量的分布是平的：最高 distant 6.34%，lone 5.67%，hiker 5.52%，
    snow 5.30%… 前十名加起来才 49%。CLIP 文本编码器是因果 transformer，
    每个 token 都吸收了它前面的内容 —— "主体"不住在某一个 token 里，
    它摊在整句上。压掉一个 token，质量重新分配给 lone / standing / ridge，
    而它们的向量里照样带着"有个人站在山脊上"。

    => 干预必须作用在【文本嵌入】上，不能作用在【注意力权重】上。

做法：
    完整 prompt 的 K/V 算一遍，去主体 prompt 的 K/V 算一遍，按 Phase 1 记录
    的主体位置图逐位置混合输出：

        out = m * out_full + (1 - m) * out_nosubj

    主体所在的那 ~3% 区域拿完整 prompt（主角照常生成），其余区域拿去主体版本
    —— 场景词一个不少，所以细节不丢，但没有可实例化的主体。

    这正是 ScaleDiff 的单次全图前向给出的空间：AccDiffusion 要喂两套 prompt
    必须跑两次 patch 前向；这里两套嵌入在同一次前向里共存，逐位置取用。

    与 AccDiffusion v2 的关系要写清楚：机制（同一句 prompt 施于所有局部视野
    导致重复）是它命名的，我们继承并引用；它的修法需要独立的 patch 前向，
    在这里不可用。相似性风险真实存在，防守靠"结构上不可用"加数字。

    python scalediff_probe/method_v1.py --s 1      # 完整干预
    python scalediff_probe/method_v1.py --s 0      # 自检：应与原版一致
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "help_code" / "ScaleDiff" / "SDXL"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from method_v0 import subject_token_ids          # noqa: E402
from subject_phrases import strip_subject        # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
PROMPT = ("a photograph of a lone hiker standing on a rocky ridge, vast forested valley "
          "and distant snow mountains behind, golden hour")
SUBJECT = "hiker"
# 去主体 prompt 现在由规则自动生成，不再手写 —— 手写的版本证明不了方法。
# 这个 assert 是回归护栏：规则的输出与当初手写、并已跑出 5->0 的那一句逐字相同，
# 所以换成自动化没有改变已有结果。
PROMPT_NOSUBJ, _REMOVED = strip_subject(PROMPT, SUBJECT)
assert PROMPT_NOSUBJ == ("a photograph of a rocky ridge, vast forested valley "
                         "and distant snow mountains behind, golden hour"), PROMPT_NOSUBJ


class BlendGate:
    """主体位置图。norm 决定它怎么从原始注意力变成混合权重。

    minmax（v1，已知有问题）：逐图 (m - min) / (max - min)。
        只保留相对高低，**同一个 0.5 在不同图上代表完全不同的注意力强度**。
        实测两类失败同源：
          08_sheep  cov 11%，图太集中 —— 羊群处处注意力不低，但 min-max 拉伸后
                    只有峰值过 0.5，89% 的画面拿到去羊群嵌入，羊群被删光
          26/23     过渡带占 49% / 61%，图太弥散 —— 大半张画面拿到的既不是完整
                    prompt 也不是去主体 prompt，而是任意插值，条件混乱
        方法的实际行为因此只在 lone 上（cov 1.6~10%）符合设计。

    rel（v2）：r = m / mean(m)，**相对均匀分布的倍数**，有绝对含义 ——
        "这个位置对主体的注意力是全图平均的几倍"。以 k 倍为界，两侧留一条
        宽 width 的过渡带。一个改动治两头：羊群处处高于均值 -> 保住；
        肖像的弥散梯度被压成清晰的界 -> 过渡带收窄。
    """

    def __init__(self, token_ids, strength=1.0, canon=64,
                 norm="rel", k=1.0, width=0.5):
        self.tok = token_ids
        self.s = strength
        self.canon = canon
        self.norm, self.k, self.width = norm, k, width
        self.phase = 1
        self.alt = None           # (B, 77, D) 去主体 prompt 的嵌入，与 batch 同序
        self._acc, self._n, self.map = None, 0, None

    def record(self, probs, hw):
        if not self.tok:
            return
        h = w = int(hw ** 0.5)
        if h * w != hw:
            return
        m = probs[-1, :, :, self.tok].sum(-1).mean(0).reshape(1, 1, h, w).float()
        m = F.interpolate(m, (self.canon, self.canon), mode="bilinear",
                          align_corners=False)[0, 0]
        self._acc = m if self._acc is None else self._acc + m
        self._n += 1

    def finalize(self):
        if self._acc is None or not self._n:
            return
        m = self._acc / self._n
        if self.norm == "rel":
            r = m / (m.mean() + 1e-8)
            self.map = ((r - self.k) / max(self.width, 1e-6) + 0.5).clamp(0, 1)
        else:
            self.map = (m - m.min()) / (m.max() - m.min() + 1e-8)
        self._acc, self._n = None, 0

    def stats(self):
        """cov / 低权重区 / 过渡带。过渡带太宽就是 v1 那个病。"""
        if self.map is None:
            return None
        m = self.map
        return (float((m > 0.5).float().mean()),
                float((m < 0.3).float().mean()),
                float(((m >= 0.3) & (m <= 0.7)).float().mean()))

    def weights(self, hw, device, dtype):
        """(1, HW, 1) 的混合权重：1 = 用完整 prompt，0 = 用去主体版本。"""
        if self.map is None or self.s <= 0:
            return None
        h = w = int(hw ** 0.5)
        if h * w != hw:
            return None
        m = F.interpolate(self.map[None, None], (h, w), mode="bilinear",
                          align_corners=False)[0, 0].reshape(1, hw, 1)
        # s=1 时完全按图切换；s<1 时向完整 prompt 靠回去，便于扫强度
        return (1.0 - self.s * (1.0 - m)).to(device=device, dtype=dtype)


class BlendCrossAttn:
    def __init__(self, gate):
        self.gate = gate

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kw):
        residual = hidden_states
        nd = hidden_states.ndim
        if nd == 4:
            b, c, hh, ww = hidden_states.shape
            hidden_states = hidden_states.view(b, c, hh * ww).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        q = attn.to_q(hidden_states)
        ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            ehs = attn.norm_encoder_hidden_states(ehs)
        k, v = attn.to_k(ehs), attn.to_v(ehs)

        B, HW, _ = q.shape
        hd = k.shape[-1] // attn.heads

        def heads(x, n=attn.heads):
            return x.view(B, -1, n, hd).transpose(1, 2)

        qh, kh, vh = heads(q), heads(k), heads(v)
        is_cross = encoder_hidden_states is not None and kh.shape[2] == 77

        # 输出【始终】走 SDPA。录主体图要显式 softmax，但那份 probs 只用来看，
        # 不参与前向 —— 否则 fp16 下两条路径的数值差会改掉基图，第一步的微小
        # 扰动一路放大，A/B 的两边就不是同一张基图了（实测 19/30 行基图计数变了）。
        out = F.scaled_dot_product_attention(qh, kh, vh)
        # phase 'r' = 放大阶段的重录窗口（v1.1，gate_refresh.RefreshGate）：
        # 与基础阶段一样只录不混，拿放大阶段更细的注意力刷新门控图
        if is_cross and self.gate.phase in (1, "r"):
            with torch.no_grad():
                probs = (qh @ kh.transpose(-1, -2) * (hd ** -0.5)).softmax(-1)
            self.gate.record(probs, HW)
        else:
            wmap = self.gate.weights(HW, q.device, q.dtype) if is_cross else None
            if wmap is not None and self.gate.alt is not None:
                alt = self.gate.alt.to(q.dtype)
                ka, va = heads(attn.to_k(alt)), heads(attn.to_v(alt))
                out_alt = F.scaled_dot_product_attention(qh, ka, va)
                # 回到 (B, HW, C) 再按位置混合，权重是逐位置的
                out = out.transpose(1, 2).reshape(B, HW, -1)
                out_alt = out_alt.transpose(1, 2).reshape(B, HW, -1)
                # 无条件分支两套嵌入都是同一个 negative prompt，混合自动是恒等
                out = (wmap * out + (1 - wmap) * out_alt)
                out = out.reshape(B, HW, attn.heads, hd).transpose(1, 2)

        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if nd == 4:
            out = out.transpose(-1, -2).reshape(b, c, hh, ww)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, default=1.0, help="混合强度；0 = 原版")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--out", default=os.environ.get("SD_OUT", "./scalediff_out"))
    a = ap.parse_args()

    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots").glob("*"):
        if not (s_ / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s_ / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()

    tok_ids = subject_token_ids(pipe, PROMPT, SUBJECT)
    print(f'主体词 "{SUBJECT}" token 位置 {tok_ids}')
    gate = BlendGate(tok_ids, strength=a.s)

    # 去主体 prompt 编码一次。顺序与 pipeline 的 CFG 拼接一致：[neg; pos]
    pe, npe, _, _ = pipe.encode_prompt(
        prompt=PROMPT_NOSUBJ, device="cuda", num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=NEG)
    gate.alt = torch.cat([npe, pe])

    procs = dict(pipe.unet.attn_processors)
    n = sum(1 for k_ in procs if k_.endswith("attn2.processor"))
    for k_ in procs:
        if k_.endswith("attn2.processor"):
            procs[k_] = BlendCrossAttn(gate)
    pipe.unet.set_attn_processor(procs)
    print(f"换掉 {n} 个 attn2 处理器,  strength={a.s}")

    orig = pipe.noise_pred_step

    def patched(latents, t, *args, **kw):
        ph = 1 if latents.shape[-1] <= 128 else 2
        if ph == 2 and gate.phase == 1:
            gate.finalize()
            cov = float((gate.map > 0.5).float().mean())
            print(f"主体图定稿，高响应区占 {cov:.1%}（其余区域将改用去主体嵌入）")
        gate.phase = ph
        return orig(latents, t, *args, **kw)

    pipe.noise_pred_step = patched

    out = Path(a.out) / "method_v1"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    imgs = pipe(PROMPT, negative_prompt=NEG, height=1024, width=1024,
                generator=torch.Generator(device="cuda").manual_seed(a.seed),
                num_inference_steps=50, guidance_scale=7.5,
                restart_ratio=0.4, scale_factor=0.125, upsample_stage=a.stage)
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    for im in imgs:
        im.save(out / f"s{a.s:g}_seed{a.seed}_{im.width}.png")
    print(f"\nstrength={a.s:g}   {dt:.1f}s   峰值 {peak:.1f} GB")
    print("对照 (v0 lam=0，同代码路径): 76.1s / 11.9 GB / 幻影 5")
    print("判据: 幻影 <=3   时间 <=79.9s   显存 <=12.5 GB   且主角还在、细节不糊")
    print(f"\n图在 {out}")


if __name__ == "__main__":
    main()
