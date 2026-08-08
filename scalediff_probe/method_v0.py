"""干预 v0：把 prompt 的主体词关在基图已经放置它的地方。

诊断（两个受控消融）：
  · 放大阶段去掉主体词  -> 幻影 5 -> 0        文本条件是必要条件
  · KV 邻域 x2         -> 幻影 5 -> 2~3      局部性是另一个必要条件
  · 2048²(邻域占 1/4) vs 4096²(1/16) -> 0 vs 5

两个条件都必要，所以打断任一即可。但两条现成的路都被堵死：
  · 删主体词 -> 真实场景用不了，而且 AccDiffusion v2 已指出
      "the absence of a prompt undermines image details"
  · 放大邻域 -> 实测 +37% 时间 / +41% 显存，把 ScaleDiff 的核心卖点吃回去

已知修法为什么搬不过来：AccDiffusion 给每个 patch 换 prompt 字符串，需要
【单独的 patch 前向】；ScaleDiff 只有一次全图前向，没有地方喂第二句话。

但一次前向恰恰给了别的东西：cross-attention 的 logits 是一张
[位置 x 77 token] 的完整表，可以逐位置逐 token 加偏置 —— 在
scaled_dot_product_attention 里就是 attn_mask 参数，几乎不加计算。

于是：
  **只压制主体词那几个 token 在【基图里没有主体的位置】上的注意力，
    场景词一个都不动。**

"基图里主体在哪"是免费的：Phase 1 生成 1024² 时模型自己的 cross-attention
就是那张图，本来就算出来了，只是被扔掉了。

    python scalediff_probe/method_v0.py --lam 4
    python scalediff_probe/method_v0.py --lam 0        # 等价于原版，自检用
"""

import argparse
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

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
PROMPT = ("a photograph of a lone hiker standing on a rocky ridge, vast forested valley "
          "and distant snow mountains behind, golden hour")
SUBJECT = "hiker"          # 要被关起来的词；场景词不动


def subject_token_ids(pipe, prompt, subject):
    """主体词在 77 长度序列里的位置。

    SDXL 两个文本编码器共用同一个 CLIP 分词器和同样的截断，所以位置在
    prompt_embeds 的拼接结果里是一致的。
    """
    tok = pipe.tokenizer
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids[0]
    sub = tok(subject, add_special_tokens=False).input_ids
    hit = []
    for i in range(len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)].tolist() == sub:
            hit += list(range(i, i + len(sub)))
    return hit


class SubjectGate:
    """记录 Phase 1 的主体注意力图，在 Phase 2 用它当门控。"""

    def __init__(self, token_ids, lam=4.0, canon=64, token_str=None):
        self.tok = token_ids
        self.lam = lam
        self.canon = canon
        self.token_str = token_str or {}
        self.phase = 1
        self._acc = None          # (canon, canon) 累积的主体注意力
        self._n = 0
        self.map = None           # 归一化到 [0,1] 的成品
        # 77 个 token 各自吸走多少注意力质量。v0 压了主体词却几乎没改变输出，
        # 第一个要排除的解释就是"主体 token 本来就没多少权重"。
        self._mass = None

    def record(self, probs, hw):
        """probs: (B, heads, HW, 77) —— 只取条件分支。"""
        if not self.tok:
            return
        h = w = int(hw ** 0.5)
        if h * w != hw:
            return
        mass = probs[-1].mean(dim=(0, 1)).float()              # (77,) head 和位置平均
        self._mass = mass if self._mass is None else self._mass + mass
        m = probs[-1, :, :, self.tok].sum(-1).mean(0)          # (HW,) 各 head 平均
        m = m.reshape(1, 1, h, w).float()
        m = F.interpolate(m, size=(self.canon, self.canon), mode="bilinear",
                          align_corners=False)[0, 0]
        self._acc = m if self._acc is None else self._acc + m
        self._n += 1

    def finalize(self):
        if self._acc is None or self._n == 0:
            self.map = None
            return
        m = self._acc / self._n
        # 逐图归一化：注意力的绝对尺度随层和步数变化，只有相对高低有意义
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        self.map = m

        if self._mass is not None:
            mass = (self._mass / self._n)
            mass = mass / mass.sum()
            share = float(mass[self.tok].sum())
            print(f"\n主体 token {self.tok} 吸走的注意力质量: {share:.2%}")
            top = torch.topk(mass, 10)
            print("注意力质量最大的 10 个 token:")
            for v, i in zip(top.values.tolist(), top.indices.tolist()):
                mark = "  <- 主体" if i in self.tok else ""
                print(f"    [{i:2d}] {self.token_str.get(i, '?'):<16} {v:6.2%}{mark}")
        self._acc, self._n, self._mass = None, 0, None

    def bias(self, hw, device, dtype):
        """返回 (HW, 77) 的加性偏置：主体词在低响应处被压 lam。"""
        if self.map is None or not self.tok or self.lam <= 0:
            return None
        h = w = int(hw ** 0.5)
        if h * w != hw:
            return None
        m = F.interpolate(self.map[None, None], size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0].reshape(-1)
        b = torch.zeros(hw, 77, device=device, dtype=torch.float32)
        b[:, self.tok] = -self.lam * (1.0 - m)[:, None]
        return b.to(dtype)


class GatedCrossAttn:
    """attn2 的处理器。Phase 1 记录，Phase 2 加偏置。"""

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
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)

        is_cross = encoder_hidden_states is not None and k.shape[2] == 77
        if is_cross and self.gate.phase == 1:
            # 记录需要显式 softmax。只在 1024² 的基阶段做，代价可以接受。
            probs = (q @ k.transpose(-1, -2) * (hd ** -0.5)).softmax(-1)
            self.gate.record(probs.detach(), HW)
            out = probs @ v
        else:
            mask = self.gate.bias(HW, q.device, q.dtype) if is_cross else None
            if mask is not None:
                # 只压条件分支；无条件分支里没有主体词，压了会破坏 CFG
                full = torch.zeros(B, 1, HW, 77, device=q.device, dtype=q.dtype)
                full[B // 2:] = mask
                mask = full
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if nd == 4:
            out = out.transpose(-1, -2).reshape(b, c, hh, ww)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install(pipe, gate):
    """只换 attn2。ScaleDiff 的 register_attention_control 在 __call__ 里会
    重建 processor 字典，但它对非 attn1 的项是原样读回当前值 —— 所以在调用
    pipe() 之前装好，我们的 attn2 处理器会被保留。"""
    procs = dict(pipe.unet.attn_processors)
    n = 0
    for name in procs:
        if name.endswith("attn2.processor"):
            procs[name] = GatedCrossAttn(gate)
            n += 1
    pipe.unet.set_attn_processor(procs)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=4.0, help="压制强度；0 = 原版")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--subject", default=SUBJECT)
    ap.add_argument("--out", default=os.environ.get("SD_OUT", "./scalediff_out"))
    a = ap.parse_args()

    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots").glob("*"):
        if not (s / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()

    tok_ids = subject_token_ids(pipe, PROMPT, a.subject)
    print(f'主体词 "{a.subject}" 在 token 位置 {tok_ids}')
    if not tok_ids:
        sys.exit("找不到主体词，检查 --subject 是否出现在 PROMPT 里")

    tk = pipe.tokenizer
    ids = tk(PROMPT, padding="max_length", max_length=tk.model_max_length,
             truncation=True, return_tensors="pt").input_ids[0].tolist()
    tstr = {i: tk.decode([t]).strip() for i, t in enumerate(ids)}
    gate = SubjectGate(tok_ids, lam=a.lam, token_str=tstr)
    print(f"换掉 {install(pipe, gate)} 个 attn2 处理器,  lam={a.lam}")

    orig = pipe.noise_pred_step

    def patched(latents, t, *args, **kw):
        # 基阶段 latent 是 128；大于它就是放大阶段
        newphase = 1 if latents.shape[-1] <= 128 else 2
        if newphase == 2 and gate.phase == 1:
            gate.finalize()
            cov = float((gate.map > 0.5).float().mean()) if gate.map is not None else -1
            print(f"主体注意力图已定稿，高响应区占画面 {cov:.1%}")
        gate.phase = newphase
        return orig(latents, t, *args, **kw)

    pipe.noise_pred_step = patched

    out = Path(a.out) / "method_v0"
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
        im.save(out / f"lam{a.lam:g}_s{a.seed}_{im.width}.png")

    print(f"\nlam={a.lam:g}   {dt:.1f}s   峰值 {peak:.1f} GB   -> {[i.width for i in imgs]}")
    print("baseline(原版): 74.8s / 11.9 GB / lone 类 4096² 幻影 5 个")
    print(f"判据: 幻影 <=3  且  时间/显存增幅 <5%  且  FID 不劣")
    print(f"\n图在 {out}")


if __name__ == "__main__":
    main()
