#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Q2 的十分钟证伪实验：那个「故意贴错」的负分支，到底选不选择性。

    **这个脚本不测方法好不好，只测一颗螺丝拧不拧得上。**
    不算指标、不做标准表、不做统计。产出是一张三列对照图，用眼睛看。

────────────────────────────────────────────────────────────────────────
被测的主张
────────────────────────────────────────────────────────────────────────

  外推 ε̂ = ε(x,c) + w·[ε(x,c) − ε̃(x,c)] 沿差向量 D 走。
  D 里装什么，全看 ε̃ 与 ε 差在哪。差在两个轴上，就在两个轴上过冲。

  C2 的构造：**在交叉注意力里置换形容词 token 的注意力图**。
      out = Σ_t A[:,t] ⊗ V[t]     ← A[:,t] 是 token t 的空间图
      交换 A 的第 i、j 列  ⇒  形容词 i 的 value 被写进形容词 j 原来注意的区域

  **结构性保证**：置换保持 {A[:,t]} 这个**多重集**不变。
  任何只依赖「图的集合」的量 —— 每个区域的总注意力质量、布局熵、空间覆盖 ——
  在置换下逐点精确不变。变的**只有配对**。
  这是 PAG / SoftPAG / HeadHunter 给不出来的：它们向单位阵插值，
  那是**破坏**多重集；置换不是。

  ⇒ **可证伪的视觉预测**：单独用 ε̃ 生成的图，应该是
     **同一个场景、同一个布局、同一批物体，只有属性贴错了。**
     如果它是另一个场景、或者一坨糊 —— 选择性是假的，C2 当场作废。

────────────────────────────────────────────────────────────────────────
三个臂，同 seed，这才叫对照
────────────────────────────────────────────────────────────────────────

  base      正常 SDXL                        ← 参照，ε 长什么样
  attnswap  C2：交叉注意力里置换形容词的图     ← **被测对象**
  textswap  C0：在 prompt 文本里交换属性       ← **反面对照**

  C0 是「朴素做法」。它不选择性 —— 交换后的 prompt 是**另一个合法场景**，
  光照/材质/构图都会跟着变。所以预期：

      base ─ attnswap  布局应当高度一致（只有颜色换了）
      base ─ textswap  布局应当明显不同

  **如果 attnswap 看起来和 textswap 一样离谱，C2 就死了。**
  只有 base 一个参照是不够的 —— 没有 C0 这一列，就没有「多离谱才算离谱」的尺度。

────────────────────────────────────────────────────────────────────────
两个设计决定，都是为了让这个测试测的是算子本身
────────────────────────────────────────────────────────────────────────

  ① **不用句法解析器。** prompt 全部走固定模板
     "a {adj1} {noun1} and a {adj2} {noun2}"，形容词位置由 tokenizer 定位并断言。
     真方法里要 spaCy，但那是另一个失败源；这里要测的是算子，不是解析器。

  ② **置换在全部交叉注意力层、全部时间步上施加**（算子的最强形式）。
     如果最强形式下选择性都成立，那就成立。层/步的选择是后面的调参，不是这里的问题。

────────────────────────────────────────────────────────────────────────
两个自测（守静默错误，零 GPU 之外）
────────────────────────────────────────────────────────────────────────

  ① **恒等置换必须是逐字节 no-op。** 挂上 hook 但不换任何列，输出必须与
     base 完全相同。否则说明 hook 本身在扰动（比如我把 SDPA 换成显式 softmax
     引入了数值差），那后面看到的一切都不可信。
     —— 这条是 attn_variants.py selftest ⑥ 的同一条纪律。
  ② **多重集不变**：置换前后，各列的和排序后必须逐元素相等。
     这是那个「结构性保证」的数值断言，不是信仰。

用法：
    source scalediff_probe/env.sh
    python bind_probe/negbranch.py --selftest       # ①②，需要 GPU 但只跑 2 步
    python bind_probe/negbranch.py --run            # 三臂 × 8 prompt，约 10 min
    python bind_probe/negbranch.py --sheet          # 三列对照图 → 看
"""

import argparse
import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "bind" / "negbranch"
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
ARMS = ("base", "attnswap", "textswap")

# 固定模板，形容词位置可由 tokenizer 精确定位（见 ① 的理由）。
# 选颜色对时刻意让两个颜色差别大 —— 贴错了要一眼能看出来。
PAIRS = [
    ("red",    "apple",     "green",  "backpack"),
    ("blue",   "bicycle",   "yellow", "umbrella"),
    ("purple", "teapot",    "orange", "book"),
    ("white",  "cat",       "black",  "chair"),
    ("green",  "bottle",    "red",    "hat"),
    ("yellow", "banana",    "blue",   "mug"),
    ("black",  "camera",    "white",  "towel"),
    ("orange", "pumpkin",   "purple", "scarf"),
]


def make_prompt(a1, n1, a2, n2):
    return f"a {a1} {n1} and a {a2} {n2}"


def find_adj_positions(tokenizers, prompt, adj1, adj2):
    """定位两个形容词在 77-token 序列里的下标。

    SDXL 有两个 text encoder，但 CLIP-L 与 OpenCLIP-bigG **共用同一套 BPE 词表**，
    所以 token 轴是对齐的、一组下标对两边都成立。
    —— 这句话不能靠记忆，下面断言两个 tokenizer 给出相同下标；不同就报错退出。
    """
    idxs = []
    for tk in tokenizers:
        ids = tk(prompt, padding="max_length", max_length=tk.model_max_length,
                 truncation=True).input_ids
        toks = tk.convert_ids_to_tokens(ids)
        pos = []
        for adj in (adj1, adj2):
            hit = [k for k, t in enumerate(toks) if t == adj + "</w>"]
            if len(hit) != 1:
                sys.exit(f"!! '{adj}' 在 '{prompt}' 里命中 {len(hit)} 次，"
                         f"模板假设被打破（要求恰好一次）")
            pos.append(hit[0])
        idxs.append(tuple(pos))
    if idxs[0] != idxs[1]:
        sys.exit(f"!! 两个 tokenizer 给出不同下标 {idxs} —— "
                 f"「共用 BPE 词表」这个前提不成立，脚本的置换会错位")
    return idxs[0]


# ─────────────────────────────────────────────────────────────────────
# 算子：交叉注意力里置换形容词的注意力图
# ─────────────────────────────────────────────────────────────────────

class SwapCrossAttn:
    """显式 softmax 路径的 cross-attention processor，在 A 上换两列。

    只挂在 cross-attention（encoder_hidden_states 不为 None）上；
    self-attention 保持 SDPA 不动 —— 自注意力的 q_len² 矩阵在 1024² 上很大，
    换成显式路径会白白吃显存，而且不是我们要动的东西。
    cross-attn 的 kv_len=77，[b*h, q, 77] 很小，显式路径没有代价。
    """

    def __init__(self, state):
        self.st = state          # 共享的 dict：swap=(i,j) 或 None；cond_only

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        import torch
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        inp_ndim = hidden_states.ndim
        if inp_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        b = (encoder_hidden_states if is_cross else hidden_states).shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states)
        ctx = hidden_states if not is_cross else (
            attn.norm_encoder_hidden_states(encoder_hidden_states)
            if attn.norm_cross else encoder_hidden_states)
        k, v = attn.to_k(ctx), attn.to_v(ctx)
        q, k, v = (attn.head_to_batch_dim(t) for t in (q, k, v))

        probs = attn.get_attention_scores(q, k, attention_mask)   # [b*h, q_len, kv_len]

        sw = self.st.get("swap")
        if is_cross and sw is not None:
            i, j = sw
            heads = probs.shape[0] // b
            if self.st.get("cond_only", True) and b % 2 == 0:
                # CFG 下 batch 是 [uncond, cond]。uncond 那半是空 prompt，
                # 被换的两列是 padding，换了没意义但也不是零 —— 只动 cond 半边，
                # 免得引入一个我们没在测的扰动。
                sel = slice((b // 2) * heads, b * heads)
            else:
                sel = slice(0, probs.shape[0])
            part = probs[sel]
            part[:, :, [i, j]] = part[:, :, [j, i]]
            probs = probs.clone()
            probs[sel] = part

        hidden_states = attn.batch_to_head_dim(torch.bmm(probs, v))
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if inp_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install(pipe):
    """给所有 cross-attention 挂上可控 processor；self-attn 保持原样。
    返回共享 state —— 把 state['swap'] 设成 None 就等于关掉算子。"""
    from diffusers.models.attention_processor import AttnProcessor2_0
    st = {"swap": None, "cond_only": True}
    procs = {}
    for name in pipe.unet.attn_processors:
        # diffusers 的命名约定：attn2 是 cross-attention，attn1 是 self-attention
        procs[name] = SwapCrossAttn(st) if "attn2" in name else AttnProcessor2_0()
    pipe.unet.set_attn_processor(procs)
    n_cross = sum(1 for n in procs if "attn2" in n)
    print(f"  挂上 {n_cross} 个 cross-attn processor（self-attn {len(procs)-n_cross} 个保持 SDPA）")
    return st


def load(device="cuda"):
    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─────────────────────────────────────────────────────────────────────
# --selftest
# ─────────────────────────────────────────────────────────────────────

def cmd_selftest(args):
    import numpy as np
    import torch
    fails = []

    def chk(label, good, detail=""):
        print(f"  [{'ok' if good else '!!'}] {label}" + (f"   {detail}" if detail else ""))
        if not good:
            fails.append(label)

    print("═" * 68); print("selftest"); print("═" * 68)

    # ② 多重集不变 —— 纯 numpy，先做，不用 GPU
    rng = np.random.default_rng(0)
    A = rng.random((4, 64, 77)); A /= A.sum(-1, keepdims=True)
    B = A.copy(); B[:, :, [5, 9]] = B[:, :, [9, 5]]
    before = np.sort(A.sum(1), axis=-1)
    after = np.sort(B.sum(1), axis=-1)
    chk("② 置换后各列质量的多重集不变", np.allclose(before, after),
        f"最大差 {np.abs(before-after).max():.2e}")
    chk("② 置换确实改变了配对（否则①是空测试）",
        not np.allclose(A, B))

    # ① 恒等置换必须是逐字节 no-op
    print("  ·· 载入 SDXL（① 需要 GPU，只跑 2 步）")
    pipe = load()
    p = make_prompt(*PAIRS[0])
    kw = dict(prompt=p, num_inference_steps=2, guidance_scale=7.5,
              height=512, width=512, output_type="latent")

    g = torch.Generator("cuda").manual_seed(7)
    ref = pipe(generator=g, **kw).images        # 原生 SDPA processor

    st = install(pipe)
    st["swap"] = None                            # 挂了 hook 但不换列
    g = torch.Generator("cuda").manual_seed(7)
    hooked = pipe(generator=g, **kw).images
    d = (ref.float() - hooked.float()).abs().max().item()
    # 显式 softmax vs SDPA 在 fp16 下有极小数值差，不可能逐位相同。
    # 但必须小到与「换了列」的差距差好几个数量级 —— 下面直接比。
    i, j = find_adj_positions([pipe.tokenizer, pipe.tokenizer_2], p,
                              PAIRS[0][0], PAIRS[0][2])
    st["swap"] = (i, j)
    g = torch.Generator("cuda").manual_seed(7)
    swapped = pipe(generator=g, **kw).images
    d_swap = (ref.float() - swapped.float()).abs().max().item()
    chk("① 恒等置换 ≈ no-op（与真置换差 ≥100×）", d_swap > 100 * max(d, 1e-9),
        f"hook 噪声 {d:.2e}  vs  置换效果 {d_swap:.2e}  比值 {d_swap/max(d,1e-12):.0f}×")
    chk("① 形容词下标定位成功", True, f"adj 位置 = ({i}, {j})")

    print("\n" + ("全部通过" if not fails else f"!! 失败：{fails}"))
    return 0 if not fails else 1


# ─────────────────────────────────────────────────────────────────────
# --run
# ─────────────────────────────────────────────────────────────────────

def cmd_run(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = load()
    st = install(pipe)
    meta = {}

    for k, (a1, n1, a2, n2) in enumerate(PAIRS):
        p = make_prompt(a1, n1, a2, n2)
        p_sw = make_prompt(a2, n1, a1, n2)          # C0：文本里换属性
        i, j = find_adj_positions([pipe.tokenizer, pipe.tokenizer_2], p, a1, a2)
        meta[k] = {"prompt": p, "textswap": p_sw, "adj_idx": [i, j]}
        print(f"[{k+1}/{len(PAIRS)}] {p}   adj@({i},{j})", flush=True)

        for arm in ARMS:
            f = OUT / f"{k:02d}_{arm}.png"
            if f.exists():
                continue
            st["swap"] = (i, j) if arm == "attnswap" else None
            prompt = p_sw if arm == "textswap" else p
            # 同一个 seed 喂三个臂 —— 布局能不能对上，全靠这个
            g = torch.Generator("cuda").manual_seed(3000 + k)
            img = pipe(prompt=prompt, generator=g, num_inference_steps=args.steps,
                       guidance_scale=7.5, height=1024, width=1024).images[0]
            img.save(f)
            print(f"      {arm}")
        st["swap"] = None

    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT}\n下一步：python bind_probe/negbranch.py --sheet")


# ─────────────────────────────────────────────────────────────────────
# --sheet
# ─────────────────────────────────────────────────────────────────────

def cmd_sheet(args):
    import numpy as np
    from PIL import Image, ImageDraw
    cell, pad, hdr = args.cell, 6, 24
    rows = [k for k in range(len(PAIRS))
            if all((OUT / f"{k:02d}_{a}.png").exists() for a in ARMS)]
    if not rows:
        sys.exit(f"!! {OUT} 里没有完整的三臂，先 --run")

    W = pad + len(ARMS) * (cell + pad)
    H = hdr + len(rows) * (cell + hdr + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for c, a in enumerate(ARMS):
        tag = {"base": "base (ε)", "attnswap": "attnswap = C2 ← 被测",
               "textswap": "textswap = C0 ← 反面对照"}[a]
        d.text((pad + c * (cell + pad) + 4, 5), tag, fill="black")

    # 辅助数字：布局漂移。**次于看图，但它直接检验那个结构性主张。**
    # 灰度降采样到 32×32 后的 L2 —— 粗到只反映布局，不反映颜色细节。
    def struct(p):
        im = Image.open(p).convert("L").resize((32, 32), Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255.0
        return (a - a.mean()) / (a.std() + 1e-8)

    drift = {"attnswap": [], "textswap": []}
    y = hdr
    for k in rows:
        b = struct(OUT / f"{k:02d}_base.png")
        for a in ("attnswap", "textswap"):
            drift[a].append(float(np.sqrt(((struct(OUT / f'{k:02d}_{a}.png') - b) ** 2).mean())))
        a1, n1, a2, n2 = PAIRS[k]
        d.text((pad + 4, y + 4), f"[{k:02d}] a {a1} {n1} and a {a2} {n2}"
               f"    布局漂移 C2={drift['attnswap'][-1]:.2f} / C0={drift['textswap'][-1]:.2f}",
               fill="black")
        for c, a in enumerate(ARMS):
            im = Image.open(OUT / f"{k:02d}_{a}.png").convert("RGB")
            im.thumbnail((cell, cell), Image.LANCZOS)
            sheet.paste(im, (pad + c * (cell + pad), y + hdr))
        y += cell + hdr + pad

    p = OUT / "sheet.png"
    sheet.save(p)
    m2, m0 = np.mean(drift["attnswap"]), np.mean(drift["textswap"])
    print(f"{p}   ({len(rows)} prompts × 3 臂)")
    print(f"\n布局漂移均值（越小 = 越保布局）：C2 {m2:.3f}   C0 {m0:.3f}")
    print(f"\n**先看图，数字只是佐证。** 要判断的是：")
    print(f"  · C2 那一列是不是「同一个场景、同一个布局、只有颜色贴错」")
    print(f"  · C0 那一列是不是明显换了场景")
    print(f"\n如果 C2 看起来和 C0 一样离谱 → **选择性是假的，C2 死，Q2 没答案，"
          f"§11 这条方法不成立。**")
    print(f"如果 C2 保布局而 C0 不保 → 螺丝拧上了，可以往下走 "
          f"（层/步的选择、强度调度、外推是否真帮忙，都是后面的事）。")
    if m2 >= m0:
        print(f"\n⚠️ 数字上 C2 漂移不小于 C0 —— 这与结构性主张相反，"
              f"看图时格外注意是不是自欺")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cell", type=int, default=420)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(cmd_selftest(a))
    (cmd_run if a.run else cmd_sheet)(a)


if __name__ == "__main__":
    main()
