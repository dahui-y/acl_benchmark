#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绑定检测器：模型自己的交叉注意力，能不能读出「这次绑对了没有」。

    起因是 §11.12 第 00 行的一个 n=1 观察：
      prompt "a red apple and a green backpack"
      base  = 红书包 + 绿苹果   ← 绑错
      置换两个形容词的注意力图后 = 绿书包 + 红苹果   ← 绑对
    也就是说 **模型算出的两张图是对的，只是分配给了错的形容词 token。**

    如果这句话成立，就有一个**不需要图像真值**的检测器：
      · 句法说 red→apple（prompt 里就有）
      · 注意力说 "red" 的图与名词 "backpack" 的图更像
      · 两者不符 = 绑错
    句法提供了外部真值 —— 这正好绕开 §11.12 判死 C2 的那个死结
    （翻转型负分支的符号依赖不可知的真值）。

    **这个脚本只回答一件事：这个检测器准不准。** 不做方法、不做修正、不做表。

────────────────────────────────────────────────────────────────────────
判准：拿我自己看图的结论当参照，并且公开声明
────────────────────────────────────────────────────────────────────────

  项目没有标注池（硬约束）。这里用**作者自评**，按规矩必须写明：
  下面 HUMAN_READ 是我 2026-08-16 看 sheet.png 逐行读出来的，
  **只标注一眼能定的那几行，含糊的一律记 None 并排除在准确率之外。**

  最低门槛（达不到就不是检测器）：
    · 第 00 行必须判成 crossed（图上明确是红书包+绿苹果）
    · 第 01 / 05 / 06 行必须判成 ok（图上明确绑对）

  ⚠️ 8 条 prompt、其中可判的只有 4–5 条。**这个 n 只够否定，不够肯定。**
     检测器不准 → n=1 是巧合，这条断。
     检测器准   → 只说明「值得扩样本」，不说明它成立。

────────────────────────────────────────────────────────────────────────
怎么读注意力
────────────────────────────────────────────────────────────────────────

  · 只取 **cond 半边**（CFG 下 batch = [uncond, cond]），uncond 是空 prompt。
  · 只取 32×32 那一档（q_len=1024）。这是跨 Prompt-to-Prompt / A&E 一系
    公认承载语义的那一档；4096 那档太细、偏纹理。
    —— 但**不写死**：脚本先打印实际出现了哪些 q_len，取不到就报错退出。
  · 头上平均、层上平均、**时间步上只取前 half**（绑定在早期定下来）。
  · 多 token 的词（BPE 可能把 backpack 切开）：span 内平均，span 由
    子词拼接还原后精确匹配，匹配不上就退出而不是猜。
  · 相似度用余弦（每张图先归一化到和为 1，再展平）。

用法：
    source scalediff_probe/env.sh
    python bind_probe/detect.py --run        # ~2 min，8 prompt
    python bind_probe/detect.py --run --qlen 4096   # 换一档分辨率复核
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from negbranch import PAIRS, make_prompt, load, OUT as NB_OUT   # noqa: E402

OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "bind" / "detect"

# 用哪一档分辨率的交叉注意力。32×32 → q_len 1024。
Q_LEN = 1024
# 时间步只用前 EARLY 比例（绑定在早期定）
EARLY = 0.5

# ── 作者自评（2026-08-16，我看 sheet.png 读的）─────────────────────
#   ok      = 图上两个属性都贴对了
#   crossed = 图上两个属性对调了
#   None    = 我看不确定 → **排除在准确率之外，不算对也不算错**
# 这是自评不是标注池，按项目规矩公开写在这里。
HUMAN_READ = {
    0: "crossed",   # 红书包 + 绿苹果，prompt 要的是红苹果 + 绿书包。最明确的一行
    1: "ok",        # 蓝自行车 + 黄伞
    2: None,        # 紫茶壶对，但多出一个橙杯子，橙书也在 —— 判不清
    3: None,        # 白猫坐白椅、黑椅在后面 —— 场景里有两把椅子，判不清
    4: "ok",        # 绿瓶 + 红帽
    5: "ok",        # 黄香蕉 + 蓝杯
    6: "ok",        # 黑相机 + 白毛巾
    7: None,        # 橙围巾和紫围巾都在，判不清
}


# ─────────────────────────────────────────────────────────────────────
# token span 定位：多子词的词也要能对上
# ─────────────────────────────────────────────────────────────────────

def word_span(tokenizer, prompt, word):
    """返回 word 在 77-token 序列里的下标列表。

    CLIP BPE 可能把一个词切成多个子词（backpack 有可能是 back+pack</w>）。
    这里扫描连续 span，把子词去掉 '</w>' 拼起来精确匹配 word。
    **匹配不上就退出，不猜。**
    """
    ids = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True).input_ids
    toks = tokenizer.convert_ids_to_tokens(ids)
    for i in range(len(toks)):
        acc = ""
        for j in range(i, min(i + 5, len(toks))):
            acc += toks[j].replace("</w>", "")
            if acc == word:
                return list(range(i, j + 1))
            if not word.startswith(acc):
                break
    sys.exit(f"!! 在 '{prompt}' 里定位不到词 '{word}'（tokens={toks[:14]}…）")


# ─────────────────────────────────────────────────────────────────────
# 记录用的 cross-attention processor
# ─────────────────────────────────────────────────────────────────────

class RecordCrossAttn:
    """只记录，不改动任何东西。累加到 st['acc'][token] 上，不存全量张量。

    存全量会爆：70 层 × 30 步 × 77 token × 1024 空间。
    我们只关心 4 个词的 span，所以在 hook 里当场取出来累加。
    """

    def __init__(self, st):
        self.st = st

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        import torch
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        nd = hidden_states.ndim
        if nd == 4:
            b_, c_, h_, w_ = hidden_states.shape
            hidden_states = hidden_states.view(b_, c_, h_ * w_).transpose(1, 2)
        b = (encoder_hidden_states if is_cross else hidden_states).shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        q = attn.to_q(hidden_states)
        ctx = hidden_states if not is_cross else (
            attn.norm_encoder_hidden_states(encoder_hidden_states)
            if attn.norm_cross else encoder_hidden_states)
        k, v = attn.to_k(ctx), attn.to_v(ctx)
        q, k, v = (attn.head_to_batch_dim(t) for t in (q, k, v))
        probs = attn.get_attention_scores(q, k, attention_mask)

        if is_cross:
            qlen = probs.shape[1]
            self.st["seen_qlen"][qlen] = self.st["seen_qlen"].get(qlen, 0) + 1
            if qlen == self.st["q_len"] and self.st["record"]:
                heads = probs.shape[0] // b
                # 只取 cond 半边。CFG 下 batch=[uncond, cond]。
                sel = slice((b // 2) * heads, b * heads) if b % 2 == 0 \
                    else slice(0, probs.shape[0])
                m = probs[sel].float().mean(0)              # 头上平均 → [q_len, 77]
                self.st["acc"] = m.cpu().numpy() if self.st["acc"] is None \
                    else self.st["acc"] + m.cpu().numpy()
                self.st["n_acc"] += 1

        hidden_states = attn.batch_to_head_dim(torch.bmm(probs, v))
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if nd == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b_, c_, h_, w_)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_recorder(pipe):
    """同 negbranch.install 的教训：不给任何人传 dict（set_attn_processor 会
    就地 pop 掏空），自己遍历 named_modules 设 .processor，0 条硬退出。"""
    st = {"acc": None, "n_acc": 0, "q_len": Q_LEN, "record": False,
          "seen_qlen": {}}
    n = 0
    for name, mod in pipe.unet.named_modules():
        if hasattr(mod, "processor") and name.endswith("attn2"):
            mod.processor = RecordCrossAttn(st); n += 1
    if n == 0:
        sys.exit("!! 一个 cross-attn 都没挂上")
    print(f"  挂上 {n} 个记录型 cross-attn processor")
    return st


def cos_maps(a, b):
    """两张空间图的余弦相似度。各自归一化到和为 1，再展平做余弦。"""
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cmd_run(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = load()
    st = install_recorder(pipe)
    tk = pipe.tokenizer

    rows, n_ok, n_judged = [], 0, 0
    for k, (a1, n1, a2, n2) in enumerate(PAIRS):
        p = make_prompt(a1, n1, a2, n2)
        spans = {w: word_span(tk, p, w) for w in (a1, n1, a2, n2)}

        st["acc"], st["n_acc"], st["record"] = None, 0, True
        # 只在前 EARLY 比例的步上记录：绑定在早期定下来。
        cutoff = max(1, int(args.steps * EARLY))
        state = {"i": 0}

        def cb(pipe_, i, t, cbk):
            state["i"] = i + 1
            st["record"] = state["i"] < cutoff
            return cbk

        g = torch.Generator("cuda").manual_seed(3000 + k)   # 与 negbranch 同 seed
        pipe(prompt=p, generator=g, num_inference_steps=args.steps,
             guidance_scale=7.5, height=1024, width=1024,
             callback_on_step_end=cb, output_type="latent")

        if st["acc"] is None:
            sys.exit(f"!! 一张图都没记到。实际出现的 q_len: {st['seen_qlen']}\n"
                     f"   Q_LEN={Q_LEN} 取不到 —— 改 --qlen 或看这份分布")
        A = st["acc"] / st["n_acc"]                          # [q_len, 77]

        def m(w):        # 词的空间图：span 内平均
            return A[:, spans[w]].mean(-1)

        # 形容词 a 更像哪个名词
        c11, c12 = cos_maps(m(a1), m(n1)), cos_maps(m(a1), m(n2))
        c21, c22 = cos_maps(m(a2), m(n1)), cos_maps(m(a2), m(n2))
        # 句法真值：a1→n1, a2→n2
        pred = "ok" if (c11 > c12 and c22 > c21) else \
               ("crossed" if (c12 > c11 and c21 > c22) else "mixed")
        human = HUMAN_READ.get(k)
        agree = None if human is None else (pred == human)
        if agree is not None:
            n_judged += 1; n_ok += int(agree)
        rows.append(dict(k=k, prompt=p, c11=c11, c12=c12, c21=c21, c22=c22,
                         pred=pred, human=human, agree=agree,
                         margin1=c11 - c12, margin2=c22 - c21))
        print(f"[{k}] {p}")
        print(f"    cos({a1},{n1})={c11:.4f}  cos({a1},{n2})={c12:.4f}   "
              f"Δ={c11-c12:+.4f}")
        print(f"    cos({a2},{n2})={c22:.4f}  cos({a2},{n1})={c21:.4f}   "
              f"Δ={c22-c21:+.4f}")
        print(f"    检测器判 {pred:8s} | 我看图读 {str(human):8s} | "
              f"{'✓' if agree else ('✗' if agree is False else '—(排除)')}",
              flush=True)

    print("\n" + "═" * 68)
    print(f"可判 {n_judged} 行（其余 {len(PAIRS)-n_judged} 行我看图判不清，已排除）")
    print(f"一致 {n_ok}/{n_judged}")
    r0 = next(r for r in rows if r["k"] == 0)
    gate0 = r0["pred"] == "crossed"
    gates = [r for r in rows if r["k"] in (1, 5, 6)]
    gate1 = all(r["pred"] == "ok" for r in gates)
    print(f"\n最低门槛（写在看数之前）：")
    print(f"  [{'ok' if gate0 else '!!'}] 第 00 行判成 crossed  → 实判 {r0['pred']}")
    print(f"  [{'ok' if gate1 else '!!'}] 01/05/06 判成 ok      → 实判 "
          f"{[r['pred'] for r in gates]}")
    if gate0 and gate1:
        print(f"\n  → 检测器过了最低门槛。**这只说明值得扩样本，不说明它成立**"
              f"（n={n_judged}）。\n"
              f"    下一步是第 5 步：查 SynGen / A-STAR / Divide-and-Bind 一系"
              f"有没有做过同一件事。")
    else:
        print(f"\n  → 没过门槛。n=1 那个观察是巧合，这条断。")
    (OUT / "detect.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT/'detect.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--qlen", type=int, default=1024,
                    help="用哪一档交叉注意力（32×32 → 1024）")
    ap.add_argument("--run", action="store_true", required=True)
    a = ap.parse_args()
    globals()["Q_LEN"] = a.qlen        # 不用 global 语句：它必须先于该名字的读取
    cmd_run(a)


if __name__ == "__main__":
    main()
