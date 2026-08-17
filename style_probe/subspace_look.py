#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StylePure 诊断：单图内跨区域对比，分出来的到底是不是 style？

    **只测这一件事。** 因为按 DESIGN.md 的公式对照，其余部件都不是新的：
      · 广义特征值形式        → DICE Eq.13 已发表（跨图三元组）
      · 参考图 PCA + 区域聚类 → StyleGallery 的 DFCC 已发表
      · K/V 注入              → StyleID (CVPR'24) 已发表
    唯一没被占的是**对比源**：DICE 用跨图，我们用**单图内的语义区域**。

    不算 ArtFID、不做标准表、不做统计。产出一张对照图，用眼睛看。

────────────────────────────────────────────────────────────────────────
三个臂（同 content、同 seed）
────────────────────────────────────────────────────────────────────────

  full   注入参考图全部 K/V（≈ StyleID）      预期：有风格 + 参考内容泄漏
  proj   只注入到 content-invariant 子空间     预期：有风格、无泄漏  ← 想要的
  comp   **只注入正交补**                      预期：有参考的物体、无风格

  `comp` 是关键证伪器。若它也带笔触 → 根本没分开。
  只有 full/proj 两列看不出这一点（§11.12 negbranch 的教训：必须有反面对照）。

────────────────────────────────────────────────────────────────────────
风格图怎么选：**故意分成"难"和"易"两组**
────────────────────────────────────────────────────────────────────────

  我对这个假设的主要技术反对是：
  > 「style = 跨区域共享」会把**随区域变化的渲染方式**当 content 丢掉，
  >   而那正是风格的重要成分（水墨的留白天空 vs 干笔皴擦的山石）。

  所以样本必须包含**笔触逐区域差异大**的（难组）**和逐区域均匀**的（易组）。
  只挑难的，失败了分不清是"假设错"还是"实现错"；
  只挑易的，等于回避了这个反对。

  **两组一起跑，才能读出精确的失败模式：**
  易组成、难组败 ⇒ 技术反对成立，假设在原理上有洞。

────────────────────────────────────────────────────────────────────────
两个刻意的简化（必须写明，否则后面自己会忘）
────────────────────────────────────────────────────────────────────────

  ① **不做 DDIM inversion，用前向加噪（closed-form）取参考特征。**
     StyleID 的正规做法是 inversion。这里不是在比数字，是在问
     "这个分解存不存在"，前向加噪足够且鲁棒得多。
     —— **所以本脚本的输出不可与任何发表数字比较。**
  ② **风格图从 StyleSSP 的 assets/ours.jpg 上裁。** 单张图不在仓库里，
     但那张拼图的第一行就是论文用的风格条（13 列 × 6 行，格子 292.6×295.0）。

用法：
    source scalediff_probe/env.sh
    python style_probe/subspace_look.py --crop       # 裁图，零 GPU
    python style_probe/subspace_look.py --selftest   # 代数自测
    python style_probe/subspace_look.py --run        # 三臂
    python style_probe/subspace_look.py --sheet      # 对照图 → 看
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "subspace"
TEASER = REPO / "help_code" / "StyleSSP" / "assets" / "ours.jpg"
# 用户已下载的真实 WikiArt 参考图（StyleSSP 协议就是 MS-COCO content × WikiArt style，
# 所以用它比从 teaser 上裁更接近后面要做的正表）。可用 WIKIART 环境变量覆盖。
WIKIART = Path(os.environ.get(
    "WIKIART", "/openbayes/input/input0/Sim2Struct-1000/temp/jdb/wikiart_ref"))
# backbone：换成 SD1.5。理由（2026-08-16，读了三篇原文 + StyleSSP 评测代码后）：
#   · StyleID / StyleGallery / DICE 全是 SD1.4/1.5 @ 512 —— 我们逐条对过公式的就是这三篇
#   · 官方 ArtFID 评测代码 eval_artfid.py:36 把一切 resize 到 512，
#     所以 1024 生成完还要降回去，纯浪费
#   · DICE 的层号（down_blocks[0] / up_blocks[3]）是 SD1.x 的，SDXL 上对不上
SD15 = os.environ.get("SD15_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5")
RES = 512
ARMS = ("full", "proj", "comp")

# ── ours.jpg 的网格：13 列 × 6 行，(0,0) 是 Style/Content 表头格 ──────
GRID_C, GRID_R = 13, 6

# 风格条（第 0 行，第 1..12 列）。**"难/易"是按逐区域笔触差异标的**，
# 这是本实验的核心设计，不是随手排的。
STYLES = {
    #  列  key                难易   为什么选它
    9:  ("ink_landscape",    "hard", "水墨：天空大片留白/淡墨 vs 山石干笔皴擦 —— 逐区域差异最大，最强测试"),
    12: ("starry_night",     "hard", "梵高：天空厚涂漩涡 vs 村庄短促笔触 —— 笔触方向与密度都随区域变"),
    5:  ("great_wave",       "hard", "浮世绘：天空平色块 vs 浪花细线 —— 平涂与线描并存"),
    6:  ("anime_landscape",  "hard", "赛璐璐：天空/车身平涂 vs 云与草丛细节"),
    1:  ("pencil_sketch",    "hard", "铅笔：主体密集排线 vs 背景大片空白"),
    4:  ("mona_lisa",        "mid",  "文艺复兴油画：脸部晕涂 vs 暗背景 —— 中等差异"),
    8:  ("geo_pattern",      "easy", "几何平面图形：**全图渲染方式一致** —— 对照组，假设该成立"),
    2:  ("line_ballerina",   "easy", "极简线稿：**全图一种线** —— 对照组"),
}
# 内容条（第 0 列，第 1..5 行）
CONTENTS = {1: "horse", 3: "portrait", 5: "bridge"}

# 用哪一档 self-attention。
#   512 输入 → latent 64×64；SD1.5 的 self-attn 分辨率是 64/32/16/8
#   → q_len ∈ {4096, 1024, 256, 64}
# **默认取 4096（64×64，最高档）**，依据是 DICE 的 style 子空间取 down_blocks[0]，
# 那正是这一档。笔触是高频的，这个先验比我原来猜的 1024 硬。
# 注入位置仍照 StyleID 的惯例放在 decoder(up_blocks)；--where 可换。
Q_LEN = 4096
RANK = 16           # 子空间维数 r，--rank 可扫
EPS = 1e-4          # 广义特征问题的正则


# ─────────────────────────────────────────────────────────────────────
# --crop
# ─────────────────────────────────────────────────────────────────────

def fit_grid(gray, n, L, axis):
    """从像素里**测**出真实网格，不假设它等分且从 0 开始。

    2026-08-16：朴素做法（step=L/n、origin=0）裁出来**左边缘漏进隔壁格**。
    实测原因：真实列起点是 +15.5px 而不是 0。

    做法：白分隔线是沿另一轴近似均匀的行/列（方差极低、亮度≈254）。
    只用**真的检测到白线**的那些边界做线性拟合 x = a·i + b，
    再用残差自证拟合可信。这张图右半边有白线、左半边没有，
    所以必须拟合而不能逐边界就近吸附。
    """
    # gray.shape=(H,W)：var(0) → 每列一个数（长 W）；var(1) → 每行一个数（长 H）
    var = gray.var(axis)
    step = L / n
    pts = []
    for i in range(1, n):
        x0 = int(round(i * step))
        win = range(max(0, x0 - 9), min(L, x0 + 10))
        b = min(win, key=lambda x: var[x])
        if var[b] < 50:               # 只收真的白线
            pts.append((i, b))
    if len(pts) < 2:
        print(f"  !! {axis} 轴只找到 {len(pts)} 条分隔线，退回等分")
        return step, 0.0, 0.0
    I = np.array([p[0] for p in pts], float)
    X = np.array([p[1] for p in pts], float)
    a, b = np.polyfit(I, X, 1)
    res = float(np.abs(np.polyval([a, b], I) - X).max())
    return float(a), float(b), res


def cmd_crop(args):
    from PIL import Image
    if not TEASER.exists():
        sys.exit(f"!! 找不到 {TEASER}")
    im = Image.open(TEASER).convert("RGB")
    W, H = im.size
    g = np.asarray(im.convert("L"), np.float32)
    cw, cx, rc = fit_grid(g, GRID_C, W, 0)
    ch, cy, rr = fit_grid(g, GRID_R, H, 1)
    d = OUT / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    print(f"{TEASER.name}  {W}×{H}")
    print(f"  实测网格：格宽 {cw:.2f} 起点 {cx:+.1f}（残差 {rc:.1f}px）"
          f" / 格高 {ch:.2f} 起点 {cy:+.1f}（残差 {rr:.1f}px）")
    print(f"  朴素等分：格宽 {W/GRID_C:.2f} 起点 0 / 格高 {H/GRID_R:.2f} 起点 0"
          f"   ← 列起点差 {cx:.1f}px，就是它把隔壁格漏进来")
    if max(rc, rr) > 4:
        print(f"  !! 残差 {max(rc,rr):.1f}px 偏大，裁完务必自己看一眼")
    inset = args.inset

    def tile(c, r):
        x0, y0 = cx + c * cw, cy + r * ch
        mx, my = cw * inset, ch * inset      # 再往内缩一点，吃掉 1–2px 的拟合残差
        box = (max(0, round(x0 + mx)), max(0, round(y0 + my)),
               min(W, round(x0 + cw - mx)), min(H, round(y0 + ch - my)))
        return im.crop(box).resize((512, 512), Image.LANCZOS)
    for c, (key, tier, why) in STYLES.items():
        tile(c, 0).save(d / f"style_{key}.png")
        print(f"  style {key:16s} [{tier:4s}] col={c:2d}   {why}")
    for r, key in CONTENTS.items():
        tile(0, r).save(d / f"content_{key}.png")
        print(f"  content {key:14s} row={r}")
    print(f"\n→ {d}\n**裁完先自己看一眼**：格子有没有错位、有没有把表头切进来。")


# ─────────────────────────────────────────────────────────────────────
# --scan：把 WikiArt 目录清点出来 + 拼一张带编号的接触图
# ─────────────────────────────────────────────────────────────────────

def cmd_scan(args):
    """清点 WikiArt 目录 + 拼带编号的接触图，供人挑难/易两组。

    2026-08-16 订正：第一版用 `sorted(rglob)` 取前 N 张 —— 25700 张分在
    **257 个艺术家目录**里，前 64 张全是同一个人。**接触图里只有一个艺术家，
    挑不出任何多样性。** 现在改成**每个艺术家抽一张**，横跨 257 人。

    同时报分辨率分布：首张是 256×256，如果整库都是这个尺寸，
    **笔触细节可能已经不在图里**，那这批图测不出核心技术反对
    （逐区域笔触差异），阴性结果会无法归因。这一条必须先看清楚。
    """
    from PIL import Image, ImageDraw
    if not WIKIART.exists():
        sys.exit(f"!! 找不到 {WIKIART}\n   用 WIKIART=<路径> 覆盖")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    by_artist = {}
    for f in WIKIART.rglob("*"):
        if f.suffix.lower() in exts:
            by_artist.setdefault(f.parent.name, []).append(f)
    if not by_artist:
        sys.exit(f"!! {WIKIART} 下没有图片")
    total = sum(len(v) for v in by_artist.values())
    print(f"{WIKIART}\n  {total} 张 / {len(by_artist)} 个艺术家目录")

    # ── 分辨率普查：随机抽样，别只看首张 ──────────────────────────
    rng = np.random.default_rng(0)
    artists = sorted(by_artist)
    probe = [by_artist[a][0] for a in artists]
    idx = rng.choice(len(probe), min(args.probe, len(probe)), replace=False)
    sizes = {}
    for i in idx:
        try:
            sizes[Image.open(probe[i]).size] = sizes.get(Image.open(probe[i]).size, 0) + 1
        except Exception:
            pass
    print(f"  分辨率普查（抽 {len(idx)} 张）：")
    for sz, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {sz[0]}×{sz[1]}   {n} 张")
    mx = max((max(sz) for sz in sizes), default=0)
    if mx <= 320:
        print(f"  ⚠️ **最大边只有 {mx}px。** 笔触细节可能已经损失，"
              f"再放大到 1024 喂 SDXL，\n"
              f"     「逐区域笔触差异」这个我们要测的东西可能根本不在图里 ——"
              f"\n     那样阴性结果无法归因（是假设错还是分辨率不够？）。"
              f"**跑之前先决定这一条。**")

    # ── 每个艺术家抽一张，横跨全部艺术家 ─────────────────────────
    pick = rng.permutation(len(artists))[:args.scan_max]
    files = [by_artist[artists[i]][0] for i in sorted(pick)]

    cell, pad, hdr = 200, 4, 16
    ncol = 8
    nrow = (len(files) + ncol - 1) // ncol
    sh = Image.new("RGB", (ncol * (cell + pad) + pad,
                           nrow * (cell + hdr + pad) + pad), "white")
    dr = ImageDraw.Draw(sh)
    for i, f in enumerate(files):
        r, c = divmod(i, ncol)
        x, y = pad + c * (cell + pad), pad + r * (cell + hdr + pad)
        dr.text((x + 2, y + 2), f"[{i}] {f.parent.name[:24]}", fill="black")
        im = Image.open(f).convert("RGB"); im.thumbnail((cell, cell), Image.LANCZOS)
        sh.paste(im, (x, y + hdr))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "wikiart_scan.png"
    sh.save(p)
    (OUT / "wikiart_files.json").write_text(json.dumps(
        {str(i): str(f) for i, f in enumerate(files)}, ensure_ascii=False, indent=2))
    print(f"\n→ {p}   ({len(files)} 位艺术家各一张，编号 0..{len(files)-1})")
    print(f"→ {OUT/'wikiart_files.json'}")
    print(f"\n把接触图贴出来，我按下面的判据挑 5–6 张 hard + 2 张 easy：")
    print(f"  hard = 笔触/渲染**逐区域明显不同**（留白天空 vs 干笔山石；平涂 vs 细线）")
    print(f"  easy = 全图渲染**基本一致**（均匀花纹、单一线条、通幅平涂）")
    print(f"  ⚠️ 两组都要 —— 只挑 hard，失败了分不清「假设错」还是「实现错」；"
          f"\n     只挑 easy，等于回避了核心技术反对。")


# 由 --scan 的接触图挑出来后填这里；空 = 仍用 teaser 裁的那批
# 形如 {"ink_xxx": ("hard", "为什么选它"), ...}，key 是 wikiart_files.json 里的文件 stem
PICKS = {}


# ─────────────────────────────────────────────────────────────────────
# 子空间：单图内跨区域的广义特征问题
# ─────────────────────────────────────────────────────────────────────

def region_labels(feat, k=5, seed=0):
    """DFCC-lite：对空间位置做 PCA + KMeans 拿区域。
    照 StyleGallery 的 DFCC（UNet 特征 → PCA → K-means），不引分割模型。
    feat: [n_pos, dim]  →  labels: [n_pos]"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    z = PCA(n_components=min(32, feat.shape[1]), random_state=seed).fit_transform(feat)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(z)


def scatter(feat, labels):
    """按区域算类间 / 类内散度。
    S_b 大 = 该方向擅长区分语义区域（= 内容敏感）
    S_w 大 = 该方向在区域内部仍有变化（笔触/纹理住在这里）"""
    mu = feat.mean(0)
    d = feat.shape[1]
    S_b = np.zeros((d, d)); S_w = np.zeros((d, d))
    for c in np.unique(labels):
        X = feat[labels == c]
        m = X.mean(0)
        dm = (m - mu)[:, None]
        S_b += len(X) * (dm @ dm.T)
        Xc = X - m
        S_w += Xc.T @ Xc
    return S_b / len(feat), S_w / len(feat)


def style_subspace(feat, labels, r=RANK, eps=EPS):
    """解 S_w u = ρ (S_b + εI) u，取 ρ 最大的 r 个方向。

    直觉：**在区域内部有变化、但不擅长区分区域**的方向。
    这与 DICE Eq.13 同构（分子 = 同风格不同内容共享，分母 = 内容共享），
    区别只在对比源：DICE 跨图，这里跨单图内的区域。**这是唯一的新颖点。**
    """
    from scipy.linalg import eigh
    S_b, S_w = scatter(feat, labels)
    d = S_b.shape[0]
    w, V = eigh(S_w, S_b + eps * np.eye(d))
    U = V[:, np.argsort(w)[::-1][:r]]           # ρ 最大的 r 个
    # 广义特征向量不是正交的；取 QR 拿一组标准正交基再做投影。
    Q, _ = np.linalg.qr(U)
    return Q, np.sort(w)[::-1][:r]


# ─────────────────────────────────────────────────────────────────────
# 注入：改写 self-attention 的 K/V
# ─────────────────────────────────────────────────────────────────────

class StyleAttn:
    """两种模式共用一个 processor：
      record  跑参考图，把每层每步的 K/V 存下来
      inject  跑内容图，用 Q_content + (投影过的) K_style/V_style   ← StyleID 的做法

    坑都踩过了（negbranch）：不给任何人传 dict、自己遍历 named_modules、
    0 条硬退出、**带调用计数器**。
    """

    def __init__(self, st, name):
        self.st, self.name = st, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        import torch
        import torch.nn.functional as F
        is_self = encoder_hidden_states is None
        residual = hidden_states
        nd = hidden_states.ndim
        if nd == 4:
            b_, c_, h_, w_ = hidden_states.shape
            hidden_states = hidden_states.view(b_, c_, h_ * w_).transpose(1, 2)
        b, qlen, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if is_self else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)

        hit = is_self and qlen == self.st["q_len"]
        if hit:
            self.st["calls"] += 1
            key = (self.name, self.st["step"])
            if self.st["mode"] == "record":
                self.st["cache"][key] = (k.detach().cpu(), v.detach().cpu())
            elif self.st["mode"] == "inject" and key in self.st["cache"]:
                ks, vs = (t.to(k.device, k.dtype) for t in self.st["cache"][key])
                P = self.st["P"].get(self.name)
                if P is not None:
                    P = P.to(k.device, k.dtype)
                    if self.st["arm"] == "proj":
                        ks, vs = ks @ P, vs @ P
                    elif self.st["arm"] == "comp":
                        I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
                        ks, vs = ks @ (I - P), vs @ (I - P)
                    self.st["applied"] += 1
                # 参考图是单张（batch 1），内容侧有 CFG（batch 2）→ 广播
                if ks.shape[0] != b:
                    ks = ks.expand(b, -1, -1); vs = vs.expand(b, -1, -1)
                k, v = ks, vs

        heads = attn.heads
        sh = lambda t: t.view(b, -1, heads, t.shape[-1] // heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, -1, heads * (out.shape[-1]))
        out = attn.to_out[1](attn.to_out[0](out.to(q.dtype)))
        if nd == 4:
            out = out.transpose(-1, -2).reshape(b_, c_, h_, w_)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def install(pipe, st, where="up_blocks"):
    n = 0
    for name, mod in pipe.unet.named_modules():
        if hasattr(mod, "processor") and name.endswith("attn1") and name.startswith(where):
            mod.processor = StyleAttn(st, name); n += 1
    if n == 0:
        sys.exit(f"!! 一个 self-attn 都没挂上（where={where}）—— 算子是惰性的")
    print(f"  挂上 {n} 个 self-attn processor（{where}）")
    return n


# ─────────────────────────────────────────────────────────────────────
# --selftest：投影代数，零 GPU
# ─────────────────────────────────────────────────────────────────────

def cmd_selftest(args):
    fails = []

    def chk(label, good, detail=""):
        print(f"  [{'ok' if good else '!!'}] {label}" + (f"   {detail}" if detail else ""))
        if not good:
            fails.append(label)

    print("═" * 68); print("selftest（纯代数，不需要 GPU）"); print("═" * 68)
    rng = np.random.default_rng(0)
    d, n, k = 64, 800, 5
    labels = rng.integers(0, k, n)
    # 造一个「区域均值不同 + 区域内共享纹理方向」的合成特征：
    #   前 8 维承载「区域身份」（内容），后 8 维承载「共享纹理」（风格）
    feat = 0.1 * rng.standard_normal((n, d))
    for c in range(k):
        m = labels == c
        feat[m, :8] += rng.standard_normal(8) * 3.0          # 区域各不同 → S_b
    feat[:, 8:16] += rng.standard_normal((n, 8)) * 3.0        # 区域内变化 → S_w

    U, rho = style_subspace(feat, labels, r=8)
    P = U @ U.T
    chk("① P 幂等（P² = P）", np.allclose(P @ P, P, atol=1e-6))
    chk("① P 对称", np.allclose(P, P.T, atol=1e-8))
    chk("② P + (I−P) 重构原特征", np.allclose(feat @ P + feat @ (np.eye(d) - P), feat, atol=1e-6))

    # ③ 合成数据上，子空间应当更偏「共享纹理」那 8 维、而不是「区域身份」那 8 维
    e_style = np.linalg.norm(P[8:16, 8:16])
    e_content = np.linalg.norm(P[:8, :8])
    chk("③ 合成数据上子空间偏向共享纹理维而非区域身份维",
        e_style > 2 * e_content, f"‖P_texture‖={e_style:.3f} vs ‖P_region‖={e_content:.3f}")

    # ④ 满秩时 proj 应是恒等
    Uf, _ = style_subspace(feat, labels, r=d)
    chk("④ r=d 时 P ≈ I（退化一致性）", np.allclose(Uf @ Uf.T, np.eye(d), atol=1e-6))

    print("\n" + ("全部通过" if not fails else f"!! 失败：{fails}"))
    print("注：③ 只说明代数在**合成数据**上按预期工作，**不说明它在真实画作上成立**"
          "\n    —— 那正是 --run 要看的。")
    return 0 if not fails else 1


# ─────────────────────────────────────────────────────────────────────
# --run
# ─────────────────────────────────────────────────────────────────────

def cmd_run(args):
    import torch
    from PIL import Image

    tiles = OUT / "tiles"
    if not tiles.exists():
        sys.exit("!! 先跑 --crop")
    d = OUT / f"r{args.rank}"
    d.mkdir(parents=True, exist_ok=True)

    pipe = _load(args)
    st = {"mode": "off", "cache": {}, "step": 0, "q_len": args.qlen,
          "P": {}, "arm": "full", "calls": 0, "applied": 0}
    install(pipe, st, args.where)

    for c, (key, tier, why) in STYLES.items():
        sp = tiles / f"style_{key}.png"
        print(f"\n[{key}] ({tier})  {why}", flush=True)

        # ── 1. 记录参考图的 K/V（前向加噪，不做 inversion，见 docstring 简化 ①）
        lat = _encode(pipe, Image.open(sp).convert("RGB"))
        st.update(mode="record", cache={}, step=0, calls=0)
        _record_reference(pipe, st, lat, args)

        if st["calls"] == 0:
            sys.exit("!! 记录阶段一次都没命中 —— q_len 不对？用 --qlen 换一档")

        # ── 2. 每层算子空间
        st["P"] = {}
        rhos = []
        for name in sorted({k[0] for k in st["cache"]}):
            ks = torch.stack([st["cache"][(name, s)][0][0].float()
                              for s in range(1, args.steps + 1)
                              if (name, s) in st["cache"]]).mean(0).numpy()
            lab = region_labels(ks, k=args.k)
            U, rho = style_subspace(ks, lab, r=args.rank)
            st["P"][name] = torch.from_numpy(U @ U.T)
            rhos.append(rho[0])
        print(f"    {len(st['P'])} 层算完子空间，ρ_max 中位数 {np.median(rhos):.2f}")

        # ── 3. 三臂
        for cr, ckey in CONTENTS.items():
            cont = Image.open(tiles / f"content_{ckey}.png").convert("RGB")
            for arm in ARMS:
                f = d / f"{key}__{ckey}__{arm}.png"
                if f.exists():
                    continue
                st.update(mode="inject", arm=arm, step=0, applied=0)
                img = _inject_generate(pipe, st, cont, args)
                img.save(f)
            print(f"    → {ckey}: 三臂完成（applied={st['applied']}）")

    (d / "meta.json").write_text(json.dumps(
        {"rank": args.rank, "k": args.k, "qlen": args.qlen, "steps": args.steps,
         "styles": {v[0]: {"tier": v[1], "why": v[2]} for v in STYLES.values()}},
        ensure_ascii=False, indent=2))
    print(f"\n→ {d}\n下一步：python style_probe/subspace_look.py --sheet --rank {args.rank}")


def _load(args):
    """SD1.5 + DDIM。DDIM 是因为下面要手写 add_noise / step 的循环，
    PNDM 那种带内部缓冲的调度器从时刻表中间切入会出错。"""
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"  backbone {args.model}  @{RES}²  scheduler=DDIM")
    return pipe


def _encode(pipe, img):
    import torch
    import numpy as _np
    x = torch.from_numpy(_np.array(img.resize((RES, RES)))).permute(2, 0, 1)[None]
    x = x.half().cuda() / 127.5 - 1
    with torch.no_grad():
        return pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor


def _empty_emb(pipe):
    import torch
    with torch.no_grad():
        ids = pipe.tokenizer([""], padding="max_length",
                             max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.cuda()
        return pipe.text_encoder(ids)[0]


def _record_reference(pipe, st, lat, args):
    """按采样时刻表对参考 latent 前向加噪，逐步跑 UNet，收 self-attn 的 K/V。
    **不是 DDIM inversion**（简化 ①）—— 只为拿到「参考图在各时刻的自注意力特征」。"""
    import torch
    sch = pipe.scheduler
    sch.set_timesteps(args.steps, device="cuda")
    emb = _empty_emb(pipe)
    with torch.no_grad():
        for i, t in enumerate(sch.timesteps):
            st["step"] = i + 1
            noise = torch.randn(lat.shape, device="cuda", dtype=lat.dtype,
                                generator=torch.Generator("cuda").manual_seed(i))
            xt = sch.add_noise(lat, noise, t.reshape(1))
            pipe.unet(xt, t, encoder_hidden_states=emb)


def _inject_generate(pipe, st, content_img, args):
    """SDEdit 式：内容图加噪到 strength 对应的时刻再去噪；
    过程中 self-attn 的 K/V 被换成（投影过的）参考图的 —— 即 StyleID 的做法。"""
    import torch
    from PIL import Image
    sch = pipe.scheduler
    sch.set_timesteps(args.steps, device="cuda")
    lat = _encode(pipe, content_img)
    emb = _empty_emb(pipe)
    start = int(args.steps * (1 - args.strength))
    ts = sch.timesteps[start:]
    with torch.no_grad():
        noise = torch.randn(lat.shape, device="cuda", dtype=lat.dtype,
                            generator=torch.Generator("cuda").manual_seed(42))
        xt = sch.add_noise(lat, noise, ts[:1])
        for i, t in enumerate(ts):
            st["step"] = start + i + 1        # 与 record 阶段的 step 对齐
            eps = pipe.unet(xt, t, encoder_hidden_states=emb).sample
            xt = sch.step(eps, t, xt).prev_sample
        img = pipe.vae.decode(xt / pipe.vae.config.scaling_factor).sample
    a = ((img / 2 + .5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255)
    return Image.fromarray(a.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────
# --sheet
# ─────────────────────────────────────────────────────────────────────

def cmd_sheet(args):
    from PIL import Image, ImageDraw
    d = OUT / f"r{args.rank}"
    tiles = OUT / "tiles"
    cell, pad, hdr = args.cell, 6, 22
    cols = ["style"] + [f"{a}" for a in ARMS]
    rows = [(v[0], v[1], v[2]) for v in STYLES.values()]
    rows = [r for r in rows if all((d / f"{r[0]}__{args.content}__{a}.png").exists()
                                   for a in ARMS)]
    if not rows:
        sys.exit(f"!! {d} 里没有完整三臂（content={args.content}），先 --run")
    # 难组在上、易组在下 —— 对比一眼可见
    rows.sort(key=lambda r: {"hard": 0, "mid": 1, "easy": 2}[r[1]])

    W = pad + len(cols) * (cell + pad)
    H = hdr + len(rows) * (cell + hdr + pad)
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    for c, nm in enumerate(cols):
        tag = {"style": "参考图", "full": "full (≈StyleID)",
               "proj": "proj ← 被测", "comp": "comp ← 证伪器"}[nm]
        dr.text((pad + c * (cell + pad) + 4, 5), tag, fill="black")
    y = hdr
    for key, tier, why in rows:
        dr.text((pad + 4, y + 4), f"[{tier}] {key}   {why[:70]}",
                fill=("red" if tier == "hard" else "black"))
        for c, nm in enumerate(cols):
            f = (tiles / f"style_{key}.png") if nm == "style" else \
                (d / f"{key}__{args.content}__{nm}.png")
            im = Image.open(f).convert("RGB"); im.thumbnail((cell, cell), Image.LANCZOS)
            sheet.paste(im, (pad + c * (cell + pad), y + hdr))
        y += cell + hdr + pad
    p = d / f"sheet_{args.content}.png"
    sheet.save(p)
    print(f"{p}")
    print(f"\n判读（先看红色的 hard 组）：")
    print(f"  · proj 保住逐区域笔触、comp 只剩物体      → 假设成立，进方法")
    print(f"  · proj 变得均匀平滑（厚涂那部分没了）      → **技术反对成立，判死**")
    print(f"  · comp 也带笔触                            → 没分开，判死")
    print(f"  · easy 组成、hard 组败                     → **精确失败模式："
          f"区域对比丢掉区域相关的渲染**")
    print(f"  · 三列看不出差别                           → 先查 applied 计数，再扫 --rank")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--k", type=int, default=5, help="区域数")
    ap.add_argument("--qlen", type=int, default=Q_LEN)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--model", default=SD15, help="backbone；SD15_PATH 环境变量可覆盖")
    ap.add_argument("--where", default="up_blocks",
                    help="在哪些 block 注入（DICE 的 style 子空间取 down_blocks[0]）")
    ap.add_argument("--strength", type=float, default=0.7, help="SDEdit 加噪强度")
    ap.add_argument("--content", default="horse")
    ap.add_argument("--cell", type=int, default=380)
    ap.add_argument("--scan-max", type=int, default=64,
                    help="接触图放几位艺术家（每人一张）")
    ap.add_argument("--probe", type=int, default=200,
                    help="分辨率普查抽样张数")
    ap.add_argument("--inset", type=float, default=0.03,
                    help="每格再往内缩的比例，吃掉拟合残差与边框")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", action="store_true",
                   help="清点 WikiArt 目录并拼接触图（挑图用）")
    g.add_argument("--crop", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    if a.scan:
        cmd_scan(a)
    elif a.crop:
        cmd_crop(a)
    elif a.selftest:
        sys.exit(cmd_selftest(a))
    elif a.run:
        cmd_run(a)
    else:
        cmd_sheet(a)


if __name__ == "__main__":
    main()
