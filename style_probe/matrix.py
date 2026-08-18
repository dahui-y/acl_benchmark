#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐 (style, content) 的失效矩阵 —— playbook 第 3 步的仪器。

    ArtFID 是**集合级**的一个数：800 张算一个 FID。它能告诉我们「整体好不好」，
    答不了「在哪坏」。而第 3 步要的恰恰是后者：一个**具体、可命名**的失效。

    所以这里把 40×20 的每一格单独量：

      · 内容项 = LPIPS(stylized, content)      —— ArtFID 的内容项本来就是它
      · 风格项 = 1 - cos(f(stylized), f(style)) —— f 是 **art_inception**，
        也就是 ArtFID 的 FID 项所用的同一组特征，只是从集合级降到逐对级

    两项都不是新造的尺子，都是在位者自己评测代码里的组件。这一点重要：
    如果失效只在我们自己发明的度量下才出现，那它多半是度量的性质而不是
    方法的性质 —— 前面已经因此死过一次（GroundingDINO 那个）。

    ⚠️ 没有用 StyleSSP 的 image_metrics.GramLoss。它的 gram_matrix 把
    (C,H,W) 压成一维再做 bmm，返回标量，等于只比较特征图总能量，丢掉全部
    通道间相关 —— 而通道相关就是风格的定义。不改别人的评测代码，绕开它。

关键读数是**方差分解**：把矩阵 M[i,j] 拆成

    M = 总均值 + 画风效应(i) + 内容效应(j) + 残差(i,j)

三项各占多少方差，直接回答止损问题：

  · 画风效应大  → 「某类画风系统性地坏」，是可命名的失效，进第 4 步
  · 内容效应大  → 「某类 content 系统性地坏」，同样可命名
  · 残差独大    → 失效是散的，**没有可写进 problem statement 的东西**，方向停

用法：
    python style_probe/matrix.py --seed 0
    python style_probe/matrix.py --seed 0 --seeds 0 1 2   # 加噪声地板
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "protocol"
EVAL = REPO / "help_code" / "StyleSSP" / "evaluation"


def _load_incep(device):
    """art_inception —— ArtFID 的 FID 项用的那一个，不是 ImageNet inception。"""
    sys.path.insert(0, str(EVAL))
    import torch
    import inception as I
    import utils as U
    ck = U.download("https://huggingface.co/matthias-wright/art_inception/"
                    "resolve/main/art_inception.pth")
    m = I.Inception3().to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    return m


def _feats(model, paths, device, bs=25):
    """[N, 2048]。变换与 eval_artfid.get_activations 一致：Resize(512) + ToTensor。"""
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor
    tf = Compose([Resize(512), ToTensor()])
    out = np.empty((len(paths), 2048), np.float64)
    for s in range(0, len(paths), bs):
        b = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + bs]])
        with torch.no_grad():
            f = model(b.to(device), return_features=True)
        out[s:s + len(b)] = f.cpu().numpy()
    return out


def _lpips(paths_a, paths_b, device, bs=20):
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor
    sys.path.insert(0, str(EVAL))
    import image_metrics as M
    met = M.LPIPS().to(device)
    tf = Compose([Resize(512), ToTensor()])
    out = np.empty(len(paths_a), np.float64)
    for s in range(0, len(paths_a), bs):
        a = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths_a[s:s + bs]])
        b = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths_b[s:s + bs]])
        with torch.no_grad():
            d = met(a.to(device), b.to(device))
        out[s:s + len(a)] = d.flatten().cpu().numpy()
    return out


def decompose(M, name, higher_is_worse=True):
    """二因素方差分解。M[i,j]：i = style，j = content。

    higher_is_worse：距离类指标（S、C）越大越差；**增益类指标（ΔS）反过来**。
    2026-08-17：第一版对所有矩阵都按「越大越差」打标签，于是 ΔS 那一段
    把增益最高的三个画风印成了「最差」。读反了会把结论整个颠倒。
    """
    g = M.mean()
    ri = M.mean(1) - g                     # 画风效应
    cj = M.mean(0) - g                     # 内容效应
    res = M - g - ri[:, None] - cj[None, :]
    v = M.var()
    vs, vc, vr = ri.var(), cj.var(), res.var()
    tot = vs + vc + vr
    print(f"\n── {name}  均值 {g:.4f}  标准差 {M.std():.4f}")
    print(f"   方差分解：画风 {vs/tot*100:5.1f}%   内容 {vc/tot*100:5.1f}%"
          f"   残差 {vr/tot*100:5.1f}%     (总方差 {v:.2e})")
    def _fmt(v, o):
        return [(int(k), round(float(v[k]), 4)) for k in o]
    o = np.argsort(ri)
    bad, good = (o[-3:][::-1], o[:3]) if higher_is_worse else (o[:3], o[-3:][::-1])
    print(f"   最差 3 个画风(行)：{_fmt(ri, bad)}")
    print(f"   最好 3 个画风(行)：{_fmt(ri, good)}")
    o = np.argsort(cj)
    badc = o[-3:][::-1] if higher_is_worse else o[:3]
    print(f"   最差 3 个 content：{_fmt(cj, badc)}")
    return dict(grand=float(g), style_eff=ri.tolist(), cnt_eff=cj.tolist(),
                frac=dict(style=float(vs/tot), content=float(vc/tot),
                          resid=float(vr/tot)))


def sheet(M, path, title):
    """把矩阵画成热图。看图比看数快 —— 结构是不是块状，一眼就知道。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xlabel("content j"); ax.set_ylabel("style i"); ax.set_title(title)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)


def _from_dirs(tdir, sdir, cdir):
    """任意目录模式：从 tar 的文件名 {style}__{content}.png 反推 40×20 网格。

    styleid_batch.py 的输出不在 protocol/seedN 布局里，也没有 manifest。
    配对关系已经写在文件名里，直接反推比再维护一份索引更不容易错。
    """
    import collections
    grid = collections.defaultdict(dict)
    for f in sorted(Path(tdir).glob("*.png")):
        if "__" not in f.stem:      # matrix_*.png 是我们自己写的热图，不是输出
            continue
        s_, c_ = f.stem.split("__", 1)
        grid[s_][c_] = f
    ss = sorted(grid)
    cs = sorted({c for v in grid.values() for c in v})
    bad = [(s_, c_) for s_ in ss for c_ in cs if c_ not in grid[s_]]
    if bad:
        sys.exit(f"!! 网格不完整，缺 {len(bad)} 格，例：{bad[0]}")

    def _find(d, stem):
        for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
            if (Path(d) / f"{stem}{ext}").exists():
                return Path(d) / f"{stem}{ext}"
        sys.exit(f"!! 在 {d} 里找不到 {stem}.*")

    sty = [_find(sdir, s_) for s_ in ss]
    cnt = [_find(cdir, c_) for c_ in cs]
    tar = [[grid[s_][c_] for c_ in cs] for s_ in ss]
    return sty, cnt, tar


def run(seed, args):
    import torch
    if getattr(args, "tar", None):
        d = Path(args.tar)
        sty, cnt, tar = _from_dirs(args.tar, args.sty, args.cnt)
        ns, nc = len(sty), len(cnt)
    else:
        d = OUT / f"seed{seed}"
        man = json.loads((d / "manifest.json").read_text())
        sty = [d / "sty" / x["file"] for x in man["style"]]
        cnt = [d / "cnt" / x["file"] for x in man["content"]]
        ns, nc = len(sty), len(cnt)
        tar = [[d / "tar" / f"{s.stem}__{c.stem}.png" for c in cnt] for s in sty]
    miss = [p for row in tar for p in row if not p.exists()]
    if miss:
        sys.exit(f"!! 缺 {len(miss)} 张 stylized，例：{miss[0].name}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flat = [p for row in tar for p in row]

    print(f"seed {seed}: {ns}×{nc} = {len(flat)} 格，device={dev}")
    m = _load_incep(dev)
    f_t = _feats(m, flat, dev).reshape(ns, nc, -1)
    f_s = _feats(m, sty, dev)
    del m
    torch.cuda.empty_cache()
    # 逐对风格距离：与 style 参考图的 art_inception 特征的余弦距离
    a = f_t / np.linalg.norm(f_t, axis=2, keepdims=True)
    b = f_s / np.linalg.norm(f_s, axis=1, keepdims=True)
    S = 1.0 - np.einsum("ijk,ik->ij", a, b)

    C = _lpips(flat, [c for _ in sty for c in cnt], dev).reshape(ns, nc)

    # ★ 零假设对照。**没有它，上面两个数不可信。**
    #
    # 行效应有一个平凡解释：某张 style 图在 art_inception 空间里本来就
    # 离所有东西都远，于是它那一行整体偏高 —— 这测的是图片的固有位置，
    # 不是「这个画风难迁移」。列效应同理：复杂的 content 做任何改动
    # LPIPS 都大。绑定检测器就是死在这类「仪器在测自己」上。
    #
    #   ΔS = d(content, style) − d(stylized, style)   风格**靠近了多少**
    #        每张 style 图的固有位置在相减中抵消
    #   Cn = LPIPS(stylized, content) / LPIPS(style, content)
    #        用「完全变成 style 图」当满量程，归掉 content 自身复杂度
    f_c = _feats(_load_incep(dev), cnt, dev)
    c_n = f_c / np.linalg.norm(f_c, axis=1, keepdims=True)
    S0 = 1.0 - c_n @ b.T                      # [nc, ns]
    S0 = S0.T                                 # [ns, nc]
    dS = S0 - S                               # 正 = 迁移让它更接近该风格
    Cmax = _lpips([s for s in sty for _ in cnt],
                  [c for _ in sty for c in cnt], dev).reshape(ns, nc)
    Cn = C / np.maximum(Cmax, 1e-6)

    np.savez(d / "matrix.npz", style_dist=S, content_lpips=C,
             style_gain=dS, style_null=S0, content_norm=Cn, content_max=Cmax,
             sty=[str(p) for p in sty], cnt=[str(p) for p in cnt])
    sheet(S, d / "matrix_style.png", f"seed{seed} style distance (1-cos, art_inception)")
    sheet(C, d / "matrix_content.png", f"seed{seed} content LPIPS")
    sheet(dS, d / "matrix_gain.png", f"seed{seed} style GAIN (null-corrected)")
    sheet(Cn, d / "matrix_cnorm.png", f"seed{seed} content LPIPS / full-replacement")
    r = {"style": decompose(S, "风格距离 1-cos（未去固有偏置）"),
         "content": decompose(C, "内容 LPIPS（未归一）"),
         "style_gain": decompose(dS, "★ 风格增益 ΔS（已去 style 图固有位置）",
                                 higher_is_worse=False),
         "content_norm": decompose(Cn, "★ 内容 LPIPS / 满量程（已归一）")}
    neg = np.argwhere(dS < 0)
    print(f"\n   注：ΔS 均值 {dS.mean():.4f}（>0 才说明迁移真的在靠近风格）；"
          f"**负增益格数 {len(neg)}/{dS.size}** —— 这些格子里，"
          f"「风格化」后的图比原 content 图**离目标风格更远**")
    if len(neg):
        import collections
        rc = collections.Counter(int(i) for i, _ in neg)
        print(f"   负增益集中在哪些画风(行)：{rc.most_common(6)}")
        cc = collections.Counter(int(j) for _, j in neg)
        print(f"   集中在哪些 content(列)：{cc.most_common(6)}")
    (d / "matrix.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"\n→ {d/'matrix.npz'} / matrix_style.png / matrix_content.png")
    return S, C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tar", default=None, help="stylized 目录（任意目录模式）")
    ap.add_argument("--sty", default=None, help="style 源图目录")
    ap.add_argument("--cnt", default=None, help="content 源图目录")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="给多个 seed 时，额外报**噪声地板**：同一格在不同抽样"
                         "下的差。任何小于地板的结构都不是结构。")
    a = ap.parse_args()
    if a.seeds:
        Ss, Cs = [], []
        for s in a.seeds:
            S, C = run(s, a)
            Ss.append(S); Cs.append(C)
        # 不同 seed 抽到的是不同的 (style, content)，格子对不上 ——
        # 所以地板只能比**效应量的分布**，不能逐格相减。这一点必须说清楚，
        # 否则会把「换了一批画」误当成「同一批画的噪声」。
        print("\n── 跨 seed（注意：不同 seed 是不同的画，格子不可逐一对应）")
        for nm, arr in (("风格距离", Ss), ("内容 LPIPS", Cs)):
            gm = [float(x.mean()) for x in arr]
            sd = [float(x.std()) for x in arr]
            print(f"   {nm}：均值 {['%.4f'%g for g in gm]}  "
                  f"标准差 {['%.4f'%s for s in sd]}")
            print(f"     → 抽样引起的均值波动 = {np.std(gm):.4f}"
                  f"（任何小于它的「改进」都读不出来）")
    else:
        run(a.seed, a)


if __name__ == "__main__":
    main()
