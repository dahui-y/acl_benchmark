"""标准表打分器：FID / KID / IS + FIDp / KIDp / ISp + CLIP。

这是项目里欠得最久的一笔债 —— 参考图 2026-08-11 就备好了（§7.1.5–§7.1.7，
3424 张已审计），但**打分代码一行都没有**。§10 转向之后它成了主路。

────────────────────────────────────────────────────────────────────────
一、patch 三列到底怎么算 —— **推出来的，不是猜的**
────────────────────────────────────────────────────────────────────────
ScaleDiff §4.1 只写 *"extract multiple patches from each image ...
following [8]"*（DemoFusion），而 DemoFusion 仓库里**没有评测代码**。
所以裁法要从同线论文的原文拼：

  AccDiffusion / v2：*"we crop **10 local patches at 1x resolution**
      (native resolution) from each generated high-resolution image
      and subsequently resize them, yielding FIDc and ISc."*
  FAM（CVPR25）：  *"we extract **10 random crops** from each image
      before calculating FID and KID, referring to these metrics as
      FIDc and KIDc."*
  PixelRush（CVPR）：*"FIDc, a variant of FID computed on local crops
      **without resizing**"*

三家一致：**每张生成图取 10 个原生分辨率（SDXL = 1024²）的随机裁块**。

**真图那一侧三家都没写。** 但 AccDiffusion Table 1 的 1× 行把它定死了：

    1024x1024 (1x)  SDXL-DI   FIDr 58.49   FIDc 58.08

在 1× 上，"10 个 1024² 裁块"退化成同一张整图重复 10 次。两个数几乎相等，
**只可能在参考侧完全相同的情况下成立** —— 若 FIDc 的参考侧换成了真图的
裁块，两个数不会这么近。

  -> **参考侧 = 真图整图，FID 与 FIDp 共用同一套特征。差别全在生成侧。**

这也解释了为什么 FIDp（38.89）反而**低于** FID（61.87）：整张 4096² 缩到
299² 后不像任何一张真实照片，而一个 1024² 裁块缩到 299² 就很像。

────────────────────────────────────────────────────────────────────────
二、为什么不用 torchmetrics 的默认路径
────────────────────────────────────────────────────────────────────────
用的。特征塔必须是 **torch-fidelity 的 InceptionV3**
（`weights-inception-2015-12-05`，即 TF 移植版），这是 pytorch-fid /
torch-fidelity / torchmetrics 共用的那一个，也是全线发表数字的来源。
换成 torchvision 自带的 `inception_v3` 数值会整体平移，**与发表值不可比**
—— 所以那条路要显式 `--allow-torchvision` 才走，并且会一路打警告。

指标本身是手算的（FID / KID / IS 各二十行），因为我们要：
  · 一次抽特征、多个臂复用、落盘缓存（4096² PNG 解码才是瓶颈）；
  · 对**同一批特征**做 bootstrap 求噪声地板（见下）。

────────────────────────────────────────────────────────────────────────
三、噪声地板 —— 这把尺也要有阳性对照
────────────────────────────────────────────────────────────────────────
项目纪律：任何尺子先报量程，再报读数（§8.9 系列）。这里的量程是
**对图像重采样的 bootstrap 标准差**：把 n 张图有放回重抽 B 次重算指标。

  · KID / IS：便宜，默认 B=100 直接算；
  · FID：每次要一个 2048×2048 的 sqrtm（~20 s），默认**不做**，
    需要时 `--boot-fid`。

**判据只在差值 > 2×bootstrap 标准差时才算数。** 这一条写死在这里，
免得看到数字再商量。

────────────────────────────────────────────────────────────────────────
四、KID 的一个陷阱
────────────────────────────────────────────────────────────────────────
KID 的数值依赖 `subset_size`（无偏 MMD² 在子集上求平均）。n=200 与
n=1000 算出来的 KID **不是一个数**。所以：

  · P0（n=200）的 KIDp **不可与发表值比**，只用 A/B 差值（§10.7 判据一）；
  · P1（n=1000）复现 ScaleDiff 行时才谈绝对值可比性。
`subset_size` 一律写进输出 json，两个臂必须相同。

────────────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────────────
    # P0：只要 patch 两列（KIDp / ISp），省掉真图之外的一切
    python scalediff_probe/std_table.py --patch-only \
        --arms "NPA=parti_npa" "NPA+shift=parti_npashift" "MD=parti_md"

    # P1：全表
    python scalediff_probe/std_table.py --res 4096 \
        --arms "ScaleDiff=laion_hi" --clip

    # 只抽特征（长，可后台）；后续跑指标秒出
    python scalediff_probe/std_table.py --arms ... --features-only
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CROPS = 10           # 三家一致
CROP_SIZE = 1024     # SDXL 原生分辨率 = "1x resolution"
CROP_SEED = 0        # 固定；两个臂必须同一颗，否则裁块位置不同不可比


# ══════════════════════════════════════════════════════════════════════
# 特征塔
# ══════════════════════════════════════════════════════════════════════
class Inception:
    """torch-fidelity 的 InceptionV3：2048-d pool 特征 + 1008-d logits。

    输入约定沿用它自己的：uint8 NCHW，[0,255]，**由它内部** resize 到 299²
    （双线性 + 它那套 TF 兼容处理）。我们不在外面先缩 —— 外面缩一次、
    里面再缩一次，数值就离发表值远了。
    """

    name = "torch-fidelity InceptionV3 (weights-inception-2015-12-05)"

    def __init__(self, weights=None):
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3)
        kw = {}
        if weights:
            kw["feature_extractor_weights_path"] = str(weights)
        self.m = FeatureExtractorInceptionV3(
            "inception", ["2048", "logits_unbiased"], **kw).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, batch_u8):
        f, lg = self.m(batch_u8.to(DEV))
        return f.float().cpu().numpy(), lg.float().cpu().numpy()


class TorchvisionInception:
    """退路。**数值与发表值不可比**，必须 --allow-torchvision 才会走到。"""

    name = "torchvision inception_v3 (!! 与发表值不可比 !!)"

    def __init__(self, weights=None):
        from torchvision.models import Inception_V3_Weights, inception_v3
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        m.fc_orig = m.fc
        m.fc = torch.nn.Identity()
        self.m = m.eval().to(DEV)

    @torch.no_grad()
    def __call__(self, batch_u8):
        x = batch_u8.to(DEV).float() / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - torch.tensor([0.485, 0.456, 0.406], device=DEV)
             .view(1, 3, 1, 1)) / torch.tensor(
                 [0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        f = self.m(x)
        lg = self.m.fc_orig(f)
        return f.float().cpu().numpy(), lg.float().cpu().numpy()


def load_inception(weights, allow_tv):
    try:
        t = Inception(weights)
        print(f"特征塔：{t.name}")
        return t
    except Exception as e:
        print(f"torch-fidelity 不可用：{type(e).__name__}: {e}")
    if not allow_tv:
        sys.exit(
            "\n拿不到参考特征塔。两条出路，按顺序试：\n"
            "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            "torch-fidelity\n"
            "  # 权重会从 github release 拉（net_probe.sh 实测 github 直连 200）；\n"
            "  # 若拉不动，手动下 weights-inception-2015-12-05-6726825d.pth\n"
            "  # 再用 --inception-weights <路径> 指过去。\n\n"
            "**不要**图省事用 --allow-torchvision 去凑数 —— 那条路算出来的\n"
            "FID/KID/IS 与 ScaleDiff/AccDiffusion/FAM 发表的数字不在同一把\n"
            "尺子上，P1 的复现判据（§10.7）当场作废。")
    t = TorchvisionInception(weights)
    print(f"⚠️⚠️ 特征塔：{t.name}\n"
          f"⚠️⚠️ 这些数**只能自比**，绝不可与任何发表值比较。")
    return t


# ══════════════════════════════════════════════════════════════════════
# 指标（手算，因为要在同一批缓存特征上反复 bootstrap）
# ══════════════════════════════════════════════════════════════════════
def _sqrtm(a):
    """不传 disp —— 新 scipy 弃用了它，旧 scipy 默认值本来就是我们要的。"""
    from scipy import linalg
    r = linalg.sqrtm(a)
    return r[0] if isinstance(r, tuple) else r


def fid(f1, f2, eps=1e-6):
    """标准 Frechet 距离。奇异协方差时的补偿沿用 pytorch-fid 的做法。

    **小样本偏差是这个指标最容易被忽视的性质**：n 远小于特征维 2048 时，
    样本协方差本身就不满秩，FID 被系统性抬高。实测（合成高斯，d=64）：

        真值 16.0    n=400 -> 20.85    n=2000 -> 17.17    n=20000 -> 16.08

    d=2048 时这个偏差要严重得多。**所以 P0（n=200）不算 FID**，
    只算 KID（无偏）与 IS（不需参考集）—— 见 §10.7。
    """
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean = _sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = _sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        # 数值噪声引入的微小虚部；虚部不小则是真出了问题，要炸出来
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise RuntimeError("sqrtm 虚部过大，协方差可能奇异")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2)
                 - 2 * np.trace(covmean))


def kid(f1, f2, subset_size, n_subsets, rng):
    """无偏 MMD²，多项式核 k(x,y) = (x·y/d + 1)³ —— 通行定义。

    数值依赖 subset_size，两个臂必须用同一个（见文件头第四节）。
    """
    d = f1.shape[1]
    m = min(subset_size, len(f1), len(f2))
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals))


def inception_score(logits, splits=10):
    """标准 IS：logits -> softmax -> 每份算 KL(p(y|x) || p(y))，取 exp 均值。"""
    x = logits - logits.max(1, keepdims=True)
    p = np.exp(x) / np.exp(x).sum(1, keepdims=True)
    n = len(p)
    out = []
    for i in range(splits):
        q = p[i * n // splits:(i + 1) * n // splits]
        if len(q) < 2:
            continue
        py = q.mean(0, keepdims=True)
        out.append(np.exp((q * (np.log(q + 1e-12) - np.log(py + 1e-12)))
                          .sum(1).mean()))
    return float(np.mean(out))


# ══════════════════════════════════════════════════════════════════════
# 取图 / 抽特征
# ══════════════════════════════════════════════════════════════════════
def arm_files(d, res):
    """该臂的 {idx: path}。res=None 取每个 idx 的最大分辨率。"""
    out = {}
    for p in Path(d).glob("[0-9]*_[0-9]*.png"):
        try:
            i, r = p.stem.split("_")
            i, r = int(i), int(r)
        except ValueError:
            continue
        if res is not None and r != res:
            continue
        if i not in out or r > out[i][0]:
            out[i] = (r, p)
    return {i: v[1] for i, v in out.items()}


def real_files(d, prompts, split):
    """真图：优先 clean.json（`real_audit.py --apply` 的产物）。

    clean.json 的键是 `alive_idx`（idx 整数表），文件名是 `{idx:05d}.jpg`
    —— 与 `fetch_real_images` / `base_run.py --alive` 同一套约定。
    split 过滤走 eval_prompts.json 里的 `split` 字段：**tune 用来开发、
    eval 只在最后出表时用一次**，两者不许混（混了就是在测试集上调参）。
    """
    d = Path(d)
    c = d / "clean.json"
    if not c.exists():
        ps = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        print(f"⚠ 真图：{len(ps)} 张（**未过 real_audit**，先跑 "
              f"`python scalediff_probe/real_audit.py --apply`）")
        return ps
    keep = set(json.loads(c.read_text())["alive_idx"])
    if split:
        pp = Path(prompts)
        if not pp.exists():
            sys.exit(f"要按 split={split} 过滤，但 {pp} 不存在")
        items = json.loads(pp.read_text())
        want = {i for i, x in enumerate(items) if x.get("split") == split}
        before = len(keep)
        keep &= want
        print(f"真图：clean.json {before} 张 -> split={split} 留 {len(keep)}")
    ps = [d / f"{i:05d}.jpg" for i in sorted(keep)]
    ps = [p for p in ps if p.exists()]
    print(f"真图：{len(ps)} 张（已过 real_audit）")
    return ps


def crops_of(im, n, size, rng):
    """n 个原生分辨率随机裁块。图小于 size 时退化为整图（1× 那一档）。"""
    W, H = im.size
    if W <= size or H <= size:
        return [im] * n
    out = []
    for _ in range(n):
        x = int(rng.integers(0, W - size + 1))
        y = int(rng.integers(0, H - size + 1))
        out.append(im.crop((x, y, x + size, y + size)))
    return out


def to_u8(ims):
    a = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ims])
    return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()


def extract(tower, paths, want_crops, cache, tag, px_budget, ncrop=CROPS,
            csize=CROP_SIZE, cseed=CROP_SEED):
    """抽特征并落盘。缓存键含文件清单指纹 + 裁块参数，参数一变自动失效。

    **不在外面做任何缩放。** 特征塔内部会 bilinear 到 299²，那是发表数字
    走的那条路；外面先缩一次、里面再缩一次，两步重采样出来的数就偏了。
    代价是一批里的图尺寸必须一致，所以攒批的规则是
    「尺寸相同 **且** 总像素不超预算」—— 4096² 一批只放得下两三张，
    1024² 裁块能放几十张，自动适应。
    """
    key = hashlib.sha256(
        (tag + "|" + str(want_crops) + f"|{ncrop}|{csize}|{cseed}|"
         + "|".join(f"{p.name}:{p.stat().st_size}" for p in paths))
        .encode()).hexdigest()[:16]
    cp = Path(cache) / f"{tag}_{key}.npz"
    if cp.exists():
        z = np.load(cp)
        print(f"  缓存命中 {cp.name}  n={len(z['feat'])}")
        return z["feat"], z["logit"], list(z["idx"])

    Path(cache).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cseed)
    F, L, I = [], [], []
    buf, bidx = [], []
    t0 = time.time()

    def flush():
        if not buf:
            return
        f, l = tower(to_u8(buf))
        F.append(f)
        L.append(l)
        I.extend(bidx)
        buf.clear()
        bidx.clear()

    for k, p in enumerate(paths):
        try:
            im = Image.open(p)
            im.load()
        except Exception as e:
            print(f"  跳过 {p.name}：{type(e).__name__}")
            continue
        for pc in (crops_of(im, ncrop, csize, rng) if want_crops else [im]):
            px = pc.size[0] * pc.size[1]
            if buf and (pc.size != buf[0].size
                        or (len(buf) + 1) * px > px_budget):
                flush()
            buf.append(pc)
            bidx.append(k)
        if (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(paths)}  {el:.0f}s  "
                  f"剩约 {el/(k+1)*(len(paths)-k-1):.0f}s")
    flush()
    feat = np.concatenate(F)
    logit = np.concatenate(L)
    np.savez_compressed(cp, feat=feat, logit=logit, idx=np.array(I))
    print(f"  -> {cp.name}  n={len(feat)}  {time.time()-t0:.0f}s")
    return feat, logit, I


# ══════════════════════════════════════════════════════════════════════
def boot_std(fn, groups, B, rng):
    """按**图**（不是按裁块）有放回重抽 B 次，返回指标的标准差。

    按图重抽而非按裁块：同一张图的 10 个裁块是相关的，按裁块抽会把
    噪声地板压低，读出来的差异就会显得比实际显著 —— 那正是我们要防的。
    """
    g = np.array(groups)
    uniq = np.unique(g)
    idx_of = {u: np.where(g == u)[0] for u in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_of[u] for u in pick])
        out.append(fn(sel))
    return float(np.std(out))


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="名字=目录")
    ap.add_argument("--real", default=str(root / "laion_real"))
    ap.add_argument("--prompts", default=str(root / "eval_prompts.json"))
    ap.add_argument("--cache", default=str(root / "incep_cache"))
    ap.add_argument("--out", default=str(root / "std_table.json"))
    ap.add_argument("--res", type=int, default=None,
                    help="只取该分辨率；不给则取每个 idx 的最大分辨率")
    ap.add_argument("--patch-only", action="store_true",
                    help="P0：只算 KIDp / ISp（不需要 FID 的 sqrtm，快）")
    ap.add_argument("--clip", action="store_true", help="加算 CLIP score")
    ap.add_argument("--crops", type=int, default=CROPS)
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE)
    ap.add_argument("--crop-seed", type=int, default=CROP_SEED)
    ap.add_argument("--kid-subset", type=int, default=None,
                    help="默认 min(1000, n)；两个臂必须相同")
    ap.add_argument("--kid-subsets", type=int, default=100)
    ap.add_argument("--boot", type=int, default=100, help="KID/IS 的 bootstrap 次数")
    ap.add_argument("--boot-fid", action="store_true",
                    help="给 FID 也做 bootstrap（每次一个 2048² sqrtm，很慢）")
    ap.add_argument("--real-split", default=None, choices=["tune", "eval"],
                    help="tune 开发用 / eval 只在最后出表时用一次")
    ap.add_argument("--px-budget", type=int, default=32_000_000,
                    help="一批的总像素上限；4096² 约两张，1024² 约三十张")
    ap.add_argument("--inception-weights", default=None)
    ap.add_argument("--allow-torchvision", action="store_true")
    ap.add_argument("--features-only", action="store_true")
    a = ap.parse_args()

    arms = {}
    for s in a.arms:
        k, v = s.split("=", 1)
        arms[k] = Path(v if "/" in v else str(root / v))
    for k, v in arms.items():
        if not v.exists():
            sys.exit(f"臂 {k} 的目录不存在：{v}")

    print(f"\n裁块：{a.crops} 个 {a.crop_size}² 随机块 / 图，seed={a.crop_seed}"
          f"   （AccDiffusion / FAM / PixelRush 三家一致，见文件头）")
    print(f"参考侧：真图**整图**，FID 与 FIDp 共用 —— 由 AccDiffusion "
          f"Table 1 的 1× 行（FIDr 58.49 / FIDc 58.08）推出\n")

    tower = load_inception(a.inception_weights, a.allow_torchvision)
    tv = isinstance(tower, TorchvisionInception)

    # ---------- 真图（一套特征，两处用） ----------
    rp = real_files(a.real, a.prompts, a.real_split)
    if not rp:
        sys.exit(f"真图目录空：{a.real}")
    print("抽真图特征：")
    rtag = "real" + (f"_{a.real_split}" if a.real_split else "")
    rf, rl, _ = extract(tower, rp, False, a.cache, rtag, a.px_budget,
                        a.crops, a.crop_size, a.crop_seed)

    # ---------- 各臂 ----------
    res = {}
    for name, d in arms.items():
        fs = arm_files(d, a.res)
        if not fs:
            print(f"⚠ 臂 {name} 在 {d} 下没有匹配的图"
                  f"{'（--res %d）' % a.res if a.res else ''}，跳过")
            continue
        idxs = sorted(fs)
        paths = [fs[i] for i in idxs]
        print(f"\n【{name}】{len(paths)} 张  {d}")
        tag = f"{name}_{a.res or 'max'}"
        print("  整图特征：")
        gf, gl, _ = extract(tower, paths, False, a.cache, tag + "_full",
                            a.px_budget, a.crops, a.crop_size, a.crop_seed)
        print("  裁块特征：")
        pf, pl, pg = extract(tower, paths, True, a.cache, tag + "_crop",
                             a.px_budget, a.crops, a.crop_size, a.crop_seed)
        res[name] = {"n": len(paths), "idxs": idxs, "gf": gf, "gl": gl,
                     "pf": pf, "pl": pl, "pg": pg, "paths": paths}

    if a.features_only:
        print("\n--features-only：特征已落盘，指标没算。")
        return 0
    if not res:
        sys.exit("没有任何臂有图。")

    # KID 的 subset 必须全臂统一，否则数值不可比
    nmin = min([r["n"] for r in res.values()] + [len(rf)])
    sub_full = a.kid_subset or min(1000, nmin)
    sub_patch = a.kid_subset or min(1000, nmin * a.crops, len(rf))
    rng = np.random.default_rng(12345)

    print(f"\nKID subset：整图 {sub_full} / 裁块 {sub_patch}"
          f"（× {a.kid_subsets} 次）  —— 数值随 subset 变，"
          f"**只在同一次运行内横比**")

    # ---------- 样本量体检：说在算之前，免得事后替数字找理由 ----------
    warn = []
    if not a.patch_only and nmin < 1000:
        warn.append(
            f"生成侧只有 {nmin} 张（<1000）。FID/FIDp 的样本协方差在 2048 维上"
            f"远不满秩，数值被系统性抬高（合成实验 d=64 时 n=400 就把 16.0 "
            f"读成 20.9）。**这一档的 FID 不可与发表值比**，"
            f"建议改用 --patch-only。")
    if len(rf) < 1000:
        warn.append(f"真图只有 {len(rf)} 张（<1000），同上。")
    if sub_patch < 100:
        warn.append(f"KID subset 只有 {sub_patch}（<100），无偏估计方差很大。")
    for w in warn:
        print(f"⚠ {w}")

    out = {"config": {
        "crops": a.crops, "crop_size": a.crop_size, "crop_seed": a.crop_seed,
        "res": a.res, "kid_subset_full": sub_full,
        "kid_subset_patch": sub_patch, "kid_subsets": a.kid_subsets,
        "boot": a.boot, "n_real": len(rf), "tower": tower.name,
        "comparable_to_published": not tv,
    }, "arms": {}}

    for name, r in res.items():
        m = {"n": r["n"]}
        # patch 三列
        m["FIDp"] = None if a.patch_only else fid(r["pf"], rf)
        m["KIDp"] = kid(r["pf"], rf, sub_patch, a.kid_subsets,
                        np.random.default_rng(7))
        m["ISp"] = inception_score(r["pl"])
        if not a.patch_only:
            m["FID"] = fid(r["gf"], rf)
            m["KID"] = kid(r["gf"], rf, sub_full, a.kid_subsets,
                           np.random.default_rng(7))
            m["IS"] = inception_score(r["gl"])

        # bootstrap 噪声地板（按图重抽）
        if a.boot:
            pg = r["pg"]
            # 每次 bootstrap 内只用 10 个子集（快），且 rng 种子固定 ——
            # 子集抽样噪声因此在各次重抽间共模，剩下的就是**图重抽**的噪声，
            # 正是我们要的那一项。
            m["KIDp_sd"] = boot_std(
                lambda s: kid(r["pf"][s], rf, sub_patch, 10,
                              np.random.default_rng(7)), pg, a.boot, rng)
            m["ISp_sd"] = boot_std(
                lambda s: inception_score(r["pl"][s]), pg, a.boot, rng)
            if not a.patch_only:
                fg = list(range(r["n"]))
                m["IS_sd"] = boot_std(
                    lambda s: inception_score(r["gl"][s]), fg, a.boot, rng)
                if a.boot_fid:
                    m["FIDp_sd"] = boot_std(
                        lambda s: fid(r["pf"][s], rf), pg,
                        max(10, a.boot // 10), rng)
        out["arms"][name] = m

    # ---------- CLIP ----------
    if a.clip:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from clip_score import load_clip
            clip = load_clip()
            pr = {}
            pp = Path(a.prompts)
            if pp.exists():
                for x in json.loads(pp.read_text()):
                    pr[int(x["idx"])] = x.get("prompt") or x.get("caption", "")
            for name, r in res.items():
                sc = [clip.score(Image.open(p).convert("RGB"), pr[i])
                      for i, p in zip(r["idxs"], r["paths"]) if i in pr]
                out["arms"][name]["CLIP"] = float(np.mean(sc)) if sc else None
                out["arms"][name]["CLIP_n"] = len(sc)
        except SystemExit:
            print("⚠ CLIP 塔拿不到，跳过该列（其余列不受影响）")

    # ---------- 输出 ----------
    cols = (["n", "KIDp", "ISp"] if a.patch_only else
            ["n", "FID", "KID", "IS", "FIDp", "KIDp", "ISp"])
    if a.clip:
        cols.append("CLIP")
    print(f"\n{'臂':<22}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (22 + 12 * len(cols)))
    for name, m in out["arms"].items():
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append("      -" if v is None else
                         (f"{v:>12d}" if c == "n" else
                          (f"{v:>12.4f}" if "KID" in c else f"{v:>12.2f}")))
        print(f"{name:<22}" + "".join(cells))
    if a.boot:
        print(f"\nbootstrap 标准差（按图重抽 {a.boot} 次）：")
        for name, m in out["arms"].items():
            bits = [f"{k[:-3]} ±{m[k]:.4f}" for k in m if k.endswith("_sd")]
            print(f"  {name:<20}" + "   ".join(bits))
        print("  **判据只在差值 > 2× 这里的标准差时才算数**（文件头第三节）。")

    Path(a.out).write_text(json.dumps(
        {k: v for k, v in out.items()}, ensure_ascii=False, indent=2,
        default=lambda o: None))
    print(f"\n-> {a.out}")

    if tv:
        print("\n⚠️⚠️ 特征塔是 torchvision 版：上面这些数**只能自比**，"
              "不可与任何发表值比较（P1 复现判据作废）。")
    if a.patch_only:
        print("\n§10.7 判据一：C(MultiDiffusion) 对 A(NPA) 的 KIDp 优势 "
              "≥0.0005 **且** ISp 优势 ≥0.2 -> 缺口复现，进 P1；否则方向判死。")
        print("§10.7 判据二：令 g=C−A、b=B−A。b≥0.6g -> 缺口住在边界"
              "（天花板低，重估）；b≤0.3g -> 住在上下文（可攻，要的就是这格）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
