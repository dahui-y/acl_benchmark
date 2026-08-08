"""否决测试：干预有没有伤到 prompt 保真度。

为什么这是最该先跑的一个数：
    我们的干预把画面上大部分位置的文本嵌入换成了"去主体版本"。它最可能的
    死法不是幻影没修干净，而是**图不再是 prompt 描述的那张图** —— 幻影没了，
    但主体的语境、景物的呼应也一起淡了。
    "细节有没有塌"这个问题挂了很多轮，一直只有肉眼判断。这就是它的定量答案。

为什么它比 FID 先跑：
    CLIP score 只比"图和它自己的 prompt"，**不需要真图参考集、不需要下载
    LAION、不需要 InceptionV3**。图已经在磁盘上了。几分钟出结果。
    而 FID 那一套要数 GB 的外网数据，服务器拿不到。

判据（跑之前定死）：
    · CLIP score 下降 < 0.5% ...... 过。方法不伤保真度，可以往下走
    · 下降 0.5% ~ 2% ............... 黄灯。要在论文里报，并配细节指标
    · 下降 > 2% .................... **否决**。先修方法，不要碰 FID 那 43 小时

同时报 1024²（未被干预，应当逐字节相同 -> 分数必须一模一样）作为自检：
这一档若有差，说明脚本或图对错了，下面的数不用看。

    python scalediff_probe/clip_score.py
    python scalediff_probe/clip_score.py --res 4096 2048 1024
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 两种来源都试。服务器上 hf-mirror 也不通了，所以**不下载任何权重**：
#   1) HF 格式的 CLIPModel（缓存里若有就用）
#   2) 缓存里已有的 open_clip 权重
#      laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup（1.4 GB）
#      那个仓库只有 open_clip_model.safetensors，没有 config —— 不要紧，
#      open_clip 的架构按名字内建，BPE 词表也打包在 wheel 里，
#      所以只需要 pip 装一个纯 Python 包：
#          pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open_clip_torch
#      它是 LAION-2B 上训的，和 ScaleDiff / AccDiffusion v2 用的塔同源。
HF_CANDIDATES = [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
]
OPENCLIP_CANDIDATES = [
    ("convnext_large_d_320",
     "models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"),
]


class HFClip:
    def __init__(self, mid, model, proc):
        self.name, self.m, self.p = mid, model, proc

    @torch.no_grad()
    def score(self, img, text):
        inp = self.p(text=[text], images=img, return_tensors="pt",
                     padding=True, truncation=True).to("cuda")
        o = self.m(**inp)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        return float(100.0 * max(0.0, (ie * te).sum().item()))


class OpenClip:
    def __init__(self, arch, ckpt):
        import open_clip
        self.name = f"open_clip:{arch}"
        self.m, _, self.pre = open_clip.create_model_and_transforms(
            arch, pretrained=str(ckpt))
        self.m = self.m.eval().to("cuda")
        self.tok = open_clip.get_tokenizer(arch)

    @torch.no_grad()
    def score(self, img, text):
        im = self.pre(img).unsqueeze(0).to("cuda")
        tt = self.tok([text]).to("cuda")
        ie = self.m.encode_image(im)
        te = self.m.encode_text(tt)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        return float(100.0 * max(0.0, (ie * te).sum().item()))


def load_clip():
    """CLIP score = 100 * max(0, cos(图嵌入, 文本嵌入))，沿用通行定义。

    CLIP 视觉塔输入是 224²/320²。4096² 喂进去会被缩十几倍 —— 这正是 FID
    看不见幻影的同一个原因。但这里问的是"整幅图还符不符合 prompt"，
    是全局语义，缩放不影响这个问法。**不要拿它去测幻影。**
    """
    errs = []
    for mid in HF_CANDIDATES:
        try:
            from transformers import CLIPModel, CLIPProcessor
            c = HFClip(mid, CLIPModel.from_pretrained(mid).eval().to("cuda"),
                       CLIPProcessor.from_pretrained(mid))
            print(f"用 {mid}")
            return c
        except Exception as e:
            errs.append(f"  {mid}: {type(e).__name__}")

    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for arch, repo in OPENCLIP_CANDIDATES:
        cks = sorted((hub / repo / "snapshots").glob("*/open_clip_model.safetensors")) \
            if (hub / repo).exists() else []
        if not cks:
            errs.append(f"  {repo}: 缓存里没有")
            continue
        try:
            c = OpenClip(arch, cks[0])
            print(f"用 {c.name}   {cks[0]}")
            return c
        except ImportError:
            errs.append(f"  {repo}: 权重在，但没装 open_clip_torch")
        except Exception as e:
            errs.append(f"  {repo}: {type(e).__name__}: {e}")

    print("拿不到可用的 CLIP：")
    print("\n".join(errs))
    print("\n最省事的一条（权重已在缓存里，只差一个纯 Python 包）：")
    print("  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open_clip_torch")
    print("\n若 pip 也不通，就把图缩到 1024² 拷到 Mac 上算 —— "
          "CLIP 视觉塔输入只有 224²，缩放不影响这个问法。")
    sys.exit(1)


def rows(d):
    p = Path(d) / "manifest.jsonl"
    if not p.exists():
        sys.exit(f"没有 {p}")
    return {json.loads(l)["idx"]: json.loads(l) for l in p.open()}


def sanity(clip, A, d):
    """尺子的阳性对照 —— 和检测器那个是同一套纪律。

    问题不是"CLIP score 算得对不对"，是"它有多少动态范围"。
    +0.02% 这个结果，只有在尺子能分辨"配对 vs 错配"时才有意义：

        错配掉到 ~20  -> 量程约 10 分，+0.02% 是真的"没变"
        错配还是 ~29  -> 尺子饱和了，+0.02% 什么也没说

    错配用【环形移位】而不是随机打乱：确定性，同输入同输出，
    且保证每张图都配到别人的 prompt。同时报同类内错配 ——
    "一个登山者"配"一个冲浪者"比配"一张猫脸"难分，同类内的落差
    才是这把尺子在我们这个任务上的真实分辨力。
    """
    ii = sorted(A)
    imgs, texts, cats = [], [], []
    for i in ii:
        f = A[i]["files"].get("4096")
        if not f:
            continue
        imgs.append(Image.open(d / f).convert("RGB"))
        texts.append(A[i]["prompt"])
        cats.append(A[i]["cat"])
    n = len(imgs)

    match = np.array([clip.score(imgs[k], texts[k]) for k in range(n)])
    shift = np.array([clip.score(imgs[k], texts[(k + 1) % n]) for k in range(n)])
    # 同类内错配：同一 cat 里换下一条
    within = []
    for k in range(n):
        same = [j for j in range(n) if cats[j] == cats[k] and j != k]
        within.append(clip.score(imgs[k], texts[same[0]]) if same else np.nan)
    within = np.array(within)

    print(f"\n===== 尺子的阳性对照   n={n}   4096² =====")
    print(f"  配对（图 + 自己的 prompt）      {match.mean():7.3f}")
    print(f"  错配（图 + 下一条 prompt）      {shift.mean():7.3f}"
          f"   落差 {shift.mean()-match.mean():+.3f}")
    print(f"  同类内错配（同 cat 换一条）     {np.nanmean(within):7.3f}"
          f"   落差 {np.nanmean(within)-match.mean():+.3f}")
    gap = match.mean() - shift.mean()
    wgap = match.mean() - np.nanmean(within)
    print(f"\n  逐图判对的比例（配对 > 错配）   "
          f"{(match > shift).mean():.0%}   同类内 {(match > within).mean():.0%}")
    print(f"\n判据: 跨类落差 {gap:.2f} 分。我们量到的干预效应是 "
          f"{0.02*match.mean()/100:+.3f} 分（+0.02%）。")
    if gap < 1.0:
        print("  **落差 <1 分 —— 尺子在这批图上没有量程，+0.02% 不能解读为'没变'**")
    else:
        print(f"  尺子有 {gap:.1f} 分的量程，干预效应比它小 {gap/max(abs(0.02*match.mean()/100),1e-9):.0f} 倍")
        print(f"  -> +0.02% 可以解读为'全局语义没变'。同类内量程 {wgap:.2f} 分，"
              "这才是它在本任务上的真实分辨力。")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--res", type=int, nargs="+", default=[4096, 1024])
    ap.add_argument("--sanity", action="store_true",
                    help="尺子的阳性对照：把图和别人的 prompt 配对，看分数掉多少")
    a = ap.parse_args()

    A, B = rows(a.base), rows(a.new)
    idxs = sorted(set(A) & set(B))
    clip = load_clip()

    if a.sanity:
        sanity(clip, A, Path(a.base))
        return

    acc = {r: {"base": [], "new": [], "cat": [], "applied": []} for r in a.res}
    for i in idxs:
        for r in a.res:
            fa, fb = A[i]["files"].get(str(r)), B[i]["files"].get(str(r))
            if not fa or not fb:
                continue
            t = A[i]["prompt"]
            acc[r]["base"].append(clip.score(Image.open(Path(a.base) / fa).convert("RGB"), t))
            acc[r]["new"].append(clip.score(Image.open(Path(a.new) / fb).convert("RGB"), t))
            acc[r]["cat"].append(A[i]["cat"])
            acc[r]["applied"].append(bool(B[i].get("applied", True)))

    for r in a.res:
        d = acc[r]
        if not d["base"]:
            continue
        ba, ne = np.array(d["base"]), np.array(d["new"])
        cats, app = np.array(d["cat"]), np.array(d["applied"])
        print(f"\n===== {r}²   n={len(ba)} =====")
        print(f"{'cat':<12}{'base':>9}{'new':>9}{'Δ':>8}{'Δ%':>8}")
        print("-" * 46)
        for c in sorted(set(cats)):
            k = cats == c
            print(f"{c:<12}{ba[k].mean():>9.3f}{ne[k].mean():>9.3f}"
                  f"{ne[k].mean()-ba[k].mean():>+8.3f}"
                  f"{100*(ne[k].mean()-ba[k].mean())/ba[k].mean():>+8.2f}%")
        dl = 100 * (ne.mean() - ba.mean()) / ba.mean()
        print(f"{'全部':<11}{ba.mean():>9.3f}{ne.mean():>9.3f}"
              f"{ne.mean()-ba.mean():>+8.3f}{dl:>+8.2f}%")
        # 只看真正被干预的那些行 —— empty 五条走原版路径，混在一起会稀释信号
        k = app
        if k.any() and not k.all():
            db = 100 * (ne[k].mean() - ba[k].mean()) / ba[k].mean()
            print(f"{'仅干预行':<9}{ba[k].mean():>9.3f}{ne[k].mean():>9.3f}"
                  f"{ne[k].mean()-ba[k].mean():>+8.3f}{db:>+8.2f}%   (n={int(k.sum())})")
            dl = db
        if r == 1024:
            same = np.allclose(ba, ne, atol=1e-4)
            print(f"自检: 1024² 未被干预，两边分数应完全相同 -> "
                  + ("一致" if same else "**不一致，脚本或图对错了，上面的数不用看**"))
        else:
            v = ("过（<0.5%）" if dl > -0.5 else
                 "黄灯（0.5~2%），要在论文里报并配细节指标" if dl > -2.0 else
                 "**否决（>2%）—— 先修方法，不要碰 FID 那 43 小时**")
            print(f"判据: {dl:+.2f}%  -> {v}")


if __name__ == "__main__":
    main()
