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

# SDXL 自带 openai/clip-vit-large-patch14 的文本塔，视觉塔不一定在缓存里。
# 按优先级试，把缓存里有的那个用上；都没有就明确报错，不要静默换模型。
CANDIDATES = [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
]


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    errs = []
    for mid in CANDIDATES:
        try:
            m = CLIPModel.from_pretrained(mid).eval().to("cuda")
            p = CLIPProcessor.from_pretrained(mid)
            print(f"用 {mid}")
            return mid, m, p
        except Exception as e:                       # 缓存里没有就换下一个
            errs.append(f"  {mid}: {type(e).__name__}")
    print("缓存里没有任何完整的 CLIP 模型（文本塔不够，要视觉塔）：")
    print("\n".join(errs))
    print("\n在能上外网的机器上抓一个，拷进 $HF_HOME/hub 即可：")
    print('  python -c "from transformers import CLIPModel, CLIPProcessor; '
          "m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); "
          'CLIPProcessor.from_pretrained(m)"')
    sys.exit(1)


@torch.no_grad()
def score(model, proc, img, text):
    """CLIP score = 100 * max(0, cos(图嵌入, 文本嵌入))，沿用通行定义。

    CLIP 视觉塔输入是 224²。4096² 直接喂进去会被缩 18 倍 —— 这正是
    FID 看不见幻影的同一个原因。但这里问的是"整幅图还符不符合 prompt"，
    是全局语义，缩放不影响这个问法。**不要拿它去测幻影。**
    """
    inp = proc(text=[text], images=img, return_tensors="pt",
               padding=True, truncation=True).to("cuda")
    out = model(**inp)
    ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float(100.0 * max(0.0, (ie * te).sum().item()))


def rows(d):
    p = Path(d) / "manifest.jsonl"
    if not p.exists():
        sys.exit(f"没有 {p}")
    return {json.loads(l)["idx"]: json.loads(l) for l in p.open()}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--res", type=int, nargs="+", default=[4096, 1024])
    a = ap.parse_args()

    A, B = rows(a.base), rows(a.new)
    idxs = sorted(set(A) & set(B))
    _, model, proc = load_clip()

    acc = {r: {"base": [], "new": [], "cat": [], "applied": []} for r in a.res}
    for i in idxs:
        for r in a.res:
            fa, fb = A[i]["files"].get(str(r)), B[i]["files"].get(str(r))
            if not fa or not fb:
                continue
            t = A[i]["prompt"]
            acc[r]["base"].append(score(model, proc, Image.open(Path(a.base) / fa).convert("RGB"), t))
            acc[r]["new"].append(score(model, proc, Image.open(Path(a.new) / fb).convert("RGB"), t))
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
