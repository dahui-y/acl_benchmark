"""查清两条管线的 1024² 为什么不同（自检 0/120 逐字节相同）。

嫌疑按便宜程度排：

  (1) **VAE tiling**：laion_hi_run 开了 `pipe.vae.enable_tiling()`（照抄
      run_one 的写法），base_run 没开。分块解码与整图解码的像素有微小
      数值差 —— latent 完全相同也会字节不同。
  (2) ScaleDiff 管线第一级注册的 attention processor 让数值路径不同。
  (3) RNG 消耗路径不同（应当不会：两边都钉了全局种子 + generator(77)）。

做法：用**普通 SDXL 管线**在 tiling 开/关两种状态下各出同一条 prompt，
与已有的两张 1024²（base_run 的、laion_hi 的）四方对 md5，
再补一手**像素差**（md5 不同不代表内容不同，PNG 元数据都可能搅局）：

  plain+tiling == hi 版   -> 就是 tiling；base_run 那 560 张内容没问题，
                              evf 不受影响，只是解码路径不同。
  仍不同但最大像素差 < 3   -> 数值路径差异（fp 顺序/attn processor），
                              内容等同，evf 仍可用；delta 继续用同跑基图。
  最大像素差大、图看着不同 -> **内容真的不同**，第一级不是"原版 SDXL"，
                              §基图=真值 的推理要重审。

    python scalediff_probe/base_diag.py            # 默认取 laion_hi 第一条
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                     # noqa: E402
from base_run import needs_fp16_variant, CKPT    # noqa: E402


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]


def pixdiff(p1, p2):
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(p1).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(p2).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None, None
    d = np.abs(a - b)
    return int(d.max()), float(d.mean())


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "laion_hi"))
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()

    hi = Path(a.hi)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    r = rows[0] if a.idx is None else next(x for x in rows if x["idx"] == a.idx)
    idx, prompt = r["idx"], r["prompt"]
    p_hi = hi / (r["files"].get("1024") or r["files"].get(1024))
    p_base = Path(a.base) / f"{idx:05d}.png"
    print(f"idx={idx}  prompt: {prompt[:64]}")

    import torch
    from diffusers import StableDiffusionXLPipeline
    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = StableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    out = {}
    for tiling in (False, True):
        if tiling:
            pipe.vae.enable_tiling()
        else:
            pipe.vae.disable_tiling()
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        im = pipe(prompt, negative_prompt=NEGATIVE, height=1024, width=1024,
                  generator=g, num_inference_steps=50,
                  guidance_scale=7.5).images[0]
        p = hi / f"diag_{idx:05d}_plain_{'tile' if tiling else 'notile'}.png"
        im.save(p)
        out[tiling] = p

    print("\nmd5：")
    files = {"base_run 版": p_base, "laion_hi 版": p_hi,
             "plain 不分块": out[False], "plain 分块": out[True]}
    for k, v in files.items():
        print(f"  {k:<14} {md5(v) if Path(v).exists() else '(缺文件)'}")

    print("\n像素差（max / mean）：")
    pairs = [("plain不分块 vs base_run", out[False], p_base),
             ("plain分块   vs laion_hi", out[True], p_hi),
             ("plain不分块 vs laion_hi", out[False], p_hi),
             ("base_run    vs laion_hi", p_base, p_hi)]
    for name, x, y in pairs:
        if Path(x).exists() and Path(y).exists():
            mx, mn = pixdiff(x, y)
            print(f"  {name:<26} max={mx}  mean={mn:.3f}")

    print("""
判读：
  plain分块 == laion_hi（md5 或 max<=2）      -> 原因就是 VAE tiling，
      base_run 560 张内容没问题，evf 不受影响。
  都不等但 max 很小（<=3）                    -> 数值路径差异，内容等同；
      delta 继续用同跑基图，560 张也仍可用。
  max 大、肉眼可见不同                        -> **内容真的不同**，
      ScaleDiff 第一级不是原版 SDXL，'基图=真值'要重审。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
