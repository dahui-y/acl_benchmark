"""把混合权重图叠在 4096² 输出和检测框上，判断变差的那几行是哪种失效。

要分开的两个解释（它们指向完全不同的修法）：

  A 去主体 prompt 泄漏 —— 摘除规则把暗示主体的词留下了，模型照样在背景里
    画一个。若成立，多出来的物体会落在**混合权重低（用去主体嵌入）的远景**里。
    修法：改 strip_subject。

  B 接缝伪影 —— 混合权重是一张平滑但任意的场，在过渡带上模型拿到的条件既不是
    完整 prompt 也不是去主体 prompt。若成立，多出来的物体会**贴着过渡带**
    （权重 0.3~0.7 的那一圈）。修法：改混合方式（二值化+羽化，或换更陡的映射）。

为什么这个诊断是必要的：26_bridge 去掉了 "bridge" 这个名词，桥反而从 1 座
变成 2 座。**A 解释不了这个**（去掉词，物体多了），所以 B 的嫌疑更大 ——
但要看图才能定。

    python scalediff_probe/method_batch.py --s 1 --only structure portrait   # 先补存 map
    python scalediff_probe/map_check.py --idx 26 27 23
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector          # noqa: E402


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--idx", type=int, nargs="+", default=[26, 27, 23])
    ap.add_argument("--view", type=int, default=1400)
    a = ap.parse_args()

    new, base = Path(a.new), Path(a.base)
    R = {json.loads(l)["idx"]: json.loads(l) for l in (new / "manifest.jsonl").open()}
    det = Detector()

    for i in a.idx:
        r = R.get(i)
        if r is None:
            print(f"[{i}] manifest 里没有")
            continue
        tag = f"{r['idx']:02d}_{r['cat']}_s{r['seed']}"
        mp = new / f"{tag}_map.npy"
        if not mp.exists():
            print(f"[{i}] 没有 {mp.name} —— 先重跑 method_batch 补存 map")
            continue
        m = np.load(mp)                                  # (64, 64)，[0,1]
        f = r["files"].get("4096") or r["files"][max(r["files"], key=int)]

        for which, d in (("new", new), ("base", base)):
            if which == "base":
                rb = {json.loads(l)["idx"]: json.loads(l)
                      for l in (base / "manifest.jsonl").open()}[i]
                fn = rb["files"].get("4096")
            else:
                fn = f
            im = Image.open(d / fn).convert("RGB")
            b, s = det.detect(im, r["subject"])
            sc = a.view / im.width
            v = im.resize((a.view, int(im.height * sc)), Image.LANCZOS)

            if which == "new":
                # 权重图：红 = 用去主体嵌入（低权重），过渡带描一圈青色
                mm = np.array(Image.fromarray((m * 255).astype(np.uint8))
                              .resize(v.size, Image.BILINEAR)) / 255.0
                ov = np.array(v).astype(np.float32)
                ov[..., 0] = np.clip(ov[..., 0] + 90 * (1 - mm), 0, 255)   # 低权重区偏红
                band = ((mm > 0.30) & (mm < 0.70))                          # 过渡带
                ov[band] = 0.45 * ov[band] + 0.55 * np.array([0, 220, 220])
                v = Image.fromarray(ov.astype(np.uint8))

            dr = ImageDraw.Draw(v)
            for bb, ss in zip(b, s):
                dr.rectangle([bb[0] * sc, bb[1] * sc, bb[2] * sc, bb[3] * sc],
                             outline=(255, 255, 0), width=3)
                dr.text((bb[0] * sc + 4, bb[1] * sc - 12), f"{ss:.2f}",
                        fill=(255, 255, 0))
            p = new / f"M_{tag}_{which}.png"
            v.save(p)
            print(f"[{i}] {which:<4} {r['subject']} x{len(b)}   -> {p.name}")

        lo = float((m < 0.30).mean())
        bandf = float(((m > 0.30) & (m < 0.70)).mean())
        print(f"      cov(>0.5)={float((m>0.5).mean()):.1%}   "
              f"低权重区={lo:.1%}   过渡带={bandf:.1%}")
        print(f"      去主体 prompt: {r['alt_prompt']!r}\n")

    print("看什么：new 那张图里，多出来的黄框落在【青色过渡带】上 -> 接缝伪影（改混合）；")
    print("        落在【偏红的远景】里 -> 去主体 prompt 泄漏（改摘除规则）。")


if __name__ == "__main__":
    main()
