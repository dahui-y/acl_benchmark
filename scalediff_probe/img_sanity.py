"""出图体检：把 NaN/溢出/全黑的坏图挡在测量之外。

为什么需要它（2026-08-14 血的教训）：
AccDiffusion 在 `lowvram=True` 下**关掉 VAE 的 fp32 升位**
（accdiffusion_sdxl.py:1247 `needs_upcasting = False  # use
madebyollin/sdxl-vae-fp16-fix in lowvram mode!`），前提是你换了
fp16-fix 的 VAE。用原版 SDXL VAE 就会 fp16 溢出出 NaN，
postprocess 里 `(images*255).round().astype("uint8")` 把 NaN 铸成
垃圾值 —— 只在 stderr 留一行 RuntimeWarning，图照样存盘。

**一张坏图以"对手的正常输出"身份进入闸门测量，足以让整个结论作废。**
所以任何跑批之后、计数之前，都先过这一关。

判据（三条，命中任一即可疑）：
    纯黑占比 > 30%      NaN 铸 uint8 在多数平台上落到 0
    整图标准差 < 4      信息量塌了（纯色/近纯色）
    单通道饱和 > 40%    fp16 溢出常见的形态（某通道全 255 或全 0）

    python scalediff_probe/img_sanity.py $SD_OUT/parti_acc
    python scalediff_probe/img_sanity.py $SD_OUT/parti_acc --rm   # 删坏图与其 manifest 行
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def check(p, thumb=512):
    im = Image.open(p).convert("RGB")
    a = np.asarray(im.resize((thumb, thumb), Image.BILINEAR), np.float32)
    black = float((a.sum(-1) == 0).mean())
    std = float(a.std())
    sat = max(float((a[..., c] >= 254).mean()) for c in range(3))
    zero = max(float((a[..., c] <= 1).mean()) for c in range(3))
    bad = []
    if black > 0.30:
        bad.append(f"纯黑 {black:.0%}")
    if std < 4:
        bad.append(f"标准差 {std:.1f}")
    if max(sat, zero) > 0.40:
        bad.append(f"单通道饱和 {max(sat, zero):.0%}")
    return bad, dict(black=black, std=std, sat=max(sat, zero))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--rm", action="store_true",
                    help="删掉坏图及其 manifest 行，好让跑批脚本重跑那几条")
    a = ap.parse_args()

    d = Path(a.run)
    files = sorted(d.glob("*.png"))
    if not files:
        raise SystemExit(f"{d} 里没有 png")

    bad_idx, n_ok = set(), 0
    for p in files:
        bad, st = check(p)
        if bad:
            print(f"  ⚠ {p.name:<22} {'  '.join(bad)}")
            if p.name[:5].isdigit():
                bad_idx.add(int(p.name[:5]))
        else:
            n_ok += 1
    print(f"\n{len(files)} 张：正常 {n_ok}，可疑 {len(files) - n_ok}"
          f"（涉及 {len(bad_idx)} 个 idx：{sorted(bad_idx)}）")

    if not bad_idx:
        print("体检通过，可以进入计数。")
        return
    print("\n**这些 idx 不能进入测量。** 常见成因：VAE 在 fp16 下未升位"
          "（AccDiffusion 的 lowvram 分支会关掉升位）。")
    if not a.rm:
        print("加 --rm 可删掉它们（含 manifest 行），跑批脚本会自动重跑。")
        return

    mp = d / "manifest.jsonl"
    if mp.exists():
        keep = [l for l in mp.read_text().splitlines()
                if l.strip() and json.loads(l).get("idx") not in bad_idx]
        mp.write_text("\n".join(keep) + ("\n" if keep else ""))
        print(f"manifest 保留 {len(keep)} 行")
    n = 0
    for i in bad_idx:
        for p in d.glob(f"{i:05d}_*"):
            p.unlink()
            n += 1
    print(f"删掉 {n} 个文件，涉及 idx {sorted(bad_idx)} —— 重跑跑批脚本即可补上。")


if __name__ == "__main__":
    main()
