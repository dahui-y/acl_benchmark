"""核对 prompt 消融的结果。一条命令，回答三件事。

1. 两次运行的 1024² 基图是不是【逐像素一致】。
   这是整个实验的前提：基阶段的 prompt 没被换过，所以基图必须完全相同。
   若不同，说明我那个"按 latent 尺寸判断阶段"的补丁在基阶段就生效了，
   实验作废，结论不能用。

2. 两张 4096² 上各有几个 person。检测器参数在 seed 77 那张上定死之后
   【没有再动过】—— 冻结是纪律，不是形式，否则跑出来的数字是调出来的。

3. 出一张能直接发出来的并排对比图。

    python scalediff_probe/diag_check.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector, annotate     # noqa: E402

D = Path(os.environ.get("SD_OUT", "./scalediff_out")) / "diag" / "prompt"
SEED = 77


def main():
    b_keep = Image.open(D / f"prompt_keep_s{SEED}_1024.png").convert("RGB")
    b_nos = Image.open(D / f"prompt_nosubj_s{SEED}_1024.png").convert("RGB")
    d = np.abs(np.asarray(b_keep, np.int16) - np.asarray(b_nos, np.int16))
    print(f"基图一致性: 最大差 {d.max()}  平均差 {d.mean():.4f}   "
          f"{'一致 ✓' if d.max() == 0 else '!! 不一致，实验前提不成立'}")

    det = Detector(box_thr=0.30, text_thr=0.25)
    panels, counts = [], {}
    for label in ("keep", "nosubj"):
        im = Image.open(D / f"prompt_{label}_s{SEED}_4096.png").convert("RGB")
        det.dropped_aspect = det.dropped_small = 0
        b, s = det.detect(im, "person", 1024, 512, 0.40, 2.0, 8.0)
        counts[label] = len(b)
        print(f"\n{label:<8} 4096²  person x{len(b)}   "
              f"(长宽比丢 {det.dropped_aspect}，尺度门槛丢 {det.dropped_small})")
        for j, (bb, ss) in enumerate(sorted(zip(b, s), key=lambda z: -z[1])):
            cx, cy = (bb[0] + bb[2]) / 2 / im.width, (bb[1] + bb[3]) / 2 / im.height
            print(f"    #{j:<2} ({cx:.3f}, {cy:.3f})  "
                  f"{bb[2]-bb[0]:.0f}x{bb[3]-bb[1]:.0f}px  {ss:.2f}")
        p = annotate(im, b, s, view=1100)
        lab = Image.new("RGB", (p.width, p.height + 30), (255, 255, 255))
        lab.paste(p, (0, 30))
        ImageDraw.Draw(lab).text((8, 8), f"{label}  40962  person x{len(b)}", fill=(0, 0, 0))
        panels.append(lab)

    sheet = Image.new("RGB", (sum(p.width for p in panels) + 12,
                              max(p.height for p in panels)), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 12
    sheet.save(D / "E_prompt_ablation.png")

    print(f"\nkeep {counts['keep']}  ->  nosubj {counts['nosubj']}")
    print("nosubj 明显更少 => 病因在 cross-attention 通路（文本条件被送到了每个位置）")
    print("两者相当       => 病因不在 prompt，方法方向要换")
    print(f"\n发这一张：{D/'E_prompt_ablation.png'}")


if __name__ == "__main__":
    main()
