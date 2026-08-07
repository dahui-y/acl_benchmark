"""核对诊断实验的结果。三个实验共用一个入口。

    python scalediff_probe/diag_check.py prompt    # keep vs nosubj
    python scalediff_probe/diag_check.py kv        # x1 vs x2
    python scalediff_probe/diag_check.py res       # 2048 / 4096 / 8192

检测器参数在 seed 77 那张阳性对照上定死之后【不再改动】。冻结是纪律：
参数是看着一张图调出来的，之后再动就等于在结果上调参数。

prompt 那一组还会额外校验两次运行的 1024² 基图是否逐像素一致 —— 基阶段的
条件没被换过，所以必须完全相同；不同就说明按 latent 尺寸判断阶段的补丁在
基阶段就生效了，对照作废。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector, annotate     # noqa: E402

ROOT = Path(os.environ.get("SD_OUT", "./scalediff_out")) / "diag"
SEED = 77

SPECS = {
    "prompt": ("prompt", [("keep", "prompt_keep_s{s}_{r}.png", 4096),
                          ("nosubj", "prompt_nosubj_s{s}_{r}.png", 4096)]),
    "kv":     ("kv", [("KV x1 (原版)", "kv_x1_s{s}_{r}.png", 4096),
                      ("KV x2 (邻域加倍)", "kv_x2_s{s}_{r}.png", 4096)]),
    "res":    ("res", [("2048", "res_stage1_s{s}_{r}.png", 2048),
                       ("4096", "res_stage2_s{s}_{r}.png", 4096),
                       ("8192", "res_stage3_s{s}_{r}.png", 8192)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(SPECS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subject", default="person")
    ap.add_argument("--view", type=int, default=1100)
    a = ap.parse_args()

    sub, items = SPECS[a.exp]
    d = ROOT / sub

    if a.exp == "prompt":
        try:
            b1 = np.asarray(Image.open(d / f"prompt_keep_s{a.seed}_1024.png").convert("RGB"), np.int16)
            b2 = np.asarray(Image.open(d / f"prompt_nosubj_s{a.seed}_1024.png").convert("RGB"), np.int16)
            m = np.abs(b1 - b2).max()
            print(f"基图一致性: 最大差 {m}  {'一致 ✓' if m == 0 else '!! 不一致，对照作废'}\n")
        except FileNotFoundError:
            pass

    det = Detector(box_thr=0.30, text_thr=0.25)
    panels, summary = [], []
    for label, pat, res in items:
        p = d / pat.format(s=a.seed, r=res)
        if not p.is_file():
            print(f"[缺] {p.name}")
            continue
        im = Image.open(p).convert("RGB")
        det.dropped_aspect = det.dropped_small = 0
        b, s = det.detect(im, a.subject, 1024, 512, 0.40, 2.0, 8.0)
        # 主角是画面里最大的那个；其余按定义是基图里没有的
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)
        big = int(areas.argmax()) if len(b) else -1
        print(f"{label:<18} {im.width}²  {a.subject} x{len(b)}   "
              f"(长宽比丢 {det.dropped_aspect}，尺度门槛丢 {det.dropped_small})")
        for j, (bb, ss) in enumerate(sorted(zip(b, s), key=lambda z: -z[1])):
            cx, cy = (bb[0] + bb[2]) / 2 / im.width, (bb[1] + bb[3]) / 2 / im.height
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            note = "  <- 最大，多半是主角" if len(b) and (w * h) == areas[big] else ""
            print(f"    ({cx:.3f}, {cy:.3f})  {w:.0f}x{h:.0f}px  w/h={w/max(h,1):.2f}  {ss:.2f}{note}")
        summary.append((label, len(b), max(len(b) - 1, 0), det.dropped_small))
        pan = annotate(im, b, s, view=a.view)
        lab = Image.new("RGB", (pan.width, pan.height + 28), (255, 255, 255))
        lab.paste(pan, (0, 28))
        ImageDraw.Draw(lab).text((8, 7), f"{label}   {im.width}2   x{len(b)}", fill=(0, 0, 0))
        panels.append(lab)

    if panels:
        sheet = Image.new("RGB", (sum(p.width for p in panels) + 12 * (len(panels) - 1),
                                  max(p.height for p in panels)), (255, 255, 255))
        x = 0
        for p in panels:
            sheet.paste(p, (x, 0))
            x += p.width + 12
        out = d / f"E_{a.exp}.png"
        sheet.save(out)
        print(f"\n{'条件':<20}{'总数':>6}{'除主角外':>10}{'尺度门槛丢掉':>14}")
        for label, n, extra, small in summary:
            print(f"{label:<20}{n:>6}{extra:>10}{small:>14}")
        print(f"\n发这一张：{out}")


if __name__ == "__main__":
    main()
