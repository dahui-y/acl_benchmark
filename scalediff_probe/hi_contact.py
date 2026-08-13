"""把 laion_hi 的 4096² 输出拼成接触表 —— 数字说"没有重复"之后，眼睛复核。

为什么必须看：`delta_content` 是**保守**的量 —— 4096 降到 1024 再数，
主体级的大副本留得住，但**小尺寸的克隆**（柯基图里那种 1/8 大小的小狗）
降采样后可能掉到检测线以下。所以 "delta_content ≈ 0" 有两种可能：
真的没有重复，或者只有小克隆而这把尺子看不见。分辨这两种，看图最便宜。

每格：4096 输出的缩略图 + 本次自带基图的缩略图并排，标 evf / delta 两个值。
先看 C 层（evf 高，最该重复），再扫 delta_raw 最大的几张。

    python scalediff_probe/hi_contact.py --stratum C_high
    python scalediff_probe/hi_contact.py --top-raw 12      # delta_raw 最大的 12 张
    python scalediff_probe/hi_contact.py --hi $SD_OUT/parti_hi   # 触发集：全取，
        # 按预注册标签 scenic -> mid -> flat 分组排（parti_trigger_predictions.json）
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "laion_hi"))
    ap.add_argument("--stratum", default=None,
                    help="LAION 跑用 A_evf0/B_low/C_high/D_undef；"
                         "--idx-file 名单跑出来的 manifest 全是 'trigger'。"
                         "不传时：有 C_high 取 C_high，否则全取")
    ap.add_argument("--top-raw", type=int, default=0,
                    help="改为取 delta_raw 最大的 N 张（跨层）")
    ap.add_argument("--idx", type=int, nargs="*", default=None,
                    help="只看这几张（可疑格放大用，配 --cell 1024 --cols 1）")
    ap.add_argument("--cell", type=int, default=384)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()

    hi = Path(a.hi)
    mani = {json.loads(l)["idx"]: json.loads(l)
            for l in (hi / "manifest.jsonl").open()}
    deltas = {}
    dp = hi / "delta.jsonl"
    if dp.exists():
        for l in dp.open():
            x = json.loads(l)
            deltas[x["idx"]] = x

    # 预注册标签（scenic/mid/flat）—— 有就用来分组排版和标注
    pred_p = Path(__file__).resolve().parent / "parti_trigger_predictions.json"
    pred = (json.loads(pred_p.read_text())["tags"]
            if pred_p.exists() else {})
    ptag = lambda r: pred.get(str(r["idx"]), {}).get("tag", "")

    rows = list(mani.values())
    if a.idx:
        rows = [mani[i] for i in a.idx if i in mani]
        tag = "pick_" + "_".join(str(i) for i in a.idx)
    elif a.top_raw:
        rows = sorted(rows, key=lambda r: -(deltas.get(r["idx"], {})
                                            .get("delta_raw", -99)))[:a.top_raw]
        tag = f"topraw{a.top_raw}"
    elif a.stratum:
        rows = [r for r in rows if r["stratum"] == a.stratum]
        tag = a.stratum
    else:
        sel = [r for r in rows if r["stratum"] == "C_high"]
        if sel:
            rows, tag = sel, "C_high"
        else:
            # --idx-file 名单跑（如 parti_hi）：全取，
            # 按预注册标签分组排：scenic（预测该重复）在前，flat 殿后
            order = {"scenic": 0, "mid": 1, "flat": 2, "": 3}
            rows = sorted(rows, key=lambda r: (order.get(ptag(r), 3), r["idx"]))
            tag = "all"

    if not rows:
        sys.exit("没有符合条件的行")
    from PIL import Image, ImageDraw

    C, cols = a.cell, a.cols
    # 每格：左 4096 缩略、右基图缩略，等宽并排
    cell_w, cap_h, pad = C * 2 + 4, 30, 8
    nrow = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell_w + pad) + pad,
                              nrow * (C + cap_h + pad) + pad), "white")
    dr = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        fhi = r["files"].get("4096") or r["files"].get(4096)
        f1 = r["files"].get("1024") or r["files"].get(1024)
        im_hi = Image.open(hi / fhi).convert("RGB").resize((C, C), Image.LANCZOS)
        im_b = Image.open(hi / f1).convert("RGB").resize((C, C), Image.LANCZOS)
        x = pad + (k % cols) * (cell_w + pad)
        y = pad + (k // cols) * (C + cap_h + pad)
        sheet.paste(im_hi, (x, y))
        sheet.paste(im_b, (x + C + 4, y))
        d = deltas.get(r["idx"], {})
        t = ptag(r)
        dr.text((x + 2, y + C + 2),
                f"[{r['idx']}] {t or r['stratum']}  evf={r.get('evf')}"
                f"  dc={d.get('delta_content','?')}  raw={d.get('delta_raw','?')}",
                fill={"scenic": (180, 0, 0), "flat": (0, 100, 0)}.get(t, "black"))
        dr.text((x + 2, y + C + 16), r["prompt"][:64], fill=(90, 90, 90))
    p = hi / f"contact_{tag}.jpg"
    sheet.save(p, "JPEG", quality=90)
    print(f"写出 {p}   每格：左 = 4096² 缩略，右 = 同跑基图缩略")
    print("看什么：左图里有而右图里没有的**主体副本**（不论大小）。"
          "有 -> delta_content 的保守性漏掉了小克隆，要记录；"
          "没有 -> '4K LAION 无重复'由眼睛第二次确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
