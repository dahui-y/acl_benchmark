"""定性对比图：照 AccDiffusion 的证据格式，把重复的物体用红框圈出来。

**为什么照他们的格式**（读 AccDiffusion.pdf 原文核定，2026-08-14）：
- Fig.6 图注原话：*"**We draw a red box** upon the generated images to
  highlight the repeated objects. Best viewed zoomed in."* —— 框是**手画的**，
  没有检测器、没有度量。
- 全文 user study / human / participant / preference / volunteer **各 0 次**
  —— 他们**没有做人工评测**。
- 正文（Fig.6 下）：*"Considering the fact that **existing quantitative
  metrics are unable to accurately reflect the extent of object
  repetition**, we choose to provide visualizations..."*

即：这条线接受的证据格式就是**带框的定性图 + 标准表**。我们照办。

**版面上我们与他们的差别（有意为之）**：
他们是「一个 prompt × 四种长宽比 × 四个方法」。我们改成
**「一行一个 prompt × 各方法一列」** —— 多几个 prompt 比多几个长宽比
有说服力得多，尤其我们要打的是"他们在更难的 prompt 上没解决"。

**一条他们没有、我们必须有的诚实规则**：
    该臂键**不存在**  -> 打 "not annotated"（还没标，不是干净）
    该臂键是**空表**  -> 打 "no repetition found"（作者已看过，判定干净）
两者绝不能都渲染成留白 —— 留白会被读成"漏画了"。

标注文件格式（坐标一律是 0~1 的图幅比例，与分辨率无关）：

    {
      "331": {
        "_note": "直升机由 1 增至 6；女神像 ScaleDiff 处 4 尊",
        "ScaleDiff":    [[0.05, 0.10, 0.12, 0.09], [0.62, 0.14, 0.10, 0.08]],
        "Ours":         [],
        "AccDiffusion": [[0.28, 0.72, 0.09, 0.14]]
      }
    }

    python scalediff_probe/box_figure.py --idx 331 734 1187
    python scalediff_probe/box_figure.py --idx 734 --crop 0.55 0.60 0.45
    python scalediff_probe/box_figure.py --template --idx 331 734   # 生成待填模板
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

BOX_RGB = (237, 28, 36)      # 与 AccDiffusion 图里的红接近
OURS_BAR = (198, 219, 239)   # 我们那一列的色条（他们用蓝标自己）
OTHER_BAR = (222, 222, 222)
_WARNED = {"note": False}


def load_delta(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return {}
    return {int(json.loads(l)["idx"]): json.loads(l)
            for l in p.open() if l.strip()}


def pick_hi(d, idx):
    """该臂该 idx 的最大分辨率图（排除 1024 基图）。"""
    cands = [p for p in Path(d).glob(f"{idx:05d}_*.png")
             if p.stem.split("_")[1] != "1024"]
    if not cands:
        return None
    return max(cands, key=lambda p: int(p.stem.split("_")[1]))


def draw_boxes(im, boxes, crop, width=5):
    """boxes 是 0~1 图幅比例的 [x, y, w, h]，在 **裁剪后**的图上作画。"""
    W, H = im.size
    dr = ImageDraw.Draw(im)
    cx, cy, s = crop if crop else (0.5, 0.5, 1.0)
    x0, y0 = cx - s / 2, cy - s / 2
    for b in boxes:
        bx, by, bw, bh = b[:4]
        # 全图比例 -> 裁剪窗内比例
        rx, ry = (bx - x0) / s, (by - y0) / s
        rw, rh = bw / s, bh / s
        if rx + rw < 0 or ry + rh < 0 or rx > 1 or ry > 1:
            continue                       # 框被裁掉了，不画半个
        dr.rectangle([rx * W, ry * H, (rx + rw) * W, (ry + rh) * H],
                     outline=BOX_RGB, width=width)


def cell(path, boxes, state, label, C, crop):
    """一格 = 图 + 红框 + 底部色条标签。state: 'ok'/'clean'/'todo'/'missing'"""
    pad = 30
    c = Image.new("RGB", (C, C + pad), "white")
    if path is None:
        dr = ImageDraw.Draw(c)
        dr.rectangle([0, 0, C, C], fill="#eeeeee")
        dr.text((10, C // 2), "image missing", fill="black")
    else:
        im = Image.open(path).convert("RGB")
        if crop:
            cx, cy, s = crop
            W, H = im.size
            h = s / 2
            im = im.crop((int((cx - h) * W), int((cy - h) * H),
                          int((cx + h) * W), int((cy + h) * H)))
        im = im.resize((C, C), Image.LANCZOS)
        draw_boxes(im, boxes, crop)
        c.paste(im, (0, 0))
    dr = ImageDraw.Draw(c)
    bar = OURS_BAR if "our" in label.lower() else OTHER_BAR
    dr.rectangle([0, C, C, C + pad], fill=bar)
    tag = {"ok": f"{len(boxes)} repeated object(s) boxed",
           "clean": "no repetition found (author-checked)",
           "todo": "NOT ANNOTATED YET",
           "missing": ""}[state]
    dr.text((6, C + 4), label, fill="black")
    dr.text((6, C + 16), tag, fill=(150, 0, 0) if state == "todo" else "black")
    return c


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="名字=目录；默认 ScaleDiff / AccDiffusion / Ours")
    ap.add_argument("--boxes", default=None,
                    help="标注 JSON；默认 scalediff_probe/boxes.json")
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--cell", type=int, default=560)
    ap.add_argument("--crop", type=float, nargs=3, default=None,
                    metavar=("CX", "CY", "S"))
    ap.add_argument("--template", action="store_true",
                    help="按 --idx 生成待填标注模板（不出图）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.arms:
        arms = {}
        for s in a.arms:
            k, v = s.split("=", 1)
            arms[k] = Path(v if "/" in v else str(root / v))
    else:
        arms = {"ScaleDiff": root / "parti_hi",
                "AccDiffusion": root / "parti_acc",
                "Ours": root / "parti_v12"}

    bp = Path(a.boxes) if a.boxes else \
        Path(__file__).resolve().parent / "boxes.json"

    if a.template:
        cur = json.loads(bp.read_text()) if bp.exists() else {}
        for i in a.idx:
            cur.setdefault(str(i), {"_note": "",
                                    **{k: [] for k in arms}})
        bp.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
        print(f"模板已写入 {bp}\n"
              f"坐标格式 [x, y, w, h]，全是 0~1 的图幅比例。\n"
              f"**空表 = 作者已看过、判定干净；删掉该键 = 还没标。**\n"
              f"两者在图上会显示成不同的字，别混。")
        return

    boxes_all = json.loads(bp.read_text()) if bp.exists() else {}
    if not boxes_all:
        print(f"⚠ {bp} 不存在或为空 —— 先跑 --template 生成模板并手工填框。\n"
              f"  （AccDiffusion 的框也是手画的，见本文件头的原文引用）\n")

    deltas = {k: load_delta(v) for k, v in arms.items()}
    C = a.cell
    rows = []
    for i in a.idx:
        ann = boxes_all.get(str(i), {})
        cells = []
        for name, d in arms.items():
            p = pick_hi(d, i)
            if name in ann:
                bs = ann[name]
                state = "ok" if bs else "clean"
            else:
                bs, state = [], ("todo" if p else "missing")
            lab = name
            if i in deltas.get(name, {}):
                lab += f"   (Δ{deltas[name][i]['delta']:+d})"
            cells.append(cell(p, bs, state if p else "missing", lab, C, a.crop))
        strip = Image.new("RGB", (C * len(cells), cells[0].height), "white")
        for j, c in enumerate(cells):
            strip.paste(c, (j * C, 0))
        prm = next((deltas[k][i].get("prompt", "") for k in deltas
                    if i in deltas[k]), "")
        note = ann.get("_note", "")
        hdr = Image.new("RGB", (strip.width, 34), "white")
        hd = ImageDraw.Draw(hdr)
        hd.text((6, 4), f'[{i}]  "{prm[:120]}"', fill="black")
        if note:
            # 图最终进英文论文，且 PIL 自带位图字体没有 CJK（中文会变豆腐块），
            # 故渲染时只留 ASCII；JSON 里仍可用中文自己看
            ascii_note = "".join(ch for ch in note if ord(ch) < 128).strip()
            if ascii_note:
                hd.text((6, 19), f"note: {ascii_note[:150]}",
                        fill=(90, 90, 90))
            elif not _WARNED["note"]:
                _WARNED["note"] = True
                print("提示：_note 含非 ASCII，图上不渲染（PIL 无 CJK 字体）。"
                      "要上图就用英文写。")
        merged = Image.new("RGB", (strip.width, strip.height + 34), "white")
        merged.paste(hdr, (0, 0))
        merged.paste(strip, (0, 34))
        rows.append(merged)

    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)),
                      "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    op = Path(a.out) if a.out else root / "box_figure.jpg"
    sheet.save(op, quality=93)
    todo = [i for i in a.idx
            if any(k not in boxes_all.get(str(i), {}) for k in arms)]
    print(f"{len(rows)} 行 -> {op}   ({sheet.width}x{sheet.height})")
    if todo:
        print(f"⚠ 还没标注完的 idx：{todo} —— 图上会打红字 NOT ANNOTATED，"
              f"别拿去当结论")


if __name__ == "__main__":
    main()
