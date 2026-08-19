#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 make-it-count 自己的评测器（YOLOv9e）给各臂打分，并**把 46% 拆开**。

    判定口径与 `help_code/make-it-count/evaluation_script.py:18-41` 一致：
        检测框里 class_name 完全等于目标类的个数 == 要求数 → 算对。
    区别只在实现路径：那边经 supervision 转一手，这里直接读
    `result.boxes.cls`，两者是同一批框、同一套类名（脚本会断言 model.names
    与 COCO-80 一致，所以这不是"假设兼容"，是"核对过兼容"）。

    ★ 真正新的东西是第三张表：**把 CountGen 的准确率拆成两段**。

    依据是源码里那条结构：`extract_mask.py:25-27` 一旦 DBSCAN 簇数 == N，
    就置 `obj_num_match=True`，`run_countgen.py:128-129` 直接输出原版图、
    **不做任何干预**。于是

        计数器说"已经对了"  ├─ 实际也对  → 白捡的分
                            └─ 实际是错的 → **这张图永远不会被修**  ← 天花板损失
        计数器说"不对"      ├─ 修正后对   → 方法真正的贡献
                            └─ 修正后仍错 → 修正失败

    所以 **CountGen 的上限 = 它 DBSCAN 计数器的准确率**。
    这张拆解表没有人报过，而且它直接回答"要动哪一段"。

    ⚠️ 两个必须说在前面的口径问题：
      1. YOLOv9e 自己也会数错。它是**在位者选定的尺子**，用它是为了跟人家的表
         对齐；但"计数器 vs YOLO 的一致率"这类读数继承了 YOLO 的误差，
         只能当**相对**指标看，不能当真值。
      2. N>9 的题目官方代码整题跳过，只有 vanilla 臂有图。**单独列，不并入总分。**

用法：
    python count_probe/yolo_eval.py --arms $SD_OUT/count/cocoount_arms
    # 权重：wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
"""

import argparse
import csv
import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

from make_arms import COCO80  # 同目录；断言用


def _pct(x, n):
    return f"{100.0*x/n:5.1f}%" if n else "   n/a"


def _photo_stats(path, side=256, blk=8, flat_thr=3.0):
    """两个零依赖的"照片感"代理，专门量我们看到的那种失效：照片 → 平涂剪贴画。

    · **colourfulness**（Hasler & Süsstrunk 2003，通行定义）：
        rg = R−G, yb = ½(R+G)−B
        C  = √(σ_rg² + σ_yb²) + 0.3·√(μ_rg² + μ_yb²)
      黑白/灰阶的剪贴画分数极低，真实照片通常在 30~80。
    · **flat**：把图切成 blk×blk 的块，块内标准差 < flat_thr 的块占比。
      纯色底一大片 → 这个值高。

    都是**代理**，不是标准指标 —— 论文主表要报的仍是 CLIPScore（--clip）。
    但它们不依赖任何权重，现在就能算，而且直接对准了肉眼看到的那件事。
    """
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    x = np.asarray(im, dtype=np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    col = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    g = x.mean(-1)
    h, w = (g.shape[0] // blk) * blk, (g.shape[1] // blk) * blk
    b = g[:h, :w].reshape(h // blk, blk, w // blk, blk).std(axis=(1, 3))
    return col, float((b < flat_thr).mean())


def evaluate(model, files, conf):
    """→ ({file: 检出的目标类框数}, {file: 这些框的平均置信度})

    置信度是**免费的质量代理**：图糊了、物体畸形，检测器的置信度就掉。
    它不是标准指标，但不需要任何新模型，而且和计数用的是同一次前向。
    真正要报进论文的质量列是 CLIPScore（见 --clip），与在位者对齐
    （CountCluster 报 CLIPScore + ImageReward，CountDiffusion 报 CLIP-score + IR）。
    """
    out, cf = {}, {}
    for i, p in enumerate(files):
        kw = {"verbose": False}
        if conf is not None:
            kw["conf"] = conf
        r = model(str(p), **kw)[0]
        target = p.name.split("__")[1]
        names = r.names
        hit = [float(s) for c, s in zip(r.boxes.cls.tolist(),
                                        r.boxes.conf.tolist())
               if names[int(c)] == target]
        out[p.name] = len(hit)
        cf[p.name] = sum(hit) / len(hit) if hit else None
        if (i + 1) % 50 == 0:
            print(f"    …{i+1}/{len(files)}")
    return out, cf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="make_arms.py 的输出目录")
    ap.add_argument("--weights", default="yolov9e.pt")
    ap.add_argument("--conf", type=float, default=None,
                    help="不给就用 ultralytics 默认（与官方脚本一致）")
    ap.add_argument("--clip", action="store_true",
                    help="加算 CLIPScore（复用 scalediff_probe/clip_score.py 的"
                         "加载器，只走本地缓存、不下载）。**跑批还在占卡时别开**，"
                         "CLIP 塔会再吃 1~2 GB 显存")
    a = ap.parse_args()
    root = Path(a.arms).resolve()

    idx = {r["file"]: r for r in json.load(open(root / "index.json"))}
    arms = sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any(d.glob("*.png")))
    print(f"臂：{arms}")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = [model.names[i] for i in range(len(model.names))]
    if names != COCO80:
        bad = [(i, x, y) for i, (x, y) in enumerate(zip(names, COCO80)) if x != y]
        raise SystemExit(f"!! {a.weights} 的类名表与 COCO-80 不符，前几处：{bad[:5]}\n"
                         f"   make_arms.py 是按 COCO-80 写文件名的，对不上就会静默全错。")
    print(f"{a.weights}: 类名表与 COCO-80 逐项一致 ✓")

    counts, confs = {}, {}
    for arm in arms:
        files = sorted((root / arm).glob("*.png"))
        print(f"\n跑 {arm}：{len(files)} 张")
        counts[arm], confs[arm] = evaluate(model, files, a.conf)

    # ---------- 质量列 ----------
    # 只报计数准确率是不够的：方法完全可能靠把图弄糟来把数弄对，而那样的
    # "提升"没有意义。在位者自己也报质量（CountCluster: CLIPScore+ImageReward；
    # CountDiffusion: CLIP-score+IR），我们不报，审稿人一定会问。
    clip = None
    if a.clip:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "scalediff_probe"))
        try:
            from clip_score import load_clip
            clip = load_clip()
        except Exception as e:
            print(f"⚠ CLIP 塔拿不到（{type(e).__name__}: {str(e)[:80]}），"
                  f"跳过该列，其余不受影响")

    # ---------- 明细 ----------
    rows = []
    for fn, meta in idx.items():
        row = {k: meta[k] for k in
               ("file", "stem", "N", "coco_class", "prompt", "seed",
                "n_dbscan", "obj_num_match", "skipped_by_official")}
        for arm in arms:
            c = counts[arm].get(fn)
            row[f"yolo_{arm}"] = c
            row[f"ok_{arm}"] = None if c is None else int(c == meta["N"])
            row[f"conf_{arm}"] = confs[arm].get(fn)
        rows.append(row)
    csv_p = root / "yolo_results.csv"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n逐题明细 → {csv_p}")

    main_rows = [r for r in rows if not r["skipped_by_official"]]
    over9 = [r for r in rows if r["skipped_by_official"]]

    # ---------- 表一：各臂总分 ----------
    print(f"\n{'='*64}\n表一 各臂准确率（N≤9，共 {len(main_rows)} 题）")
    print(f"{'臂':<12}{'张数':>6}{'准确率':>9}{'MAE':>8}")
    for arm in arms:
        ok = [r for r in main_rows if r[f"ok_{arm}"] is not None]
        if not ok:
            continue
        acc = sum(r[f"ok_{arm}"] for r in ok)
        mae = sum(abs(r[f"yolo_{arm}"] - r["N"]) for r in ok) / len(ok)
        print(f"{arm:<12}{len(ok):>6}{_pct(acc, len(ok)):>9}{mae:>8.3f}")
    print("对齐目标【一手】：CountCluster 表 CountGen/SDXL 46.20 / 27.88（CountGD 评测器）；"
          "\n              CountDiffusion 表 CoCoCount CountGen/SDXL 51 / 34（Grounded SAM）。"
          "\n              评测器不同，别指望对上小数点；量级对得上就算复现成功。")

    # ---------- 表一b：质量列 ----------
    # 计数准确率单独看是可以被"把图弄糟"骗到的。这一张就是防那个的。
    print(f"\n表一b 质量（只报，不做判据 —— 但计数涨、质量跌就不算赢）")
    print(f"{'臂':<12}{'YOLO 置信度':>11}{'colourfulness':>14}{'平涂块占比':>11}"
          f"{'CLIPScore':>11}")
    base = {}
    for arm in arms:
        cs = [r[f"conf_{arm}"] for r in main_rows if r.get(f"conf_{arm}")]
        cols, flats = [], []
        for r in main_rows:
            f = root / arm / r["file"]
            if f.exists():
                c, fl = _photo_stats(f)
                cols.append(c)
                flats.append(fl)
        cl = ""
        if clip is not None:
            from PIL import Image
            v = [clip.score(Image.open(root / arm / r["file"]).convert("RGB"),
                            r["prompt"])
                 for r in main_rows if (root / arm / r["file"]).exists()]
            cl = f"{sum(v)/len(v):11.3f}" if v else f"{'n/a':>11}"
        mc = sum(cols) / len(cols) if cols else float("nan")
        base[arm] = mc
        print(f"{arm:<12}{(sum(cs)/len(cs) if cs else float('nan')):>11.3f}"
              f"{mc:>14.1f}{(sum(flats)/len(flats) if flats else float('nan')):>11.1%}{cl}")
    if "vanilla" in base:
        for arm in arms:
            if arm != "vanilla" and base["vanilla"]:
                d = 100.0 * (base[arm] - base["vanilla"]) / base["vanilla"]
                print(f"  {arm} 相对 vanilla 的 colourfulness：{d:+.1f}%"
                      + ("   ⚠️ 掉得很厉害，看图确认是不是塌成平涂了"
                         if d < -20 else ""))
    if clip is None:
        print("  （CLIPScore 未算：装好 open_clip 后加 --clip。它是论文主表要的那一列，"
              "colourfulness/平涂块只是代理）")

    # ---------- 表二：按 N 分档 ----------
    print(f"\n表二 按要求个数分档")
    hdr = f"{'N':>3}{'题数':>6}" + "".join(f"{arm:>12}" for arm in arms)
    print(hdr)
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["N"]].append(r)
    for N in sorted(by_n):
        rs = by_n[N]
        line = f"{N:>3}{len(rs):>6}"
        for arm in arms:
            ok = [r for r in rs if r[f"ok_{arm}"] is not None]
            line += f"{_pct(sum(r[f'ok_{arm}'] for r in ok), len(ok)):>12}"
        print(line + ("   ← 官方跳过修正" if N > 9 else ""))

    # ---------- 表三：天花板拆解 ----------
    if "vanilla" in arms and "countgen" not in arms:
        # --vanilla-only 的先行档：四个格子里能算出三个，只差"修正成功率"。
        v = [r for r in main_rows
             if r["obj_num_match"] is not None and r["ok_vanilla"] is not None]
        if v:
            n = len(v)
            m_ok = sum(1 for r in v if r["obj_num_match"] and r["ok_vanilla"])
            m_bad = sum(1 for r in v if r["obj_num_match"] and not r["ok_vanilla"])
            f_all = sum(1 for r in v if not r["obj_num_match"])
            print(f"\n{'='*64}\n表三（先行档：只有 vanilla 臂，N≤9 共 {n} 题）")
            print(f"{'计数器说对 & 实际对（会被原样输出，白捡）':<40}{m_ok:>5}{_pct(m_ok, n):>9}")
            print(f"{'计数器说对 & 实际错 ← 永远修不到':<40}{m_bad:>5}{_pct(m_bad, n):>9}")
            print(f"{'计数器说错 → 会进修正（成功率待测）':<40}{f_all:>5}{_pct(f_all, n):>9}")
            print(f"\n读法：CountGen 的准确率 = {_pct(m_ok, n)} + "
                  f"{_pct(f_all, n)}×修正成功率。")
            print(f"      中间那 {m_bad} 题（{_pct(m_bad, n).strip()}）是**结构性损失**——"
                  f"管线信任自己的计数器，这些图不会被送进修正，")
            print(f"      所以无论修正做得多好都拿不到。换掉计数器才拿得到。")
            _agree(v)
        _over9(over9)
        return
    if "vanilla" not in arms or "countgen" not in arms:
        print("\n（缺 vanilla 臂，跳过拆解）")
        return
    dec = [r for r in main_rows
           if r["obj_num_match"] is not None and r["ok_vanilla"] is not None
           and r["ok_countgen"] is not None]
    if not dec:
        print("\n（counter_log.jsonl 缺失或没对上，出不了拆解表）")
        return
    n = len(dec)
    # match 分支里管线交付的就是那张原版图（run_countgen.py:128-129），
    # 所以两个臂应当是同一张图、同一个读数。不等就是链路有问题，要喊出来。
    odd = [r for r in dec
           if r["obj_num_match"] and r["ok_vanilla"] != r["ok_countgen"]]
    if odd:
        print(f"\n⚠️ {len(odd)} 题 obj_num_match=True 但两臂读数不同 —— "
              f"这两个臂本该是同一张图，检查分臂或缓存。例：{[r['stem'] for r in odd][:3]}")
    m_ok = [r for r in dec if r["obj_num_match"] and r["ok_countgen"]]
    m_bad = [r for r in dec if r["obj_num_match"] and not r["ok_countgen"]]
    f_ok = [r for r in dec if not r["obj_num_match"] and r["ok_countgen"]]
    f_bad = [r for r in dec if not r["obj_num_match"] and not r["ok_countgen"]]
    print(f"\n{'='*64}\n表三 CountGen 的准确率卡在哪一段（N≤9，{n} 题）")
    print(f"{'情形':<34}{'题数':>6}{'占比':>9}")
    print(f"{'计数器说对 & 实际对（白捡）':<34}{len(m_ok):>6}{_pct(len(m_ok), n):>9}")
    print(f"{'计数器说对 & 实际错 ← 永远修不到':<34}{len(m_bad):>6}{_pct(len(m_bad), n):>9}")
    print(f"{'计数器说错 & 修好了（方法的贡献）':<34}{len(f_ok):>6}{_pct(len(f_ok), n):>9}")
    print(f"{'计数器说错 & 没修好':<34}{len(f_bad):>6}{_pct(len(f_bad), n):>9}")
    print(f"\nCountGen 实测准确率 = (白捡 + 修好) / n = "
          f"{_pct(len(m_ok)+len(f_ok), n)}")
    print(f"**若计数器完美**（把「说对却是错的」那 {len(m_bad)} 题也送进修正，"
          f"且修正成功率与现在相同 {_pct(len(f_ok), max(len(f_ok)+len(f_bad),1))}）：")
    rate = len(f_ok) / max(len(f_ok) + len(f_bad), 1)
    print(f"  上限 ≈ {_pct(len(m_ok) + (len(f_ok)+len(f_bad)+len(m_bad))*rate, n)}"
          f"   ← 换计数器最多能拿到这么多")

    _agree(dec)
    _over9(over9)


def _agree(rows):
    """DBSCAN 计数器与 YOLO 在同一张原版图上的一致程度。"""
    agree = [r for r in rows if r.get("n_dbscan") is not None
             and r.get("yolo_vanilla") is not None]
    if not agree:
        return
    same = sum(1 for r in agree if r["n_dbscan"] == r["yolo_vanilla"])
    mae = sum(abs(r["n_dbscan"] - r["yolo_vanilla"]) for r in agree) / len(agree)
    print(f"\nDBSCAN 计数器 vs YOLO（同一张原版图，{len(agree)} 题）："
          f"完全一致 {_pct(same, len(agree))}，MAE {mae:.2f}")
    print("  注意：这是两个都会错的数数器在互比，只能当相对读数。")


def _over9(over9):
    if not over9:
        return
    ok = [r for r in over9 if r["ok_vanilla"] is not None]
    print(f"\n{'='*64}\nN>9 单列（{len(over9)} 题，官方 run_countgen.py:104 整题跳过）")
    print(f"  vanilla 准确率 {_pct(sum(r['ok_vanilla'] for r in ok), len(ok))}"
          f"，CountGen 无输出")
    print("  这一档不能并入总分；报数时要写明它被在位者的实现排除在外。")


if __name__ == "__main__":
    main()
