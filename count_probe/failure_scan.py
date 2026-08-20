#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把三种失效**全批**扫一遍，不再靠抽样看图。零 GPU（--clip 除外）。

    这一轮反复踩的坑是：拿三五张图下全局结论，或者拿整批均值回答尾巴问题。
    好在人工核对时点出的三种失效，有两种可以逐题量出来：

      · **意图执行失败** —— 引导画出来的个数 ≠ mask 要求它改的个数。
            残差 = (本配置数出 − vanilla 数出) − (要求 N − DBSCAN 数出)
        残差 0 = 忠实执行；|残差| 变大 = 比 CountGen 执行得更糟。
      · **物体形变** —— 逐题 YOLO 平均置信度（CSV 里的 conf_countgen）。
        物体畸形、糊掉，检测器置信度会掉。注意它只判「像不像这个类」，
        不判「像不像一个真实物体」；实测汽车那题人眼觉得变形、置信度反而更高，
        所以**它抓不到的那部分只能靠人工核对**，这一点要写进 limitations。
      · **计数改坏** —— CountGen 对而我们错。

    判据一律**相对 CountGen**，不是相对 vanilla：我们建在它上面，它自己造成的
    退化不该记在我们头上（同 quality_probe --ref 的理由）。

    两条支路要分开报。它们的机制不同（删多余靠轻推、补缺失靠持续优化），
    混在一起会互相掩盖 —— 绵羊那题就是「补缺失被迫跑满」造成的，
    而它在总表里会被删多余那 23 题稀释掉。

用法：
    python count_probe/failure_scan.py \\
        --ref  $SD_OUT/count/tune_orig_arms \\
        --arms $SD_OUT/count/tune_{fg,rhoover,both}_arms
"""

import argparse
import csv
from pathlib import Path


def _f(v):
    if v in ("", None, "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(v):
    x = _f(v)
    return None if x is None else int(x)


def _ok(v):
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    x = _i(v)
    return None if x is None else bool(x)


def _rows(arms):
    p = Path(arms) / "yolo_results.csv"
    if not p.exists():
        raise SystemExit(f"!! 没有 {p}")
    out = {}
    for r in csv.DictReader(p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        r["_N"] = _i(r["N"])
        r["_nd"] = _i(r["n_dbscan"])
        r["_ym"] = _i(r.get("yolo_countgen"))
        r["_yv"] = _i(r["yolo_vanilla"])
        r["_ok"] = _ok(r.get("ok_countgen"))
        r["_conf"] = _f(r.get("conf_countgen"))
        r["_match"] = r["obj_num_match"] in ("True", "true", "1")
        r["_resid"] = (None if None in (r["_ym"], r["_yv"], r["_nd"], r["_N"])
                       else (r["_ym"] - r["_yv"]) - (r["_N"] - r["_nd"]))
        out[r["stem"]] = r
    return out


def _dir(r):
    if r["_match"] or r["_nd"] is None:
        return None                      # 没进修正的题，谈不上「执行得好不好」
    return "删多余" if r["_nd"] > r["_N"] else ("补缺失" if r["_nd"] < r["_N"] else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="参照（原版 CountGen）的 _arms 目录")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--conf-drop", type=float, default=0.03,
                    help="逐题 YOLO 置信度掉多少算「可能形变」")
    ap.add_argument("--worst", type=int, default=6, help="点名前几题")
    ap.add_argument("--clip", action="store_true",
                    help="把逐题 CLIPScore 也算进来（要 GPU，每配置 120 次前向）")
    a = ap.parse_args()

    ref = _rows(a.ref)
    clip = None
    if a.clip:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scalediff_probe"))
        from clip_score import load_clip
        clip = load_clip()

    def clip_of(arms, r):
        from PIL import Image
        p = Path(arms) / "countgen" / r["file"]
        return (clip.score(Image.open(p).convert("RGB"), r["prompt"])
                if p.exists() else None)

    ref_clip = {}
    if clip is not None:
        ref_clip = {s: clip_of(a.ref, r) for s, r in ref.items()}

    # ---- CountGen 自己的基线：它执行得有多忠实 ----
    base = [r for r in ref.values() if _dir(r) and r["_resid"] is not None]
    print(f"\n参照（CountGen 自己，{len(base)} 道进过修正的题）：")
    for tag in ("删多余", "补缺失", "全部"):
        sub = [r for r in base if tag == "全部" or _dir(r) == tag]
        if not sub:
            continue
        z = sum(1 for r in sub if r["_resid"] == 0)
        big = sum(1 for r in sub if abs(r["_resid"]) >= 2)
        print(f"  {tag:<6} {len(sub):>3} 题   残差=0 {z:>3} ({z/len(sub):>5.1%})   "
              f"|残差|≥2 {big:>3} ({big/len(sub):>5.1%})")
    print("  这是我们要比的基准 —— 它自己就有大半的修正没有忠实执行。")

    rows = []
    for arms in a.arms:
        cur = _rows(arms)
        name = Path(arms).name.replace("_arms", "")
        rec = dict(name=name, hits=[])
        for s, r in cur.items():
            b = ref.get(s)
            if b is None or _dir(r) is None:
                continue
            worse_resid = (r["_resid"] is not None and b["_resid"] is not None
                           and abs(r["_resid"]) > abs(b["_resid"]))
            dconf = (None if None in (r["_conf"], b["_conf"])
                     else r["_conf"] - b["_conf"])
            worse_conf = dconf is not None and dconf < -a.conf_drop
            broke = bool(b["_ok"]) and not bool(r["_ok"])
            dclip = None
            if clip is not None and ref_clip.get(s) is not None:
                v = clip_of(arms, r)
                dclip = None if v is None else v - ref_clip[s]
            if worse_resid or worse_conf or broke or (dclip is not None and dclip < -1.0):
                rec["hits"].append(dict(
                    stem=s, dir=_dir(r), resid_ref=b["_resid"], resid=r["_resid"],
                    dconf=dconf, broke=broke, dclip=dclip,
                    sev=(2 if broke else 0) + (1 if worse_resid else 0)
                        + (1 if worse_conf else 0)
                        + (1 if dclip is not None and dclip < -1.0 else 0)))
            rec.setdefault("n", 0)
            rec["n"] += 1
            for k, v in (("wr", worse_resid), ("wc", worse_conf), ("bk", broke)):
                rec[k] = rec.get(k, 0) + int(v)
            for tag in ("删多余", "补缺失"):
                if _dir(r) == tag:
                    rec[f"n_{tag}"] = rec.get(f"n_{tag}", 0) + 1
                    for k, v in (("wr", worse_resid), ("wc", worse_conf),
                                 ("bk", broke)):
                        rec[f"{k}_{tag}"] = rec.get(f"{k}_{tag}", 0) + int(v)
        rows.append(rec)

    # ---- 表一：全批 ----
    print(f"\n{'='*74}\n表一 相对 CountGen 的失效扫描（只算进过修正的题）")
    print(f"{'配置':<12}{'题数':>5}{'残差变差':>10}{'置信度掉>' + f'{a.conf_drop:g}':>13}"
          f"{'计数改坏':>10}{'任一命中':>10}")
    for r in rows:
        n = r.get("n", 0) or 1
        print(f"{r['name']:<12}{r.get('n',0):>5}{r.get('wr',0):>8}/{n:<3}"
              f"{r.get('wc',0):>10}/{n:<3}{r.get('bk',0):>8}/{n:<3}"
              f"{len(r['hits']):>8}/{n:<3}")
    print("  「任一命中」= 这三条（或 CLIPScore 掉超过 1 分）里中了至少一条的题数，"
          "\n  也就是**该人工复核的清单**。它不等于「坏图数」—— 检测器和残差都会误报。")

    # ---- 表二：分支拆开 ----
    print(f"\n表二 分支拆开（两支机制不同，混在一起会互相掩盖）")
    print(f"{'配置':<12}{'支路':<8}{'题数':>5}{'残差变差':>10}{'置信度掉':>10}{'计数改坏':>10}")
    for r in rows:
        for tag in ("删多余", "补缺失"):
            n = r.get(f"n_{tag}", 0)
            if not n:
                continue
            print(f"{r['name']:<12}{tag:<8}{n:>5}{r.get('wr_'+tag,0):>8}/{n:<3}"
                  f"{r.get('wc_'+tag,0):>8}/{n:<3}{r.get('bk_'+tag,0):>8}/{n:<3}")

    # ---- 表三：点名 ----
    print(f"\n表三 该人工复核的题（按严重度排，前 {a.worst}）")
    for r in rows:
        if not r["hits"]:
            print(f"\n  {r['name']}：**一题都没命中**")
            continue
        print(f"\n  {r['name']}（共 {len(r['hits'])} 题）")
        for h in sorted(r["hits"], key=lambda x: -x["sev"])[:a.worst]:
            bits = []
            if h["broke"]:
                bits.append("计数改坏")
            if h["resid"] is not None and h["resid_ref"] is not None:
                bits.append(f"残差 {h['resid_ref']:+d}→{h['resid']:+d}")
            if h["dconf"] is not None:
                bits.append(f"置信度 {h['dconf']:+.3f}")
            if h["dclip"] is not None:
                bits.append(f"CLIP {h['dclip']:+.2f}")
            print(f"    {h['stem'][:36]:<38}{h['dir']:<8}" + "  ".join(bits))

    print("\n判读：")
    print("  · 「任一命中」占比高 = 失效是普遍的，不是那三张个例；低 = 个例。")
    print("  · 分支拆开后若集中在**补缺失**，多半是阈值置 0 让那一支被迫跑满"
          "\n    （--thresh-frac-over 的连带效果），修法是给补缺失支保留原版阈值。")
    print("  · 分支拆开后若集中在**删多余**，那是收 mask 收太狠，修法是 --fg-separate。")
    print("  · 置信度这一列抓不到「像不像一个真实物体」，只抓「像不像这个类」——"
          "\n    实测汽车那题人眼觉得变形、置信度反而更高。这部分只能人工核对，"
          "\n    并且必须写进 limitations，不能当作没有。")


if __name__ == "__main__":
    main()
