"""主体级漏检：4096² 上还认得出基图里那个主体吗？两个 arm 一样吗？

起因（03/1234 我们这版的标注图）：主船清清楚楚地在，GroundingDINO
一个框都没给它；同时检出的两个小框里有一个不是船。
**同一行里漏检和误检方向相反、恰好抵消** —— 靠抵消得到的正确不是正确。

逐行不准这件事本身可以接受（我们报的是 45 行的聚合量）。真正致命的是
另一件：**如果漏检在两个 arm 里频率不同，A/B 差值就有系统偏差** ——
我们这版更容易被漏检，计数就被系统性压低，看起来更好，但那是尺子偏心。

锚点是现成的、而且很硬：**基图 1024² 两个 arm 逐字节相同（90/90 已验证）**，
主体在基图上检得到。所以：

    1. 基图上检出主体框 -> 坐标 x4 得到 4096² 上的锚点（只用最高分那个）；
    2. 每个 arm 的 4096² 里，有没有框覆盖那个锚点？没有 = 主体级漏检；
    3. 漏检的行再分：锚点区域的像素在两个 arm 之间变了没有？
       没变而框没了 -> 检测器漏检；变了 -> 主体真的被改动了。
       阈值同 box_audit：用**命中行**的像素差分布当零假设，不自己拍数。

预注册判据（写在跑之前）：
    ① 两个 arm 的主体命中率之差 <= 2/45 行 -> 无差别偏差，A/B 差值可用；
       差得更多 -> **MAE 必须连同命中率一起报，并讨论方向。**
    ② 我们这版的"主体真被改动"行数 == 0 -> 方法没有删主体；
       > 0 -> 那些行进失败分析，逐个看图。

    python scalediff_probe/subject_track.py
    python scalediff_probe/subject_track.py --cat lone
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from box_audit import iou, pix_diff, region              # noqa: E402
from count_objects import Detector                       # noqa: E402

HIT_IOU = 0.20          # 锚点命中判据：宽松，我们问的是"认出来没有"不是"框得准"


def whole_diff(a, b, s=8):
    """整张图的平均绝对灰度差（1/8 缩略图上算，够用且快）。

    它是"这一行整体变了多少"的标度。锚点差除以它，才是"主体变得比
    其余部分更厉害吗"。
    """
    ga = np.asarray(a.convert("L").resize((a.width // s, a.height // s)),
                    dtype=np.int16)
    gb = np.asarray(b.convert("L").resize((b.width // s, b.height // s)),
                    dtype=np.int16)
    return float(np.abs(ga - gb).mean())


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def files(rec):
    lo = rec["files"][str(min(int(k) for k in rec["files"]))]
    hi = rec["files"][str(max(int(k) for k in rec["files"]))]
    return lo, hi


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--cat", default=None)
    ap.add_argument("--pctl", type=float, default=95)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))
    if a.cat:
        keys = [k for k in keys if A[k]["cat"] == a.cat]

    det = Detector()
    rows, hit_d, miss = [], [], []
    print(f"{'idx':<5}{'seed':>6}{'cat':<10}{'基图主体':>9}"
          f"{'基线命中':>9}{'我们命中':>9}")
    print("-" * 50)
    for k in keys:
        lo_a, hi_a = files(MA[k])
        _, hi_b = files(MB[k])
        base = Image.open(Path(a.base) / lo_a).convert("RGB")
        b0, s0 = det.detect(base, A[k]["subject"])
        if len(b0) == 0:                      # 基图就没主体（empty 类，或检测失败）
            rows.append((k, None, None, None))
            print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{'无':>9}{'-':>9}{'-':>9}")
            del base
            continue
        j = int(np.argmax(s0))
        scale = Image.open(Path(a.base) / hi_a).width / base.width
        anchor = [float(x) * scale for x in b0[j]]
        del base

        res = {}
        imgs = {}
        for arm, d, fn in (("base", a.base, hi_a), ("ours", a.new, hi_b)):
            im = Image.open(Path(d) / fn).convert("RGB")
            imgs[arm] = im
            bb, _ = det.detect(im, A[k]["subject"])
            res[arm] = any(iou(anchor, list(map(float, x))) >= HIT_IOU for x in bb)

        # **相对锚点变化 = 锚点区域的像素差 ÷ 整张图的像素差。**
        # 绝对像素差不能用：分类别看，中位数从 portrait 4.3 到 texture 8.0
        # 差近两倍，而全局 95 分位是 9.61 —— 对 texture 而言阈值只比中位数
        # 高一点，那一类几乎必然被判"主体被改动"（15_texture 两行就是这么
        # 被误标的）。比值问的是**"锚点变得比图像其余部分更厉害吗"**，
        # 自动消掉类别间的尺度差异，也消掉"整张图本来就发散"这个混淆。
        whole = whole_diff(imgs["base"], imgs["ours"])
        d_anchor = (pix_diff(imgs["base"], imgs["ours"], anchor)
                    / max(whole, 1e-6))
        if res["base"] and res["ours"]:
            hit_d.append((k, d_anchor))       # 两边都认出来 -> 零假设样本
        else:
            miss.append((k, res["base"], res["ours"], d_anchor))
        rows.append((k, True, res["base"], res["ours"]))
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{'有':>9}"
              f"{'√' if res['base'] else '×':>9}{'√' if res['ours'] else '×':>9}")
        del imgs

    # 分组报。**单锚点对 crowd/texture 无意义**（"许多车""数千朵花"取最高分
    # 那一个框当锚点，漏检是必然的），而这两类 card=None，本来就不在主指标里。
    # 第一版把它们和主指标行混在一个命中率里，稀释了要看的那个数。
    MAIN = {"lone", "empty", "portrait", "structure"}
    have = [r for r in rows if r[1]]
    groups = [("主指标行 (lone/empty/portrait/structure)",
               [r for r in have if A[r[0]]["cat"] in MAIN]),
              ("crowd + texture（单锚点不适用，仅供参考）",
               [r for r in have if A[r[0]]["cat"] not in MAIN]),
              ("全部", have)]
    for gname, g in groups:
        if not g:
            continue
        hb = sum(1 for r in g if r[2])
        ho = sum(1 for r in g if r[3])
        n = len(g)
        mark = ""
        if gname.startswith("主指标"):
            mark = ("  -> 判据① 过：无差别偏差，A/B 差值可用"
                    if abs(hb - ho) <= 2 else
                    "  -> **判据① 没过：漏检不对称，MAE 必须连同命中率一起报**")
        print(f"\n{gname}  {n} 行")
        print(f"  基线认出主体 {hb}/{n} = {hb/n:.0%}   "
              f"我们 {ho}/{n} = {ho/n:.0%}   差 {abs(hb-ho)} 行{mark}")

    # 偏差上界：把"我们漏检、基线命中"的行各补回 1 个计数，重算 MAE。
    # 这是最保守的修正 —— 假定每一次漏检都让我们白占了一个便宜。
    # 偏差上界。**基线被单独漏检的行也要补**，否则修正是单边的。
    only_ours = [r[0] for r in have if r[2] and not r[3] and A[r[0]]["cat"] in MAIN]
    only_base = [r[0] for r in have if r[3] and not r[2] and A[r[0]]["cat"] in MAIN]
    # 措辞注意：这是**漏检修正**，不是严格意义的上界。excess 为负的行
    # （检测器本来就少数了）补回 1 反而使 |excess| 变小 —— 23/77 就是这样，
    # 和 03/1234 的变大互相抵消，于是主指标全体 0.75 -> 0.75 纹丝不动。
    # 修正的方向是对的（漏了一个主体，真值就该 +1），但别叫它"上界"。
    print(f"\n漏检修正：我们被单独漏检 {len(only_ours)} 行，"
          f"基线被单独漏检 {len(only_base)} 行，各补回 1 个计数")
    print(f"  {'组':<12}{'基线 MAE':>18}{'我们 MAE':>18}{'降幅':>18}")
    for gname, cats in (("lone", {"lone"}), ("主指标全体", MAIN)):
        ks = [k for k in keys if A[k]["cat"] in cats
              and A[k].get("excess") is not None and B[k].get("excess") is not None]
        if not ks:
            continue
        ma0 = sum(abs(A[k]["excess"]) for k in ks) / len(ks)
        mb0 = sum(abs(B[k]["excess"]) for k in ks) / len(ks)
        ma1 = sum(abs(A[k]["excess"] + (1 if k in only_base else 0))
                  for k in ks) / len(ks)
        mb1 = sum(abs(B[k]["excess"] + (1 if k in only_ours else 0))
                  for k in ks) / len(ks)
        print(f"  {gname:<12}{f'{ma0:.2f} -> {ma1:.2f}':>18}"
              f"{f'{mb0:.2f} -> {mb1:.2f}':>18}"
              f"{f'{(1-mb0/ma0)*100:.1f}% -> {(1-mb1/ma1)*100:.1f}%':>18}")
    print("  **论文里报修正后的保守值，并在脚注写明修正方式。**")

    if not hit_d:
        print("\n没有两边都命中的行，建不起零假设。")
        return 1
    thr = float(np.percentile([d for _, d in hit_d], a.pctl))
    print(f"\n零假设（两边都命中的行，**相对锚点变化** = 锚点差/整图差，"
          f"n={len(hit_d)}）：")
    print(f"  中位 {np.median([d for _, d in hit_d]):.2f}  "
          f"{a.pctl:.0f} 分位 {thr:.2f}  -> 超过它才算主体被特意改动")
    bycat = {}
    for kk, d in hit_d:
        bycat.setdefault(A[kk]["cat"], []).append(d)
    print("  分类别：" + "  ".join(
        f"{c} n={len(v)} 中位{np.median(v):.2f}" for c, v in sorted(bycat.items())))
    print("  （改成比值之后各类别应当靠拢；若仍相差一倍以上，说明还要分层。）")

    print(f"\n漏检的 {len(miss)} 行：")
    changed_ours = []
    for k, rb, ro, d in sorted(miss, key=lambda x: -x[3]):
        who = "基线" if not rb else ""
        who += ("我们" if not ro else "")
        # **判据②只看"基线命中 + 我们漏检 + 像素变了"。** 第一版写成
        # `d > thr and not ro`，把"两个 arm 都漏检"的行也算作我们改动了主体
        # —— 两边都漏的行不可归因于我们（15_texture 两条就是这么被误算的）。
        attributable = rb and (not ro) and d > thr
        if attributable:
            changed_ours.append(k)
            kind = "**基线命中/我们漏检 + 像素变了 -> 可归因于方法，要看图**"
        elif d > thr:
            kind = "像素变了，但两边都漏检 -> 不可归因"
        else:
            kind = "检测器漏检（像素基本没变）"
        print(f"  {k[0]:02d}_{A[k]['cat']}_s{k[1]:<6} 谁漏={who:<6} "
              f"相对锚点变化 {d:6.2f}   {kind}")

    main_changed = [k for k in changed_ours if A[k]["cat"] in MAIN]
    print(f"\n  判据②  可归因于方法的主体改动：{len(changed_ours)} 行"
          f"（其中主指标类 {len(main_changed)} 行）")
    print("     " + ("过：主指标行里方法没有删主体" if not main_changed else
                     f"**没过：{main_changed} 进失败分析**"))
    if changed_ours and not main_changed:
        print(f"     主指标外的 {[f'{k[0]:02d}_s{k[1]}' for k in changed_ours]} "
              f"仍要看图 —— 它是唯一的疑似损害证据。")

    # 基图上就检不出主体的行：锚点测不了，这是尺子的另一个边界，要报不要静默
    nobase = [r[0] for r in rows if r[1] is None]
    card_nobase = [k for k in nobase if A[k]["cat"] in MAIN and A[k]["cat"] != "empty"]
    print(f"\n基图检不出主体的行 {len(nobase)} 条；其中**有基数、非 empty** 的 "
          f"{len(card_nobase)} 条锚点测不了："
          f"{[f'{k[0]:02d}_s{k[1]}' for k in card_nobase]}")
    print("  （empty 类基图检不出主体是正确行为，不算边界。）")

    print("""
这张表要进论文的敏感度部分。它回答的不是"检测器准不准"（不准），
而是**"检测器的不准在两个 arm 之间是否对称"** —— 只有对称，A/B 差值才成立。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
