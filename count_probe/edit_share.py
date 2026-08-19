#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检验一个机制假设：**收 mask 之所以有效，是因为它抬高了「这次编辑」在损失总权重里的份额。**
零 GPU，只读跑批时存下的 {stem}_masks.npz（32×32，几百字节）。

    背景。`loss_utils.py:5` 把 N 个实例标签压成二值前景，再用带 pos_weight=10 的
    BCE。于是每个 token 在损失里的权重只有两种：前景 10、背景 1。
    整张图的总权重 W = 10·n_fg + n_bg。

    「这次修正想干什么」是一件很局部的事：relayout 只动了几个 blob。
    把 bin(corrected) 与 bin(vanilla) 不一致的那些 token 叫**编辑区**。
    引导要克服的是模型本来的倾向，能不能扳过来，取决于编辑区在总权重里
    占多大份额 —— 如果 76.6% 的 token 都在以权重 10 喊「这里要有物体」，
    那几十个权重 1 的「这里不要有物体」就被淹掉了。

        edit_share = (编辑区 token 在最终目标里的权重之和) / W

    收 mask 不改变编辑区是哪些 token（vanilla / corrected 两张图跟收不收没关系），
    但它把 W 从 10·0.766+0.234 ≈ 7.89 压到 10·0.117+0.883 ≈ 2.05。
    所以假设给出一个**可证伪的定量预言**：

        tune_fg 的 edit_share 应当约为 tune_orig 的 7.89/2.05 ≈ 3.8 倍。

    对不上 → 假设当场作废，别再用「权重份额」解释 tune_fg 的增益。
    对上了 → 再问第二问：edit_share 高的题，是不是真的更容易修成功？
    这一问才决定它是不是一条能继续挖的设计原则。用置换检验，不看均值之差。

用法：
    python count_probe/edit_share.py --runs $SD_OUT/count/tune_{orig,fg,inst,inst2,hinge}
"""

import argparse
import csv
from pathlib import Path

import numpy as np

POS_WEIGHT = 10.0          # loss_utils.py 里写死的


def _ok(v):
    """ok_countgen 这一列在不同版本里写成过 1/0 和 True/False，两种都吃下来。"""
    if v in ("", None, "None"):
        return None
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    try:
        return bool(int(float(v)))
    except ValueError:
        return None


def _w(binary, pos_weight):
    """每个 token 在 BCEWithLogits(pos_weight) 里的权重：前景 pos_weight，背景 1。"""
    return np.where(binary, pos_weight, 1.0)


def _rows(run, arms_suffix, pos_weight):
    run = Path(run)
    arms = run.parent / (run.name + arms_suffix)
    csv_p = arms / "yolo_results.csv"
    if not csv_p.exists():
        return None, f"没有 {csv_p}"
    out = []
    for r in csv.DictReader(csv_p.open()):
        if r["skipped_by_official"] in ("True", "true", "1"):
            continue
        if r["obj_num_match"] in ("True", "true", "1"):
            continue                       # 没进修正的题没有「编辑」可言
        npz = run / f"{r['stem']}_masks.npz"
        if not npz.exists():
            continue
        ok = _ok(r.get("ok_countgen"))
        if ok is None:
            continue
        z = np.load(npz)
        van = z["vanilla"] != 0
        cor = z["corrected"] != 0
        post = z["postprocess"] != 0
        edit = van ^ cor                   # 这次 relayout 动了哪些 token
        if edit.sum() == 0:
            continue
        w = _w(post, pos_weight)
        n_d, N = int(z["n_dbscan"]), int(z["N"])
        out.append(dict(
            stem=r["stem"], N=N, n_dbscan=n_d,
            direction="删多余" if n_d > N else ("补缺失" if n_d < N else "?"),
            ok=ok,
            share=float(w[edit].sum() / w.sum()),
            n_edit=int(edit.sum()),
            fg_post=float(post.mean()),
            w_mean=float(w.mean()),
        ))
    return out, None


def _perm(x, y, n=20000, seed=0):
    """置换检验：中位数之差。样本小的时候 t 检验的正态假设不成立，这个不用假设。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    obs = abs(np.median(x) - np.median(y))
    pool = np.concatenate([x, y])
    rs = np.random.RandomState(seed)
    hits = 0
    for _ in range(n):
        rs.shuffle(pool)
        hits += abs(np.median(pool[:len(x)]) - np.median(pool[len(x):])) >= obs - 1e-12
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="跑批目录（不是 _arms）")
    ap.add_argument("--arms-suffix", default="_arms")
    ap.add_argument("--pos-weight", type=float, default=POS_WEIGHT)
    a = ap.parse_args()

    data = {}
    for run in a.runs:
        rows, err = _rows(run, a.arms_suffix, a.pos_weight)
        if err or not rows:
            print(f"（跳过 {Path(run).name}：{err or '没有可用的 npz'}）")
            continue
        data[Path(run).name] = rows

    if not data:
        raise SystemExit("!! 一个配置都没读到。npz 是跑批时存的，早期的跑批可能没有。")

    # ---------- 第一问：份额本身，以及那个 3.8 倍的预言 ----------
    print(f"\n第一问 编辑区在损失总权重里的份额（pos_weight={a.pos_weight:g}）")
    print(f"{'配置':<12}{'题数':>5}{'前景占比':>10}{'平均权重':>10}"
          f"{'编辑区token':>12}{'edit_share 中位':>16}")
    base = None
    for k, rs in data.items():
        sh = np.median([r["share"] for r in rs])
        wm = float(np.mean([r["w_mean"] for r in rs]))
        line = (f"{k:<12}{len(rs):>5}{np.mean([r['fg_post'] for r in rs]):>9.1%}"
                f"{wm:>10.2f}{np.mean([r['n_edit'] for r in rs]):>12.0f}{sh:>15.2%}")
        if base is None:
            base = (k, sh, wm)
        elif base[1] > 0:
            # 预言的倍数不写死，用**实测的平均权重**算：份额 ∝ 1/W，
            # 所以 share 之比应当等于 W 之比的倒数。这样这条检验自带标定。
            print(f"{line}   实测 ×{sh / base[1]:.2f}   预言 ×{base[2] / wm:.2f}"
                  f"（相对 {base[0]}）")
            continue
        print(line)
    print("  「编辑区 token」两两之间应当**基本相同** —— vanilla / corrected 两张 mask 与收不收无关，"
          "\n  收 mask 只作用在最终目标上。若这一列在配置间差很多，说明我上面的推理前提就不对。")
    print("  份额 ∝ 1/W，所以「实测倍数」应当对上「预言倍数」（预言是拿实测的平均权重算的，没写死）。"
          "\n  对不上 → 「权重份额」这个解释作废，tune_fg 的增益要另找原因。")

    # ---------- 第二问：份额高的题，是不是更容易修成功 ----------
    print(f"\n第二问 edit_share 高 → 更容易修成功？（这一问才决定它是不是设计原则）")
    print(f"{'配置':<12}{'方向':<8}{'修成功':>18}{'没修成':>18}{'置换检验 p':>12}")
    for k, rs in data.items():
        for tag, sub in [("全部", rs),
                         ("删多余", [r for r in rs if r["direction"] == "删多余"]),
                         ("补缺失", [r for r in rs if r["direction"] == "补缺失"])]:
            ok = [r["share"] for r in sub if r["ok"]]
            no = [r["share"] for r in sub if not r["ok"]]
            if not ok or not no:
                print(f"{k:<12}{tag:<8}{'（一边是空的，检不了）':>36}")
                continue
            p = _perm(ok, no)
            print(f"{k:<12}{tag:<8}{np.median(ok):>10.2%} (n={len(ok):>2})"
                  f"{np.median(no):>10.2%} (n={len(no):>2}){p:>12.3f}")
    print("  p 是对**中位数之差**做 20000 次标签置换得到的，不假设正态、不受离群值支配。")
    print("  修成功的那组 share 显著更高 → 机制成立，设计原则是「把编辑量在损失里的权重份额顶上去」。")
    print("  两组分不开 → tune_fg 的 +6.7 与这个量无关，当作一次碰巧调对的超参，别据此往下设计。")


if __name__ == "__main__":
    main()
