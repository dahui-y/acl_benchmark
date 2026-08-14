"""基图审计：同一张图在不同臂里被数出不同的数，说明尺子有噪声底。

**触发这个脚本的事实（2026-08-14）**：
parti_v12 的 1024 基图与 parti_hi **逐像素相同**（跑批自检 31/31 通过），
但 vlm_delta.jsonl 里报的基图均值是 1.83 vs 2.08 —— 12 行差了 3 个实例。
同一张图、同一个 prompt，数出来的数不一样。

两个可能，本脚本分辨：
  A 主体词漂移。`vlm_subjects.json` 是**按目录**缓存的，四个臂各算各的。
    同一 prompt 若在不同臂被解析成不同主体词（"statue" vs "monument"），
    计数当然不同 -> 修法：全局共用一份主体缓存。
  B VLM 本身不确定。同图同词仍给不同答案 -> 那是尺子的**噪声底**，
    必须量出来、写进论文，并据此判断我们 0.333 vs 0.583 的差是否显著。

**为什么非查不可**：Δdelta = count(hi) - count(base)。若 count(base)
本身带噪声，两臂的 delta 差里就混进了与方法无关的抖动。这是审稿人
一击致命的地方，必须我们自己先量、先说。

    python scalediff_probe/base_audit.py
    python scalediff_probe/base_audit.py --arms hi=parti_hi ours=parti_v12
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def load_delta(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return {}
    return {int(json.loads(l)["idx"]): json.loads(l)
            for l in p.open() if l.strip()}


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    if a.arms:
        arms = {}
        for s in a.arms:
            k, v = s.split("=", 1)
            arms[k] = Path(v if "/" in v else str(root / v))
    else:
        arms = {"hi": root / "parti_hi", "v1": root / "parti_v1",
                "v12": root / "parti_v12", "acc": root / "parti_acc"}
    arms = {k: v for k, v in arms.items() if (v / "vlm_delta.jsonl").exists()}
    if len(arms) < 2:
        raise SystemExit("至少要两臂")

    data = {k: load_delta(v) for k, v in arms.items()}
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    names = list(arms)
    print(f"共同 idx {len(common)} 条，臂：{names}\n")

    # ---------- 1. 主体词是否一致 ----------
    subs = {}
    for k, d in arms.items():
        p = d / "vlm_subjects.json"
        subs[k] = json.loads(p.read_text()) if p.exists() else {}
    print("=" * 70)
    print("① 主体词一致性（各臂各缓存一份，漂了就会数出不同的数）")
    drift = []
    for i in common:
        vals = {k: subs[k].get(str(i)) for k in names if subs[k]}
        uniq = set(v for v in vals.values() if v)
        if len(uniq) > 1:
            drift.append(i)
            print(f"  ⚠ {i}: " + "  ".join(f"{k}={v!r}" for k, v in vals.items()))
    print(f"  主体词漂移 {len(drift)}/{len(common)} 条"
          + ("" if drift else " —— 全部一致"))

    # ---------- 2. 基图是否同一张 ----------
    print("\n" + "=" * 70)
    print("② 基图文件是否逐字节相同（相同却数出不同 = 尺子噪声）")
    hashes, mismatch = {}, []
    for i in common:
        h = {}
        for k, d in arms.items():
            f = d / f"{i:05d}_1024.png"
            if f.exists():
                h[k] = md5(f)
        hashes[i] = h
        if len(set(h.values())) > 1:
            mismatch.append(i)
    print(f"  基图各臂逐字节相同的行：{len(common) - len(mismatch)}/{len(common)}")
    if mismatch:
        print(f"  基图不同的行（管线本就不同，预期之内）：{mismatch[:8]}"
              f"{' ...' if len(mismatch) > 8 else ''}")

    # ---------- 3. 同图不同计数 = 噪声底 ----------
    print("\n" + "=" * 70)
    print("③ **同一张基图被数出不同的数** —— 这就是尺子的噪声底")
    same_img, disagree, diffs = 0, [], []
    for i in common:
        h = hashes[i]
        groups = {}
        for k, v in h.items():
            groups.setdefault(v, []).append(k)
        for _, ks in groups.items():
            if len(ks) < 2:
                continue
            same_img += 1
            cs = {k: data[k][i].get("n_base") for k in ks}
            vals = set(v for v in cs.values() if v is not None)
            if len(vals) > 1:
                disagree.append(i)
                diffs.append(max(vals) - min(vals))
                print(f"  ⚠ {i}: 同一张基图，"
                      + "  ".join(f"{k}={v}" for k, v in cs.items())
                      + f"   主体={data[ks[0]][i].get('subject')!r}")
    if same_img:
        rate = len(disagree) / same_img
        print(f"\n  可比对的同图组 {same_img} 个，计数不一致 {len(disagree)} 个"
              f"（{rate:.0%}）")
        if diffs:
            print(f"  不一致时的幅度：均值 {sum(diffs)/len(diffs):.2f}，"
                  f"最大 {max(diffs)}")
            print(f"  -> **噪声底 ≈ {len(disagree)*sum(diffs)/len(diffs)/same_img:.3f} "
                  f"实例/行**。任何小于这个量的方法间差异都不可信。")
        else:
            print("  -> 噪声底 = 0：同图同词计数完全可复现，Δdelta 的差可以直接归因于方法。")
    else:
        print("  没有逐字节相同的基图可比对（各臂管线不同），此项无法检验。")

    # ---------- 4. 逐臂基图均值 ----------
    print("\n" + "=" * 70)
    print("④ 各臂基图计数均值（起点是否可比）")
    for k in names:
        v = [data[k][i].get("n_base") for i in common]
        v = [x for x in v if x is not None]
        print(f"  {k:<6} 均值 {sum(v)/len(v):.2f}   逐行 {v}")
    print("\n  两臂基图**逐像素相同**却均值不同 -> 走 ①/③ 定位；"
          "\n  基图本就不同（跨管线）-> Δdelta 仍可比，但起点差异要在论文里说明。")


if __name__ == "__main__":
    main()
