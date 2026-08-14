"""闸门表：ScaleDiff 基线 / v1 / v1.2 / AccDiffusion 在**同一批 idx** 上对齐。

这是整个去留判断的交付物，也是论文主表的缩微版。

读法上的三条纪律：

1. **只比 Δdelta，不比绝对计数。** 这是跨骨干比较（AccDiffusion 建在
   DemoFusion 上，我们建在 ScaleDiff 上），两边的 1024 基图**不是同一张**
   （管线不同、negative prompt 不同）。Δdelta 各自与**自己的基图**相减，
   量的是"放大过程新加了多少重复"，这才是可比的量。
2. **Rep⁺ 与 Dmg⁻ 分开看，两列都要好才算赢。** 签名均值允许"把主体删光"
   （负 delta）冲抵重复（正 delta）—— 该缺陷已于 2026-08-14 在闸门实验
   **之前**锁死（§3.14d）。
3. **基图计数一并报**。若两边的基图主体数差得离谱，说明起点就不同，
   Δdelta 的可比性要打折 —— 这个必须自己先查，不能等审稿人查。

    python scalediff_probe/gate_table.py
    python scalediff_probe/gate_table.py --arms hi=parti_hi ours=parti_v12 acc=parti_acc
"""

import argparse
import json
import os
from pathlib import Path


def load(d):
    p = Path(d) / "vlm_delta.jsonl"
    if not p.exists():
        return {}
    out = {}
    for ln in p.open():
        r = json.loads(ln)
        if r.get("delta") is not None:
            out[int(r["idx"])] = r
    return out


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="名字=目录，默认 hi/v1/v12/acc 四臂")
    a = ap.parse_args()

    if a.arms:
        arms = {}
        for s in a.arms:
            k, v = s.split("=", 1)
            arms[k] = v if "/" in v else str(root / v)
    else:
        arms = {"ScaleDiff 基线": root / "parti_hi",
                "v1": root / "parti_v1",
                "v1.2 (ours)": root / "parti_v12",
                "AccDiffusion": root / "parti_acc"}

    data = {k: load(v) for k, v in arms.items()}
    data = {k: v for k, v in data.items() if v}
    if len(data) < 2:
        raise SystemExit("至少要两臂才有得比。检查各目录下有没有 vlm_delta.jsonl")

    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if not common:
        raise SystemExit("各臂没有共同的 idx")
    print(f"共同 idx {len(common)} 条：{common}\n")

    # ---------- 逐行 ----------
    names = list(data)
    w = max(len(n) for n in names) + 2
    print(f"{'idx':>5} " + "".join(f"{n:>{w}}" for n in names) + "  prompt")
    for i in common:
        row = "".join(f"{data[n][i]['delta']:>{w}}" for n in names)
        p = next(data[n][i].get("prompt", "") for n in names)
        print(f"{i:>5} {row}  {p[:44]}")

    # ---------- Rep+ / Dmg- ----------
    print(f"\n{'='*72}")
    print(f"{'方法':<16}{'Rep+ 重复':>11}{'Dmg- 误伤':>11}"
          f"{'正/负实例':>12}{'基图均值':>10}{'≥1 行数':>9}")
    res = {}
    for n in names:
        d = data[n]
        pos = sum(max(d[i]["delta"], 0) for i in common)
        neg = sum(max(-d[i]["delta"], 0) for i in common)
        nb = [d[i].get("n_base") for i in common]
        nb = [x for x in nb if x is not None]
        k = len(common)
        res[n] = (pos / k, neg / k)
        print(f"{n:<16}{pos/k:>11.3f}{neg/k:>11.3f}"
              f"{f'{pos} / {neg}':>12}"
              f"{(sum(nb)/len(nb) if nb else float('nan')):>10.2f}"
              f"{sum(1 for i in common if d[i]['delta'] >= 1):>9}")

    # ---------- 闸门判读 ----------
    base = next((n for n in names if "基线" in n or n == "hi"), None)
    ours = next((n for n in names if "ours" in n or "v1.2" in n), None)
    acc = next((n for n in names if "Acc" in n or n == "acc"), None)
    if not (acc and ours):
        return
    print(f"\n{'='*72}\n闸门判读（判据在跑之前就写死，见 acc_batch.py 尾部）")
    ra, ro = res[acc][0], res[ours][0]
    rb = res[base][0] if base else None
    print(f"  AccDiffusion Rep+ = {ra:.3f}     我们 Rep+ = {ro:.3f}"
          + (f"     ScaleDiff 基线 Rep+ = {rb:.3f}" if rb is not None else ""))
    if ra < 0.05:
        print("  -> **对手已在零**。继续压我们的残留没有意义，"
              "这条线判死，转向。")
    elif rb is not None and ra > rb:
        print("  -> **AccDiffusion 比未处理的 ScaleDiff 基线还差**。"
              "卖点变成'它以重复为卖点却从未被测量过，我们测了，没解决'"
              "—— 这是最有杀伤力的一种，但务必先眼睛复核几张图再下结论。")
    elif ro < ra:
        print("  -> **我们更低**。在场上，主表照打。仍需注意："
              "跨骨干比较，且此子集是按基线重复最重挑的（非无偏抽样），"
              "正式表格要跑全集。")
    else:
        print("  -> **对手更低**。我们的机制主张要重新审视；"
              "此时'1 次前向 vs N 次'的成本轴与'窗口注意力家族结构上"
              "跑不了'才是仅剩的差异化，须据此重排卖点。")
    print("\n  提醒：此子集共 12 条，是按 ScaleDiff 基线上重复最重挑出来的，"
          "\n  属**筛查性**测试，够分辨'≈0 vs 明显>0'这种粗判，"
          "不够做精细排序。")


if __name__ == "__main__":
    main()
