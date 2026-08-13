"""触发集 delta 与预注册标签对账 —— ③ 的正式判读，全 CPU，秒出。

三件事：
  1. scenic / mid / flat 分组统计，对 parti_trigger_predictions.json 里
     预注册的两条门槛（scenic 组均值 >= +1.5；flat 组均值 <= +0.5）判读；
  2. 逐条表：tag、delta、主体词、眼睛终审的六个嫌疑样本单独标记，
     delta >= 3 的悬案置顶（眼睛最大只给到 +1，超出必须回看图）；
  3. 第二场空对照：眼睛判干净的样本（非嫌疑、非实锤）的 delta 分布 ——
     含"重新生成"扰动的噪声地板，与 --null 的重采样地板并列报。

    python scalediff_probe/parti_reconcile.py
"""

import json
import os
import sys
from pathlib import Path

# 眼睛终审（§5.49e）：实锤 / 疑似。583 是纹理泄漏，计数应读 0，单列。
CONFIRMED = {263: "+1 整尊女神（实锤）"}
SUSPECT = {157: "+1 人", 146: "+1 杯", 553: "+1 小象", 994: "+1 画框（弱）"}
TEXTURE = {583: "网纹克隆——计数看不见，应读 0"}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    hi = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "parti_hi"
    dp = hi / "vlm_delta.jsonl"
    if not dp.exists():
        sys.exit(f"没有 {dp} —— 先跑 vlm_count.py --delta")
    pred = json.loads((Path(__file__).resolve().parent /
                       "parti_trigger_predictions.json").read_text())["tags"]
    recs = [json.loads(l) for l in dp.open()]
    byidx = {r["idx"]: r for r in recs}

    mean = lambda v: sum(v) / max(len(v), 1)
    groups = {}
    for r in recs:
        t = pred.get(str(r["idx"]), {}).get("tag", "?")
        if r.get("delta") is not None:
            groups.setdefault(t, []).append(r["delta"])

    print("== 分组统计（预注册：scenic 均值 >= +1.5；flat 均值 <= +0.5）==")
    for t in ("scenic", "mid", "flat", "?"):
        v = groups.get(t)
        if not v:
            continue
        print(f"  {t:<7} n={len(v):<3} 均值 {mean(v):+.2f}   "
              f">=1 的 {sum(1 for d in v if d >= 1)}   "
              f">=3 的 {sum(1 for d in v if d >= 3)}")
    sc, fl = mean(groups.get("scenic", [0])), mean(groups.get("flat", [0]))
    print("\n  判读：", end="")
    if sc >= 1.5 and fl <= 0.5:
        print("scenic 高 + flat 低 -> 第三因子成立，门的判据升级为 evf×场景性")
    elif sc < 1.5 and fl <= 0.5 and sc > fl:
        print(f"中间态（scenic {sc:+.2f} 未到 +1.5，但方向 scenic > flat）->"
              " 稀疏 +1 泄漏图景成立，组均值门槛过严；按 §5.49e 的分布读法走")
    elif sc >= 1.5 and fl > 0.5:
        print("都高 -> evf 单独够，第三因子不需要")
    else:
        print("都低 -> §5.0 止损支，触发集舞台移 DemoFusion")

    print("\n== 悬案：delta >= 3（眼睛最大只给 +1，必须回看图）==")
    for r in sorted(recs, key=lambda r: -(r.get("delta") or 0)):
        if (r.get("delta") or 0) >= 3:
            print(f"  [{r['idx']:>4}] delta={r['delta']:+d}  "
                  f"n {r['n_base']}->{r['n_hi_dn']}  subj={r['subject']!r}"
                  f"\n         {r['prompt'][:70]}")

    print("\n== 眼睛终审样本的盖章情况 ==")
    for d_, label in ((CONFIRMED, "实锤"), (SUSPECT, "疑似"), (TEXTURE, "纹理")):
        for i, note in d_.items():
            r = byidx.get(i)
            if not r:
                continue
            got = r.get("delta")
            stamp = ("✓ 盖章" if label != "纹理" and (got or 0) >= 1 else
                     "✓ 按设计读 0" if label == "纹理" and got == 0 else
                     "✗ 未盖章" if label != "纹理" else f"? 读了 {got:+d}")
            print(f"  [{i:>4}] {label} {note:<28} delta={got:+d}  {stamp}")

    print("\n== 第二场空对照：眼睛判干净的样本（含重新生成扰动的地板）==")
    dirty = set(CONFIRMED) | set(SUSPECT) | set(TEXTURE)
    clean = [r["delta"] for r in recs
             if r["idx"] not in dirty and r.get("delta") is not None]
    if clean:
        nz = sum(1 for d in clean if d != 0)
        print(f"  n={len(clean)}  mean|Δ|={mean([abs(d) for d in clean]):.2f}  "
              f"bias={mean(clean):+.2f}  非零 {nz} 条"
              f"（重采样地板 --null 为全零；这里的非零 = 重新生成扰动"
              f" + 眼睛在缩略图上漏掉的真差异，逐条回看图分账）")
        for r in sorted(recs, key=lambda r: -abs(r.get("delta") or 0)):
            if r["idx"] not in dirty and (r.get("delta") or 0) != 0:
                print(f"    [{r['idx']:>4}] {r['delta']:+d}  "
                      f"subj={r['subject']!r}  {r['prompt'][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
