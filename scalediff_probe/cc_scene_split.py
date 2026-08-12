"""CoCoCount 基准内部的单变量对照：scene 行 vs no-scene 行的 evf。

这是 `caption_struct` 那个假设（"场景短语 -> 空视野"）能得到的**最干净的
检验**：同一批物体、同一批数字、同一个基准，唯一在动的变量是场景短语，
分组字段还是基准作者自己标的。

聚合数已经很不妙（200 条 p90 = 0.00，触发 0.6%，**比裸模板的 GenEval
还低**），这里把预注册的 H1 在正主对照上判掉：

    H1  scene 组 evf 均值明显高于 no-scene 组（至少 2 倍且差 >= 0.05）
    判据来自 caption_struct 的预注册：不成立 -> **结构解释被否**，
    "换承载面"不得再拿它当理由。

    python scalediff_probe/cc_scene_split.py
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "cococount_base"))
    ap.add_argument("--items", default=str(root / "count_cococount.json"))
    a = ap.parse_args()

    items = json.loads(Path(a.items).read_text())["items"]
    cands = sorted(Path(a.base).glob("trigger_all_*.jsonl"))
    if not cands:
        sys.exit(f"没有 {a.base}/trigger_all_*.jsonl")
    src = max(cands, key=lambda c: sum(1 for _ in c.open()))
    rows = [json.loads(l) for l in src.open()]
    print(f"用 {src.name}   {len(rows)} 条")

    mean = lambda v: sum(v) / max(len(v), 1)
    groups = {"scene": [], "no_scene": []}
    undef = {"scene": 0, "no_scene": 0}
    for r in rows:
        it = items[r["idx"]]
        g = "scene" if it.get("scene") else "no_scene"
        if r["nbox"] >= 1:
            groups[g].append(r)
        else:
            undef[g] += 1

    print(f"\n{'组':<10}{'n':>5}{'nbox=0':>8}{'evf均值':>10}"
          f"{'evf>0':>8}{'evf>0.25':>10}{'evf>0.5':>9}")
    print("-" * 52)
    for g in ("scene", "no_scene"):
        v = [r["evf"] for r in groups[g]]
        if not v:
            continue
        print(f"{g:<10}{len(v):>5}{undef[g]:>8}{mean(v):>10.3f}"
              f"{sum(1 for x in v if x > 0):>8}"
              f"{sum(1 for x in v if x > .25):>10}"
              f"{sum(1 for x in v if x > .5):>9}")

    # R=8 那一档也看（存了 evf_R）
    if groups["scene"] and groups["scene"][0].get("evf_R"):
        print("\nR=8（8192²）下同一对照：")
        for g in ("scene", "no_scene"):
            v = [r["evf_R"].get("8", 0.0) for r in groups[g] if r.get("evf_R")]
            if v:
                print(f"  {g:<10} evf 均值 {mean(v):.3f}   "
                      f">0.25 的 {sum(1 for x in v if x > .25)}/{len(v)}")

    ms = mean([r["evf"] for r in groups["scene"]]) if groups["scene"] else 0
    mn = mean([r["evf"] for r in groups["no_scene"]]) if groups["no_scene"] else 0
    ok = ms >= 2 * mn and (ms - mn) >= 0.05
    print(f"\nH1（scene 组 evf 至少 2 倍于 no-scene 且差 >= 0.05）："
          f"{ms:.3f} vs {mn:.3f}  -> " + ("**成立**" if ok else "**不成立**"))
    if not ok:
        print("  按 caption_struct 的预注册：**结构解释被否**。"
              "\n  '场景短语'不驱动空视野 —— SDXL 在 1024 上不管有没有"
              "'on the grass'\n  都倾向把物体铺满画面。触发区的真正变量是"
              "**基图构图本身（evf）**，\n  它不能从 caption 预测，"
              "只能在基图上量 —— 而这正是门本来的定义。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
