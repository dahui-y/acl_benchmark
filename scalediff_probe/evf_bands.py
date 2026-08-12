"""把已有的 120 条 delta 按细 evf 档重切 —— 零 GPU，定位重复的"起燃点"。

为什么是下一步：三个语料都证明触发构图罕见（<3%），于是触发评测集
只能取**自然语料的 evf 尾部**。但去给 eval split 出基图要 ~5 小时 GPU，
而"尾部到底从哪个 evf 起才真的重复"这个问题，**手上这 120 条已经能给
初步答案**：C 层（evf>0.25）取的是全体 30 条，其中就含 tune 里 evf>0.5
的那 12 条。按 0.25/0.5/0.75 细分重读一遍 delta.jsonl 即可 —— 不生成
任何新图。

已知的两端：
    evf = 0        （LAION A 层）      delta_content = 0.00
    evf 0.75–0.94  （诊断集 lone 类）  excess = +5.40（另一把尺，仅作方向参照）
中间在哪里起燃，就看这一刀。

判读（写在看结果之前）：
    0.5+ 档 delta_content 明显抬头（≥1，即使 n 只有十来条）
        -> 起燃点在 0.5–0.75 之间，eval 尾部值得去取（评估 ~80 条）；
    0.5+ 档也平
        -> 起燃点更高或者 LAION 尾部构图与诊断集有质的差别，
           先取 eval 尾部里 evf>0.6 的看基图长相再决定烧不烧 4096。

    python scalediff_probe/evf_bands.py
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
    a = ap.parse_args()

    dp = Path(a.hi) / "delta.jsonl"
    if not dp.exists():
        sys.exit(f"没有 {dp}")
    rows = [json.loads(l) for l in dp.open()]
    scored = [r for r in rows if r.get("evf") is not None
              and r["stratum"] != "D_undef"]
    mean = lambda v: sum(v) / max(len(v), 1)

    bands = [(0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    print(f"{len(scored)} 条有 evf（盲区层不进）\n")
    print(f"{'evf 档':<14}{'n':>4}{'delta_content':>15}{'delta_raw':>11}"
          f"{'>=1 的':>7}{'>=3 的':>7}")
    print("-" * 58)
    for lo, hi in bands:
        if lo == hi:
            g = [r for r in scored if r["evf"] == 0]
            tag = "= 0"
        else:
            g = [r for r in scored if lo < r["evf"] <= hi]
            tag = f"({lo}, {hi if hi <= 1 else 1}]"
        if not g:
            print(f"{tag:<14}{0:>4}   (空)")
            continue
        dc = [r["delta_content"] for r in g]
        print(f"{tag:<14}{len(g):>4}{mean(dc):>15.2f}"
              f"{mean([r['delta_raw'] for r in g]):>11.2f}"
              f"{sum(1 for x in dc if x >= 1):>7}"
              f"{sum(1 for x in dc if x >= 3):>7}")

    top = sorted(scored, key=lambda r: -r["evf"])[:8]
    print("\nevf 最高的 8 条（逐条看，n 太小时均值骗人）：")
    for r in top:
        print(f"  evf={r['evf']:.2f}  dc={r['delta_content']:+d}  "
              f"raw={r['delta_raw']:+d}  {r['prompt'][:56]}")
    print("""
判读（预注册在脚本头）：0.5+ 档抬头 -> 去取 eval 尾部（~80 条）跑触发评测；
0.5+ 档也平 -> 先看 eval 尾部基图长相，再决定烧不烧 4096。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
