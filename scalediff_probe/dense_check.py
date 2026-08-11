"""被主指标排除的 10 条（crowd + texture）到底能不能说点什么。

主指标是 excess = 检出数 - prompt 声明的基数。"many cars" / "thousands of
blooms" 没声明数字，CARD=None，所以算不出 excess —— 这 10 条被排除了。

排除规则本身是干净的（CARD 在 subject_phrases.py 里定死，早于任何主结果），
**但排除掉的恰好是检测器表现最差的密集场景，说"不影响结论"太轻巧。**
诚实的说法是：我们对拥挤场景没有定量结论。

不过还有一个量可用，而且不需要跑 GPU：

    delta = 4096 计数 - 基图计数

基图两个 arm **逐字节相同**（90/90 已验证），所以 delta 之差完全归因于
外推阶段。没有"应该几辆车"的真值，但"比基图多出来几个"是有意义的。

必须同时声明的两条限制：
  ① 密集场景的检测本身不可靠（一框套两辆、半个框、漏检），所以这是
     **补充分析**，不能当主结果；
  ② delta 跨分辨率比较，尽管 tile=width/4 已修掉 4 倍表观尺度偏差，
     残余漂移约 2.0（见 count_objects 的注释）—— 所以看的是**两个 arm
     的 delta 之差**，那个漂移在相减时抵消。

    python scalediff_probe/dense_check.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_phrases import CARD                       # noqa: E402


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def cnt(rec, res):
    c = rec["counts"]
    return c.get(str(res), c.get(res, 0))


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    keys = sorted(set(A) & set(B))
    dense = [k for k in keys if CARD[k[0]] is None]
    if not dense:
        sys.exit("没有 card=None 的行")

    # 基图两个 arm 应当逐字节相同 -> 基图计数也必须相同。先自检。
    bad = [k for k in dense if cnt(A[k], 1024) != cnt(B[k], 1024)]
    print(f"基图计数一致性自检：{len(dense)-len(bad)}/{len(dense)} 相同"
          + ("" if not bad else f"   **不一致的行 {bad} —— 先查这个**"))

    print(f"\n{'idx':<5}{'seed':>6}{'cat':<10}{'基图':>6}"
          f"{'4096 基线':>10}{'4096 我们':>10}{'delta 基线':>11}"
          f"{'delta 我们':>11}{'差':>6}")
    print("-" * 76)
    da, db = [], []
    for k in dense:
        c0 = cnt(A[k], 1024)
        ca, cb = cnt(A[k], 4096), cnt(B[k], 4096)
        da.append(ca - c0); db.append(cb - c0)
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<10}{c0:>6}{ca:>10}{cb:>10}"
              f"{ca-c0:>+11}{cb-c0:>+11}{(cb-c0)-(ca-c0):>+6}")

    print(f"\n{'组':<12}{'n':>4}{'delta 基线':>12}{'delta 我们':>12}{'差':>8}")
    for name in ("crowd", "texture", "全部"):
        ks = [i for i, k in enumerate(dense)
              if name == "全部" or A[k]["cat"] == name]
        if not ks:
            continue
        ma = np.mean([da[i] for i in ks]); mb = np.mean([db[i] for i in ks])
        print(f"{name:<12}{len(ks):>4}{ma:>12.2f}{mb:>12.2f}{mb-ma:>+8.2f}")

    print("""
读法（这是补充分析，不是主结果）：
  两个 arm 的 delta 差 ≈ 0  -> 我们的介入在密集场景上没有可测的作用。
     这与门的设计一致（主体铺满画面 -> 空视野比例低 -> 关门 -> 不介入），
     也与 gate_eval 里 crowd/texture 15/15 中 13 行关门吻合。
  差为负（我们更少）        -> 密集场景上也有抑制作用，可作为补充证据。
  差为正（我们更多）        -> **要查**，说明介入在不该起作用的地方起了作用。

无论哪种，limitation 里那句话都要写成：
  "我们的计数指标只覆盖 prompt 声明了基数的 20/30 条；被排除的 10 条
   是密集与纹理场景，我们对这类场景没有定量结论，只给出 delta 的补充分析。"
不能写成"不影响结论"。""")


if __name__ == "__main__":
    main()
