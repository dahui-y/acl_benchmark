#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
另抽一批 prompt 当调参集 —— **不许在那 167 题上调参**。

    换了损失，`thresholds {0:1.3,10:1.2,20:1.15}` 和 `scale_factor=50` 就必须重定：
    原版损失的量纲是带 pos_weight=10 的 BCE（取值 0.8~2.4），我们的各项都是
    [0,1] 上的均值/最大值（取值 0~1）。沿用原值等于没设阈值。

    但重定阈值就是调参，而调参必须在**与评测集不相交**的 prompt 上做，
    否则报出来的数是在测试集上选出来的。这个项目一路都在守这条线
    （风格线上"不调参的那一档 q=0.5"也是同一个考虑）。

    做法：用**他们自己的生成脚本** `dataset/create_data_CoCoCount.py` 另抽一批，
    先 `random.seed(seed)` 使其可复现，再按 id 剔除与评测集重合的题。
    抽出来的 json 会存下来并进版本库 —— 他们的脚本没有 seed 参数，
    可复现性靠**保留那份 json**，不靠重跑。

用法：
    python count_probe/make_tune_set.py --n 60 --out $SD_OUT/count/tune.json
"""

import argparse
import json
import os
import random
import runpy
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIC = REPO / "help_code" / "make-it-count"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="调参集大小")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--exclude", nargs="+",
                    default=[str(MIC / "dataset" / "CoCoCount.json")],
                    help="要剔重的题目 json，可给多份 —— 新方法阶段必须同时排除"
                         "评测集**和已烧掉的旧调参集**（旧集上评过 16 个配置，"
                         "在它上面再选任何东西都是在挑噪声）")
    ap.add_argument("--drop-over9", action="store_true", default=True,
                    help="剔掉 N>9（官方代码整题跳过，调参用不上）")
    a = ap.parse_args()

    excl = set()
    for p in a.exclude:
        got = {f"{d['object']}_num={d['int_number']}_seed={d['seed']}"
               for d in json.load(open(p))}
        excl |= got
        print(f"剔重来源 {p}：{len(got)} 题")
    print(f"合计剔除名单 {len(excl)} 题")

    # 多抽一些，剔除后再截断
    want = a.n * 3
    random.seed(a.seed)
    with tempfile.TemporaryDirectory() as td:
        cwd, argv = os.getcwd(), sys.argv[:]
        os.chdir(MIC)
        sys.argv = ["create_data_CoCoCount.py",
                    "--output_directory", td, "--N_samples", str(want)]
        try:
            runpy.run_path(str(MIC / "dataset" / "create_data_CoCoCount.py"),
                           run_name="__main__")
        finally:
            os.chdir(cwd)
            sys.argv = argv
        js = list(Path(td).glob("*.json"))
        if not js:
            raise SystemExit(f"!! 他们的脚本没在 {td} 里生成 json")
        data = json.load(open(js[0]))

    out, seen = [], set()
    for d in data:
        i = f"{d['object']}_num={d['int_number']}_seed={d['seed']}"
        if i in excl or i in seen:
            continue
        if a.drop_over9 and d["int_number"] > 9:
            continue
        seen.add(i)
        out.append(d)
        if len(out) >= a.n:
            break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2)
    import collections
    print(f"调参集 {len(out)} 题 → {a.out}")
    print(f"  N 分布 {dict(sorted(collections.Counter(d['int_number'] for d in out).items()))}")
    print(f"  与剔除名单重合 0 题（已剔除）")
    print(f"\n⚠️ 把这份 json 提交进版本库。他们的脚本没有 seed 参数，"
          f"可复现性靠保留文件，不靠重跑。")


if __name__ == "__main__":
    main()
