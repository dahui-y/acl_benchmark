"""重复程度 vs 主体覆盖率 —— 把"lone 坏、portrait 没事"从分类观察变成连续关系。

起因：用户拿一组肖像图问"这类图就没有重复现象"。对，而且我们自己的数据早就
说了：基线 MAE 里 lone 是 5.40（5/5 有重复），portrait / structure / empty
都是 0.20（1/5）。我把动机写成普适的，少了后半截。

机制上为什么：重复要发生，需要一个受限视野【里面本来没有主体，却仍然收到
完整的文本条件】—— 它于是把那个视野当成空白画布，照 prompt 画一个完整场景。

    主体占满画面（肖像）：每个视野里都有主体的一部分 -> 画的是"延续" -> 不重复
    主体很小（山脊上一个人）：绝大多数视野里没有主体 -> 每个都新建 -> 重复

所以支配变量不只是邻域/画布比，而是

    不含主体的邻域数 ≈ R² x (1 − 主体覆盖率)

R² 是外推倍数（控制不了），主体覆盖率是内容决定的 —— 而 cov 就是在测它。

这个脚本不跑 GPU：cov 在 method 的 manifest 里（来自 phase 1，两个 arm 相同），
excess 在两边的 counts.json 里。

    python scalediff_probe/law_check.py
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"),
                    help="只用它的 manifest 取 cov（phase 1 与基线相同）")
    a = ap.parse_args()

    cj = Path(a.base) / "counts.json"
    mj = Path(a.new) / "manifest.jsonl"
    if not cj.exists() or not mj.exists():
        sys.exit(f"需要 {cj} 和 {mj}")

    C = {r["idx"]: r for r in json.loads(cj.read_text())}
    M = {json.loads(l)["idx"]: json.loads(l) for l in mj.open()}

    rows = []
    for i in sorted(set(C) & set(M)):
        cov = M[i].get("cov", -1)
        exc = C[i].get("excess")
        if cov is None or cov < 0 or exc is None:
            continue
        rows.append((i, C[i]["cat"], M[i].get("head"), cov, abs(exc),
                     C[i]["counts"].get("4096", 0)))

    if not rows:
        sys.exit("没有同时有 cov 和 excess 的行（empty 类 head=None 没有 cov）")

    print(f"{'idx':<5}{'cat':<11}{'head':<13}{'cov':>8}{'|excess|':>10}{'4096计数':>9}")
    print("-" * 56)
    for i, cat, head, cov, e, n4 in sorted(rows, key=lambda r: r[3]):
        print(f"{i:<5}{cat:<11}{str(head):<13}{cov:>7.1%}{e:>10}{n4:>9}")

    xs = [r[3] for r in rows]
    ys = [r[4] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n) or 1e-9
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n) or 1e-9
    r_p = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)

    # Spearman：秩相关，对非线性单调关系更合适，也不怕 lone 那几个大值拉偏
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        rk = [0.0] * len(v)
        k = 0
        while k < len(order):
            j = k
            while j + 1 < len(order) and v[order[j + 1]] == v[order[k]]:
                j += 1
            avg = (k + j) / 2 + 1
            for t in range(k, j + 1):
                rk[order[t]] = avg
            k = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mrx, mry = sum(rx) / n, sum(ry) / n
    srx = math.sqrt(sum((v - mrx) ** 2 for v in rx) / n) or 1e-9
    sry = math.sqrt(sum((v - mry) ** 2 for v in ry) / n) or 1e-9
    r_s = sum((u - mrx) * (v - mry) for u, v in zip(rx, ry)) / (n * srx * sry)

    print(f"\nn={n}   Pearson r = {r_p:+.3f}   Spearman ρ = {r_s:+.3f}")
    print("预期：**负相关** —— 主体覆盖率越低，不含主体的邻域越多，重复越严重。")

    lo = [r for r in rows if r[3] < 0.15]
    hi = [r for r in rows if r[3] >= 0.15]
    if lo and hi:
        print(f"\n  cov < 15%（主体局部化）  n={len(lo):<3} 平均 |excess| = "
              f"{sum(r[4] for r in lo)/len(lo):.2f}")
        print(f"  cov >= 15%（主体铺开）   n={len(hi):<3} 平均 |excess| = "
              f"{sum(r[4] for r in hi)/len(hi):.2f}")
        print("\n  这条分界线若清晰，就是【适用性门】的依据 —— 主体铺满画面时"
              "根本不存在\n  '没有主体的邻域'，也就没有要抑制的东西，方法不该介入。")


if __name__ == "__main__":
    main()
