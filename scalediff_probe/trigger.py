"""触发条件：直接在基图上数"有多少受限视野里本来没有主体"。

一个量办两件事：

  A【验证律】失效不是普适的 —— 特写肖像三档都只有一个人，lone 类才重复。
    机制上：重复要发生，需要一个受限视野里本来没有主体、却仍收到完整文本条件，
    它于是把那块当空白画布重画一遍。所以自变量是**空视野的数量**。

  B【回答风险】"如果触发条件在评测集里很罕见，头条表就不会动。"
    这个量能直接统计一个 prompt 语料里有多大比例落在触发区。

为什么用它而不是 cov：
    cov 来自 phase 1 的注意力图，依赖 head word（要 strip_subject 那条脆规则），
    而且经过逐图 min-max 归一化，绝对含义已经丢了。
    **空视野比例是机制里那个变量本身**：在【基图】上跑检测器，按 R x R 切块，
    数有多少块不含主体框。不需要注意力图、不需要 head word、不需要跑放大阶段。

        空视野比例 = 不含主体框的 tile 数 / R²

    R=4 对应 4096²（16 块），R=2 对应 2048²（4 块）。

预注册的判据（跑之前定死）：
    · 律：空视野比例与 4096² 的 |excess| 应当**正相关**（Spearman ρ > 0.5）
    · 普遍性：语料里 **≥25%** 的 prompt 落在空视野比例 > 0.75 的区间
      -> 触发条件足够常见，头条表值得跑
      < 10% -> 条件太罕见，必须改成分层报告为主、总表为辅

    python scalediff_probe/trigger.py                    # 我们的 30 条
    python scalediff_probe/trigger.py --R 2              # 2048² 那一档
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector          # noqa: E402


def empty_view_fraction(boxes, W, H, R):
    """把图切成 R x R 块，返回不含任何框的块所占比例。

    "含"的判定用交集面积 > 块面积的 5% —— 框只擦到块角一点点，那块对模型来说
    仍然基本是空白画布，仍会长出一个新主体。
    """
    if R <= 1:
        return 0.0
    tw, th = W / R, H / R
    empty = 0
    for r in range(R):
        for c in range(R):
            x0, y0, x1, y1 = c * tw, r * th, (c + 1) * tw, (r + 1) * th
            hit = False
            for b in boxes:
                ix = max(0.0, min(x1, b[2]) - max(x0, b[0]))
                iy = max(0.0, min(y1, b[3]) - max(y0, b[1]))
                if ix * iy > 0.05 * tw * th:
                    hit = True
                    break
            empty += not hit
    return empty / (R * R)


def spearman(xs, ys):
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
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in rx) / n) or 1e-9
    sy = math.sqrt(sum((v - my) ** 2 for v in ry) / n) or 1e-9
    return sum((u - mx) * (v - my) for u, v in zip(rx, ry)) / (n * sx * sy)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(root / "batch"))
    ap.add_argument("--R", type=int, default=4, help="4 -> 4096²（16 块）；2 -> 2048²")
    ap.add_argument("--hi", type=int, default=0, help="0 = 用 1024*R 那一档的 excess")
    a = ap.parse_args()

    batch = Path(a.batch)
    cj = batch / "counts.json"
    if not cj.exists():
        sys.exit(f"没有 {cj}，先跑 count_objects.py --batch {batch}")
    C = {(r["idx"], r["seed"]): r for r in json.loads(cj.read_text())}
    M = {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
         for l in (batch / "manifest.jsonl").open()}
    hi = a.hi or 1024 * a.R

    det = Detector()
    rows = []
    for k in sorted(set(C) & set(M)):
        r, m = C[k], M[k]
        f1 = m["files"].get("1024")
        card = r.get("card")
        if not f1 or card is None:
            continue                       # 只用 prompt 声明了基数的那些
        im = Image.open(batch / f1).convert("RGB")
        b, _ = det.detect(im, r["subject"])
        evf = empty_view_fraction(b, im.width, im.height, a.R)
        n_hi = r["counts"].get(str(hi), r["counts"].get(hi, 0))
        rows.append((k[0], k[1], r["cat"], r["subject"], len(b), evf,
                     abs(n_hi - card)))

    if not rows:
        sys.exit("没有可用的行")

    print(f"R={a.R}（{a.R*a.R} 块），excess 取 {hi}²\n")
    print(f"{'idx':<5}{'seed':>6}{'cat':<11}{'subj':<10}"
          f"{'基图框数':>9}{'空视野比例':>12}{'|excess|':>10}")
    print("-" * 66)
    for i, sd, cat, subj, nb, evf, e in sorted(rows, key=lambda t: -t[5]):
        print(f"{i:<5}{sd:>6}{cat:<11}{subj:<10}{nb:>9}{evf:>11.0%}{e:>10}")

    xs = [r[5] for r in rows]
    ys = [r[6] for r in rows]
    rho = spearman(xs, ys)
    print(f"\nn={len(rows)}   Spearman ρ(空视野比例, |excess|) = {rho:+.3f}")
    print("  判据：ρ > 0.5 -> 律成立（空视野越多，重复越严重）")
    print("  " + ("-> **过**" if rho > 0.5 else "-> **没过，这条律要重想**"))

    hi_g = [r for r in rows if r[5] > 0.75]
    lo_g = [r for r in rows if r[5] <= 0.75]
    if hi_g and lo_g:
        print(f"\n  空视野 > 75%  n={len(hi_g):<3} 平均 |excess| = "
              f"{sum(r[6] for r in hi_g)/len(hi_g):.2f}")
        print(f"  空视野 ≤ 75%  n={len(lo_g):<3} 平均 |excess| = "
              f"{sum(r[6] for r in lo_g)/len(lo_g):.2f}")

    # 普遍性 —— 这就是"触发条件罕不罕见"那个风险的答案
    frac = len(hi_g) / len(rows)
    print(f"\n触发区（空视野 > 75%）占比 = {frac:.0%}   n={len(rows)}")
    print("  判据：≥25% -> 条件足够常见，头条表值得跑；"
          "<10% -> 太罕见，必须以分层报告为主")
    print("  " + ("-> 足够常见" if frac >= 0.25
                  else "-> **太罕见，改分层报告为主**" if frac < 0.10
                  else "-> 中间地带，两种都报"))
    print("\n注意：我们这 30 条是【人为按六类均衡】造的，这个比例**不代表真实语料**。"
          "\n      要回答风险，必须在 LAION 那 1000 条上重跑这个脚本"
          "（只需基图，约 7s/条）。")


if __name__ == "__main__":
    main()
