"""caption 的句法结构能不能预测空视野比例 —— 预注册的假设，跑之前写死。

`caption_audit` 报出 tune 的 token 数中位数只有 **13**（≈8 个词）。
回头看那些 caption：

    Paneer Biryani with Cucumber Raita
    Lime Green Hot Rod Classic Round Sticker
    How to Make Cheddar Cheese Popcorn

**全是"只有主体、没有场景"。** 而我们那 30 条诊断 prompt 是
"a lone hiker standing **on a rocky ridge**, distant snow-capped mountains"
—— **主体 + 场景，两样都有**。

律要凑齐的条件是：受限视野里**没有主体、却仍收到完整文本条件**，
于是把那块当空白画布重画。这需要 prompt **同时**说了主体和场景：
场景让模型在空处有东西可画，文本里的主体又在每个视野里坚持要一个主体。
**只有主体没有场景 -> SDXL 把主体铺满画面（商品图构图）-> 不存在空视野。**

所以低触发率可能不是"摄影惯例"这种模糊说法，而是**caption 的句法结构**。
这个说法可证伪，本脚本就是去证伪它。

**预注册（写在看结果之前）：**

  H1  含方位介词短语的 caption，其 evf 的**均值更高**；
  H2  evf 与 caption 长度**正相关**，Spearman ρ > 0.2；
  判据：两条都成立 -> 结构解释站得住，可以据此说明为什么
        LAION-aesthetic 低触发、而带场景短语的基准应当高触发；
        只成立一条 -> 记为弱证据，不写进主线；
        都不成立 -> **结构解释被否**，回到"就是语料如此"，
        并且不得再用它去论证换承载面的合理性。

**这条预测不是事后找补**：`fetch_count_bench.py` 的文档里，在拿到任何 LAION
结果之前就写下了"CoCoCount 带场景短语 -> 门该开 / GenEval 裸模板 -> 门该关"。
LAION-aesthetic 属于"裸主体"那一类，按我们自己的理论本来就该低触发。

    python scalediff_probe/caption_struct.py --split tune
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import spearman                      # noqa: E402

# 方位介词：**只收能引出"场景"的那些**。"with / of / for" 不收 ——
# "Biryani **with** Raita" 说的是主体的构成，不是它在哪里。
LOCATIVE = {
    "on", "in", "at", "near", "beside", "among", "amongst", "against",
    "over", "under", "underneath", "beneath", "across", "along", "atop",
    "behind", "inside", "outside", "around", "between", "beyond", "amid",
    "overlooking", "surrounded",
}
# 这些搭配里的 "on/in/at" 不表方位，是商业/网页套话，必须排掉，
# 否则 "Jeep **for sale on** eBay" 会被算成场景。
STOP_AFTER = {
    "sale", "ebay", "etsy", "amazon", "pinterest", "instagram", "facebook",
    "line", "stock", "white", "black", "sale.", "order", "demand",
    "display", "request",
}


def has_locative(caption):
    """有没有引出场景的方位短语。**规则很粗，粗得要说清楚**：
    只看介词后面紧跟的那个词是不是套话，不做句法分析。"""
    w = re.findall(r"[a-z']+", caption.lower())
    for i, t in enumerate(w):
        if t in LOCATIVE:
            nxt = w[i + 1] if i + 1 < len(w) else ""
            nxt2 = w[i + 2] if i + 2 < len(w) else ""
            # 跳过冠词再看一个词
            probe = nxt2 if nxt in {"a", "an", "the"} else nxt
            if probe and probe not in STOP_AFTER:
                return True
    return False


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--split", default="tune")
    ap.add_argument("--tag", default=None,
                    help="trigger_select 写出的那个 tag；省略则自动找")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    base = Path(a.base)
    cands = ([base / f"trigger_{a.tag}.jsonl"] if a.tag
             else sorted(base.glob(f"trigger_{a.split}_*.jsonl")))
    cands = [p for p in cands if p.exists()]
    if not cands:
        sys.exit(f"没找到 {base}/trigger_{a.split}_*.jsonl，先跑 trigger_select.py")
    src = cands[0]
    rows = [json.loads(l) for l in src.open()]
    rows = [r for r in rows if r.get("nbox", 0) >= 1]      # 无定义的不进统计
    if len(rows) < 20:
        sys.exit(f"{src} 里可用行只有 {len(rows)} 条，太少")
    print(f"用 {src.name}，有定义的 {len(rows)} 条")

    from caption_audit import load_tokenizer
    tok = load_tokenizer()
    for r in rows:
        r["ntok"] = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        r["loc"] = has_locative(r["prompt"])

    loc = [r for r in rows if r["loc"]]
    noloc = [r for r in rows if not r["loc"]]
    mean = lambda v: sum(v) / max(len(v), 1)

    print(f"\n含方位短语 {len(loc)} 条（{len(loc)/len(rows):.1%}）"
          f"   不含 {len(noloc)} 条")
    print(f"  H1  evf 均值：含 {mean([r['evf'] for r in loc]):.3f}"
          f"   vs 不含 {mean([r['evf'] for r in noloc]):.3f}"
          f"   -> " + ("**方向对**" if loc and noloc and
                       mean([r['evf'] for r in loc]) >
                       mean([r['evf'] for r in noloc]) else "**方向不对**"))

    rho = spearman([r["ntok"] for r in rows], [r["evf"] for r in rows])
    print(f"  H2  Spearman(token 数, evf) = {rho:+.3f}"
          f"   -> " + ("**过**" if rho > 0.2 else "**没过**（判据 > 0.2）"))

    print("\n按 token 数分组的 evf 均值：")
    for lo, hi in [(0, 8), (8, 14), (14, 22), (22, 999)]:
        g = [r for r in rows if lo <= r["ntok"] < hi]
        if g:
            print(f"  {lo:>3}–{hi if hi < 999 else '∞':<4} n={len(g):>4}"
                  f"  evf {mean([r['evf'] for r in g]):.3f}"
                  f"  含方位 {mean([r['loc'] for r in g]):.0%}")

    top = sorted(rows, key=lambda r: -r["evf"])[:a.show]
    print(f"\nevf 最高的 {len(top)} 条（看结构解释对不对得上）：")
    for r in top:
        print(f"  evf={r['evf']:.2f} tok={r['ntok']:>3} "
              f"{'方位' if r['loc'] else '  —  '}  {r['prompt'][:60]}")

    ok1 = bool(loc) and bool(noloc) and (mean([r["evf"] for r in loc]) >
                                         mean([r["evf"] for r in noloc]))
    ok2 = rho > 0.2
    print("\n判读（预注册）：")
    if ok1 and ok2:
        print("  两条都成立 -> **结构解释站得住**。可以据此说明"
              "LAION-aesthetic 为何低触发、带场景短语的基准为何应当高触发。")
    elif ok1 or ok2:
        print("  只成立一条 -> **弱证据，不写进主线**，仅作讨论。")
    else:
        print("  都不成立 -> **结构解释被否**。回到'就是语料如此'，"
              "并且**不得再用它去论证换承载面的合理性**。")
    print("\n注意：`has_locative` 是很粗的规则（只看介词后一个词是不是套话，"
          "不做句法分析）。它只用于分组统计，不进方法。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
