"""计数基准：用现成的、可引用的，不自己从 caption 挖。

两版从 LAION alt-text 挖计数的尝试都失败了（见 fetch_eval_prompts.py 顶部
的失败清单）。根因不是正则不够好 —— **LAION 里的数字绝大多数不描述画面**
（型号、规格、容量、排行榜、住宿人数）。

而这条线根本没人从 caption 挖：

  GenEval (NeurIPS'23 D&B)   counting 任务用**模板 prompt**，
      静态文件 prompts/evaluation_metadata.jsonl，
      形如 {"tag":"counting","include":[{"class":"dog","count":3}],
            "prompt":"a photo of three dogs"}
  CountGen / Make It Count (CVPR'25)   专门造 CoCoCount，
      由 dataset/create_data_CoCoCount.py 生成，
      形如 "A photo of four donuts on the road"（**带场景短语**）

计数写在 prompt 里，不存在抽错的可能。

**两个都取，而且这本身是律的又一次检验：**
  CoCoCount 带场景短语 -> 主体可能只占画面一小部分 -> 空视野比例高
      -> 按律**应当有重复，我们的方法应当有效**；
  GenEval 是裸模板（"a photo of three dogs"）-> 主体多半占满画面
      -> 空视野比例低 -> 按律**本就不该有重复，门应当关闭、逐字节不动**。
  这是事前预测，不是事后解释 —— 两批的空视野比例可以在基图上直接量。

    python scalediff_probe/fetch_count_bench.py
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

GENEVAL_URL = ("https://raw.githubusercontent.com/djghosh13/geneval/main/"
               "prompts/evaluation_metadata.jsonl")
COCOCOUNT_REPO = "https://github.com/Litalby1/make-it-count.git"


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def geneval(out_dir, tune_frac=0.2, seed=0):
    """GenEval 的 counting 子集。静态文件，无依赖。"""
    print(f"取 GenEval: {GENEVAL_URL}")
    raw = get(GENEVAL_URL).decode("utf-8")
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    cnt = [r for r in rows if r.get("tag") == "counting"]
    print(f"  全部 {len(rows)} 条，counting {len(cnt)} 条")
    if not cnt:
        print("  **没有 counting 标签 —— 文件格式可能变了，先看几行原始内容：**")
        for l in raw.splitlines()[:3]:
            print("   ", l[:160])
        return None

    items = []
    for r in cnt:
        inc = r.get("include", [])
        if len(inc) != 1:          # 只收单类计数，多类的没法和我们的指标对齐
            continue
        items.append({"prompt": r["prompt"],
                      "card": inc[0]["count"],
                      "subject": inc[0]["class"],
                      "source": "geneval"})
    import random
    random.Random(seed).shuffle(items)
    k = int(len(items) * tune_frac)
    for i, it in enumerate(items):
        it["split"] = "tune" if i < k else "eval"
    p = Path(out_dir) / "count_geneval.json"
    p.write_text(json.dumps({
        "source": "GenEval (NeurIPS 2023 D&B), prompts/evaluation_metadata.jsonl",
        "tag": "counting", "n": len(items),
        "note": "裸模板，无场景短语；按律主体多半占满画面 -> 门应当关闭",
        "items": items}, ensure_ascii=False, indent=1))
    print(f"  写出 {p}   {len(items)} 条（tune {k} / eval {len(items)-k}）")
    for it in items[:5]:
        print(f"    - [{it['card']} {it['subject']}] {it['prompt']}")
    return items


def cococount_hint():
    print(f"""
CoCoCount 需要克隆仓库后用它自己的生成脚本（**不要我们重写，
重写就等于自造基准，失去可引用性**）：

    git clone {COCOCOUNT_REPO}
    cd make-it-count
    python dataset/create_data_CoCoCount.py \\
        --output_directory <out.json> --N_samples 400

生成的 prompt 形如 "A photo of four donuts on the road"，
文件名规范 {{count}}__{{class}}__{{prompt}}.png。
拿到 json 后跑：
    python scalediff_probe/fetch_count_bench.py --cococount <out.json>
""")


def adapt_cococount(path, out_dir, tune_frac=0.2, seed=0):
    """把 CoCoCount 的输出转成我们的统一格式。

    它的字段名未核实，所以这里做**宽松适配 + 明确报错**，
    不猜结构（这轮已经因为猜字段名/仓库名栽过两次）。
    """
    raw = json.loads(Path(path).read_text())
    rows = raw if isinstance(raw, list) else raw.get("data", raw.get("items"))
    if not isinstance(rows, list):
        print(f"**读不懂 {path} 的结构。顶层键：**"
              f"{list(raw)[:10] if isinstance(raw, dict) else type(raw)}")
        return None
    items = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        prompt = r.get("prompt") or r.get("text") or r.get("caption")
        card = r.get("count") or r.get("number") or r.get("expected_count")
        subj = r.get("class") or r.get("class_name") or r.get("object")
        if prompt and card and subj:
            items.append({"prompt": prompt, "card": int(card),
                          "subject": str(subj), "source": "cococount"})
    if not items:
        print(f"**没解析出条目。第一行长这样：** {rows[0] if rows else '(空)'}")
        return None
    import random
    random.Random(seed).shuffle(items)
    k = int(len(items) * tune_frac)
    for i, it in enumerate(items):
        it["split"] = "tune" if i < k else "eval"
    p = Path(out_dir) / "count_cococount.json"
    p.write_text(json.dumps({
        "source": "CoCoCount, CountGen/Make It Count (CVPR 2025)",
        "n": len(items),
        "note": "带场景短语；按律主体可能只占一小部分 -> 门应当开启",
        "items": items}, ensure_ascii=False, indent=1))
    print(f"  写出 {p}   {len(items)} 条（tune {k} / eval {len(items)-k}）")
    for it in items[:5]:
        print(f"    - [{it['card']} {it['subject']}] {it['prompt'][:70]}")
    return items


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root))
    ap.add_argument("--cococount", default=None,
                    help="CoCoCount 生成脚本的输出 json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    g = geneval(a.out, seed=a.seed)
    if a.cococount:
        print("\n适配 CoCoCount:")
        adapt_cococount(a.cococount, a.out, seed=a.seed)
    else:
        cococount_hint()

    print("""
事前预测（写在跑之前，两批各自的空视野比例在基图上量）：
  CoCoCount（带场景）  空视野比例高 -> 门开 -> 计数 MAE 明显下降；
  GenEval（裸模板）    空视野比例低 -> 门关 -> **逐字节不动，MAE 不变**。
两条都成立才说明律和门都对；GenEval 那批若也大幅变化，说明门没起作用，
要回头查阈值。""")
    return 0 if g else 1


if __name__ == "__main__":
    sys.exit(main())
