"""取 PartiPrompts（P2）+ 从本仓库的论文 PDF 里收割展示 prompt。

为什么是这两个来源（2026-08-12 定，接在 §5.49c 的裁决后面）：

  三个语料（LAION / GenEval / CoCoCount）都证明触发构图在评测语料里
  基本不存在；LAION 的 evf 尾部还被排版插画和素底商品图污染 ——
  **自然语料选不出触发构图，自己写 prompt 又不可辩护。** 剩下的正路：

  1. **PartiPrompts**：Parti 论文的标准 prompt 套件（~1600 条，带
     Category/Challenge 标签），社区公认、可引用。其中 Outdoor Scenes
     等类别正是广景构图。**我们不按类别筛选**（那也是挑），全量过
     基图 -> evf，让预注册的判据自己分层；类别标签只用于事后分组画像。
  2. **展示 prompt（showcase）**：从这条线论文的 PDF 里抽出图注 prompt
     （"A cute corgi on the lawn" 等）。它们直接支撑"图表错位"论点：
     **在他们自己展示失效的 prompt 上量失效。** PDF 抽取有噪声，
     打印出来人工过目后才可用。

预注册预测（写在跑之前）：
  · PartiPrompts 整体 evf 右移于 LAION；Outdoor Scenes / World Knowledge
    高于 Artifacts / Food & Beverage（商品图式类别）；
  · showcase 组 evf 明显高；
  · **判死条件：连 Outdoor Scenes 都趴在 0 -> 该失效只活在手工构造的
    prompt 里，论文按再降一档的诚实口径写。**

    python scalediff_probe/fetch_parti.py
    python scalediff_probe/base_run.py --prompts $SD_OUT/parti.json --out $SD_OUT/parti_base
    python scalediff_probe/trigger_select.py --base $SD_OUT/parti_base
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

PARTI_URL = ("https://raw.githubusercontent.com/google-research/parti/"
             "main/PartiPrompts.tsv")

# 图注 prompt 的常见排版；抽出来必须人工过目，PDF 文本层噪声很大
PDF_PATTERNS = [
    r'[Pp]rompt[:\s]+["“]?([A-Z][^."”\n]{15,120})',
    r'["“]([A-Z][a-z][^"”\n]{20,120})["”]',
]
# 已核实的展示 prompt（来源标注到论文）；PDF 抽取只是给这张表补候选
SHOWCASE_VERIFIED = [
    ("A cute corgi on the lawn", "AccDiffusion v1/v2, Fig. qualitative"),
    ("A cute robot on the beach", "AccDiffusion v2, PDF 抽取已核"),
    ("Astronaut on mars during sunset", "AccDiffusion v2 / ScaleDiff run 脚本"),
    ("Summer landscape, vivid colors, a work of art, grotesque, mysterious",
     "AccDiffusion v2, PDF 抽取已核"),
]


def fetch_parti(out_dir, tune_frac=0.2, seed=0, limit=None):
    print(f"取 PartiPrompts: {PARTI_URL}")
    with urllib.request.urlopen(PARTI_URL, timeout=60) as r:
        raw = r.read().decode("utf-8")
    lines = raw.splitlines()
    head = lines[0].split("\t")
    print(f"  表头: {head}")
    ip = next(i for i, c in enumerate(head) if c.lower().startswith("prompt"))
    ic = next((i for i, c in enumerate(head) if c.lower().startswith("cat")), None)
    items = []
    for l in lines[1:]:
        f = l.split("\t")
        if len(f) <= ip or not f[ip].strip():
            continue
        items.append({"prompt": f[ip].strip(),
                      "category": f[ic].strip() if ic is not None and len(f) > ic else "?",
                      "source": "parti"})
    print(f"  共 {len(items)} 条")
    from collections import Counter
    cats = Counter(x["category"] for x in items)
    print("  类别分布: " + "  ".join(f"{k}={v}" for k, v in cats.most_common()))

    rng = random.Random(seed)
    rng.shuffle(items)
    if limit:
        items = items[:limit]
    k = int(len(items) * tune_frac)
    for i, it in enumerate(items):
        it["split"] = "tune" if i < k else "eval"
    p = Path(out_dir) / "parti.json"
    p.write_text(json.dumps({
        "source": "PartiPrompts (P2), google-research/parti",
        "n": len(items), "seed": seed,
        "note": "全量不按类别筛（筛也是挑）；类别只用于事后分组画像",
        "items": items}, ensure_ascii=False, indent=1))
    print(f"  写出 {p}（tune {k} / eval {len(items)-k}）")
    return items


def harvest_pdfs(repo_root):
    """从仓库根的论文 PDF 里抽图注 prompt 候选。只打印，不自动入库。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  (无 pypdf，跳过 PDF 收割)")
        return
    seen = set()
    for pdf in sorted(Path(repo_root).glob("*.pdf")):
        try:
            t = "\n".join((pg.extract_text() or "")
                          for pg in PdfReader(str(pdf)).pages)
        except Exception as e:
            print(f"  {pdf.name}: 读不了 ({type(e).__name__})")
            continue
        hits = []
        for pat in PDF_PATTERNS:
            hits += re.findall(pat, t)
        # 过滤明显的正文句子：太多逗号从句/包含引用记号的丢掉
        cand = []
        for h in hits:
            h = " ".join(h.split())
            if h in seen or len(h) > 130:
                continue
            if re.search(r"\[\d|et al|Fig\.|Table|Section", h):
                continue
            seen.add(h)
            cand.append(h)
        if cand:
            print(f"  {pdf.name}:")
            for c in cand[:8]:
                print(f"    ? {c}")
    print("  （'?' 开头的是**候选**，逐条人工核对图注后才加进 SHOWCASE_VERIFIED）")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root))
    ap.add_argument("--limit", type=int, default=None,
                    help="只取前 N 条（打乱后）。全量 ~1600 条基图约 3 小时")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-pdf", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    items = fetch_parti(a.out, seed=a.seed, limit=a.limit)

    sc = [{"prompt": p, "provenance": src, "split": "eval", "source": "showcase"}
          for p, src in SHOWCASE_VERIFIED]
    p = Path(a.out) / "showcase.json"
    p.write_text(json.dumps({
        "source": "figure prompts of the DemoFusion/AccDiffusion/ScaleDiff line",
        "n": len(sc),
        "note": "在他们自己展示失效的 prompt 上量失效；每条 provenance 到论文",
        "items": sc}, ensure_ascii=False, indent=1))
    print(f"\nshowcase 已核 {len(sc)} 条 -> {p}")

    if not a.skip_pdf:
        print("\nPDF 里的更多候选（人工核对后手动加进 SHOWCASE_VERIFIED）：")
        harvest_pdfs(Path(__file__).resolve().parent.parent)

    print(f"""
预注册预测（跑之前定死）：
  · Parti 整体 evf 右移于 LAION；Outdoor Scenes / World Knowledge 高于
    Artifacts / Food & Beverage；
  · showcase 组 evf 明显高；
  · **连 Outdoor Scenes 都趴 0 -> 失效只活在手工 prompt 里，按降档口径写。**

下一步：
  python scalediff_probe/base_run.py --prompts {a.out}/parti.json --out {a.out}/parti_base
  python scalediff_probe/base_run.py --prompts {a.out}/showcase.json --out {a.out}/showcase_base
  python scalediff_probe/trigger_select.py --base {a.out}/parti_base
  python scalediff_probe/trigger_select.py --base {a.out}/showcase_base""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
