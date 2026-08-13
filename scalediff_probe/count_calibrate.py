"""表 C：计数器对真值的检定 —— 把"看着不准"换成"量出多准"。

背景：Parti 触发集的接触表看着检测很差。那张表是按 evf>0.5 选的 ——
**构造上就是检测失败的尾巴集**（evf 高 = 框少框小），观感放大了问题。
但观感的反驳只能是数字：在**计数写在 prompt 里**的基准上，
数基图、对声明数，给出 MAE / 严格命中率。

数据就是已生成的 geneval_base（80 张，"a photo of four chairs"，card=4）
和 cococount_base（200 张，"A photo of three ties on the ground"，card=3）。
真值来自基准作者，不是我们标的。

两种 grounding 文本并列（这就是仪器的两个工作点）：
    caption   条件全句 —— delta 实际用的方式，检定"如实使用状态"
    subject   只喂主体名词（这两个基准有官方 subject 字段）——
              检定"上限状态"，也回答"框差是不是喂全句喂出来的"

预注册判读（写在跑之前）：
    caption 模式 MAE <= 1.0 且 |bias| < 0.5      -> 尺子在真值面前站得住；
    subject 模式明显好于 caption 模式            -> 框的观感差主要来自
        "全句 grounding 把场景词也框进来/挤掉主体"，选择器有明确升级路径
        （规整 prompt 套件上按主体词 ground）；
    两种模式都差（MAE > 2）                       -> 计数器不合格，
        表 B 必须换仪器（更强的开放词表检测器）再跑。

    python scalediff_probe/count_calibrate.py                    # 两个基准都跑
    python scalediff_probe/count_calibrate.py --which geneval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_audit import conditioned_text, load_tokenizer   # noqa: E402


def run_bench(name, base_dir, items_path, det, tok, min_score, limit=None):
    from PIL import Image
    base = Path(base_dir)
    if not (base / "manifest.jsonl").exists():
        print(f"  {name}: 没有 {base}/manifest.jsonl，跳过")
        return None
    meta = json.loads(Path(items_path).read_text())["items"]
    rows = [json.loads(l) for l in (base / "manifest.jsonl").open()]
    if limit:
        rows = rows[:limit]

    out_path = base / f"calib_ms{min_score:g}.jsonl"
    done = {}
    if out_path.exists():
        for l in out_path.open():
            x = json.loads(l)
            done[x["idx"]] = x
    todo = [r for r in rows if r["idx"] not in done]
    print(f"  {name}: {len(rows)} 张，已算 {len(done)}，待算 {len(todo)}")

    t0 = time.time()
    with out_path.open("a") as f:
        for n, r in enumerate(todo, 1):
            it = meta[r["idx"]]
            card, subj = it.get("card"), it.get("subject")
            if card is None or not subj:
                continue
            im = Image.open(base / r["file"]).convert("RGB")
            cap, _, _ = conditioned_text(tok, r["prompt"])
            n_cap = len(det.detect(im, cap, min_score=min_score)[0])
            n_sub = len(det.detect(im, subj, min_score=min_score)[0])
            rec = {"idx": r["idx"], "card": card, "subject": subj,
                   "n_caption": n_cap, "n_subject": n_sub,
                   "prompt": r["prompt"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[r["idx"]] = rec
            el = time.time() - t0
            print(f"\r    {n}/{len(todo)}  {el/60:.0f} 分钟  "
                  f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟", end="", flush=True)
    if todo:
        print()
    recs = [done[r["idx"]] for r in rows if r["idx"] in done]
    return recs


def report(name, recs):
    if not recs:
        return
    mean = lambda v: sum(v) / max(len(v), 1)
    print(f"\n  == {name}  n={len(recs)} ==")
    print(f"  {'grounding':<10}{'MAE':>7}{'bias':>7}{'严格命中':>9}{'±1 内':>8}")
    for key, label in (("n_caption", "caption"), ("n_subject", "subject")):
        err = [r[key] - r["card"] for r in recs]
        mae = mean([abs(e) for e in err])
        bias = mean(err)
        exact = mean([e == 0 for e in err])
        within = mean([abs(e) <= 1 for e in err])
        print(f"  {label:<10}{mae:>7.2f}{bias:>+7.2f}{exact:>9.1%}{within:>8.1%}")
    # 按声明数分层 —— size_sweep 教训：均值会藏住某一档的塌方
    print("  按声明数分层（subject 模式）：")
    from collections import defaultdict
    by = defaultdict(list)
    for r in recs:
        by[r["card"]].append(r["n_subject"] - r["card"])
    for c in sorted(by):
        v = by[c]
        print(f"    card={c:<3} n={len(v):<4} MAE={mean([abs(e) for e in v]):.2f}"
              f"  bias={mean(v):+.2f}")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="*", default=["geneval", "cococount"])
    ap.add_argument("--min-score", type=float, default=0.50,
                    help="delta 实际用 0.50；怀疑二次筛吃框时并列跑 0.30")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from count_objects import Detector
    det = Detector()
    tok = load_tokenizer()

    cfg = {"geneval": (root / "geneval_base", root / "count_geneval.json"),
           "cococount": (root / "cococount_base", root / "count_cococount.json")}
    all_recs = {}
    for name in a.which:
        b, i = cfg[name]
        recs = run_bench(name, b, i, det, tok, a.min_score, a.limit)
        if recs:
            all_recs[name] = recs
            report(name, recs)

    print("""
判读（预注册在 docstring）：
  caption 模式 MAE <= 1.0 且 |bias| < 0.5  -> 尺子在真值面前站得住（表 C 成立）；
  subject 明显好于 caption               -> 框的观感差主要是全句 grounding 造成，
                                            规整 prompt 上按主体词 ground 是升级路径；
  两种都差（MAE > 2）                     -> 计数器不合格，表 B 换仪器再跑。
注意：这里检定的是"基图上数得准不准"。delta 的抗噪还有配对相消那一层，
所以 delta 的实际误差 <= 这里的单图误差。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
