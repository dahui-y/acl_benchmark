"""数 LAION 上的重复量 —— 消掉检测器尺度漂移之后的那个数。

    delta_content = count₁₀₂₄(把 4096 输出降采样回 1024) − count₁₀₂₄(基图)
    delta_raw     = count₄₀₉₆(4096 输出)                 − count₁₀₂₄(基图)

前者两边落在同一个检测尺度上，**漂移被完全消掉**，剩下的是"内容上真的
多了东西"；后者是朴素做法。**两者之差就是漂移的实测值**（`scale_check`
在诊断集上量到残余 2.00 个物体，这里在 LAION 上再量一次）。

重复出来的是**整个主体的副本**，降采样到 1024 照样看得见，
所以 delta_content **只会低估不会虚报** —— 这正是我们要的方向。

**一个免费的自检（不做的话整个测量的参照物就是悬空的）：**
`laion_hi_run` 的 `pipe()` 同时输出 1024²，它**应当与 `base_run.py` 出的
基图逐字节相同**（同 ckpt / 同 seed / 同 steps / 同 CFG / 同 negative，
ScaleDiff 第一级就是原版 SDXL）。不同就说明两边的"基图"不是一张图，
delta 的参照物错位，必须先对齐再谈结论。脚本先查这个，再算别的。

**判读（预注册，写在看结果之前）：**

  C 层 delta_content 明显 > A 层，且与 evf 正相关
      -> 律在 LAION 上成立，触发变量有效 -> 定 τ，走原路线
  各层 delta_content 都接近 0
      -> 重复在 4K LAION 上真的罕见（2% 得到独立确认）-> 换承载面或换分辨率
  各层都有明显 delta_content，但与 evf 无关
      -> **问题常见，只是 evf 不是对的预测因子** -> 保留问题，换预测因子。
         **这其实是好消息**：换尺子比换题目容易。
  D 层（盲区）delta_content 高
      -> 门会整批漏掉这些图。**方法硬伤**，必须先修检测覆盖。

**功效提醒**：n=30/层，delta 标准差若约 2，则只能检出 >=1.5 个物体的差异。
"各层都接近 0" 应读作"没检出 >=1.5 的效应"，**不是"效应为 0"**。

    python scalediff_probe/laion_delta.py
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import spearman                      # noqa: E402
from caption_audit import conditioned_text, load_tokenizer   # noqa: E402


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "laion_hi"))
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--min-score", type=float, default=0.50)
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    hi, base = Path(a.hi), Path(a.base)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        sys.exit(f"{hi}/manifest.jsonl 是空的")
    print(f"{len(rows)} 条  <-  {hi}")

    # ---- 自检：两条管线出的 1024² 是不是同一张图 ----
    same, diff, nocmp = 0, [], 0
    for r in rows:
        f1 = r["files"].get("1024") or r["files"].get(1024)
        b1 = base / f"{r['idx']:05d}.png"
        if not f1 or not b1.exists():
            nocmp += 1
            continue
        if md5(hi / f1) == md5(b1):
            same += 1
        else:
            diff.append(r["idx"])
    print(f"\n自检 基图一致性：逐字节相同 {same}   不同 {len(diff)}   无法比 {nocmp}")
    if diff:
        # **md5 不同 ≠ 内容不同。** base_diag 已定案（2026-08-12）：
        # plain 管线逐字节复现 base_run（max=0），tiling 开关无影响；
        # ScaleDiff 管线的 1024² 与之 max=13 / mean=0.057 —— 纯数值路径差，
        # 同一张图。所以这里补一手像素差，把两种情况当场分开。
        from PIL import Image
        import numpy as np
        worst = 0.0
        for i in diff[:5]:
            r0 = next(r for r in rows if r["idx"] == i)
            f1 = r0["files"].get("1024") or r0["files"].get(1024)
            a1 = np.asarray(Image.open(hi / f1).convert("RGB"), dtype=np.int16)
            a2 = np.asarray(Image.open(base / f"{i:05d}.png").convert("RGB"),
                            dtype=np.int16)
            if a1.shape == a2.shape:
                worst = max(worst, float(np.abs(a1 - a2).mean()))
        print(f"  抽 5 对算像素差：mean 最大 {worst:.3f}"
              + ("  -> **内容等同（数值路径差异），参照物没有错位**；"
                 "delta 用同跑基图，560 张基图与 evf 照常可用。"
                 if worst < 1.0 else
                 "  -> **内容可能真的不同，先跑 base_diag.py 查清再信下面的数。**"))

    out_path = hi / "delta.jsonl"
    done = {}
    if out_path.exists():
        for l in out_path.open():
            x = json.loads(l)
            done[x["idx"]] = x
        print(f"delta.jsonl 里已有 {len(done)} 条，跳过")

    todo = [r for r in rows if r["idx"] not in done]
    if todo:
        from PIL import Image
        from count_objects import Detector
        det = Detector(box_thr=a.box_thr)
        tok = load_tokenizer()
        import time
        t0 = time.time()
        with out_path.open("a") as f:
            for n, r in enumerate(todo, 1):
                text, _, _ = conditioned_text(tok, r["prompt"])
                fhi = r["files"].get(str(a.res)) or r["files"].get(a.res)
                if not fhi:
                    continue
                im_hi = Image.open(hi / fhi).convert("RGB")
                # **基图优先用本次自带的 1024²** —— 自检若不一致，
                # 至少保证 delta 的两边来自同一条管线。
                f1 = r["files"].get("1024") or r["files"].get(1024)
                p_base = (hi / f1) if f1 else (base / f"{r['idx']:05d}.png")
                im_b = Image.open(p_base).convert("RGB")

                b_base, _ = det.detect(im_b, text, min_score=a.min_score)
                b_raw, _ = det.detect(im_hi, text, min_score=a.min_score)
                im_dn = im_hi.resize(im_b.size, Image.LANCZOS)
                b_dn, _ = det.detect(im_dn, text, min_score=a.min_score)

                rec = {"idx": r["idx"], "stratum": r["stratum"],
                       "evf": r.get("evf"), "nbox_base": len(b_base),
                       "n_raw": len(b_raw), "n_dn": len(b_dn),
                       "delta_raw": len(b_raw) - len(b_base),
                       "delta_content": len(b_dn) - len(b_base),
                       "prompt": r["prompt"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[r["idx"]] = rec
                el = time.time() - t0
                print(f"\r{n}/{len(todo)}  {el/60:.0f} 分钟  "
                      f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟", end="", flush=True)
        print()

    recs = [done[r["idx"]] for r in rows if r["idx"] in done]
    mean = lambda v: sum(v) / max(len(v), 1)

    print(f"\n{'层':<10}{'n':>4}{'基图框':>8}{'delta_content':>15}"
          f"{'delta_raw':>11}{'漂移':>8}")
    print("-" * 58)
    order = ["A_evf0", "B_low", "C_high", "D_undef"]
    for s in order:
        g = [r for r in recs if r["stratum"] == s]
        if not g:
            continue
        dc, dr = mean([r["delta_content"] for r in g]), mean([r["delta_raw"] for r in g])
        print(f"{s:<10}{len(g):>4}{mean([r['nbox_base'] for r in g]):>8.1f}"
              f"{dc:>15.2f}{dr:>11.2f}{dr-dc:>8.2f}")

    A = [r["delta_content"] for r in recs if r["stratum"] == "A_evf0"]
    C = [r["delta_content"] for r in recs if r["stratum"] == "C_high"]
    D = [r["delta_content"] for r in recs if r["stratum"] == "D_undef"]
    if A and C:
        gap = mean(C) - mean(A)
        print(f"\nC − A = {gap:+.2f} 个物体   "
              f"（功效：n=30/层只检得出 >=1.5 的差异）")
    ok = [r for r in recs if r["stratum"] != "D_undef" and r.get("evf") is not None]
    if len(ok) >= 10:
        rho = spearman([r["evf"] for r in ok], [r["delta_content"] for r in ok])
        print(f"Spearman(evf, delta_content) = {rho:+.3f}   "
              f"（不含盲区层；律要求正相关）")

    print("\n判读（预注册）：")
    if A and C:
        gap = mean(C) - mean(A)
        strong = gap >= 1.5
        anyrep = mean([r["delta_content"] for r in recs]) >= 1.0
        if strong:
            print("  C 明显高于 A -> **律在 LAION 上成立，触发变量有效** -> 定 τ")
        elif anyrep:
            print("  **各层都有重复但与 evf 无关 -> 问题是真的，evf 不是对的"
                  "预测因子。** 换尺子比换题目容易，这是好消息：保留问题，"
                  "改预测因子。")
        else:
            print("  **各层都接近 0 -> 4K LAION 上重复确实罕见**（2% 得到独立"
                  "确认）。注意这只说明'没检出 >=1.5 的效应'。-> 换承载面或换分辨率。")
    if D and mean(D) >= 1.0:
        print(f"  **盲区层 delta_content = {mean(D):.2f} —— 门会整批漏掉这些图。**"
              "\n     这是方法硬伤（27% 的语料），必须先修检测覆盖，不能当注脚。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
