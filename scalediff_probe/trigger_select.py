"""在 LAION 基图上量空视野比例 —— 回答"触发区在真实语料里有多常见"，
并切出触发子群。

**这是整条链上第一个按"我们研究的问题"筛的步骤。** 在它之前的筛选
（caption 词数、短边 ≥ 299、真图存活、去重）全是"数据能不能用"，
一条都不涉及图的内容。

两件事必须分开，别混：

  · **标准表（FID/KID/IS/CLIP）跑在未经挑选的 eval-1000 上**，
    与 ScaleDiff §4.1 同协议。按"有没有重复"去挑会让那张表不再可比。
  · **重复指标跑在触发子群上**，分层报告。这一层不需要真图
    （Δdelta 只比 4096 与基图）。

所以基图里出现"不像我们问题"的图是**正常且必要的** —— 它们正是用来
证明门在非触发场景上关闭、逐字节不动、指标不退化的那批。

预注册的东西（**都在看 LAION 数据之前就定死了**）：

  τ = 0.75    **这个值已被证据推翻，见 §5.4，保留在这里只为对照。**
              它定在我们那 30 条诊断 prompt 上，而那批是为展示现象刻意
              构造的（"a lone hiker on a rocky ridge"）。τ=0.75 要求
              16 块里有 12 块空 —— 看图实测，LAION 上即便框完全准确，
              Jeep 约 0.62、木屋约 0.50、贴纸约 0.75，**几乎不可达**。
              律本身是连续的（重复 ≈ R²×(1−覆盖率)），τ 只是把连续量切成
              二值的**部署选择**，该在 tune 上按 Δdelta 的代价/收益定。
              所以这里四档全报，**哪一档是主判据等 tune 上的 Δdelta 出来再定**，
              eval 全程不回看。
  nbox >= 1   基图一个框都检不出时，空视野比例恒等于 1，那是**仪器失效
              不是内容属性**，预测量在这些行上没有定义（§5.5）。
  普遍性判据  >= 25% -> 触发区常见，总表值得跑；
              10–25% -> 分层表为主、总表小幅改善即可；
              < 10%  -> **老实承认这是小众但真实的失效模式**，
                        主张改成"触发区大幅改善、非触发区零代价"。

检测器为什么喂 caption 而不是类别表：GroundingDINO 是开放词表的，
LAION 的 caption 正是条件生成时用的那句话，**用它去 ground 就是在找
"prompt 要求的那些主体"** —— 而律说的正是这些主体被在空视野里重画。
换成 COCO 80 类会把"连衣裙""缝纫图样""教堂立面"这类主体整片漏掉，
于是它们全被 nbox=0 排除，触发子群被偷偷偏向 COCO 式内容。

    python scalediff_probe/trigger_select.py --split tune          # 先跑 tune
    python scalediff_probe/trigger_select.py --split eval --limit 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import empty_view_fraction        # noqa: E402
from caption_audit import conditioned_text, load_tokenizer   # noqa: E402

TAU = 0.75
TAUS = (0.50, 0.625, 0.75, 0.875)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--split", default=None, choices=["tune", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--R", type=int, default=4,
                    help="4 -> 4096²（16 块）。与放大倍数一致")
    ap.add_argument("--box-thr", type=float, default=0.30,
                    help="**门要召回，指标要精度 —— 两个工作点不同**（§5.5）。"
                         "基图漏检 -> 那块看着空 -> 空视野比例虚高 -> 假触发。"
                         "但阈值放低也会引入假框把块填满 -> 漏触发。两个方向"
                         "都存在，所以不拍脑袋：tune 上跑 0.20/0.30 两档，"
                         "**普遍性若对阈值不敏感就沿用 0.30（与计数指标同值），"
                         "敏感则如实记下来**。")
    ap.add_argument("--min-score", type=float, default=0.50,
                    help="**detect() 在 box_thr 之后还筛的第二道**。"
                         "trigger_debug 实测它把两张图整图一遍拿到的 3 个框"
                         "全部吃掉，evf 记成 1.00 再被 nbox>=1 当成'无定义'"
                         "排除掉 —— 门要召回，这道二次筛正好反着来。"
                         "默认沿用 0.50（诊断集当时的工作点），但必须并列报"
                         "0.30 那一档。")
    ap.add_argument("--tag", default=None, help="输出文件后缀，便于并列两档")
    a = ap.parse_args()

    base = Path(a.base)
    mpath = base / "manifest.jsonl"
    if not mpath.exists():
        sys.exit(f"没有 {mpath}，先跑 base_run.py")
    rows = [json.loads(l) for l in mpath.open()]
    if a.split:
        rows = [r for r in rows if r.get("split") == a.split]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        sys.exit("没有符合条件的基图")

    tag = a.tag or f"{a.split or 'all'}_thr{a.box_thr:g}_ms{a.min_score:g}"
    out = base / f"trigger_{tag}.jsonl"
    done = {}
    if out.exists():                      # 断点续跑
        for l in out.open():
            r = json.loads(l)
            done[r["idx"]] = r
        print(f"已有 {len(done)} 条，跳过")
    todo = [r for r in rows if r["idx"] not in done]
    print(f"{len(rows)} 条，待跑 {len(todo)} 条   R={a.R}"
          f"（{a.R*a.R} 块）  box_thr={a.box_thr}")

    if todo:
        from PIL import Image
        from count_objects import Detector
        det = Detector(box_thr=a.box_thr)
        # **ground 的文本必须恰好等于条件过图像的那段。** LAION 的 alt-text
        # 里混着图片 ID / 发布者 / 日期（"1 1 may. 2 0 1 6" 每字符一个 token），
        # 一条 60 词的 caption 也能撑到 95 token，超出 77 的部分 SDXL 根本没看见
        # —— 拿它去 ground 图像只会长出噪声框，而这个量对假阳性框最敏感（§5.4）。
        tok = load_tokenizer()
        t0 = time.time()
        with out.open("a") as f:
            for n, r in enumerate(todo, 1):
                im = Image.open(base / r["file"]).convert("RGB")
                text, _, _ = conditioned_text(tok, r["prompt"])
                b, _ = det.detect(im, text, min_score=a.min_score)
                evf = empty_view_fraction(b, im.width, im.height, a.R)
                rec = {"idx": r["idx"], "split": r.get("split"),
                       "nbox": len(b), "evf": evf, "prompt": r["prompt"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[r["idx"]] = rec
                el = time.time() - t0
                print(f"\r{n}/{len(todo)}  {el/60:.0f} 分钟已用  "
                      f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟", end="", flush=True)
        print()

    recs = [done[r["idx"]] for r in rows if r["idx"] in done]
    defined = [r for r in recs if r["nbox"] >= 1]
    n_undef = len(recs) - len(defined)
    print(f"\n共 {len(recs)} 条；**基图检不出框的 {n_undef} 条"
          f"（{n_undef/len(recs):.1%}）预测量无定义，按 §5.5 排除**")
    if n_undef / max(len(recs), 1) > 0.25:
        print("  ⚠️ 排除比例过高 —— 这不是内容属性而是检测器覆盖不足，"
              "会把触发子群系统性地偏向'检得到的那类内容'。**必须记进论文局限**。")

    if not defined:
        sys.exit("没有可用的行")

    ev = sorted(r["evf"] for r in defined)
    qs = [0, .1, .25, .5, .75, .9, 1.0]
    print("\n空视野比例分位：" + "  ".join(
        f"p{int(q*100)}={ev[min(int(q*len(ev)), len(ev)-1)]:.2f}" for q in qs))

    print("\n普遍性（分母 = 有定义的那些）：")
    for t in TAUS:
        k = sum(1 for r in defined if r["evf"] > t)
        star = "  <- 预注册主判据" if t == TAU else ""
        print(f"  τ={t:<6} 触发 {k:>5} / {len(defined)} = {k/len(defined):6.1%}{star}")

    k = sum(1 for r in defined if r["evf"] > TAU)
    frac = k / len(defined)
    print("\n判读（§1.2.1a 预注册的三档）：")
    if frac >= 0.25:
        print(f"  {frac:.1%} >= 25%  -> **触发区常见**，总表值得跑，"
              "整体也该看得出改善。")
    elif frac >= 0.10:
        print(f"  10% <= {frac:.1%} < 25%  -> **中间地带**：分层表为主，"
              "总表小幅改善即可。")
    else:
        print(f"  {frac:.1%} < 10%  -> **触发区罕见**。不粉饰：主张改成"
              "'触发区大幅改善、非触发区零代价'，\n"
              "     总表退为'不退化'的证据。这仍是一篇成立的论文，"
              "但故事必须照实讲。")

    sel = sorted(r["idx"] for r in defined if r["evf"] > TAU)
    p = base / f"trigger_set_{tag}.json"
    p.write_text(json.dumps(
        {"base": str(base), "split": a.split, "R": a.R, "tau": TAU,
         "box_thr": a.box_thr, "min_score": a.min_score, "n_scored": len(recs), "n_defined": len(defined),
         "n_trigger": len(sel), "prevalence": frac,
         "note": "τ 沿用 trigger.py 的预注册值，未在 LAION 上重新拟合",
         "alive_idx": sel}, ensure_ascii=False, indent=1))
    print(f"\n触发子群 {len(sel)} 条写入 {p}（键名沿用 alive_idx）")
    print("\n下一步：在 tune 上把 box_thr 0.20 那一档也跑一遍，"
          "看普遍性对阈值敏不敏感：\n"
          f"    python {sys.argv[0]} --split tune --box-thr 0.20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
