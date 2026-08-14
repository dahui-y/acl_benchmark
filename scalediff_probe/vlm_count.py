"""VLM 计数器 —— 把 Δdelta 的刀从检测器换成 Qwen2.5-VL。

为什么换（2026-08-13，用户否决检测器方案，否决是对的）：
    触发集接触表上 GroundingDINO 肉眼可见地失败 —— 死在剪影、水墨、
    插画、商品图上，全是风格化内容的分布外失效。headline metric 底下
    压着一个一眼不准的仪器，校准数字救不回第一印象。
    VLM 数风格化图像是强项，7B 单卡 4090 装得下，"VLM 当裁判"是
    2026 年评测的通行做法。

**Δdelta 的设计一个字不改**：配对（同基图两臂）、同尺度比较（4096
降采样回 1024 再数，与基图同分辨率）、基图为参照、真值检定后才上岗。
换的只是"数数"这一步。GroundingDINO 降级为附录里的交叉验证副尺。

预注册（写在跑之前）：
    检定门槛   --calibrate 在 GenEval/CoCoCount 基图上 MAE <= 1.0 且
               |bias| < 0.5，VLM 才准上岗成为主尺；
    判死       VLM 也检定不过 -> 弃计数，改成对偏好判决
               （VLM 只答"哪张图有主体被重复"，不数数）；
    一致性     同图同问必须同答（temperature=0, do_sample=False），
               --probe 会重复问 3 次验证。

主体词从哪来：GenEval/CoCoCount 有官方 subject 字段；Parti/showcase 的
prompt 是规整英文，用 VLM 自己抽一次主体名词（每条 prompt 只抽一次，
落盘复用，temperature=0 可复现）。LAION alt-text 太脏，那批不走 VLM 计数。

    python scalediff_probe/vlm_count.py --probe                  # 模型能不能跑
    python scalediff_probe/vlm_count.py --calibrate              # 表 C 检定
    python scalediff_probe/vlm_count.py --delta --hi $SD_OUT/parti_hi \\
        --base $SD_OUT/parti_base                                # 34 条触发集

**2026-08-13 补：原 --calibrate 那张考卷作废（§8.9b）。**
`card` 是 prompt 声明的数，不是图里实际有的数；GenEval 这个基准存在的
理由恰恰是 T2I 数不对（SDXL counting 准确率 ~0.4），所以量到的误差是
"计数器误差 + 生成器没听懂数量"的和，不可分。**VLM 的计数本身照常有效
且已落盘**，作废的只是拿 `card` 当真值这件事。三个补丁模式：

    --null      空对照：基图 vs 它的重采样往返版（1024->4096->1024），
                内容同源、物体数必然相同，**Δ 的真值恒为 0**。
                零标注、零人眼，直接量 delta 的噪声地板 —— 这比单图
                绝对精度更贴近 delta 的实际用法。
                预注册：mean|Δ| <= 0.3 且 95% 分位 <= 1 -> 通过。
    --gt-sheet  渲染分层接触表，供**作者逐图核对实际物体数**（不是标注员，
                论文里写明"由作者核对"），产出 gt_template.json。
    --regt      拿核对好的真值重算 MAE/bias，**复用已落盘的 VLM 计数，
                不再动 GPU**；同时给出 SDXL 的数量服从率
                （= 实际数 == 声明数 的比例），它本身就是考卷作废的证据。
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MODELS = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]


class VlmCounter:
    def __init__(self):
        import torch
        self.torch = torch
        last = None
        for mid in MODELS:
            try:
                from transformers import AutoProcessor
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as M
                    if "2.5" not in mid:
                        raise ImportError
                except ImportError:
                    from transformers import Qwen2VLForConditionalGeneration as M
                    if "2.5" in mid:
                        # transformers 版本不带 2.5 类时退到 2-VL 模型
                        continue
                self.proc = AutoProcessor.from_pretrained(mid)
                self.model = M.from_pretrained(
                    mid, torch_dtype=torch.bfloat16, device_map="cuda")
                self.model.eval()
                self.mid = mid
                print(f"载入 {mid}")
                return
            except Exception as e:
                last = e
                print(f"  {mid}: {type(e).__name__}: {str(e)[:120]}")
        raise RuntimeError(f"两个模型都载入失败，最后错误：{last}")

    def ask(self, image, question, max_new=16):
        """单轮问答，贪心解码（可复现）。"""
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image],
                           return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False)
        ans = self.proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0]
        return ans.strip()

    def count(self, image, subject):
        q = (f"How many {subject}(s) are visible in this image? "
             f"Count every distinct instance, including partial or "
             f"background ones. Answer with a single integer only.")
        ans = self.ask(image, q)
        m = re.search(r"\d+", ans)
        return (int(m.group()) if m else None), ans

    def subject_of(self, prompt):
        q = (f'For the image-generation prompt "{prompt}", what is the main '
             f"countable subject? Answer with one or two words (a noun), "
             f"singular form, no articles.")
        # 抽主体不需要图；给一张 1x1 白图占位
        from PIL import Image
        blank = Image.new("RGB", (28, 28), "white")
        ans = self.ask(blank, q, max_new=8)
        return re.sub(r"[^a-zA-Z ]", "", ans).strip().lower() or None


def load_img(p, size=1024):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if max(im.size) > size:
        im = im.resize((size, size), Image.LANCZOS)
    return im


def do_probe(v):
    from PIL import Image
    import numpy as np
    # 合成一张 3 圆图，真值已知，重复问 3 次验证一致性
    a = np.full((512, 512, 3), 240, dtype=np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    for cx, cy in ((100, 120), (300, 250), (420, 400)):
        a[(xx - cx) ** 2 + (yy - cy) ** 2 < 45 ** 2] = (200, 30, 30)
    im = Image.fromarray(a)
    answers = [v.count(im, "red circle")[0] for _ in range(3)]
    print(f"3 个红圆，三次回答：{answers}")
    ok = all(x == 3 for x in answers)
    print("**probe 过：数对且三次一致**" if ok else
          "**probe 没过** —— " + ("答案不稳定（贪心解码下不该发生，查版本）"
                                  if len(set(answers)) > 1 else "数错了"))
    subj = v.subject_of("a small house on a mountain top")
    print(f'主体抽取自检："a small house on a mountain top" -> "{subj}"'
          + ("  ✓" if subj and "house" in subj else "  <- 预期含 house，检查"))
    return ok


def do_calibrate(v, root, which, limit):
    cfg = {"geneval": (root / "geneval_base", root / "count_geneval.json"),
           "cococount": (root / "cococount_base", root / "count_cococount.json")}
    mean = lambda x: sum(x) / max(len(x), 1)
    for name in which:
        bdir, ipath = cfg[name]
        if not (bdir / "manifest.jsonl").exists():
            print(f"{name}: 没有基图，跳过")
            continue
        meta = json.loads(ipath.read_text())["items"]
        rows = [json.loads(l) for l in (bdir / "manifest.jsonl").open()]
        if limit:
            rows = rows[:limit]
        outp = bdir / "vlm_calib.jsonl"
        done = {json.loads(l)["idx"]: json.loads(l)
                for l in outp.open()} if outp.exists() else {}
        todo = [r for r in rows if r["idx"] not in done]
        print(f"\n{name}: {len(rows)} 张，待算 {len(todo)}")
        t0 = time.time()
        with outp.open("a") as f:
            for n, r in enumerate(todo, 1):
                it = meta[r["idx"]]
                cnt, raw = v.count(load_img(bdir / r["file"]), it["subject"])
                rec = {"idx": r["idx"], "card": it["card"],
                       "subject": it["subject"], "n_vlm": cnt, "raw": raw}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[r["idx"]] = rec
                el = time.time() - t0
                print(f"\r  {n}/{len(todo)}  {el/60:.1f} 分钟  "
                      f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟", end="", flush=True)
        print()
        recs = [x for x in done.values() if x["n_vlm"] is not None]
        bad = len(done) - len(recs)
        err = [x["n_vlm"] - x["card"] for x in recs]
        print(f"  n={len(recs)}（解析失败 {bad}）  "
              f"MAE={mean([abs(e) for e in err]):.2f}  "
              f"bias={mean(err):+.2f}  "
              f"严格命中={mean([e == 0 for e in err]):.1%}  "
              f"±1 内={mean([abs(e) <= 1 for e in err]):.1%}")
        from collections import defaultdict
        by = defaultdict(list)
        for x in recs:
            by[x["card"]].append(x["n_vlm"] - x["card"])
        for c in sorted(by):
            vv = by[c]
            print(f"    card={c:<3} n={len(vv):<4} "
                  f"MAE={mean([abs(e) for e in vv]):.2f} bias={mean(vv):+.2f}")
    print("\n检定门槛（预注册）：MAE <= 1.0 且 |bias| < 0.5 -> VLM 上岗主尺；"
          "\n不过 -> 判死：弃计数，改成对偏好判决。")


def roundtrip(im, up=4096, back=1024):
    """基图 -> 双三次放大 -> 走与 delta 相同的降采样路径回 1024。

    内容逐像素同源，**物体数必然不变**，所以这一对的 Δ 真值恒为 0，
    不需要任何标注。量到的非零就是仪器在 delta 实际配置下的噪声。
    注意披露：这一关只含"重采样"这层扰动，不含"重新生成"那层，
    所以它是噪声地板的**下界**。含生成扰动的空对照来自 --delta 里
    那 25 对眼睛判为无重复的样本（见 §5.49e），两者并列报。
    """
    from PIL import Image
    return im.resize((up, up), Image.BICUBIC).resize((back, back), Image.LANCZOS)


def do_null(v, hi, limit):
    hi = Path(hi)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    if limit:
        rows = rows[:limit]
    subj_p = hi / "vlm_subjects.json"
    subj = json.loads(subj_p.read_text()) if subj_p.exists() else {}
    outp = hi / "vlm_null.jsonl"
    done = {json.loads(l)["idx"]: json.loads(l)
            for l in outp.open()} if outp.exists() else {}
    todo = [r for r in rows if r["idx"] not in done]
    print(f"空对照：{len(rows)} 条，待算 {len(todo)}")
    t0 = time.time()
    with outp.open("a") as f:
        for n, r in enumerate(todo, 1):
            key = str(r["idx"])
            if key not in subj:
                subj[key] = v.subject_of(r["prompt"])
                subj_p.write_text(json.dumps(subj, ensure_ascii=False))
            f1 = r["files"].get("1024") or r["files"].get(1024)
            im = load_img(hi / f1)
            n_a, _ = v.count(im, subj[key])
            n_b, _ = v.count(roundtrip(im), subj[key])
            rec = {"idx": r["idx"], "subject": subj[key], "n_base": n_a,
                   "n_roundtrip": n_b,
                   "d": (n_b - n_a) if None not in (n_a, n_b) else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[r["idx"]] = rec
            print(f"\r  {n}/{len(todo)}  {(time.time()-t0)/60:.1f} 分钟",
                  end="", flush=True)
    print()
    ds = [x["d"] for x in done.values() if x.get("d") is not None]
    if not ds:
        print("没有有效样本")
        return
    mean = lambda x: sum(x) / max(len(x), 1)
    a = sorted(abs(d) for d in ds)
    p95 = a[min(len(a) - 1, int(0.95 * len(a)))]
    m = mean(a)
    print(f"\nn={len(ds)}  mean|Δ|={m:.2f}  95% 分位={p95}  "
          f"Δ=0 的占 {mean([d == 0 for d in ds]):.1%}  "
          f"bias={mean(ds):+.2f}")
    ok = m <= 0.3 and p95 <= 1
    print("**空对照过（预注册 mean|Δ|<=0.3 且 95%<=1）：噪声地板够低，"
          "delta 读到的 +1 是内容，不是仪器**" if ok else
          "**空对照没过** —— 仪器在同源内容上就会自己抖出 Δ，"
          "delta 的 +1 无法与噪声区分。走判死支：弃计数，改成对偏好判决。")


def do_gt_sheet(root, cards, per, cell, cols):
    """渲染分层接触表，供作者逐图核对**实际**物体数。"""
    from PIL import Image, ImageDraw
    base = root / "cococount_base"
    meta = json.loads((root / "count_cococount.json").read_text())["items"]
    rows = [json.loads(l) for l in (base / "manifest.jsonl").open()]
    by = {}
    for r in rows:
        c = meta[r["idx"]].get("card")
        if c in cards:
            by.setdefault(c, []).append(r)
    pick = []
    for c in cards:
        pick += by.get(c, [])[:per]
    print(f"取 {len(pick)} 张（每档 {per}，档 {cards}）")
    tmpl, nsheet = {}, 0
    for s in range(0, len(pick), cols * 2):
        chunk = pick[s:s + cols * 2]
        nrow = (len(chunk) + cols - 1) // cols
        pad, cap = 8, 22
        sheet = Image.new("RGB", (cols * (cell + pad) + pad,
                                  nrow * (cell + cap + pad) + pad), "white")
        dr = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            im = Image.open(base / r["file"]).convert("RGB").resize(
                (cell, cell), Image.LANCZOS)
            x = pad + (k % cols) * (cell + pad)
            y = pad + (k // cols) * (cell + cap + pad)
            sheet.paste(im, (x, y))
            it = meta[r["idx"]]
            dr.text((x + 2, y + cell + 4),
                    f"[{r['idx']}] subject={it['subject']}  card={it['card']}",
                    fill="black")
            tmpl[str(r["idx"])] = {"subject": it["subject"],
                                   "card": it["card"], "n_actual": None}
        p = base / f"gt_sheet_{nsheet:02d}.jpg"
        sheet.save(p, "JPEG", quality=94)
        print(f"  写出 {p}")
        nsheet += 1
    tp = base / "gt_template.json"
    tp.write_text(json.dumps(tmpl, ensure_ascii=False, indent=1))
    print(f"\n模板 {tp} —— 逐图填 n_actual（图里**实际**数得出几个 subject），"
          f"填完跑 --regt")


def do_regt(root, which, gt_path):
    """用核对好的真值重算，**复用已落盘的 VLM 计数，不动 GPU**。"""
    mean = lambda x: sum(x) / max(len(x), 1)
    gp = Path(gt_path)
    if not gp.exists():
        print(f"没有真值文件 {gp} —— 先跑 --gt-sheet，核对后填 n_actual")
        return
    raw = json.loads(gp.read_text())
    # 真值文件属于哪个基准就只用于哪个基准 —— 两个基准的 idx 都是小整数，
    # 混用会索引撞车（GenEval 的第 7 张被拿去对 CoCoCount 第 7 张的真值，
    # 2026-08-14 实跑出过一块 MAE=2.02 的假结果）
    gt_bench = raw.get("_bench")
    # 跳过 _note 之类的说明字段（值是字符串不是 dict）
    gt = {k: v for k, v in raw.items()
          if isinstance(v, dict) and v.get("n_actual") is not None}
    if not gt:
        print(f"{gt_path} 里没有填好的 n_actual")
        return
    cfg = {"geneval": root / "geneval_base", "cococount": root / "cococount_base"}
    for name in which:
        if gt_bench and name != gt_bench:
            print(f"\n== {name}: 真值文件属于 {gt_bench}，跳过 ==")
            continue
        p = cfg[name] / "vlm_calib.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.open()]
        recs = [r for r in recs
                if str(r["idx"]) in gt and r.get("n_vlm") is not None]
        if not recs:
            continue
        act = lambda r: gt[str(r["idx"])]["n_actual"]
        err = [r["n_vlm"] - act(r) for r in recs]
        derr = [r["n_vlm"] - r["card"] for r in recs]
        comply = mean([act(r) == r["card"] for r in recs])
        print(f"\n== {name}  n={len(recs)} ==")
        print(f"  对**实际数**：MAE={mean([abs(e) for e in err]):.2f}  "
              f"bias={mean(err):+.2f}  严格命中={mean([e == 0 for e in err]):.1%}"
              f"  ±1 内={mean([abs(e) <= 1 for e in err]):.1%}")
        print(f"  对**声明数**（作废的旧算法，对照用）："
              f"MAE={mean([abs(e) for e in derr]):.2f}  bias={mean(derr):+.2f}")
        print(f"  **SDXL 数量服从率={comply:.1%}** "
              f"（实际数 == 声明数的比例 —— 这就是旧考卷作废的直接证据）")
        from collections import defaultdict
        byc = defaultdict(list)
        for r in recs:
            byc[r["card"]].append(r["n_vlm"] - act(r))
        for c in sorted(byc):
            vv = byc[c]
            print(f"    card={c:<3} n={len(vv):<4} "
                  f"MAE={mean([abs(e) for e in vv]):.2f} bias={mean(vv):+.2f}")
    print("\n门槛不变（预注册）：对**实际数** MAE <= 1.0 且 |bias| < 0.5 "
          "-> VLM 上岗主尺；不过 -> 弃计数，改成对偏好判决。"
          "\n运行域限定：结论只在 <= 6 个实例的场景上声明。")


def do_armdelta(v, d):
    """method_v2 的四臂输出：按 (idx, arm) 数 delta，判 P2a/P2b。

    为什么单独一趟：SDXL 管线还占着显存时装不下 7B VLM，所以生成与
    权威计数必须分开跑。为什么不用 method_v2 里那一列：那是检测器，
    §8.9a 已判死（饱和在 2–3 个），而 P2a/P2b 恰恰是纯计数判据。

    **同一 idx 的四个臂必须用同一个主体词**，否则臂间不可比 —— 所以
    主体按 idx 缓存，不按 (idx, arm)。
    """
    d = Path(d)
    rows = [json.loads(l) for l in (d / "manifest.jsonl").open()]
    rows = [r for r in rows if "arm" in r and r.get("files")]
    if not rows:
        print(f"{d}/manifest.jsonl 里没有带 arm 的行")
        return
    subj_p = d / "vlm_subjects.json"
    subj = json.loads(subj_p.read_text()) if subj_p.exists() else {}
    outp = d / "vlm_armdelta.jsonl"
    key = lambda r: f"{r['idx']}_{r['arm']}"
    done = {json.loads(l)["key"]: json.loads(l)
            for l in outp.open()} if outp.exists() else {}
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(rows)} 个臂样本，待算 {len(todo)}")
    t0 = time.time()
    with outp.open("a") as f:
        for n, r in enumerate(todo, 1):
            ik = str(r["idx"])
            if ik not in subj:
                subj[ik] = v.subject_of(r["prompt"])
                subj_p.write_text(json.dumps(subj, ensure_ascii=False))
            fl = r["files"]
            f_lo = fl.get("1024") or fl.get(1024)
            f_hi = max((k for k in fl if str(k).isdigit()), key=lambda k: int(k))
            n_b, _ = v.count(load_img(d / f_lo), subj[ik])
            n_h, _ = v.count(load_img(d / fl[f_hi]), subj[ik])
            rec = {"key": key(r), "idx": r["idx"], "arm": r["arm"],
                   "subject": subj[ik], "n_base": n_b, "n_hi_dn": n_h,
                   "delta": (n_h - n_b) if None not in (n_b, n_h) else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[key(r)] = rec
            print(f"\r  {n}/{len(todo)}  {(time.time()-t0)/60:.1f} 分钟",
                  end="", flush=True)
    print()
    by = {}
    for x in done.values():
        by.setdefault(x["idx"], {})[x["arm"]] = x["delta"]
    arms = sorted({x["arm"] for x in done.values()})
    print(f"\n{'idx':<5}" + "".join(f"{a:>16}" for a in arms))
    for i in sorted(by):
        print(f"{i:<5}" + "".join(
            f"{by[i].get(a, '-'):>16}" for a in arms))
    gs = sorted({a.split("_g")[1] for a in arms if "_g" in a})
    for g in gs:
        p2a = p2b = n = 0
        for i, ad in by.items():
            d1, db = ad.get("v1"), ad.get("base")
            d2, dvo = ad.get(f"v2_g{g}"), ad.get(f"v2only_g{g}")
            if None in (d1, d2):
                continue
            n += 1
            if None not in (dvo, db):
                p2a += (dvo >= db + 1)
            p2b += (d2 <= d1 + 0.5)
        print(f"\ng_bg={g}   P2a 互锁(v2only 回潮) {p2a}/{n}"
              f"   P2b 门压得住 {p2b}/{n}   （VLM 计数，权威读数）")


def do_delta(v, hi, base):
    hi, base = Path(hi), Path(base)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    # 主体缓存必须**全局共用**。按目录各存一份会漂：实测 157 在 hi/v1
    # 被解析成 'person'（6 个人），在 v12/acc 被解析成 'team'（2 支队）——
    # 同一张逐字节相同的基图数出 6 和 2，两个答案都对，只是问题不同。
    # Δdelta 在臂内仍自洽（base 与 hi 用同一个词），但**跨臂比较就废了**。
    # 优先读 SD_OUT 根目录的全局缓存；没有才回落到本目录（旧行为）。
    _g = Path(os.environ.get("SD_OUT", ".")) / "vlm_subjects_global.json"
    subj_cache_p = _g if (_g.exists() or os.environ.get("VLM_GLOBAL_SUBJ")) \
        else hi / "vlm_subjects.json"
    if subj_cache_p == _g:
        print(f"主体缓存：全局 {_g}")
    subj_cache = (json.loads(subj_cache_p.read_text())
                  if subj_cache_p.exists() else {})
    outp = hi / "vlm_delta.jsonl"
    done = {json.loads(l)["idx"]: json.loads(l)
            for l in outp.open()} if outp.exists() else {}
    todo = [r for r in rows if r["idx"] not in done]
    print(f"{len(rows)} 条，待算 {len(todo)}")
    t0 = time.time()
    with outp.open("a") as f:
        for n, r in enumerate(todo, 1):
            key = str(r["idx"])
            if key not in subj_cache:
                subj_cache[key] = v.subject_of(r["prompt"])
                subj_cache_p.write_text(json.dumps(subj_cache, ensure_ascii=False))
            subj = subj_cache[key]
            f1 = r["files"].get("1024") or r["files"].get(1024)
            fh = max((k for k in r["files"] if str(k).isdigit()),
                     key=lambda k: int(k))
            n_b, raw_b = v.count(load_img(hi / f1), subj)
            n_h, raw_h = v.count(load_img(hi / r["files"][fh]), subj)
            rec = {"idx": r["idx"], "subject": subj, "n_base": n_b,
                   "n_hi_dn": n_h,
                   "delta": (n_h - n_b) if None not in (n_b, n_h) else None,
                   "prompt": r["prompt"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[r["idx"]] = rec
            el = time.time() - t0
            print(f"\r{n}/{len(todo)}  {el/60:.1f} 分钟", end="", flush=True)
    print()
    recs = [x for x in done.values() if x.get("delta") is not None]
    mean = lambda x: sum(x) / max(len(x), 1)
    # 签名均值单独报是**不安全**的：负 delta（主体被毁）会冲抵正 delta
    # （重复）。实测 v1 与 v1.2 的 delta 总和都是 +3，但 v1 是 6 个 +1
    # 减一个 -3、v1.2 是 5 个 +1 减两个 -1 —— 行为完全不同、均值相同。
    # 一个"把主体全删光"的方法能靠这个刷分。故 Rep+ / Dmg- 必须同时报，
    # **两列都变好才算赢**。（2026-08-14 锁定，在闸门实验之前。）
    pos = sum(max(x['delta'], 0) for x in recs)
    neg = sum(max(-x['delta'], 0) for x in recs)
    n = max(len(recs), 1)
    print(f"\nn={len(recs)}  delta 均值 {mean([x['delta'] for x in recs]):+.2f}"
          f"   >=1 的 {sum(1 for x in recs if x['delta'] >= 1)}"
          f"   >=3 的 {sum(1 for x in recs if x['delta'] >= 3)}")
    print(f"  Rep+ (重复) {pos/n:.3f}  [{pos} 个实例]     "
          f"Dmg- (主体误伤) {neg/n:.3f}  [{neg} 个实例]")
    print(f"  判读：两列都要变好才算赢；只看签名均值会把'把主体删光'"
          f"当成进步。")
    print("分组判读（scenic/flat）用 parti_trigger_predictions.json 对账 ——"
          "见 vlm_delta.jsonl 逐条。")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--which", nargs="*", default=["geneval", "cococount"])
    ap.add_argument("--delta", action="store_true")
    ap.add_argument("--hi", default=str(root / "parti_hi"))
    ap.add_argument("--base", default=str(root / "parti_base"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--null", action="store_true",
                    help="空对照：基图 vs 重采样往返版，Δ 真值恒为 0")
    ap.add_argument("--gt-sheet", action="store_true",
                    help="渲染分层接触表供作者核对实际物体数（不用 GPU）")
    ap.add_argument("--regt", action="store_true",
                    help="用核对好的真值重算，复用已落盘计数（不用 GPU）")
    ap.add_argument("--gt", default=None, help="--regt 读的真值文件")
    ap.add_argument("--armdelta", action="store_true",
                    help="method_v2 四臂输出的权威计数（P2a/P2b 以它定案）")
    ap.add_argument("--dir", default=str(root / "method_v2"),
                    help="--armdelta 的目录")
    ap.add_argument("--resubject", default=None,
                    help="JSON {\"157\": \"person\"}：主体词规则修订"
                         "（具体可数名词；集体名词映射到成员）。更新主体缓存、"
                         "删掉受影响的 delta 行；配合 --delta 同跑即只重数这几条")
    ap.add_argument("--cards", type=int, nargs="*", default=[2, 3, 4, 5],
                    help="--gt-sheet 的分层档；默认只取运行域内（<=6 实例）")
    ap.add_argument("--per", type=int, default=15, help="--gt-sheet 每档张数")
    a = ap.parse_args()

    # 这两个模式不碰模型，先分流，免得白等 40 秒载入
    if a.gt_sheet:
        do_gt_sheet(root, a.cards, a.per, 512, 3)
        return 0
    if a.regt:
        do_regt(root, a.which,
                a.gt or (root / "cococount_base" / "gt_template.json"))
        return 0

    if a.resubject:
        # 主体词修订不需要模型：改缓存、删行；重数交给随后的 --delta
        hi = Path(a.hi)
        over = json.loads(Path(a.resubject).read_text())
        sp = hi / "vlm_subjects.json"
        subj = json.loads(sp.read_text()) if sp.exists() else {}
        for k, s_new in over.items():
            print(f"  [{k}] 主体 {subj.get(k)!r} -> {s_new!r}")
            subj[k] = s_new
        sp.write_text(json.dumps(subj, ensure_ascii=False))
        dp = hi / "vlm_delta.jsonl"
        if dp.exists():
            keep = [l for l in dp.open()
                    if str(json.loads(l)["idx"]) not in over]
            dropped = sum(1 for _ in dp.open()) - len(keep)
            dp.write_text("".join(keep))
            print(f"删除 {dropped} 条旧 delta 行；接下来 --delta 只会重数这几条")
        if not (a.probe or a.calibrate or a.delta or a.null or a.armdelta):
            print("没有带 --delta，只改了缓存和删行，未加载模型。")
            return 0

    v = VlmCounter()
    if a.probe:
        return 0 if do_probe(v) else 1
    if a.calibrate:
        do_calibrate(v, root, a.which, a.limit)
    if a.null:
        do_null(v, a.hi, a.limit)
    if a.delta:
        do_delta(v, a.hi, a.base)
    if a.armdelta:
        do_armdelta(v, a.dir)
    if not (a.probe or a.calibrate or a.delta or a.null or a.armdelta):
        print("选一个：--probe / --calibrate / --null / --delta "
              "/ --armdelta / --gt-sheet / --regt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
