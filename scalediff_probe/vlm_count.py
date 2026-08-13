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


def do_delta(v, hi, base):
    hi, base = Path(hi), Path(base)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    subj_cache_p = hi / "vlm_subjects.json"
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
    print(f"\nn={len(recs)}  delta 均值 {mean([x['delta'] for x in recs]):+.2f}"
          f"   >=1 的 {sum(1 for x in recs if x['delta'] >= 1)}"
          f"   >=3 的 {sum(1 for x in recs if x['delta'] >= 3)}")
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
    a = ap.parse_args()

    v = VlmCounter()
    if a.probe:
        return 0 if do_probe(v) else 1
    if a.calibrate:
        do_calibrate(v, root, a.which, a.limit)
    if a.delta:
        do_delta(v, a.hi, a.base)
    if not (a.probe or a.calibrate or a.delta):
        print("选一个：--probe / --calibrate / --delta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
