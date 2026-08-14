"""v1 的真考：34 条 Parti 触发集上跑 v1 臂（门开，s=1）。

这是 v1 第一次离开诊断集（那 30 条是设计脚手架，§6.9）在第三方语料上
接受检验。看三件事：
    263  第二尊自由女神消不消失（实锤展品，论文方法部分第一图的候选）
    583  网眼纹理还克隆不克隆进空挡风布
    其余 25 条眼睛判干净的样本 —— 构图与质量是否保持

主体词从哪来：Parti prompt 没有手写 HEADS 表。中心词从 VLM 主体缓存
（vlm_subjects.json，--null / --delta 跑的时候落盘）解析出**表面形**：
VLM 给单数（"helicopter"），prompt 里可能是复数（"helicopters"），
按 简单复数规则 对齐到 prompt 里实际出现的那个词 —— strip_subject 和
token 匹配都要表面形。两个保险：
    --plan        不加载模型，打印 34 条的 head/摘除结果/token 命中，
                  人工过目后再烧 GPU；解析失败的条目列出原因
    --heads-file  JSON {"263": "statue", ...}，手工覆盖解析坏的条目

自带一致性检查：v1 在基础阶段只录图不混合（BlendCrossAttn 输出始终走
SDPA），所以 v1 自己的 1024 基图应与 parti_hi 的 1024 **逐像素相同**
（同管线同 seed）。跑完逐条比对并报告 —— 这就是"门开也不碰基础阶段"
的直接证据，白捡的。

    python scalediff_probe/v1_trigger_run.py --plan
    python scalediff_probe/v1_trigger_run.py --idx 263 583 146 157 553 994
    nohup python scalediff_probe/v1_trigger_run.py > v1run.log 2>&1 &

之后计数（v1 臂自己的 delta，与 parti_hi 基线臂逐条对照）：
    python scalediff_probe/vlm_count.py --delta --hi $SD_OUT/parti_v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                                  # noqa: E402
from subject_phrases import strip_subject, _base              # noqa: E402
from method_v1 import BlendGate, BlendCrossAttn               # noqa: E402
from gate_refresh import RefreshGate                          # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
STOP = {"of", "a", "an", "the"}


def surface_head(prompt, vlm_subject):
    """把 VLM 给的（可能是单数/两个词的）主体名词对齐到 prompt 里
    实际出现的那个词。返回 (表面词, 失败原因)。

    两词主体的取词顺序：含 "of" 取第一个名词（"statue of liberty" ->
    statue，中心词在前）；否则取最后一个（"toy car" -> car，中心词在后）。
    """
    if not vlm_subject:
        return None, "VLM 没给主体"
    words = [w for w in vlm_subject.lower().split() if w not in STOP]
    if not words:
        return None, f"主体全是虚词: {vlm_subject!r}"
    cands = [words[0]] + words[1:] if "of" in vlm_subject.lower() \
        else [words[-1]] + words[:-1]
    pwords = [_base(w) for w in prompt.split()]
    for c in cands:
        # 表面形对齐：原形，或 简单复数（cats / boxes / puppies）
        forms = {c, c + "s", c + "es"}
        if c.endswith("y"):
            forms.add(c[:-1] + "ies")
        if c.endswith("s"):
            forms.add(c[:-1])          # VLM 偶尔给了复数、prompt 是单数
        for f in forms:
            if f in pwords:
                return prompt.split()[pwords.index(f)].strip(",.;:"), None
    return None, f"主体 {vlm_subject!r} 的任何形态都不在 prompt 里"


def token_ids_of(tok, prompt, subject):
    """subject 在 77 长度截断序列里的位置（与 method_v0 同逻辑，
    只依赖 tokenizer，--plan 不用加载整个管线）。"""
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids[0]
    sub = tok(subject, add_special_tokens=False).input_ids
    hit = []
    for i in range(len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)].tolist() == sub:
            hit += list(range(i, i + len(sub)))
    return sorted(set(hit))


def resolve(rows, subjects, overrides, tok):
    """逐条解析，返回 (可跑列表, 跳过列表)。"""
    ok, skip = [], []
    for r in rows:
        key = str(r["idx"])
        if key in overrides:
            head, why = overrides[key], None
        else:
            head, why = surface_head(r["prompt"], subjects.get(key))
        if head is None:
            skip.append((r, why))
            continue
        nosubj, removed = strip_subject(r["prompt"], head)
        if not removed:
            skip.append((r, f"head={head!r} 摘不出短语"))
            continue
        tids = token_ids_of(tok, r["prompt"], head)
        if not tids:
            skip.append((r, f"head={head!r} 无 token 命中"))
            continue
        ok.append((r, head, nosubj, removed, tids))
    return ok, skip


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default=str(root / "parti_hi"),
                    help="基线臂目录（读 prompt 名单 + 比对 1024 基图）")
    ap.add_argument("--out", default=str(root / "parti_v1"))
    ap.add_argument("--subjects", default=None,
                    help="VLM 主体缓存；默认 <hi>/vlm_subjects.json")
    ap.add_argument("--heads-file", default=None,
                    help='JSON {"263": "statue"} 覆盖自动解析')
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--push", action="store_true",
                    help="v2a：背景位置的负分支加入被摘掉的主体短语，"
                         "把'不要求画'升级为'主动排斥'。零额外成本 —— "
                         "CFG 的负分支本来每步都在算。"
                         "动机见代码内注释（v1.2 锐化门零增益）")
    ap.add_argument("--uniform", action="store_true",
                    help="消融：门控图拍平成常数（均值不变、空间结构去掉）。"
                         "回答\"逐位置自适应是不是真的在起作用\" —— "
                         "若均匀门也能拿到同样的 delta，空间主张即告死亡")
    ap.add_argument("--layers", default="all",
                    help="v1.2 选层：top4 / top8 / all（gate_layers.json）。"
                         "全层平均对比度仅 1.50，top4 为 3.68")
    ap.add_argument("--refresh", type=int, default=0,
                    help="v1.1：放大阶段前 N 步重录门控图（建议 1）。"
                         "0 = 原版 v1。预注册判据见 gate_refresh.py")
    ap.add_argument("--refresh-canon", type=int, default=256)
    ap.add_argument("--plan", action="store_true",
                    help="只打印解析结果，不加载模型不烧 GPU")
    a = ap.parse_args()

    hi = Path(a.hi)
    rows = [json.loads(l) for l in (hi / "manifest.jsonl").open()]
    if a.idx:
        want = set(a.idx)
        rows = [r for r in rows if r["idx"] in want]
    subj_p = Path(a.subjects) if a.subjects else hi / "vlm_subjects.json"
    if not subj_p.exists():
        sys.exit(f"没有 {subj_p} —— 先跑 vlm_count.py --null（或 --delta），"
                 f"它会把 34 条的主体词落盘；或用 --heads-file 全量手工给")
    subjects = json.loads(subj_p.read_text())
    overrides = (json.loads(Path(a.heads_file).read_text())
                 if a.heads_file else {})

    LAYERS = None
    if a.layers != "all":
        _g = json.loads((Path(__file__).resolve().parent
                         / "gate_layers.json").read_text())
        LAYERS = list(_g["top4"])
        if a.layers == "top8":
            LAYERS += _g["top8_extra"]
        print(f"选层：{a.layers} -> {len(LAYERS)} 层"
              f"（对比度 {_g['contrast_measured'].get(a.layers, '?')} "
              f"vs 全层 {_g['contrast_measured']['all_layers']}）")

    from caption_audit import load_tokenizer
    tok = load_tokenizer()
    ok, skip = resolve(rows, subjects, overrides, tok)

    print(f"{len(rows)} 条：可跑 {len(ok)}，跳过 {len(skip)}\n")
    for r, head, nosubj, removed, tids in ok:
        print(f"[{r['idx']:>4}] head={head!r:<14} tok={tids}"
              f"\n       摘掉: {removed!r}"
              f"\n       剩下: {nosubj[:72]}")
    if skip:
        print("\n跳过（--heads-file 可救）：")
        for r, why in skip:
            print(f"[{r['idx']:>4}] {why}\n       {r['prompt'][:72]}")
    if a.plan:
        print("\n--plan：没有加载模型。过目无误后去掉该参数再跑。")
        return 0
    if not ok:
        sys.exit("没有可跑的条目")

    import torch

    def pin(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
               / "snapshots").glob("*"):
        if not (s_ / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s_ / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mani = out / "manifest.jsonl"
    done = set()
    if mani.exists():
        done = {json.loads(l)["idx"] for l in mani.open()}
        print(f"\nmanifest 已有 {len(done)} 条，跳过")
    todo = [t for t in ok if t[0]["idx"] not in done]

    orig_step = pipe.noise_pred_step
    t_all = time.time()
    with mani.open("a") as mf:
        for n, (r, head, nosubj, removed, tids) in enumerate(todo, 1):
            idx = r["idx"]
            gate = (RefreshGate(tids, strength=a.s, layers=LAYERS,
                                refresh_canon=a.refresh_canon,
                                refresh_steps=a.refresh)
                    if a.refresh > 0
                    else BlendGate(tids, strength=a.s, layers=LAYERS,
                                   uniform=a.uniform))
            # v2a 推拉：背景位置的**负分支**里加入被摘掉的主体短语。
            #
            # 为什么要这一刀 —— 我们自己的数据逼出来的：
            #   v1.2 把门锐化（过渡带 100%->40%，20/31 真开）后 Rep+ 纹丝
            #   不动（0.500），说明**残留重复不是定位不准造成的**。定位对了，
            #   副本照样长。病根在于 v1 的干预是**被动**的：只是"没要求你
            #   画女神像"，而律的高段（空视野大）本就缺约束 —— 空天上结构
            #   引导几乎不提供信息，文本是唯一信号，删一个词拦不住它。
            #
            # CFG 是 eps = eps_neg + s*(eps_pos - eps_neg)，负分支每一步都
            # 在算却一直没用。把主体短语放进背景位置的负分支，性质就从
            # "不要求"变成"**主动排斥**"，且**额外成本为零**（只多一次
            # 文本编码，前向次数不变）。
            neg_bg = NEGATIVE
            if a.push and removed:
                neg_bg = f"{NEGATIVE}, {removed}"
            pe, npe, _, _ = pipe.encode_prompt(
                prompt=nosubj, device="cuda", num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=neg_bg)
            gate.alt = torch.cat([npe, pe])
            if a.push and n == 1:
                print(f"    推拉开：背景负分支 = NEGATIVE + {removed!r}")
            procs = dict(pipe.unet.attn_processors)
            for k_ in procs:
                if k_.endswith("attn2.processor"):
                    procs[k_] = BlendCrossAttn(gate,
                                               k_.replace(".processor", ""))
            # 选层名单必须真的对得上处理器键名。对不上的话门控图一张都录不到，
            # map 恒为 None -> weights() 返回 None -> 完全不混合，输出与基线
            # 逐字节相同 —— 那是一次“看起来跑了、其实什么都没做”的静默失败。
            if LAYERS is not None:
                have = {k_.replace(".processor", "") for k_ in procs}
                miss = [n for n in LAYERS if n not in have]
                if miss:
                    raise SystemExit(
                        f"选层名单有 {len(miss)} 个名字不在 UNet 里，"
                        f"门会静默失效：{miss[:3]}")
            pipe.unet.set_attn_processor(procs)

            def patched(latents, t, *args, _o=orig_step, _g=gate, **kw2):
                ph = 1 if latents.shape[-1] <= 128 else 2
                if ph == 2 and _g.phase == 1:
                    _g.finalize()                       # 基础阶段那张先定格
                    if isinstance(_g, RefreshGate):
                        _g.begin_refresh()              # phase -> 'r'，开重录窗口
                    else:
                        _g.phase = 2
                elif ph == 1:
                    _g.phase = 1                        # 'r' 不许被踩回去
                out = _o(latents, t, *args, **kw2)
                if getattr(_g, "phase", None) == "r":
                    _g.step_done()
                return out
            pipe.noise_pred_step = patched

            pin(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                imgs = pipe(r["prompt"], negative_prompt=NEGATIVE,
                            height=1024, width=1024,
                            generator=torch.Generator(device="cuda")
                            .manual_seed(a.seed),
                            num_inference_steps=a.steps, guidance_scale=7.5,
                            restart_ratio=0.4, scale_factor=0.125,
                            upsample_stage=a.stage)
            except torch.cuda.OutOfMemoryError:
                print(f"\n{idx} OOM，跳过")
                torch.cuda.empty_cache()
                pipe.noise_pred_step = orig_step
                continue
            dt = time.time() - t0
            files = {}
            for im in imgs:
                p = out / f"{idx:05d}_{im.width}.png"
                im.save(p)
                files[im.width] = p.name
            if gate.map is not None:
                torch.save(gate.map.cpu(), out / f"{idx:05d}_mask.pt")
            cov = gate.stats()
            band = gate.band_stats() if isinstance(gate, RefreshGate) else None
            if band:
                print(f"\n    门控图重录：{band['before']} -> {band['after']}"
                      f"  图尺寸 {band['map_size']}  重录 {band['n_refresh']} 次")
            elif cov:
                # R1 是预注册判据（§3.14c），每一行都得留下带宽读数，
                # 不能只在 RefreshGate 那条支路上才有
                print(f"\n    门控图：主体 {cov[0]:.0%}  背景<0.3 {cov[1]:.0%}"
                      f"  过渡带 {cov[2]:.0%}"
                      + ("  ← 仍是半开" if cov[2] > 0.5 or cov[1] < 0.1 else ""))
            mf.write(json.dumps({
                "idx": idx, "stratum": "v1", "prompt": r["prompt"],
                "head": head, "removed": removed, "s": a.s, "seed": a.seed,
                "files": files, "gate_cov": cov, "band": band,
                "refresh": a.refresh, "layers": a.layers,
                "uniform": a.uniform, "push": a.push,
                "sec": round(dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            pipe.noise_pred_step = orig_step
            el = time.time() - t_all
            print(f"\r{n}/{len(todo)}  {el/60:.0f} 分钟  "
                  f"剩约 {el/n*(len(todo)-n)/60:.0f} 分钟  cov={cov}",
                  end="", flush=True)

    # ---------- 一致性检查：v1 的 1024 基图 == 基线的 1024 基图 ----------
    print("\n\n基础阶段一致性（v1 在 phase1 只录不混，应逐像素同基线）：")
    import numpy as np
    from PIL import Image
    same = diff = missing = 0
    for l in mani.open():
        r = json.loads(l)
        f1 = r["files"].get("1024") or r["files"].get(1024)
        b_rows = {json.loads(x)["idx"]: json.loads(x)
                  for x in (hi / "manifest.jsonl").open()}
        b = b_rows.get(r["idx"])
        fb = b and (b["files"].get("1024") or b["files"].get(1024))
        if not (f1 and fb):
            missing += 1
            continue
        x = np.asarray(Image.open(out / f1), dtype=np.int16)
        y = np.asarray(Image.open(hi / fb), dtype=np.int16)
        if x.shape == y.shape and int(np.abs(x - y).max()) == 0:
            same += 1
        else:
            diff += 1
            print(f"  [{r['idx']}] 基图不同！max|Δ|="
                  f"{int(np.abs(x - y).max()) if x.shape == y.shape else '形状不同'}"
                  f"  <- 录图路径污染了前向，必须查")
    print(f"  相同 {same} / 不同 {diff} / 缺文件 {missing}")
    print("\n下一步：\n"
          "  python scalediff_probe/hi_contact.py --hi " + str(out) +
          "   # 眼睛：263 女神还在吗\n"
          "  python scalediff_probe/vlm_count.py --delta --hi " + str(out) +
          "   # v1 臂 delta，与 parti_hi 的逐条对照")
    return 0


if __name__ == "__main__":
    sys.exit(main())
