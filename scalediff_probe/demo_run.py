"""DemoFusion（带门 fork）跑批：E4 战场的两臂生成器。

臂：
    base   pipe.gate = None      —— 钩子全是 no-op，应与原版逐字节相同
    v1     pipe.gate = DemoGate  —— phase-1 录图，patch/dilated 支逐窗混合

冒烟顺序（先小后大，别一上来烧 4096）：
    python scalediff_probe/demo_run.py --smoke                # 1 条 2048，两臂
    python scalediff_probe/demo_run.py --smoke --compare-original
                                  # 外加原版管线跑同 seed，md5 对比 base 臂
    python scalediff_probe/demo_run.py --idx 331 263 583 --size 4096
                                  # 触发集样本上正式两臂

主体词与 v1_trigger_run 同一来源（vlm_subjects.json 表面形对齐 +
v1_heads.json 覆盖），保证跨管线用的是同一个门。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                                   # noqa: E402
from subject_phrases import strip_subject                      # noqa: E402
from demo_gate import DemoGate                                 # noqa: E402
from method_v1 import BlendCrossAttn                           # noqa: E402
from v1_trigger_run import surface_head, token_ids_of          # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
SMOKE_PROMPT = ("The Statue of Liberty surrounded by helicopters", "Statue")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def load_pipe(gated=True):
    import torch
    if gated:
        from pipeline_demofusion_gated import DemoFusionSDXLPipeline
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "help_code" / "DemoFusion"))
        from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
               / "snapshots").glob("*"):
        if not (s_ / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s_ / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = DemoFusionSDXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    return pipe


def run_one(pipe, prompt, head, seed, size, arm, out, tag,
            gate_dilated=True, vb=8, lowvram=False):
    import torch
    gate = None
    if arm == "v1":
        tok = pipe.tokenizer
        tids = token_ids_of(tok, prompt, head)
        nosubj, removed = strip_subject(prompt, head)
        if not tids or not removed:
            print(f"    v1 不可用（tok={tids}, removed={removed!r}），跳过")
            return None
        gate = DemoGate(tids, strength=1.0, gate_dilated=gate_dilated)
        pe, npe, _, _ = pipe.encode_prompt(
            prompt=nosubj, device="cuda", num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=NEGATIVE)
        gate.set_alt(torch.cat([npe, pe]))
        procs = dict(pipe.unet.attn_processors)
        for k in procs:
            if k.endswith("attn2.processor"):
                procs[k] = BlendCrossAttn(gate)
        pipe.unet.set_attn_processor(procs)
        print(f"    门已装：tok={tids} 摘掉 {removed!r}")
    else:
        # base 臂：卸回默认 processor，防止上一臂的门残留
        pipe.unet.set_default_attn_processor()
    pipe.gate = gate

    # DemoFusion 的窗口抖动走 Python random（get_views random_jitter），
    # 不钉它的话臂间抖动序列不同 —— 配对性被破坏，md5 也对不上。
    # 首轮冒烟 2048 md5 就是栽在这里：phase-1（无 get_views）一致，
    # phase-2 不一致。三个 RNG 全钉。
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    images = pipe(prompt, negative_prompt=NEGATIVE,
                  height=size, width=size,
                  generator=torch.Generator(device="cuda").manual_seed(seed),
                  num_inference_steps=50, guidance_scale=7.5,
                  view_batch_size=vb, stride=64,
                  cosine_scale_1=3., cosine_scale_2=1., cosine_scale_3=1.,
                  sigma=0.8, multi_decoder=True, show_image=False,
                  lowvram=lowvram)
    dt = time.time() - t0
    files = {}
    for im in images:
        p = out / f"{tag}_{arm}_{im.width}.png"
        im.save(p)
        files[im.width] = p.name
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"    {arm:<5} {dt/60:.1f} 分钟  峰值 {peak:.1f}GB  "
          f"出图 {sorted(files)}")
    if gate is not None:
        gate.clear_views()
    pipe.gate = None
    return {"files": files, "sec": round(dt, 1), "peak_gb": round(peak, 2)}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1 条 showcase prompt @2048，两臂")
    ap.add_argument("--compare-original", action="store_true",
                    help="smoke 时再用原版管线跑同 seed，md5 对比 base 臂")
    ap.add_argument("--idx", type=int, nargs="*", default=None,
                    help="parti 触发集样本（读 parti_hi manifest 拿 prompt）")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--arms", nargs="+", default=["base", "v1"])
    ap.add_argument("--no-gate-dilated", action="store_true")
    ap.add_argument("--out", default=str(root / "demo_e4"))
    ap.add_argument("--vb", type=int, default=8, help="view_batch_size")
    ap.add_argument("--lowvram", action="store_true",
                    help="4096 建议开：2048 冒烟峰值已 18GB")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mani = out / "manifest.jsonl"
    done = set()
    if mani.exists():
        for l in mani.open():
            r = json.loads(l)
            done.add((r["tag"], r["arm"], r["size"]))

    jobs = []
    if a.smoke or not a.idx:
        jobs.append(("smoke331",) + SMOKE_PROMPT)
    if a.idx:
        hi = root / "parti_hi"
        rows = {json.loads(l)["idx"]: json.loads(l)
                for l in (hi / "manifest.jsonl").open()}
        subj = json.loads((hi / "vlm_subjects.json").read_text())
        heads = json.loads((Path(__file__).resolve().parent
                            / "v1_heads.json").read_text())
        for i in a.idx:
            r = rows.get(i)
            if not r:
                print(f"[{i}] 不在 parti_hi manifest，跳过")
                continue
            h = heads.get(str(i)) or surface_head(r["prompt"],
                                                  subj.get(str(i)))[0]
            if not h:
                print(f"[{i}] 无 head，跳过")
                continue
            jobs.append((f"{i:05d}", r["prompt"], h))

    pipe = load_pipe(gated=True)
    with mani.open("a") as mf:
        for tag, prompt, head in jobs:
            print(f"\n[{tag}] {prompt[:70]}")
            for arm in a.arms:
                if (tag, arm, a.size) in done:
                    print(f"    {arm} 已有，跳过")
                    continue
                rec = run_one(pipe, prompt, head, a.seed, a.size, arm, out,
                              tag, gate_dilated=not a.no_gate_dilated,
                              vb=a.vb, lowvram=a.lowvram)
                if rec:
                    mf.write(json.dumps({
                        "tag": tag, "arm": arm, "size": a.size,
                        "prompt": prompt, "head": head, "seed": a.seed,
                        **rec}, ensure_ascii=False) + "\n")
                    mf.flush()

    if a.compare_original and (a.smoke or not a.idx):
        print("\n== 原版管线对照（证明门关 = 逐字节原版）==")
        del pipe
        import torch
        torch.cuda.empty_cache()
        pipe0 = load_pipe(gated=False)
        tag, prompt, head = ("smoke331",) + SMOKE_PROMPT
        rec = run_one(pipe0, prompt, head, a.seed, a.size, "orig", out, tag,
                      vb=a.vb, lowvram=a.lowvram)
        if rec:
            for wdt, f in rec["files"].items():
                g = out / f"{tag}_base_{wdt}.png"
                if g.exists():
                    same = md5(out / f) == md5(g)
                    print(f"    {wdt}px  md5 {'一致 ✓' if same else '不一致 ✗ 必须查'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
