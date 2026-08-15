"""C2 存在性检查：close-up 图上的"局部不合理内容"在我们手上复现得出来吗？

**为什么查这个**（两篇顶会在各自 Limitations 里独立点名，机制完全不同）：

- AccDiffusion（ECCV）限制 (3)：
  *"Relying on LDMs' prior knowledge of cropped images, it may produce
  **local irrational content in sharp close-up image generation**."*
- ScaleDiff（NeurIPS）附录 D：
  *"being a patch-based approach, it relies heavily on the diffusion
  model's prior knowledge of cropped image regions. This can sometimes
  lead to **inconsistent local content when generating sharp close-up
  images**."*

两支队伍、两种机制（patch 分解 vs 窗口注意力）、几乎同一句话。
**都只是"承认"，无人当成问题去解。**

**和我们已有资产的咬合**：我们的律是 重复 ∝ R²×(1−主体覆盖)。
close-up 的主体覆盖 ≈ 1，正是律的**另一端** —— 那一端不重复，
但文献说那一端有**另一种**失效。同一根轴的两头、两种失效，
而那根轴我们已经量过。

**预注册判据（写在跑之前）**：
    看得见（>=6/20 条出现可指认的局部瞎编）-> C2 成立，进入设计；
    3~5 条                                  -> 边缘，扩到 40 条再判；
    <=2 条                                  -> 在我们的配置下不复现，
                                              **C2 当场判死，转 C1**。
"局部瞎编"的判准（先说死，免得看到图再放宽）：
    在 4096 的**局部裁块**里出现基图同位置**没有**的、且**语义上说不通**
    的结构 —— 多出来的眼睛/牙齿/手指、纹理里长出人脸、器官重复拼接。
    仅仅"更清晰"或"细节更多"**不算**。

close-up prompt 的选取原则：主体铺满画面、几乎没有背景（覆盖≈1），
且是模型见过大量特写的类别（否则分不清是"外推失败"还是"本来就不会画"）。

    python scalediff_probe/closeup_probe.py --plan     # 只看名单
    nohup python scalediff_probe/closeup_probe.py > closeup.log 2>&1 &
    python scalediff_probe/multi_look.py --arms "ScaleDiff=$SD_OUT/closeup" --idx 0 1 2
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

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"

# 20 条 close-up：主体铺满、背景极少（覆盖≈1 = 律的另一端）。
# 分四组，便于事后看失效是否集中在某一类 —— 若集中在"生物/面部"，
# 那更像是模型先验问题；若四组都有，才是外推机制的问题。
PROMPTS = [
    # A 生物特写（模型见得最多，最不该出错）
    "extreme close-up of a human iris, macro photography, sharp focus",
    "extreme close-up of a cat's face filling the frame, whiskers in focus",
    "macro photograph of a bee's compound eye, extreme detail",
    "close-up portrait of an elderly man's face, every wrinkle visible",
    "macro shot of a butterfly wing scales, iridescent",
    # B 材质/纹理特写（无语义结构，考察纯纹理外推）
    "extreme close-up of woven denim fabric, macro",
    "macro photograph of cracked dry earth filling the frame",
    "extreme close-up of rusted metal surface with peeling paint",
    "macro shot of honeycomb cells filled with honey",
    "close-up of tree bark texture, deep grooves, filling the frame",
    # C 有精细结构的人造物（考察"结构说不说得通"）
    "extreme close-up of a mechanical watch movement, gears and jewels",
    "macro photograph of a circuit board, components and solder joints",
    "close-up of a typewriter keyboard, keys filling the frame",
    "extreme close-up of a violin's f-hole and strings",
    "macro shot of a bicycle chain and sprocket teeth",
    # D 食物/自然（高频细节，常见于展示图）
    "extreme close-up of a strawberry surface, seeds and texture",
    "macro photograph of a dandelion seed head, filling the frame",
    "close-up of coffee beans filling the entire frame",
    "extreme close-up of ice crystals on glass, macro",
    "macro shot of a peacock feather, eye pattern filling the frame",
]
GROUP = ["A 生物"] * 5 + ["B 材质"] * 5 + ["C 人造物"] * 5 + ["D 食物/自然"] * 5


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "closeup"))
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--stage", type=int, default=2, help="2 = 到 4096")
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    items = list(enumerate(zip(PROMPTS, GROUP)))
    if a.idx:
        want = set(a.idx)
        items = [t for t in items if t[0] in want]

    print(f"C2 存在性检查：{len(items)} 条 close-up prompt，"
          f"ScaleDiff 原生跑到 4096²，seed={a.seed}\n")
    for i, (p, g) in items:
        print(f"  [{i:>2}] {g:<10} {p}")
    print("\n预注册判据：>=6/20 出现可指认的局部瞎编 -> C2 成立；"
          "3~5 边缘扩样本；<=2 -> 判死转 C1")
    print("判准：4096 局部裁块里出现基图同位置没有、且语义说不通的结构"
          "（多余的眼/齿/指、纹理里长人脸、器官重复拼接）。"
          "\n      **仅仅更清晰、细节更多不算。**")
    if a.plan:
        print("\n--plan：没有加载模型。")
        return 0

    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    kw = {"torch_dtype": torch.float16}
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
               / "snapshots").glob("*"):
        u = s_ / "unet"
        if not (u / "diffusion_pytorch_model.safetensors").exists() \
                and list(u.glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mani = out / "manifest.jsonl"
    done = ({json.loads(l)["idx"] for l in mani.open()}
            if mani.exists() else set())
    todo = [t for t in items if t[0] not in done]
    if done:
        print(f"\nmanifest 已有 {len(done)} 条，跳过")

    t_all = time.time()
    with mani.open("a") as mf:
        for n, (i, (prompt, grp)) in enumerate(todo, 1):
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                imgs = pipe(prompt, negative_prompt=NEGATIVE,
                            height=1024, width=1024,
                            generator=torch.Generator(device="cuda")
                            .manual_seed(a.seed),
                            num_inference_steps=a.steps, guidance_scale=7.5,
                            restart_ratio=0.4, scale_factor=0.125,
                            upsample_stage=a.stage)
            except torch.cuda.OutOfMemoryError:
                print(f"\n[{i}] OOM，跳过")
                torch.cuda.empty_cache()
                continue
            dt = time.time() - t0
            files = {}
            for im in imgs:
                p = out / f"{i:05d}_{im.width}.png"
                im.save(p)
                files[str(im.width)] = p.name
            mf.write(json.dumps({
                "idx": i, "prompt": prompt, "group": grp, "seed": a.seed,
                "files": files, "sec": round(dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "stratum": "closeup",
            }, ensure_ascii=False) + "\n")
            mf.flush()
            el = (time.time() - t_all) / 60
            print(f"{n}/{len(todo)} [{i:>2}] {dt:5.1f}s  已 {el:.0f} 分钟  "
                  f"剩约 {el/n*(len(todo)-n):.0f} 分钟  {prompt[:44]}")

    print(f"\n总计 {(time.time()-t_all)/60:.1f} 分钟")
    print(f"""
下一步 —— **用眼睛看，别先算任何数**：
  python scalediff_probe/multi_look.py --arms "ScaleDiff={out}" \\
      --idx 0 1 2 3 4 5 6 7 8 9
  # 逐条裁块放大（局部瞎编只在放大后看得见）：
  python scalediff_probe/multi_look.py --arms "ScaleDiff={out}" \\
      --idx 0 --crop 0.5 0.5 0.25 --cell 900

判据回顾：>=6/20 可指认 -> C2 成立；3~5 扩样本；<=2 -> 判死转 C1。
另记录每条属于 A/B/C/D 哪组 —— 若失效只集中在"A 生物"，那更可能是
模型先验问题而非外推机制问题，结论要相应收窄。""")


if __name__ == "__main__":
    sys.exit(main() or 0)
