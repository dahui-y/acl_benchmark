"""三个诊断实验。目的是拿到【病因】，不是拿到指标。

为什么要做：我们手上只有第 1 环（失效：4096² 会幻觉出 prompt 主体）。
从失效到方法，中间必须有病因这一环 —— 三篇在位者无一例外：

    StyleID     Fig.5 同时注入 style 的 Q/K/V，颜色仍跟 content
                => 颜色不住在 self-attention 里 => 改初始 latent
    HiWave      Fig.11 去掉低频上的 CFG，重复就消失
                => 病因是低频上的 CFG => 频段选择性 CFG
    AccDiffusion 所有 patch 共用一句 prompt => patch 感知 prompt

方法都是从诊断【推】出来的，不是猜的。

假说（来自代码 + 2048/4096 对照）：
    NPA 的 KV 邻域固定是"一张基分辨率画布"。2048² 时它占整图 1/4，
    4096² 时占 1/16 —— 邻域看起来像一张完整的风景照，而 cross-attention
    把整句 prompt 送给了每个位置，于是每个邻域各画了一个主体。
    （实测：2048² 幻影 0 个，4096² 至少 3 个。）

    res     同一张基图，扫 2048 / 4096 / 8192
    prompt  固定基图，只把放大阶段的 prompt 换成纯风景  <- 最关键
    kv      放大 NPA 的 KV 邻域，query 分块不变

判定表（三个结果一起看）：
                       prompt 消融后幻影消失      幻影仍在
    KV 放大后减少      局部邻域 x 全局文本条件     纯局部性
    KV 放大后无效      文本条件通路本身            假说全错，回去看图

    python scalediff_probe/diag.py res
    python scalediff_probe/diag.py prompt
    python scalediff_probe/diag.py kv --mult 2
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

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

# seed 77 那一张，已经肉眼确认了三个幻影。三个实验都以它为基准，
# 换一条 prompt 就没有可比的基线了。
PROMPT = ("a photograph of a lone hiker standing on a rocky ridge, vast forested valley "
          "and distant snow mountains behind, golden hour")
# 同一句话去掉主体。景物词一个不改 —— 只有"有没有可数主体"这一个变量。
PROMPT_NOSUBJ = ("a photograph of a rocky ridge, vast forested valley "
                 "and distant snow mountains behind, golden hour")


def load_pipe():
    import torch
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for snap in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots").glob("*"):
        u = snap / "unet"
        if not (u / "diffusion_pytorch_model.safetensors").exists() and list(u.glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    return pipe


def run(pipe, out, tag, seed, stage, prompt=PROMPT, steps=50):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        imgs = pipe(prompt, negative_prompt=NEG, height=1024, width=1024,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    num_inference_steps=steps, guidance_scale=7.5,
                    restart_ratio=0.4, scale_factor=0.125, upsample_stage=stage)
    except torch.cuda.OutOfMemoryError:
        print(f"{tag:<28}OOM  峰值 {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
        torch.cuda.empty_cache()
        return None
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    for im in imgs:
        im.save(out / f"{tag}_{im.width}.png")
    print(f"{tag:<28}{dt:7.1f}s  峰值 {peak:5.1f} GB  -> {[i.width for i in imgs]}")
    return {"tag": tag, "sec": round(dt, 1), "peak_gb": round(peak, 2),
            "res": [i.width for i in imgs]}


# ---------------------------------------------------------------- res
def exp_res(a, out):
    """分辨率扫描。假说预测幻影数随分辨率单调增长。"""
    pipe = load_pipe()
    rows = []
    for st in a.stages:
        r = run(pipe, out, f"res_stage{st}_s{a.seed}", a.seed, st)
        if r:
            rows.append(r)
    return rows


# ---------------------------------------------------------------- prompt
def exp_prompt(a, out):
    """固定基图，只换放大阶段的 prompt。

    这是 StyleID 式的做法：把一个杠杆推到极限（主体词完全去掉），看什么还动。
    实现上不改 ScaleDiff 的代码 —— noise_pred_step 把 prompt_embeds 当参数收，
    在外面包一层，按 latent 尺寸判断当前在哪个阶段就行：基阶段的 latent 是
    128x128，放大阶段是 256 或 512。
    """
    import torch
    pipe = load_pipe()
    rows = []

    for label, up_prompt in (("keep", None), ("nosubj", PROMPT_NOSUBJ)):
        if up_prompt is None:
            alt = None
        else:
            pe, npe, pool, npool = pipe.encode_prompt(
                prompt=up_prompt, device="cuda", num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=NEG)
            # 与 __call__ 里一致：CFG 时把 negative 拼在前面
            alt = (torch.cat([npe, pe]), torch.cat([npool, pool]))

        orig = pipe.noise_pred_step

        def patched(latents, t, prompt_embeds, add_text_embeds, add_time_ids,
                    *args, _alt=alt, _orig=orig, **kw):
            # 基阶段 latent 是 128；大于它就是放大阶段
            if _alt is not None and latents.shape[-1] > 128:
                prompt_embeds, add_text_embeds = _alt
            return _orig(latents, t, prompt_embeds, add_text_embeds, add_time_ids,
                         *args, **kw)

        pipe.noise_pred_step = patched
        r = run(pipe, out, f"prompt_{label}_s{a.seed}", a.seed, a.stage)
        pipe.noise_pred_step = orig
        if r:
            r["variant"] = label
            rows.append(r)

    print("\n两张的【基图必须逐像素一致】—— 基阶段的 prompt 没动过。")
    print("若 nosubj 的 4096² 不再长人 => 病因在 cross-attention 通路。")
    return rows


# ---------------------------------------------------------------- kv
def exp_kv(a, out):
    """放大 NPA 的 KV 邻域，query 分块不变。

    原实现里 window_size 不是自由参数：h = height_scale * window_size 必须等于
    该层特征图的边长，所以 window_size 就等于"这张特征图在基分辨率下的边长"，
    也就是 KV 邻域恰好是【一张基分辨率画布】。直接改 window_size 会让 reshape
    对不上。要变的是 KV 的范围 p1，query 的分块 p2 保持不变。
    """
    import torch
    import pipeline_scalediff_sdxl as P
    from attn_scalediff_sdxl import AttnControl

    orig_get = AttnControl.get_kv_view

    def wide(self, window_size, _mult=a.mult):
        h = int(self.height_scale * window_size)
        w = int(self.width_scale * window_size)
        p2 = int(window_size) // 2                      # query 分块，不动
        p1 = min(int(window_size * _mult), h, w)        # KV 邻域，放大
        off = (p1 - p2) // 2                            # 保持以 query 块为中心
        idx = []
        for r in torch.clamp(torch.arange(0, h, p2) - off, 0, max(h - p1, 0)):
            for c in torch.clamp(torch.arange(0, w, p2) - off, 0, max(w - p1, 0)):
                gh, gw = torch.meshgrid(torch.arange(r, r + p1),
                                        torch.arange(c, c + p1), indexing="ij")
                idx.append((gh * w + gw).flatten())
        return torch.stack(idx, dim=0)

    rows = []
    pipe = load_pipe()
    for m in ([1.0] + list(a.mult_list)):
        a.mult = m
        AttnControl.get_kv_view = orig_get if m == 1.0 else wide
        r = run(pipe, out, f"kv_x{m:g}_s{a.seed}", a.seed, a.stage)
        if r:
            r["kv_mult"] = m
            rows.append(r)
    AttnControl.get_kv_view = orig_get
    print("\nKV 邻域放大 m 倍 => 该层注意力代价约 m^2 倍。")
    print("若幻影随 m 减少 => 病因确认在局部性。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["res", "prompt", "kv"])
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--mult-list", type=float, nargs="+", default=[2.0])
    ap.add_argument("--out", default=os.environ.get("SD_OUT", "./scalediff_out"))
    a = ap.parse_args()

    out = Path(a.out) / "diag" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    rows = {"res": exp_res, "prompt": exp_prompt, "kv": exp_kv}[a.exp](a, out)
    (out / "log.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n图在 {out}")
    print(f"数幻影：python scalediff_probe/count_objects.py --check  的参数已冻结，"
          f"改用 --batch {out} 需要 manifest；先用 view.py 肉眼过一遍。")


if __name__ == "__main__":
    main()
