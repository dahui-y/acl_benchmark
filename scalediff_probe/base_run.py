"""只生成 1024² 基图 —— 触发子群筛选的前提，也是 4096² 的第一步。

**基图不是额外工作**：ScaleDiff 的流水线本来就是 1024 -> 2048 -> 4096，
1024 那张是必经的第一步。区别只在于我们给**比最终要用的更多**的 prompt
先跑这一步，然后据此挑。

为什么必须先出基图 —— **基图上看的是构图，不是重复**：

    重复只发生在放大阶段。基图是**对的**（一个 hiker 就是一个；实测
    lone 类 excess：1024² +0.20 / 2048² +1.00 / 4096² +5.40）。
    我们在基图上数的是**预测因子**，不是症状：

        基图 -> 检测器找出所有物体 -> 按 R×R 切块 -> 数有多少块没有物体
        -> 空视野比例（= 构图属性：画面上多大比例的区域没有东西）

    律说：不含主体的邻域数 ≈ R² × (1 − 主体覆盖率)。R² 由放大倍数决定，
    **主体覆盖率是基图的性质**。主体占满画面 -> 每个受限视野里都有主体
    的一部分 -> 放大时画的是延续 -> 不重复；主体只占一小块 -> 绝大多数
    视野没有主体 -> 每个都当空白画布重画 -> 重复。

    所以是**从基图的构图，预测放大后会不会重复** —— 像用今天的气压预报
    明天的雨，气压不是雨。

    **这个判断不看 4096 的结果，所以不是循环论证**（按律的自变量挑，
    不是按因变量挑）。而且基图两个 arm 逐字节相同，挑选对两边完全对称。
    基图为真的另一个用处：delta = count(4096) − count(基图) 可以拿基图
    当真值，因为它对。

算一笔账：
    基图 3000 条 ~6 小时  ->  决定那 90 GPU 小时（4 天）花在哪些 prompt 上，
    顺带回答 §1.2.1a 的预注册问题（触发区在真实语料里的占比）。
    产物第 ③ 步原样复用，一张都不浪费。
    **跳过这步直接跑 1000 条随机 prompt，四天后大概率拿到一张"几乎没变化"
    的表** —— 因为多数 LAION prompt（商品图）根本不在触发区。

这里**不走 ScaleDiff 的放大流水线**，就是原版 SDXL 出 1024²，
和 ScaleDiff 第一级完全一致（同 ckpt / 同 steps / 同 CFG / 同 negative）。

断点续跑：manifest 里已有的直接跳过，可随时 Ctrl-C 再接。

    python scalediff_probe/base_run.py                       # 默认 eval_prompts.json 全部
    python scalediff_probe/base_run.py --split tune          # 先跑 tune split 探路
    python scalediff_probe/base_run.py --limit 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                   # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for snap in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
                 / "snapshots").glob("*"):
        u = snap / "unet"
        if (u / "diffusion_pytorch_model.safetensors").exists():
            return False
        if list(u.glob("*.fp16.safetensors")):
            return True
    return False


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(root / "eval_prompts.json"))
    ap.add_argument("--out", default=str(root / "laion_base"))
    ap.add_argument("--split", default=None, choices=["tune", "eval"],
                    help="只跑某个 split；省略则两个都跑")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alive", default=None,
                    help="fetch_real_images.py 出的 alive.json —— "
                         "**只给真图还活着的 prompt 生成基图**，"
                         "别给失效的烧 GPU（第一版就是这么排错的）。"
                         "触发子群不需要真图，所以那部分不用加这个参数。")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.5)
    a = ap.parse_args()

    meta = json.loads(Path(a.prompts).read_text())
    items = [{**x, "_idx": i} for i, x in enumerate(meta["items"])]
    if a.split:
        items = [x for x in items if x.get("split") == a.split]
    if a.alive:
        alive = set(json.loads(Path(a.alive).read_text())["alive_idx"])
        items = [{**x, "_idx": i} for i, x in enumerate(meta["items"])
                 if i in alive and (not a.split or x.get("split") == a.split)]
        print(f"按 {a.alive} 过滤：真图存活 {len(alive)} 条")
    if a.limit:
        items = items[:a.limit]
    if not items:
        sys.exit(f"{a.prompts} 里没有符合条件的 prompt")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for line in manifest_path.open():
            done.add(json.loads(line)["idx"])
        print(f"manifest 里已有 {len(done)} 条，跳过")

    # idx 必须是 eval_prompts.json 里的原始下标，否则和 alive.json、
    # 真图文件名对不上。所以带上 x 自己记的 _idx（--alive 路径下已设）。
    todo = [(x.get("_idx", i), x) for i, x in enumerate(items)
            if x.get("_idx", i) not in done]
    print(f"{len(items)} 条，待跑 {len(todo)} 条 -> {out}")
    print(f"来源 {meta.get('repo')}   split={a.split or '全部'}   seed={a.seed}")
    if not todo:
        print("没有要跑的了。")
        return 0

    import torch
    from diffusers import StableDiffusionXLPipeline

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = StableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    t0 = time.time()
    with manifest_path.open("a") as mf:
        for n, (i, x) in enumerate(todo, 1):
            # 全局 RNG 也钉住，和其余批次同一纪律
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            g = torch.Generator(device="cuda").manual_seed(a.seed)
            im = pipe(x["prompt"], negative_prompt=NEGATIVE,
                      height=1024, width=1024, generator=g,
                      num_inference_steps=a.steps,
                      guidance_scale=a.cfg).images[0]
            fn = f"{i:05d}.png"
            im.save(out / fn)
            mf.write(json.dumps({
                "idx": i, "file": fn, "prompt": x["prompt"],
                "url": x.get("url"), "split": x.get("split"),
                "seed": a.seed,
            }, ensure_ascii=False) + "\n")
            mf.flush()
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print(f"\r{n}/{len(todo)}  {el/60:.0f} 分钟已用  "
                  f"剩约 {eta/60:.0f} 分钟   {x['prompt'][:44]}",
                  end="", flush=True)
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"\n下一步：python scalediff_probe/trigger_select.py --base {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
