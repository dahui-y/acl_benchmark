"""开跑前检查。目标是把会在 20 分钟后才炸的东西，提前 20 秒查出来。

最要紧的一条是 fp16 变体。ScaleDiff 的代码是

    CustomStableDiffusionXLPipeline.from_pretrained(ckpt, torch_dtype=torch.float16)

注意它【没有】传 variant="fp16"。diffusers 因此会去找不带后缀的
`diffusion_pytorch_model.safetensors`（fp32 命名）。而我们缓存里的 SDXL 是
6.7 GB —— 完整 fp32 应该在 13 GB 以上，所以那多半是【只有 fp16 变体】的缓存。
真是这样的话，跑起来会在 from_pretrained 报 FileNotFoundError，
而且报错不会说"你该加 variant='fp16'"。

    python scalediff_probe/preflight.py
"""

import os
import sys
from pathlib import Path

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"
fails = []


def check(name, fn):
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = BAD, f"{type(e).__name__}: {e}"
    print(f"{status} {name:<26}{detail}")
    if status == BAD:
        fails.append(name)


def c_env():
    missing = [v for v in ("HF_HOME", "HF_HUB_OFFLINE", "SD_OUT") if not os.environ.get(v)]
    if missing:
        return BAD, f"未设置: {', '.join(missing)} —— 先 source scalediff_probe/env.sh"
    return OK, f"HF_HOME={os.environ['HF_HOME']}"


def c_torch():
    import torch
    if not torch.cuda.is_available():
        return BAD, f"torch {torch.__version__} 但 CUDA 不可用"
    p = torch.cuda.get_device_properties(0)
    return OK, f"torch {torch.__version__}  {p.name}  {p.total_memory / 2**30:.1f} GB"


def c_pkgs():
    import diffusers, numpy, einops, transformers
    want = {"diffusers": ("0.35.1", diffusers.__version__),
            "einops": ("0.8.1", einops.__version__),
            "numpy": ("2.", numpy.__version__)}
    off = [f"{k}: 要 {v[0]} 实际 {v[1]}" for k, v in want.items() if not v[1].startswith(v[0])]
    detail = (f"diffusers {diffusers.__version__}, transformers {transformers.__version__}, "
              f"numpy {numpy.__version__}")

    # transformers 5.x 会让 `from diffusers import StableDiffusionXLPipeline` 直接炸：
    # diffusers 0.35.1 从 transformers.utils 导入 FLAX_WEIGHTS_NAME，v5 删了它。
    # 报错栈指向 diffusers，真正的原因是 transformers 太新 —— 所以单独查一条。
    if int(transformers.__version__.split(".")[0]) >= 5:
        return BAD, (detail + "\n       transformers 必须 < 5（diffusers 0.35.1 要 "
                     "FLAX_WEIGHTS_NAME，v5 已删）。\n"
                     '       修：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple '
                     '"transformers<5"')
    return (WARN, detail + "  << " + "; ".join(off)) if off else (OK, detail)


def c_sdxl():
    """SDXL 在不在，以及【是否必须加 variant='fp16'】。"""
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
    if not root.is_dir():
        return BAD, f"缓存里没有 SDXL: {root}"
    snaps = list((root / "snapshots").glob("*"))
    if not snaps:
        return BAD, "有目录但没有 snapshots"
    snap = snaps[0]
    unet = snap / "unet"
    plain = list(unet.glob("diffusion_pytorch_model.safetensors"))
    fp16 = list(unet.glob("*.fp16.safetensors"))
    tot = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**30
    if plain:
        return OK, f"{tot:.1f} GB, unet 有 fp32 命名的权重 —— 原样跑即可"
    if fp16:
        return WARN, (f"{tot:.1f} GB, unet 【只有】 *.fp16.safetensors。\n"
                      f"       ScaleDiff 没传 variant='fp16'，原样跑会 FileNotFoundError。\n"
                      f"       run_one.py 已自动加 variant='fp16'，用它跑就行。")
    return BAD, f"{tot:.1f} GB 但 unet 下既无 fp32 也无 fp16 权重"


def c_code():
    p = Path("help_code/ScaleDiff/SDXL/pipeline_scalediff_sdxl.py")
    if not p.is_file():
        return BAD, f"找不到 {p} —— 请在仓库根目录运行"
    return OK, "help_code/ScaleDiff/SDXL/ 就位"


def c_space():
    out = Path(os.environ.get("SD_OUT", "."))
    st = os.statvfs(out)
    gb = st.f_bavail * st.f_frsize / 2**30
    # 一张 4096² png 约 25–40 MB，一轮 30 条 prompt × 3 分辨率 ≈ 2 GB
    return (OK if gb > 20 else WARN), f"{out} 剩余 {gb:.0f} GB"


print()
for n, f in [("环境变量", c_env), ("torch / GPU", c_torch), ("包版本", c_pkgs),
             ("SDXL 权重", c_sdxl), ("ScaleDiff 代码", c_code), ("输出盘空间", c_space)]:
    check(n, f)
print()
if fails:
    print(f"有 {len(fails)} 项失败：{', '.join(fails)}。先修这些，别急着跑。")
    sys.exit(1)
print("全部通过。下一步：python scalediff_probe/run_one.py")
