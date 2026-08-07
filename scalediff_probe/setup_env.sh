#!/usr/bin/env bash
# ScaleDiff 环境。在服务器上执行：bash setup_env.sh
#
# 为什么不用 README 的 `pip install -r requirements.txt`：
#   1. 服务器连不上 pypi.org，必须走清华源；
#   2. requirements.txt 只列了 4 个包，但 diffusers 加载 SDXL 需要
#      transformers（两个 CLIP text encoder）、accelerate、safetensors，
#      FLUX 还要 sentencepiece/protobuf（T5）。照原样装会在 from_pretrained
#      时报 ImportError，而且报错信息不会直说缺哪个。
set -e

ENV=scalediff
PY=3.13                     # torch 2.6.0 有 cp313 轮子；3.11/3.12 也可以
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 1. 建 conda 环境 =="
conda create -n "$ENV" python=$PY -y

echo
echo "== 2. 激活并装包（清华源）=="
echo "下面的命令需要在【激活后】跑。如果 conda activate 在脚本里失效，"
echo "请手动执行第 2 步的内容。"
eval "$(conda shell.bash hook)"
conda activate "$ENV"

python -m pip install --upgrade pip -i $MIRROR

# 与 requirements.txt 完全一致的四个
pip install -i $MIRROR \
    torch==2.6.0 \
    diffusers==0.35.1 \
    einops==0.8.1 \
    numpy==2.3.4

# requirements.txt 漏掉但实际必需的
#
# transformers 必须钉在 4.x：diffusers 0.35.1 的 pipeline_loading_utils.py 里有
#     from transformers.utils import FLAX_WEIGHTS_NAME
# 而这个常量在 transformers v5 被删了。不钉版本 pip 会装 5.x，然后在
#     from diffusers import StableDiffusionXLPipeline
# 报 ImportError: cannot import name 'FLAX_WEIGHTS_NAME'
# —— 报错指向 diffusers，真正的原因却是 transformers 太新。第一次装就踩了。
#
# torchvision 0.21.0 是配 torch 2.6.0 的版本。不装的话 transformers 会退回
# PIL 后端并每次打印两条警告，不致命但吵。
pip install -i $MIRROR \
    "transformers<5" torchvision==0.21.0 \
    accelerate safetensors \
    sentencepiece protobuf \
    pillow

echo
echo "== 3. 自检 =="
python - <<'PY'
import torch, diffusers, numpy, einops, transformers
print(f"torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          {p.name}  {p.total_memory/2**30:.1f} GB")
print(f"diffusers    {diffusers.__version__}")
print(f"transformers {transformers.__version__}")
print(f"numpy        {numpy.__version__}")
print(f"einops       {einops.__version__}")
assert numpy.__version__.startswith("2."), "ScaleDiff 要 numpy 2.x"
PY

echo
echo "完成。下一步："
echo "  conda activate $ENV"
echo "  source scalediff_probe/env.sh"
echo "  python scalediff_probe/preflight.py"
