#!/usr/bin/env bash
# StyleSSP 环境。独立于 aspect 环境 —— 不要在 aspect 里降级 torch/diffusers，
# 那会打死 mmdit_probe。
#
#   bash setup_env.sh
#
# 不用 conda env create -f environment.yaml 的理由：那个 yaml 把 defaults 频道的
# 系统库逐个钉死（openssl/ncurses/tk 那些），在国内解算慢且常因频道 ToS 卡住。
# 依赖真正的约束在 requirements.txt 里，走 pip 更稳。失败时的退路见文件末尾。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../help_code/StyleSSP" && pwd)"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "==> 建 py3.9 环境"
conda create -y -n StyleSSP python=3.9
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate StyleSSP

python -m pip install -U pip -i "$PIP_MIRROR"

# torch 2.3.0 的 linux 默认 PyPI 轮子就是 cu121 构建（requirements.txt 里那一串
# nvidia-*-cu12 就是它的依赖），所以不需要 download.pytorch.org 那个 index，
# 直接走国内 PyPI 镜像即可。cu121 轮子在 cu13x 驱动上向后兼容。
echo "==> torch 2.3.0 (cu121)"
pip install -i "$PIP_MIRROR" torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

echo "==> StyleSSP 依赖"
pip install -i "$PIP_MIRROR" -r "$REPO/requirements.txt"

# requirements.txt 漏了这三个：
#  - sentencepiece/protobuf: BLIP2 的 T5 分词器要，缺了会在很后面炸出
#    一个看不懂的 tokenizer 错误
#  - openai-clip: infer_style.py 第 35 行 import clip，但全仓库没有 clip.* 调用。
#    装它只是为了让 import 过去；装不上就把那行注释掉，功能无影响。
echo "==> 补漏"
pip install -i "$PIP_MIRROR" sentencepiece protobuf openai-clip

echo "==> 自检"
python - <<'PY'
import torch, diffusers, transformers, numpy
print(f"torch        {torch.__version__}   cuda={torch.cuda.is_available()}")
print(f"diffusers    {diffusers.__version__}   (需 0.30.x)")
print(f"transformers {transformers.__version__}   (需 4.44.x)")
print(f"numpy        {numpy.__version__}   (必须 <2，否则 torch 2.3 会炸)")
assert numpy.__version__.startswith("1."), "numpy 被升到 2.x 了，pip install 'numpy<2'"
for m in ("sentencepiece", "google.protobuf", "pyrallis", "controlnet_aux"):
    __import__(m); print(f"  ok  {m}")
try:
    import clip; print("  ok  clip")
except ImportError:
    print("  !!  clip 缺失 —— 把 infer_style.py 第 35 行 'import clip' 注释掉即可")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          {p.name}  {p.total_memory/2**30:.1f} GB")
PY

cat <<'EOF'

==> 环境好了。conda activate StyleSSP

下载权重时注意：hf-mirror.com 已经不再代理（会 308 跳回 huggingface.co，
本会话早前核实过），所以 HF_ENDPOINT 那招无效。走 ModelScope 或从 Mac 传。
运行时务必 export HF_HUB_OFFLINE=1，否则 from_pretrained 会去连 HF 而挂住。

下一步：
  python check_assets.py --roots /openbayes/input /openbayes/home ~/.cache

退路（上面失败时）：
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  conda env create -f ../help_code/StyleSSP/environment.yaml
EOF
