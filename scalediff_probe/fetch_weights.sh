#!/usr/bin/env bash
# 在服务器上下权重。**能下** —— 挡住的是我们自己在 env.sh 里设的
# HF_HUB_OFFLINE=1，不是网络：hf-mirror.com 在国内是通的，SDXL 就是这么来的。
# 那行注释写的"服务器连不上 huggingface.co"对官方域名成立，对镜像不成立。
#
# HF_HOME 已指向 temp/weights/hf，下载自动落到唯一有空间的那个盘。
#
#   conda activate scalediff && source scalediff_probe/env.sh
#   bash scalediff_probe/fetch_weights.sh clip
#   bash scalediff_probe/fetch_weights.sh clip-big      # 正式表格用的大塔
#   bash scalediff_probe/fetch_weights.sh dino          # 计数器（多半已有）

set -u
export HF_HUB_OFFLINE=0
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
: "${HF_HOME:?先 source scalediff_probe/env.sh}"

echo "HF_HOME     = $HF_HOME"
echo "HF_ENDPOINT = $HF_ENDPOINT"
echo

fetch_clip () {   # $1 = repo id
    python - "$1" <<'PY'
import sys
from transformers import CLIPModel, CLIPProcessor
m = sys.argv[1]
CLIPModel.from_pretrained(m)
CLIPProcessor.from_pretrained(m)
print(f"OK  {m}")
PY
}

case "${1:-clip}" in
  clip)
    # ~600 MB。CLIP score 的最小可用塔。两个 arm 用同一个模型，
    # 相对比较就成立；正式表格要对齐 ScaleDiff 时再换大塔，判据不变。
    fetch_clip openai/clip-vit-base-patch32 ;;
  clip-l)
    fetch_clip openai/clip-vit-large-patch14 ;;
  clip-big)
    # ~10 GB。ScaleDiff / AccDiffusion v2 报 CLIP score 用的量级
    fetch_clip laion/CLIP-ViT-bigG-14-laion2B-39B-b160k ;;
  dino)
    python - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
m = "IDEA-Research/grounding-dino-base"
AutoProcessor.from_pretrained(m)
AutoModelForZeroShotObjectDetection.from_pretrained(m)
print(f"OK  {m}")
PY
    ;;
  *)
    echo "用法: bash scalediff_probe/fetch_weights.sh [clip|clip-l|clip-big|dino]"
    exit 1 ;;
esac

echo
echo "缓存里现有的仓库："
ls -1 "$HF_HOME/hub" | grep '^models--' | sed 's/^/  /'
