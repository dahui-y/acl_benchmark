# 每次开跑前 source。忘了设 HF_HOME，from_pretrained 会去 ~/.cache（那个盘没空间），
# 要么静默重下 7 GB，要么在离线模式下挂住而不是快速报错。
#
#   conda activate scalediff && source scalediff_probe/env.sh

TEMP=/openbayes/input/input0/Sim2Struct-1000/temp

export HF_HOME=$TEMP/weights/hf          # 已有的 SDXL 缓存就在 $HF_HOME/hub 下
# 跑实验时钉成离线：避免 from_pretrained 静默去网上摸一遍，也让缺权重时
# 快速报错而不是挂住。**这不代表下不了东西** —— hf-mirror.com 在国内是通的
# （SDXL 就是这么来的），要下新权重用 scalediff_probe/fetch_weights.sh，
# 它会临时把这个变量置 0。
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export SD_OUT=$TEMP/scalediff_out        # 结果也写到唯一有空间的盘
export FLUX_PATH=$TEMP/weights/FLUX.1-dev

mkdir -p "$HF_HOME/hub" "$SD_OUT"

if ! touch "$SD_OUT/.wtest" 2>/dev/null; then
    echo "!! $SD_OUT 不可写"
else
    rm -f "$SD_OUT/.wtest"
    free_gb=$(df -BG --output=avail "$SD_OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "SD_OUT       = $SD_OUT   (可写, 剩余 ${free_gb:-?} GB)"
fi
echo "HF_HOME      = $HF_HOME"
echo "HF_HUB_OFFLINE = $HF_HUB_OFFLINE"
echo "FLUX_PATH    = $FLUX_PATH  $([ -d "$FLUX_PATH" ] && echo '(存在)' || echo '(!! 不存在)')"
echo
echo "缓存里的仓库："
ls -1 "$HF_HOME/hub" 2>/dev/null | grep '^models--' | sed 's/^/  /' || echo "  (空)"
