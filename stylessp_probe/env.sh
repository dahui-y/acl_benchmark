# 每次开跑前 `source env.sh`。不是可选项 —— 忘了设 HF_HOME，from_pretrained
# 会去找 ~/.cache（那个盘没空间），要么静默重下 26 GB，要么在离线模式下挂住。
#
#   conda activate StyleSSP && source env.sh
#
# 为什么指到这里：服务器上唯一可写且有空间的路径是 .../temp，而已有的 SDXL
# 缓存正好已经在 .../temp/weights/hf/hub/ 下，所以把 HF_HOME 设成它的父目录，
# 新下载就落在旧的旁边，不用软链、不用搬 6.7 GB。
export HF_HOME=/openbayes/input/input0/Sim2Struct-1000/temp/weights/hf

# 离线。服务器连不上 huggingface.co，不设这个的话 from_pretrained 即使路径
# 全在本地也会去复核一次，表现为长时间挂起而不是快速报错。
export HF_HUB_OFFLINE=1

# 生成结果也写到同一个盘 —— 仓库所在盘没空间。
export SSP_OUT=/openbayes/input/input0/Sim2Struct-1000/temp/stylessp_out

mkdir -p "$HF_HOME/hub" "$SSP_OUT"

# 自检：可写 + 空间够。剩下 7 项约 19 GB，加上生成结果，20 GB 是下限。
if ! touch "$HF_HOME/hub/.wtest" 2>/dev/null; then
    echo "!! $HF_HOME/hub 不可写 —— 换一个 HF_HOME"
else
    rm -f "$HF_HOME/hub/.wtest"
    free_gb=$(df -BG --output=avail "$HF_HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "HF_HOME     = $HF_HOME   (可写, 剩余 ${free_gb:-?} GB)"
    if [ -n "$free_gb" ] && [ "$free_gb" -lt 20 ]; then
        echo "!! 剩余空间不足 20 GB —— 还要下 ~19 GB 权重，先清一下"
    fi
fi
echo "HF_HUB_OFFLINE = $HF_HUB_OFFLINE"
echo "SSP_OUT     = $SSP_OUT"
echo
echo "已在缓存里的仓库："
ls -1 "$HF_HOME/hub" 2>/dev/null | grep '^models--' | sed 's/^/  /' || echo "  (空)"
