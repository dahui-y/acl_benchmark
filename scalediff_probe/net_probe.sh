#!/usr/bin/env bash
# 服务器（OpenBayes，中国境内）到底能拿到哪些评测资产 —— 逐项测，不猜。
#
# 背景：之前一次 CLIP 权重拉取失败，我就写下了"服务器拿不到评测数据、
# Mac 中转是关键路径"。这是过度推广。实测结论（2026-08-10）：
#   - **huggingface.co 直连可用**（真取到字节，不只是 HEAD 200）；
#   - github.com 直连可用；清华 pip 源可用；
#   - **hf-mirror 反而不可用**（SSL 握手超时）—— 仓库里
#     stylessp_probe/setup_env.sh:63 早就记着它已不再代理，
#     我却把它设成了默认，第一版这个脚本因此误报。
#   - env.sh 里的 HF_HUB_OFFLINE=1 是我们自己设的，不是网络的限制。
# 结论：Mac 中转这条路线删除。
#
#     bash scalediff_probe/net_probe.sh
#
# 输出每行 OK / FAIL。注意 HEAD 200 不等于下得来，真下载见 eval_assets.py。

set -u
export HF_HUB_OFFLINE=0
# 直连优先。mirror 只在下面单独列一行做对照，不作为默认端点。
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
PIP_IDX=${PIP_IDX:-https://pypi.tuna.tsinghua.edu.cn/simple}

ok()   { printf '  \033[32mOK  \033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

# $1 = 说明, $2 = URL。只发 HEAD，不下载正文。
head_ok() {
  code=$(curl -sIL -m 25 -o /dev/null -w '%{http_code}' "$2" 2>/dev/null)
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then ok "$1  ($code)"; return 0
  else bad "$1  (HTTP ${code:-timeout})  $2"; return 1; fi
}

echo
echo "== 1. 基础出口 =="
head_ok "huggingface.co 直连（主通道）"  "https://huggingface.co"
head_ok "清华 pypi 可达"                "$PIP_IDX"
head_ok "hf-mirror.com（备胎；实测握手超时，不要当默认）" "https://hf-mirror.com"
head_ok "github.com 直连（FID 权重的默认来源）"     "https://github.com"

echo
echo "== 2. FID/KID/IS 需要的 InceptionV3 =="
# pytorch-fid 的权重默认从 github release 拉；torchmetrics 走 torch hub 同一处。
head_ok "pytorch-fid pt_inception (github release)" \
  "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
# 备选：HF 上有同一份权重的镜像
head_ok "同一权重的 HF 镜像（备选）" \
  "$HF_ENDPOINT/nateraw/inception-v3-fid/resolve/main/pt_inception-2015-12-05-6726825d.pth"

echo
echo "== 3. CLIP-Score 用的文本/图像塔 =="
head_ok "open_clip ViT-bigG-14 laion2b (HF 镜像)" \
  "$HF_ENDPOINT/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin"
head_ok "openai/clip-vit-large-patch14（更小的备选）" \
  "$HF_ENDPOINT/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"

echo
echo "== 4. LAION prompt 列表（1000 条 caption）=="
# ScaleDiff / DemoFusion 的协议：随机采 LAION-5B caption。
# 全量 parquet 极大，我们只要 caption 列的一小片。
head_ok "laion2B-en-aesthetic 单个 parquet 分片" \
  "$HF_ENDPOINT/datasets/laion/laion2B-en-aesthetic/resolve/main/part-00000-ce9c1d54-8e7c-4e5c-a4c3-1d0e34ad0f6b-c000.snappy.parquet"
head_ok "datasets 仓库 API（列文件用）" \
  "$HF_ENDPOINT/api/datasets/laion/laion2B-en-aesthetic"

echo
echo "== 5. pip 侧（装包比拉权重容易，很多东西可以绕）=="
python -c "import open_clip" 2>/dev/null && ok "open_clip 已装（清华源装成功的证据）" \
  || bad "open_clip 未装"
pip download --no-deps -d /tmp/_np pytorch-fid -i "$PIP_IDX" -q 2>/dev/null \
  && ok "pip 能装 pytorch-fid" || bad "pip 装 pytorch-fid 失败"
pip download --no-deps -d /tmp/_np clean-fid -i "$PIP_IDX" -q 2>/dev/null \
  && ok "pip 能装 clean-fid" || bad "pip 装 clean-fid 失败"
rm -rf /tmp/_np

cat <<'EOF'

判读：
  §2 里任一行 OK   -> FID/KID/IS 的骨干拿得到。
  §3 任一行 OK     -> CLIP-Score 拿得到。
  §4 任一行 OK     -> prompt 列表拿得到（真图参考集另说，见下）。

真图参考集是唯一可能真需要中转的一项：FID 要和真实图片比，
LAION 的图要按 URL 逐张下载（境外 CDN）。两条不用中转的出路：
  a. 用 HF 上已经打包好的图片集（如 laion/relaion2B-en-research-safe 的
     img2dataset 产物、或 COCO-2014 val）—— 走 hf-mirror，同一条通道；
  b. FIDp/KIDp 用的是**我们自己生成的图之间**的 patch 分布，
     基线 arm 就是参考，本来就不需要外部真图。

只有 a、b 都不成立时才谈 Mac。
EOF
