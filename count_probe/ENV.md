# 建环境（CountGen 线）

**要新建，不能沿用风格线那套。** make-it-count 钉的是 torch 2.1.2 / diffusers 0.25.0 /
transformers 4.29 系 / numpy 1.23.3，和 StyleID·StyleSSP 那套是两代人。

Python 选 **3.10**：`scikit-image 0.23.2` 要 `>=3.10`，`spacy 3.5.2` 和
`torch 2.1.2` 上限到 3.11，交集就是 3.10/3.11，取 3.10 更稳。

```bash
# ★ 必须带 --override-channels -c conda-forge，原因见下一节
conda create -n countgen python=3.10 -y --override-channels -c conda-forge
conda activate countgen

# PyPI 上 torch 2.1.2 的 linux 轮子默认就是 cu121 构建，走国内 PyPI 镜像即可，
# 不必绕 download.pytorch.org。装完确认：python -c "import torch;print(torch.version.cuda)" → 12.1
pip install torch==2.1.2 torchvision==0.16.2

pip install -r count_probe/requirements_infer.txt
```

### OpenBayes 上 `conda create` 报 `UnavailableInvalidChannel ... anaconda/pkgs/r`

不是我们的配置问题，是**镜像源整个下架了**。2024 年 Anaconda 改商业授权条款后，
国内镜像站陆续停掉了 `repo.anaconda.com` 官方 pkgs 的镜像，而 OpenBayes 预置的
`.condarc` 还把 `default_channels` 指在那儿。实测（2026-08）：

| 地址 | 状态 |
|---|---|
| `mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/repodata.json` | **404** |
| `mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/linux-64/repodata.json` | **404** ← 报错的就是它 |
| `mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json` | **200** |

`defaults` 没了，conda-forge 还在。所以要么像上面那样每次带
`--override-channels -c conda-forge`（不动全局配置，最省事），要么永久改：

```bash
conda config --show-sources          # 先看是哪个文件写的
conda config --remove-key default_channels
conda config --remove-key channels
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

### 不想碰 conda：venv 一样够

我们用 conda 只是为了拿一个解释器，其余全是 pip。

```bash
python -V     # 3.10 或 3.11 都行
python -m venv /openbayes/home/venv/countgen      # 放可写盘，别放容器临时目录
source /openbayes/home/venv/countgen/bin/activate
pip install -U pip
pip install torch==2.1.2 torchvision==0.16.2
pip install -r count_probe/requirements_infer.txt
```

3.11 也可以：`numpy==1.23.3` 的 linux wheel 覆盖 cp38/39/310/311（查过 PyPI），
不是非 3.10 不可。

## 三样要单独弄的东西

### 1. spacy 的 en_core_web_trf（≈460MB，GitHub release，国内常被卡）

```bash
python -m spacy download en_core_web_trf     # 先直接试
# 不通就手动取这个 wheel，传到服务器再 pip install 本地文件：
# https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/en_core_web_trf-3.5.0-py3-none-any.whl
```

> 退而求其次可以用 `en_core_web_sm`（12MB），但**不能直接换**：
> `self_counting_sdxl_pipeline.py:357` 取的 `object_token_idx` 依赖依存分析结果，
> 换模型可能换 token 下标，就不是复现了。真要换，先在 200 条 prompt 上
> 逐条比对两个模型给出的下标是否完全一致，一致才算数。

### 2. ReLayout 权重（Google Drive）

```
help_code/make-it-count/pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth
```

链接在 make-it-count 的 README 里。国内取不到就得换机器下载再传。
**没有它，只有"少了要补物体"那条分支跑不了**——`relayout_overgeneration`（多了删）
和 `obj_num_match`（直接输出原版图）两条都不需要权重。所以真拿不到，
仍可先跑出 vanilla 臂和一部分拆解，但那不是完整复现，报数时必须写明。

### 3. ReLayout U-Net 的骨架来自 torch.hub

`relayout.py:10` 每张图都会调

```python
torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', ...)
```

要连 GitHub。`countgen_batch.py` 已经把它 memoize 成只调一次，并在失败时
提示怎么用本地缓存；预热办法是在能联网的机器上跑一次同样的调用，
再把 `~/.cache/torch/hub/` 拷过来（或设 `TORCH_HOME` 指过去）。

### 4. 模型权重

```bash
export SDXL_PATH=/openbayes/.../stable-diffusion-xl-base-1.0    # fp16 分支，约 7GB
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
export SD_OUT=/openbayes/input/input0/Sim2Struct-1000/temp/scalediff_out
```

若本地 SDXL 目录没有 `fp16` variant，跑批时传 `--variant ""`。

## 装完先自检

```bash
python count_probe/env_check.py
```

它只做静态检查（版本、约束、四样资产在不在），不占 GPU、不生成图。
过了再去跑 `countgen_batch.py --limit 5`。

## 已知的坑

- `huggingface-hub==0.20.1` 是他们 requirements 里的原值。若报
  `cannot import name 'cached_download'`，降到 `0.19.4`。
- `opencv-python-headless` 而不是 `opencv-python`：服务器无显示，装全版会拖
  一堆 GUI 依赖；代码只用 `cv2.findContours / dilate / imread` 这类纯计算 API。
- `matplotlib` 必须走 Agg。`countgen_batch.py` 里已经 `MPLBACKEND=Agg` 兜底了
  （他们的 `db_scan` 每次聚类都画一张图）。
- 评测（ultralytics）和跑批可以同一个环境——ultralytics 8.2.12 的约束很松，
  torch 2.1.2 / numpy 1.23.3 都在范围内。不想冒险的话单独开一个 `yolo` 环境
  也行，两步之间只通过磁盘上的 PNG 交接，没有别的耦合。
