# 建环境（CountGen 线）

**要新建，不能沿用风格线那套。** make-it-count 钉的是 torch 2.1.2 / diffusers 0.25.0 /
transformers 4.29 系 / numpy 1.23.3，和 StyleID·StyleSSP 那套是两代人。

Python 选 **3.10**：`scikit-image 0.23.2` 要 `>=3.10`，`spacy 3.5.2` 和
`torch 2.1.2` 上限到 3.11，交集就是 3.10/3.11，取 3.10 更稳。

```bash
# ★ 必须带 --override-channels -c conda-forge，原因见下一节
conda create -n countgen python=3.10 -y --override-channels -c conda-forge
conda activate countgen

# ★ 先把 pip 指到国内镜像，然后**不要**加 --index-url download.pytorch.org（见下）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# ★ numpy 和 Pillow 必须和 torch 一起钉住，理由见下面那节
pip install torch==2.1.2 torchvision==0.16.2 numpy==1.23.3 Pillow==10.1.0
python -c "import torch,numpy;print(torch.__version__, torch.version.cuda, numpy.__version__)"
# 期望 2.1.2+cu121 12.1 1.23.3

pip install -r count_probe/requirements_infer.txt
```

### 单独装 torch 会顺手拉来 numpy 2.x，然后 torch 的 numpy 桥就废了

症状：`import torch` 打出

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 ...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

torch 版本号照样正确（`2.1.2+cu121 12.1`），但 `torch.from_numpy` / `.numpy()`
这条通路是坏的 —— 而它在整条链上到处都是。

成因是**安装顺序**：torch 2.1.2 自己没声明 numpy 依赖，torchvision 声明了
但没写上界，所以 pip 装了最新的 2.x；而 torch 2.1.2 的 C 扩展是按 NumPy 1.x
的 ABI 编译的。所以上面那条命令把 `numpy==1.23.3` 和 torch 一起装。
已经装坏了也不用重建环境，补一句就行：

```bash
pip install "numpy==1.23.3" "Pillow==10.1.0"
```

Pillow 一并钉住是同理：默认会装到 12.x，而 make-it-count 钉的是 10.1.0，
torchvision 0.16.2 是同年代的东西，不值得在这上面赌。

### 装 torch 千万别加 `--index-url https://download.pytorch.org/whl/cu121`

那个源在美国、没有国内镜像，实测只有 ~400 kB/s（2.2 GB 要一个半小时）；
而且它发的是**胖轮子**——把 CUDA 运行时静态打进 whl 里，所以才 2.2 GB。

走普通 PyPI（国内镜像）就行：`torch 2.1.2` 的 cp310 linux 轮子只有 **639 MB**，
它的依赖里带

```
nvidia-cuda-nvrtc-cu12==12.1.105   nvidia-cudnn-cu12==8.9.2.26
nvidia-cublas-cu12==12.1.3.1       ...共 11 个
```

**它本来就是 cu121 构建**，只是把 CUDA 运行时拆成独立 pip 包，而这些包国内镜像都有。

镜像备选（实测都通）：

| 镜像 | index-url |
|---|---|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple/` |
| 中科大 | `https://mirrors.ustc.edu.cn/pypi/simple` |
| 南大 | `https://mirror.nju.edu.cn/pypi/web/simple` |

torch 加那 11 个 nvidia 包解包后约 5 GB。先 `df -h /openbayes/home` 看配额；
不够就把 pip 缓存挪走：`export PIP_CACHE_DIR=$SD_OUT/pipcache`。

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

**直接装 wheel，别用 `python -m spacy download`：**

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/en_core_web_trf-3.5.0-py3-none-any.whl
```

这个 URL 就是 make-it-count 的 `requirements.txt` 里钉的那一个（连 sha256 都是），
版本一定对。取不到就在别的机器下好传过去，`pip install` 本地文件。

> **为什么不走 `python -m spacy download`**：spacy 3.5.2 依赖 `typer 0.7.0`，
> 而 typer 0.7.0 只写了 `click>=7.1.1,<9.0.0`，pip 会装最新的 click 8.4.x ——
> 差三年多，click 8.2 之后的参数解析改动 typer 0.7 没适配。症状是 spacy 拼出
> ```
> .../download/-en_core_web_trf/-en_core_web_trf.tar.gz#egg===en_core_web_trf
> ```
> 模型名成了空串。`requirements_infer.txt` 已经把 `click==8.1.7` 钉住了；
> 若是先前装的环境，补一句 `pip install "click==8.1.7"`。
> 这条不只影响下模型 —— spacy 3.5.2 的 `__init__.py` 里有
> `from .cli.info import info`，**`import spacy` 就会拉起 typer/click**。

> **别用 hf-mirror 上的 `spacy/en_core_web_trf`。** 我查过：主分支是 **3.7.3**，
> `meta.json` 写 `spacy_version >=3.7.2,<3.8.0` 且依赖 `spacy-curated-transformers`，
> 和我们钉的 spacy 3.5.2 + spacy-transformers 1.2.5 不兼容，仓库里也没有 3.5 的 tag。
> 这是条看起来能走、实际会把环境搞乱的岔路。

> 退而求其次可以用 `en_core_web_sm`（12MB），但**不能直接换**：
> `self_counting_sdxl_pipeline.py:357` 取的 `object_token_idx` 依赖依存分析结果，
> 换模型可能换 token 下标，就不是复现了。真要换，先在 200 条 prompt 上
> 逐条比对两个模型给出的下标是否完全一致，一致才算数。

### 2. ReLayout 权重（Google Drive）

**不要放进仓库**——GB 级文件，而且仓库所在盘不一定有空间。放到可写盘，
用 `RELAYOUT_CKPT` 指过去：

```bash
export CK=/openbayes/input/input0/Sim2Struct-1000/temp/weights/relayout_checkpoint.pth
mkdir -p "$(dirname "$CK")"
pip install gdown
gdown 1xyfkwmX9plMB5-c0VDwl7WiPuQ2qt5yb -O "$CK"    # 若下下来是压缩包，解开再放
export RELAYOUT_CKPT="$CK"                           # env_check 和跑批都认这个
```

也可以 `python count_probe/countgen_batch.py --relayout-ckpt "$CK" ...`。
脚本会把它写成绝对路径塞回 config——**必须绝对路径**，因为跑批时已经
`chdir` 到了 make-it-count，相对路径会解到仓库里去。

HF 上没有任何人镜像过这个权重（`make-it-count` / `countgen` / `relayout` 都搜过），
所以国内取不到就只能换机器下载再传。

**没有它也能先开跑**：它只在 `relayout_undergeneration`（数少了要补物体）里用。
加 `--vanilla-only` 可以先把不依赖它的三个读数拿到手——
vanilla SDXL 的 baseline、DBSCAN 计数器与 YOLO 的一致率、
以及「计数器说对但实际错」那一桶有多大（那是天花板损失）。
只有"修正成功率"要等权重。而且那一档是一次前向无梯度，快三四倍。

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
