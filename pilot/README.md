# 第三步：pilot（294 张，4090 上约 40 分钟）

前两步已过：仪器标定见 `../calibrate/RESULTS.md`，别名封口见 `../survey_t2i_entailed_count.md` §7b。
这一步回答三个问题，**任何一个不过都要改设计，而不是往下走**。

| # | 问题 | 不过的话 |
|---|---|---|
| 1 | 模型对 `each` / `together` **有没有反应** | 若 dist 与 coll **连画面都一样**，结论退化成"模型忽略副词"，不是分配性 |
| 2 | **锚点**（`explicit_n` / `explicit_1`）能不能过 | 若模型连"三个气球"都画不出，dist 的失败是计数失败，跟我们无关 |
| 3 | 两个检测器在**生成图**上还一致吗 | 真实照片上 0.887；掉太多说明域偏移是真问题 |

外加四个**筛查族**（reciprocal / pair / respectively / part_whole）——
它们的真值可能不单值，**是拿来看、拿来砍的，不是拿来打分的**。

---

## 需要装什么

**大概率你已经有了。** 这台机器跑过 Wan2.2，`torch` / `diffusers` / `transformers`
都在，只要确认版本够新：

```bash
python -c "import torch,diffusers,transformers;print(torch.__version__,diffusers.__version__,transformers.__version__)"
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

缺了再装（国内源）：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    diffusers transformers accelerate safetensors pillow numpy
```

**不需要**：openai、opencv、vllm。**不需要下载 COCO / LVIS**——
那是第一步标定用的，这一步我们数的是自己生成的图。

## 需要下载什么权重

国内机器**必须先设镜像**，否则 huggingface.co 连不上：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

| 权重 | 大小 | 干什么 |
|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | 生成 |
| `facebook/mask2former-swin-large-coco-instance` | ~0.8 GB | 计数（COCO 80 类） |
| `google/owlv2-base-patch16-ensemble` | ~0.6 GB | 计数（开放词表，气球/泰迪熊归它） |

不用手动下，脚本首次运行时自动拉。想先拉好：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub
hf download stabilityai/stable-diffusion-xl-base-1.0
hf download facebook/mask2former-swin-large-coco-instance
hf download google/owlv2-base-patch16-ensemble
```

---

## 完整命令

```bash
cd pilot
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. 建刺激集（不需要 GPU，秒级）
python stimuli.py

# 2. 先看 8 条 prompt，确认没写错
python generate.py --dry-run

# 3. 生成（SDXL，24 GB 不用 offload；约 6 s/张 × 294 ≈ 30 分钟）
python generate.py --model sdxl

# 4. 计数：两个检测器都跑全部图
python count.py --images images/sdxl --detector mask2former
python count.py --images images/sdxl --detector owlv2

# 5. 报告
python analyze.py --counts images/sdxl/counts_mask2former.jsonl \
                           images/sdxl/counts_owlv2.jsonl

# 6. 联系表——一定要看，不要只看数字
python sheet.py --images images/sdxl \
                --counts images/sdxl/counts_mask2former.jsonl \
                         images/sdxl/counts_owlv2.jsonl
```

生成和计数都**断点续跑**，中断了重跑同一条命令即可。

先小跑一段试速度：`python generate.py --model sdxl --limit 10`。

### 显存不够 / 想换模型

```bash
python generate.py --model sdxl --offload          # SDXL 不需要，FLUX 需要
python generate.py --model sd35m                   # SD3.5-medium
python generate.py --model flux-schnell --offload  # FLUX.1-schnell，12B
```

---

## 目录长这样

```
pilot/
  stimuli.jsonl                 # 26 items / 294 images
  images/sdxl/
    manifest.jsonl              # 每张图连同产生它的 prompt
    item0000/
      dist__seed11.png  coll__seed11.png  bare__seed11.png
      explicit_n__seed11.png  explicit_1__seed11.png
    counts_mask2former.jsonl
    counts_owlv2.jsonl
    sheets/sheet_seed11_00.png
```

---

## 设计上必须保住的三条

**一、同一 item 的所有条件共用 seed。** 同种子 = 同初始噪声，
`dist` 与 `coll` 之间唯一的差异源就是那个副词。视频侧实测过是逐像素相同的
（max |Δpixel| = 0.000000）。**这一条丢了，条件间的差异就可能只是两次不同的采样。**

代码里每张图都新建一个 generator 再设种子，而不是复用一个——
复用会让状态在条件之间前进，共享噪声这条就没了。

**二、检测器看不见句子。** 它只被问"这张图里有几个 balloon"。
它在结构上无法知道这张图来自 `dist` 还是 `coll`，
所以条件间的差异不可能从仪器那侧漏进来。这比视频侧的 MLLM 判官更彻底——
那个至少还得被告知动词。

**三、断点续跑按 prompt 匹配，不按路径。** 重建套件会让 item 重新编号
（多加一个物体，后面所有 id 全平移），而磁盘上的图还带着旧编号。
只按坐标匹配会**悄悄把过期的图当成已完成**——视频侧就是这样把
"擀面皮"配上了一段擦胡萝卜的视频。

---

## 阈值是标定给的，别在这里调

`count.py` 里 `THRESHOLD = {"mask2former": 0.40, "owlv2": 0.20}`，
来自 LVIS 标定的**误差平衡点**，不是准确率最高点。理由：
把 1 个误判成"多个"污染的是集体条件，漏检污染的是分配条件——
两个方向落在不同的实验条件上，只追准确率会让偏差正好压在这个基准要做的那个比较上。

**在生成图上重调阈值 = 拿被测对象来调仪器。** 不要做。

---

## 看什么

`analyze.py` 出四节。最要紧的是第 1 节里那张**像素差表**：

```
pair                      n  mean |Δpixel|  identical
dist vs coll             54          31.42       0.00
```

- **均值接近 0 / identical 接近 1** → 模型压根没读那个副词。
  数量上没差异就成了平凡结论，得改 prompt 模板（换更强的分配标记，
  比如 "each ... of her own"）再试。
- **均值大 / identical 为 0** → 模型读了句子。这时候数量差异（或没有差异）
  才是关于分配性的结论。

第 2 节的锚点：`explicit_n` 若准确率很低，**先别下任何关于分配性的结论**——
说明这批 N 对这个模型就是数不出来，应该降 N 或换物体。

第 4 节的筛查族：**逐条看图**，判断"正确答案是不是唯一的"。
凡是你自己都要犹豫的族，直接砍掉。宁可只留三族，不留一族说不清的。

---

## 跑完把什么传回来

判读只需要这三样，都很小：

```bash
tar czf pilot_out.tgz \
    images/sdxl/counts_*.jsonl images/sdxl/manifest.jsonl \
    images/sdxl/sheets stimuli.jsonl
```

图本身（294 张 PNG，约 1 GB）不用传。
