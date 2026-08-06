# StyleSSP break test：内容细节是否被它自己的频率操纵吃掉

**这是找失效的探针，不是复现指标。** 产物是一个**有名字的失效 + 一组图**，不是一张表。

## 假设

StyleSSP 论文的核心论据：**z_T 的高频载布局**，所以要保住高频、只削低频（α=0.7）。
但发布代码用的是**带通**（`filter_type="gaussian_b"`），我们按发布参数算出的实际响应：

| 归一化频率半径 | z_T 上的有效乘子 |
|---|---|
| DC | **0.702** |
| 中频 0.5–0.7 | **0.898（衰减最少）** |
| 最高频 | **0.823** |

**它把自己论证"载布局"的那一段衰减了约 18–20%，论文没提。**

由此的可证伪假设（**升级版才是有价值的那个**）：

> 内容的细结构和风格的笔触**住在同一频段**。一个**全局的、逐频率的标量**
> 无法一边保住前者一边削掉后者——**频率不是分离内容与风格的正确坐标轴。**

## 跑

```bash
# 1. 分层：从图片自身的频谱算，不靠人工判断"哪张算细致"
python stratify.py --content DIR --style DIR --per-bucket 3

# 2. 用 pairs.jsonl 跑 StyleSSP 两遍：α=0.7（发布值）和 α=1.0（关掉频率操纵）
#    α 在 infer_style.py 的 freq_exp(...) 调用里

# 3. 读结果
python sheet.py --pairs pairs.jsonl --out-dir RESULTS --ablation RESULTS_alpha1
```

## 两处设计要点（都是冒烟测试逼出来的）

**① 判据必须是消融 delta，不能是跨格绝对值。** 第一版用
`loC×loS − hiC×hiS` 当读数，在注入了真实失效的合成数据上**判成了"无失效"**——
因为平滑内容**本来就没有细结构可保**，retention 基线接近 0，跨格比较测的是输入不是方法。
现在：主读数是 **α=0.7 vs α=1.0 的差**，且 α=1.0 基线 < 0.2 的格子直接丢弃
（没东西可丢的地方，"丢没丢"无定义）。**没有 `--ablation` 时脚本拒绝下任何结论。**

**② 用边际趋势，不用 argmax。** 九个格子每格几对，argmax 是噪声。
现在按内容 HF 和风格 HF 分别报边际损失——**预测是"两个方向都单调上升"**，
只有一个成立就说明不是频谱竞争，是别的东西。

## 结论有四档

| 结果 | 含义 |
|---|---|
| 任何格子的 delta < 0.10 | 频率操纵几乎不吃细节，**假设死** |
| delta 大但**各格均匀** | 是一笔统一的税，**不是频谱竞争**，最多算 errata |
| **两个边际趋势都为正** | 预测的模式，往下走 |
| 只有一个趋势为正 | 有东西，但不是这个假设，先诊断 |

---

## 第 0 步（占位核查）结果：**强版假设已被占，探针不跑**

2026-08-06 检索。

**[FreSCo: Joint Frequency-Aware and Spatial Control for Image Zero-Shot Style
Transfer](https://dl.acm.org/doi/10.1145/3805622.3810798)（ICMR 2026，training-free）**

它的问题陈述：*"a key oversight being the **neglect of frequency-domain
distinctions in visual signals**, which leads to issues like **content drift and
style leakage**"* —— **和我们的强版假设是同一句话。**

它的解法正是我们会提的那个：**Dynamic Wavelet Latent Fusion**——用 **DWT**
分解 latent（小波 = 空间局部化的频率，恰恰是"全局标量分不开"的标准答案），
**只把风格注入高频纹理子带**；外加 VAE-Compressed Masking 做空间控制。
而且**直接对比 StyleSSP**：结构保持上声称大幅胜出，风格化上与 StyleSSP 持平。

另有 **FFTDiff**（tuning-free）在频域解耦纹理/内容/颜色。

### 判定

| 版本 | 状态 |
|---|---|
| 弱版：StyleSSP 发布代码用带通、削了自己论证要保的最高频 | **仍是我们的发现，但它是 errata，不是论文** |
| **强版：频率不是分离内容与风格的正确坐标轴，需空间自适应** | **被 FreSCo 占**（问题陈述 + 解法 + 对 StyleSSP 的对比全中） |

**探针不跑，SDXL 环境不建。** 这正是第 0 步存在的意义——
在 26 GB 下载和 conda 环境之前，用一轮检索杀掉它。

**这是第四次"以为是空位、查了才知道有人"**（state change / MMDiT 文本流 /
调制空间 / 频域坐标轴）。四次里有三次是在花掉 GPU 或环境成本之前查到的。
流程有效——但**命中率本身是个信号**，见下。

---

## 跑 StyleSSP：环境与运行方案

目标是**看输出、找失效**，不是复现指标。顺序严格如下，每步都有停下来的理由。

### 第 0 步：查已有权重（先跑这个，别盲下 26 GB）

```bash
cd stylessp_probe
python check_assets.py                      # 扫默认路径
python check_assets.py --roots /openbayes/input /openbayes/home ~/.cache
```

它按 HF 缓存名和 README 的目录名两种形式找，报 `ok / PARTIAL / MISSING`
（体积不足预期 60% 判 PARTIAL，并单独报悬空符号链接），
然后**只为缺的那些**打印带 `--include` 的下载命令，以及可从服务器直连的
ModelScope 镜像。

> **不要照 README 跑 `huggingface-cli download h94/IP-Adapter`**——
> 那个仓库 40 GB+，真正要的单文件只有 700 MB。`check_assets.py` 打印的命令带过滤。

### 第 1 步：建独立环境

**必须独立**：StyleSSP 要 torch 2.3 / diffusers 0.30 / transformers 4.44 / py3.9，
当前 `aspect` 环境是 2.13 / 0.39 / 5.14。**就地降级会打死 `mmdit_probe`。**

```bash
conda env create -f ../help_code/StyleSSP/environment.yaml   # env 名 StyleSSP
conda activate StyleSSP
```

`pip install git+https://github.com/openai/CLIP.git` **跳过**——
`infer_style.py` 有 `import clip` 但全仓库无 `clip.` 调用。
用 `pip install openai-clip`，装不上就注释掉那一行。

### 第 2 步：改 `src/config.py` 的路径

`base_model_path` / `IP_path` / `tile_controlnet_path` / `canny_controlnet_path`
指向第 0 步确认的实际位置。**`style_image_dir` / `content_image_dir` 不用改**——
`run_batch.py` 从 pair 列表读，不走这两个字段。

### 第 3 步：分层输入 → 跑 → 看

```bash
python stratify.py --content DIR --style DIR --per-bucket 3
python run_batch.py --pairs pairs.jsonl --out results --limit 1   # 先跑一对
python run_batch.py --pairs pairs.jsonl --out results
python sheet.py --pairs pairs.jsonl --out-dir results
```

### `run_batch.py` 为什么不是"循环调用 infer_style.py"

他们的 `__main__` 是给**单对**写的：BLIP2 → instruct 管线 → 反演管线 →
ControlNet 管线，一对跑完就结束。用 subprocess 循环等于**每对重载 ~26 GB**，
30 对要多花一个多小时纯加载。

`run_batch.py` **import 他们的函数**（不复制、不改动 vendored 目录），
把日程重排成四段，换来两件东西：

1. **显存。** 原顺序里 BLIP2(~8GB) + instruct(~10GB) + 反演管线(~7GB)
   在反演期同时驻留 ≈ 25 GB，**24 GB 卡装不下**。
   这里先把所有 caption 出完并**释放 BLIP2**，反演全部做完再**释放反演管线**，
   最后才建 ControlNet 管线——两个最重的阶段**永不共存**。
2. **α 消融。** `--alpha 1.0` 关掉频率操纵、其他零件全不动，
   这是 `sheet.py` 归因所必需的内部对照。`d_s/d_t/filter_type` 全部照发布值，
   **只有 α 是我们动的**。

**未测试。** 本机无 GPU、无 diffusers，代码是照着读到的接口写的。
每段都打印加载了什么、峰值多少 GB。**先 `--limit 1`，读完输出再往下。**

### 看图的时候看什么

**不要先看指标。** 按 `sheet_content_hi.png` → `mid` → `lo` 的顺序看，
重点盯**细结构**：栏杆、电线、文字、发丝、密集树叶。
然后问 StyleSSP 自己那两个问题——**布局变了吗？风格图的东西漏进来了吗？**
以及第三个只有跑起来才看得见的：**还有什么它没命名的错？**

那个"没命名的错"，才是我们要找的东西。
