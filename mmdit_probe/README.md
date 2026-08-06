# MMDiT 风格探针：一票否决实验

**这不是方法，是杀方向用的。** 目的只有一个：判断 MMDiT 上的风格迁移，
如果做出来，是**移植**还是**新机制**。跑完之前不推荐这个方向。

## 判据

| 换掉哪一路 K/V | 它是什么 | 结果的含义 |
|---|---|---|
| **image 流** (`to_k` / `to_v`) | StyleID 换 U-Net self-attention K,V 的直接对应物 | **移植 → 方向死** |
| **text 流** (`add_k_proj` / `add_v_proj`) | 注入风格分支**演化过的文本表示**。U-Net 里文本是冻结条件，**没有这个东西** | **新机制 → 可以往下走** |

已录用+开源的那条线全部依赖 U-Net 结构：StyleID (CVPR 2024)、Cross-Image
Attention (SIGGRAPH 2024)、Ctrl-X (NeurIPS 2024)、PnP (CVPR 2023)。
MMDiT 没有独立的 self-attention 层，也没有 decoder ResNet；而且**文本表示在每个
block 都被重写**。所以上面两路的差别不是实现细节，是"有没有论文"的差别。

## 怎么做的（以及为什么这样做）

**不重写 attention processor。** 手写 FLUX 的 RoPE / fused 投影 / single block
布局，正好是那种"数学写错了但图看起来很合理"的改法，这里出一张假图比不做还糟。

改成：两个分支放在**同一个 batch、同一份初始噪声**里生成，
在 K/V 的 `nn.Linear` 上挂 forward hook，把风格行拷到内容行。**attention 数学一行没动。**

batch 布局是每 2 个一组 `[style, content]`——有无 CFG 都成立
（diffusers 排成 `[neg_s, neg_c, pos_s, pos_c]`），所以 `out[1::2] = out[0::2]` 两种情况都对。
**这一点不靠假设，靠 `--selftest` 证明。**

FLUX 的 single block 是单流 `[text; image]` 拼接，按 token 下标切分处理——
否则三分之二的模型探不到。

## 跑

```bash
cd mmdit_probe
export HF_HUB_OFFLINE=1        # 见下

# 0. 环境 + 权重体检（几秒，不加载权重）
python preflight.py

# 1. 自检（~2 分钟）。不过不要往下走。
python run.py --selftest --model sd35

# 2. 主实验：6 组 (风格, 内容) × 2 seed × 4 条件
python run.py --model sd35                 # ~15 min，2.5B，24GB 宽裕
python analyze.py --model sd35

# 3. FLUX 确认（12B，必须 offload）
python run.py --selftest --model flux
python run.py --model flux
python analyze.py --model flux
```

显存不够就 `--sequential-offload` 或 `--size 768`。

### 环境 / 权重

**大概率不用重建环境。** 这里比跑 Wan2.2 的要求低：SD3.5 只要 `diffusers>=0.31`、
FLUX 要 `>=0.30`，而 Wan2.2 要 `>=0.36`。唯一可能缺的是 T5 分词器要的
`sentencepiece` + `protobuf`——缺了会在很后面才炸出一个看不懂的 tokenizer 错误，
所以 `preflight.py` 单独查这两个。

**权重路径已经写死在 `run.py` 的 `MODELS` 里**，`resolve_path()` 同时接受
普通模型目录和 HF hub 缓存目录（自动进 `snapshots/*` 找 `model_index.json`）。
路径不对就 `--path` 覆盖。

`preflight.py` 会逐个组件核对，重点抓两件事：

1. **下载不全**。SD3.5 的 `text_encoder_3` 是 T5-XXL，约 9GB，是最常缺的一个。
   缓存目录里是指向 `../../blobs` 的符号链接，**没下完的 blob 会留下悬空链接**，
   目录看起来存在但读不出来——单独查了这个。
2. **`HF_HUB_OFFLINE=1` 没设**。即使路径完全在本地，`from_pretrained` 仍会去
   huggingface.co 复核一次。这台服务器连不上外网，结果是**长时间挂起**而不是快速报错。
   设了就变成立刻、可读的错误。

### `--selftest` 在测什么

只有内容行该被覆盖。所以**换过之后，风格分支的图必须和没换时一样，内容分支必须变**。
如果下标写反了，这两个数会对调——而后面每一张 contact sheet 都会是一张
言之凿凿的废图。风格分支给 1/255 的容差（clone 改内存布局会换 attention kernel），
反向下标不可能只差 1 级，仍然抓得住。

三条 FAIL 各自对应一个具体病因：hook 没触发 / 下标反了 / 模块名不对（这个
diffusers build 用的不是 `to_k`）。

## 怎么读结果

**先看 contact sheet，再看表。两者冲突时以图为准。**
`analyze.py` 每个 seed 出一张 6 行 × 5 列的图：
`content 基线 | style 参考 | 换 image 流 | 换 text 流 | 两个都换`。

每组风格分支都带**自己的物体**（向日葵 / 海浪 / 茶壶 / 自行车 / 狮子 / 城堡）——
没有这个就分不清"风格过来了"和"风格图整张过来了"，而**内容泄漏**正是注意力交换类
风格迁移在实践中死掉的地方。图上标了每行该找哪个物体。

表里三个量（全部纯 numpy 算，不依赖任何下载的模型，因为服务器连不上外网）：

| | 含义 | 期望 |
|---|---|---|
| **S** | 到风格图的风格距离（纹理 Gram + 颜色直方图） | **降**=风格过来了 |
| **C** | 到内容基线的结构相似度（梯度图相关） | **保持高**=内容还在 |
| **L** | 到风格图的结构相似度 | **不升**=没泄漏 |

**三个必须同时成立才算数**：只看 S 的话，直接复制风格图得满分；只看 L 的话，
一团灰泥得满分。`C < 0.5` 的条件会被单独警告并要求折价。

结论分四种，其中"两路差不多"是**单独一档**，不是绿灯——
它只说明 2 路测不开（联合 softmax 把两流缠住了），那时候才轮到去写 4 路 processor 版本。

## 已知限制（先说清楚）

- 风格参考是**同一个模型按风格 prompt 生成**的，不是真实画作反演。对一个否决实验够用
  （问的是"风格住在哪一路"，不是"怎么做得最好"），但不能当成方法的评测。
- S / C / L 是像素级代理，不是 CLIP/VGG/CSD。**图是主证据，表是旁证。**
  如果后面要正经数字，在 Mac 上下 `openai/clip-vit-large-patch14` 传上来即可，
  但**否决判断不依赖它**。
- 结构相似度的 64px + 3×3 模糊不是随手定的：128px 不模糊时，给图加高频纹理
  （正是风格化在做的事）会把平滑图的自相似度打到 0，正确的风格化输出会被判成
  "内容毁了"。调到 64px+模糊后纹理控制从 0.975 升到 0.997，同时两个不同布局仍然
  分得开（~0.03）。低于 48px 就开始把无关布局判成相似。
- **本机没有 GPU 也没装 diffusers**，所以模块名（`to_k` / `add_k_proj`）没能对着真实
  build 验证过。`--selftest` 和 `attach()` 里的 `SystemExit`/`RuntimeError` 就是为这个
  准备的：名字不对会**报错**，不会静默给出错误结果。
