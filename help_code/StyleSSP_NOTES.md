# StyleSSP 代码分析 + 跑起来的方案

读的是 `help_code/StyleSSP`（bytedance/StyleSSP）。目的：**第 1 步——跑出它的输出，
找它还有什么可见的错**。不是复现指标，是找失效。

---

## 1. 代码实际在做什么（与论文逐条核对）

`infer_style.py` 主流程：

```
BLIP2 生成 style/content 的 caption
  → IP-Adapter-Instruct 抽 style/content 的解耦 embedding（4 个）
  → ReNoise DDIM 反演内容图（50 步，num_renoise_steps=1）
     · 反演时用 NPI 负引导（inv_guidance_scale=1.5）
  → freq_exp() 频率操纵起点
  → SDXL + 双 ControlNet + IP-Adapter 采样（50 步）
     · latents=latent_l（低频削弱后的起点）
```

### 三处代码与论文不一致（**这是找失效的线索，不是 bug**）

| | 论文写的 | 代码实际 |
|---|---|---|
| 低通滤波器 | "Gaussian filter with variance σ = 0.3" | `filter_type="gaussian_b"` = **带通**（`high_pass_mask + low_pass_mask`），`d_s=0.3, d_t=0.9`。该函数自己的注释：*"Consider that the highest part of image is noise. Filter it as well as filter the low-frequency components"* —— **最高频也被压了**，论文没提 |
| 采样调度器 | DDIM，50 步 | `UniPCMultistepScheduler`。DDIM 那行被注释掉了，**而且注释写着 `# works the best`** |
| 内容约束 | "pre-trained SDXL and tile ControlNet" | `control_type = "tile_canny"` → **同时加载 tile(0.25) + MistoLine canny(0.40) 两个 ControlNet**。canny 那个论文的 implementation details 里没有 |

另外：IP-Adapter 注入是 InstantStyle 式的块选择，`{"up": {"block_0": [0.0, 2.5, 0.0]}}`，
scale 2.5。`src/config.py` 的默认是 SDXL_Turbo/4 步，但 `__main__` 覆盖成 SDXL/50 步——
以 `__main__` 为准。

**为什么这三条重要**：它们说明"training-free"背后挂着一串**没写进论文的现成训练好的组件**
（第二个 ControlNet、UniPC、带通而非低通）。失效搜索的"目标间权衡"轴可以直接问：
**去掉 canny ControlNet 之后，它宣称修好的 content preservation 还剩多少？**

---

## 2. 需要的权重（8 项，约 26 GB）

| # | 仓库 | 用途 | 约 |
|---|---|---|---|
| 1 | `stabilityai/stable-diffusion-xl-base-1.0` (fp16 variant) | 基座，**三处加载** | 6.9 GB |
| 2 | `Salesforce/blip2-flan-t5-xl` | 自动 caption | 7.9 GB |
| 3 | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | IP-Adapter 图像编码器 | 3.9 GB |
| 4 | `madebyollin/sdxl-vae-fp16-fix` | VAE | 0.2 GB |
| 5 | `h94/IP-Adapter` → **只要** `sdxl_models/ip-adapter_sdxl_vit-h.safetensors` | 风格注入 | 0.7 GB |
| 6 | `CiaraRowles/IP-Adapter-Instruct` → `ip-adapter-instruct-sdxl.bin` | 解耦 embedding | 1.2 GB |
| 7 | `xinsir/controlnet-tile-sdxl-1.0` | tile 约束 | 2.5 GB |
| 8 | `TheMistoAI/MistoLine`（需 `variant="fp16"`） | canny 约束 | 2.5 GB |

**坑**：README 让你 `huggingface-cli download h94/IP-Adapter --local-dir ...`，
那个仓库整包 **40 GB+**（含 SD15/SDXL 全套 + 多个 image encoder）。**必须加 `--include`**：

```bash
huggingface-cli download h94/IP-Adapter \
  --include "sdxl_models/ip-adapter_sdxl_vit-h.safetensors" \
  --local-dir checkpoints/IP-Adapter
```

**网络**：服务器在国内连不上 HF。三条路，按优先级：
1. 先在服务器上**查一遍已有权重**（你已有 FLUX/SD3.5，SDXL 可能也在）；
2. ModelScope 有 1/3/4/7 的镜像；
3. 剩下的（BLIP2、两个 IP-Adapter、MistoLine）从 Mac 下了传上去。

---

## 3. 环境：**必须新建，且必须独立**

| | 当前 `aspect` 环境 | StyleSSP 要求 |
|---|---|---|
| python | 3.x（未查） | **3.9** |
| torch | 2.13.0+cu130 | **2.3.0** |
| diffusers | 0.39.0 | **0.30.0** |
| transformers | 5.14.1 | **4.44.0** |

**完全不兼容**，且降级会打死 `mmdit_probe` 那套。所以：

```bash
conda env create -f help_code/StyleSSP/environment.yaml   # 名字就是 StyleSSP
conda activate StyleSSP
```

国内装 torch 建议先配镜像源。torch 2.3.0 是 cu121 轮子，在 cu130 驱动上**向后兼容能跑**。

**`pip install git+https://github.com/openai/CLIP.git` 这步可以跳过。**
`infer_style.py:35` 有 `import clip`，但全仓库**没有任何 `clip.` 调用**
（`CSD_Score` 那两个路径也只在 config 里定义、从未被引用）。国内装 GitHub 包很痛，
所以：`pip install openai-clip`（PyPI 上有，镜像可达），装不上就把那一行注释掉。

---

## 4. 显存：**24 GB 大概率不够，需要一处改动**

论文跑在 **A100**。按加载顺序估峰值：

```
BLIP2 fp16                        ~8 GB
+ ip_instruct_model（SDXL+IPA-I+CLIP-H）  ~10 GB
+ get_pipes 的 SDXL（inv/inf 共用 components，只算一次） ~7 GB
                                   ≈ 25 GB  →  4090 OOM
```

`del pipe_inversion, pipe_inference, model` 在**反演之后**才执行（第 276 行），
所以三者在反演期间同时驻留。

**最小改动**：BLIP2 只用来出两句 caption，把它挪到最前面、出完 caption 立刻释放，
再 `init_models`。峰值降 8 GB。改动只是**重排**，不动方法本身：

```python
# 现在的顺序（第 209–239 行）
load BLIP2 → init_models → load images → captions → embeddings
# 改成
load images → load BLIP2 → captions → del BLIP2 → init_models → embeddings
```

第二处峰值在采样阶段（ip_instruct 10 GB + ControlNet 管线 ~14 GB ≈ 24 GB），
备用手段按优先级：`pipe_inference.enable_model_cpu_offload()` →
`control_type="tile"`（少一个 ControlNet，但**偏离了发布默认值，要单独记录**）→
分辨率降到 768（**偏离论文，不推荐，会污染失效判断**）。

---

## 5. 仓库里缺的东西

- `data/`、`data_evl/`、`checkpoints/`、`CSD_Score/` 都不在（`.gitignore` 只挡了
  `checkpoints/` 和 `results/`，其余是上游就没有）。
- README 说 800 张评测集（40 风格 × 20 内容）在 `./data`，**实际没有**。
  第 1 步不需要 800 张；先用 20–30 对看失效即可。评测集等到要出数字时再解决。
- `src/config.py` 里 `style_image_dir` / `content_image_dir` 是占位字符串，
  **必须改**，否则一跑就 FileNotFoundError。

---

## 6. 第 1 步怎么跑（找失效，不是复现指标）

1. **先单对跑通**，确认 OOM 与否、单对耗时。
2. **再跑一组有结构的输入**——不是随机 20 对。按可能暴露失效的维度分层：
   - 风格家族：厚涂 / 平涂版画 / 素描 / 彩玻 / 像素 / 低频色场
   - 内容类型：单主体 / 多主体 / 有文字 / 强透视 / 细密纹理
   - **风格图与内容图语义重叠**的对（负引导最可能误伤的情况）
3. 出 contact sheet：`content | style | StyleSSP 输出`，**逐张看**。
4. 给看到的错**命名**，做成像它 Fig.1 那样的对照图。

**这一步的产物不是数字，是一个有名字的失效 + 一组图。** 没有它，后面都不该动。
