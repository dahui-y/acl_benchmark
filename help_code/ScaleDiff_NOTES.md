# ScaleDiff（NeurIPS 2025）代码通读

日期 2026-08-07。源：用户推送的 `help_code/ScaleDiff`（2105 行，SDXL 588+205，FLUX 379+850）。
配合论文 arXiv:2510.25818。

---

## 1. 依赖与权重 —— 这是全场最干净的一个

```
diffusers==0.35.1   einops==0.8.1   numpy==2.3.4   torch==2.6.0
python 3.13
```
**四个包。** 对比 StyleSSP（8 个 checkpoint、26GB、40GB 下载陷阱）和
Pixelsmith（diffusers 0.25.1 旧栈）。

| 要什么 | 我们有吗 |
|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` fp16 | ✅ **已核实完整，6.7GB，无悬挂链接** |
| `black-forest-labs/FLUX.1-dev` | ✅ **已在服务器** |

**不需要 IP-Adapter、ControlNet、BLIP2、CLIP、SwinIR，不需要任何额外训练权重。**

**README 的内联代码块写 `restart_ratio=0.6`（SDXL），实际脚本 `run_scalediff_sdxl.py`
写 `0.4`，与论文 τ=400 一致。** 文档笔误，不是代码/论文不符。

---

## 2. 管线，逐行译出

```python
# Phase 1：标准 SDXL，NPA 关闭，50 步 → 1024² 基图
attnController.disable()
latents, image = 普通采样(1024²)

# Phase 2：p = 1, 2（→ 2048² → 4096²）
restart_step = 50 * (1 - 0.4) = 30        # 每级只跑 20 步
base_alpha   = 1 - alphas_cumprod[t_30]

# --- LFM：Latent Frequency Mixing ---
latents_LU = bicubic(latents, ×2^p)        # latent 域上采样
image_RU   = bicubic(image,   ×2^p)        # 像素域上采样
latents_RU = VAE.encode(image_RU)
latents_LFM = refine(latents_LU, latents_RU, scale=1, scale_factor=0.125)
#   refine(x_ref, x_pred, s, sf) = x_pred + s * (LP(x_ref) - LP(x_pred))
#   其中 LP(·) = up(down(·, sf))，sf=0.125 ⇒ 截止频率 Nyquist/8
#   ⇒ latents_LFM = HP(latents_RU) + LP(latents_LU)
#     高频来自【像素域上采样后 re-encode】，低频来自【latent 域上采样】

# --- diffuse ---
latents = add_noise(latents_LFM, noise, t_30)
attnController.enable()                     # NPA 只在放大阶段开

for i, t in timesteps[30:]:
    noise_pred = UNet(...)
    # --- SG：Structure Guidance ---
    scale   = (1-alphas_cumprod[t]) / base_alpha      # 1.0 → ~0，纯时间函数
    pred_x0 = get_pred_x0(latents, noise_pred, i+30)
    pred_x0 = refine(latents_LFM, pred_x0, scale, sf=0.125)
    #        = pred_x0 + scale * (LP(latents_LFM) - LP(pred_x0))
    #        把 pred_x0 的低频按 scale 拉向 latents_LFM 的低频
    noise_pred = get_noise(latents, pred_x0, i+30)
    latents = scheduler.step(...)
```

**LFM 把 Pixelsmith/HiWave 的"像素域上采样"和 LSRNA 的"latent 域上采样"合并了**——
各取一半频段。这就是它不需要训练 latent 超分（LSRNA 要训 1.29M 参数）的原因。

---

## 3. NPA：window 与 query/KV 的几何

```
down_blocks.1 → window 64      down_blocks.2 / mid / up_blocks.0 → 32      up_blocks.1 → 64
只挂在 attn1（自注意力），cross-attention 不动
```

- **query**：`rearrange` 成**不重叠**的 `(w/2)²` 小块
- **KV**：以该块为中心、边长 `w` 的邻域（起点 `clamp(idx - w/4, 0, h - w)`）

两条直接推论：

1. **这些层里没有任何全局注意力。** 两个相距很远的背景块**永远看不见彼此**，
   各自独立作画 —— 这与它自己承认的
   *"repetitive artifacts may still occur in **background regions**"* 完全对应。
2. **边界处 KV 窗口被 clamp 成偏心**，且 **query 块之间零重叠、零平滑**
   （Pixelsmith 做重叠平均，HiWave 用 50% overlap，ScaleDiff-SDXL 两者都没有）。

---

## 4. 【已证】两条从代码读出来的事实

### ① 锚是**静态**的

`latents_LFM` 在循环**外**算一次，之后 20 步每步都拿它当参照，**从不更新**。

对照同届 NeurIPS 2025 的 HiFlow，它的命名失效恰恰是：
> *"they typically rely solely on the **endpoint** of the low-resolution sampling
> trajectory while **neglecting intermediate states**"*

**ScaleDiff 用的比"终点"还静态**——是 LFM 混合后的一个固定张量。
**同一届会议，两篇互不比较**（ScaleDiff 2025-10 晚于 HiFlow 2025-04，本可以比）。

### ② 引导强度仍然是一个全局标量的时间函数

`scale = curr_alpha / base_alpha`，从 1.0 单调衰减到 ~0，**全图共用，不看内容**。

对齐四代：

| | 引导强度 | 空间 | 时间 | 反馈 |
|---|---|---|---|---|
| Pixelsmith | Slider（手调标量） | 全图一个 | **阶跃**（第 30 步关） | ❌ |
| HiWave | τ、w_d（常数） | 全图一个 | 阶跃 + 频段 | ❌ |
| LSRNA | Canny 图 | ✅ 逐像素 | 一次性注入 | ❌ |
| **ScaleDiff** | `curr_alpha/base_alpha` | 全图一个 | **斜坡**（连续衰减） | ❌ |

**ScaleDiff 把阶跃换成了斜坡，其余不变。**

### ③ FLUX 有边界伪影缓解，SDXL 没有 —— 代码可证

```
FLUX/transformer_scalediff_flux.py:389   if self.query_random_jitter:
                                  391      random_h = random.randint(0, base_height//2)
                                  392      random_w = random.randint(0, base_width//2)
SDXL/                              (grep jitter → 无)
```
README 自述：*"query_random_jitter — Reduce boundary artifacts with minimal
computation cost (**FLUX only**)"*

> **作者加了 jitter 就等于承认 query 分块会产生边界伪影，而 SDXL 版没有这个缓解。**
> **⇒ 可检验预测：SDXL 版在 query tile 边界上应有可见接缝，FLUX 版更轻。**
> 这是从代码读出来的，不是猜的，**第一次跑就能判**。

---

## 5. 显存与速度：唯一的未知数

- 论文：**A6000（48GB）**，4096² SDXL **113s** / FLUX 407s
- 代码：`self.vae.to(dtype=torch.float32)` —— **VAE 跑 fp32**；`enable_tiling()` 已开
- 每一级都要对全图做一次 `vae.encode`（4096² → 需要 tiling）
- UNet 在 4096² 全图上跑（NPA 只切注意力，其余层是全图）

**24GB 够不够，论文没说，代码看不出来。这是第一个必须实测的数。**
若不够，可退到 `upsample_stage=1`（2048²）。

---

## 6. 看图时的靶子（全部来自代码/论文自述，不是方法假设）

1. **背景区的重复**——它自己在 Limitations 里点名（NPA 无全局注意力，机制吻合）
2. **query tile 边界的接缝**——SDXL 无 jitter、无重叠平滑（代码可证）
3. **close-up 时局部内容不一致**——它自己点名
4. **静态锚在后期失效**——`scale` 衰减到 ~0 后，最后若干步完全无约束

**这四条是"先看哪里"，不是结论。名字要等图出来自己起。**

---

## 7. 判定

**ScaleDiff 是目前最好的在位者，而且是操作成本最低的一个：**
四个 pip 包、零额外权重、两个骨干、SDXL 和 FLUX 权重我们都有、
4096² 113 秒（若 24GB 跑得动）意味着 1000 条 prompt ≈ 31 小时，
**这是唯一一个我们做得起全量评测的在位者。**

**下一步：建 env → 跑一条 → 测显存和秒数。** 不做任何方法上的推测。
