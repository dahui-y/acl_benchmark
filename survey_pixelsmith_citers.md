# Pixelsmith 的引用图：谁引了，谁真的比了

日期 2026-08-07。来源：Semantic Scholar Graph API，`arXiv:2406.07251` 的全部引用（分页拉完）。

**全部引用只有 14 篇。**

---

## 1. 引用 × 是否定量比较

| 工作 | venue | 引用 | **定量比较** | 备注 |
|---|---|---|---|---|
| **LSRNA**（Latent Space SR for HR Generation） | **CVPR 2025** | ✅ | ✅ **比了，并搭在它上面** | 自己的方法就叫 `LSRNA-Pixelsmith` |
| **HiWave** | **ACM SIGGRAPH**（S2 记录） | ✅ | ✅ **主 baseline** | Table 1 + 81.2% 用户研究 |
| **AccDiffusion v2** | **TPAMI** | ✅ | ❌ **引而不比** | |
| **ScaleDiff** | **NeurIPS 2025** | ✅ | ❌ **引而不比** | baseline：ScaleCrafter / HiDiffusion / DiffuseHigh / FreeScale / DemoFusion / AccDiffusion v2 / UltraPixel |
| **ResDiT** | CVPR 2026 | ❌ **未引** | ❌ | DiT，单阶段 |
| PixelRush | **CVPR（已核定：CVF Open Access 水印，页码 35946）** | ✅ | ❌ **引而不比** | 2026-08-14 读原文订正（原记"2026 arXiv 未确认"有误）。baseline 仅 SDXL-DI/FouriScale/DemoFusion/FreeScale；**无开源码** |
| PhotoQuilt | 2026 arXiv | ✅ | 未确认 | |
| Latent Wavelet Diffusion / APT | 2025 arXiv | ✅ | 未确认 | |
| TNNLS 2026（超高分辨率**编辑**） | TNNLS | ✅ | — | 非本任务 |
| RefGC-SR² / UPLiFT / Convex Opt / HaineiFRDM | arXiv | ✅ | — | 非本线 |

> **14 篇引用，只有 2 篇做了定量比较。**

---

## 2. 四条读法

### ① Pixelsmith 不是这条线的公认基准

一年半、14 次引用、2 次比较。公认基准是 **DemoFusion**（几乎每篇都比）和
**HiDiffusion / ScaleCrafter**。
**⇒ 只拿 Pixelsmith 当唯一在位者，审稿人不一定认。必须同时比 DemoFusion。**

### ② 这条线的评测阵营是分裂的 —— 这条最要紧

四篇 SOTA，四套 baseline：

| 论文 | 它选的对手 |
|---|---|
| HiWave | SDXL、**Pixelsmith**、HiDiffusion |
| LSRNA (CVPR25) | SDXL+BSRGAN、SDXL、ScaleCrafter、FouriScale、HiDiffusion、Self-Cascade、**DemoFusion**、**Pixelsmith** |
| ScaleDiff (NeurIPS25) | ScaleCrafter、HiDiffusion、DiffuseHigh、FreeScale、**DemoFusion**、**AccDiffusion v2**、UltraPixel、BSRGAN、OSEDiff |
| ResDiT (CVPR26) | **DemoFusion**、DiffuseHigh、I-Max、HiFlow |

**交集只有 DemoFusion 和 HiDiffusion。** 指标也各不相同
（FID / FIDp / FIDc / HPS-v2 / ImageReward / 20 人 / 548 人），
prompt 集也各不相同（LAION2B-en-aesthetic 1000 / LAION-Aesthetics 500 / 自建）。

> **这条线连"跟谁比、比什么"都没有共识 —— 谁都可以挑一组对自己有利的对手和指标。**
> 这把 P0 的价值抬高了一级：唯一能穿透这个的，是一个
> **不依赖对手选择、只依赖基图**的自动量。

### ③ "架构过时"的质疑，有现成的解法先例

**ScaleDiff（NeurIPS 2025）同时在 SDXL(U-Net) 和 FLUX(DiT) 上做实验**，
自称 model-agnostic，**单张 A6000**。
**⇒ 我上一轮建议的 B 方案（在两个骨干上都验证）不是设想，有 NeurIPS 论文趟过。**

### ④ LSRNA 占了 P1 的一半 —— 精确边界

**已被占**：用**基图导出的一张空间图**去调制自由度。
- 区域 = 对解码后的低分辨率参考图做 **Canny 边缘检测**
- 逐像素调制噪声幅度：`g^HR ← g^HR + T(E_resized)·ε`
- 对 Pixelsmith 用 `[0.4, 0.8]`，对 DemoFusion 用 `[0.0, 1.2]`

**我上一轮"把全局标量换成空间图"的措辞，到此作废。**

**看起来没被占的四条（全部标为待验，不是结论）**：

1. **图的语义不同。** RNA 的信号是 **Canny 边缘 = "哪里有细节可加"**，
   不是 **"哪里编造物体最危险"**。
   天空无边缘 → 低噪声（可天空恰恰最安全）；
   人脸边缘密 → 高噪声（可人脸恰恰最不能编）。
   **两张图可能是反相关的。**
   → **这是一个可证伪的预测**，拿到真实的"基图 + 有幻影的输出"配对当天就能判。
2. **RNA 是一次性注入**（上采样后立刻加），
   而 Slider / τ / w_d 控制的是**整条轨迹**。逐时间步 × 逐位置的引导强度没人碰。
3. **LSRNA 自己承认 `[e_min,e_max]` 要按方法手调** ——
   全局标量没死，变成了两个手调常数。
4. **LSRNA 不是 training-free**：SwinIR-light + LIIF，1.29M 参数，
   200K iter，4.7M latent 对。整条线其余全是 training-free。
5. 它命名的失效是 **manifold deviation / 过度平滑**（细节保真），
   **不是幻觉**；**它也没测复制。**

---

## 3. 这一轮改变了什么

- **P1 的措辞作废**，必须重新瞄准到"图的语义"上，而且要**先用实验判**
  （Canny 图 vs 幻影位置是否反相关），不能再靠"看起来没人做"。
- **P0 反而更硬了**：不只是"没有指标能测复制"，而是
  **这条线连评测协议都没有共识**。
- **在位者不能只有 Pixelsmith**：必须加 **DemoFusion**（唯一的公认基准）。
- **跨骨干验证有先例**（ScaleDiff, NeurIPS 2025）。

**下一步不变，而且更急了**：拿到真实的"基图 + 有幻影的输出"配对。
它同时决定 P0 的检测器和 P1 的重新瞄准。
