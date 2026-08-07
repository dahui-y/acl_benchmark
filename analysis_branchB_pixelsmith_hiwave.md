# B 支线（基图锚定式）：Pixelsmith 与 HiWave 的问题、动机、机制、以及它们留下的开口

日期 2026-08-07。材料：Pixelsmith 论文全文（arXiv 2406.07251v2，28 页 PDF 已抽文）+
官方代码 `Thanos-DB/Pixelsmith`；HiWave 论文全文（arXiv 2506.20452v1）。

**全文纪律：【已证】= 从论文/代码直接可验证，现在就成立；【待验】= 必须跑出图才能确认或推翻。**

---

## 1. Pixelsmith（NeurIPS 2024）

### 它的动机与命名的失效

| 它命名的失效 | 原文 |
|---|---|
| 直接放大 → **prompt 被复制满整张图** | *"even scaling up by a factor of 2 would lead to the **duplication of the image produced by the prompt** across the higher-resolution image"* |
| patch 去噪的根因 | *"each patch is denoised with the same condition—the text-prompt. This leads to **multiple repetitions of the condition** and results in poor-quality generations"* |
| **DemoFusion 的多级中间分辨率 → 多尺度重复** | *"approaches like [8] where a fixed number of intermediate resolution steps may **amplify this issue progressively with every step**"*（附录 D 专门画了图） |
| 显存/时间 | ScaleCrafter 显存随分辨率涨；DemoFusion 慢 |

> **注意**：Pixelsmith 第 3.2 节末尾那句"每个 patch 用同一个条件 ⇒ 条件被重复"，
> 跟 **AccDiffusion（ECCV 2024，A 支）命名的病因一模一样**。两支各自独立发现了同一条。
> 但**修法不同**：AccDiffusion 给每个 patch 单独的 **prompt**；
> Pixelsmith 给每个 patch 一个**结构条件**（基图对应的 patch）。
> 这是两支真正的分野点，也是它们从此互不比较的起点。

### 机制（论文 + 代码）

```
SDXL 生成 1024² 基图
  → 在【像素空间】Lanczos 上采样到目标分辨率（论文明说：latent 空间上采样会引入
    scheduler 没见过的噪声）
  → VAE encode 得 z_guid，前向扩散得 z_guid_t
  → 高分辨率过程【从纯噪声 z_T ~ N(0,I) 起步】
  → 每个 timestep：随机裁一个 128²(latent) = 1024²(pixel) 的 patch
       · 频域融合（见下）
       · chess/条纹 mask Λ 把结果与 z_guid_{t-1} 交错
       · 重叠区取平均
       · DiffInfinite 式记账：每个像素每个 timestep 只去噪一次
  → 走到 Slider 那一步之后，【完全不再用 guidance】，退化成普通 patch 去噪
```

### 【已证】三处可以现在就说的东西

**(a) 论文与代码不符，而且论文的解释在数学上是错的。**

论文（式 5）：平均 **虚部**，保留实部——
> *"The **imaginary part** in the frequency space contains most of the **low-frequency
> information** of the image."*

这句话不成立。实部/虚部的划分对应偶/奇对称分量，等价于**相位**信息，
**与频段（低频/高频）无关**——低频住在 DC 附近，实部虚部里都有。

代码 `pixelsmith_pipeline.py` 实际做的是：
```python
magnitude_latents = torch.abs(fft_sub_latents)                    # 当前 latent 的幅度
mixed_phase      = torch.angle(complex_latents + complex_guid_latents)   # 相位取复数和的辐角
```
**这是幅度/相位分解，不是实部/虚部分解。** 而且这个做法是对的——
"相位携带结构、幅度携带纹理"是图像处理里的经典结论，
**代码等于在"把结构锚在相位上、把纹理留给幅度"**。

> **论文用了正确的机制，却给了错误的解释，因而没有深究它。**
> 这是 StyleSSP 开口的教科书形态——StyleSSP 的原话是前任
> *"manipulates the startpoint but only by rescaling, **without fully investigating its role**"*。
> 这里比那还强一级：**在位者连自己在做什么都写错了。**

（顺带修正我自己：论文描述的"平均虚部"版本**不是平移等变**的，
但代码的幅度/相位版本**是**平移等变的——`|A|·e^{i·angle((A+B)e^{iθ})} = C·e^{iθ}`。
我一度以为随机裁剪会让引导强度随裁剪偏移抖动，**核对代码后这条不成立，撤回。**）

**(b) `angle(A+B)` 不是相位平均，是"按幅度加权的相位融合"——而这个权重没有人选过。**

复数和的辐角，偏向模长大的那一方。所以：
- 在 |z_guid| 大的频率上 → 锚定强
- 在 |z_guid| 小的频率上 → 几乎不锚

而 **z_guid 来自 Lanczos 上采样**：它在基图 Nyquist 以上**幅度基本为零**。
4× 外推时，目标频谱有 3/4 的带宽落在那以上。
**⇒ 那 3/4 的频谱上，锚定强度自动趋近于零。**

这不是设计出来的，是 `angle(A+B)` 这个写法的副产品。
**它意味着"哪些频率被锚住"由 Lanczos 的截止频率决定，而不是由任务决定。**

**(c) 作者自己承认 Slider 必须逐图手调。**

> *"The optimal value for the Slider **varies depending on the base image**.
> **Adjusting the Slider manually is necessary** to achieve the best output. …
> **Experimentation and iterative adjustments are often required** to find the most
> effective value for each specific image and resolution."*

而消融表证实了这是个真权衡：
| Slider 位置 | 结果 |
|---|---|
| 0（全放开） | *"introduces numerous artifacts"*，FID 76.2 |
| 30 | 最优，FID 63.9 |
| 49（全锚死） | *"lacks fine detail"*，FID 65.4 |

**结构保真 ↔ 细节丰富，由一个全局标量控制，且最优值逐图不同、只能手调。**
README 的示例也是手填的：×2 用 `slider=20`，×4 用 `slider=30`。

---

## 2. HiWave（arXiv 2506.20452，venue 未核实，**无代码**）

### 它命名的失效

> *"Patch-based approaches often produce **duplicated objects**, while direct inference
> methods **struggle to maintain global coherence** at very high resolutions
> (e.g., beyond 2048×2048)"*

对 Pixelsmith 的具体判词：
> *"Pixelsmith generates more detailed images but **frequently suffers from object duplication**"*

用户研究：对 Pixelsmith 的偏好率 **>80%**。

### 机制

```
1024² 基图 → 【同样是】Lanczos 在像素域上采样
  → 【patch-wise DDIM 反演】取回"能生成这块图的噪声"，从反演噪声开始采样
     （对比 Pixelsmith：从纯噪声起步）
  → DWT 细节增强器（sym4）：
       低频  D̃^L = D_c^L                                （只用条件预测，不做 CFG 放大）
       高频  D̃^H = D_u^H + w_d · (D_c^H − D_u^H)，w_d = 7.5
       逆 DWT 重建
  → 50% patch 重叠
  → skip residual 只在前 τ 步用：τ = 15 (2048²) / 30 (4096²)，总 50 步
  → 渐进：1024² → 2048² → 4096²
```

### 它自己承认的问题

> *"current metrics are often **unreliable at high resolutions**, as they typically
> downscale images to lower resolutions (e.g., 224×224) before computing the score"*

**在位者自己说这一格的评测是坏的。** 这既是风险（我们也得面对），也是一个信号：
**这一格的"怎么量"本身还没有共识。**

---

## 3. 两代对齐：HiWave 修了什么，又留下了什么

| | Pixelsmith | HiWave 的改动 | 【已证】留下什么 |
|---|---|---|---|
| 高分辨率起点 | 纯噪声 | **基图 patch 的 DDIM 反演噪声** | 反演是**逐 patch 独立**做的 ⇒ 全局噪声场不连贯 |
| 锚定机制 | 相位融合 + 条纹 mask（相位/幅度切法） | **DWT 分频**（频段切法） | **两种切法互不兼容，没人比较过哪种对** |
| 锚定强度 | Slider，**逐图手调**（作者承认） | τ = 15/30，**冻成全局常数** | Pixelsmith 说最优值逐图不同；**HiWave 没有解决这个，只是不再暴露它** |
| 细节强度 | 由幅度自由决定 | w_d = 7.5，**一个全局标量**，所有高频子带 / 所有 patch / 所有 timestep 共用 | 同上 |
| 基图来源 | Lanczos | **同样是 Lanczos** | 上采样后**没有任何真实高频**，所有细节都是模型编的，且**没有任何东西约束"编什么"** |

> **关键：Pixelsmith 把"该锚多紧"暴露成一个必须手调的旋钮并诚实承认；
> HiWave 把它冻成常数，问题没被解决，只是被藏起来了。**

**这正是 StyleID(γ) → StyleSSP(α) → StyleFM(三分带) 那条链每一代的开口形状：
一个空间上/逐图变化的量，被在位者用一个全局标量控制。**

---

## 4. 由此得到的候选研究问题

### P1【待验，最强】锚定强度是空间变化的量，两个在位者都用标量

**假设的失效（必须跑图确认）**：
天空、平墙这类区域完全放开也不会长出东西，语义主体区（人、手、脸、动物）一放开就长幻影。
Slider / τ 是全图一个数，只能取折中 ⇒ **主体区仍长幻影，同时平坦区细节被压**。

**这一格的免费监督信号，而且比我们之前说的那个更强**：
> 高分辨率结果**下采样回 1024² 之后，应当与基图逐像素一致**。
> 任何偏离，**按构造就是伪影**——不需要标注、不需要判别器、不需要人。
> 可微、精确、逐像素、同 seed 完全可复现。

**注意**：这个信号被这一格所有方法用过，但**只用作早期 timestep 的软先验**
（DemoFusion 的 skip residual、Pixelsmith 的 Slider 前引导、HiWave 的 τ），
**从来没有被当作显式的一致性目标，而且所有方法都在中途把它关掉。**
Pixelsmith 的消融表说明了为什么要关：不关就没细节。
**⇒ "什么时候、在哪里可以安全地关"，就是问题本身。**

**非平凡的地方不在"把标量变成图"**（那是显然的一步，只值一篇 AAAI），
**而在"用什么量来决定每一处该锚多紧"** ——
基图自身的哪个内在量，能预测"这里放开会长幻影"？
**这个量必须靠跑图找，不能靠想。**

### P2【已证一半】锚定的坐标系可能选错了：按频段切 vs 按结构/纹理切

- HiWave **按 DWT 频段切**：低频保、高频引。
- Pixelsmith 的代码**按相位/幅度切**（= 结构/纹理），但**论文写错了，作者没意识到**。

"该保的"是**结构**，"该放的"是**纹理**——而结构与纹理**在每个频段里都同时存在**
（4096² 上一个 200px 的幻影人物，是中频里的结构；一片草地的纹理也是中频）。
**按尺度切，切不开结构和纹理。**

> 这条与风格迁移那一格的 StyleFM 是**同一个数学处境**：
> StyleFM 的 buffer band 承认了"内容和风格在频域重叠"；
> 这里是"结构和纹理在频段上重叠"。**同一个病，在另一格还没被命名。**

**而且这条可以直接用实验回答**：同 seed、同基图，把锚定分别放在
(i) 低频带、(ii) 相位、(iii) 两者组合 上，看幻影出现率与细节丰富度。
**这是我们能做的第一个真正的对照实验，而且它同时是 Fig. 1。**

### P3【工具，不是选题】分支互盲 ⇒ 三方同 seed 对照

```
1024² 基图（语义真值） | A 支 AccDiffusion v2 的 4096² | B 支 Pixelsmith 的 4096²
```
"没人比过"是空位，**空位不是选题**（这是我犯了五次的错）。
但它是**看见坏图最便宜的仪器**：两支互为对照，坏在哪一眼定位。

---

## 5. 操作性：Pixelsmith 是我们目前最便宜的在位者

| | 值 | 来源 |
|---|---|---|
| 显存 | **7.4 GB，所有分辨率都是这个数** | 原文："tested on a single RTX 3090 GPU, with all tested resolutions requiring 7.4GB" |
| 我们的卡 | RTX 4090 23.5 GB | 已核实 |
| 基座 | SDXL base | **我们已有 fp16，6.7 GB，已核实完整无悬挂链接** |
| 代码 | 单文件 `pixelsmith_pipeline.py` + `autoencoder_kl.py` + `vae.py`，diffusers 系 | 已看仓库 |
| 数据 | LAION-5B 随机 1000 对 prompt（我们可以先用几十条自建 prompt） | 原文 |
| 依赖 | `diffusers==0.25.1 transformers==4.37.0 accelerate==0.26.1 xformers==0.0.25` | requirements.txt |

**风险**：依赖版本比我们已建的 StyleSSP 环境（diffusers 0.30.0 / transformers 4.44.0）旧，
可能要么另建一个 env，要么试试直接用现有的。**这是第一个要测的东西，几分钟就知道。**

对比 StyleFM：要另建 LDM 老栈（py3.8/torch1.8 时代）**并且**要下 SD1.4（4 GB，服务器下不了，得走 Mac）。
**Pixelsmith 到第一张图的成本更低——权重已在本地。**

---

## 6. 下一步（顺序不许乱）

1. **拉 `Thanos-DB/Pixelsmith`，用现有 conda env 直接试跑一张 2048²。**
   只测三件事：能不能 import、显存、单图秒数。
2. 跑 20–30 条自选 prompt 的 **1024² 基图 + 2048² + 4096²**，同 seed。
   **靶子**：幻影出现在哪类区域？出现在 Slider 之前还是之后？
3. **Slider 扫描**：同一张基图，slider ∈ {0,10,20,30,40,49}，看幻影何时出现、细节何时消失。
   这一步直接检验 P1 的前提——**权衡是不是真的存在，以及最优值是不是真的逐图不同**。
4. 看到具体失效 → 命名 → **才**回来查这条具体失效的占位。
5. 诊断 → 让诊断挑杠杆。

**在第 2 步看到图之前，P1/P2 都只是靶子，不是结论。**
