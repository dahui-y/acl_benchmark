# 用 FreeInpaint 的思路扫 Image Manipulation (40) + Style Transfer (18)

调研日期 2026-08-06。来源：`littlewhitesea/training-free-methods` 全文下载后逐条解析
（不是靠摘要），加针对性检索。筛子沿用用户设定：**开源 + 已录用**才算可依托的地基。

## 0. FreeInpaint 的模板拆成可执行的判据

FreeInpaint 真正可复用的不是"inpainting"，是这条链：

> **(1) 失败模式有名字且可观察** → **(2) 成功样本和失败样本在某个"内部量"上有对比差异**
> → **(3) 这个内部量的目标值由任务输入免费给出（零标注）** → **(4) 写成损失，回传**

inpainting 的填法：(1) prompt-unaligned，物体根本没生成；(2) 注意力集不集中在 mask 内；
(3) **mask 是用户给的**；(4) `L = Σ[(1-M)·A − M·A]`，只在第一步算（首步注意力≈全时序均值）。

所以扫这两格，就是问：**哪些免费监督信号还没被当成"目标"用？**

---

## 1. 免费信号清单 —— 逐条查占位（Image Manipulation）

| 任务输入自带的免费信号 | 被谁吃掉 | venue / 代码 |
|---|---|---|
| **mask（空间）** | FreeInpaint | **AAAI 2026 ✅代码** |
| 未编辑区必须像素相同 | KV-Edit | arXiv ✅代码 |
| 源 prompt ↔ 目标 prompt 的 token 差 | Prompt-to-Prompt / CannyEdit | CVPR 2023 ✅ |
| 用户指定的几何变换（拖拽/旋转/缩放） | FreeFine | **ICCV 2025 ✅代码** |
| 从源图免费抽的 depth / canny / seg | FreeControl | **CVPR 2024 ✅代码** |
| 参考图（exemplar） | ReInversion / Analogist | SIGGRAPH 2024 ✅ |
| 多视角一致性 | Coupled Diffusion | arXiv ✅代码 |
| **任意可微 reward（DeGu 的一般化）** | **ITOC** | **ICLR 2026 ✅ `jinhojsk515/ITOC`** |

**ITOC 是这一格的封门砖。** "Training-Free Reward-Guided Image Editing via Trajectory
Optimal Control"，把 FreeInpaint 的 DeGu 推广到通用编辑，还额外处理了 reward hacking。
**"给编辑latent挂现成 reward"作为一篇论文已经没了。**

### 我另外单查并且判死的四个候选

| 候选 | 占位情况 |
|---|---|
| 风格迁移的 **content leakage**（参考图的物体漏进输出） | **MaskST（ICLR 2025 ✅代码）**、CleanStyle、StyleGallery、Only-Style —— 满 |
| 物体插入的**光照/阴影合理性** | SpotLight、TF-GPH、Light-Guided T2I、3DGS Harmonizers —— 满 |
| **多轮编辑的漂移累积** | **FreqEdit**（training-free, 2512.01755）、AnchorEdit、GeoEdit —— 满 |
| **空间可变 / 多风格迁移** | MAST、HAM、Semantix —— 满 |

**结论要直说：这两格在"任务 × 信号"这一层已经没有天窗了。**
这条线的消耗速度是：任何一个新基座放出来，明显的信号 3–6 个月内被吃干净。

---

## 2. 但 FreeInpaint 自己交出了另一条线索：**架构断层**

它的表里有一个被作者一笔带过、但对我们很要紧的数字。
HD-Painter（唯一的 training-free 前作）加到四个 U-Net 基座上都有增益，
**加到 DiT 基座 SD3I 上直接崩掉**：

| SD3I on EditBench | ImageReward | HPSv2 | L.CLIP | InpaintReward |
|---|---|---|---|---|
| Base | 0.2993 | 25.48 | 26.26 | -0.2170 |
| **+HD-Painter** | **-0.5020** | **21.56** | **22.83** | -0.2988 |
| +FreeInpaint | 0.5248 | 25.70 | 26.98 | -0.0694 |

作者原话：*"HDP is incompatible with the DiT-based SD3I"*。
而且 FreeInpaint 自己上 DiT 也得换参数：τ_round 5→1、lr 0.0125→**0.1（8×）**。
它的 `L_c`/`L_s` 是写在"cross-attention 和 self-attention 是两个独立对象"这个
**U-Net 假设**上的——MMDiT 里根本没有这个划分，只有一个 `[text; image]` 的联合注意力矩阵。

### 全索引 195 条的架构关键词统计

```
MMDiT 2   MM-DiT 0   FLUX 0   Flux 3   SD3 0   Qwen-Image 0   rectified flow 0
```

**整个索引的标题里，MMDiT / flow-matching 时代几乎不存在。**

### 两格的"已录用 + 开源"地基，全是 U-Net 机制

Image Manipulation：**14/40 已录用**（26 条是纯 arXiv）。
Style Transfer：**5/18 已录用**，其中真正做风格迁移的只有三篇：

| 工作 | venue | 机制 | 依赖 U-Net 的什么 |
|---|---|---|---|
| **StyleID** | CVPR 2024 ✅代码 | 风格 K,V 注入 + query preservation + query AdaIN | **self-attention 层是独立模块** |
| **Cross-Image Attention** | SIGGRAPH 2024 ✅代码 | 跨图注意力 | 同上 |
| **Ctrl-X** | NeurIPS 2024 ✅代码 | 特征 + 注意力注入 | **decoder ResNet 的空间特征** |
| （PnP，编辑格） | CVPR 2023 ✅代码 | 空间特征注入 | **decoder ResNet 块** |

**这四样东西 MMDiT 一样都没有**：没有独立 cross-attention、没有 ResNet decoder、没有 U-Net 跳连。

---

## 3. 由此得到的唯一还活着的候选

**任务：MMDiT / flow-matching 基座（FLUX、SD3.5、Qwen-Image）上的参考图风格迁移，training-free。**

按模板四条逐个填：

| 条件 | 状态 |
|---|---|
| (a) 任务定义清楚 | ✅ 风格迁移，评测协议现成（StyleID 那套） |
| (b) **目前要训练** | ⚠️ **有条件成立**：U-Net 上有免训练解法；**但在 MMDiT 上，实际部署的是训练出来的 IP-Adapter / InstantStyle / style LoRA。这个任务在这个架构上退回到了训练侧** |
| (c) 结构性质 | ❓ **待验证**：MMDiT 的联合注意力**每个 block 都更新 text stream**，U-Net 里文本是冻结条件。风格是否落在 text stream 上，是一个 U-Net 没有对应物的问题 |
| (d) 索引里没有 | ✅ Style Transfer 那 18 条**没有一条**是 MMDiT/FLUX/SD3 |

### 必须先说的风险（不在核实前推荐）

1. **最大的风险是退化成"移植"**。"把 StyleID 搬到 FLUX 上"不是论文。
   **只有 (c) 成立——即找到一个 U-Net 没有对应物的性质——它才是论文。**
2. **邻居正在往里挤，且是同一个组**：ConsistEdit（SIGGRAPH Asia 2025，✅代码）已经
   拿下"MMDiT 注意力性质 + 编辑"，ColorCtrl / FreeFlux 同方向。
   他们做的是**编辑**不是**风格迁移**，但离得很近。
3. 风格侧也已有预印本在动：MAST、HAM、Inversion-Free Style Transfer with Dual
   Rectified Flows（2511.20986）。**都还没 venue，但不能当作空的。**

### 一票否决实验（半天，一张 4090）

在 FLUX.1-dev 或 SD3.5 上，对 content prompt + style 参考，
**逐 block 导出联合注意力的四个子块**（text→img / img→text / img→img / text→text），
测：**换掉哪一个子块能迁风格而不漏内容**。

- 若答案落在 `img→img`（= U-Net self-attention 的等价物）→ **就是移植，这条路死**
- 若落在 text stream 相关的子块 → **(c) 成立，这是 U-Net 没有的机制，可以往下走**

**这个实验必须先跑。跑完之前不推荐这个方向。**

---

## 4. 备选：把上面的veto实验做成一篇分析论文

如果不想赌 (c)，同样的机器可以换个出口：

> **U-Net 时代的 training-free 方法，有多少能过架构这一关？为什么过不了？**

- 素材现成：这两格 **19 篇已录用 + 开源**的方法
- 动机现成：FreeInpaint 表里 SD3I+HDP 从 0.2993 掉到 -0.5020，是**已发表的**失败
- 不训练、不标注、4090 跑得动
- 它同时**就是**上面那个方向的前置实验——先做它，method 论文的 (c) 顺带被证伪或证实

风险：是分析/benchmark 类，不是 method 类；CVPR 收，但不是主流赛道。

---

## 5. 尚未核实的项（不要当结论用）

- Style Transfer 18 条的**实际基座**只按标题判断，没有逐篇开源码确认
- FreqEdit / MAST / HAM / Dual Rectified Flows 的 venue 状态是"未见"，不是"确认没有"
- MMDiT text stream 承载风格，目前是**从架构推的猜想**，没有任何测量支持
