# B 的结果：高分辨率外推这一格

日期 2026-08-06。问题定死为：**HiWave 之后，"patch 边界伪影 + 跨 patch 语义不一致"
这条失效，有没有已录用的后继者修了？**

---

## 1. 谱系（每一代都开在前任的残余上）

| 代 | venue | 命名的失效 | 修法 | 留下什么（**由下一代点名**） |
|---|---|---|---|---|
| ScaleCrafter | ICLR 2024 | 重复图案、结构畸变 | 扩张卷积 | 仍重复 |
| **DemoFusion** | CVPR 2024 | ↑ | patch + 低分辨率参考引导 | **物体重复** |
| **FouriScale** | ECCV 2024 | 重复+畸变 | 频域扩张卷积 + 低通 | **>2048² 全局语义丢失** |
| **AccDiffusion** | ECCV 2024 | 物体重复 | patch-wise 内容感知 prompt | 同上 |
| **Pixelsmith** | 2024 | ↑ | patch | **重复物体**（被 HiWave 点名） |
| **HiWave** | **SIGGRAPH Asia 2025** | patch 法 >2048² **边界伪影 / 内容重复 / 跨 patch 语义不一致** | 两阶段：基图 → patch-wise DDIM 反演 → **DWT 细节增强器**（低频保结构、高频引细节） | **？** ← 我们的问题 |

**HiWave 原文判词**：*"current training-free approaches either **fail to maintain
global coherence** compared to the base diffusion model or suffer from
**duplicated objects and artifacts** at ultra-high resolutions (e.g.,
4096×4096)"*

---

## 2. 占位核查：**未见已录用后继者**

三轮检索（含专门查 2026 CVPR/ICLR/AAAI），**没有打到 HiWave 之后的已录用工作**。
2025–2026 打到的是 **SEGA / InfoScale / ScaleDiff，全是 arXiv 预印本**。

⚠️ **状态：未见占位。** 但按前四次的教训——这不等于"确认为空"。
HiWave 是 2025-06 的 arXiv、2025 年底的 SIGGRAPH Asia，**窗口本来就还没走完评审周期**。

---

## 3. 这一格对我们的三个决定性优势

### ① **HiWave 就是在一张 4090 上跑的**

原文两处：
> *"enabling 4096×4096 image generation on **consumer GPUs with 24GB of VRAM**"*
> *"All experiments were conducted on a **single RTX 4090 GPU with 24GB of VRAM**"*

**这是我们整轮调研里第一次遇到"当前 SOTA 的实验条件 = 我们的硬件"。**
风格迁移那边 StyleSSP 跑 A100；这里在位者自己就是为 24GB 设计的。
对位从"轻装备打主场"变成了**同等装备**。

### ② 失效肉眼一秒可见，且**自带免费监督信号**

原文描述前任的失效："**duplicated humans in rows 1 through 4 and a phantom
figure in the grass background of row 5**"——图里长出第二个人。
比"风格保真度"那种主观量好抓一个量级。

而且这一格有一个风格迁移没有的东西：
**同 prompt、同 seed 的 1024² 基图，就是语义真值。**
4096² 版本多长出来的东西，一比就知道。**零标注、程序化、无需人工。**

### ③ 评测程序化为主，可复现

FID / KID / CLIP / LPIPS + LAION2B-en-aesthetic **1000 条 prompt**、
SDXL 基座、**同一组种子**、官方 baseline 代码。
用户研究（548 份 A/B，81.2%）是补充，不是主体。

---

## 4. 必须直说的四条风险

1. **venue 层级**。这条链的落点是 ICLR/CVPR/ECCV/**SIGGRAPH Asia**，
   不是清一色 CVPR 主会。SIGGRAPH Asia 分量足够，但如果你只认 CVPR，
   这一格的历史落点分布要先接受。
2. **时间成本**。4096² patch-based 生成，50% overlap，逐 patch 反演——
   **单张图的时间远高于 1024² 风格迁移**。1000 条 prompt 的完整评测跑不动，
   得先用小子集找失效，最后才跑全量。**具体秒数未测，这是第一个要测的数。**
3. **HiWave 有没有开源，我没核实。** SIGGRAPH Asia 论文 + arXiv，
   但代码状态**未确认**。没代码的话在位者要降级成 Pixelsmith 或 DemoFusion
   （两者都开源），故事要重排。
4. **"未见占位"只是三轮检索。** HiWave 才半年，CVPR 2027 周期里
   大概率有人在做。**我们的三个月窗口和这个周期是重叠的。**

---

## 5. 判定与下一步

**B 通过，且比风格迁移那一格更适合我们。** 理由不是"更空"——
是**在位者的实验条件和我们完全一致**，这一条在整轮调研里是独一份。

下一步严格按正确顺序，**第一步是核实代码，不是跑**：

1. **查 HiWave 代码状态**（几分钟）。有 → 它是在位者；无 → 降级到
   Pixelsmith / DemoFusion（确认开源）。
2. **跑 1 张 4096²，测时间和显存。** 这个数决定后面所有实验的规模。
3. **小子集（20–30 prompt）跑当前在位者，看输出。**
   靶子：patch 边界、重复物体、4096² 相对 1024² 基图多长出来的东西。
4. 失效命名 → 才回来查这条具体失效的占位。

**在第 3 步看到图之前，不做任何方法上的推测。**
