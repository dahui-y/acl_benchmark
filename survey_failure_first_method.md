# 失效优先的方向搜索法（把"找方向"变成流水线）

日期 2026-08-06。前提：放弃"在清单里找空白任务"，改为
**"哪个现有 training-free 方法有一个可测量、没人修的失效"**。
本文把这句话变成可执行的流程，并给出候选清单和第一批 break test。

---

## 0. 一篇论文需要的完整链条（FreeInpaint 逆向出来的）

失效只是入口。一个失效要长成论文，四环都得在：

> **(1) 失效可观察、有名字** → **(2) 成败样本在某个内部量上有对比差异**
> → **(3) 那个内部量的目标值由任务输入免费给出（零标注）** → **(4) 写成推理期干预**

FreeInpaint：(1) prompt-unaligned；(2) 注意力集不集中在 mask 内；
(3) mask 用户给的；(4) 注意力损失回传初始噪声。

**只有 (1) 没有 (2)(3) 的失效是一句抱怨，不是方法。**
但顺序上 (1) 必须先拿到——(2)(3)(4) 是拿到失效之后才回答的问题。

---

## 1. 流水线的五步

### 第 1 步：候选池（已经有了）

已录用 + 开源 + 4090 能跑。从覆盖图和这几轮调研里筛出的池子：

ITOC (ICLR 2026) / FreeInpaint (AAAI 2026) / Z-Sampling (ICLR 2025) /
ReNO (NeurIPS 2024) / SLD (CVPR 2024) / StyleID (CVPR 2024) /
Ctrl-X (NeurIPS 2024) / PnP (CVPR 2023) / FreeControl (CVPR 2024) /
FreeU (CVPR 2024) / DemoFusion (CVPR 2024) / AccDiffusion (ECCV 2024) /
FouriScale (ECCV 2024) / FreeCustom (CVPR 2024) / KV-Edit / FreeFine (ICCV 2025) /
OmniVTON (ICCV 2025) / ConsistEdit (SIGGRAPH Asia 2025) / ZeST (ECCV 2024)

### 第 2 步：对每个方法，沿四条失效轴生成假设

| 失效轴 | 问法 | 已知先例 |
|---|---|---|
| **架构迁移** | 换到 MMDiT / flow matching 上还成立吗？ | HD-Painter 在 SD3I 上崩（FreeInpaint 表 1） |
| **目标间权衡** | 它修 A 的时候，B 是不是在变差？ | HD-Painter：对齐↑，HPSv2/InpaintReward/LPIPS 全↓ |
| **输入分布** | 哪类输入上系统性失败？（风格家族 / mask 形状 / prompt 结构 / 分辨率） | DemoFusion 的重复伪影 |
| **指标泄漏** | 它报的增益，在它没直接优化的指标上还剩多少？ | FreeInpaint 自己：EditBench 上 LPIPS 多数反而变差，干净指标只有 HPSv2/G.CLIP |

### 第 3 步：筛假设（三条硬性）

1. **可程序化测量，零标注**——在**他们自己的 benchmark + 指标**上可见。
   用在位者自己的坐标系，别人无法说"你换了尺子"。
2. **4090 一天内可测一个假设**。超过一天的假设先放。
3. **没有后续工作已修**——每个假设动手前单独检索一次
   （吸取教训：索引薄 ≠ 世界薄）。

### 第 4 步：break test（每个假设 ≤ 1 天 GPU）

不追求修，只回答"失效存在吗、多大、系统性吗"。产物是一张表 + contact sheet。

### 第 5 步：只有失效坐实后，才回到第 0 节问 (2)(3)(4)

**什么样的失效算"论文级"**（四个都要）：
- **系统性**：某个可命名的子类上高频出现，不是 cherry-pick
- **可归因**：能指出机制层面的原因（这正是 (2)）
- **在位者坐标系内可见**：他们的 benchmark、他们的指标
- **有免费信号可修**：任务输入里有个没被当目标用的监督量（这是 (3)）

---

## 2. 候选 × 失效假设（按 break test 成本排）

| # | 方法 | 失效假设 | 测法 | 成本 |
|---|---|---|---|---|
| 1 | **StyleID / Ctrl-X / PnP** | 架构迁移：原始代码在 SD3.5 上崩或退化 | 跑原仓库，SD1.5 vs SD3.5 各 20 张 | **1–2 天，已定为第一步** |
| 2 | **FreeInpaint** | 指标泄漏：干净指标（HPSv2/G.CLIP/LPIPS）上的增益远小于泄漏指标 | 跑原仓库 EditBench 子集，按泄漏/干净分列 | 1 天 |
| 3 | **Z-Sampling** | 输入分布：自反射在哪类 prompt 上反而更差 | 原仓库 + GenEval 类别分桶 | 1 天 |
| 4 | **ITOC** | 权衡/reward hacking：它声称处理了，验证在哪类编辑上仍破 | 原仓库 + 非 reward 指标 | 1–2 天 |
| 5 | **DemoFusion / AccDiffusion** | 输入分布：重复伪影在哪类布局必现；后续修了没 | 原仓库 + 4×外推 | 1 天（先查占位） |
| 6 | **FreeCustom / ConsistEdit** | 架构 / 多主体数量上限 | 原仓库 | 1–2 天 |

不进池的：FreeU（失效轴是"美学"，无法零标注测量）、加速类（军备竞赛主场）。

---

## 3. 风险与对策

- **风险 1：跑了 N 个 break test 一个论文级失效都没有。**
  对策：break test 便宜（每个 ≤1 天），且**结论确定**（找到/没有，不悬着）。
  预算上限先定 **5 个 GPU 天**——5 个假设全空就回覆盖图换搜索方式，不恋战。
- **风险 2：找到的失效太浅，修起来是工程不是机制。**
  对策：第 5 步的四条判据里"可归因"挡这个——归因不到机制的失效不立项。
- **风险 3：我们能找到的失效，别人也能。**
  对策：优先选**要跑两个架构才看得见**的假设（#1、#6）——
  多数人只在方法的原生基座上复现，跨架构的测量本身就是壁垒。

---

## 4. 执行顺序

1. **#1 StyleID 原始代码上 SD3.5**（这一步同时服务三个目的：
   审计选项 A 的第一格、失效搜索 #1、以及裁决 mmdit_probe 第二轮
   那个负结论到底是"我的复现失败"还是"方法本身迁不过去"）
2. #1 跑的同时，把 #2–#5 的"后续工作已修没修"检索做完（不占 GPU）
3. 按检索结果重排 #2–#5，逐个 break test，5 GPU 天封顶

---

## 附：Style Transfer 这一格的时间线核实（2026-08-06，回答"StyleID 之后是谁"）

用 Westlake-AGI-Lab 的专门清单（维护到 2026-06）交叉核对——**又一次证明
littlewhitesea 那份索引的薄格不可信**：它给 Style Transfer 记 18 条，专门清单里
2025 一年就有 5 篇已录用。

### 已录用 + 时间线（training-free 参考图/文本风格迁移）

| 年份 | 工作 | venue | 代码 |
|---|---|---|---|
| 2024 | StyleID | CVPR 2024 | ✅ |
| 2024 | Cross-Image Attention | SIGGRAPH 2024 | ✅ |
| 2024 | Ctrl-X | NeurIPS 2024 | ✅ |
| 2025 | **StyleSSP**（字节） | **CVPR 2025** | ✅ `bytedance/StyleSSP` |
| 2025 | StyleStudio / RB-Modulation / Attention Distillation | CVPR 2025 | ✅ |
| 2025 | Semantix | ICLR 2025 | — |
| 2025 | **SADis（Free-Lunch Color-Texture Disentanglement）** | **NeurIPS 2025** | ✅ `deepffff/SADis` |
| 2026 | HAM | CVPR 2026（据清单，未自行核实） | 未见 |

**最新的已录用+开源在位者是 StyleSSP（CVPR 2025）**，不是 StyleID。
它做的正是"初始噪声/起点增强 + 频率操纵 + 负引导"，声称专修 content leakage。

### "free-lunch 引入 style transfer 了吗"——早就有了

- **FreeStyle**（arXiv 2024-01，标题就叫 *Free Lunch for Text-guided Style
  Transfer*）：FreeU 同款思路，利用 U-Net 双流编码结构。期刊 Pattern
  Recognition 2026 发表，无顶会 venue。
- **SADis / Free-Lunch Color-Texture Disentanglement：NeurIPS 2025 poster，开源。**
- 实质上整条 StyleID→StyleSSP 线都是 free-lunch（training-free）做风格迁移。

**"把 free-lunch 带进风格迁移"作为立论，2023–2024 就没了。**

### 两条要命的预印本动态

1. **Scheduled Style Injection**（arXiv 2605.26538）：副标题就是
   *Expanding the Style-Content **Pareto Frontier** in Training-Free
   Diffusion-based Style Transfer*——**和我们 sweep 第二轮独立推出的
   "权衡线/脱线"框架是同一个东西**。好消息：框架被独立印证。
   坏消息：这个框架本身已不新。
2. **Dual Rectified Flows**（2511.20986）：inversion-free 风格迁移做在
   rectified flow 上——**MMDiT/流匹配那边的空正在被预印本填**，
   进一步确认杀掉那个方向是对的。

### 对流水线的修订

- 失效搜索 #1 的**在位者从 StyleID 升级为 StyleSSP**：它是最新已录用+开源，
  而且**自称修好了 content leakage**——它修剩下的残余失效才是最值钱的靶子。
- 阳性对照仍可用 StyleID（配方最简、引用最canonical），但 break test 的
  "在位者坐标系"应该是 StyleSSP 的 benchmark 和指标。
- StyleSSP 的基座是哪个（SD1.5/SDXL）**未核实**，动手前先看仓库。

---

## 附 2：StyleID → StyleSSP 拆解（同一格连中两次顶会的机制）

读了 StyleSSP 全文（arXiv 2501.11319v2）。

### StyleSSP 相对 StyleID 的增量

| | StyleID (CVPR 2024) | StyleSSP (CVPR 2025) |
|---|---|---|
| 干预点 | **采样期的注意力**（decoder self-attn 注入风格 K/V + query preservation + 初始 AdaIN） | **采样的起点 z_T**（DDIM 反演产物）+ **反演过程本身** |
| 命名的失效 | （它自己是开创者之一） | ① 布局被改（content preservation）② 风格图内容泄漏（content leakage） |
| 机制 ① | — | FFT 分解反演 latent：**低频 ×α=0.7 削弱、高频保留**、按 1−α 补高斯噪声。依据 FlexiEdit 的观察：latent 的高频载轮廓/布局 |
| 机制 ② | — | **反演阶段加负引导**（ω=1.5，风格图作负条件），让起点远离风格图内容 |
| 基座 | SD1.5 | **SDXL + tile ControlNet + InstantStyle 式注入 + CLIP ViT-L**（注意：training-free ≠ adapter-free，它用了现成训练好的 ControlNet/IP 组件） |
| 对 StyleID 的数字 | — | ArtFID 28.80→**21.50**，FID 18.13→**13.45**，LPIPS 0.5055→**0.4881**，**指标和实验设置全部沿用 StyleID**（"consistent with StyleID"） |

### 为什么这一格能连续出顶会——四个结构性原因

1. **任务目标本身是权衡（Pareto 前沿），不是可解问题。** 风格 vs 内容此消彼长，
   没有方法能"做完"它——每篇已录用工作都只是推一下前沿，**必然留下可测量的残余失效**。
   权衡型任务不会像可解型任务那样饱和，它持续产出论文位。
2. **每一任继任者都开在前任的命名失效上，且用前任自己的坐标系。**
   StyleSSP 的指标、数据、设置逐项沿用 StyleID——审稿风险极低：
   动机是已发表、可复现的失效；尺子是已被接受的尺子。
3. **干预点清单还没用完。** 扩散管线有一排互相独立的旋钮：
   初始噪声 / 反演 / 注意力 K,V / 引导项 / 频域 / 调度器。
   StyleID 拿走"注意力"，StyleSSP 拿走"起点 + 反演引导"。
   **同一个失效 × 一个新旋钮 = 一篇新论文。**
4. **评测全程序化**（ArtFID/FID/LPIPS，零标注），复现便宜，审稿人可自己验证。

### 一个跨任务的重复模式（值得记住）

**StyleSSP 之于 StyleID，恰好等于 FreeInpaint 之于 HD-Painter**：
前任在采样期动注意力 → 继任者**优化起点噪声 + 在采样中加引导**，双双中会。
"起点优化 + 采样引导"是 2025–2026 这条线的胜型，且是**按任务逐个套用**的。

### 对我们流水线的印证与提示

- 这正是失效优先四环链的活例：失效命名 → 对比内部量（z_T 的频率成分 /
  反演轨迹被风格污染）→ 免费信号（内容图自身、风格图作负条件）→ 推理期干预。
- StyleSSP 留下的口子（break test 靶点）：它把生态绑在 **SDXL + ControlNet +
  IP-Adapter** 上——这套组件在 SD3/FLUX 生态**不成熟**，架构迁移轴依然打得到它；
  另外 Scheduled Style Injection（2605.26538）说明前沿仍未收敛。
