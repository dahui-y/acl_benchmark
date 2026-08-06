# A：动作导致的物体状态变化——方法谱系（按时间）

调研日期 2026-08-06。逐篇读原文（arXiv HTML / 全文），不采信检索摘要。

> **最重要的一条，先说：**
> **"只要打败最好的方法就成为新 SOTA"这个前提，在这个方向上不成立——
> 因为这个方向没有共享基准。** 六篇里没有任何两篇在同一个评测上比过。
> 每篇自己造评测、自己选基座、自己挑基线。
>
> 这既是风险也是机会：**OSCBench 是第一个真正针对这个构念的基准（ACL 2026 主会，
> 1119 条 prompt，6 个模型），而且到今天为止没有任何方法在它上面报过数。**

---

## 1. 谱系表

| # | 工作 | 时间 | 需要训练？ | 基座 | 训练/评测数据 | 对比基线 | 指标 |
|---|---|---|---|---|---|---|---|
| 1 | **GenHowTo** (CVPR 2024) `2312.07322` | 2023-12 | **训练**。SD 2.1 + 复制的 ControlNet 编码分支，原编码器冻结 | Stable Diffusion 2.1 | 训练：COIN + ChangeIt 的 **35k 视频 → 20 万三元组**（自监督挖掘）；评测：ChangeIt 留出 5 个交互类，233 条人工核验 + 1.4 万条 | SD、Edit Friendly DDPM、InstructPix2Pix、CLIP(手写 prompt) | 分类准确率（在生成图上训线性分类器、真图上测）、FID、10 人用户研究 |
| 2 | **TVG** `2408.13413` | 2024-08 | **免训练** | 视频扩散模型（未指明） | 基准 + 自采图像对（未列名） | 未列（只对照传统 morphing） | 未列 |
| 3 | **ShowHowTo** (CVPR 2025) `2412.01987` | 2024-12 | **训练**。视频扩散模型 | 视频扩散 | **100 万教学视频 → 0.6M 图文序列** | — | — |
| 4 | **From Prompt to Progression** `2509.19690` | 2025-09 | **免训练**，纯推理期 | VideoCrafter2（+ OpenSora 验证兼容） | **自建 CAT-Bench**（120 对 prompt，8 类属性）+ TC-Bench-T2V 的属性/背景子集 | 单 prompt：AnimateDiff、ModelScope、Latte、VideoCrafter2；多 prompt：**Free-Bloom、VideoTetris、Gen-L、FreeNoise** | 整体/逐帧转变分（CLIP 方向相似度）、VBench 五项、38 人用户研究 |
| 5 | **Show Me** `2511.17839` | 2025-11 | **训练**。两阶段 LoRA：先空间、后时序 + 运动奖励 | DynamiCrafter | 训练：Something-Something V2 **112,321** + EPIC-KITCHENS-100 **57,602**；评测：SSv2 2,048 / EK100 8,236 / Ego4D | 图像端：ControlNet、SDEdit、InstructPix2Pix、**GenHowTo**、AURORA、**ShowHowTo**；视频端：AnimateAnything、Seer、ConsistI2V、DynamiCrafter | CLIP-I/T、DINO-I、FID、PSNR、LPIPS、FVD、ViCLIP、EgoVLP、VQAScore、Motion Score |
| 6 | **CausalMotion** `2606.14317` | 2026-06 | **免训练**，全程推理期 | **LTX-Video**（选它是因为支持在指定时间位置上做多关键帧条件） | **PhyGenBench**（力学/光学/热学/材料）+ VBench + Gemini 2.5 Flash 当判官 | CogVideoX-T2V-5B、LTX-Video、OpenSora、CogVideoX-I2V、SVD-XT、SG-I2V、LLM-Grounding Video Diffusion、**PhyT2V、VideoDPO、Diffphy、PhyGDPO、VLIPP**；VBench 表另比 Wan2.1-1.3B、VChain | PhyGenBench 四项、VBench 质量分、VLM 判官三项 |

---

## 2. 它们之间的联系：**两条几乎不互引的支线**

### 支线甲：视觉/教学视频，**都要训练**
```
GenHowTo (CVPR'24) → ShowHowTo (CVPR'25) → Show Me (2025-11)
```
- 共同点：从**真实教学视频**自动挖三元组/序列当训练数据（ChangeIt / COIN / HowTo100M / SSv2 / EK100）
- 输出是**图像或图像序列**，评测用 FID / CLIP / FVD 这一套
- Show Me 明确把 GenHowTo 和 ShowHowTo 当基线比 —— **这条线内部是接续的**
- **数据规模 10 万–60 万级，我们跑不动**

### 支线乙：训练自由的推理期控制
```
TVG (2024-08) →（Free-Bloom / FreeNoise / Gen-L / VideoTetris 这一批多 prompt 方法）
   → From Prompt to Progression (2025-09) → CausalMotion (2026-06)
```
- 共同点：不动权重，在**去噪过程里加引导**——插值控制、潜空间方向、关键帧软约束
- 评测各自为政：Prompt-to-Progression 自建 CAT-Bench，CausalMotion 用 PhyGenBench
- **这条线单卡可行，是我们唯一能进的那条**

**两条线互不引用。** Show Me 的基线表里没有 Prompt-to-Progression，
CausalMotion 的基线表里没有 GenHowTo/Show Me。

---

## 3. 关键：它们**没有一篇**在做 OSCBench 的那个构念

| 工作 | 它实际处理的 | 是不是"动作导致的物体状态变化" |
|---|---|---|
| GenHowTo / ShowHowTo / Show Me | 教学步骤的图像/序列生成，条件是**给定初态图** | 接近，但是 **I2V/I2I**，不是纯文本条件的 T2V |
| TVG | 两张给定图之间的**转场** | 否，是 morphing |
| **From Prompt to Progression** | **属性转变**：年龄、胡子、妆容、发型、颜色、材质、光照、天气 | **否**——是主体属性渐变，不是动作把物体变成另一个状态 |
| **CausalMotion** | **物理合理性**：力学、光学、热学、材料 | **否**——是物理，不是动作语义 |

**最像的那篇（Prompt to Progression）测的是"人变老"，不是"苹果被切开"。**
削皮 / 裹粉 / 按压这些 OSCBench 的核心动作，六篇里没有一篇做。

而且 OSCBench 的 **novel / compositional 两个划分**（不常见但合理的动作—物体配对、
多动作复合）**完全没人碰过**——那正是 OSCBench 报出退化最严重的地方。

---

## 4. 对"打败 SOTA 即可"这个计划的实际影响

**风险：没有现成的排行榜可以爬。** 要在 OSCBench 上声称 SOTA，
必须**自己把基线跑出来**，而且：

1. 基线们**基座不同**（VideoCrafter2 / LTX-Video / DynamiCrafter），
   公平比较要么统一基座（要移植），要么各跑各的基座（要解释）。两条都是活。
2. 算力按 §之前算的：1119 条 × 208 s = **2.7 天/条件**（480p）。
   baseline + 我们 + 两个消融 + 两个竞争方法 = 6 条件 = **16 天**，还是最后一跑。
3. 支线甲的三篇（要训练的）我们**根本复现不了**——10 万级教学视频数据 + 训练。
   只能在 related work 里说明"不可比：需要 I2V 条件 / 需要训练"。

**机会：位置是真的。**
- **第一个在 OSCBench 上报数的方法**——这个位置现在空着，而 OSCBench 是 ACL 2026 主会
- 支线乙没人做过**动作语义**，只做过属性渐变和物理
- CVPR / ACL 双投的说法成立：**方法在文本侧就投 ACL，在去噪侧就投 CVPR**

---

## 5. 如果做，方法应该长什么样（尚未核查占用）

从 §3 的空白反推，**没被占的那格是：把动作—物体的状态变化语义，
在推理期从"模型见过的配对"迁移到"没见过的配对"上**——
正对 OSCBench 的 novel 划分，也正是支线乙从没碰过的东西。

CausalMotion 提供了一个可借的机制形状（VLM 分解 → 关键帧软约束），
但它分解的是**物理因果链**；换成**动作的状态变化链**是不同的东西。
Prompt-to-Progression 提供了"潜空间方向"的形状，但它的方向来自**属性词对**，
换成**动作—物体对**同样不是一回事。

**这两点是"形状可借、内容不同"，不是"改个名字"。** 但它离"薄壳"只有一步之遥，
**必须先做占用核查再动手**，我不在核查前推荐。

---

# C：分析文章——占用核查结果

> **结论：C 的"机制可解释性工具"这一片已经很热，但"状态变化/动作语义在哪一环丢失"
> 这个具体靶子看起来还空。代价是：我们会变成"用已有工具打一个新靶子"，
> 这通常是 Findings 的量级，除非发现本身够意外。**

已存在的邻居：

| 工作 | 做了什么 |
|---|---|
| **Diffusion Lens** `2403.05846` | 用文本编码器**各层**的中间表示去引导扩散，看语义在编码器里怎么逐层成形 |
| **Follow the Flow** `2504.01137` | T2I 里**跨文本 token 的信息流**分析 |
| **Mechanistic Interpretability of T2I Diffusion Models**（**ACL 2026 Findings**） | 因果的、基于范数的可解释性框架，测**token 的真实贡献**而非原始注意力对齐 |
| Wang et al. 2026 | 用机制可解释性研究**空间关系**的生成；发现 T5 会把多个词项的信息**融进单个 token** |
| **Concept-Layer Alignment in T2V** `2605.25941` | T2V 里概念擦除**应该发生在哪一层** |
| **Analysis of Attention in Video Diffusion Transformers** `2504.10317` | 视频 DiT 的注意力分析 |
| Zero-shot editing via cross-attention `2404.05519` | T2V/V2T 交叉注意力各自编码什么 |

**空着的**：没有一篇专门问"**动作导致的状态变化语义**在 T2V 的哪一环丢失"。
Wang et al. 做的是空间关系，Concept-Layer 做的是概念擦除。

**但要诚实**：工具是现成的（Diffusion Lens 的逐层探针、范数式因果归因、
交叉注意力分析），我们是**换靶子**。这在 ACL 能不能进主会，取决于发现有多反直觉。
我们手上那条线索（**模型从第一帧就把末态画出来，把句子当静态场景描述读**）
如果能定位到具体层/模块，是有分量的；如果只能说"文本编码器里有、去噪器丢了"，
就偏薄。

---

# 两条路的对照

| | **A 方法** | **C 分析** |
|---|---|---|
| 动机 | 从 OSCBench 借 ✅ | 从 OSCBench 借 ✅ |
| 有没有清晰的成功判据 | ✅ 数字赢就是赢 | ⚠️ "发现够不够意外"是主观的 |
| **有没有现成 SOTA 可打** | ❌ **没有共享基准，基线要自己跑** | 不适用 |
| 单卡可行 | ⚠️ 16 天最后一跑 + 多轮迭代 | ✅ 子集探针 |
| 风险画像 | ❌ 打不过 = 零 | ✅ 定位到了就能写 |
| 投稿口 | **ACL + CVPR 双口**（文本侧投 ACL） | ACL（2026 theme 正是 Explainability） |
| 复现别人 | ❌ 支线甲三篇复现不了 | ✅ 不需要 |
| 位置 | ✅ **第一个在 OSCBench 上报数** | ⚠️ 用已有工具打新靶 |

**A 的最大优点是判据清晰、双投稿口；最大风险是"没有现成 SOTA 可打"意味着
基线成本要我们自己扛，而这一条在原计划里是被当成优势算的。**
