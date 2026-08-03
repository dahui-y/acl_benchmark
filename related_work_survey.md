# 相关工作调研 —— Idea 1 / Idea 2 的新颖性核验

调研日期：2026-08-03。目的：核验 `paper_ideas.md` 中 Idea 1（多步程序性指令下的实体状态追踪）
是否已被做过，以及能否达到 OSCBench 的 ACL main 标准。

> **结论：Idea 1 原样做达不到 ACL main 标准，存在实质撞车。Idea 2（结果义类型学）的空白确认成立。**

---

## 1. 与 Idea 1 直接竞争的工作

| 工作 | 出处 | 覆盖了 Idea 1 的哪一部分 | 冲突程度 |
|---|---|---|---|
| **RecipeGen** | ACM MM 2025 (arXiv 2506.06733) | 菜谱多步 T2I / I2V / **T2V** 基准。26,453 菜谱、196,724 图、4,491 视频；GPT-4o 清洗 + 逐步骤关键帧对齐（CLIP 相似度选帧）；指标 Goal Faithfulness / **Step Faithfulness** / **Cross-Step Consistency** / Ingredient Accuracy / Interaction Faithfulness；2,000 菜谱人工验证（步骤—图像对应、跨步骤视觉连续性）。失败模式已命名：Position Misalignment / Ingredient Missing / Ingredient Inconsistency | **高**。Idea 1 的"多步菜谱 + 逐步骤核对 + 跨步一致性"骨架被基本覆盖 |
| **SeqBench** | arXiv 2510.13042（2025-10，venue 未确认） | 序列叙事 T2V 基准。320 prompt / 32 子类 × 8 模型 = 2,560 标注视频；prompt 按 **Strictly Sequential / Flexible Order / Simultaneous** 三种时序关系分类，按 Single/Multi Subject × Single/Multi Action 分难度；Dynamic Temporal Graph 指标含**依赖过滤**（前置步骤未解析正确则后续问题不计分）；11 名标注者 × 每对 5 人，overall Spearman ρ=0.857 | **高**。我原方案里的 "first-failure step / 崩溃步分布" 实质等价于其依赖过滤机制 |
| **TC-Bench** | **ACL Findings 2025** (arXiv 2406.08656) | 时序组合性基准。prompt 显式描述初态与末态以消歧；三类变化：属性转变、物体关系、背景切换；配有真实对照视频，兼容 T2V 与 I2V；提出"组分转变完成度"指标，与人工判断相关性显著优于既有指标。结论：当代生成器只完成 **<20%** 的组合变化 | **高**（对 Idea 3 尤其）。"初态→末态是否完成"的指标已被做且发在 ACL |
| **YoCausal** | arXiv 2605.30346 | 视频扩散模型因果认知基准。用**时序倒放的真实视频**作天然反事实样本，无需合成数据；13 个开源模型；提出 Reverse Surprise Index（时间箭头感知）与 Causality Cognition Index（真实因果理解），并证明"感知时间箭头 ≠ 理解因果" | **高**。Idea 1 的扰动轴 D（不可逆性）被覆盖 |
| **EntityBench** | arXiv 2605.15199 | 长程多镜头视频生成的实体一致性。140 集 / 2,491 镜头，显式的**每镜头实体日程表**，同时追踪角色、物体、地点；难度分档至 50 镜头、跨镜头 22 个物体、间隔达 48 镜头 | 中。"跨步骤实体追踪"在多镜头设定下已有对应物 |
| **VBench-2.0** | arXiv 2503.21755 | 内在忠实性评测套件，已包含"中途修改属性（颜色/尺寸/纹理）"与空间指令重定位，采用 video-based multi-QA | 中。属性中途变化已被纳入通用套件 |

### 其他已被占位的扰动轴

- **否定**：`NEGATE`（arXiv 2603.06533）已构建 T2V 否定约束评测套件，8 个类别覆盖不同语言与动态失败模式。
- **反事实/违反物理**：`T2VPhysBench`（arXiv 2505.00337）已测"模型面对物理不可能的反事实 prompt 时会照做"。
- **理解端的隐含论元**：`Implicit-VidSRL` / Predicting Implicit Arguments in Procedural Video Instructions（ACL 2025, arXiv 2505.21068）——多模态烹饪程序中推断显式与隐含论元；论文明确指出程序性指令高度省略（"(i) add cucumber to the bowl (ii) add sliced tomatoes"，第二步的 where 论元需从上下文推断）。

## 2. 仍然空白的部分

1. **生成端的指称与省略**。`Implicit-VidSRL` 做的是**理解端**的 SRL 预测任务。
   没有任何工作把共指 / 代词 / 零形式 / 省略当作**视频生成端的受控自变量**。
   RecipeGen 与 SeqBench 明确**未**测试步骤顺序置换、共指或省略（已逐篇核实）。
2. **同内容的最小对立顺序置换**。SeqBench 的时序类别是 prompt 的**分类标签**，
   不是同一内容的 A→B / B→A 配对置换，无法分离"遵循指令"与"回退菜谱先验"。
3. **结果义的跨语言词汇化**（Idea 2）。见下节。

## 3. Idea 1 的重构方案（若坚持做）

原样做 = Findings 量级，且大概率吃到 "incremental over RecipeGen / SeqBench" 的致命意见。
要进 main，必须把支点从"更长的多步 prompt"换成一个**语言学命题**：

> **T2V 模型不做指代消解；它对 prompt 执行的是名词袋（bag-of-nouns）检索。**

**验证设计（最小对立三元组）**，指称对象恒定、名词袋变化：

| 变体 | 示例 | 名词袋 | 指称对象 |
|---|---|---|---|
| 显式名词 | 把黄瓜切好放进碗里，然后把**黄瓜**搅拌 | {黄瓜, 碗} | 黄瓜 |
| 代词 | 把黄瓜切好放进碗里，然后把**它**搅拌 | {黄瓜, 碗} | 黄瓜 |
| 零形式 | 把黄瓜切好放进碗里，然后搅拌 | {黄瓜, 碗} | 黄瓜 |
| 干扰项对照 | 把黄瓜切好放进**碗**里，然后把它搅拌 | 加入竞争先行词 | 需消歧 |

若生成结果随表层形式而非指称关系变化，命题成立。
RecipeGen 与 SeqBench 的 prompt 全部使用显式名词，因此**无法反驳该主张**。

即便如此，Idea 1 的相邻工作过于密集，审稿人容易将其读作 SeqBench 的一次消融。

## 4. Idea 2 的空白核验（结论：成立）

多语言 T2V 的现有工作仅三类，**均不涉及语义组合**：

| 工作 | 做了什么 | 是否触及结果义 |
|---|---|---|
| 多语言计数（arXiv 2504.04051） | prompt 语言变化时的数量约束遵循 | 否 |
| **T2VTextBench**（arXiv 2505.04946） | 视频内**文字渲染**能力，73 prompt/模型，含"Multilingual (Chinese)"类别 | 否 |
| **MAVEN**（arXiv 2605.16716） | 多文化 T2V，243 prompt × 4 条精修流水线 = 972 视频，覆盖中/美/罗马尼亚文化 | 否 |
| 现有粗粒度观察 | "Kling / Wan2.1 / Hailuo 在中文最好，Sora / Pika 在英文最好" | 否 |

**没有任何工作研究结果义由动结式补语（切开 / 切碎 / 煮熟）承载 vs 由单动词承载（slice / mince / cook）
时，模型状态变化能力的差异。** Talmy 类型学（verb-framed vs satellite-framed）在 T2V 评测中完全未被使用。

## 4b. Idea 2 能否达到 OSCBench 的 ACL main 标准（2026-08-03 追加核验）

### 补充调研

| 核验点 | 结果 |
|---|---|
| **Manner/result complementarity**（Levin & Rappaport Hovav：一个动词不能同时词汇化方式与结果）是否被用于生成模型评测 | **未被用过**。仅在发展语言学的测量工具中出现（arXiv 2605.16654）。理论文献：Levin, *Lexicalized Meaning and Manner/Result Complementarity* |
| 动词语义类评测是否已有 | 有，但全在**理解端**：SVO-probes（421 动词，图文模型）、ActionBench / Paxion（Action Antonym + Video Reversal 探针）、Verbs in Action（arXiv 2304.06708）。**生成端空白** |
| 语言变异 × 多模态生成 是否有先例 | 有：**DialectGen**（arXiv 2510.14949），英语方言 × 图像**与视频**生成，**词汇层面**，含人工评测。**不涉及动词语义、结果义构式或跨语言类型学** → 是体裁先例，不是竞争者 |
| 多语言 T2I 现有工作 | AltDiffusion、PEA-Diffusion 等，关注的是**模型多语言能力**（编码器替换、知识蒸馏），不是语义组合探针 |

### 判定

- **Idea 2 原始形式（多语言状态变化对比）：Findings 量级。**
  主张停留在"不同语言表现有差异"，是描述性的，且天然被"多语言能力弱"这一混淆变量解释掉。
- **Idea 2 重构版：可达 ACL main。**

### 重构：把支点从 Talmy 类型学换成方式/结果互补性

三个可证伪假设，**全部是语言内部对比**，因此"多语言能力弱"无法解释：

| 假设 | 内容 | 性质 |
|---|---|---|
| **H1** | 结果动词（melt / break / open，目标状态被词汇化）的 OSC 分数显著高于方式动词（stir / wipe / scrub，目标状态需推理） | 观察 |
| **H2** | 中文动结式可把方式动词补足出结果义（搅**匀** / 擦**净** / 切**开**）；若 H1 成立，加结果补语应**修复**该差距 | **干预** |
| **H3** | 英语 resultative construction（`wipe the table clean`）在**句法层面**提供同样的结果义。若中文补语能修复而英语构式不能 → 模型只处理词汇层面的结果义，不处理句法层面的组合 | **双重解离** |

跨语言（日/西/法/韩）仅用于类型学推广，是**屋顶而非承重结构**。

### 对"需要多语言母语标注者"这一风险的重估

- 主实验 H1/H2/H3 只需**中文 + 英文** → 对 NUS / SMU / 复旦是零成本，且**足以独立成篇**。
- 类型学推广节可缩至每语言 60–80 prompt、仅标 OSC 准确性单一维度、2 名标注者。
  西班牙语走 Prolific；日语难招可换**法语**（同为动词框架语言，同样缺乏 resultative 构式，论证效力等价）或韩语。
- **该节被砍掉不影响论文成立**——这是相对原方案最重要的降险改动。

### 必须最先排除的风险：地板效应

OSCBench 报告英文下 OSC accuracy 仅 0.38–0.79。若模型在 `slice watermelon` 上本就做不对，
则"它区分不了 切开 / 切碎"毫无信息量——所有对比被压到地板上，测不出目标变量。

> **Pilot 协议（写任何 prompt 之前先做）**：先在英文基线上筛选动作—物体对，
> 只保留模型 OSC 准确性 ≥ 0.7 的组合进入主实验，确保存在 headroom。
> 这是整个方案最脆弱的一环。

### 其余防守点

| 风险 | 防守 |
|---|---|
| 文本编码器混淆（Wan2.2 用 **umT5**（多语言）；HunyuanVideo 用 MLLM 编码器；Kling / Hailuo 中文原生；Veo / Sora 英文为主） | 报告每个模型的文本编码器类型；构造"中文原生模型 × 英文原生模型"的 2×2，不能只测一侧 |
| 切开 / 切碎 的语料频率差异 → 可能只是频率效应而非组合语义 | 用中文语料统计频率，报告频率与分数的偏相关 |
| "这是 CV 问题，不是语言学发现" | H3 直接反驳：同一语言内部、无多语言因素，仍然失败 |
| 借 OSCBench Figure 5（rolling/heating 高，peeling/coating/pressing 低）作为动机 | **只能当线索，不能当证据**——peeling 本身也蕴含结果状态，事后附会会被抓。必须独立做词汇语义分类 |

### 为什么重构版可能高于 OSCBench

OSCBench 的贡献本质是"我们发现 X 很难"（**描述性**）。
重构后的 Idea 2 是"我们预测并验证了 X 在何时难、何时可被修复，并给出机制解释"（**解释性**），
且包含一个**干预实验**而非纯观察。ACL 审稿人对解释性贡献的评分系统性更高。
前提：pilot 先排除地板效应。

## 5. 修正后的推荐

**主推 Idea 2 的重构版（方式/结果互补性 + 结果补语干预），并把重构后的 Idea 1 降级为其中一条扰动轴。**

> 注意：Idea 2 的**原始形式**（纯多语言状态变化对比）只有 Findings 量级，
> 必须按第 4b 节换支点后才具备 ACL main 的竞争力。

理由：中文的零形式共指在英文中根本无法构造，因此"指称"与"结果义词汇化"这两条线
在类型学框架下天然合并为同一篇论文，而不是两篇。合并后的核心主张：

> 结果义与指称在不同语言中由不同的形式手段承载；T2V 模型对承载这些语义的形式成分
> （补语、零形式）系统性失明，因此其状态变化能力在英文评测上被系统性高估。

该主张避开了 RecipeGen / SeqBench / TC-Bench / YoCausal / NEGATE 的全部射程。

## 6. 场地适配佐证

- OSCBench —— ACL 2026 main（本仓库论文）
- TC-Bench —— ACL Findings 2025
- Implicit-VidSRL —— ACL 2025

视频生成评测在 *ACL 系列有明确先例，前提是主张必须锚定在语言现象上而非视觉质量上。

---

## 参考链接

- RecipeGen: https://arxiv.org/abs/2506.06733
- SeqBench: https://arxiv.org/html/2510.13042
- TC-Bench: https://aclanthology.org/2025.findings-acl.241/ ; https://arxiv.org/abs/2406.08656
- YoCausal: https://arxiv.org/html/2605.30346
- EntityBench: https://arxiv.org/html/2605.15199v1
- VBench-2.0: https://arxiv.org/html/2503.21755v1
- NEGATE: https://arxiv.org/html/2603.06533v1
- T2VPhysBench: https://arxiv.org/html/2505.00337v1
- Implicit-VidSRL: https://arxiv.org/abs/2505.21068
- OpenPI-C: https://aclanthology.org/2023.findings-acl.452/
- DialectGen: https://arxiv.org/pdf/2510.14949
- Verbs in Action: https://ar5iv.labs.arxiv.org/html/2304.06708
- Paxion / ActionBench: https://arxiv.org/pdf/2305.10683
- Levin, Lexicalized Meaning and Manner/Result Complementarity: https://web.stanford.edu/~bclevin/barcel11rev.pdf
- Manner/result 测量工具: https://arxiv.org/html/2605.16654
- AltDiffusion: https://arxiv.org/pdf/2308.09991
- T2VTextBench: https://arxiv.org/html/2505.04946v1
- MAVEN: https://arxiv.org/pdf/2605.16716
- 多语言计数: https://arxiv.org/pdf/2504.04051
