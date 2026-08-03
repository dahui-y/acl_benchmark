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

## 5. 修正后的推荐

**主推 Idea 2（结果义类型学），并把重构后的 Idea 1 降级为其中一条扰动轴。**

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
- T2VTextBench: https://arxiv.org/html/2505.04946v1
- MAVEN: https://arxiv.org/pdf/2605.16716
- 多语言计数: https://arxiv.org/pdf/2504.04051
