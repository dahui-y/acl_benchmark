# 方向提案：语法体与终结性在视频生成中的实现

调研日期：2026-08-03。
筛选标准：**空白 + 与 OSCBench 同等稳健（资源型交付、结果风险低）+ 标注成本可承受**。

---

## 1. 观察起点

OSCBench 的 1,120 条 prompt **全部是现在进行时**（`A man is chopping lettuce`）。
而"状态变化是否完成"在语言中正由**体（aspect）**编码。
该基准测量了状态变化，却从未操纵决定其完成与否的语法范畴。

## 2. 最小对立组（事件恒定，仅改体貌/时态标记）

| 变体 | 例 | 视频应当 | 语义范畴 |
|---|---|---|---|
| 进行体 | `A man is slicing an apple` | 有切的过程，末态可未完成 | imperfective |
| 完成体 / 一般过去 | `A man sliced an apple` | 末态必须已切好 | perfective |
| 完成结果态 | `A man has sliced an apple` | **只呈现结果，不应出现切的过程** | resultant state |
| 将行体 | `A man is about to slice an apple` | **状态变化不应发生** | prospective |
| 未遂 | `A man tried to slice an apple but failed` | 有动作、无结果 | 非蕴含终结 |
| 非终结（光杆复数） | `A man is slicing apples` | 无确定终结点 | atelic |

## 3. 评判协议（关键：只需两个二元问题）

对每个视频：

- **Q1**：视频中是否出现该动作的过程？（是 / 否）
- **Q2**：末帧中物体是否处于目标末态？（是 / 否）

两问构成 2×2 四格，**直接映射体貌语义**：

| | 有过程 | 无过程 |
|---|---|---|
| **末态达成** | perfective ✓ | resultant state ✓ |
| **末态未达成** | imperfective / 未遂 ✓ | prospective ✓ |

**主图**：六个变体在（动作出现率, 末态达成率）平面上的散点。
若六点聚成一团 → 模型对体貌完全失明。

> **判据必须锚在末帧状态**，不能是整段观感——5 秒视频难以从整体区分"进行"与"完成"。
> 该要求须写入标注指南，否则标注者一致性会垮。

## 4. 空白核验

| 核验点 | 结果 |
|---|---|
| 体貌 / 终结性 × 视频**理解** | **已被占**。`Perfect Times`（arXiv 2506.00928）：MCQA 基准，覆盖 perfectivity 与 telicity，英/意/俄/日四语，3,739 QA，源自 400 段 Charades 视频、157 个动词类；标注者 Fleiss **κ = 0.8**，gold 准确率 93.36%。`ViLMA`（ICLR 2024）：含 change-of-state 维度的语言学与时间锚定基准 |
| 体貌 / 时态 × 视频或图像**生成** | **空**。T2V-CompBench（1,400 prompt / 7 类）、GenAI-Bench（1,600 组合 prompt）、UniGenBench++（600 prompt / 27 子标准）、TC-Bench 均只覆盖属性、关系、计数、逻辑等组合性，**无一操纵时态或体** |
| Perfect Times 是否涉及生成 | **明确不涉及**（已逐项核实） |

**辩护先例**：OSCBench 自身即是"理解侧数据集 → 生成侧基准"的搬运
（HowToChange 是理解侧资源），并进入 ACL 2026 main。本方向执行同一动作。

## 5. 稳健性对照（对照 `idea_invariance.md` §7b 的三个不稳来源）

| 不稳来源 | 不变性方向 | 本方向 |
|---|---|---|
| **结果依赖** | **有**：模型若稳健则无发现 | **几乎没有**。训练字幕压倒性为进行时/一般现在时，模型忽略体貌近乎必然；且 `has sliced` 要求**生成状态而非事件**，是模型明显不具备的能力 |
| **指标单点故障** | **有**：全压在一个自动指标上 | **没有**。地面真值为人工二元判断，与 OSCBench 同构；MLLM 自动评测仅作可扩展代理并报告相关性 |
| **资源体量** | 60 base，单薄 | 600 prompt + 生成视频 + 协议 + 标注指南，与 OSCBench 同量级 |

**交付物独立于结果方向存在** → 满足"稳"的核心条件。

## 6. 成本估算

| 项 | 估计 |
|---|---|
| base 事件 | 100（可直接复用本仓库 `action_object_taxonomy/` 的动作—物体组合） |
| 体貌变体 | 6 |
| prompt 总数 | 600 |
| 模型数 | 4（建议 3 开源 + 1 闭源） |
| **视频总数** | **2,400** |
| 人工判断 | 2,400 × 2 问 × 3 标注者 = **14,400 次二元判断** |
| Prolific 成本 | 约 **$300–700** |

对比 OSCBench：840 视频 × 8 维 × 3 人 ≈ 20,000 次 **5 级 Likert** 打分。
本方案在判断数量相近的情况下，**单次判断的认知负荷低一个量级**，
且不需要领域专家或特定语言母语者（主实验用英语即可）。

可行性佐证：Perfect Times 在体貌判断上取得 Fleiss **κ = 0.8**（substantial），
说明该类判断本身具有高标注者一致性。

## 7. 必须处理的三个问题

1. **商业模型内部的 prompt 改写层**。Kling / Veo 使用 LLM 改写 prompt，很可能抹平体貌标记。
   → **主实验必须放在开源模型上**（Wan2.2、HunyuanVideo-1.5 直接消费原始 prompt）；
   闭源模型单列一节。"评测流水线的改写层擦除语法信息"本身即为有价值的发现，非纯损失。
2. **视频时长混淆**。5 秒视频难以从整体区分进行与完成 → 判据锚定末帧状态（见 §3）。
3. **"模型忽略体貌"可能被判为不意外**。
   → 重心放在 `has sliced`（要结果不要过程）与 `is about to`（要它不发生）两格：
   它们不是鲁棒性问题而是**能力**问题，且对应真实需求——生成一个**状态**而非一个**事件**，
   是视频编辑与数据合成中的常见场景。

## 8. 可选扩展（屋顶，非承重）

跨语言体貌标记：中文 了 / 着 / 过，俄语体貌对偶（Perfect Times 已覆盖俄语理解侧，可对照）。
砍掉不影响论文成立。

## 9. ACL 适配

体貌与终结性属形式语义学核心范畴。同脉络已被接受的工作：
OSCBench（ACL 2026 main）、TC-Bench（ACL Findings 2025）、Implicit-VidSRL（ACL 2025）、
ViLMA（ICLR 2024）、Perfect Times。

## 参考链接

- Perfect Times: https://arxiv.org/html/2506.00928
- ViLMA (ICLR 2024): https://cyberiada.github.io/ViLMA/ ; https://arxiv.org/pdf/2311.07022
- T2V-CompBench: https://arxiv.org/pdf/2407.14505
- GenAI-Bench: https://arxiv.org/abs/2406.13743
- UniGenBench++: https://arxiv.org/abs/2510.18701
- TC-Bench (ACL Findings 2025): https://aclanthology.org/2025.findings-acl.241/
