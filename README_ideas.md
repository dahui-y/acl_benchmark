# 选题调研索引与最终结论

基于对 `OSCBench`（ACL 2026 main）的分析，经五轮调研后的选题决策记录。

**筛选标准**：空白 + 与 OSCBench 同等稳健 + 达到 ACL main 标准 + 标注成本可承受（无标注团队）。

---

## 最终推荐

> ### 预设投射在视觉生成中（`idea_presupposition.md`）
>
> **核心主张**：模型只生成 prompt 中被断言的内容，忽略被预设的内容。
>
> **杀手锏**：`The man didn't stop slicing the apple` —— 按预设的定义，
> "他之前在切"在否定环境下依然成立，视频中必须出现切削动作。

## 五轮取舍对照

| 方向 | 文档 | 空？ | 稳？ | 结论 |
|---|---|---|---|---|
| ① 多步程序性指令的实体状态追踪 | `paper_ideas.md` §Idea 1 | **否** | — | **淘汰**。被 RecipeGen（ACM MM'25）、SeqBench、TC-Bench（ACL Findings'25）、YoCausal 四面撞车 |
| ② 结果义类型学（manner/result 互补性） | `related_work_survey.md` §4b | 是 | 中 | **受资源约束否决**。需中/日/西母语标注者；且英文基线存在地板效应 |
| ③ 语义等价改写下的不变性 | `idea_invariance.md` | 是 | **低** | **淘汰**。结果依赖、指标单点故障、seed 方差可能吞掉效应 |
| ④ 语法体 / 终结性 | `idea_aspect.md` | 是 | 中高 | **降级并吸收进 ⑤**。gold 由作者规定（可被形式语义审稿人打穿）；轴过窄 |
| ⑤ **预设投射** | `idea_presupposition.md` | **是** | **高** | **推荐** |

## 为何 ⑤ 胜出

| 标准 | 依据 |
|---|---|
| **空** | 预设文献（PROPRES、CONFER、ACL 2018 adverbial triggers）**全在纯文本侧**；视觉侧仅 CP-Bench（理解侧问答）。已逐个核对 Awesome-Evaluation-of-Visual-Generation 清单、GenAI-Bench、T2I-CompBench、VBench-2.0（全 18 子维度）、UniGenBench++、DrawBench、TC-Bench、NEGATE —— 生成侧无人涉足 |
| **稳** | 交付物是资源（触发语 × 环境 × 事件套件），不依赖结果方向；判断二元客观，无需专家或母语者（约 $300–700）；结果风险低（NEGATE 已证明模型连基础否定都处理不好）；**gold 由语义学推导而非作者规定** |
| **ACL 标准** | 预设与投射属语义/语用学核心，有成熟 ACL 引用脉络 |

## 前几轮的成果如何继承

方向 ④ 并非丢弃，而是被**吸收**：相位动词（stop / begin / finish）
**既是体貌算子、也是预设触发语**，直接构成触发语矩阵的一行。

以下设计要求由前几轮继承，适用于 ⑤：

| 要求 | 出处 |
|---|---|
| 主实验放开源模型，避开商业模型的 LLM prompt 改写层；闭源单列并标注 confound | `idea_aspect.md` §7d P1 |
| 判据锚在「首帧 + 过程 + 末帧」三元组，而非整段观感 | `idea_aspect.md` §7d P2 |
| 所有变体的视频时长与帧率固定一致 | `idea_aspect.md` §7d P2 |
| 文本编码器前提探针先行（纯文本、不烧算力、两种结果都可发表） | `idea_aspect.md` §7c |
| 机制定位 + 免训练差向量干预，作为"发现不够意外"的解药 | `idea_aspect.md` §7c/§7d P3 |

## 唯一可能翻车之处

**双向对照是必需的，不是可选的。**

若模型对 `didn't` 完全无反应、照常生成切削动作，则"预设存活"属**偶然正确**
——原因是忽略了否定，而非理解了投射。必须配真否定组
（`The man is not slicing the apple` → 不应出现切削）做解离。
**漏掉该对照，全文结论即为假。** 解离表见 `idea_presupposition.md` §5。

## 尚未完成的核验（需自行补上）

1. **UniGenBench++ 的 27 条子标准清单**：arXiv 摘要页与项目页均取不到，
   需下载 PDF 核对是否有涉及否定或预设的子项。属 T2I 而非 T2V，碰撞程度弱，但洞仍在。
2. **CP-Bench**：仅依据二手描述判定为理解侧问答，未读原文。
3. **Perfect Times 的场地**：arXiv 2506.00928 至今仅 v1、无场地标注（已核实），
   但若后续中稿，方向 ④ 相关的 related work 需加强区分。

## 下一步（两条路，都不贵）

- **验证路线**：先跑文本编码器探针，确认编码器是否区分
  `didn't stop slicing` 与 `stopped slicing`。纯文本、不需标注、两种结果都能写成一节。
- **构造路线**：按 PROPRES 的 6 触发语 × 5 环境矩阵，
  配合本仓库 `action_object_taxonomy/`，生成约 600 条 prompt 及双向对照。

## 文档清单

| 文件 | 内容 |
|---|---|
| `paper_ideas.md` | 第一轮：OSCBench 配方拆解 + 五个初始方向（已标注修订） |
| `related_work_survey.md` | 第二、三轮：Idea 1 / 2 的新颖性核验与判定 |
| `idea_invariance.md` | 第四轮：不变性方向 + 稳健性评估（已淘汰） |
| `idea_aspect.md` | 第四轮：体貌 / 终结性方向 + 前提探针 + 三问题解法（已吸收进 ⑤） |
| `idea_presupposition.md` | **第五轮：预设投射方向（当前推荐）** |
