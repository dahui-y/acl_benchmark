# 方向重调研：T2I / 蕴含数量（entailed count）

调研日期：2026-08-05。触发条件：用户要求重新调研方向，可以是 T2I，
**但"中 ACL 的标准"和"中 ACL 的稳"两条不能丢，且不做人工评测**。

> **结论：推荐 T2I × 蕴含数量（distributivity 及同类现象）。**
> 空白经 9 轮检索确认成立；关键是它把 T2V 体态路线上那个补不上的洞
> （**没有已知正例 ⇒ 敏感度不可测**）补上了——本方向的正确答案是**算出来的**，
> 而且**测量仪器本身可以用现成标注数据（COCO/LVIS）校准，不新增任何标注**。

---

## 0. 先把约束写死

"和 OSCBench 一样稳" + "不做人工评测"这两条同时成立，只有一条路：

> **正确答案必须是程序可判定的。**

OSCBench 的稳，很大一部分来自它的人工评测和人—机相关性那一节。
我们不做人工评测，就必须拿**可计算的真值**去顶那一格：检测器计数、OCR、包围框算术。
不是"换个自动指标"，是**换一类真值**。

T2V 体态路线在这一条上死掉了，死因很具体（见 `RISKS.md`）：

| | T2V 体态 | 本方向 |
|---|---|---|
| 正确答案来源 | MLLM 判官的判断 | 检测器数出来的**整数** |
| 已知负例 | 有（`other_verb` 的末态） | 有 |
| **已知正例** | **没有**——设计里没有任何东西能认证"这段视频确实达成了末态" | **有**——显式数量条件的答案先验已知 |
| 敏感度 | **测不出来**，只能写进 limitation | 可测 |
| 仪器校准 | 无外部参照 | COCO/LVIS 的实例数标注就是参照 |
| 主结果形态 | 条件间的**差值**（试跑 CI [0.85, 2.67]，含 1） | 对程序真值的**绝对准确率** |

最后一行是要害。体态路线的主结果是一个需要置信区间才能解释的差值，试跑打出来的
区间含 1；本方向的主结果是"模型在 X% 的分配性条件下产出了被蕴含的数量，
而在数量被明说时是 Y%"——**绝对数，不需要靠差值显著性活着**。

---

## 1. 已被占位的部分（逐条核实）

| 现象 | 占位工作 | 覆盖了什么 |
|---|---|---|
| **显式数词计数** | **GenEval**（arXiv 2310.11513）、**T2I-CompBench++**（TPAMI 2025，generative numeracy）、**GeckoNum**（arXiv 2406.14774）、**Make It Count**（arXiv 2406.10210）、**T2ICountBench**（arXiv 2503.06884，"Cannot Count"）、**QUOTA** | prompt 里**写明**数字："three red apples"、"between three and five"、"few / several"、"more apples than pears" |
| **否定** | LMD Negation、NEG-TTOI、NegBench、**NEGATE**（T2V） | "no elephant in the room" |
| **组合性 / 属性绑定 / 空间关系** | T2I-CompBench、ConceptMix、GenAI-Bench、VISOR、GenEval | 颜色—物体绑定、左右上下、共现 |
| **逻辑等价 prompt 的鲁棒性** | **MetaLogic**（arXiv 2510.00796，ICFEM 2025） | 交换律、结合律、**逻辑分配律** A∧(B∨C) ≡ (A∧B)∨(A∧C) |
| **结构歧义（理解端）** | **LaViSA**（arXiv 2606.19552） | 七类：VP / PP 附着、指代、省略、形容词辖域、动词辖域、连词辖域。**是 VLM 理解任务，不是生成**；七类里**没有**分配性或量化辖域（已核实全文） |
| **推理驱动生成** | **R2I-Bench**（EMNLP 2025 main）、WISE、T2I-ReasonBench | 常识 / 数学 / 逻辑 / 因果 / 数值推理，QA 式指标 |
| **失败模式清单** | **FineGRAIN**（arXiv 2512.02161，27 个失败模式）、Right Looks Wrong Reasons（arXiv 2511.10136 综述） | "Counts or Multiple Objects" 一格 = 精确数词；两篇都**不涉及** each/every/together 与分配性 |
| **文本端的分配性** | **DistNLI**（BlackboxNLP 2022, arXiv 2209.04761）、**"Respectively" 论文**（arXiv 2305.19597） | 纯文本 NLI / 纯文本推理，**没有视觉，没有生成** |

### 两个必须在 related work 里主动拆掉的同名地雷

1. **MetaLogic 的 "distributivity" 不是我们的 distributivity。**
   它指命题逻辑的分配律（合取对析取），我们指复数谓述的分配读法
   （"each" 让谓语作用到每个个体上）。同名不同物，
   但审稿人扫一眼会以为撞车——**必须在正文里点名区分，不能等 rebuttal**。

2. **GeckoNum 用的是人工标注**：52,721 张图 × 5 人 = 479,570 条标注。
   这不是撞车，这是**对照组**——同一类问题，别人靠标注做，我们靠程序真值做，
   而且我们能给出仪器的校准数。方法论差异本身就是贡献的一部分。

### 检索到的直接证据：这块确实空着

- GeckoNum 全文核实：12 个 prompt 类型，全部是**显式数词 1–10** 加 many/few/no，
  **不系统覆盖** each / every / together / a pair of，也没有分配—集体的对立。
- Right Looks Wrong Reasons 综述（覆盖 2022–2025 的 15 个基准）明确：
  计数一支的覆盖是"exact numerals / bounded ranges / vague quantifiers /
  relational counts / multi-type"，**"不显式处理 each/every/together，
  也不分析分配与集体的语义解读；焦点是基数控制，不是量化辖域或约束现象"**。
- 文本端有 DistNLI（BlackboxNLP 2022）和 respectively 论文——
  **现象在 NLP 里是立住的、可引的**，但**没有视觉端、没有生成端的对应物**。

这是最理想的空白形状：**现象已被 NLP 承认为重要，模态完全空着。**

---

## 2. 空白是什么：数量不是读出来的，是算出来的

所有计数基准的共同前提：**目标数字写在 prompt 里**。模型只要把那个数字复制到画布上。

我们要问的是另一件事：**数字在 prompt 里根本没出现，它是从语义算出来的。**

```
The three girls are each holding a balloon.        → 气球 = 3
The three girls are holding a balloon together.    → 气球 = 1
The three girls are holding a balloon.             → 歧义（1 或 3 都可接受）
```

三句话里"气球"前面永远是 "a"。**没有任何数字可以复制。**
3 这个数是 "each" 的分配算子和 "three girls" 一起算出来的。

这条线立刻推出一整族现象，共性是**同一条**：数量被蕴含，从不被陈述。

| 族 | 例 | 被蕴含的数 |
|---|---|---|
| **分配 vs 集体** | "the three girls are each holding a balloon" / "…together" | 3 / 1 |
| **相互（reciprocal）** | "the two boys are shaking hands with each other" | 一次握手，四只手，不是两次 |
| **词汇复数 / 成对** | "a pair of shoes"、"a trio of musicians"、"twins" | 2 / 3 / 2 |
| **respectively 构式** | "Anna and Bob are holding a red and a blue balloon respectively" | 2 个气球，且颜色—人绑定 |
| **部分—整体量化** | "two of the three apples have been bitten" | 3 个苹果，其中 2 个有咬痕 |
| **否定辖域下的量化** | "none of the four cups is full" | 4 个杯子，0 个满 |

六族都满足：**答案是整数，整数是算出来的，整数可以被检测器数出来。**

---

## 3. 为什么这个设计能"稳"——锚点结构

体态路线上我们学到的教训是：**没有锚点的差值没法解释**。这里锚点是天然的，
而且比 T2V 那边强，因为它给出的是**已知正例**。

一个 item 的条件组（同一 seed，跨条件共用，与 T2V 侧同一套做法）：

| 条件 | prompt | 蕴含的气球数 | 作用 |
|---|---|---|---|
| `dist` | "The three girls are **each** holding a balloon." | 3 | 主条件 |
| `coll` | "The three girls are holding a balloon **together**." | 1 | 主条件（对立） |
| `bare` | "The three girls are holding a balloon." | 1 或 3 | 欠定，看模型的默认读法 |
| `explicit_n` | "Three girls and **three** balloons." | 3 | **上锚 / 已知正例**：模型到底能不能画出 3 个气球 |
| `explicit_1` | "Three girls and **one** balloon." | 1 | **下锚 / 已知正例** |

`explicit_n` 是整条路的地基：如果模型在 `explicit_n` 上就画不出 3 个气球，
那 `dist` 上的失败不是分配性失败，**只是计数失败**，不能算到我们头上。
主结果因此不是裸准确率，而是**条件化的**：

> 在模型**能**正确画出 N 个物体的那些 item 上（`explicit_n` 正确），
> `dist` 的正确率是多少？

这一条同时回答了体态路线上那个"37 个 item 只有 9 个带信息"的问题——
这里"带信息"的子集是**由已知正例定义的**，不是事后挑的。

**难度分档**（对应 OSCBench 的 difficulty regimes）：目标数 N ∈ {2, 3, 4, 5}。
已知计数准确率随 N 超线性衰减，所以 N 本身就是天然的难度轴，
且能画出"分配性失败 vs 计数失败"随 N 分离的曲线。

---

## 4. 仪器怎么校准——不用一条新标注

这是最关键的一节，也是与 GeckoNum / FineGRAIN 的方法论分水岭。

**a) 检测器在计数上的既有成绩，可引。**
GenEval 用 Mask2Former（COCO 训练）做检测，计数一项与人工标注的
**Cohen's κ = 0.823**（5 折交叉验证），整体与人工一致率 83%，
在标注者全体一致的图上 91%（人—人一致率 88%）。
它们同时报告了具体的工程细节：同类多实例时检测器会吐大量低置信框，
所以**计数任务把置信阈值从 0.3 提到 0.9**，且 NMS 无助于此。
——这是我们可以直接继承的既有实践，不是我们自己发明的可疑做法。

**b) 我们自己的仪器，用现成标注数据现测。**
COCO / LVIS 的实例标注里本来就有"这张图有几个 X"。
把检测器放到 val 集上、限定到我们用的物体类和数量区间（2–5），
测出计数准确率与 κ。**零新增标注，且这是已知正例——
体态路线上补不上的那一格，在这里是白送的。**

**c) 域偏移，双检测器交叉。**
COCO 是真实照片，我们的图是生成的。这个偏移是真的，要写进 limitation，
同时用两个独立检测器（Mask2Former + OWLv2 / Grounding DINO 开放词表）
在**我们自己的生成图上**互测一致率；主结果可另报一份"两器一致子集"的版本。

**d) 判官不看句子——这条从 T2V 侧原样搬过来。**
检测器只被问"这张图里有几个 balloon"，它从头到尾不知道 prompt 是
`dist` 还是 `coll`。条件差异不可能从仪器侧泄漏。这一点比 MLLM 判官更彻底：
检测器**在结构上**不可能看到句子。

---

## 5. 算力：约束基本消失

| | T2V 体态路线 | 本方向 |
|---|---|---|
| 单样本 | 480p 121 帧，**208 s/条**（实测） | 1024² 单图，**5–20 s/张** |
| 全量 | 9 天，且只有 1–2 个模型 | **1–2 天**，4–6 个模型 |
| 显存 | 24 GB 紧到要开 offload + VAE tiling | SDXL/SD3.5 宽裕；FLUX.1-dev 12B 开 offload 可跑 |
| 能不能测多模型 | 不能（这是审稿人一定会问的） | **能**——横向对比是基准论文的基本盘 |

粗算：500 prompt × 4 seed × 4 模型 = 8,000 张，按 10–20 s/张 = **22–44 小时**。
再加 1–2 个 API 模型（在 Mac 上跑，走外网）做闭源对照。

体态路线上"单卡 4090 只能跑一个模型的 480p"是一条要写进 deviation 段落去解释的伤；
这里它不存在。

---

## 6. 与 OSCBench 的骨架对齐

| OSCBench | 本方向 |
|---|---|
| 从现有数据集出发 | 从现有物体类（COCO/LVIS 词表）与既有计数基准的物体词表出发 |
| human-in-the-loop 抽象 | **程序化生成 prompt**：模板 × 物体 × N × 条件，全部可复现 |
| 难度分档 | 目标数 N ∈ {2,3,4,5}；现象族 6 类 |
| 人工 + MLLM 评测 | **检测器程序判定**（无人工），MLLM 作第二仪器交叉 |
| 人—机相关性 | **仪器—真值相关性**：检测器 vs COCO/LVIS 金标注（κ），外加双检测器交叉 |

对应关系是逐格的，且每一格我们都有替代物——这正是"不做人工评测"能顶得住的形式。

---

## 7. 还没核实的、以及真实风险

**已核实（第一步、第二步）：**

- ~~检测器准确率~~ —— **第一步做完，过。** 见 `calibrate/RESULTS.md`：
  LVIS 干净金标上，N∈{3,4,5} 留出法不筛类的 N-vs-1 有序率 **0.957**，加权 κ **0.803**，
  已核实为零的一档 **0.987**，两器一致时 **0.981**（覆盖 88.7%）。
  代价：N=2 弃用（余量只有 1，落在计数噪声里）。
- ~~别名封口~~ —— **第二步做完，空白确认成立。** 见下节。

**仍未核实（留给第三步 pilot）：**

1. 六族现象里，**reciprocal 与 part-whole 两族的真值是否真的单值**。
   "shaking hands with each other" 画面上到底该有几只手、几次握手，
   可能存在合法变体——**凡真值不唯一的族直接砍掉**，
   宁可只留 3 族也不留一族说不清的。
2. 现有 T2I 模型对 "each" / "together" 是否**完全无反应**。
   如果 `dist` 与 `coll` 的数量分布完全相同，那是干净的系统性失败（可发表）；
   但要先确认这不是因为**两句话都退化成同一张图**
   （那就变成"模型忽略副词"的平凡结论）。
3. **两器在生成图上的一致率**。真实照片上是 0.887，生成图上是多少只能实测——
   这是域偏移那条 limitation 唯一能给出数字的地方。

---

## 7b. 第二步：别名封口（2026-08-06，已完成）

把"看起来空"推到"已核实为空"。做法是**读发布的 prompt 全表，不读论文摘要**，
再按六种别名各扫一轮。

### 逐条核实的结果

| 核实对象 | 做法 | 结果 |
|---|---|---|
| **GeckoNum** | 取回**全部 12 个 prompt 模板** | `<num> <noun>` / `There <verb> <num> <noun>` / additive / attribute-color / attribute-spatial / approx / fractional / part-whole。**没有一个含 each / every / together / apiece / respectively / a pair of；数字永远显式**（数词或 many/few/no/as many as） |
| **T2I-CompBench++ numeracy** | 取回构造方法与例子 | 1000 条，30% 单物体 / 30% 双物体 / 40% 多物体，**数量 1–8 全部显式数词**，模板由 ChatGPT 生成 150 类物体随机组合。**不含隐式量化词** |
| `distributive predication` | 检索 | 只打到形式语义学本行的文献（*Natural Language Semantics* 2026 的 distributive kind predication），**无视觉、无生成** |
| `cumulative reading` / `plural quantification` | 检索 | 同上，无对应物 |
| `quantifier scope` + 生成 | 检索 | 只打到 MetaLogic（命题逻辑分配律，见下） |
| `implicit / entailed count` | 检索 | 打到 T2I-ReasonBench 的 "implicit meaning"，但那是**世界知识**（成语、实体、科学常识），不是语义蕴含 |
| `a pair of` / reciprocal `each other` | 检索 | 无对应物 |
| **理解端（VLM）** | 检索 | LaViSA 七类不含分配性（已核实）；VAGUE / ClearVQA 是**一般性歧义**，不是复数谓述 |

### 唯一一处提到量词的工作，恰好是助力

**DyEval**（arXiv 2411.15509）观察到 "SDXL 和 SD3 在量词与代词这类语言成分上持续失败"。
但它是**人在环的交互式探索框架**：没有固定 prompt 集，没有指标，**明确依赖人工评测者**。

所以它不是竞品，而是**独立旁证**——别人用交互式探针撞到了这个现象、点名了它，
但没有人把它做成可复现的评测。这一句在引言里比我们自己论证有力。

### 三个必须在 related work 里主动拆掉的同名地雷（比原来多一个）

| 撞名的词 | 别人指的 | 我们指的 |
|---|---|---|
| **MetaLogic 的 "distributivity"** | 命题逻辑分配律 A∧(B∨C) ≡ (A∧B)∨(A∧C) | 复数谓述的分配读法 |
| **T2I-ReasonBench 的 "implicit"** | 世界知识（成语、实体、科学） | 语义**蕴含**的数量 |
| **"one-and-only alignment"**（arXiv 2606.30262） | 唯一性实体（埃菲尔铁塔、蒙娜丽莎）的反事实生成 | 与单复数无关 |

第三个是这一轮新查出来的：名字听起来像"单数对齐"，实际毫不相干，但审稿人扫标题会误会。

### 结论与它的边界

> **空白确认成立。** 两个最近的基准的 prompt 全表已逐条读过，
> 六种别名各扫一轮，生成端与理解端都扫了，没有任何一条打出竞品。

边界要说清楚：**检索证不了否命题**，同期投稿的风险永远存在。
这一轮做到的是——**不再依赖论文摘要的转述，而是核对了发布的 prompt 表本身**，
这和第一轮"看起来空"是两回事。这也是当初给体态方向做过的同一级别的核验。

**真实风险：**

| 风险 | 说法 | 应对 |
|---|---|---|
| "单一现象太窄" | 只做 each/together 撑不起一篇 main | 用**蕴含数量**做统摄构念，6 族（砍剩 4 族也行）+ 4 档难度 + 锚点结构 |
| "检测器不可信" | 生成图上的计数误差没保证 | §4 的 b/c 两条：COCO 金标注校准 + 双检测器交叉 + 一致子集副表 |
| "和 MetaLogic 撞车" | 同名 distributivity | §1 已拆：命题逻辑分配律 ≠ 复数谓述分配读法，正文点名 |
| "模型压根画不出 3 个物体" | 效应被计数失败淹没 | `explicit_n` 锚点把两种失败分开，主结果条件化在锚点通过的子集上 |
| "纯 T2I 评测能进 *ACL 吗" | venue fit | R2I-Bench 在 **EMNLP 2025 main**；OSCBench（T2V）在 ACL 2026 main；ACL 2026 Findings 有 T2I 创造性评测。先例存在 |

---

## 8. 体态路线上哪些资产可以直接搬过来

不是从零开始。已经写好并验证过的东西里，下面这些是**模态无关**的：

- **共用 seed 的 item 内对照**：同一 item 的所有条件共用 seed，
  已实测 `max |Δpixel| = 0.000000`，唯一差异源是文本条件。同一套逻辑，T2I 上更便宜。
- **锚点结构**（上锚必须变、下锚必须不变）与效应/地板比，以及 bootstrap CI。
- **断点续跑按 prompt 而非坐标匹配**（`generate/generate.py` 里那个教训）。
- **仪器不看句子**这一条设计原则。
- **`RISKS.md` 的写法**：七条审稿意见 × 状态 × 修法 × 代价。

T2V 那条线上真正废掉的只有：Wan/Hunyuan 的模型注册表、抽帧、MLLM 判官的三问 prompt。
**方法论骨架全部保留。**

---

## 9. 建议的下一步

1. **pilot（半天）**：4 族 × 5 个物体 × N∈{2,3} × 5 条件 × 2 seed ≈ 400 张，
   一个模型（SDXL，最快）。目的只有三件事：
   真值是否单值、检测器数得准不准、`dist`/`coll` 有没有可测的差别。
2. **仪器校准（一小时）**：检测器在 COCO val 上、限定我们的物体类与 N∈{2..5}
   的计数准确率与 κ。这一步**先于**任何生成，因为它决定整条路成不成立。
3. 两步都过了，再谈全量。任一步不过，这份文档里的失败判据是明写的，
   不会再出现"跑完才发现区间含 1"的情况。

**第 2 步应该排在第 1 步前面**——它最便宜，且它是地基。

---

### Sources

- [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)
- [T2I-CompBench++](https://arxiv.org/abs/2307.06350)
- [Evaluating Numerical Reasoning in Text-to-Image Models (GeckoNum)](https://arxiv.org/html/2406.14774v2)
- [Make It Count: Text-to-Image Generation with an Accurate Number of Objects](https://arxiv.org/pdf/2406.10210)
- [Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help](https://arxiv.org/pdf/2503.06884)
- [MetaLogic: Robustness Evaluation of Text-to-Image Models via Logically Equivalent Prompts](https://arxiv.org/html/2510.00796)
- [LaViSA (structural ambiguity, VLM understanding)](https://arxiv.org/abs/2606.19552)
- [R2I-Bench: Benchmarking Reasoning-Driven Text-to-Image Generation (EMNLP 2025 main)](https://aclanthology.org/2025.emnlp-main.636/)
- [FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with VLM Judges](https://arxiv.org/html/2512.02161v1)
- [Right Looks, Wrong Reasons: Compositional Fidelity in Text-to-Image Generation](https://arxiv.org/html/2511.10136v1)
- [Testing Pre-trained Language Models' Understanding of Distributivity (DistNLI, BlackboxNLP 2022)](https://aclanthology.org/2022.blackboxnlp-1.26/)
- [What does the Failure to Reason with "Respectively" Tell Us about Language Models?](https://arxiv.org/pdf/2305.19597)
- [Evaluation of Text-to-Image Generation from a Creativity Perspective (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.26/)
