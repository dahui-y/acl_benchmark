# 方向提案：语义等价改写下的生成不变性

调研日期：2026-08-03。

**约束前提**：无标注团队、无特定语言母语标注者。
因此选题标准为：**空白 + 达到 ACL main 标准 + 正确性由构造决定（不需要人工打分）**。

---

## 1. 核心命题

> **T2V 模型的输出由 prompt 的表层形式驱动，而非其语义内容。**

判据（配对检验）：

```
Δ_改写(同义改写对之间的输出距离)  >  Δ_seed(同一 prompt 不同随机种子之间的输出距离)
```

若成立 → 模型对表层形式敏感而对语义不敏感。
Δ_seed 是**自动获得的噪声下界**，构成内置对照组，无需任何人工标注。

## 2. 双重解离设计

| 组 | 操作 | 模型应当 | 示例 |
|---|---|---|---|
| **保义组** | 词汇同义替换 | 不变 | `slicing an apple` → `cutting an apple into slices` |
| | 被动化 | 不变 | `A man is slicing an apple` → `An apple is being sliced by a man` |
| | 话题化 / 分裂句 | 不变 | `It is an apple that the man is slicing` |
| | 关系从句 | 不变 | `The man, who is in the kitchen, is slicing an apple` |
| | 定指性 / 数 | 不变 | `a man` → `the man` |
| | 冗余修饰（不改变事件） | 不变 | 附加不影响事件语义的状语 |
| **改义组（对照）** | 论旨角色互换 | **改变** | `A man slices an apple` → `An apple slices a man` |
| | 施受倒置（可逆事件） | **改变** | `The dog chases the cat` → `The cat chases the dog` |

**双重解离**：保义组敏感 + 改义组不敏感 → bag-of-words 假说在生成端确证。
单看任一组都可被"模型能力弱"解释掉；两组合起来不能。

## 3. 为什么标注成本近乎为零

| 环节 | 是否需要人工 | 量级 |
|---|---|---|
| 生成正确性的 gold label | **不需要**——测的是"变不变"，不是"对不对" | 0 |
| 噪声下界 Δ_seed | **不需要**——同 prompt 换种子，全自动 | 0 |
| 核验改写确实保义 | 需要，但是**文本**判断（"这两句意思一样吗"），非视频判断 | 作者自行完成，1–2 天 |
| 验证自动指标效度 | 需要，二元配对（"这两个视频描绘同一事件吗"） | ~200 次判断，Prolific 数十美元 |

对比：OSCBench 为 840 视频 × 8 维度 × 3 评分者 ≈ **20,000 次 Likert 打分**。
本方案 ≈ **200 次二元判断**，相差两个数量级，且不需要领域专家或特定语言母语者。

## 4. 空白核验

| 核验点 | 结果 |
|---|---|
| T2V 的保义改写不变性 benchmark | **不存在**。现有相关工作均为**方法**：POS（arXiv 2311.00949）、VPO（arXiv 2503.20491）用改写去**提升**生成质量，不测不变性 |
| paraphrase robustness | 已有，但限于**纯文本 LLM**（ParaConsist 等 900 prompt / 150 base × 5 变体设计），未进入生成模型 |
| 改义方向（bag-of-words / 论旨角色） | **已被占**。理解侧：CLIP 的 "the grass is eating the horse" 问题、ARO、Winoground。生成侧：GenAI-Bench / VQAScore（ECCV 2024，1,600 组合 prompt）、T2I-CompBench。→ **改义只能作对照组，不能作卖点** |
| 种子方差 | 已被用作 diversity / 稳定性指标（如 Dynamics Perspective, NeurIPS 2024），但**未被用作语言效应的噪声下界**。该用法为新 |

**卖点必须落在"不变性"这一侧**，改义组仅用于构成解离。

## 5. 评测指标设计（技术难点）

直接用像素距离或 CLIP 距离会被 seed 噪声淹没。建议路线：

1. 用 MLLM 从视频抽取**结构化事件表示** `(agent, action, patient, target_state, scene)`；
2. 在**符号层面**计算两组视频的表示距离，而非像素层面；
3. 分别对保义组与改义组、以及 seed 基线计算该距离，做配对显著性检验。

优势：**抽取比打分容易得多**，从而绕开 OSCBench 自陈的 MLLM 打分不可靠问题
（其 consistency 维度 human–MLLM Kendall τ 仅 0.317，而人–人为 0.501）。

## 6. 成本估算

| 项 | 估计 |
|---|---|
| base prompt | 60 |
| 每 base 的变体 | 6（5 保义 + 1 改义对照） |
| 每变体种子数 | 3 |
| 模型数 | 3（建议：2 开源 + 1 闭源子集） |
| **总视频数** | **≈ 3,240**（约为 OSCBench 6,720 的一半） |

开源侧：HunyuanVideo-1.5 为轻量模型；Wan2.2 可量化运行。
闭源侧（Kling / Veo）按 API 计费，仅跑子集用于验证结论可推广。

> **真正的瓶颈从标注转移到了算力。** 若算力也受限，可进一步压缩：
> 减至 40 base × 5 变体 × 3 seed × 2 开源模型 ≈ 1,200 视频。
> 该设计的统计效力来自**每 base 内部的配对比较**，而非 base 的数量，因此缩减 base 数
> 对结论的损害小于缩减变体数或种子数。

## 7. 必须主动承认的问题

1. **"保义"并不严格成立**。被动化改变信息结构与焦点；话题化改变主位。
   必须**按改写类型分层报告**，不能笼统声称语义等价。
   主动写入 Limitations 反而加分——审稿人一定会挑这一点。
2. **自动指标的效度需被验证**，否则整篇论文悬空。这是 §3 中那 200 次人工判断的用途。
3. **改义组可能存在天花板效应**：若模型对 `An apple slices a man` 生成了合理的"苹果切人"，
   说明它其实处理了论旨角色，则解离不成立。需先做小规模 pilot 确认改义组确有区分度。

## 7b. 与 OSCBench 的稳健性对比（关键评估）

**结论：本方向不如 OSCBench 稳。它是高方差方案——上限更高，但存在真实的失败概率。**

OSCBench 稳的原因：交付物是**资源**。无论 6 个模型跑出什么数字，都留下 1,120 条 prompt、
一套动作/物体分类体系、840 个人工标注视频。**结果方向不影响论文成立。**

本方向的四个不稳来源：

| # | 风险 | 说明 |
|---|---|---|
| 1 | **结果依赖（最根本）** | 若 Δ_改写 ≈ Δ_seed（模型确实不敏感），则无发现。"模型稳健"是弱得多的结论，难支撑 main。探针式论文的固有风险，资源型论文没有 |
| 2 | **seed 方差可能吃掉效应** | T2V 换种子会改变整个场景构图，Δ_seed 可能接近天花板，导致检验**先天欠功效**。缓解：**固定种子做改写对比**，利用扩散模型在相同噪声下对相近文本嵌入产生相近构图的性质。**但此性质必须由 pilot 验证，不能假定** |
| 3 | **指标是单点故障** | 全文压在"结构化事件抽取距离"一个指标上。OSCBench 有人工评分兜底，本方案**无兜底**，指标一噪则全盘皆空 |
| 4 | **资源体量小** | 60 个 base prompt，期待"benchmark"的审稿人会认为单薄 |

软风险："模型对措辞敏感"是从业者已相信的事。若发现不够反直觉，易被判为确认已知常识。
OSCBench 的发现同样不算意外，但有资源作缓冲垫，本方案没有。

### 根本困境：稳 / 无标注 / 空白 —— 三选二

| 组合 | 可行性 | 代价 |
|---|---|---|
| 稳 + 空白 | OSCBench 的配方 | **必须大规模标注** |
| 无标注 + 空白 | 本方向 | **不稳，结果依赖** |
| 稳 + 无标注 | 只能做正确性可程序化验证的题（计数、文字渲染等） | **空白没了，均已被占** |

这是约束组合本身的紧致性，不是选题质量问题。

### 若"稳"是第一优先级：改为 resource-first

交付一套**不变性评测套件**（改写类型体系 + prompt 集 + 生成视频 + 指标实现 + 模型排行），
使交付物**独立于结果方向**存在——即使模型表现出不变性，仍交付了可复用的鲁棒性评测协议。

代价：prompt 集扩至 200+ base、模型 5–6 个、指标需扎实效度验证
→ 算力成本回到 OSCBench 量级。**约束从标注转移到算力，并未消失。**

### 建议：先跑 pilot，由数据决定

几十个视频即可，一次跑完三件事：

1. **固定种子下，改写是否产生可测的输出差异**（若否，方向直接毙掉——最要紧的一条）
2. Δ_seed 的实际量级
3. 改义组（`An apple slices a man`）是否有区分度

pilot 通过 → 本方向期望值高于继续找新题；不通过 → 省下全部投入。
**不建议在 pilot 之前承诺方向。**

## 8. ACL 适配

主张为"模型是否表征语义而非表层形式"——即组合性与 bag-of-words 假说，
这是 NLP 的核心问题，而非画质问题。与 OSCBench（ACL 2026 main）、
TC-Bench（ACL Findings 2025）、Implicit-VidSRL（ACL 2025）同属一条已被接受的脉络。

## 参考链接

- POS (Prompt Optimization Suite for T2V): https://arxiv.org/pdf/2311.00949
- VPO (Aligning T2V with Prompt Optimization): https://arxiv.org/html/2503.20491v1
- ParaConsist / paraphrase-induced output-mode collapse: https://arxiv.org/html/2605.04665v2
- On Robustness and Reliability of Benchmark-Based Evaluation of LLMs: https://arxiv.org/pdf/2509.04013
- VQAScore / GenAI-Bench: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01435.pdf
- Evaluation of T2V Models: A Dynamics Perspective (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/c6483c8a68083af3383f91ee0dc6db95-Paper-Conference.pdf
- Artifact-Bench: https://arxiv.org/pdf/2605.18984
