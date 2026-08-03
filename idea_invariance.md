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
