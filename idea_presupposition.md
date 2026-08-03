# 方向提案：预设投射在视觉生成中的实现

调研日期：2026-08-03。第五轮调研。

> ## ⚠️ 本方向已被降级（2026-08-03 修正）
>
> 后续核实 NEGATE 的实证发现后，判定本方向存在**结构性缺陷**，不再作为主推：
>
> 1. **生态效度接近于零**。真实用户不会写 `The man didn't stop slicing the apple`。
>    OSCBench 记录的失败影响每一个教学视频合成 / 机器人 / 世界模拟用例；
>    本方向的失败影响的用例接近不存在。
> 2. **最可能的实验结果恰使研究失效**。NEGATE 已证明模型的主要失败模式是
>    **"inclusion despite negation"——照样生成被否定的内容**（"no vehicles" 仍生成车辆）。
>    则对 `didn't stop slicing`，模型会生成切削动作，**看似投射成功，实则只是忽略了 `didn't`**。
>    §5 解离表中"两组都出现 → 结论无效"一行，按 NEGATE 的证据**正是最可能发生的情况**。
>
> **关键区别**：方向 ④（体貌）的缺陷"gold 由作者规定"**可靠设计修复**；
> 本方向的缺陷是**结构性的**，无法通过设计消除。
>
> **保留价值**：相位动词（stop / begin / finish）本就同时是体貌算子，
> 可作为体貌矩阵中的一行保留，但**不作为主张的承重结构**。
>
> 当前推荐见 `idea_aspect.md`。

**方法变更**：前四轮均为"先提出点子、再验证"，因而每次都在验证阶段被打回。
本轮改为**先绘制已占领地图、再定位空白**，并一次性核验全部候选。

**筛选标准（严格）**：空白 + 与 OSCBench 同等稳健 + 达到 ACL main 标准。

---

## 1. 本轮报废的候选

| 候选 | 判定 | 依据 |
|---|---|---|
| 空间参照框架（intrinsic vs relative "to the left of"） | **已被占，死** | `GenSpace`（arXiv 2505.24870）明确划分 Egocentric / Allocentric / Intrinsic 三个子域；另有 `SpatialBench-UC`（arXiv 2601.13462）、`Rel3D`（arXiv 2012.01634，刻意不指定参照框架以反映人类使用分布） |
| 量化辖域 / 分配性（every, each；distributive vs collective） | 空白，但**不推荐** | **地板效应**：模型本就不会计数——见 *Text-to-Image Diffusion Models Cannot Count, and Prompt Refinement Cannot Help*（arXiv 2503.06884）。无法区分"不懂分配性"与"不会数数" |

## 2. 存活方向：预设投射

### 核心主张

> **模型只生成 prompt 中被断言（asserted）的内容，忽略被预设（presupposed）的内容。**

### 杀手锏测试

```
The man didn't stop slicing the apple.
```

按预设的定义，"他之前在切"这一内容**在否定环境下依然成立**
（negation test 是预设的经典判别式），因此视频中必须出现切的动作。
模型若将 `didn't` 当作"删除"信号，则会生成不含切削动作的视频。

> **关键优势：gold 由语义学推导得出，而非作者规定。**
> 这修复了体貌方案（`idea_aspect.md`）最要害的缺陷——
> "`has sliced` 不应出现切的过程"是作者的规定，懂形式语义的审稿人可直接打穿；
> 而预设的投射行为是学界公认的判别式，不可辩驳。

### 触发语类型（提供广度）

| 触发语类型 | 例 | 视频必须出现 |
|---|---|---|
| 状态改变动词 | `The man didn't stop slicing the apple` | 切的动作 |
| 事实性动词 | `The chef regrets slicing the apple` | 切好的苹果 |
| 迭代副词 | `The man sliced the apple again` | 先前的切 |
| 断裂句 | `It was the chef who sliced the apple` | 有人切了苹果 |
| 定指描述 | `the chef's sliced apple` | 存在切好的苹果 |
| 焦点小品词 | `Even the chef sliced the apple` | 他人也切了 |

**设计模板**：可直接借用 `PROPRES`（arXiv 2312.08755）的
**6 触发语 × 5 环境**（否定、疑问、条件、情态、态度谓词）矩阵设计
——既获得学界认可的结构，又天然解决"轴太窄"的问题。

## 3. 空白核验

| 核验点 | 结果 |
|---|---|
| 预设 × **文本** | **已被充分占领**：PROPRES（6 触发语 × 5 环境的投射性研究）、CONFER（arXiv 2506.06133）、Adverbial Presupposition Triggers（ACL 2018, P18-1256）。**全部为纯文本 NLI** |
| 预设 × 视觉**理解** | 仅 `CP-Bench`——LVLM 的反事实预设问答，属**幻觉缓解**研究，非生成 |
| 预设 × 视觉**生成** | **空**。已核对 Awesome-Evaluation-of-Visual-Generation 清单、GenAI-Bench、T2I-CompBench、VBench-2.0（18 个子维度全清单）、UniGenBench++、DrawBench、TC-Bench、NEGATE，**无一涉及** |
| 与 `NEGATE` 的关系 | **方向相反**。NEGATE 测"不要出现 X"；本方向测"**否定句中 X 仍必须出现**"。模型若把否定当删除，恰在此处系统性失败 |

## 4. 三个标准的判定（**已被顶部修正推翻，以下为降级前的原始判断，保留备查**）

| 标准 | 判定 | 理由 |
|---|---|---|
| **空** | **是** | 五轮调研中信心最高的一次。预设文献全在文本侧，视觉生成侧完全空白 |
| **稳** | **四个方向中最高** | 资源型交付（触发语 × 环境 × 事件套件）；判断二元客观；结果风险低（NEGATE 已证明模型连基础否定都处理不好，投射失败近乎必然）；**gold 可推导而非规定** |
| **ACL 标准** | **是** | 预设与投射属语义/语用学核心范畴，ACL 有成熟引用脉络（PROPRES、CONFER、P18-1256） |

### 与前四个方向的稳健性对比

| 方向 | 空 | 稳 | 致命问题 |
|---|---|---|---|
| Idea 1 多步实体状态追踪 | 否 | — | RecipeGen / SeqBench / TC-Bench / YoCausal 撞车 |
| Idea 2 结果义类型学 | 是 | 中 | 需多语言母语标注者；地板效应 |
| 不变性 | 是 | 低 | 结果依赖；指标单点故障；seed 方差可能吞掉效应 |
| 体貌 / 终结性 | 是 | 中高 | **gold 由作者规定**；轴过窄 |
| **预设投射** | 是 | ~~高~~ **低** | 原以为 §5 可通过设计消除；后经核实为**结构性缺陷**，见顶部修正 |

## 5. 必须写进设计的陷阱：正确结果可能来自错误原因

若模型对 `didn't` 完全无反应——照常生成切削动作——则"预设存活"会**偶然正确**，
但原因是它忽略了否定，而非理解了投射。

**双向对照是必需的，不是可选的。** 必须配一组**真·否定**对照：

```
The man is not slicing the apple.     → 不应出现切的动作
```

### 判据（解离表）

| 预设组（`didn't stop slicing`） | 真否定组（`is not slicing`） | 结论 |
|---|---|---|
| 出现切 | 不出现切 | 模型正确处理投射（预期概率低） |
| 出现切 | **出现切** | **模型忽略否定**；预设组的"成功"无效 |
| **不出现切** | 不出现切 | **模型把否定当删除 —— 预期的主发现** |
| 不出现切 | 出现切 | 异常，需个案分析 |

**漏掉该对照，全文结论即为假。** 这是整个设计中最关键的一条。

## 6. 评判协议

每个视频两个二元问题（与 `idea_aspect.md` 同构）：

- **Q1**：视频中是否出现被预设的事件（如切削动作）？
- **Q2**：末帧中物体是否处于被预设蕴含的状态？

二元、客观、无需领域专家或特定语言母语者。

## 7. 成本估算

| 项 | 估计 |
|---|---|
| 触发语类型 | 6 |
| 嵌入环境 | 5 |
| 每格事件数 | 8（可复用本仓库 `action_object_taxonomy/`） |
| prompt 总数 | 6 × 5 × 8 = **240**，加真否定对照与基线共约 **600** |
| 模型数 | 4（3 开源 + 1 闭源） |
| 视频总数 | **≈ 2,400** |
| 人工判断 | 2,400 × 2 问 × 3 标注者 = **14,400 次二元判断**，Prolific 约 **$300–700** |

## 8. 沿用自前几轮的设计要求

- **商业模型的 prompt 改写层**会抹平否定与触发语 → 主实验放开源模型，闭源单列并标注 confound（见 `idea_aspect.md` §7d P1）
- **文本编码器前提探针**同样适用：先验证编码器是否区分 `didn't stop slicing` 与 `stopped slicing`。
  纯文本、不烧算力、两种结果都可发表（见 `idea_aspect.md` §7c）
- 所有变体的视频时长与帧率固定一致

## 参考链接

- PROPRES: https://arxiv.org/pdf/2312.08755
- CONFER: https://arxiv.org/pdf/2506.06133
- Adverbial Presupposition Triggers (ACL 2018): https://aclanthology.org/P18-1256.pdf
- Presupposition (SEP): https://plato.stanford.edu/entries/presupposition/
- NEGATE: https://arxiv.org/html/2603.06533v1
- GenSpace: https://arxiv.org/pdf/2505.24870
- SpatialBench-UC: https://arxiv.org/html/2601.13462
- Rel3D: https://arxiv.org/pdf/2012.01634
- T2I diffusion cannot count: https://arxiv.org/pdf/2503.06884
- Awesome-Evaluation-of-Visual-Generation: https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation
