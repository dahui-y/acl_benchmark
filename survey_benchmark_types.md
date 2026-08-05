# 不限方向的重调研：什么**类型**的基准符合这四条标准

调研日期：2026-08-05。触发条件：用户取消方向限制，问"什么类型的 benchmark 更符合我的标准"。

四条标准（用户历次明确）：
1. 中 ACL main 的**标准**（对标 OSCBench）
2. 一样的**稳**
3. 方向**未被占用**，且要在动手前核实
4. **不做人工评测 / 不需要标注**

隐含硬约束：单张 RTX 4090，不能加卡；生成机在国内无外网，Mac 在英国有外网；无标注队伍。

> **结论：先定类型，再定题目。**
> 四条标准联立，能活下来的只有一类：
> **最小对立 × 答案由程序算出 × 校验器能用现成金标注标定。**
> 在这一类里，按"占位程度 × 可行性 × ACL 契合度"排序，
> **T2I × 蕴含数量**仍然排第一；语音韵律排第二但最好的构念已被 CoNLL 2025 拿走；
> 纯文本最小对立（CxMP 那一型）最便宜、形态已被 ACL 2026 main 验证，但竞争最狠。

---

## 1. 从约束反推类型

标准 4（不做人工评测）是最硬的一条，它直接决定"正确答案从哪来"。
穷举一下，真值只有五个可能来源：

| # | 真值来源 | 例 | 能不能用 |
|---|---|---|---|
| A | **由 prompt 的语义算出**（蕴含） | "each ... a balloon" ⇒ 3 | ✅ 强 |
| B | **继承现成金标注** | COCO 实例数、UniMorph 屈折表、ToBI 断句 | ✅ 强（别人付过标注的钱） |
| C | **由构造保证**（逆向任务） | 从已知结构出发生成表层形式 | ✅ 强 |
| D | **执行/解析可验** | 代码跑单测、SVG 解析坐标 | ✅ 强 |
| E | **只有条件间的一致性** | 语义等价的两个 prompt 应该给同样的图 | ❌ **弱** |

**E 就是体态路线最后退化成的样子，它为什么弱要说清楚：**
E 没有已知正例 ⇒ 敏感度不可测 ⇒ 主结果只能是差值 ⇒ 差值要靠置信区间活着 ⇒
跑完才知道区间含不含 1。这不是运气不好，是**这一类真值的结构性缺陷**。

A、C、D 都需要一个**校验器**去读模型的输出；校验器本身的误差就成了新的软肋。
解法是把 B 用在校验器身上：**校验器的准确率在现成金标注上实测**。
于是得到应该走的组合：

> **A 或 C 给题目的真值，B 给校验器的标定。**

这就是"类型"。再叠上一条使基准好看的形态学要求：

> **最小对立**（同一 item 只改一个语言学维度，其余全同，共用 seed）。
> 它把"模型差"和"模型对这个维度不敏感"分开，这是 OSCBench 的 difficulty regime
> 和我们锚点结构的共同祖先。

**三条合起来 = 可投的类型：最小对立 × 程序真值 × 金标注标定的校验器。**

---

## 2. ACL 是不是真收这一型——有直接证据

**ACL 2026 main 收了 `CxMP: A Linguistic Minimal-Pair Benchmark for Evaluating
Constructional Understanding in Language Models`**，基于构式语法（let-alone、
caused-motion、双及物）。这说明：

- **"语言学构念 + 最小对立 + 无标注"就是一个被认证过的 ACL main 形态**，不需要论证；
- 且 ACL 2026 的评测/基准类论文数量与 2025 持平（增长的是训练/对齐），
  说明这条赛道没有被挤爆到发不进去。

参照系：ACL 2026 主会 12,145 投 / ~2,400 收，19%。OSCBench（T2V）在主会。
EMNLP 2025 main 收了 R2I-Bench（纯 T2I 评测）。**venue fit 不是风险点。**

---

## 3. 候选类型排序（各自的占位核实）

### ① T2I × 蕴含数量 —— 推荐

真值 = A（语义算出）；校验器标定 = B（COCO/LVIS 实例数金标注）。

- **占位**：显式数词计数被 GenEval / GeckoNum / T2I-CompBench++ / Make It Count /
  T2ICountBench 占满；**数字不写在 prompt 里、由量化语义算出**这一支空着
  （9 轮检索，详见 `survey_t2i_entailed_count.md`）。
- **可行**：8,000 张图 ≈ 22–44 小时，4–6 个模型横向对比跑得起。
- **可扩**：新发现一条能显著加厚的轴——**数量分类词（numeral classifier）**。
  "一**只**鸟" vs "一**群**鸟"：数词相同，个体化不同，蕴含的数量不同。
  多语 T2I 评测目前全是**把英文基准翻译过去**（m-GenEval、UniGenBench++、SoS、
  NeoBabel），而**翻译在结构上不可能测到分类词——英文原句里没有分类词可翻**。
  这个论证很硬，可以直接写进 related work。
- **弱点**：结构上弱于 OSCBench 一处——最终指标没在**我们自己的生成图上**做过人工核验
  （COCO 是真实照片，有域偏移）。

### ② 语音 / TTS × 韵律—句法 —— 次选，但最好的构念已被拿走

真值 = C（构造：句法歧义句的两个读法各自要求不同的韵律断句）；
校验器 = 强制对齐的停顿时长 + 基频轨迹；标定 = B（Boston Radio News 的 ToBI 金标注）。

这一型的**校验精度最高**——停顿是毫秒，F0 是赫兹，没有检测器漏检那种噪声。
算力几乎为零（TTS 模型比视频模型小两个数量级，4090 绰绰有余）。

- **占位（要害）**：`A Linguistically Motivated Analysis of Intonational Phrasing
  in TTS: Revealing Gaps in Syntactic Sensitivity`（arXiv 2505.22236，**CoNLL 2025**）
  做的正是"花园径句 / 附着歧义句上 TTS 的韵律断句敏感度"，结论是系统依赖逗号这种表层线索。
  **最显然的那个构念被拿走了。**
- `EmergentTTS-Eval`（NeurIPS 2025 D&B）覆盖 syntactic complexity，
  但用的是 **LALM model-as-judge**——**程序化韵律校验这条路仍然空着**，
  这是可以攻的缝，但要换构念（对比焦点、信息结构、辖域）而不是重做附着歧义。
- **弱点**：新基建；\*ACL 的语音轨较小，审稿人可能推去 Interspeech；
  "有没有正确断句"是梯度量，阈值要标定（好在 ToBI 金标注能标定）。

### ③ 纯文本最小对立（CxMP 型）—— 最便宜，但竞争最狠

真值 = A/C；无需校验器（答案是 log-prob 或选择题，精确）。
算力几乎为零，API 就能跑完。形态已被 ACL 2026 main 验证。

- **弱点**：**12,145 篇投稿都在这个池子里**。构念的"为什么重要"要过的门槛远高于多模态。
  文本端的分配性已有 DistNLI（BlackboxNLP 2022），respectively 也有人做过。
- **战略含义**：选多模态**不是题目选择，是避开竞争的选择**。同样一个语言学构念，
  放在生成模态上，赛道厚度差一个量级。

### ④ 已核实占位、建议划掉的

| 类型 | 状态 |
|---|---|
| **文本→SVG / 矢量图生成** | ❌ 挤满：VGBench（**EMNLP 2024 main**）、SGP-GenBench、SVGenius（ACM MM 2025）、UniSVG、LLM4SVG、IntroSVG、GeoSVG-RL |
| **多语 T2I（翻译式扩展）** | ❌ 挤满：m-GenEval / m-DPG（NeoBabel）、UniGenBench++、SoS、When Cultures Meet。**但分类词那一支不在其中**（见 ①） |
| **比较级 / 最高级 × 包围框算术** | 🟡 没查到专门基准，但**单独做太窄**，"incremental over 空间关系"几乎必吃。**作为 ① 的一个族并进去** |
| **代码 / text-to-SQL 执行验证** | ❌ 红海 |
| **布局→图、场景图→图** | ❌ 7Bench、OverLayBench、HRS-Bench 等 |

---

## 4. 一条可复用的通用配方（下次换题目也能用）

把 ① 抽象出来，得到一个可以套在任何构念上的模板：

```
取一个语言学构念 X，使得：
  (1) X 决定输出中某个【可被程序数出来/量出来】的量 q
  (2) q 在 prompt 里【从不出现】——它是算出来的，不是抄来的
  (3) 存在一个"把 q 明写出来"的条件作为【上锚】，它的答案先验已知
  (4) 校验器在【某个现成金标注数据集】上能测出自己的准确率
```

- (2) 是**新颖性**的来源：所有既有基准都在测"抄得准不准"。
- (3) 是**稳**的来源：它是已知正例，同时把"构念失败"和"底层能力失败"分开，
  并且**事先**定义了哪些 item 带信息（体态路线上"37 个只有 9 个带信息"是事后才发现的）。
- (4) 是**免标注**的来源，也是唯一能顶替 OSCBench 人工评测那一格的东西。

四条同时满足，才既新又稳还不用标注。**缺 (3) 就退化成 §1 的 E 类——体态路线的死法。**

---

## 5. 建议

**留在 ①，但按这份文档加厚**：蕴含数量作统摄构念，下辖分配/集体、相互、成对词汇、
respectively、部分—整体、比较级六族，再加**数量分类词的跨语言轴**。
理由是四条标准里它是唯一能同时满足的，而且**两个致命前提一天之内能证伪**：

1. 检测器在 COCO val（限定我们的物体类、N∈{2..5}）上的计数准确率与 κ —— **1 小时**
2. 别名封口 + 读 GeckoNum / T2I-CompBench++ 发布的 prompt 全表 —— **半天**
3. pilot 400 张，验真值单值性与 dist/coll 可测差异 —— **半天**

**② 留作备选**，触发条件是第 1 步不过：若检测器在同类小物体上就是数不准，
整个"检测器当校验器"的路都塌，那时应该换到毫秒/赫兹级的声学校验，
并且换构念（焦点、信息结构），避开 CoNLL 2025 那篇。

---

### Sources

- [ACL 2026 Accepted Main Conference Papers](https://2026.aclweb.org/program/accepted_papers/)（CxMP 最小对立基准）
- [ACL 2026 acceptance statistics](https://aiweekly.co/alerts/acl-2026-accepts-2400-main-track-papers-from-12145-submissions)
- [R2I-Bench (EMNLP 2025 main, 纯 T2I 评测)](https://aclanthology.org/2025.emnlp-main.636/)
- [VGBench (EMNLP 2024 main)](https://aclanthology.org/2024.emnlp-main.213.pdf) / [SVGenius (ACM MM 2025)](https://arxiv.org/pdf/2506.03139) / [SGP-GenBench](https://www.emergentmind.com/topics/sgp-genbench)
- [A Linguistically Motivated Analysis of Intonational Phrasing in TTS (CoNLL 2025)](https://arxiv.org/pdf/2505.22236)
- [EmergentTTS-Eval (NeurIPS 2025 D&B)](https://arxiv.org/abs/2505.23009)
- [NeoBabel (m-GenEval / m-DPG)](https://arxiv.org/pdf/2507.06137) / [UniGenBench++](https://arxiv.org/abs/2510.18701) / [SoS](https://arxiv.org/pdf/2601.16803)
- [GenEval](https://arxiv.org/abs/2310.11513) / [T2I-CompBench++](https://arxiv.org/abs/2307.06350)
- [7Bench](https://arxiv.org/pdf/2508.12919) / [OverLayBench](https://arxiv.org/html/2509.19282) / [HRS-Bench](https://openaccess.thecvf.com/content/ICCV2023/papers/Bakr_HRS-Bench_Holistic_Reliable_and_Scalable_Benchmark_for_Text-to-Image_Models_ICCV_2023_paper.pdf)
