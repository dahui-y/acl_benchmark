# 数量（object count）格子：在位者与「SOTA 到天花板」的距离

查证日期：2026-08-18。触发：风格线按事先写死的判据判负后，选题准则从
「找空格子」改成 **「SOTA 离天花板还有多远」** —— 这个量在花任何 GPU 之前
就能从已发表的数上读出来。

标注约定（沿用全项目）：【一手】= 论文正文/表格本身；【二手】= 综述页或
他人转述，**不能当判据用**；【未查】。

---

## 一、事先写死的判据（在查之前就说好的，原话）

> 如果 SOTA 绝对值仍在 40–60%，天花板 100%，**这个格子的 headroom 比风格线
> 大一个数量级**，值得投；如果已经到 85%+，那这一格也满了。

---

## 二、绝对数（不是相对增益）

### 表 A — CountCluster 的协议【一手】
（arXiv 2508.10710 Table 1；prompt 模板 `A photo of [N] [object]`，N=2–10；
评测器 = CountGD）

| 方法 | 骨干 | Accuracy | MAE | RMSE |
|---|---|---:|---:|---:|
| Naive | SD2.1 | 24.56 | 1.990 | 3.127 |
| Counting-Guidance | SD2.1 | 24.95 | 1.885 | 2.982 |
| CountCluster | SD2.1 | **33.33** | 1.791 | 3.282 |
| Naive | SDXL | 27.88 | 5.045 | 9.302 |
| Zafar et al. | SDXL | 25.93 | 1.815 | 2.924 |
| CountGen (Make It Count) | SDXL | 46.20 | 1.661 | 3.562 |
| CountCluster | SDXL | **55.75** | 0.821 | 1.849 |

耗时【一手】：SD2.1 9.14 s/图，SDXL 36.38 s/图，无外部模块、无训练。

### 表 B — CountDiffusion 的协议【一手】
（arXiv 2505.04347 Table I；评测器 = Grounded SAM）

| 方法 | CoCoCount (2–10) | GPTSingleCount (2,3,5,7,10) | GPTMultiCount (多类，各 1–3) |
|---|---:|---:|---:|
| SDXL | 34 | 21 | 5 |
| CountGen | 51 | 35 | — |
| CountDiffusion (SDXL) | **59** | **45** | **31** |
| Pixart-Σ | 40 | 22 | 23 |
| CountDiffusion (Pixart-Σ) | **60** | 37 | **43** |

> **最破的一格是多类**：SDXL 在 GPTMultiCount 上只有 **5%**，SOTA 也才 31–43%。

### 表 C — T2ICountBench【一手】
（arXiv 2503.06884，标题就是 "Cannot Count, and Prompt Refinement Cannot Help"；
15 个模型，含闭源；**评测是 5 名研究生人工判**）

- 最好：Imagen-3 **43%**、Gemini 2.0 Flash 39%；最差：Recraft V3 25%、SD3.5 26%
- **没有任何一个模型平均超过 50%**
- 按难度：1–5 个 60–80% ／ 6–10 个 **10–30%** ／ 11–15 个 **<10%**
- 四种 prompt 改写策略**全部把准确率改低了**：42% → 26 / 23 / 34 / 20

### 表 D — CountLoop（多智能体迭代，非单次前向）【一手】
（arXiv 2508.16644；评测器 = GroundingDINO；A100 80G，10/50/100 实例
分别 28.4 / 75.3 / 120.1 秒）

| 方法 | COCO-Count Acc | T2I-CompBench Acc | CountLoop-S (30–200 实例) | CountLoop-M |
|---|---:|---:|---:|---:|
| SDXL | 42.13 | 44.00 | 24.49 | 67.25 |
| FLUX | 54.19 | 57.00 | 29.59 | 78.00 |
| CountGen | 50.00 | 19.78 | 41.40 | 45.33 |
| CountLoop | 93.33 | 78.50 | **55.00** | 83.67 |

CountLoop 的 93.33 是**反复重生成直到检测器点头**换来的，不是单次前向；
它自己的高实例档仍只有 55.00。

---

## 三、判据的裁决

SOTA 绝对值：单类 2–10 → **45–60%**；多类 → **31–43%**；6–10 个 → **10–30%**。
落在事先写死的 40–60% 区间里（多类和高计数还更低）。

> **按判据：这一格值得投。** 天花板 100%，风格线上 ArtFID 的可动空间是
> 小数点后一位，这里是几十个百分点。

---

## 四、在位者过筛（基础必须「有开源码 + 有会议接收」）

| 工作 | 会议 | 开源码 | 单卡 24G 可跑 | 备注 |
|---|---|---|---|---|
| **CountGen / Make It Count** | **CVPR 2025**【一手 openaccess】 | **有**（`Litalby1/make-it-count`） | SDXL，是 | ⚠️ **训练**了一个布局预测小模型（权重已放），不是纯 training-free |
| **YOLO-Count** | **ICCV 2025** | **有**（`mlpc-ucsd/YOLO-Count`） | SDXL-Turbo | T2I 控制是推理期引导；**40–180 s/图**（V100）；论文只给了图，**没给 T2I 绝对数表** |
| CountCluster | under review | ✗（README 说"will be released"，仓库 1 个 commit，空） | — | **SOTA 数（55.75）在它手上** |
| CountDiffusion | 预印本 | ✗（全文无任何代码链接） | — | |
| CountSteer | AAAI'26 **workshop** | 【未查】 | — | 只报了 "+约 4%" |
| CountLoop | 预印本 | 项目页称有 | A100 级 | 迭代式，非单次前向 |

**只有 CountGen 同时满足两条筛子。**

---

## 五、必须先说清的两个坑

1. **没有共享的表。** 三篇三套评测器（CountGD / Grounded SAM / GroundingDINO）、
   三套 prompt 集，数之间**不可横向比**。所以「在它自己的表上打赢它」只有一条
   落地方式：**取 CountCluster 的协议**（它是唯一一张把 Naive SDXL、CountGen、
   SOTA 放在同一列的表），先用 CountGen 的公开代码复现 46.20，再去打 55.75。
   这与风格线上"先复现 StyleID 的 28.801"是同一套做法，那一步是有效的。

2. **CountGD 是 GroundingDINO 改的。** 用户此前否掉过一个我自己造的
   GroundingDINO 指标。区别要讲明白：那次是**我们发明一个尺子**；这次是
   **沿用在位者论文自己用的、已发表的评测器**。是否接受由用户定。
   备选：Grounded SAM（CountDiffusion 用的）。两者都不需要人工标注。

3. T2ICountBench 的数**是人工判的**，我们复现不了，只能引用，不能作为我们的表。

---

## 六、从数本身读出的、方法可以打的位置

- **多类计数**（"three cats and two dogs"）：SDXL 5% → SOTA 31–43%，
  是所有档里最破的，且 CountGen 在这一档**没有报数**。
- **6–10 个**：10–30%，而 1–5 个已经 60–80%。断崖在 5 和 6 之间。
- **prompt 改写整条路已被证伪**（表 C 四种全部变差），
  这是一条已发表的负结果，把可做的空间收窄到 latent / attention 干预。
