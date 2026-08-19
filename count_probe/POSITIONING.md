# 相对 CountGen 的定位：接着做，不是打倒

写在动手写方法之前。核心问题：**CountGen 既然有这些缺陷，它是怎么中 CVPR 的？**
答案决定我们的论文能怎么写。

标注约定：【一手】= 论文正文/表格或我们自己跑出来的；【二手】= 转述。

---

## 一、它成功在哪：删多余

| | 题数 | vanilla | 修正后 | 增益 | p（McNemar 精确）|
|---|---:|---:|---:|---:|---:|
| 数多了 → 删最小 blob（8 行启发式，无模型）| **65** | 9.2% | 49.2% | **+40.0** | **<0.001** |
| 数少了 → ReLayout U-Net（474MB 训练权重）| 33 | 21.2% | 27.3% | +6.1 | 0.774 |

65:33 —— 三分之二的干预走在能成的那条路上，整体 +16.8 个点几乎全来自这里。

**机制上为什么不对称**：删多余时，desired mask 是从**原版图自己的布局**里删掉最小的
几个 blob 得来的，剩下的 N 个 blob 正好压在已存在的物体上。此时二值损失说的两句话
恰好都有利 ——「这些区域要有物体感」本来就满足（白送），「其它地方压低」正是要
干的事（压掉多余的）。`pos_weight=10` 让前景项主导，还顺带保护了该留的物体。

补缺失时反过来：要在空白处**凭空造一个实例**，而二值损失只说"这片要有物体感"、
不说"要恰好一个"，那片区域中位占画面 35.3%。**同一个缺陷，删的方向上是保护，
补的方向上是灾难。**

---

## 二、为什么它该中：整体结果是真的，科学贡献是真的

- **我们复现到了它自己的数**：它们 Table 2 用 YOLOv9 报 **50%**【一手】，
  我们用 YOLOv9e 在 200 题上得 **48.5%**——同一评测器，差 1.5 个点。
- **消融像样**【一手 Table 2】：

  | | Acc（YOLOv9）|
  |---|---:|
  | CountGen 布局 + CountGen 生成 | 50% |
  | CountGen 布局 + Bounded Attention | 40% |
  | 随机 mask + CountGen 生成 | 37% |
  | 随机 mask + Bounded Attention | 29% |

  布局相对随机 mask 值 +13 个点。这个结论站得住。
- **科学贡献是新的**：首次指出扩散模型自身的自注意力特征携带实例身份，
  可以据此分离并计数实例（abstract：*"we first identify features within the
  diffusion model that can carry the object identity information"*）。

> **所以论文不能写成「CountGen 是错的」。它不是错的。**

---

## 三、我们的位置：他们 Limitations 只有三句话，我们对每一句都有数和机制

Limitations 原文【一手】：

> "Occasionally, our optimization results in **multiple instances of an object in an
> area intended for just one by the layout**. In other cases CountGen generates
> **plain backgrounds** compared to SDXL. Finally, the scope of our experiments may
> seem narrow, since we focus on generating scenes with **up to 10 instances and a
> single object per prompt**."

| 他们写的 | 我们量的【一手，自跑】 | 我们找到的机制【一手，读码+复算】 |
|---|---|---|
| "**Occasionally** … multiple instances in an area intended for just one" | **不是 occasionally**：\|残差\|≥2 占 **44.9%**（n=98）；极端例 1 个 blob → 多出 6 匹马 | `loss_utils.py:5` 的 `(desired_mask != 0)` 把 N 抹掉；`blob_merger` 凸包使前景中位占 **35.3%**、最大 68.3%；归一化后过 sigmoid 造成损失下界 `0.693+2.439f`，**76–85%** 的题退出阈值数学上不可达 → refinement 每次跑满 20 步 |
| "generates **plain backgrounds**" | 【未量化，待做】 | `attention_processors.py:45` 的 `for j in range(0, max+1)` 把背景（标号 0）也算进 `blob_coordinates`，使其恒为全 1 → **背景 query 的整行自注意力被清零**（`check_selfattn_mask.py` 已复算）。一行改动可 A/B |
| "up to **10** instances、**single object** per prompt" | N=10 那 33 题官方代码整题跳过（`run_countgen.py:104`），vanilla 仅 **6.1%** | `relayout.py:7` 的 `in_channels=9, out_channels=10`；`self_counting_sdxl_pipeline.py:357` 的 `paired_indices[0]` 只取第一个数词 |

**论文形状**：把在位者一句话带过的 limitation 量化成主导性失败 → 给机制 → 修好。
接着前作往下做，不是推翻前作。

**他们没报的**：over/under 两支的分开数字（Section 3.2 描述了两套机制，全文无分支报数）；
DBSCAN 计数器自身的准确率（我们测：与 YOLO 逐题一致 64.1%、MAE 0.89）。

---

## 四、这对方法设计的两条硬约束

1. **不能把"删多余"那一支弄坏。** 它贡献了几乎全部增益。我们的损失会同时作用于
   两支，所以**结果必须按两支分开报**（`decompose.py --arms` 的表四/表六b），
   不能只看总分：否则可能"补"涨 5 分、"删"跌 10 分而总分难看，却看不出原因。
2. **"plain backgrounds" 可以顺手拿下**，作为一个独立消融行：修掉 `range(0,...)`，
   看背景是否恢复、计数是否受影响。这一行不属于我们的主方法，但它是一个
   便宜且可验证的附带贡献。

---

## 五、诚实性备忘

- 我们的拆解建立在 **一次 CoCoCount 抽样（200 题）**、**YOLOv9e 单一评测器**、
  以及 **补缺失那一支只有 33 题** 之上。方法确认有效后必须扩样本。
- 空对照的读数：新增 blob 真实位置命中 68.9%，**随机位置 56.0%**，差 +12.9。
  所以"引导有空间服从性"这句话要按 +12.9 说，不能按 68.9 说。
  我最初写的判据「命中率 ≫50% 即听话」定得不好——没预料到随机基线有 56%。
- 定性图（horse_num=7_seed=1932）属**作者自查**，论文里要写明是作者本人看的、
  不是第三方标注，且只用于定性说明，所有定量结论由自动评测器给出。
