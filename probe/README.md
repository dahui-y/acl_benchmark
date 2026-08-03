# 文本编码器探针

回答一个问题：**T2V 模型的文本编码器，到底有没有把"事件是否达成"编码进条件向量？**

这决定论文的叙事走向（见 `../idea_aspect.md` §7c）：

- **信息存在但幅度低** → 生成器拿到了信号却没用 → 原方案成立，外加一节机制定位，
  且免训练差向量干预值得一试
- **信息不存在** → 编码器就是瓶颈 → 叙事从语义学转为架构分析

**两种结果都能写成论文的一节。** 所以这不是生死门，是分叉器——但它必须先跑，
因为它决定 intro 怎么写。

不生成任何视频。单卡数分钟，或 CPU 上用 `t5-base` 冒烟测试。

---

## 快速开始

**先建 venv**，不要装进系统 Python：

```bash
cd ..                      # 仓库根目录
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r probe/requirements.txt

cd probe
python build_stimuli.py --n-items 200                  # -> stimuli.jsonl
python encode.py --model t5-base --out emb_t5base.npz  # CPU 冒烟测试，约 2 分钟
python analyze.py --emb emb_t5base.npz
```

`torch` 与 `transformers` 只有 `encode.py` 需要；只想复跑分析的话装
`numpy scipy scikit-learn` 就够。

真正要看的编码器（需要 GPU）：

```bash
# Wan 2.1 / 2.2 用的 umT5-XXL，约 11 GB (bf16)
python encode.py --model google/umt5-xxl --device cuda --dtype bfloat16 \
                 --layers all --out emb_umt5xxl.npz
python analyze.py --emb emb_umt5xxl.npz --all-layers --out result_umt5xxl.json

# 多数扩散模型用的 T5-v1.1-XXL
python encode.py --model google/t5-v1_1-xxl --device cuda --dtype bfloat16 --out emb_t5xxl.npz

# CLIP 文本塔，已知的 bag-of-words 参照系
python encode.py --model openai/clip-vit-large-patch14 --device cuda --out emb_clip.npz
```

HunyuanVideo 用的是 MLLM（LLaVA 系）编码器，接口不同，需另写加载器——建议先跑上面三个。

---

## 刺激材料设计

每个 item 固定一个事件（动词 + 物体 + 主语 + 场景），**只改变语言如何编码达成**：

| 条件 | 例 |
|---|---|
| `prog` | A man is crushing a lime in the kitchen. |
| `perf` | A man crushed a lime in the kitchen. |
| `result` | A man has crushed a lime in the kitchen. |
| `prospective` | A man is about to crush a lime in the kitchen. |
| `failed` | A man tried to crush a lime in the kitchen but failed. |
| `atelic` | A man is crushing limes in the kitchen. |

外加四个对照：

| 对照 | 例 | 作用 |
|---|---|---|
| `filler` | A man is crushing a lime in the kitchen**, as the footage shows**. | **锚点**：事件与体貌均不变，只加语义无关的词——量化"纯靠改字面能买到多少距离" |
| `paraphrase_min` | A man is crushing a lime **inside** the kitchen. | 紧噪声下界：单 token 同义替换 |
| `paraphrase` | **In the kitchen,** a man is crushing a lime. | 宽松下界，仅作参考 |
| `other_verb` | A man is **rolling** a lime in the kitchen. | 事件不同、体貌相同（**实测表明它不是上界**，见下） |

两处对自己假设保守的设计：

1. **下界用 `paraphrase_min` 而非 `paraphrase`**。后者移动大量 token，会抬高噪声下界、
   压低体貌的相对显著性——正好偏向我们期待的结论。紧的下界才保守。
2. **体貌条件的编辑距离（0.21–0.47）大于下界（0.10）**，即它们改动的 token 更多。
   若嵌入距离仍与下界相当，就无法用"改的字面更少"来解释。

物体只取可数名词（`lexicon.MASS_OR_GENERIC` 过滤掉 mass noun 与类别标签），
否则 `atelic` 的光杆复数对立不成立。

---

## 三项测量

### (a) 几何：Surface-Normalised Salience

```
SNS = d(prog, 体貌条件) / d(prog, filler)
```

- **SNS < 1** — 改变"事件是否达成"对条件向量的影响，**小于**追加几个无意义词
- **SNS > 1** — 体貌的影响大于纯表层扰动

**为什么锚点是 `filler` 而不是 `other_verb`**：实跑之后才发现，
`other_verb` 根本不是上界——换动词（crushing → rolling）只改 1 个 token，
移动量反而**小于**改了好几个 token 的体貌条件。以它作分母的比值会爆掉（出现 ASI=40）。
`filler` 保持事件与体貌不变、只加入语义无关的词，
直接量化"纯粹靠改字面能买到多少距离"，才是可比的锚点。

同时报告配对 Wilcoxon（d_cond vs d_filler）以及距离与编辑距离的相关 r。

### 实跑参考（t5-base，200 items）

| 条件 | d_cond | SNS | 编辑距离 |
|---|---|---|---|
| perf | 0.0720 | 0.86 | 0.206 |
| result | 0.0712 | 0.85 | 0.206 |
| prospective | 0.0474 | **0.56** | 0.256 |
| failed | 0.1563 | 1.86 | 0.472 |
| atelic | 0.0366 | **0.43** | 0.206 |
| *filler（锚点）* | 0.0849 | 1.00 | 0.310 |
| *other_verb* | 0.0493 | 0.59 | 0.103 |

**5 个体貌条件里 4 个低于 1.0** —— 改变事件是否达成，比追加 4 个无意义词
移动得更少。判定为 INFORMATION PRESENT BUT LOW-MAGNITUDE。

注意 t5-base 不是任何视频模型的编码器，这只是管线验证。
真正要看的是 umT5-XXL（Wan 2.2）与 T5-v1.1-XXL。

### (b) 线性探针

从池化向量做 6 类体貌分类，**按动词分组留出**（再按物体分组各做一次），
使探针无法靠词汇身份取巧。两个参照：

- **置换基线** —— 打乱标签后的准确率，即真实 chance
- **bag-of-words 上界** —— TF-IDF 直接在句子上分类

**探针远低于 bag-of-words** 意味着：表层形式携带了这个区别，而编码器把它衰减掉了。
这本身就是一个机制结论。

> 注意解释边界：探针成功只说明"信息在条件向量里线性可取"，
> 不说明编码器"理解"了体貌。但对本研究这就够了——
> 我们要问的正是"生成器有没有拿到这个信号"。

### (c) 逐层定位

`--all-layers`，看区别在哪一层出现或消失。

---

## 已知局限

1. **池化是近似，而且对句长敏感**。扩散模型通过 cross-attention 使用**整个 token 序列**，
   不是池化向量。t5-base 的结果里 `other_verb`（换掉整个动词）距离仅 0.049、
   低于 filler 的 0.085，说明均值池化后的几何很大程度上由句长与表层重叠支配。
   **几何这一节的结论必须以此为限**，`--save-tokens` 可保留逐 token 状态重做分析。
   写论文时这一条必须进 Limitations。
2. **padding 已 mask**。`failed` 条件系统性更长，不 mask 会让句长伪装成条件效应。
   代码里已处理，改动时别破坏它。
3. **探针不等于因果**。信息可线性解码 ≠ 生成器会使用它。
   因果证据要靠差向量注入实验（`../idea_aspect.md` §7c）。

---

## 文件

| 文件 | 作用 |
|---|---|
| `lexicon.py` | 动词变位表、复数规则、mass noun 过滤、场景同义映射 |
| `build_stimuli.py` | 从 `../action_object_taxonomy/` 生成 `stimuli.jsonl` |
| `encode.py` | 加载 HF 文本编码器，输出 `.npz`（逐层池化向量） |
| `analyze.py` | 三项测量 + 判定规则，`--out` 存 JSON |
