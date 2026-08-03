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

```bash
pip install -r requirements.txt

python build_stimuli.py --n-items 200            # -> stimuli.jsonl
python encode.py --model t5-base --out emb_t5base.npz    # CPU 冒烟测试
python analyze.py --emb emb_t5base.npz
```

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

外加三个对照：

| 对照 | 例 | 作用 |
|---|---|---|
| `paraphrase_min` | A man is crushing a lime **inside** the kitchen. | **主噪声下界**：单 token 同义替换，意义与体貌均不变 |
| `paraphrase` | **In the kitchen,** a man is crushing a lime. | 宽松下界，仅作参考 |
| `other_verb` | A man is **rolling** a lime in the kitchen. | 上界：事件不同、体貌相同 |

**为什么下界必须用 `paraphrase_min` 而不是 `paraphrase`**：
后者移动了大量 token，会**抬高**噪声下界，从而压低 ASI ——
正好偏向我们期待的结论。用紧的下界才是对自己假设保守。

材料还有一层保守性：体貌条件的 token 编辑距离（0.21–0.47）**大于**下界（0.10）。
也就是说体貌条件改动的 token 更多。若它们的嵌入距离仍与下界相当，
就不能用"改动的字面更少"来解释——证据方向对我们不利，这是应该的。

物体只取可数名词（`lexicon.MASS_OR_GENERIC` 过滤掉 mass noun 与类别标签），
否则 `atelic` 的光杆复数对立不成立。

---

## 三项测量

### (a) 几何：Aspect Salience Index

```
ASI = (d_aspect − d_floor) / (d_other_verb − d_floor)
```

- **ASI ≈ 0** — 体貌对嵌入的影响不超过一次同义替换 → 编码器几乎不表征达成
- **ASI ≈ 1** — 影响与换掉动词相当 → 体貌是一等公民

同时报告配对 Wilcoxon（d_aspect vs d_floor）。

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

1. **池化是近似**。扩散模型通过 cross-attention 使用**整个 token 序列**，不是池化向量。
   `--save-tokens` 可保留逐 token 状态以便重做分析。写论文时这一条必须写进 Limitations。
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
