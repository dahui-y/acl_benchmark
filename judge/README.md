# 自动判官：零人工标注的评测与验证

| 文件 | 作用 |
|---|---|
| `criteria.py` | 三个问题、每个动词的目标末态定义、以及**自带正确答案的控制条件对** |
| `judge.py` | 对抽帧跑 MLLM，输出结构化判断，断点续跑 |
| `validate.py` | 用控制条件验证判官可靠性——**这一节替代 OSCBench 的人工评测** |

---

## 零标注下能建立什么、不能建立什么

`validate.py` 报四个数，**它们不是同一类数**：

| # | 名称 | 测的是什么 | 需要标注吗 |
|---|---|---|---|
| 1 | **信度** | 同一视频判两遍的一致率。纯判官属性 | 否 |
| 2 | **特异度** | 拿视频问一个它不可能达成的末态，答案先验为 `no` | 否 |
| 3 | **模型不稳定性** | 语义等价条件间的不一致，减掉判官噪声 | 否 |
| 4 | **动作执行率** | 第一级结果：事件到底有没有被渲染 | 否 |

**缺的是敏感度，而且这条路拿不到。** 设计里没有任何东西能认证"某个视频确实达成了末态"，
所以没有已知正例。由此推出：**可辩护的主张是"条件之间的差异"，
不是"绝对达成率"**——后者未经校准。

一条容易搞混的：**第 3 行不是判官效度检验。** `prog` 与 `paraphrase_min` 是
**两段不同的视频**（试跑已证实一个介词就能改画面），判官给出不同答案可能是对的。
减掉判官噪声之后，那个数量的是**视频模型**，而且它就是所有体态效应必须跨过的地板。

判官效度只来自第 1 行（信度）和第 2 行（特异度）。

### 特异度怎么做到不用标注

拿一段擦丝的视频，问它"有没有被切成极碎的丁"——**答案先验是 no**，
因为我们知道这段视频演的是别的动作。`stimuli.jsonl` 里本来就有 `other_verb_base`，
所以**已有的每一段视频都是一个已知负例**，不用多生成任何东西，只多问一次。

```bash
python judge.py --frames ... --model ... --known-negative
```

这把特异度从单个格子扩到几十个格子。

### 那条缺口怎么补

敏感度要么靠**外部已标注的真实视频**（ChangeIt / Ego4D 一类），
要么靠**作者人工核验一个小样本**（约 150 段、一小时），
以"we manually verified a random subsample"的形式报告，不作为主结果。

**这一条必须写进 limitation，不能藏。**

## 判官看不到句子

这是与 OSCBench 最大的方法差异。他们的评测者能看到 prompt；我们**不能**——
我们的条件差异**就在** prompt 里，让判官看到句子等于告诉它答案。

判官只拿到：抽帧 + 动词原形 + 宾语 + **该动词目标末态的文字定义**。
同一个 item 的 17 个条件被问的是**完全相同的问题**。

## 三个问题

| id | 问题 | 用途 |
|---|---|---|
| `action` | 有没有任何一帧显示动作正被施加于宾语？ | **第一级结果**：模型到底会不会渲染这个事件 |
| `initial` | 第一帧里，宾语是否处于未受影响的原始状态？ | 试跑发现模型主要改的是**首帧状态**，这一格是信号所在 |
| `final` | 最后一帧，宾语是否到达目标末态？ | 体态编码的主判断 |

二元（yes / no / unclear），不是 1–5 Likert。**只有二元 + 明确末态定义，
控制条件才能承载正确答案**——两个 Likert 分数"应该相同"是说不清的。

目标末态的定义写在 `criteria.py` 的 `TARGET_STATES`，每个动词一条，
要求**单帧可判**。五个 activity 类动词（fry / saute / grill / roast / roll）
没有内在终点，它们的"目标末态"是约定的停止点而非语义终点——
这不是定义的缺陷，正是体态类这个变量要追踪的性质。

---

## 用法

后端是任意 OpenAI 兼容的 chat endpoint，托管 API 和本地起的服务（vLLM / SGLang）
走同一条代码路径。

```bash
pip install openai

export JUDGE_API_KEY=sk-...
export JUDGE_BASE_URL=https://api.openai.com/v1      # 本地服务改成 http://localhost:8000/v1

# 先看一眼实际发出去的 prompt，不调用任何接口
python judge.py --frames /data/frames/wan2.2-ti2v-5b-480p --model gpt-5.2 --dry-run

# 先判 20 个测速和验证格式
python judge.py --frames /data/frames/wan2.2-ti2v-5b-480p --model gpt-5.2 --limit 20

# 正式跑，两遍（第二遍是判官噪声的测量，不是浪费）
python judge.py --frames /data/frames/wan2.2-ti2v-5b-480p --model gpt-5.2 --repeat 2

# 效度报告
python validate.py --judgments /data/frames/wan2.2-ti2v-5b-480p/judgments.jsonl
```

```bash
# 特异度：每段视频再问一次「错误的末态」，答案先验为 no
python judge.py --frames ... --model gpt-5.2 --known-negative
```

`--repeat 2` 不是可选项。**没有它就没有判官噪声，没有判官噪声就分不开
"判官不准"和"模型不稳"**，控制条件的一致性率也就无法解释。

## 输出

`judgments.jsonl`，每个 (item, condition, seed, pass) 一行：

```json
{"item_id": 9, "condition": "prog", "seed": 42, "pass": 1,
 "gerund": "whipping", "noun": "egg", "aspectual_class": "degree_achievement",
 "judge_model": "gpt-5.2", "status": "ok", "n_frames": 20,
 "action":  {"answer": "yes", "evidence": "..."},
 "initial": {"answer": "yes", "evidence": "..."},
 "final":   {"answer": "no",  "evidence": "..."}}
```

出错的格子写 `status: "error"` 并继续，续跑时重试。
答案不在 yes/no/unclear 三者之内**直接判为错误而不做纠正**——
判官编出第四种答案说明它没理解任务，静默归一化会把这件事藏起来。

## 抽帧数

默认 20 帧，与 OSCBench 的 MLLM 评测一致，所以两边的数字是可比的。
`extract_frames.py` 的抽样公式也和他们完全相同（`linspace`，含首末帧）。

## 两个判官跑在两个地方

生成机在国内、能上外网的机器在国外，这个约束正好把两个判官拆开——
而"两个 MLLM 交叉一致"本来就是效度论证里要补的一格：

| 判官 | 跑在哪 | 需要外网 |
|---|---|---|
| **GPT-5.2**（主判官） | 本地 Mac | 是 |
| **Qwen2.5-VL**（第二判官） | 生成机的 4090，vLLM 本地服务 | **否** |

主判官选 GPT-5.2 的理由是**可比性**：OSCBench 的主结果就是用它报的，
而且他们测出 GPT-5.2 + CoT 与人工相关最高。同一个判官，
两边的数字在同一坐标系里。

抽帧要搬到 Mac 上：111 个视频 × 20 帧 ≈ 2200 张图、一两百 MB。
判官脚本只依赖 `openai`，不需要 torch。

```bash
# 生成机：打包抽帧
tar czf frames.tgz -C $FRAMES wan2.2-ti2v-5b-480p

# Mac：解包，判，把 judgments.jsonl 传回来（或就在 Mac 上做分析）
pip install openai
python judge.py --frames ./wan2.2-ti2v-5b-480p --model gpt-5.2 --repeat 2
```

```bash
# 生成机：起本地服务，第二判官
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
JUDGE_BASE_URL=http://localhost:8000/v1 \
python judge.py --frames $FRAMES/wan2.2-ti2v-5b-480p \
  --model Qwen/Qwen2.5-VL-7B-Instruct --repeat 2 --out judgments_qwen.jsonl
```

两份 judgments 合到一起出报告，`validate.py` 会多出交叉一致率一节：

```bash
python validate.py --judgments judgments_gpt.jsonl judgments_qwen.jsonl
```

`--model` 可以把前四节限定到某一个判官。

## 与生成侧的接口

```
generate.py  →  videos/<model>/item0007/prog__seed42.mp4
extract_frames.py  →  frames/<model>/item0007/prog__seed42/frame_001.jpg …
judge.py  →  frames/<model>/judgments.jsonl
validate.py  →  效度报告
```
