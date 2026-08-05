# 自动判官：零人工标注的评测与验证

| 文件 | 作用 |
|---|---|
| `criteria.py` | 三个问题、每个动词的目标末态定义、以及**自带正确答案的控制条件对** |
| `judge.py` | 对抽帧跑 MLLM，输出结构化判断，断点续跑 |
| `validate.py` | 用控制条件验证判官可靠性——**这一节替代 OSCBench 的人工评测** |

---

## 为什么这套不需要标注

一般评测基准必须找人标，是因为"这个视频好不好"没有先验答案。

我们不一样：**suite 里有几个条件的答案是从语义推出来的。**

| 条件对 | 先验正确答案 | 依据 |
|---|---|---|
| `prog` / `paraphrase_min` | 三个问题**答案必须相同** | 只换了一个介词，真值条件不变 |
| `prog` / `filler` | 必须相同 | 补的是一句关于"画面显示"的废话，与宾语无关 |
| `atelic` / `telic_plural` | 必须相同 | 光杆复数 vs 定复数，都不给事件划界 |
| `atelic` / `atelic_some` | 必须相同 | 同上 |
| `prog` / `other_verb` | `final` **必须不同** | 换了动作，本 item 的末态不可能达成 |

前四对量的是**一致性**，最后一对量的是**区分度**——没有它，一个"全答 yes"的判官会在前四对上拿满分。

**suite 自己就是判官的测试集，一条人工标注都不需要。**

## 但一致性率本身是混淆的

```
disagreement(prog, paraphrase_min) = 判官噪声 + 视频模型不稳定性
```

试跑已经证明第二项不为零：一个介词就把画面改了。所以判官要**对同一批视频跑两遍**：

```
判官噪声   = disagreement(pass 1, pass 2)      同一个视频，两次判断
模型不稳定 = 控制条件的不一致 − 判官噪声
```

两个数都是可写进论文的结果：**第一个是本来要靠人工评测提供的效度论证，
第二个是关于模型的发现，而且它设定了所有体态效应必须跨过的地板。**

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

## 与生成侧的接口

```
generate.py  →  videos/<model>/item0007/prog__seed42.mp4
extract_frames.py  →  frames/<model>/item0007/prog__seed42/frame_001.jpg …
judge.py  →  frames/<model>/judgments.jsonl
validate.py  →  效度报告
```
