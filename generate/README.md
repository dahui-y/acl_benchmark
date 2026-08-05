# 视频生成

把 `probe/stimuli.jsonl` 的生成子集变成视频。三个文件：

| 文件 | 作用 |
|---|---|
| `models.py` | 模型注册表。分辨率/帧数/FPS/negative prompt 全在这里，命令行改不了 |
| `generate.py` | 主驱动。排程、断点续跑、manifest、settings.json |
| `extract_frames.py` | 抽帧，抽样公式与仓库根目录 OSCBench 的 `extract_frames.py` 完全一致 |
| `contact_sheet.py` | 把一个 item 的多个条件排成"行=条件、列=时间"的拼图（OSCBench Fig. 4 版式）|
| `BUDGET.md` | 实测速度、算力约束、每项设置偏离及其代价。论文设置表与 limitation 从这里取材 |

---

## 设计：为什么种子这样安排

**item 内共用一个种子。** 每个事件 15 个视频条件只在"语言如何编码终结"上不同。
科学主张是**同一 item 内两个条件之差**，所以 item 内除 prompt 外一切必须固定，
而最容易压倒一切的就是初始噪声——种子基本决定了画面里是谁、机位在哪、台面长什么样。
不同种子的两个视频同时因两个原因不同，效应无法归因。共用种子后，
分辨率与帧数是模型常量 ⇒ 潜变量形状相同 ⇒ **初始噪声逐位相同**，剩下唯一的差异源就是文本条件。

**每个 item 三个种子。** 单个种子只是模型分布里的一次抽样：有的种子会遮住宾语，
有的根本没渲染出状态变化。三个种子买到的是误差棒、条件差异的稳定性，
以及把 item 方差和种子方差分开（item × seed 交叉随机效应）。

**种子在 item 之间也共用**（42/43/44 全局固定）。若每个 item 用不同种子，
"item 7 的 seed 42"与"item 8 的 seed 42"不可比，交叉设计白白丢掉一个因子。

这对终结性轴另有一层用处：那条轴的预期是视频**不该**有差别。
共用噪声把这条软预期变成硬预期——相同噪声 + 几乎相同的条件，就该给出几乎相同的视频。

**迭代顺序 item → seed → condition**，所以中断后留下的是完整的 (item, seed) 块。
块是可分析的最小单位，半块没有价值。`--shard i/n` 也按 item 切，理由相同。

## 模型与设置：对齐 OSCBench Table 6

OSCBench 的 Appendix B 只说一句"we follow the official and default
implementations"，Table 6 只给分辨率/帧数/FPS/时长，**不给步数和引导强度**——
意思就是跑的是 pipeline 自己的默认值。我们照办：

| | OSCBench Table 6 | `models.py` |
|---|---|---|
| Wan-2.2 | 1280×720, 81f, 16fps | `wan2.2-t2v-a14b` 逐格相同 |
| HunyuanVideo-1.5 | 1280×720, 121f, 24fps | `hunyuanvideo-1.5` 逐格相同 |

注意他们那行 Wan-2.2 是 **81 帧 @16fps，这是 A14B 的配置**，TI2V-5B 是 121f@24。
要跟他们可比就得用 A14B。`wan2.2-ti2v-5b` 留在注册表里只作冒烟测试、
确定性检查和测速用，`--list` 会把它标成 `[----]`，它不在 Table 6 里，
用它生成的东西不能跟他们的 Wan-2.2 并排报。

步数和引导强度在 `models.py` 里**故意不填**，不传给 pipeline，
让官方默认生效；`generate.py` 再从 pipeline 签名把实际默认值读回来写进
`settings.json` 的 `pipeline_defaults`。这样"我们用的是默认值"是一个被记录下来的
数字，而不是一句一年后无法核实的话。

同一模型内，这些值对所有条件严格相同——一旦随条件变化，
`prog` 与 `result` 的差异就不再能归因于 prompt。论文的设置表从 `settings.json`
生成，也就是从运行现场生成，而不是从我们的意图生成。

**negative prompt 默认为空，这是刻意偏离默认值的一处。** Wan 官方 negative prompt 里
含"静止不动的画面"等项，而我们有若干条件（`perf`、`result`、`phase_finish`）描述的
正是状态而非持续动作——用官方 negative prompt 等于把混淆直接写进数据。
论文里要写明这一处偏离。

## 生成哪些条件

默认 16 条 = 14 条目标条件 + **两个锚点**：

| 锚点 | 预期 | 作用 |
|---|---|---|
| `other_verb`（换动词） | 视频**必须**变 | 上锚。不变就说明模型根本没在读事件描述 |
| `paraphrase_min`（换一个同义词） | 视频**不该**变 | 下锚。变了就说明模型对任何改词都敏感，体态效应无从谈起 |

两个锚点把所有体态效应**夹在中间**。这不是锦上添花——它决定了结论的形式：
有了双锚，"模型做不到"就变成"**模型对词汇内容有反应，却唯独对体态形态盲**"。
**选择性缺陷不会被"你模型太小"解释掉，笼统的失败会。**

`paraphrase`（语序重排）与 `filler`（长度配平）仍只在文本侧，
它们是用来校准编码器距离的，视频侧没有对应的标注问题。`--conditions` 接受 `video`（默认 16 条）、`all`（18 条），或**直接点名**：

```bash
python generate.py --model ... --conditions prog perf other_verb paraphrase_min
```

点名这件事比看起来重要：`--limit` 是按列表顺序截断的，
所以 `--conditions all --limit 8` 给的是**列表里的前 8 条**，不是你心里想的那 8 条。

---

## 用法

```bash
# 先看清单：多少视频、注册表哪几项在当前 diffusers 下能解析
python generate.py --list

# 干跑：只排程和写 settings.dry-run.json，不碰模型、不写 manifest
python generate.py --model wan2.2-t2v-a14b --dry-run

# 先用小模型烧 5 个测速和验证流程，脚本会打印 s/video 和 ETA
python generate.py --model wan2.2-ti2v-5b --out-dir /data/videos --limit 5

# 正式跑（可随时 Ctrl-C，重跑自动续）
python generate.py --model wan2.2-t2v-a14b --out-dir /data/videos
python generate.py --model hunyuanvideo-1.5 --out-dir /data/videos

# 显存不够时
python generate.py --model wan2.2-t2v-a14b --out-dir /data/videos --offload

# 多卡：按 item 切分，每张卡一个 shard
CUDA_VISIBLE_DEVICES=0 python generate.py --model wan2.2-t2v-a14b --shard 0/4 --out-dir /data/videos &
CUDA_VISIBLE_DEVICES=1 python generate.py --model wan2.2-t2v-a14b --shard 1/4 --out-dir /data/videos &

# 抽帧：标注只覆盖一个种子，另外两个种子只进 MLLM 自动评测
python extract_frames.py --videos /data/videos/wan2.2-t2v-a14b \
                         --out /data/frames/wan2.2-t2v-a14b --seeds 42

# 定性检查 / 论文定性图：一个 item 的各条件排成 Fig.4 那样的拼图
python contact_sheet.py --videos /data/videos/wan2.2-ti2v-5b-480p --item 0
python contact_sheet.py --videos /data/videos/wan2.2-ti2v-5b-480p --all-items \
                        --conditions prog perf result other_verb paraphrase_min
```

## 显存

24GB 卡（4090）上，整条 pipeline 常驻会在第一个去噪步之前就把卡填满——
文本编码器 + transformer + VAE 加起来就没给激活值留位置。必须加 `--offload`：

| 开关 | 作用 | 代价 |
|---|---|---|
| `--offload` | 一次只把一个组件放在 GPU 上 | PCIe 传输，每个视频多几十秒 |
| `--sequential-offload` | 逐层换入换出，几乎什么卡都能跑 | 慢很多，最后手段 |
| VAE tiling | **默认开**，分块解码，避开 121 帧 720p 一次性解码的峰值 | 略慢，可能有极淡的拼缝 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 脚本自动设，防碎片化 | 无 |

VAE tiling 的拼缝值得说一句：它对同一个 item 的所有条件是完全相同的，
所以在本设计唯一做的那种比较（item 内条件差）里会抵消。真正在意画质时用
`--no-vae-tiling`，但要有显存。

实际峰值显存记在 `settings.json` 的 `memory` 字段里，进度行也会打印 `peak NNGB`。

### 正式跑之前做一次

```bash
python generate.py --model wan2.2-t2v-a14b --out-dir /data/videos --determinism-check --limit 1
```

同一个 (prompt, seed) 生成两次比像素。可识别性的前提是"种子 + prompt 固定视频"，
而非确定性的 attention kernel 会破坏这个前提——真被破坏了，item 内条件差异里
就掺着采样噪声。这件事早查五分钟，晚查就是整批数据的解释权。

---

## 规模与开销

| | |
|---|---|
| 生成子集 | 37 items × 16 conditions × 3 seeds = **1776 视频/模型** |
| 开源主实验 | 两个 480p 模型 = **3552 视频**，≈ 8 天（实测 208 s/视频） |
| 人工标注覆盖 | 只标 seed 42：592 × 2 模型 = 1184 视频 |
| 标注判断数 | 1184 × 2 个二元问题 × 3 名标注者 ≈ **7.1k**（OSCBench ≈ 20k） |
| 另两个种子 | 只进 MLLM 自动评测，作稳定性检查 |

**上表是设计规模，不是可执行计划。** 单张 4090 实测 536 s/视频（720p 官方设置），
两个模型要 20 天不间断，跑不下来。降配的优先级、各项代价和实测数据都在
`BUDGET.md`，配置定稿前以那份为准。

算力紧张时退到 2 个种子，**但不要退到 1 个**——退到 1 个就同时失去误差棒
和方差分解，而这两样正是这个设计相对 OSCBench 的增量。

闭源参照模型（Kling / Veo）不走这个脚本：API 模型没有我们能在条件间共用的种子控制，
所以它们用单独的驱动、更小的子集，只作参照，不参与 item 内对比的主分析。

## 注册表状态

三项的 `verified` 都还是 `False`——`--list` 报 `ok` 只说明类名在当前 diffusers 里
存在，不代表 repo id 对、权重能下、能跑通。真正的核验是第一次 `--limit 5` 跑通，
跑通后把 `settings.json` 里 `pipeline_defaults` 读到的值抄回 `models.py` 的注释、
把 `verified` 改成 `True`。

A14B 是双专家 MoE，diffusers 可能还有第二个 guidance scale（`guidance_scale_2`），
它会出现在 `settings.json` 的 `pipeline_defaults` 里——第一次跑完看一眼，
确认默认值是我们想要的，不是的话再决定要不要在 `extra` 里显式钉住。

## 产物

```
/data/videos/wan2.2-t2v-a14b/
    settings.json                    # 论文设置表的唯一来源
    manifest.jsonl                   # 每个视频一行：prompt、seed、全部生成参数、耗时、状态
    item0000/prog__seed42.mp4
            /perf__seed42.mp4
            ...
/data/frames/wan2.2-t2v-a14b/
    frames_index.jsonl
    item0000/prog__seed42/frame_001.jpg ... frame_020.jpg
```

manifest 每行都带 `status`；出错的格子不会中断整批，续跑时会重试。
写视频走 `.part.mp4` 再原子改名，所以被 kill 不会留下半截文件被误判为已完成。
