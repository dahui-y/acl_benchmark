# count_probe：复现 CountGen 并把它的准确率拆开

三步，一条命令接一条。目的不是"跑通"，是**先对齐发表数，再看天花板卡在哪一段**
——和风格线上先复现 StyleID 的 28.801 是同一套做法。

## 0. 建环境 + 备资产

**要新建环境**，不能沿用风格线那套（那边是另一代 torch/diffusers）。
完整步骤见 **[ENV.md](ENV.md)**，一句话版：

```bash
conda create -n countgen python=3.10 -y && conda activate countgen
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r count_probe/requirements_infer.txt
python -m spacy download en_core_web_trf
```

⚠️ **不要直接 `pip install -r help_code/make-it-count/requirements.txt`**：
那份文件里 `transformers` 指向一个 GitHub commit，与同文件的
`spacy-transformers==1.2.5`（要求 `transformers<4.31`）冲突，国内也多半拉不动。
`requirements_infer.txt` 换成了 `transformers==4.29.2`——依据是那个 commit 自己的
`__version__ = "4.29.0.dev0"`——并删掉了推理路径根本没 import 的一堆包。

装完先自检（不占 GPU，只查版本约束和四样资产在不在）：

```bash
python count_probe/env_check.py
```

## 1. 跑批

```bash
export SD_OUT=/openbayes/input/input0/Sim2Struct-1000/temp/scalediff_out
export RELAYOUT_CKPT=/openbayes/input/input0/Sim2Struct-1000/temp/weights/relayout_checkpoint.pth

# 先冒烟 5 题，确认能跑通、量一下单张耗时
python count_probe/countgen_batch.py --limit 5 --out $SD_OUT/count/cocoount
# 全量 200 题
python count_probe/countgen_batch.py --out $SD_OUT/count/cocoount
```

**还没拿到 ReLayout 权重**（Google Drive，HF 无镜像）也能先开跑：

```bash
python count_probe/countgen_batch.py --vanilla-only --out $SD_OUT/count/cocoount
```

这一档只跑原版 SDXL + DBSCAN 计数，一次前向、无梯度，快三四倍。
表三四个格子里能先拿到三个（baseline、计数器一致率、「计数器说对但实际错」
那一桶＝天花板损失），只差修正成功率。拿到权重后**不加该开关重跑一遍**，
脚本会认出哪些题是先行档跑的并补上修正那一步，不会重复劳动。

不改 make-it-count 一行源码。与直接跑 `pipeline/run_countgen.py` 的差别只有三处，
脚本头部逐条写明了理由：**记下 DBSCAN 计数器读数**、**N>9 仍跑 vanilla**、
**两个只影响速度的缓存**（spacy 和 torch.hub 原本每张图重载一次）。
随机数调用顺序与原脚本逐行一致。可中断续跑。

粗估耗时：vanilla 一次 50 步 + 修正一次 50 步、期间最多三段 20 次带梯度的
UNet 前后向 → 单张约 40–80 s，200 题 2.5–4.5 小时（4090）。所以务必先 `--limit 5`。

## 2. 改名分臂

```bash
python count_probe/make_arms.py --src $SD_OUT/count/cocoount \
    --out $SD_OUT/count/cocoount_arms
```

它们自己的两个脚本对不上名字：`run_countgen.py` 存 `{obj}_num={N}_seed={S}.png`，
`evaluation_script.py` 要 `{count}__{class}__{...}.png`；而且 `*_vanilla.png`
和 CountGen 图混在同一个目录里。这一步只建软链，产出 `vanilla/` 与 `countgen/`
两个臂加一份 `index.json`。类名会映射到 YOLO 认识的 COCO 名
（`ball → sports ball` 等），不映射会静默全错。

## 3. 评测

```bash
python count_probe/yolo_eval.py --arms $SD_OUT/count/cocoount_arms
```

判定口径与 `evaluation_script.py` 一致（脚本会断言 `model.names` 与 COCO-80
逐项相同）。出三张表：

- **表一** 各臂准确率 / MAE。对齐目标【一手】：CountCluster 表里
  CountGen 46.20 / SDXL 27.88（CountGD 评测器）；CountDiffusion 表里
  CoCoCount CountGen 51 / SDXL 34（Grounded SAM）。评测器不同，量级对上即可。
- **表二** 按要求个数分档。N>9 单列——官方 `run_countgen.py:104` 整题跳过。
- **表三** ★ **把准确率拆成两段**：
  「计数器说对 & 实际错」这一类**永远不会被修正**（`extract_mask.py:25-27`
  + `run_countgen.py:128-129` 直接输出原版图），所以
  **CountGen 的上限 = 它 DBSCAN 计数器的准确率**。这张表还给出"换一个完美
  计数器最多能拿到多少"，直接回答该动哪一段。

想对照它们原装的评测脚本（证明我们没动评测）：

```bash
cd help_code/make-it-count
python evaluation_script.py --images_dir "$SD_OUT/count/cocoount_arms/countgen" \
    --output_dir "$SD_OUT/count/official_eval"
```

## 我们的方法（`--loss instance` / `--fg-max`）

默认全关时行为与原版 CountGen 一致，所以 A/B 干净。三条根因、三处改动，
详见 `inst_loss.py` 文件头和 `POSITIONING.md`。

```
L = L_peak + w_sep·L_sep + w_bg·L_bg + w_conc·L_conc

L_peak = 1 − min_k ( max_{p∈blob k} A(p) )    ← N 从这里进入目标函数
L_sep  = mean_{p∈间隔带} A(p)                  ← 相邻 blob 之间挖谷
L_bg   = mean_{p∈背景} A(p)
L_conc = mean_k ( mean_blob A / max_blob A )   ← 默认权重 0，留作消融
```

`inst_loss.py` 的自测把「本方法存在的理由」写成了断言而不是声明：

| 解 | 原版 BCE | 我们的 |
|---|---:|---:|
| 理想（每 blob 一个尖峰）| 1.1389 | **0.0200** |
| 铺满前景 | **0.8718** | 0.5000 |
| 相邻粘连 | 0.9051 | 0.5580 |

值越小越优 —— **原版损失把「铺满」排在「理想解」前面，它不是看不见 N，
是主动奖励铺满。** 跑 `python count_probe/inst_loss.py` 可复核。

### 调参必须在另一批 prompt 上做

换了损失，`thresholds` 和 `scale_factor` 就必须重定（原版量纲是带 pos_weight=10
的 BCE，取值 0.8~2.4；我们的各项都是 [0,1] 上的均值/最大值）。但重定阈值就是调参，
**不能在那 167 题上调**，否则报出来的数是在测试集上选出来的。

```bash
python count_probe/make_tune_set.py --n 60 --out $SD_OUT/count/tune.json
```

用他们自己的生成脚本另抽 60 题，按 id 剔除与评测集重合的。**那份 json 要进版本库**
——他们的脚本没有 seed 参数，可复现性靠保留文件。

### 跑法

```bash
# 调参档（60 题，约 40 分钟一轮）
python count_probe/countgen_batch.py --loss instance --fg-max 0.25 \
    --thresholds 0:0.5,10:0.4,20:0.35 --scale-factor 50 \
    --dataset $SD_OUT/count/tune.json --out $SD_OUT/count/tune_inst

# 定下超参后，评测档（167 题，约 105 分钟）——**必须换目录**
python count_probe/countgen_batch.py --loss instance --fg-max 0.25 \
    --thresholds <定下来的> --scale-factor <定下来的> \
    --out $SD_OUT/count/cocoount_inst
python count_probe/make_arms.py --src $SD_OUT/count/cocoount_inst \
    --out $SD_OUT/count/cocoount_inst_arms
python count_probe/yolo_eval.py --arms $SD_OUT/count/cocoount_inst_arms
python count_probe/decompose.py --arms $SD_OUT/count/cocoount_inst_arms
```

同一个输出目录里混进两种配置会被脚本挡住（续跑只按 id 判断，混了就是
一个两种配置各跑一半的目录，而且是静默的）。

**判读必须按两支分开**（`decompose.py` 的表四/表六b）：「删多余」那一支贡献了
原版几乎全部的增益（+40.0，p<0.001），新损失不能把它弄坏。只看总分会掩盖
「补涨了、删跌了」。

## 其他

- `check_selfattn_mask.py`：复算 `attention_processors.py:45` 的
  `range(0, ...)` 使 `blob_coordinates` 恒为全 1、背景 query 整行注意力被清零。
  零依赖、零 GPU，`python count_probe/check_selfattn_mask.py` 直接看。
- `CODE_READ_countgen.md`：源码通读，硬限制逐条带行号。
- `INCUMBENTS.md`：这一格的在位者与「SOTA 到天花板」的距离。

## 24G 卡上的三处显存改动（必须在论文里交代）

改完实测（RTX 4090 23.5 GB，`--mem-log`）：

| | |
|---|---|
| 引导前向后 | 15.28 GB |
| 进修正步（已释放上一张图） | **6.48 GB**（改之前是 18.58，等于没释放） |
| 峰值 | 18.73 GB，余量 4.8 GB |
| 速度 | **38.5 s/张**，CoCoCount 200 题约 2.1 小时 |

另一个副产品读数：每次进修正步都跑满 21 次前向（20 次迭代 + 最后一次），
即 `while loss > target_loss` **从未靠达到阈值退出**，每次都是撞
`max_refinement_steps=20` 才出来。三个触发点（step 0/10/20）都如此 ——
说明 `thresholds {0:1.3, 10:1.2, 20:1.15}` 对这个带 `pos_weight=10` 的 BCE
来说定得太低，实际等价于「固定跑 20 步」。


原版在 24G 上跑不动：修正步 `perform_iterative_refinement_step` 带梯度，
SDXL 1024² 的注意力图被 autograd 全留住。三处改动，前两处**数值完全等价**：

| # | 改动 | 依据 | 数值影响 |
|---|---|---|---|
| ① | `update_latent` 的 `create_graph=True` → `False` | 全仓库无二阶导；拿到 grad 后下一轮立刻 `clone().detach()`。`create_graph` 只决定"梯度计算本身可不可再求导"，**grad 的值不变**，且它隐含的 `retain_graph=True` 让图算完不释放 | **无** |
| ② | 切断对上一张图的三条引用：(a) `loss_and_plot` 改返回 float，带图的张量存到 `self` 上由 `update_latent` 取用；(b) 进修正步前清掉它和 attention store；(c) **引导前向的 UNet 返回值 detach** | 一张图实测约 11.6 GB（`--mem-log`：前向后 18.59 GB，减去 6.95 GB 权重），两张装不下。三条引用缺一不可：<br>· 返回值只被用于比较（`:171/:188/:490/:497`）和传给 `update_latent`（`:189/:498`）<br>· UNet 的返回值在引导路径上是 `_ =`（`:175/:205/:476`），从未使用；loss 是从 attention store 算的。主去噪路径的 `noise_pred`（`:516`）在 `@torch.no_grad()` 下、本来无图，用 `torch.is_grad_enabled()` 分开<br>（走过的弯路：只在函数入口把形参转 float **没用**——调用方的局部变量要等函数返回才重新绑定，Python 3.10 改不到 `f_locals`；实测显存只掉了 0.01 GB） | **无** |
| ③ | probs 不会被任何地方读的注意力层改走 SDPA | 存的门限是 `shape[1]==attn_res²`；`aggregate_attention` 全仓库只有 `:152` 一处且 `get_cross=True`（`all_self_attention` 从不被读）；`self_step_store` 只在 `loss=False` 时写；屏蔽要求 `shape[0]==40` 而修正前向是 `latent.unsqueeze(0)` batch=1 → 屏蔽在那趟不生效 | **浮点级**（SDPA 与 baddbmm+softmax+bmm 归约顺序不同） |

③ 用 `--no-mem-attn` 可关掉验等价（`--vanilla-only` 那一档无梯度、显存够，
两条路都跑得动，比 `n_dbscan`）。已知：三题里有一题的 `Best Epsilon` 从 0.14
变成 0.17，簇数仍相同 —— 所以**这不是逐比特复现，是同分布的两次采样**，
最终判据只能是准确率能否落在发表值附近。

## 已知口径风险（先写下来，免得事后争）

1. **YOLOv9e 自己会数错。** 它是在位者选定的尺子，用它是为了跟人家的表对齐；
   「计数器 vs YOLO 一致率」继承了 YOLO 的误差，只能当相对读数。
2. **三篇论文三套评测器**（CountGD / Grounded SAM / YOLOv9），横向不可比。
   我们只跟 CountGen 自己那张表对齐。
3. **CoCoCount 是随机生成的**（`create_data_CoCoCount.py`），仓库里那份 json
   是一次抽样。换种子会换题目，报数时要说明用的是仓库自带的那份。
