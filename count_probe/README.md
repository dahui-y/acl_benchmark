# count_probe：复现 CountGen 并把它的准确率拆开

三步，一条命令接一条。目的不是"跑通"，是**先对齐发表数，再看天花板卡在哪一段**
——和风格线上先复现 StyleID 的 28.801 是同一套做法。

## 0. 先备齐四样东西

```bash
cd help_code/make-it-count
# 1) ReLayout 权重（Google Drive，见 README）
#    → pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth
ls pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth
# 2) YOLOv9e（评测器，官方指定）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt
# 3) spacy 解析器
python -m spacy download en_core_web_trf
# 4) SDXL base 1.0；本地权重就用 SDXL_PATH 指过去
export SDXL_PATH=/path/to/stable-diffusion-xl-base-1.0
```

`relayout_undergeneration` 里那句 `torch.hub.load('mateuszbuda/brain-segmentation-pytorch', ...)`
要连 GitHub。连不上时 `countgen_batch.py` 会告诉你怎么把 hub 缓存拷过来。

## 1. 跑批

```bash
export SD_OUT=/openbayes/input/input0/Sim2Struct-1000/temp/scalediff_out
# 先冒烟 5 题，确认能跑通、量一下单张耗时
python count_probe/countgen_batch.py --limit 5 --out $SD_OUT/count/cocoount
# 全量 200 题
python count_probe/countgen_batch.py --out $SD_OUT/count/cocoount
```

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

## 其他

- `check_selfattn_mask.py`：复算 `attention_processors.py:45` 的
  `range(0, ...)` 使 `blob_coordinates` 恒为全 1、背景 query 整行注意力被清零。
  零依赖、零 GPU，`python count_probe/check_selfattn_mask.py` 直接看。
- `CODE_READ_countgen.md`：源码通读，硬限制逐条带行号。
- `INCUMBENTS.md`：这一格的在位者与「SOTA 到天花板」的距离。

## 已知口径风险（先写下来，免得事后争）

1. **YOLOv9e 自己会数错。** 它是在位者选定的尺子，用它是为了跟人家的表对齐；
   「计数器 vs YOLO 一致率」继承了 YOLO 的误差，只能当相对读数。
2. **三篇论文三套评测器**（CountGD / Grounded SAM / YOLOv9），横向不可比。
   我们只跟 CountGen 自己那张表对齐。
3. **CoCoCount 是随机生成的**（`create_data_CoCoCount.py`），仓库里那份 json
   是一次抽样。换种子会换题目，报数时要说明用的是仓库自带的那份。
