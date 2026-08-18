# CountGen（make-it-count, CVPR 2025）源码通读

读的是仓库里 `help_code/make-it-count/`（用户已推送）。行号都指该目录下的文件。

**身份确认**：CountGen 就是 *Make It Count: Text-to-Image Generation with an
Accurate Number of Objects*（Binyamin et al., CVPR 2025, arXiv 2406.10210）
提出的方法名，README 抬头写明。所以它是我们在 `INCUMBENTS.md` 里筛出来的
唯一同时满足「有会议 + 有开源码」的在位者。

---

## 一、它到底怎么跑的（三段）

1. **先跑一次原版 SDXL**（`perform_counting=False`），把 cross / self attention
   存下来（`dbscan_mask_extract.py:163-173`）。
2. **数数**：取 **t=25、层 `up_52`** 的 self-attention 亲和矩阵（32×32=1024 个
   token），对称化后做 **DBSCAN 聚类**，簇数就是它认为图里有几个物体
   （`dbscan_mask_extract.py:181-205`）。
3. **决策**（`extract_mask.py:25-43` + `run_countgen.py:128-137`）：
   - 簇数 == 要求数 → **直接输出第 1 步那张原版图，不做任何干预**
   - 多了 → `relayout_overgeneration`：按面积排序，**把最小的几个 blob 删掉**
   - 少了 → `relayout_undergeneration`：用训练好的 ReLayout U-Net **一次补一个**
   然后拿修好的 mask 再生成一次，中间带梯度的 iterative refinement。

所以**每张图至少两次 SDXL 前向**；需要修正时第二次还要在 i∈{0,10,20} 三个步上
做最多 20 次「UNet forward + backward」（`self_counting_sdxl_pipeline.py:160-219`,
阈值表在 `pipeline_config.yaml` 的 `thresholds`）。这是它慢的来源。

---

## 二、代码里读出来的硬限制（每条都可指认）

### 1. 超过 9 个物体，官方代码直接跳过
`run_countgen.py:104`：
```python
if required_object_num > 9:
    print(f"Skipping ... as it is not supported.")
    continue
```
根因在 `relayout.py:7`：ReLayout U-Net 是 `in_channels=9, out_channels=10`，
**通道数写死了最多 9 个实例**。不是调参能改的，是架构上限。

而它自己的数据集生成脚本 `dataset/create_data_CoCoCount.py:38`：
```python
numbers = ["two", "three", "four", "five", "seven", "ten"]
```
仓库里那份 `dataset/CoCoCount.json` 200 题中 **33 题 int_number=10**，
**官方 pipeline 会把这 33 题全部跳过**。跑之前必须先说清这一档怎么算。

### 2. 整条链的计数分辨率只有 32×32
DBSCAN 在 1024 个 token 上聚类，参数写死 `min_samples=10`、
`min_cluster_size=15`（`dbscan_mask_extract.py:189-191, 202-204`），
再加 `remove_sparse_blobs(min_blob_size=10)`（`utils_masks.py`）。
→ **面积小于约 15/1024 ≈ 1.5% 的物体被当成噪声删掉**。1024² 的图上
那大约是 128×128 像素。小物体在这个计数器眼里根本不存在。

### 3. 只看一个时间步、一个层
`dbscan_mask_extract.py:181`：
```python
self_attn_feats_mean = get_agg_self_attn(pipe.attention_store.self_step_store, 25,
                                         layers=['up_52'], cross_mask=cross_mask)
```
config 里 `save_timesteps` 存了 9 个时间步，**实际只用了 t=25 一个**；
层也只用 `up_52` 一个。没有任何跨步 / 跨层的一致性检验。

### 4. 选 eps 的准则里没有 N
eps 在 0.1–0.2 上扫 10 个值，**按 silhouette score 挑**
（`dbscan_mask_extract.py:186-205`）。轮廓系数衡量的是「簇分得干不干净」，
不是「数得对不对」。题目里已经给了 N，这个信息在选 eps 时一点没用到。

### 5. 管线完全信任自己的计数器 —— 这条最要紧
`extract_mask.py:25-27` 置 `obj_num_match=True`，`run_countgen.py:128-129`
就直接把原版图当输出。**计数器说对了但实际错了 → 这张图永远不会被修。**

> 推论：**CountGen 的准确率上限 = 它 DBSCAN 计数器的准确率。**
> 这一条是可测的：拿 YOLOv9 去数原版图，和 DBSCAN 的数逐题对一下，
> 就能把 46% 拆成「计数器数错」与「修正没修好」两部分。
> 这个数**没有人报过**，而且它直接指出天花板卡在哪一段。

### 6. 引导损失把实例身份丢掉了
`utils/loss_utils.py:5`：
```python
foreground_mask = (desired_mask != 0).to(dtype=object_attention_map.dtype)
```
逐实例的标签在这里被压成**一张二值前景图**，cross-attention BCE 只说
「物体应该出现在这些位置」，没有说「应该是 N 个彼此分开的东西」。
实例分离完全靠另一处的 self-attention masking（见下）。

### 7. 只支持单一物体类别
`self_counting_sdxl_pipeline.py:357`：`object_token_idx = paired_indices[0][1]`
只取第一个 nummod；`utils/counting_words_extract.py` 的正则也只匹配一个
`(number)(noun)`。**"three cats and two dogs" 这一类它做不了** ——
这正好解释了 CountDiffusion 表里 GPTMultiCount 那一栏 CountGen 空着。

---

## 三、一个实现缺陷（已本地复算，见 `check_selfattn_mask.py`）

`pipeline/attention_processors.py:45`：
```python
for j in range(0, max_blob_index + 1):      # ← 从 0 开始
```
`desired_mask` 的约定是 **0 = 背景、1..N = 实例**
（`utils_masks.py: from_channels` 里 `combined_mask[...] = i + 1`，其余留 0）。
循环从 j=0 起，就把背景也算进 `blob_coordinates`，于是它**在整张 32×32 上恒为 1**。
接着 56–61 行对每个背景位置做
`attention_probs[:, x, y, blob_coordinates_flat] = 0`，
等价于**把背景 query 的整行注意力清零**。

复算（4 个实例、背景占 85.9%）：

| | blob_coordinates 置 1 比例 | 整行被清零的 query 比例 |
|---|---:|---:|
| 原码 `range(0, ...)` | **1.000** | **0.859** |
| 改成 `range(1, ...)` | 0.141 | 0.000 |

即：在 step 0–10、所有 up-block 的 32×32 自注意力上，
**背景 token 的自注意力输出被整体置零**，而不是注释暗示的「背景不许看前景」。

是不是有意为之、对最终分数有没有影响，**要跑了才知道** —— 它有可能正是把内容
挤进 blob 的真实机制。但它是一行改动、有明确的对照臂，属于可以当场测的东西。

---

## 四、评测口径（好消息）

自带评测是 **YOLOv9e（ultralytics）**，不是 CountGD 也不是 GroundingDINO
（`evaluation_script.py:13`）：数出 `class_name` 完全等于目标类的框数，
等于 N 才算对。**复现 CountGen 自己那张表用不到 GroundingDINO 血统的东西**，
而且 YOLOv9e 在 4090 上是白菜价、全自动、零标注。

两处必须先补的口径：
- 文件名格式要求 `{count}__{class}__....png`（`evaluation_script.py:61-70`），
  但 `run_countgen.py:137` 存的是 `{obj}_num={N}_seed={S}.png`，**对不上，要改名**；
  而且它同时把 `*_vanilla.png` 写进同一个目录（正好是免费的 baseline 臂，
  但会被评测脚本一起数进去，要分开）。
- 数据集里的 `ball / glove / phone` 要映射到 COCO 的
  `sports ball / baseball glove / cell phone`（`create_data_CoCoCount.py` 里有
  `coco_object_name_dict`），否则类名比对全 False。

---

## 五、跑起来还缺什么

- `pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth`（Google Drive）
- `yolov9e.pt`（ultralytics release）
- `en_core_web_trf`（spacy）
- 依赖钉在 diffusers 0.25.0 / torch 2.1.2。管线自带了 `retrieve_timesteps` 和
  `rescale_noise_cfg`，主要 API 风险在 `encode_prompt` 签名和 attention processor
  接口（它走的是老的 `attn.get_attention_scores` 路径，新版 diffusers 仍保留）。
- 显存：SDXL 1024² fp16 两次前向没问题；峰值在 iterative refinement 那段要留
  UNet 计算图，24G 应当够但要实测。
- 本容器无 GPU，跑要在 OpenBayes 上。

---

## 六、建议的第一步（一件事，一次跑出三个数）

在 CoCoCount 200 题上跑一遍，同时落三样东西：

1. **vanilla SDXL 的 YOLOv9 准确率**（`*_vanilla.png` 免费得到）
2. **CountGen 的 YOLOv9 准确率** → 对齐论文/CountCluster 表里的 46–51%
3. **DBSCAN 计数器 vs YOLOv9 的逐题一致率**（第 5 条推论）

第 1、2 项是「先复现发表数再动方法」——和风格线上先复现 StyleID 28.801 是同一套
做法，那一步被证明是有效的。第 3 项没人报过，且直接决定天花板卡在计数还是修正。
N=10 的 33 题要单独列出来说明是被官方代码跳过的，不能混进总分。
