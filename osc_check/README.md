# HunyuanVideo-1.5 问题核验：状态变化失败在我们的设置下存不存在

方法路线的第 0 个检查点。OSCBench 报告的失败是在 720p、A14B 级算力上测的；
我们要用的基座是 **HunyuanVideo-1.5 在 480p、单张 4090** 上——
**在自己手里复现出问题本身，是做任何方法之前的第一件事。**
问题若在这个设置下不显现（或者模型根本跑不动），后面全部计划重排。

## 这一步回答三个问题

1. **跑不跑得动**：HunyuanVideo-1.5(8.3B, 文本编码器是 Qwen2.5-VL)在 24GB 上的
   实测显存与 s/video —— 注册表里这条一直是 `verified: False`
2. **问题在不在**：OSCBench 说 peeling/coating/pressing 最差、rolling/heating 最好。
   我们按它的分层选了 24 条(弱 12 / 中 4 / 强 8，三个 split 都覆盖)，
   一眼看帧条就知道弱类是不是真的不变化
3. **失败长什么样**：末态从第一帧就画好？动作演了但状态不变？还是物体直接画错？
   ——这决定方法往哪个部位下手

## 跑法：先用已经在机器上的 Wan2.2-TI2V-5B

**HunyuanVideo-1.5 在国内暂时下不了**（见下节）。而这一步问的是"问题在不在、
长什么样"，用哪个基座都能答——**TI2V-5B 权重已在这台机器上、已实测过
208 s/条 / 峰值 11.66 GB**，零下载，现在就能开跑。

```bash
git pull
cd osc_check
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install imageio imageio-ffmpeg av      # 抽帧条要用

# 1. 冒烟：一条
python generate.py --model wan2.2-ti2v-5b-480p --limit 1

# 2. 剩下 23 条（约 80 分钟，断点续跑）
python generate.py --model wan2.2-ti2v-5b-480p
```

**代价要记住**：TI2V-5B 不在 OSCBench 的表里，所以这批结果**不能和已发表数字并排放**。
它回答"问题在不在"，不回答"我们比 0.524 高多少"。后者要等 HunyuanVideo-1.5。

而且 TI2V-5B 比表里四个开源模型都小，**大概率失败得更多**——
所以"问题存在"这一格几乎必然通过，信息量在**弱/强动词档的模式**和**失败长什么样**上，
不在"失败率高"这个事实上。

### HunyuanVideo-1.5：下载受阻，单独解决

```bash
python generate.py --model hunyuanvideo-1.5-480p --limit 1   # 有外网时
```

`HF_ENDPOINT=https://hf-mirror.com` **已经失效**——实测它现在只做 308 重定向回
huggingface.co，不再代理，所以国内会报
`LocalEntryNotFoundError / Distant resource does not seem to be on huggingface.co`。

已核实的现状：
- ModelScope 有 `Tencent-Hunyuan/HunyuanVideo-1.5`，但是**官方原始格式**——
  没有 `model_index.json`，而且**连 text_encoder 目录都没有**，diffusers 装不进去
- ModelScope 上**没有** diffusers 转换版
- 试过的其他镜像（gitmirror / sukaka / gitee / aifasthub）从我这边都不通，
  但**这不能证明国内也不通**，值得你在服务器上各试一次

可行的路：在英国的 Mac 上 `hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`
然后传到 OpenBayes 数据集。体积不小（含 Qwen2.5-VL 文本编码器），先在 Mac 上看一眼实际大小再决定。

## 传回来什么

每条视频旁边自动生成一张 8 帧条(`*_strip.png`，首帧到末帧均匀采样)：

```bash
tar czf osc_check_out.tgz videos/hunyuanvideo-1.5-480p/*_strip.png \
    videos/hunyuanvideo-1.5-480p/manifest.jsonl
```

约 24 张 PNG，几 MB。**视频本身先不用传。**

## 判读标准(拿到帧条后)

| 看到什么 | 结论 |
|---|---|
| 弱类(peel/coat/mash)物体全程不变，强类(roll/melt)有变化 | **问题复现**，与 OSCBench 分层一致 → 方法路线开工 |
| 末态第一帧就在(削好皮的土豆直接出现) | 问题复现，且失败模式 = 体态试跑看到的那种 → 方法要管**初态锚定** |
| 弱类也大都变化正常 | 480p/HunyuanVideo-1.5 上问题不显著 → **重新评估**，可能要换基座或升分辨率 |
| 物体/动作大面积画错(画的根本不是 prompt 里的东西) | 失败在更上游，先解决 prompt 遵循，状态变化无从谈起 |

## prompt 子集怎么选的

`select_prompts.py`，确定性选取(每个 (动词,split) 槽位按文件顺序取首个匹配)：

- **弱** peeling ×4 / coating ×4 / squeezing+crushing+mashing ×4 —— OSCBench 图 5 最差档
- **中** chopping ×2 / slicing ×2 —— 它们图 4 的示例族
- **强** rolling ×4 / melting ×2 / browning ×2 —— 图 5 最好档
- 覆盖 regular / novel / compositional 三个 split(novel 是它们报告退化最狠的)

强类在场是**对照**：如果连 rolling 都不动，那是模型/设置的问题，不是状态变化特有的失败。
