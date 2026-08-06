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

## 跑法(在 4090 那台机器上)

```bash
git pull
cd osc_check

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -U "diffusers>=0.36" imageio imageio-ffmpeg av   # HunyuanVideo15Pipeline 需要 >=0.36

# 1. 冒烟测试：一条视频，看显存和速度
python generate.py --limit 1

# 2. 剩下 23 条(断点续跑，中断重跑同一条命令)
python generate.py
```

权重是 `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`
(**官方 tencent 仓库没有 model_index.json，diffusers 加载不了**，这条已写进注册表)。
首次运行自动下载，约 20+ GB(含 Qwen2.5-VL 文本编码器)。

OOM 的话：`python generate.py --sequential-offload`(更慢但更省)。

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
