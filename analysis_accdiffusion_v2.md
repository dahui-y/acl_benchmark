# AccDiffusion v2 —— 占位核查结果（2026-08-08）

论文：*AccDiffusion v2: Towards More Accurate Higher-Resolution Diffusion Extrapolation*，
Zhihang Lin, Mingbao Lin, Wengyi Zhan, Rongrong Ji。arXiv 2025-06-15，13 页，TPAMI 体例。
代码 https://github.com/lzhxmu/AccDiffusion_v2 。仓库里有全文 PDF。

核查的问题：**它有没有物体重复的定量指标？** 有的话，"尺子"这条贡献就塌一半。

## 结论：没有。而且论文里明说了两遍。

§5.2，Table 1 正下方：

> "Note that **FID, IS, and CLIP-Score may not directly indicate the presence of
> repetitive generation or local distortion** in the generated images. Therefore,
> we perform a qualitative comparison in next section to confirm the efficacy of
> AccDiffusion v2 in reducing such artifacts."

§5.5 Ablation Study 开头：

> "**Since current quantitative metrics cannot intuitively reflect the extent of
> object repetition or local distortion**, we provide visualizations to show how
> our core modules effectively prevent repetitive generation and local distortion."

Table 1 的五列是 FID_r / IS_r / FID_c / IS_c / CLIP —— 没有一列在测重复。
FID_c / IS_c 是"每张图裁 10 个原分辨率 patch 再缩到 299²"（沿用 DemoFusion 的做法），
仍然是分布距离，不数物体。**三个核心模块的消融，证据全部是图。**

这比"没被占"更强：**缺口是被在位者公开声明存在的。**

## 它的三条腿（我们要照着搭）

机制新颖不在其中。这条线上没有一篇靠机制新颖发出来。

| | AccDiffusion v1 | AccDiffusion v2 |
|---|---|---|
| ① 命名前作留下的残余失效 | DemoFusion 给每个 patch 喂同一句 prompt -> 物体重复 | v1 删了 patch 里的词 -> **局部畸变**（prompt 不再描述局部结构）；patch 之间不通信 -> 全局语义不连贯 |
| ② 结构上新的干预 | prompt 逐 patch 解耦 | ControlNet canny 补局部结构 + dilated sampling 加窗口交互 |
| ③ 领域标准指标上赢的表格 | — | Table 1，2048²/3072²/4096² 三档全面超 DemoFusion 与 v1 |

**v2 的问题正是 v1 那个修法的残余。** 谱系生成原理在同一个团队内部走了一遍。

我们的位置：① 有（ScaleDiff 自己的 Limitations，且已定位到邻域/画布比）；
② 有（单次前向内逐位置混合两套嵌入，AccDiffusion 的修法需要独立 patch 前向，
在 ScaleDiff 里结构上不可用）；**③ 零 —— 而这条腿才是真正让论文进去的。**

## 其余可用事实

- 对比方法名单（全部 training-free）：SDXL-DI, Attn-SF, ScaleCrafter,
  MultiDiffusion, HiDiffusion, DiffuseHigh, DemoFusion, AccDiffusion。
  **明确不与超分方法比**，理由是输入不同（图 vs 文）。
- 评测协议：LAION-5B 里随机 10,000 张真图作参考集，1,000 条 prompt 作输入。
  与 ScaleDiff 同源，我们照抄即可，两边可比。
- 耗时（他们的机器）：4096² 需 35 min。ScaleDiff 在我们的 4090 上是 **74.7s**。
  这个量级差要在论文里用上 —— 但注意是不同硬件，只能作数量级陈述。
- 它自己的 Limitations（Fig. 15）：超过 8K (64x) 出现 detail degradation。
  这是**它的**残余，不是我们要开的那个口子（我们的入口是 ScaleDiff 的）。
- ControlNet 依赖：v2 需要 `xinsir/controlnet-canny-sdxl-1.0` 额外权重。
  ScaleDiff 线是纯 training-free 无额外模型 —— 这是一条可用的对比维度。

## 对方法设计的直接约束

我们的干预与 AccDiffusion 的关系必须写清楚，不能回避：

- 机制（同一句 prompt 施于所有局部视野导致重复）**是 AccDiffusion v1 命名的**，
  我们继承并引用。
- 它的修法是 **patch-content-aware prompt**：给每个 patch 换一串 token，
  需要**独立的 patch 前向**。ScaleDiff 只有一次全图前向，没有地方喂第二句话。
- 我们的构造是**软的、逐位置的、单次前向内的**：两套嵌入同时存在，
  按 Phase 1 的主体图逐位置取用。
- 相似性风险真实存在（"这不就是软版的 patch-content-aware prompt 吗"）。
  防守 = "结构上不可用" + 数字，不是"我们不一样"。

而 v2 又给了一条新的防守面：它为了修 v1 的局部畸变引入了 ControlNet。
我们的做法保留全部景物词（只摘主体短语），**不会产生 v1 那种 "absence of a
prompt undermines image details"**，所以不需要 ControlNet 这类补丁。
这一点要用细节指标证明，不能只是声称。
