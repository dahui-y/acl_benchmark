# "Free lunch" 这条线：只认**开源 + 已录用**的工作

调研日期 2026-08-06。筛子由用户定：**必须有开源代码，且已被会议录用**。
arXiv 预印本单列，不与已录用工作同等对待。

方向定义：**不训练、不改权重，在推理期让模型（或一个判据）改进它自己的生成。**

> **核心结论：这条线的机器早就成熟且开源，2023–2025 一路铺到今天。**
> UniRect-CoT（"Free Lunch"，2026-04，**无代码、无 venue**）的增量只有一条：
> **把 critic 换成统一模型自己的理解分支**。其余零件全是已录用且开源的旧件。
> 而且它**没有和这条线上任何一个已录用方法比过**——只和 base model 比。

---

## 1. 祖先（2023–2024，全部开源 + 已录用）

| 工作 | venue | 代码 | 这条线上的贡献 |
|---|---|---|---|
| **Prompt-to-Prompt** | ICLR 2023 | ✅ `google/prompt-to-prompt` | 交叉注意力可编辑 |
| **Attend-and-Excite** | SIGGRAPH 2023 | ✅ `yuval-alaluf/Attend-and-Excite` | 推理期改注意力，无需外部 critic |
| **Universal Guidance** | CVPR 2023 W | ✅ `arpitbansal297/Universal-Guided-Diffusion` | 任意外部模型当 critic 引导去噪 |
| **FreeDoM** | **ICCV 2023** | ✅ `vvictoryuki/FreeDoM` | 训练自由的能量引导 + time-travel |
| **DOODL** | **ICCV 2023** | ✅ `salesforce/DOODL` | **把损失回传进 latent**（UniRect-CoT 的核心动作） |
| **LMD** | TMLR 2024 | ✅ `TonyLianLong/LLM-groundedDiffusion` | LLM 先出布局再生成 |
| **SLD** | **CVPR 2024** | ✅ `tsunghan-wu/SLD` | **检测器 + LLM 自我纠错**（生成后重绘） |
| **ReNO** | **NeurIPS 2024** | ✅ `ExplainableML/ReNO` | 用奖励模型优化初始噪声 |

---

## 2. 2025：这一年的两块基石（**都开源、都已录用**）

| 工作 | venue | 代码 | 机制 |
|---|---|---|---|
| **Z-Sampling**（Zigzag Diffusion Sampling） | **ICLR 2025** | ✅ `xie-lab-ml/Zigzag-Diffusion-Sampling` | 副标题就是 **"Diffusion Models Can Self-Improve via **Self-Reflection**"**。利用去噪与反演之间的**引导间隙**逐步累积语义，全程不训练、不需要外部 critic |
| **Golden Noise for Diffusion Models** | **ICCV 2025** | ✅ `xie-lab-ml/Golden-Noise-for-Diffusion-Models` | 学一个"噪声变换器"把随机噪声变成"黄金噪声" |
| **Frame Guidance** | 2025（video） | ✅ `agwmon/frame-guidance` | 视频扩散的帧级训练自由引导 |
| **TITAN-Guide** | **ICCV 2025** | ❌ 未找到仓库（猜名未命中，**不等于没有**） | 推理期对齐，声称解决 DOODL 的显存问题 |

**Z-Sampling 尤其要紧**：它是 "free lunch / 自我改进" 这个说法在 2025 年**已录用且开源**的
代表作，而且它**根本不需要一个 critic**——靠去噪/反演的不对称性自己积累语义。

---

## 3. 2026：热闹，但几乎全是无 venue 的预印本

| 工作 | 时间 | venue | 代码 |
|---|---|---|---|
| **UniRect-CoT / "Free Lunch"** | 2026-04 | ❌ arXiv | **❌ 无**（已核实） |
| FiRe | 2026-04 | ❌ arXiv | 未核实 |
| Iterative Partial Refinement (IPR) | 2026-05 | OpenReview 在审 | 未核实 |
| Adaptive Inference-Time Scaling via Early-Step Latent Verification | 2026-06 | ❌ arXiv | 未核实 |
| Flash-BoN | 2026-07 | ❌ arXiv | 未核实 |
| Beyond VLM-Based Rewards: Diffusion-Native Latent Reward | 2026-02 | ❌ arXiv | 未核实 |
| Prism（离散扩散语言模型的自验证） | 2026-02 | ❌ arXiv | 未核实 |

**已录用的 2026 工作，在这条线上我一个都没打到。** 这有两种读法，我分不开：
一是这条线在 2026 还没走完评审周期；二是我的检索没覆盖到。
**这一格是"未核实"，不是"确认为空"。**

---

## 4. 把 UniRect-CoT 拆开，逐个零件找祖先

| 零件 | 谁先做的 | 是否已录用+开源 |
|---|---|---|
| 前瞻估计干净图 `ẑ₀\|t` | loss-based guidance 的标配 | 是 |
| **语义损失回传进 latent** | **DOODL** | **ICCV 2023 ✅** |
| 训练自由的引导框架 | FreeDoM / Universal Guidance | ICCV 2023 ✅ |
| CLIP 当损失 | Universal Guidance / ReNO | NeurIPS 2024 ✅ |
| 多候选择优 | ReNO / Golden Noise / best-of-N | ICCV 2025 ✅ |
| 自我改进、不用外部 critic | **Z-Sampling** | **ICLR 2025 ✅** |
| 自我纠错（检测器+LLM） | **SLD** | **CVPR 2024 ✅** |
| **critic = 模型自身的理解分支** | **UniRect-CoT** | ❌ 无 venue 无代码 |
| 窗口 [5,10]、K=3 | 网格搜出来的超参 | — |

**只有倒数第二行是新的。而且它还打折**：CSA 损失和 GITO 择优**都走 CLIP**，
所谓"内在理解"只负责吐一句 caption。

---

## 5. 由此看到的两个真空

**A. 没有人做过"critic 是谁"的受控比较。**

UniRect-CoT 的对照只有 base model（BAGEL 0.776→0.799、OmniGen2 0.786→0.799），
**没有和 DOODL / FreeDoM / Universal Guidance / ReNO / SLD / Z-Sampling 中的任何一个比过**。

所以这个问题至今没有答案：

> 在同一套训练自由的引导框架下，同一个基座、同一个基准上，
> **外部 CLIP（ReNO 一路）、外部检测器+LLM（SLD 一路）、无 critic 的自反射（Z-Sampling 一路）、
> 模型自身理解分支（free lunch 一路）**，到底谁强、强在哪类 prompt 上？

对照组**全部开源且已录用**，不依赖任何拿不到的东西。

**B. Z-Sampling 没有被搬到统一模型上。**

它是 ICLR 2025、开源、"自我改进"这条线最干净的代表，但做在标准扩散模型上。
**统一模型有一个标准扩散模型没有的东西：一个能回答问题的理解分支。**
Z-Sampling 的自反射 + 统一模型的理解分支，是否互补？没人测过。

---

## 6. 可行性（沿用之前实测的参数量）

- 对照组 ReNO / SLD / Z-Sampling / FreeDoM 多在 SD / SDXL 规模 → **4090 跑得动**
- 统一模型侧用 **Janus-Pro-1B / Show-o**；**BAGEL 14.7B / 27.4GB 不碰**
- **DOODL 要 EDICT 可逆采样、双倍模型驻留** → 单独评估，可能跑不动

---

## 7. 下一步（先证伪，再推荐）

1. **拉 Z-Sampling 和 ReNO 两个仓库，确认能跑通**，并测能不能接到统一模型上。
   ——这是整条路的一票否决：对照组接不上统一模型，受控比较就做不成。
2. 把 2026 那批预印本的**代码状态逐个查实**（去项目页，不是猜仓库名）。
   若其中有已开源的，第 3 节要重排。
3. TITAN-Guide（ICCV 2025）的仓库要找到——它声称解决 DOODL 的显存问题，
   而显存正是我们的约束。
