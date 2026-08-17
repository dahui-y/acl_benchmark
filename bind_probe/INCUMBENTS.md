# 在位者清单：训练无关的组合性 / 属性绑定

建于 2026-08-16。用途：playbook 第 1 步「选一个在位者（有会议、有代码、跑得动）」。

**出处标记**（上一轮我给出 0.6369，一手表格出来是 0.6734，所以这一列现在是强制的）：

| 记号 | 含义 |
|---|---|
| **【一手】** | 从论文 PDF / 摘要页原文抠出 |
| 【二手】 | 综述页面或他人转述，**未经原文核对，不可用于任何判据** |
| 【未查】 | 还没查 |

---

## ⚠️ 第一件事：两条线用的不是同一张表

| | 表 | 报数形式 |
|---|---|---|
| SDXL / UNet 那一系 | **T2I-CompBench**（NeurIPS'23 D&B） | 逐子类（color / shape / texture 各一个数） |
| SD3.5 / MMDiT 那一系 | **T2I-CompBench++**（TPAMI 2025） | 常报 overall；四大类八子类，每子类 1000 条（700 训 / 300 测） |

**两边的数字不可直接比。** 我在对话里一度把 Detail++ 的 0.7241（++）和 SynGen 的
0.7267（原版）并列，那是错的，此处订正。

---

## A. SDXL / UNet 线 —— T2I-CompBench

数据来源：**【一手】** ELDiff (arXiv 2606.20924) Table 2，从 PDF 逐行抠出。
该表只列了三个训练无关方法，**它是作者自选的对比集，不是排行榜。**

| 方法 | venue | 代码 | color | shape | texture | 延迟 | 要测试期优化？ |
|---|---|---|---|---|---|---|---|
| SDXL 基线 | — | — | **0.6734** | **0.5064** | **0.6243** | 8.07s | — |
| **SynGen** | NeurIPS'23 | ✅ | **0.7267** | **0.5249** | 0.6476 | 11.62s | 是（注意力损失） |
| **InitNO** | CVPR'24 | ✅ | 0.6823 | 0.5229 | **0.6547** | 19.42s | 是（初始噪声） |
| R2F | 【未查】 | 【未查】 | 0.7135 | 0.5194 | 0.6625 | 10.14s | LLM 参与 |
| ELDiff | arXiv 2606，无 venue | ✅ | 0.7768 | 0.5663 | 0.6941 | 8.07s | **要训练**，不同档 |

**这一系里未进 ELDiff 对比集、需要补的：**

| 方法 | venue | 代码 | 数字 | 备注 |
|---|---|---|---|---|
| Attend-and-Excite | SIGGRAPH'23 | ✅ | 【未查】 | latent 梯度优化 |
| Structured Diffusion | ICLR'23 | ✅ | 【未查】 | 语言结构进 K/V，**无优化** |
| Divide-and-Bind | BMVC'23 | ✅ | 【未查】 | |
| CONFORM | CVPR'24 | ✅ | 【未查】 | 对比损失 |
| **ToMe** | **NeurIPS'24** | ✅ | 【未查】 | token 合并 + **熵/绑定双损失迭代**（要优化） |
| SE-Guidance | Pattern Recognition 2025 | 【未查】 | 【未查】 | |
| **Detail++** | arXiv 2507，无 venue | ❌ "will be released" | color 0.7241 / texture 0.5582【二手】 | **用的是 ++**；多分支 + Centroid Alignment Loss（要优化） |
| STEDiff | 【未查】 | 【未查】 | 【未查】 | 文本嵌入空间 + [EOT] token |
| ColorWave | **WACV 2026** | 【未查】 | — | 做的是 IP-Adapter 的 RGB 级精确配色，**问题不同**，术语撞车 |
| EPIC | arXiv 2605.11722 | 【未查】 | 【未查】 | predicate-guided，含重采样 → 贵 |

---

## B. SD3.5 / MMDiT 线 —— T2I-CompBench++

| 方法 | venue | 代码 | 数字 | 要优化？ |
|---|---|---|---|---|
| SD3.5 基线 | — | — | CompBench++ **56.92** / GenEval **66.42**【一手】 | — |
| **TexTailor** | **ECCV 2026** | ❌ **摘要页无代码链接** | CompBench++ **63.00** / GenEval **71.63**【一手】 | **否，纯推理期** |
| Stitch | arXiv 2509.26644 | 【未查】 | 【未查】 | MMDiT 的**位置控制**，不是属性绑定 |

TexTailor 的机制：逐 block 做 remove / disable / enhance 的消融，发现
**语义在靠前的 block 涌现、细节在靠后的 block**，然后按 block 裁剪文本引导。
声称 Shape +12% / Texture +10% / Color +8%【一手摘要】。

---

## 这张表已经能回答的两件事

### ① 「为什么是 SynGen」——**不该是。**

我原来的依据只有 ELDiff Table 2，而那是作者自选的三个方法。SynGen 是 NeurIPS'23，
到现在快三年。而且它**要测试期优化**，我们的主张若是「不做优化」，
拿它当在位者反而是打一个已经过时的靶。

### ② 真正的分叉：**最好的方法没有代码**

| backbone | 最强在位者 | 能跑吗 | 后果 |
|---|---|---|---|
| SDXL | SynGen (0.7267) | ✅ | **能跑，但是 2023 年的** |
| **SD3.5** | **TexTailor (ECCV'26, 63.00)** | ❌ **无码** | **数字要比，但没法「跑它、看它在哪坏」** |

playbook 第 2 步要求「跑它」。**在 SD3.5 上，第 2 步做不了**——只能拿 SD3.5 裸基线
当起点，把 TexTailor 当成一个必须超过的**数字**而不是一个可解剖的**对象**。

这不是死路（AccDiffusion 也没解剖 FouriScale），但要说清楚：
在 SD3.5 上我们是「对着一个数字做」，在 SDXL 上才是「对着一个可跑的对象做」。

---

## 还欠的（下一轮补）

1. **A 表里 8 个【未查】的方法**：venue / 代码 / 原表数字。重点是 **ToMe (NeurIPS'24)**
   —— 它是 A 系里最新的、有码的、且明确做 semantic binding 的。
2. **每个数字的协议**：多少 seed、什么 CFG、CompBench 还是 ++。ELDiff Table 2
   没写协议，直接比可能有系统偏差。
3. **TexTailor 有没有放出代码**（摘要页没有，要去 project page / GitHub 再找一遍）。
4. **SD3.5-Medium 在 24GB 上的实测吞吐**（估计 1024² / 28 步约 4–6 s，未实测）。


---

# 第二轮补充（2026-08-16）：补 ①（未查方法）与 ③（TexTailor 代码）

## ③ TexTailor 有没有代码 —— **没有**

摘要页、arXiv v2、以及一轮专门检索都没找到 GitHub 或 project page。
但补到两条比代码更重要的：

* **它不只在 SD3.5 上做**：原文说跨 **SD3.5 / FLUX / Qwen-Image** 三个 backbone，
  跨 GenEval / T2I-CompBench 两张表，跨生成 / 编辑 / 加速三个任务。【一手】
* **它是纯推理期，不做测试期优化。**【一手】

## ① 未查方法补齐

| 方法 | venue | 代码 | backbone | **要测试期优化？** | 备注 |
|---|---|---|---|---|---|
| Structured Diffusion | ICLR'23 | ✅ | SD1.x | **否** | 语言结构进 K/V。老、弱 |
| Attend-and-Excite | SIGGRAPH'23 | ✅ | SD1.x | 是 | latent 梯度 |
| SynGen | NeurIPS'23 | ✅ | SD1.4/2.1（SDXL 上被别人报过数） | 是 | 句法 + 注意力损失 |
| **CONFORM** | CVPR'24 | ✅ gemlab-vt/CONFORM | **SD v1.5**【一手】 | 是 | 对比损失。**不是 SDXL，不可直接比** |
| InitNO | CVPR'24 | ✅ | SD | 是 | 初始噪声优化 |
| **ToMe** | **NeurIPS'24** | ✅ hutaihang/ToMe（**确认已放出**：run_demo.py / configs / environment.yaml） | README 未写 | **是** | token 合并 + **熵损失 + 绑定损失迭代更新** composite token【一手摘要】 |
| **R2F** | **ICLR'25 Spotlight** | ✅ krafton-ai/Rare-to-Frequent | **SDXL / SD3 / FLUX / IterComp**【一手】 | **否** | LLM（GPT-4o 或 **LLaMA3**）规划 rare→frequent 概念引导。T2I-CompBench +0.1~3.6 pp |
| Detail++ | arXiv 2507，无 venue | ❌ "will be released" | SDXL | 是（Centroid Alignment Loss） | 用 CompBench**++** |
| STEDiff | 【未查】 | 【未查】 | 【未查】 | 【未查】 | [EOT] token + 语义增强损失 → 大概率要优化 |
| **TexTailor** | **ECCV 2026** | ❌ | **SD3.5 / FLUX / Qwen-Image** | **否** | 逐 block 裁剪文本引导 |

**本轮新打到、尚未归档的：** CARINOX（2509.17458，类别感知奖励的初始噪声优化，
属于 inference-time scaling 那一档，贵）、"Infinity and Beyond: Compositional
Alignment in VAR and Diffusion T2I Models"（2512.11542，是研究不是方法）。

---

# 这张表现在给出的结论

## 一条清晰的结构：**「不做优化」这一栏几乎是空的，而且空得有理由**

| | 要测试期优化 | 不要 |
|---|---|---|
| 有 venue + 有码 | A&E、SynGen、CONFORM、InitNO、ToMe（**五个**） | Structured Diffusion（ICLR'23，老且弱）、**R2F（ICLR'25 Spotlight）** |
| 有 venue、无码 | Detail++（无 venue）… | **TexTailor（ECCV'26）** |

**这一栏在 2025 年之前基本是空的**，而 2025–26 出现的两个填补者
（R2F、TexTailor）都不是靠自扰动，而是靠**引入外部结构**：
R2F 引入 **LLM 的概念知识**，TexTailor 引入 **block 级功能的先验**。

这与 §11.12 那条一般性怀疑一致：
> 用「退化模型自身」造的信号，方向是模型自己已经相信的那个，包括它错的那个。
> **要修绑定，必须引入模型之外的东西。**

C2 死于此。R2F 用 LLM 绕开、TexTailor 用 block 结构绕开，都是**外部**的。

## 在位者是谁：**R2F**

按「能跑 > 够新 > 分数 > 失效可见」排：

* **能跑** ✅ 代码在 krafton-ai/Rare-to-Frequent
* **够新** ✅ ICLR'25 **Spotlight**
* **backbone 对** ✅ 支持 SDXL **和 SD3 和 FLUX** —— 无论我们最后选哪个都能比
* **与我们的立场一致** ✅ 不做测试期优化
* **有可攻击的弱点** ✅ **每条 prompt 要一次 LLM 调用**

最后一条是重点：R2F 的代价不是 GPU 秒数，是**一次 LLM 调用**。
「不需要 LLM 也能拿到同样甚至更好的绑定」是一个**具体、可测、审稿人听得懂**的
贡献陈述，而且它不要求我们找到任何「未被认领的机制」。

## 还欠

1. R2F 在 T2I-CompBench 上的**逐子类原表数字**（ELDiff 转述的 color 0.7135 是【二手】）
2. R2F 用 LLaMA3 时的实际显存与耗时（GPT-4o 在国内不通；LLaMA3-8B 本地可跑，
   但要看能不能与 SDXL 分时共存于 24GB —— 规划可以离线预算好，应当不难）
3. STEDiff 的 venue / 码
4. SD3.5-Medium 在 24GB 上的实测吞吐
