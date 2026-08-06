# StyleSSP 的引用图：已被认领的失效 vs 剩下的

日期 2026-08-06。输入：用户提供的 StyleSSP 引文列表。
**这一轮检索是合法的**——它不是"找空位"，而是问：
**哪些失效已经被后继者命名并认领了？** 这是正确顺序里的第 5 步，
只是提前用在"我们该不该进这一格"上。

---

## 1. 已录用的后继者：命名的失效 × 认领的旋钮

| 工作 | venue | 它命名的失效 | 它拿走的旋钮 |
|---|---|---|---|
| **StyleFM** | **AAAI 2026** ✅代码 `YingnanMa/StyleFM` | **内容与风格表征在频域重叠** | **三分频段 + "buffer band" 专门处理重叠**；递归注意力做时序一致注入 |
| **FreSCo** | ICMR 2026 | 忽视频域区分 → content drift + style leakage | DWT 小波分解，风格只注入高频纹理子带；VAE 压缩掩码 |
| **TransferAnything** | ICASSP 2026 | 内容保持与风格强度失衡 | **频率感知的 latent 优化** |
| **ACID-Style** | AAAI 2026 | 内容/风格注入的**时机**不对 | 两个轻量 adapter + **自适应注入时机** |
| **AHAI** | ICASSP 2026 | （未取得摘要） | 自适应混合注意力推理 |
| **SAAST** | IEEE Trans. 2026 | 分步的内容/风格学习 | step-aware 学习 |
| **Dual-seed EA** | AAAI 2026 | 噪声优化 | 演化算法搜初始噪声 |
| **Scheduled Style Injection** | arXiv | 风格-内容 Pareto 前沿 | 注入时间表 |
| **DRF** | arXiv | 反演误差沿链放大 | 双轨迹 + 中点速度插值 |

---

## 2. 结论一：**频域这条轴已经被三篇已录用工作占满**

我们上一轮提的"强版假设"——
**内容细节与风格纹理住在同一频段，全局逐频率标量无法分离**——

**StyleFM（AAAI 2026）原文**：*"tripartite frequency design with a
**buffer band accounting for the overlap of content and style representations**"*

**这就是我们的假设，一字不差，而且它给了解法（三分频段 + 重叠缓冲带），还开源。**

加上 FreSCo（小波空间局部化）和 TransferAnything（频率感知 latent 优化），
**2026 年一年内三篇已录用工作占了同一条轴。** 频域方向彻底死透，
连"errata"级的弱版都没意义了——StyleFM 的三分设计本身就覆盖了 StyleSSP 带通的问题。

---

## 3. 结论二：**旋钮清单已经被扫干净了**

把这一格所有已录用工作的旋钮列出来：

| 旋钮 | 认领者 |
|---|---|
| 采样期注意力 K/V | StyleID (CVPR24)、DRF |
| 起点频率操纵 | StyleSSP (CVPR25) |
| 反演期负引导 | StyleSSP (CVPR25) |
| 频域分带 / 重叠处理 | **StyleFM (AAAI26)**、FreSCo (ICMR26)、TransferAnything (ICASSP26) |
| 注入**时机 / 调度** | **ACID-Style (AAAI26)**、Scheduled Style Injection |
| 注意力混合策略 | **AHAI (ICASSP26)** |
| 初始噪声搜索 | **Dual-seed EA (AAAI26)** |
| 分步内容/风格学习 | SAAST (IEEE Trans 26) |
| 反演轨迹 / inversion-free | DRF |
| AdaLN 调制通路 | Modulation Guidance (ICLR26，非风格任务) |

**我们之前盘点时以为"旋钮清单还没用完"（`survey_failure_first_method.md` 附 2 第 3 条）。
这份引用图证明那句话已经过期：2026 年一年之内，清单被扫完了。**

---

## 4. 这对方向意味着什么

### 冷冰冰的事实

- StyleSSP 是 CVPR 2025。**一年之内至少 8 篇后继者**，其中 **4 篇顶会/A 类已录用**
  （StyleFM、ACID-Style、Dual-seed EA @AAAI 2026；TransferAnything、AHAI @ICASSP 2026）。
- 我们上一轮找到的假设，被 **AAAI 2026 一篇开源工作**逐字覆盖。
- 剩余可用旋钮：**0 个明显的**。

### 这一格对我们的对位

**换手周期 ≈ 3–6 个月，而我们光是搭 StyleSSP 环境就要 2–3 天、
找失效要 1 周、做方法要 1 个月。** 等我们跑出结果，
这份引用列表上会再多五篇。

**这不是"我们找得不够好"，是任务本身的边际实验太便宜**——
挑个旋钮、跑 MS-COCO×WikiArt 800 张、写出来。
一个中等规模的组一个季度能出三篇。

---

## 5. 判定

**风格迁移这一格，作为"我们做 method 论文"的目标，到此为止。**

不是因为没问题可做，而是因为**问题被消耗的速度快过我们能工作的速度**。

保留的资产（不作废）：
- `analysis_styleid_vs_stylessp.md` 里那两条**发现机制**（把旋钮推到极限看什么不动；
  找前任"碰过但没深究"的东西）——这是方法论，跨领域有效
- `survey_failure_first_method.md` 的四环链和 break test 纪律
- `mmdit_probe/` 的实验设计基建（双分支同 batch、权衡线、消融 delta、阳性对照）
- StyleSSP 环境已建好、SDXL 权重已就位——如果将来要用，成本已经付了

**下一步不是再选一个格。**
下一个决定应该是：**在什么条件下，我们的工作速度能快过问题被消耗的速度？**
这个问题没回答之前，选任何一格都是同样的结局。
