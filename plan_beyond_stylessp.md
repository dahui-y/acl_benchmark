# 超越 StyleSSP 的作战方案（按失效优先流水线推导）

日期 2026-08-06。目标 venue：CVPR 2027（截稿约 2026-11，剩 ~3 个月）。

## 0. 胜型（已两次验证）

> **前任的命名失效 × 一个没人用过的旋钮 × 前任自己的坐标系。**

StyleID→StyleSSP、HD-Painter→FreeInpaint 都是这个形状。
所以问题分解为：StyleSSP 留下什么失效？还剩哪个旋钮？

## 1. StyleSSP 的候选残余失效（占位核查后）

| 失效假设 | 占位状态 |
|---|---|
| ① 反演误差随链条放大 → 内容失真 | **已被占**：Dual Rectified Flows（DRF, arXiv 2511.20986）开场白就是它，"even minor inversion inaccuracies can propagate…(see StyleSSP and Zstar results)" |
| ② 生态锁死在 SDXL + ControlNet + IP-Adapter，流匹配模型上没有对应物 | **部分被占**：DRF 已做在 SD3 上（双轨迹 + 中点速度插值 + 注意力注入），且直接对比并声称赢 StyleSSP。但 DRF **无 venue、无代码线索、实验跑在 P40 上** |
| ③ 全局固定 α=0.7 → 风格家族依赖的失败（低频型风格：平涂、色场） | **未见占位，我们可测** |
| ④ 负引导在"风格图与内容图语义重叠"时误删合法内容 | **未见占位，我们可测** |

## 2. 旋钮清单盘点（关键发现）

| 旋钮 | 谁用了 |
|---|---|
| 采样期注意力 K/V | StyleID（U-Net）、DRF（SD3，照搬） |
| 起点频率操纵 | StyleSSP |
| 反演期负引导 | StyleSSP |
| 双轨迹速度插值 | DRF |
| 注入时间表/调度 | Scheduled Style Injection（预印本 2605.26538） |
| **AdaLN 调制通路（pooled → scale/shift/gate）** | **零命中。** DRF 全文 `AdaLN`/`modulat` 0 次；检索未见任何"风格经调制通路注入"的工作 |

**而调制通路恰恰是我们自己在 mmdit_probe 第一轮撞见的**：swap-txt 的 50 倍方差残
差，来源就是 pooled CLIP → AdaLN 这条 K/V 碰不到的路。当时记了一句
"任何 MMDiT 风格主张都得把它算进去"——现在它成了唯一没人碰过的旋钮。

## 3. 论文的候选故事线（若探针通过）

> U-Net 里风格住在 self-attention 的 K/V（StyleID 证明的）。
> **MMDiT 里 K/V 替换只在"无效果↔整图复制"之间插值，没有迁移区间**
> （我们 sweep 已测得：corr(1-C,dS)=0.874，24 格无一脱线）。
> 风格必须从**调制通路**进入——这是 U-Net 没有的结构。

注意：**sweep 第二轮那个"失败的"负结果直接变成这篇论文的动机实验（Fig.2）。**
一条已经付过成本的证据链，别人要复制得先想到再跑一遍。

架构事实支撑这个分工的合理性：SD3 的 AdaLN 调制向量由 pooled 文本嵌入 + 时间步
算出，**每 block 全局、不含空间信息**——天然只能载"全局外观"（调色板、色调、
纹理统计），载不了布局。全局风格走调制、内容走其余通路，是一个干净的解耦故事。
（sweep 里"颜色过来了、纹理没过来"的观察与此相容但不构成证明。）

## 4. 执行计划（每步带 kill 判据）

| 步 | 内容 | 成本 | kill 判据 |
|---|---|---|---|
| 0 | **深度占位核查**：SADis（NeurIPS 2025）的机制是不是调制注入？FreeFlux / ColorCtrl / RB-Modulation 各自碰没碰调制通路？ | 1–2 天，无 GPU | 任何一个已用调制通路做风格 → 旋钮报废，回第 1 节 ③④ |
| 1 | **调制交换探针**：复用 mmdit_probe harness，hook 从 K/V 换到 norm1/norm1_context 的调制输出，风格分支的 per-block scale/shift/gate 移植进内容分支。对照 sweep 已有的权衡线 | **0.5 天 GPU** | 落在权衡线上 → 旋钮死，方向终止 |
| 2 | 阳性对照：StyleID 原代码（SD1.5）校准 S/C/L 尺子；StyleSSP 原代码（SDXL）小子集复现 | 1 天 GPU | StyleID 不脱线 → 尺子坏，先修尺 |
| 3 | 真实风格图输入：风格分支从文本 prompt 换成风格图（CLIP image embedding 顶替 pooled 槽位，或反演风格图取轨迹调制） | 1 天 GPU | 只对生成风格图有效、对真实画作无效 → 降级为分析论文 |
| 4 | 方法成型 + 在 **StyleSSP 坐标系**评测：MS-COCO × WikiArt 800 张，ArtFID/FID/LPIPS，对 StyleID / StyleSSP / DRF | 1–2 周 | 赢不了 StyleSSP 的 LPIPS 或 FID 任何一项 → 不投主会 |

前三步合计 ≤3 GPU 天，全部有确定结论，最深的沉没成本是三天。

## 5. 必须直说的风险

1. **第 0 步可能直接杀掉整条线**（SADis 的机制我还没读，它的标题里
   "color-texture disentanglement"和调制通路的"全局外观"嫌疑很重）。
   所以第 0 步在任何 GPU 时间之前。
2. **调制只载得动全局外观**：笔触类局部纹理可能进不来。真发生的话，
   故事降级为"调色板+色调层面的解耦控制"——还有救（SADis 证明色彩解耦
   单独可发 NeurIPS），但窄很多。
3. **DRF 若中会**（现在无 venue），②那条失效轴的坐标系要换成对 DRF 比。
4. 3 个月做完 4 步偏紧；第 4 步若赶不上 CVPR 2027，顺延 ICCV 2027（2027-03）。

---

## 第 0 步结果（2026-08-06）：旋钮已被半占，方案修订

逐篇读了机制。四个原嫌疑人全部排除，但检索甩出来一个新的、致命度更高的：

| 工作 | 它的"modulation"实际是什么 | 与 AdaLN 通路 |
|---|---|---|
| SADis (NeurIPS 2025) | CLIP image embedding 加性 + 白化/上色，**经 IP-Adapter 的 cross-attention 注入** | 无关（adapter 通路；白化技巧可借用） |
| RB-Modulation (CVPR 2025) | 反向 SDE 漂移的随机最优控制 | 无关 |
| ColorCtrl | 文本→视觉注意力分数缩放 | 无关 |
| Unraveling MMDiT Blocks | 逐 block 移除/增强**文本 hidden-states** | 无关（14 个 `gate` 命中是 investi**gate** 的碎片） |
| DRF | StyleID 式注意力注入搬到 SD3 | 无关（全文 AdaLN/modulat 0 次） |
| **Modulation Guidance（Rethinking Global Text Conditioning, ICLR 2026, Yandex+Adobe, 开源 `quickjkee/modulation-guidance`）** | **就是调制空间**：y(p,t) → ŷ(p,p⁺,p⁻)，在调制空间做文本驱动的 guidance，training-free，用于生成质量/属性控制和指令编辑 | **同一空间** |

### Modulation Guidance 占了什么、没占什么

占了：**"调制空间是可干预点"这个机制主张本身**——ICLR 2026 已录用+开源。
"我们发现了这条通路"这句话从此不能作为贡献。

没占：
1. **信号源是文本**（p⁺/p⁻ prompt），不是参考图像。图像派生的调制信号没做。
2. **任务是质量/属性增强和指令编辑**，不是参考图风格迁移，没进 StyleSSP 坐标系。
3. 他们顺带证明了两件对我们有用的事：常规用法下 pooled 贡献小，
   但**放大后能造成全局外观改变（他们自己的例子：car style）**——
   即这条通路确实载得动"风格"级别的信号。

### 修订后的立论（从"新旋钮"降级为"新信号源+新任务+独有测量"）

三个差异化零件，缺一不可：
1. **不可能性测量**（独有）：MMDiT 上 K/V 替换无迁移区间（sweep，corr 0.874）
   ——论文的动机图，别人没有；
2. **图像派生的调制信号**（Modulation Guidance 只做了文本）；
3. **任务与坐标系**：参考图风格迁移，StyleSSP 的 benchmark/指标。

先例支持这个形状：StyleSSP 自己就是"InitNO 的起点旋钮 + FlexiEdit 的频率观察
+ 风格迁移任务"组合成的 CVPR。**旋钮被 ICLR 2026 合法化，反而降低了审稿人
对"调制空间干预是否合理"的质疑成本。**

### 判定：第 1 步探针照跑，framing 按上面修订

- 探针问题不变：**图像派生的调制移植，能否落在权衡线之外**。
  Modulation Guidance 的放大实验提高了它的先验成功率。
- 新增一条 kill 判据：若探针成功但效果与"文本 prompt 描述该风格 + Modulation
  Guidance"无法区分，则图像信号源这个零件不成立，方向死。
- 风险登记：这是**第三次**"以为是空位、查了才知道有人"（state change、
  MMDiT 文本流、调制空间）。但这次是在 GPU 时间之前查到的——第 0 步的存在
  就是为了这个。流程有效，保持"先查再跑"。

---

## 修正（2026-08-06，用户指出）：本方案的动机类型是错的

### 两篇在位者的动机，逐字对照

**StyleID（CVPR 2024）**
- 问题：风格迁移要训练或逐样本优化。
- 观察：self-attention 里 **K,V 载风格、Q 载结构**。
- 干预：把风格图的 K,V 换进去。
- 其内部零件（query preservation、注意力温度缩放、初始 AdaIN）**也各自对应一个它自己观察到的输出缺陷**。

**StyleSSP（CVPR 2025）** —— 原文链条：
1. **先看到输出错了**："layout changes of original content and content leakage
   from style images"，Fig.1 (a)(b) 各配一组图（那条被草坪盖掉的河）。
2. **做实验定位病因**："Through a series of experiments, we discovered that an
   effective startpoint in the sampling stage significantly enhances the style
   transfer process"；再做频域分析确认 "high-frequency components in z_T are
   more crucial in determining the layout"。
3. **干预点由病因推出**，不是由"哪个旋钮没人用"推出：布局病在起点频谱 → 削低频；
   泄漏病在反演轨迹被风格图污染 → 反演期负引导。
4. 评测沿用 StyleID 的 benchmark 与指标。

**它为什么跟 StyleID 比**：同任务、同 benchmark（MS-COCO×WikiArt）、同指标
（ArtFID/FID/LPIPS）、同 training-free 约束。**比较本身就是论证**——
"你没修的，我修了，用你的尺子量"。

### 我们方案的动机是什么类型

"U-Net 的 K/V 配方在 MMDiT 上没有对应形式。"

这是**方法可移植性**的陈述，不是**输出错了**的陈述。没有任何用户会因为
"K/V 替换在 MMDiT 上无迁移区间"而抱怨——他们抱怨的是图里布局丢了、风格图的
东西漏进来了。**我们继承了他们的任务和指标，唯独没继承他们的问题类型。**

### 更糟的一点：我把已经废弃的搜索方式换层重新引入了

`survey_free_lunch_round2.md` 的结论是"不要在清单里找空白"。然后本方案：
- 第 1 节挑架构轴，理由是 "SD3/FLUX 生态没有对应物" —— **架构层的空白**；
- 第 2 节挑调制通路，理由是 "**唯一没人碰过的旋钮**" —— **旋钮层的空白**。

**同一个错误换了两层。** 而 StyleSSP 选起点频率不是因为没人用过，
是因为它做实验测出布局病在那里。

### 一个方法论上的连带问题

当"换基座"本身就是卖点时，与在位者的比较无法隔离贡献：我们在 SD3.5 上拿到
ArtFID 更好，审稿人问的是"这是你的方法还是 SD3.5 比 SDXL 强"。
（诚实补充：StyleSSP 自己是 SDXL 对 StyleID 的 SD1.5，这个混淆该领域一定程度上
容忍。但那里基座只是背景；在我们这里基座**就是**论点，混淆会吃掉整篇论文。）

### 正确的推导顺序（本方案作废，按此重来）

1. **跑 StyleSSP 原代码，看它的输出还有什么可见的错。** 这不是"校准尺子"，
   是**找失效**——失效优先流水线第 4 步的原意。
2. 给失效命名，配一组像 StyleSSP Fig.1 那样的对照图。
3. **做定位实验找病因**（这才是 StyleSSP 那个 "series of experiments"）。
4. **干预点由病因决定**。若病因恰好指向架构或调制，那时 sweep 的测量和
   `mod_probe.py` 才有位置——作为**支持证据**，不是作为动机。

`mod_probe.py` 暂不跑：它是一个没有失效要解释的机制探针。
sweep 的权衡线测量仍然是真实结果，只是**不能当动机用**。
