# 两条谱系并排：顶会级研究问题是怎么被"找"出来的

日期 2026-08-07。触发：用户的问题——
> "每次新文章出现都觉得这个最新文章都做得很好了，很难找出新方法进行突破了，
> 然后不知道为什么还是有人能进一步进行方法创新发顶会。"

这份文档回答两件事：
1. 风格迁移 / 高分辨率外推 这两条链，**每一代的问题是从哪来的、方法怎么赢前任的**；
2. 为什么"看起来已经做完了"这个感觉几乎总是错的。

---

## 0. 先回答那个感觉：为什么每篇新论文都像是"做完了"

三个独立的原因叠在一起，制造了这个错觉。**三个都不是关于领域的事实，是关于你的观察渠道的事实。**

### ① 你看到的图是作者挑的

论文里的 Fig. 1 / Fig. 5 是**为了展示"我修好的那条失效"而选的**。
作者不会放一张"我的方法在这里也塌了"的图。所以从论文看，方法总是成功的。

> **只有跑它、用你自己的输入、看它的输出，才能看到没被挑进论文的那一半。**

这是我们这一轮最重要的操作教训：我们枪毙了六个方向，**六个全部死在检索台上，零个死在实验台上**。
检索能告诉你"什么被占了"，永远不能告诉你"什么是坏的"——因为一条被写进文献的失效，
定义上已经被修了。

### ② 修复本身在制造下一个问题

这是这两条链真正的生成机制。看具体的：

- StyleID 为了防止内容被打乱，加了 query preservation，一个**全局标量 γ**
  → 所有区域一视同仁 → 布局仍会变。
- StyleSSP 为了压布局变化，在 z_T 上削低频，一个**全局逐频率标量 α**
  → 但内容和风格在频域是重叠的 → 一刀切不开。
- StyleFM 为了处理重叠，做三分频段 + buffer band
  → 但**分带边界仍然是全局固定的**，而重叠程度逐图不同 → ？

每一代的修法都是"在前任的粗粒度上加一层结构"，而**新加的那层结构自己又是粗粒度的**。
这个过程不收敛，因为任务目标是权衡（风格 ↔ 内容 / 细节 ↔ 全局语义），
**权衡问题没有"解完"的状态，只有前沿上的位置**。每挪一次位置，就暴露一批上一个位置看不见的东西。

### ③ "很完备"是论文修辞的产物

每篇论文在写作上都**必须**把自己写成一个闭环：摘要不承认遗留限制，
相关工作把前任写成有明确缺陷、把自己写成填上了那个缺陷。
AccDiffusion v2（TPAMI 2025）的摘要里一条遗留限制都没有——这不代表没有，
只代表摘要不是用来写遗留限制的地方。

> **推论：一篇论文越是读起来"没留下什么"，越说明你只读了它的自我叙述。
> 残余不在论文里，在它的输出图里。**

---

## 1. 风格迁移：StyleID → StyleSSP → StyleFM

### 逐代拆解

| 代 | 它命名的失效（问题从哪来） | 它怎么定位到病因（发现机制） | 干预在哪 | 它留下的残余（被下一代点名） |
|---|---|---|---|---|
| **StyleID**<br>CVPR 2024<br>SD1.5，无 adapter | 已有 training-free 方法要么 per-style 优化，要么把内容搞烂 | **把自己的杠杆推到极限，看什么还不动**：Fig.5 同时注入 style 的 Q、K、V，**颜色依然跟着 content 走** ⇒ 颜色不住在 self-attention 里 ⇒ 去看初始 latent | 解码器 self-attention 换 style 的 K,V；query preservation（γ）；attention temperature scaling；初始 latent AdaIN | **布局仍会变；风格图的"内容"泄漏进结果** |
| **StyleSSP**<br>CVPR 2025<br>SDXL + IP-Adapter + ControlNet | 直接点名上面两条 | **找前任"碰过但没深究"的旋钮**——原话：StyleID manipulates the startpoint *"but only by rescaling … **without fully investigating its role in style transfer**"*；再用 "a series of experiments" 把因果指到采样起点 z_T | 对 DDIM 反演得到的 z_T 做 FFT、削低频（α=0.7）；反演期用 style image 做负条件的负引导（ω=1.5） | **内容与风格在频域重叠，全局逐频率标量分不开** |
| **StyleFM**<br>AAAI 2026<br>SD1.4，LDM 代码库 | 直接点名上面那条：*"buffer band accounting for the **overlap of content and style representations**"* | （待读原文/跑图确认） | 三分频段 + 重叠缓冲带；递归注意力做注入的时序一致 | **？ ← 这是我们的位置** |

### 它是怎么"赢"前任的

**在前任自己的坐标系里赢，一个数都不换。**
StyleSSP 用的是 StyleID 的评测协议（MS-COCO × WikiArt，800 张）：

| | ArtFID ↓ | FID ↓ | LPIPS ↓ |
|---|---|---|---|
| StyleID | 28.80 | 18.13 | 0.5055 |
| StyleSSP | **21.50** | **13.45** | **0.4881** |

这一点是纪律性的：**新方法不许换 benchmark**。换了 benchmark 的提升，审稿人不认。

### 三代共享的动作序列

```
跑前任 → 看到一张坏图 → 给失效起个名字
   → 找一个能把"成功样本"和"失败样本"分开的内部量
   → 这个量必须有一个免费的目标值（任务输入自带，零标注）
   → 在推理期干预这个量
   → 在前任的坐标系里比
```

四环链缺一环就不成立。**而第一环只能靠跑和看，别的都替代不了。**

### 我们在这条链上的三条待验证靶子

跑 StyleFM 时专门盯这三处（**是靶子，不是结论**）：
1. 三分带的**边界是全局固定的**，而内容/风格的重叠程度逐图、逐区域不同
   （肖像的发丝区 vs 皮肤区）；
2. buffer band 是"两边都不给满"的折中，那一段的**内容和风格可能都不达标**；
3. 递归注意力保证了时序一致，但**一致 ≠ 正确**——"一致地错"是另一种失效。

---

## 2. 高分辨率外推：ScaleCrafter → … → AccDiffusion v2

同样的结构，但失效**肉眼一秒可见**（图里长出第二个人），比"风格保真度"这种主观量好抓一个量级。

| 代 | venue / 代码 | 它命名的失效 | 定位到的病因 | 干预 | 留下的残余 |
|---|---|---|---|---|---|
| **ScaleCrafter** | ICLR 2024 ✅`YingqingHe/ScaleCrafter` | 直接放大 → **重复图案、结构畸变** | 卷积感受野与训练分辨率不匹配 | 扩张卷积 / re-dilation | 仍重复 |
| **DemoFusion** | CVPR 2024 ✅`PRIS-CV/DemoFusion` | ↑ | patch 之间互相不知道对方在画什么 | 渐进放大 + 低分辨率参考的 skip residual + 膨胀采样 | **物体重复 (object duplication)** ← 被 AccDiffusion 点名 |
| **AccDiffusion** | ECCV 2024 ✅`lzhxmu/AccDiffusion` | **物体重复** | **所有 patch 共用同一句 prompt** ⇒ 每个 patch 都尽责地画一整个主体 | patch 内容感知的独立 prompt | 全局语义丢失 |
| **FouriScale** | ECCV 2024 ✅`LeonHLJ/FouriScale` | 重复 + 畸变 | 频域视角下的 scale/structure 不一致 | 扩张卷积 + 低通 | >2048² 全局语义丢失 |
| **HiDiffusion** | ECCV 2024 ✅`megvii-research/HiDiffusion` | ↑ | | | |
| **Pixelsmith** | **NeurIPS 2024** ✅`Thanos-DB/Pixelsmith` | ↑ | | 单卡 patch 流程 | 重复物体（被 HiWave 点名） |
| **AccDiffusion v2** | **TPAMI 2025** ✅`lzhxmu/AccDiffusion_v2`<br>**← 当前可跑在位者** | **重复生成** + **局部畸变** | prompt 对局部结构描述不准；全局语义不足 | patch 内容感知 prompt + **ControlNet 提供局部结构** + 带窗口交互的膨胀采样 | **摘要里零承认** ← 只能靠跑它看图 |
| HiWave | venue 未核实、**无代码** | >2048² **patch 边界伪影 / 内容重复 / 跨 patch 语义不一致** | | 两阶段 + DWT 细节增强 | 只能当"已发表的失效描述"引用，不能跑 |

### 这条链的判词（可直接引用）

> *"current training-free approaches either **fail to maintain global coherence**
> compared to the base diffusion model or suffer from **duplicated objects and
> artifacts** at ultra-high resolutions (e.g., 4096×4096)"* —— HiWave

> *"ScaleCrafter, FouriScale, and HiDiffusion **partially alleviate** the object
> repetition problem, but often **fail to capture correct global semantics**,
> particularly at higher resolution"*

### 这一格的一个结构性好处

**同 prompt、同 seed 的 1024² 基图，就是语义真值。**
4096² 版本多长出来的东西，程序化一比就知道——**零标注、零人工**。
风格迁移那边没有这个东西（"风格对不对"没有免费真值）。

---

## 3. 两条路并排

| | 风格迁移 | 高分辨率外推 |
|---|---|---|
| 当前可跑在位者 | **StyleFM** (AAAI 2026) ✅ | **AccDiffusion v2** (TPAMI 2025) ✅ |
| 基座 / 权重成本 | SD 1.4，**单个 ~4GB ckpt** | SDXL（**我们已有 fp16，6.7GB 已核实完整**） |
| 代码栈 | LDM 老代码库（Python 3.8 / torch 1.8 时代），**现有 torch-2.3 环境大概率不通用** | diffusers 系，与现有环境同代 |
| 测试数据 | **仓库自带** `./data`（20 content + 40 style） | LAION2B-en-aesthetic prompt 列表 |
| 失效可见性 | 中（风格保真度偏主观） | **高**（多长一个人，一眼） |
| 免费监督信号 | 无现成的 | **有**（同 seed 低分辨率基图 = 语义真值） |
| 单图耗时 | 低（1024²） | **高**（4096² patch-wise 反演，**未测**） |
| venue 落点 | CVPR / CVPR / AAAI | ICLR / CVPR / ECCV / NeurIPS / TPAMI |
| 换手速度 | **快**（一年 8 篇后继）→ 竞争风险高，但也证明格子丰饶 | 中 |
| 已付成本 | StyleSSP 读透（含 3 处代码-论文不符）、环境已建、`stratify.py`/`sheet.py` 已写并冒烟 | 基本为零（除 SDXL 权重） |

**两条都合格**（第 2 节六条特征全中：权衡目标 / 失效可命名 / 程序化零标注评测 / 推理期入口 /
≥3 篇的可见链 / 每环开源）。**差别不在"哪条更空"——那是格子属性比较，是我们已经犯了五次的错。
差别在"哪条能最快让我们看到第一张坏图"。**

按这个标准：**StyleFM 只需要一个 4GB ckpt、自带测试集，最快到第一张图**——
唯一的成本是要另建一个 LDM 老栈的 conda 环境。

---

## 4. 结论

**"每篇新文章都做得很好"这个感觉，来自只读论文不跑代码。**
论文是作者对自己工作的自我叙述，它的职业要求就是显得完备。
残余失效不在论文里，只在它没挑进论文的那些输出里。

**所以下一步不是再检索一轮，是跑。**
在看到 StyleFM（或 AccDiffusion v2）自己产出的一张坏图之前，
任何方法上的推测都是第五、第六次犯同一个错。

顺序不许乱：
```
1. 跑在位者，用它自带的数据 + 我们自己挑的难例
2. 看图，找到一条肉眼可见的失效，给它起名字
3. 才回来检索：这条**具体的**失效有没有被已录用工作修了
4. 诊断病因（用 StyleID 式"推到极限看什么不动"或 StyleSSP 式"前任碰过但没深究"）
5. 让诊断挑杠杆，不是让杠杆清单挑问题
6. 在前任的坐标系里比
```
