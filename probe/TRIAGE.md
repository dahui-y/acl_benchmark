# Prompt suite 分诊记录

**这不是人工核验。** 分诊由生成这批 prompt 的同一个模型完成，
自审不构成独立验证，论文中不能据此声称 human-in-the-loop。
它的作用是把人工核验的工作量从 37 个事件压到 8 个。

---

## 四轮分诊：明显淘汰率 40% → 3%

| 轮次 | 明显该删 | 主要修法 |
|---|---|---|
| 第 1 轮 | ~15/40（**40%**） | 发现四类系统性失败，见下 |
| 第 2 轮 | 3/37（8%） | 加动词级覆盖、丢弃 `melting`、坚果移出 Heating |
| 第 3 轮 | 3/37（8%） | 加动词—物体黑名单、`grating` 收紧 |
| 第 4 轮 | 1/37（**3%**） | `celery` 入不可数表、`squeezing` 去掉香蕉、`grilling` 去掉糕点 |

## 第 1 轮发现的四类系统性失败

| 问题 | 例 | 根因与修法 |
|---|---|---|
| **坚果全线崩** | `frying an almond`、`grilling a hazelnut`、`mashing a pecan` | 坚果的烹饪义只在复数/物质义上成立，而单数不定式是每个 item 的参照格 → **Nuts_Seeds 移出 Heating** |
| **`zesting` 越界** | `zesting a scallion`、`zesting a carrot` | zest 只对柑橘皮成立，但它继承了 Grating 整类 → **动词级覆盖限定 Citrus** |
| **`melting` 无合法可数宾语** | `melting an oreo`、`melting a cake` | 能融化的东西（黄油/奶酪/巧克力/糖）**全是不可数名词**，而终结性轴要求可数名词。这是设计层面的不相容 → **整个动词丢弃**，非筛物体 |
| **`whipping`/`mashing` 越界** | `whipping a coconut`、`mashing an almond` | 动作类粒度太粗 → **动词级覆盖** |

其中第三条值得记入论文的方法一节：**终结性轴的 a/the/bare-plural 交替要求可数名词，
因此本质上只取物质名词宾语的动词无法参与本设计。** 这是设计约束，不是数据缺陷。

## 现存过滤机制（`lexicon.py`）

| 机制 | 作用 |
|---|---|
| `ACTION_OBJECT_COMPATIBILITY` | 动作类 × 物体子类，粗粒度 |
| `VERB_OBJECT_OVERRIDE` | 动词级，比类更严（`zesting`、`whipping`、`grating` 等 11 个） |
| `VERB_OBJECT_BLOCK` | 动词—具体物体黑名单（硬壳/纤维类，如 `mashing` × coconut） |
| `MASS_OR_GENERIC` | 不可数名词与类别标签，逐子类编制 |

---

## 待人工确认清单（只需看这些）

**明显该删（1）**

| item | prompt | 问题 |
|---|---|---|
| 16 | `A man is roasting a cracker in the kitchen.` | 饼干不烤制，已成品 |

**需要母语判断（7）** —— 我的判断可能有偏，请确认是否保留

| item | prompt | 疑问 |
|---|---|---|
| 9 | `rolling a biscuit` | 擀的是饼干**面团**，成品饼干不擀 |
| 12 | `peeling a radish` | 萝卜通常不削皮 |
| 18 | `grating a capsicum` | 甜椒质地是否适合擦丝 |
| 20 | `shredding a tomato` | 番茄常见的是擦泥，不是擦丝 |
| 31 | `mincing a pumpkin` | 南瓜通常切丁，不剁末 |
| 82 | `frying a cucumber` | 黄瓜下锅煎炸是否成立 |
| 4 | `crushing a cucumber` | 拍黄瓜在中餐成立，英语语境是否自然 |

**其余 29 个事件我判定无问题。**

## 确认后的处理

- 判定该删的，把 `(动词, 物体)` 告诉我，我加进 `VERB_OBJECT_BLOCK` 重新生成
- 若淘汰数 ≥ 5，说明规则仍有系统性缺口，应先改规则再重新分诊
- 全部通过则 prompt suite 定稿，进入视频生成
