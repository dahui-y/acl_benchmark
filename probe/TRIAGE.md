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

## 判据：不是"常不常见"，而是"末态定不定义得出来"

想清楚那 7 个边缘条目时才明确下来的，比逐条裁决更有用：

> 评分问的是**"物体有没有到达目标末态"**。
> 因此**罕见但末态明确**的组合可用；**常见但末态说不清**的组合不可用。

两条支撑：

1. **同 item 内比较**。每个事件的 18 个变体共享同一个动作—物体对，
   合理性对全部条件影响相同，在组内对比中抵消。**本设计对条目合理性天然稳健**，
   只要它可被渲染。
2. **末态未定义则无法打分**。`grating a capsicum` 擦出来是浆不是丝，
   "擦好了"是什么状态说不出来——这类必须删，即使动作本身可以被渲染。

## 第 4 轮 7 个边缘条目的裁决

**删（3）**

| 条目 | 理由 |
|---|---|
| `rolling a biscuit` | `biscuit` 指**成品**，擀的是面团 |
| `grating a capsicum` | 甜椒壁薄多汁，擦出来是浆不是丝，**末态无定义** |
| `shredding a tomato` | 同上。番茄可擦泥，但 shred 蕴含成丝，番茄不产生丝状物 |

**留（4）**

| 条目 | 理由 |
|---|---|
| `peeling a radish` | 萝卜通常不削皮，但削了就是削了，**末态明确且视觉清楚**。只是低频 |
| `mincing a pumpkin` | 通常切丁，但 mince 的末态（极碎小块）定义清晰 |
| `frying a cucumber` | **是真菜**——美国南方炸黄瓜、中式炒黄瓜。activity 类动词，末态为"熟/上色" |
| `crushing a cucumber` | 拍黄瓜是真实技法（英文菜单作 smashed cucumber），末态视觉极明显 |

## 由这 7 条反推出的两个结构性 bug

`rolling a biscuit` 与 `roasting a cracker` **不是两个坏例子，是同一个 bug**：

| bug | 修法 |
|---|---|
| `Carb_Foods` 把**面团阶段**（dough / batter / crust / pastry）与**成品**（bread / biscuit / cracker / tortilla）混在一个子类 | 新增 `FINISHED_BAKED`，从所有塑形与加热动词中屏蔽 |
| `Fruiting` 子类过宽——番茄、甜椒、茄子、西葫芦、黄瓜、南瓜同列，而"擦丝""挤汁"只对其中一部分成立 | `shredding` / `grating` 去掉 `Fruiting`；`squeezing` 收到仅 `Citrus` |

## 待人工确认清单（只需看这些）

第 5 轮后（37 个事件，动词分布 2/个、语义类 16/11/10）我只想标记一条：

| prompt | 疑问 |
|---|---|
| `A woman is roasting an egg at a market stall.` | 烤蛋存在但少见；烤箱蛋（baked egg）更常说。末态（蛋凝固）明确，**按判据应保留**，但值得母语确认 |

**其余 36 个我判定可用。** 前几轮被我裁掉的组合已由规则屏蔽，不会再出现。

## 回归测试

四层过滤机制现在相互作用，改任一层都可能静默撤销某一轮的裁决。
`test_lexicon.py` 把这些裁决固化为断言：

```bash
python test_lexicon.py
# ok: 18 rejections, 8 retentions, 7 mass nouns, all 20 verbs classed
```

内容为 18 条必须被拒的组合（各自注明是哪一轮、因何被拒）、
8 条必须保留的组合（含 4 条"罕见但末态明确"的），以及不可数名词与动词分类的完整性检查。
已验证：把 `Fruiting` 放回 `shredding` 会立刻触发失败。

## 确认后的处理

- 判定该删的，把 `(动词, 物体)` 告诉我，我加进 `VERB_OBJECT_BLOCK` 重新生成
- 若淘汰数 ≥ 5，说明规则仍有系统性缺口，应先改规则再重新分诊
- 全部通过则 prompt suite 定稿，进入视频生成
