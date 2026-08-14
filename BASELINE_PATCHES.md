# 对 baseline 源码的改动（全部披露）

我们跑对手方法时对其源码做过的**每一处**改动都记在这里，含理由与
恒等性证明。原则三条：

1. **只修让代码跑不起来的东西**，绝不碰算法、超参、默认开关；
2. 每一处都要给出**数值恒等性证明**或明确标注"会改变行为"；
3. 论文的可复现性附录直接引用本文件。**批评别人不可复现的人，
   自己得可复现**（§9.5c 我们拿"两篇顶会给出相反排序"当论据，
   就必须守这条）。

---

## P1 · AccDiffusion `utils.py:735` —— Python 3.12+ 兼容，数值恒等

**日期** 2026-08-14 ｜ **状态** 已验证恒等 ｜ **影响** 无

**症状**（跑到 Phase 1 结束、`use_md_prompt` 生成逐 patch prompt 时崩）：

```
File "help_code/AccDiffusion/utils.py", line 741, in get_views
    w_jitter = random.randint(-jitter_range, 0)
TypeError: 'float' object cannot be interpreted as an integer
```

**成因**：调用处 `utils.py:808` 传 `stride=window_size/2`，Python 3 的
`/` 恒为浮点，于是

```python
jitter_range = (window_size - stride) // 4     # -> 8.0，浮点
```

而 `random.randrange` 自 **Python 3.12** 起不再接受浮点实参（此前是
DeprecationWarning）。上游的运行环境是更早的 Python，故未暴露。

**改动**（照抄上游自己在别处的写法）：

```python
-                jitter_range = (window_size - stride) // 4
+                jitter_range = int((window_size - stride) // 4)
```

**为什么不算改动算法**：
- `// 4` 已经取整，结果**本来就是整数值**，`int()` 只去掉浮点外壳；
- 老版本 `randint` 收到整数值浮点时的行为与收到 `int` 完全一致；
- **上游自己**在 `utils.py:811` 与 `accdiffusion_sdxl.py:1369` 的同一
  表达式上就写了 `int(...)` —— 735 行是漏网，不是设计。

**恒等性证明**（脚本见提交记录）：对 `window_size ∈ {32,64,128}` ×
`scale ∈ {2,3,4}` 共 9 组配置，固定 seed 后把补丁后的 `get_views` 与
"补丁前在 Python ≤3.11 上的等价语义"逐视图比对：

```
window=  32 scale=2  视图数   9  逐元素相同 True
...（9 组全部 True）
补丁数值恒等性: OK —— 一个视图坐标都没变
```

**范围**：只改这一处。`accdiffusion_sdxl.py:582` 有同样的裸表达式，
但那条路径的 `stride` 由我们以 `stride=64`（整数）传入、`window_size`
取自 `unet.config.sample_size`（整数），乘除后仍是整数，**实测不触发**，
故不动 —— 少碰对手一行代码，就少一分可争议之处。

---

## C1 · **不得使用 `--lowvram`**（配置决定，非代码改动，但会静默污染结果）

**日期** 2026-08-14 ｜ **代价** 已烧掉一条 19 分钟的样本

`lowvram=True` 不是一个中性的显存开关，它**改数值**：

```python
# accdiffusion_sdxl.py:1247
if self.lowvram:
    needs_upcasting = False   # use madebyollin/sdxl-vae-fp16-fix in lowvram mode!
```

即 lowvram 分支**关掉 VAE 的 fp32 升位**，前提是使用者换了 fp16-fix
的 VAE。用原版 SDXL VAE（`force_upcast=True`）时 VAE 在 fp16 下解码 ->
溢出出 NaN -> `postprocess` 里 `(images*255).round().astype("uint8")`
把 NaN 铸成垃圾像素。**图照样存盘**，只在 stderr 留一行
`RuntimeWarning: invalid value encountered in cast`。

第二个副作用：lowvram 在调用结束时把 `unet`/`vae` 留在 CPU
（L1245），下一次调用 `self._execution_device` 解析成 cpu，与 cuda
generator 冲突 -> `Cannot generate a cpu tensor from a generator of
type cuda`。**管线在 lowvram 下不可重复调用。**

**结论**：4096² 实测峰值仅 **10.1 GB**，24GB 卡不需要 lowvram。
一律不开。三道防线已就位：
- `acc_batch.py` 开了就打印显式警告；
- 逐条生成后**当场体检**（纯黑 >30% 或标准差 <4 即判可疑），
  可疑条目**不写入 manifest**，不会进入测量；
- `img_sanity.py` 跑批后全量复检，`--rm` 可删坏图让脚本重跑。

> 教训写在这里：**一张坏图以"对手的正常输出"身份进入闸门测量，
> 足以让整个结论作废，而它只会留下一行 warning。** 任何对手方法的
> 输出在计数之前都必须过体检。

---

## 未改动但需在论文中声明的配置选择

这些不是代码改动，是**调用参数的选择**，同样可能被质疑，先记在这里：

| 项 | 我们的取值 | 依据 |
|---|---|---|
| 七个方法开关 | 全部 `True` | AccDiffusion `Readme.md` 的官方命令行。**argparse 的 default 全是 `False`**，照 default 跑等于把 `use_md_prompt`（他们的核心贡献 patch-content-aware prompt）关掉，那不是 AccDiffusion |
| negative prompt | 用**他们 Readme 的**（`blurry, ugly, duplicate, ...`） | 换成我们的会改掉他们的方法。注意他们的 negative 里本来就带 `duplicate` |
| `c`（重复阈值） | 0.3 | Readme 默认 |
| steps / guidance | 50 / 7.5 | Readme 默认 |
| seed | 77 | 与我们两臂同 seed，保证逐条配对 |
| `--lowvram` | **关** | 见 C1 —— 它会关掉 VAE 升位、静默产生 NaN 图。实测峰值 10.1 GB，不需要 |
