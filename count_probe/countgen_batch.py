#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CountGen（make-it-count, CVPR'25）跑批驱动 —— **不改它一行源码**。

    与直接跑 `pipeline/run_countgen.py` 的差别，只有三处，每一处都有理由：

    1. **把 DBSCAN 计数器的读数落盘**（`counter_log.jsonl`）。
       原脚本把 `n_dbscan_clusters` 和 `obj_num_match` 用完就丢。但那两个量
       决定了这条线的天花板 —— `extract_mask.py:25-27` + `run_countgen.py:128-129`
       里，簇数 == N 就**直接输出原版图、不做任何干预**，所以
       **CountGen 的准确率上限 = 它计数器的准确率**。不记下来就没法拆。

    2. **N > 9 的题目仍然跑 vanilla + 计数**，只是不做修正。
       原脚本 `run_countgen.py:104` 对 N>9 直接 `continue`，连原版图都不存
       （根因是 `relayout.py:7` 的 ReLayout U-Net `in_channels=9`）。
       而它自带的 CoCoCount 里 **1/6 的题就是 N=10**
       （`dataset/create_data_CoCoCount.py:38`），仓库那份 json 里 33/200。
       我们把这一档单独留着并标 `skipped_by_official=true`，
       报数时单独列 —— 不能混进总分，也不该假装它不存在。

    3. **两个只影响速度、不影响数值的缓存**：
       · `spacy.load("en_core_web_trf")` 写在 `__call__` 里
         （`self_counting_sdxl_pipeline.py:277`），每次调用都从磁盘加载一个
         RoBERTa。每张图调 2 次。memoize 掉。
       · `torch.hub.load(...)` 在 `relayout_undergeneration` 里**逐图调用**
         （`relayout.py:10`），要连 GitHub。memoize + 离线回退。
       两者都只是"同样的对象少造几次"，不动任何随机数与算子。

    ⚠️ 调用顺序、随机数流程与原脚本逐行一致（先 set_seed 再造 latents，
       再进 relayout —— relayout 内部会再 set_seed 一次），
       否则复现不出发表的数。

输出布局 = **官方布局**（这样 `make_arms.py` 一个转换器就能同时吃我们的输出
和官方脚本的输出）：

    <out>/{obj}_num={N}_seed={S}.png            ← CountGen 臂
    <out>/{obj}_num={N}_seed={S}_vanilla.png    ← 原版 SDXL 臂（白送的 baseline）
    <out>/{obj}_num={N}_seed={S}_masks.png      ← 三张 mask 的可视化
    <out>/metadata.json                         ← 与官方同格式
    <out>/counter_log.jsonl                     ← 我们加的：逐题计数器读数

用法：
    export SD_OUT=/openbayes/input/input0/Sim2Struct-1000/temp/scalediff_out
    # 先冒烟 5 题，确认能跑通、看单张耗时
    python count_probe/countgen_batch.py --limit 5 --out $SD_OUT/count/cocoount
    # 全量
    python count_probe/countgen_batch.py --out $SD_OUT/count/cocoount
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---- 损失的解析下界 L_min(f) = A + B·f ----------------------------------
# CountGen 把 attention map 归一化到 [0,1] 再喂 binary_cross_entropy_with_logits
# （内部还会过一次 sigmoid），所以「预测值」只能落在 [σ(0), σ(1)] = [0.500, 0.731]，
# 两端都够不到。前景带 pos_weight=10，于是每题的损失有一个只取决于前景占比 f
# 的下界：
#     L_min(f) = f·10·(−log 0.731) + (1−f)·(−log 0.500) = 0.693 + 2.439·f
# 原版的退出阈值是全局常数 {0:1.3, 10:1.2, 20:1.15}（pipeline_config.yaml）。
# 实测：f 中位 32.2% → L_min 中位 1.48，13 道「数少了」的题里 11 道连 step0 的
# 1.3 都数学上不可达，13/13 够不到 step20 的 1.15。够不到就退不出，于是精修
# **必然**跑满 max_refinement_steps，把 latent 一路推到预算用完。
L_MIN_A, L_MIN_B = 0.6931, 2.4394


def l_min(fg_frac):
    """该题损失能取到的最小值。低于它的阈值等于「永不退出」。"""
    return L_MIN_A + L_MIN_B * float(fg_frac)


def adaptive_thresholds(fg_frac, orig, margin):
    """把阈值锚到逐题的解析下界上，保留原版各步之间的相对形状。

    原版 {0:1.3, 10:1.2, 20:1.15} 的形状是「越往后要求越高」（相对 step0
    分别 0 / −0.10 / −0.15）。这里只把**基准**从全局常数换成 L_min(f)+margin，
    形状原样保留 —— 改一个变量，别顺手改两个。
    """
    ks = sorted(orig)
    base = orig[ks[0]]
    return {k: l_min(fg_frac) + margin + (orig[k] - base) for k in ks}

os.environ.setdefault("MPLBACKEND", "Agg")      # 无头环境；他们的代码会画图
# 修正步的显存峰值很尖（带梯度的 UNet 前向），碎片化会让本来够的显存不够
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parent.parent
MIC = REPO / "help_code" / "make-it-count"


def _patch_caches():
    """把两个"每张图都重造一次"的重对象 memoize 掉。不改数值。"""
    import spacy
    import torch

    _spacy_cache = {}
    _orig_spacy_load = spacy.load

    def spacy_load(name, **kw):
        if name not in _spacy_cache:
            print(f"   [cache] spacy.load({name}) 首次加载…")
            _spacy_cache[name] = _orig_spacy_load(name, **kw)
        return _spacy_cache[name]

    spacy.load = spacy_load

    _hub_cache = {}
    _orig_hub_load = torch.hub.load

    def hub_load(repo_or_dir, model, *a, **kw):
        key = (str(repo_or_dir), model, tuple(sorted(kw.items())))
        if key in _hub_cache:
            return _hub_cache[key]
        try:
            m = _orig_hub_load(repo_or_dir, model, *a, **kw)
        except Exception as e:
            # 离线回退：torch.hub 缓存目录里若已 clone 过就用本地
            hub_dir = Path(torch.hub.get_dir())
            local = hub_dir / (str(repo_or_dir).replace("/", "_") + "_master")
            if not local.exists():
                raise SystemExit(
                    f"!! torch.hub.load({repo_or_dir}) 失败：{e}\n"
                    f"   服务器大概连不上 GitHub。先在能联网的机器上跑一次：\n"
                    f"     python -c \"import torch;torch.hub.load('{repo_or_dir}',"
                    f"'{model}',in_channels=9,out_channels=10,init_features=128,"
                    f"pretrained=False)\"\n"
                    f"   再把 {hub_dir}/ 整个拷到本机同路径（或设 TORCH_HOME）。")
            print(f"   [cache] torch.hub 走本地 {local}")
            m = _orig_hub_load(str(local), model, *a, source="local", **kw)
        _hub_cache[key] = m
        return m

    torch.hub.load = hub_load


MEM_LOG = False
# 每进一次修正步记一条 (触发的 step, 迭代次数, 进入时 loss, 出来时 loss)。
# 没有它就看不出阈值定得对不对：跑满 max_refinement_steps 说明阈值太严（或不可达），
# 一两步就退出说明太松。原版实测是 100% 跑满 20 步。
REFINE = []


def _mem(tag):
    if not MEM_LOG:
        return
    import torch
    print(f"    [显存] {tag}: 已分配 {torch.cuda.memory_allocated()/2**30:5.2f} GB / "
          f"峰值 {torch.cuda.max_memory_allocated()/2**30:5.2f} GB", flush=True)


def _patch_mem_graph():
    """两处**数值完全等价**的改动，专治修正步的显存峰值。

    ① `update_latent`（self_counting_sdxl_pipeline.py:222）写的是
           torch.autograd.grad(loss, latents, create_graph=True)
       `create_graph=True` 会为"梯度的梯度"再建一张图，并且隐含 retain_graph=True
       —— **算完不释放**。但全仓库没有任何地方用二阶导：拿到 grad 后是
           latents = latents - step_size * grad
       下一轮立刻 `clone().detach()`（refinement 第 174 行 / __call__ 第 462 行）。
       改成 False 后 grad 的**数值一模一样**，只是图算完就放。

    ② 进修正步时同时活着两张完整的图：`__call__:476` 先做了一次带梯度前向得到
       `loss`，再把它传进 `perform_iterative_refinement_step`，函数里立刻又建第二张。
       而那个 `loss` 在函数内**只用作 `while loss > target_loss` 的比较量**，
       第 186 行就被重算覆盖。所以进门先把它变成 Python float，
       并清掉 attention store 里指向那张图的引用 —— 两者都在被读之前会被
       本函数的第一次前向重新填好（`between_steps` 在前向末尾触发，
       `loss_and_plot` 在其后才读）。

    这两条都不改任何一个会被使用的数值，只改张量的存活期。
    """
    import torch
    import pipeline.self_counting_sdxl_pipeline as SP

    cls = SP.SelfCountingSDXLPipeline
    orig_loss_and_plot = cls.loss_and_plot
    orig_refine = cls.perform_iterative_refinement_step

    def loss_and_plot(self, object_token_idx, i):
        """返回 float，把带图的张量存到 self 上。

        ② 的关键：`__call__:491` 是 `loss, latent = self.perform_iterative_...(loss, ...)`
        —— **调用方的局部变量要等函数返回才重新绑定**，所以在函数内部把形参转成
        float 是没用的，外层那张图仍被调用方的栈帧钉着（Python 3.10 改不到 f_locals）。
        釜底抽薪：让 `loss_and_plot` 压根不把张量交出去。

        调用方对它的返回值只做两件事：`loss > thresholds[i]`（:490）、`loss != 0`
        （:497、:188）—— float 完全够。唯一需要计算图的是 `update_latent`，
        而那个函数正好也在我们手里，让它去取存下来的张量即可。
        `loss_and_plot` 的调用点只有 :156 / :186 / :216 / :487 四处，都在这条链上。
        """
        v = orig_loss_and_plot(self, object_token_idx, i)
        self._n_loss_calls = getattr(self, "_n_loss_calls", 0) + 1
        self._last_loss_val = float(v) if torch.is_tensor(v) else float(v or 0)
        _mem(f"step {i} 前向后")
        if torch.is_tensor(v):
            self._loss_tensor = v            # update_latent 要用的就是这一个对象
            return float(v)
        self._loss_tensor = None             # loss 可能是 int 0（步范围之外）
        return v

    def update_latent(self, latents, loss, step_size):
        # ① create_graph=True → False：全仓库无二阶导，grad 的值不变，
        #    但 create_graph 隐含 retain_graph=True，会让图算完不释放。
        t = getattr(self, "_loss_tensor", None)
        grad = torch.autograd.grad(t if t is not None else loss, latents,
                                   create_graph=False)[0]
        return latents - step_size * grad

    def perform_iterative_refinement_step(self, loss, latents, *a, **kw):
        # 进了这里就说明外层那张图不会再被用到（返回值会覆盖调用方的 loss/latent）。
        # 把所有还指向它的引用清掉：存下来的张量、attention store 里的中间量。
        # 两者都会被本函数第一次前向重新填好（between_steps 在前向末尾触发，
        # loss_and_plot 在其后才读），所以清掉是安全的。
        self._loss_tensor = None
        self.attention_store.all_cross_attention = {}
        self.attention_store.all_self_attention = {}
        self.attention_store.cross_attention_store = {}
        self.attention_store.self_attention_store = {}
        torch.cuda.empty_cache()
        _mem("进修正步（已丢掉上一张图）")
        self._n_loss_calls = 0
        loss_in = float(loss)
        out = orig_refine(self, loss_in, latents, *a, **kw)
        REFINE.append((int(getattr(self.attention_store, "curr_step_index", -1)),
                       int(self._n_loss_calls), round(loss_in, 4),
                       round(float(getattr(self, "_last_loss_val", 0.0)), 4)))
        _mem("出修正步")
        return out

    cls.loss_and_plot = loss_and_plot
    cls.update_latent = update_latent
    cls.perform_iterative_refinement_step = perform_iterative_refinement_step


def _patch_mem_attn():
    """只对「概率矩阵算完就丢」的注意力层改走 SDPA，省掉 24G 装不下的那部分显存。

    OOM 出在 `perform_iterative_refinement_step`：那里 `latents.requires_grad_(True)`，
    整个 UNet 前向的中间量被 autograd 留住。而 `CountingProcessor` 走的是**显式**
    注意力（`get_attention_scores` → `baddbmm`），SDXL 1024² 下 64×64 那一档有
    4096 个 token，一层的概率矩阵是 [2×10, 4096, 4096] fp16 = **671 MB**，
    这一档约 10 个 block → 光概率矩阵就 6–7 GB。

    但很多层的概率矩阵是**白留的**。把「谁会读它」查全（见 `_probs_are_read`）：

      · **存**：`attention_store_counting.py:56` 要求 `shape[1] == attn_res²(=1024)`。
        4096 那一档不满足 → 不存。
      · **读**：`aggregate_attention` 全仓库只有 `self_counting_sdxl_pipeline.py:152`
        一处调用，且 `get_cross=True` → **`all_self_attention` 从来没被读过**。
        `self_step_store` 只有 `dbscan_mask_extract.py:181` 读，而它只在
        `loss=False` 的原版那趟被写入（`:73` 的 `and not self.loss`）。
      · **屏蔽**：`attention_processors.py:34-38` 要求 `shape[0]==40`，即
        batch2×20heads。而修正前向是 `latent.unsqueeze(0)`（`:469`）→ **batch=1**，
        `shape[0]=20`，屏蔽在那一趟**根本不生效**。

    结论：真正需要显式概率矩阵的只有三种层 ——
      ① 32² 的 **cross**-attn（loss 要用）
      ② **原版趟**的 32² self-attn（喂 DBSCAN）
      ③ **主 CFG 前向**里 up 块、步 0–10 的 32² self-attn（要被屏蔽）
    其余一律走 `scaled_dot_product_attention`：算的是同一个东西，
    但 SDPA 的后端在反向时**重算**而不保存那个矩阵。

    保守起见，只要遇到任何本函数没覆盖的情形（有 attention_mask、
    upcast_attention、group_norm…）就退回原路，不赌。

    ⚠️ 这一条是**浮点级**改动（SDPA 与 baddbmm+softmax+bmm 归约顺序不同），
       不是逐比特复现。`_patch_mem_graph` 那两条才是数值完全等价的。
       等价性验法：`--vanilla-only` 跑两遍（带/不带 `--no-mem-attn`）比 n_dbscan。
    """
    import torch
    import torch.nn.functional as F
    import pipeline.self_counting_sdxl_pipeline as SP
    from pipeline.attention_processors import CountingProcessor

    class MemAttnCountingProcessor(CountingProcessor):
        def _probs_are_read(self, attn, hidden_states, is_cross):
            """这一层的 attention_probs 会不会真的被读？读才留显式路径。"""
            st = self.attnstore
            res2 = st.attn_res[0] ** 2
            n = hidden_states.shape[1]
            if n != res2:
                return False                     # 别的分辨率：存和屏蔽的门限都不满足
            if is_cross:
                return True                      # loss 要 aggregate_attention(get_cross=True)
            if not st.loss:
                return True                      # 原版那趟：self_step_store 要喂 DBSCAN
            # counting 那趟的 self-attn：只会进 all_self_attention，而
            # aggregate_attention 全仓库仅 :152 一处调用且 get_cross=True → 从不读。
            # 唯一还会用到它的是屏蔽块，条件见 attention_processors.py:34-38。
            m = st.masking_dict
            return (bool(m.get("enable"))
                    and hidden_states.shape[0] * attn.heads == 40     # batch2×20 heads
                    and m["start_step"] <= st.curr_step_index <= m["end_step"]
                    and "up" in self.place_in_unet)

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, **kwargs):
            is_cross = encoder_hidden_states is not None
            keep = (self._probs_are_read(attn, hidden_states, is_cross)
                    or attention_mask is not None
                    or getattr(attn, "upcast_attention", False)
                    or getattr(attn, "group_norm", None) is not None
                    or getattr(attn, "spatial_norm", None) is not None
                    or getattr(attn, "residual_connection", False))
            if keep:
                return super().__call__(attn, hidden_states, encoder_hidden_states,
                                        attention_mask, **kwargs)

            ehs = encoder_hidden_states if is_cross else hidden_states
            q = attn.head_to_batch_dim(attn.to_q(hidden_states))
            k = attn.head_to_batch_dim(attn.to_k(ehs))
            v = attn.head_to_batch_dim(attn.to_v(ehs))
            # scale 显式传：diffusers 的 attn.scale 与 SDPA 的默认值都是 d^-0.5，
            # 但写出来就不必依赖"默认值恰好相同"这个假设
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                 dropout_p=0.0, is_causal=False,
                                                 scale=attn.scale)
            # 仍要调 store —— 它管着 cur_att_layer 的推进和 between_steps 的时序，
            # 少调一次整条时序就错位了。给一个 meta 张量（不占显存），
            # shape[1]=0 必然 ≠ attn_res²，于是它只推进计数器、不落任何张量。
            # 走到这里的层，其 probs 本来也不会被任何地方读（见 _probs_are_read）。
            self.attnstore(torch.empty((1, 0, 1), device="meta"),
                           is_cross, self.place_in_unet, attn.heads)
            out = attn.batch_to_head_dim(out)
            # 与它们的处理器一致：不做 group_norm / residual / rescale
            return attn.to_out[1](attn.to_out[0](out))

    SP.CountingProcessor = MemAttnCountingProcessor   # register_attention_control 用的是这个名字
    return MemAttnCountingProcessor


def _patch_instance_loss(a):
    """把 `compute_loss` 换成实例感知的目标。**我们的方法。**

    替换点选在 `SelfCountingSDXLPipeline.compute_loss`
    （self_counting_sdxl_pipeline.py:144-148），它是原版调用
    `utils/loss_utils.object_layout_loss` 的唯一入口。这样：
      · make-it-count 一行源码都不用改
      · 改动被限制在**一个函数**里，跑批结果的任何变化都只能归因到它
      · `--loss orig` 一切照旧，A/B 干净

    详细动机与三条根因见 count_probe/inst_loss.py 的文件头。
    """
    import pipeline.self_counting_sdxl_pipeline as SP
    sys.path.insert(0, str(REPO / "count_probe"))
    from inst_loss import instance_layout_loss

    def compute_loss(self, object_attention_map):
        return instance_layout_loss(object_attention_map, self.desired_mask,
                                    w_sep=a.w_sep, w_bg=a.w_bg,
                                    w_conc=a.w_conc, w_cov=a.w_cov,
                                    dilate=a.dilate,
                                    tau_fg=a.tau_fg, tau_bg=a.tau_bg,
                                    hinge_pow=a.hinge_pow)

    SP.SelfCountingSDXLPipeline.compute_loss = compute_loss


def _patch_unet_detach(pipe):
    """把**引导前向**的 UNet 返回值 detach 掉。数值等价，因为它从来没被用过。

    实测（--mem-log）：清掉 `_loss_tensor` 和 attention store 之后，
    已分配显存从 18.59 GB 只掉到 18.58 GB —— 图根本没释放。
    漏掉的引用是 `self_counting_sdxl_pipeline.py:476` 的

        _ = self.unet(latent, t, ...)[0]

    那个 `_` 是**调用方栈帧里的局部变量**，握着 UNet 输出，把整张图钉住；
    我们清不掉它（还是 f_locals 那个问题）。

    但它从来没被用过：`_ = self.unet(...)` 出现在 :175 / :205 / :476 三处，
    全是丢弃赋值；loss 是从 attention store 里算的，与 UNet 的输出无关。
    所以把它 detach 掉，图就只剩 attention store 那一条引用 —— 那条我们能清。

    主去噪路径的 `noise_pred`（:511 之后）是要用的，但那一段在 `@torch.no_grad()`
    下、本来就没有图，所以用 `torch.is_grad_enabled()` 正好把两者分开。
    """
    import torch

    unet = pipe.unet
    orig = unet.forward

    def forward(*a, **kw):
        out = orig(*a, **kw)
        if not torch.is_grad_enabled():
            return out                       # 主去噪路径：本来就无图，原样返回
        if isinstance(out, tuple):
            return (out[0].detach(),) + tuple(out[1:])
        return out.__class__(sample=out.sample.detach())

    unet.forward = forward


def _patch_counter_probe():
    """把 DBSCAN 的簇数原样取出来，而不是从 mask 反推。

    `extract_mask.relayout()` 只返回 mask，不返回簇数；从 `mask.max()` 反推
    在正常情况下是准的（`remove_sparse_blobs` 把标签连续重编号成 0..k-1），
    但在 **k=0 的兜底分支**（`extract_mask.py:21-23` 往中心塞一个 6×6 方块并
    令 n=1）下会把 0 误读成 1。所以这里直接包一层 `remove_sparse_blobs`，
    把它的第二个返回值记下来 —— 不改它们的源码，只在模块命名空间里替换引用。
    """
    import pipeline.mask_extraction.extract_mask as EM
    holder = {}
    orig = EM.remove_sparse_blobs

    def wrapped(grid, *a, **kw):
        out = orig(grid, *a, **kw)
        holder["n"] = int(out[1])
        return out

    EM.remove_sparse_blobs = wrapped
    return holder


def _load_pipeline(cfg):
    from pipeline.self_counting_sdxl_pipeline import SelfCountingSDXLPipeline
    import diffusers
    import torch

    # ★ local_files_only 必须显式传。diffusers 0.25 的 download() 会**无条件**
    #   先调 model_info() 问 HF 元数据；此时若设了 HF_HUB_OFFLINE，它是抛
    #   OfflineModeIsEnabled 而不是回退到缓存。传 True 才会整条跳过网络。
    kw = dict(use_safetensors=True, torch_dtype=torch.float16,
              variant=cfg["model"].get("variant") or None, use_onnx=False)
    try:
        pipe = SelfCountingSDXLPipeline.from_pretrained(
            cfg["model"]["sdxl_path"],
            local_files_only=cfg["model"]["local_only"], **kw)
    except Exception as e:
        # 权重全在本地缓存里，没理由让一个 45 分钟的任务在第 0 秒死于网络。
        # nohup 起的 shell 常常没继承 HF_HOME/HF_HUB_OFFLINE，就会走到这里。
        if cfg["model"]["local_only"]:
            raise
        print(f"⚠️ 联网加载失败（{type(e).__name__}），改用本地缓存重试。\n"
              f"   原因：{str(e).splitlines()[0][:120]}\n"
              f"   （想彻底避免：跑之前 export HF_HUB_OFFLINE=1，"
              f"或加 --local-files-only）")
        pipe = SelfCountingSDXLPipeline.from_pretrained(
            cfg["model"]["sdxl_path"], local_files_only=True, **kw)
    pipe.to(torch.device(cfg["pipeline"]["device"]))
    pipe.counting_config = cfg["counting_model"]
    if cfg["counting_model"]["use_ddpm"]:
        pipe.scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config)
        print("scheduler = DDPM（与官方一致）")
    return pipe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="输出目录（官方布局）")
    ap.add_argument("--dataset", default=str(MIC / "dataset" / "CoCoCount.json"))
    ap.add_argument("--config", default=str(MIC / "pipeline" / "pipeline_config.yaml"))
    ap.add_argument("--sdxl", default=os.environ.get(
        "SDXL_PATH", "stabilityai/stable-diffusion-xl-base-1.0"))
    ap.add_argument("--variant", default="fp16",
                    help="本地权重目录若没有 fp16 分支，传空字符串")
    ap.add_argument("--local-files-only", action="store_true",
                    default=bool(os.environ.get("HF_HUB_OFFLINE")),
                    help="只用本地缓存，一次网络都不发。设了 HF_HUB_OFFLINE 时自动打开。"
                         "批量跑两三小时，中途一次元数据探测超时就白费，建议开着")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--only-ids", default=None,
                    help="只跑这个文件里列出的 id（每行一个），**并忽略续跑记录**。"
                         "用于给已经跑过的子集补存 mask 数组")
    ap.add_argument("--skip-over9", action="store_true",
                    help="连 vanilla 都不跑 N>9（完全等同官方行为）")
    ap.add_argument("--no-masks", action="store_true", help="不存 mask 可视化")
    ap.add_argument("--relayout-ckpt", default=os.environ.get("RELAYOUT_CKPT"),
                    help="ReLayout 权重路径。默认用 pipeline_config.yaml 里的相对路径"
                         "（仓库内），但那是个 GB 级文件，不该塞进仓库——"
                         "指到有空间的盘即可，或设环境变量 RELAYOUT_CKPT")
    ap.add_argument("--no-mem-attn", action="store_true",
                    help="关掉注意力改路（那一条是浮点级改动）。留着是为了验等价性")
    # ---- 我们的方法 ----
    g = ap.add_argument_group("方法（默认全关 = 原版 CountGen）")
    g.add_argument("--loss", default="orig", choices=["orig", "instance"],
                   help="orig=原版二值前景 BCE；instance=实例感知目标（组件 1）")
    g.add_argument("--w-sep", type=float, default=1.0, help="间隔带项权重")
    g.add_argument("--w-bg", type=float, default=1.0, help="背景项权重")
    g.add_argument("--w-cov", type=float, default=0.0,
                   help="逐 blob 覆盖度项权重。第一轮实测：w_cov=0 时 75 个前景格里"
                        "只有 **1 格** 拿到梯度（L_peak 用 min_k，只剩最弱 blob 的"
                        "argmax），而原版 BCE 对整片前景施压且带 pos_weight=10。"
                        "这是第一轮'删多余'从 49.2%% 塌到 17.4%% 的病根")
    g.add_argument("--w-conc", type=float, default=0.0,
                   help="blob 内集中度项权重。最不确定的一项，默认关，留作消融")
    g.add_argument("--dilate", type=int, default=1, help="间隔带的膨胀半径")
    g.add_argument("--tau-fg", type=float, default=None,
                   help="hinge 余量：前景只要 ≥ 此值就不再推（建议 0.8）。"
                        "不给则无余量 —— 第二轮实测无余量会把 attention 一路推成"
                        "硬 0/1 图，图从照片塌成白底剪影")
    g.add_argument("--tau-bg", type=float, default=None,
                   help="hinge 余量：背景只要 ≤ 此值就不再压（建议 0.2）")
    g.add_argument("--hinge-pow", type=float, default=1.0,
                   help="hinge 的幂次。1=relu，其梯度是**阶跃**（背景里只有 0 和"
                        "一个常数两种取值），怀疑是 tune_hinge 那些轴对齐矩形断层的"
                        "来源；2=平方 hinge，梯度随超出量连续变化，边界处渐进到 0")
    g.add_argument("--fg-max", type=float, default=1.0,
                   help="组件 2：把 desired_mask 逐 blob 腐蚀到总前景占比 ≤ 此值。"
                        "1.0=不约束（原版）。0.25 是我们推出的阈值可达线")
    g.add_argument("--thresholds", default=None,
                   help="覆盖 refinement 的退出阈值，形如 0:0.5,10:0.4,20:0.35。"
                        "换了损失就必须重定 —— 量纲完全不同。"
                        "传 max 则全部置 0 = 每次跑满 max_refinement_steps，"
                        "与原版实际行为一致，省掉这个旋钮")
    g.add_argument("--scale-factor", type=float, default=None,
                   help="覆盖 latent 更新步长的系数（原版 50）")
    g.add_argument("--thresh-margin", type=float, default=None,
                   help="组件 3：**逐题**把退出阈值定成「解析下界 + 该余量」，"
                        "而不是全局常数。见 L_MIN_A/L_MIN_B 的推导。"
                        "原版的 {1.3,1.2,1.15} 在多数题上低于下界、数学上不可达，"
                        "于是精修必然跑满 20 步。给了这个就能真正在达标时停下来。"
                        "与 --thresholds 互斥")

    ap.add_argument("--mem-log", action="store_true",
                    help="在修正步前后打印显存占用。再 OOM 就开它，别继续猜")
    ap.add_argument("--no-mem-graph", action="store_true",
                    help="关掉两条数值完全等价的显存改动（create_graph=False、"
                         "进修正步前丢掉上一张图）。只在审计时用；关了 24G 必 OOM")
    ap.add_argument("--vanilla-only", action="store_true",
                    help="只跑原版 SDXL + DBSCAN 计数，不做任何修正。"
                         "不需要 ReLayout 权重，一次前向无梯度，快三四倍。"
                         "拿到的是表三四个格子里的前三个（baseline、计数器一致率、"
                         "「计数器说对但实际错」这一桶），只差修正成功率。")
    a = ap.parse_args()

    _patch_caches()
    sys.path.insert(0, str(MIC))
    os.chdir(MIC)                       # 他们的 config 用相对路径

    import numpy as np
    import torch
    import yaml
    from diffusers.utils.torch_utils import randn_tensor
    from tqdm import tqdm

    sys.path.insert(0, str(REPO / "count_probe"))
    from inst_loss import shrink_mask
    from pipeline.run_countgen import set_seed, run_counting_pipeline_corrected_masks
    from pipeline.mask_extraction.extract_mask import relayout
    from pipeline.mask_extraction.dbscan_mask_extract import dbscan_extract_mask
    from pipeline.mask_extraction.utils_masks import remove_sparse_blobs
    from utils.generate_random_masks import show_mask_list

    cfg = yaml.safe_load(open(a.config))
    cfg["model"] = {"sdxl_path": a.sdxl, "variant": a.variant,
                    "local_only": a.local_files_only}
    if a.local_files_only:
        print("SDXL 只从本地缓存加载（local_files_only=True）")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg["pipeline"]["output_path"] = str(out)

    data = json.load(open(a.dataset))
    if a.limit:
        data = data[:a.limit]
    n_over9 = sum(1 for d in data if d["int_number"] > 9)
    print(f"数据集 {Path(a.dataset).name}：{len(data)} 题，其中 N>9 的 {n_over9} 题"
          f"（官方 run_countgen.py:104 会整题跳过）")

    # ★ 权重缺失要在**启动时**就报。它只在 relayout_undergeneration 里被 load，
    #   即"DBSCAN 数少了"才走到；等跑到第 N 张才炸，前面的 GPU 就白烧了。
    if a.relayout_ckpt:
        # 绝对路径写回 config —— relayout_undergeneration 是从 config 里读它的，
        # 而我们已经 chdir 到 make-it-count，相对路径会解到仓库里去。
        cfg["mask_creation"]["dbscan_mask"]["unet_checkpoint_path"] = str(
            Path(a.relayout_ckpt).resolve())
    ckpt = Path(cfg["mask_creation"]["dbscan_mask"]["unet_checkpoint_path"])
    if not a.vanilla_only and not ckpt.exists():
        sys.exit(f"!! 缺 ReLayout 权重：{ckpt if ckpt.is_absolute() else ckpt.resolve()}\n"
                 f"   （Google Drive，见 make-it-count README；HF 上没有镜像）\n"
                 f"   放在别处就用 --relayout-ckpt /那个/路径（或 export RELAYOUT_CKPT）。\n"
                 f"   想先把不依赖它的三个读数拿到手，加 --vanilla-only。")
    if not a.vanilla_only:
        print(f"ReLayout 权重：{ckpt}")

    global MEM_LOG
    MEM_LOG = a.mem_log
    # ---- 方法开关。全部默认关闭时，行为与原版 CountGen 逐位一致 ----
    if a.thresh_margin is not None and a.thresholds:
        raise SystemExit("!! --thresh-margin 与 --thresholds 互斥：一个是逐题的、"
                         "一个是全局常数，同时给等于把变量搅在一起。")
    global ORIG_THRESHOLDS
    ORIG_THRESHOLDS = {int(k): float(v) for k, v in
                       cfg["counting_model"]["loss"]["thresholds"].items()}
    if a.thresh_margin is not None:
        print(f"★ 退出阈值 = 逐题的解析下界 L_min(f)={L_MIN_A:.3f}+{L_MIN_B:.3f}·f "
              f"加余量 {a.thresh_margin:g}，保留原版形状 {ORIG_THRESHOLDS}")
        print(f"  例：f=32.2%（原版中位）→ {adaptive_thresholds(0.322, ORIG_THRESHOLDS, a.thresh_margin)}"
              f"\n      f=16.6%（收 mask 后中位）→ "
              f"{adaptive_thresholds(0.166, ORIG_THRESHOLDS, a.thresh_margin)}")
    if a.thresholds == "max":
        # 去掉阈值这个旋钮：原版实测就是"每次跑满 max_refinement_steps"，
        # 阈值置 0 让我们的预算与它一致，对比里少一个自由参数。
        cfg["counting_model"]["loss"]["thresholds"] = {
            k: 0.0 for k in cfg["counting_model"]["loss"]["thresholds"]}
    elif a.thresholds:
        cfg["counting_model"]["loss"]["thresholds"] = {
            int(k): float(v) for k, v in
            (kv.split(":") for kv in a.thresholds.split(","))}
    if a.scale_factor is not None:
        cfg["counting_model"]["loss"]["scale_factor"] = a.scale_factor
    if a.loss == "instance":
        _patch_instance_loss(a)
        print(f"★ 损失 = 实例感知（w_cov={a.w_cov} w_sep={a.w_sep} "
              f"w_bg={a.w_bg} w_conc={a.w_conc} dilate={a.dilate}"
              + (f" hinge^{a.hinge_pow:g} τ_fg={a.tau_fg} τ_bg={a.tau_bg}"
                 if a.tau_fg is not None or a.tau_bg is not None
                 else " 无余量") + "）")
        print(f"  阈值 {cfg['counting_model']['loss']['thresholds']}"
              f"  步长系数 {cfg['counting_model']['loss']['scale_factor']}")
    if a.fg_max < 1.0:
        print(f"★ mask 面积约束：前景占比 ≤ {a.fg_max:.0%}")

    counter = _patch_counter_probe()
    if not a.no_mem_graph:
        _patch_mem_graph()
        print("显存改动（数值等价）：create_graph=False；进修正步前丢掉上一张图")
    if not a.no_mem_attn:
        _patch_mem_attn()
        print("显存改动（浮点级）：probs 不会被任何地方读的层改走 SDPA")
    pipe = _load_pipeline(cfg)
    if not a.no_mem_graph:
        _patch_unet_detach(pipe)
    phase1 = cfg["pipeline"]["phase1_type"]
    phase2 = cfg["pipeline"]["phase2_type"]
    assert phase1 == "dbscan_mask" and phase2 == "ours_counting_loss", \
        f"本驱动只覆盖论文配置，当前 config 是 {phase1}/{phase2}"

    meta_p, log_p = out / "metadata.json", out / "counter_log.jsonl"
    meta = json.load(open(meta_p)) if meta_p.exists() else []
    # 续跑判据要跟模式走：先跑过 --vanilla-only 的题，在完整模式下**不算做完**，
    # 否则补跑时会把它们全跳过，永远拿不到 CountGen 臂。
    last = {}
    if log_p.exists():
        for line in log_p.open():                 # 追加式日志，同 id 以最后一条为准
            if line.strip():
                r = json.loads(line)
                last[r["id"]] = r
    done = {i for i, r in last.items()
            if a.vanilla_only or r.get("has_countgen_img") or r.get("skipped_by_official")}
    partial = len(last) - len(done)
    if a.only_ids:
        want = {l.strip() for l in Path(a.only_ids).read_text().splitlines() if l.strip()}
        data = [d for d in data
                if f"{d['object']}_num={d['int_number']}_seed={d['seed']}" in want]
        done, partial = set(), 0        # 指定了 id 就重跑，不看续跑记录
        print(f"--only-ids：只跑 {len(data)} 题（忽略续跑记录）")
        if len(data) != len(want):
            print(f"⚠️ 文件里有 {len(want)} 个 id，数据集里只找到 {len(data)} 个")
    # ★ 同一个输出目录里混进两种配置 = 结果作废，而且是静默的。
    #   续跑逻辑只看 id，不看配置，所以必须在这里挡住。
    prev = {r.get("loss", "orig") for r in last.values()}
    if prev and prev != {a.loss}:
        sys.exit(f"!! {out} 里已有 loss={prev} 的记录，而本次是 loss={a.loss}。\n"
                 f"   续跑只按 id 判断，混在一起会得到一个两种配置各跑一半的目录。\n"
                 f"   换个 --out（例如 .../cocoount_inst）。")
    if done:
        print(f"续跑：已完成 {len(done)} 题")
    if partial:
        print(f"其中 {partial} 题此前是 --vanilla-only 跑的，本次会补上修正那一步")

    t_all = time.time()
    for item in tqdm(data, desc="CoCoCount"):
        prompt, seed = item["prompt"], item["seed"]
        N, obj = item["int_number"], item["object"]
        img_id = f"{obj}_num={N}_seed={seed}"
        if img_id in done:
            continue
        over9 = N > 9
        if over9 and a.skip_over9:
            continue

        t0 = time.time()
        fg0 = fg1 = float("nan")        # --vanilla-only 那一支不走 mask 这段
        thr_used = None
        # ★ 顺序必须与 run_countgen.py:94-98 一致：先 set_seed 再造 latents
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        shape = (1, pipe.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=generator,
                               device=pipe.device, dtype=torch.float16)

        if over9 or a.vanilla_only:
            # 只跑 vanilla + 计数，**不进 relayout**。两种情况会走到这里：
            #   · N>9：官方在更早的地方就 continue 了（run_countgen.py:104），
            #     根因是 ReLayout U-Net 只有 9 个通道（relayout.py:7），
            #     送 10 进去是未定义行为。官方连 baseline 都不留，我们留。
            #   · --vanilla-only：还没拿到 ReLayout 权重时的先行档。
            # 这里复刻 extract_mask.py:12-16 的前两步（DBSCAN + 去稀疏 blob），
            # 得到的 n_dbscan 与完整流程里那个是同一个量。
            raw, _, vanilla_img = dbscan_extract_mask(prompt, pipe, cfg, seed)
            _, n_dbscan = remove_sparse_blobs(raw)
            n_used, match, image = n_dbscan, n_dbscan == N, None
        else:
            counter.pop("n", None)
            vanilla_masks, correct_mask, object_masks, vanilla_img, match = relayout(
                pipe, prompt, N, cfg, seed)
            n_dbscan = counter.get("n")                 # DBSCAN 真实簇数（可能是 0）
            n_used = int(vanilla_masks.max().item())    # 兜底之后管线实际用的数
            # 组件 2：把 desired_mask 收紧。必须在这里做 —— 它是喂给
            # run_counting_pipeline_corrected_masks 的那张图，也是存进 npz 的那张。
            fg0 = fg1 = float((object_masks > 0).float().mean())
            if a.fg_max < 1.0:
                shrunk, fg0, fg1 = shrink_mask(object_masks.cpu().numpy(),
                                               fg_max=a.fg_max)
                object_masks = torch.tensor(shrunk, dtype=object_masks.dtype,
                                            device=object_masks.device)
            # 组件 3：逐题把退出阈值锚到解析下界上。必须在收完 mask 之后算 ——
            # 下界只取决于最终喂进去的那张 mask 的前景占比。
            if a.thresh_margin is not None:
                thr_used = adaptive_thresholds(fg1, ORIG_THRESHOLDS,
                                               a.thresh_margin)
                cfg["counting_model"]["loss"]["thresholds"] = thr_used
            # 把三张 mask 存成数组（32×32，几百字节）。可视化 png 看得见但量不了，
            # 而"新增的 blob 位置上到底长没长出物体"这个问题必须拿数组去问。
            np.savez_compressed(out / f"{img_id}_masks.npz",
                                vanilla=vanilla_masks.cpu().numpy(),
                                corrected=correct_mask.cpu().numpy(),
                                postprocess=object_masks.cpu().numpy(),
                                n_dbscan=n_dbscan, N=N)
            if not a.no_masks:
                show_mask_list([vanilla_masks, correct_mask, object_masks],
                               titles=[f"Vanilla: {n_used}",
                                       f"Corrected: {int(correct_mask.max())}",
                                       f"Postprocess: {int(object_masks.max())}"],
                               save_path=str(out / f"{img_id}_masks.png"))
            # ★ 与官方一致：计数器认为已经对了，就直接输出原版图，不做任何干预
            if match:
                image = vanilla_img
            else:
                # 修正步是显存峰值所在。原版那一趟存下来的 9 个时间步 × 各层
                # 32×32 注意力图（约 0.7 GB）到这里已经用完了 —— 下一次 __call__
                # 会新建 store，所以现在清掉是安全的。
                for st in (pipe.attention_store.self_step_store,
                           pipe.attention_store.cross_step_store):
                    st.clear()
                pipe.attention_store.all_cross_attention = {}
                pipe.attention_store.all_self_attention = {}
                torch.cuda.empty_cache()
                image = run_counting_pipeline_corrected_masks(
                    pipe, prompt, generator, object_masks, latents, cfg)
        t_count = time.time() - t0

        vanilla_img.save(out / f"{img_id}_vanilla.png")
        if image is not None:
            image.save(out / f"{img_id}.png")

        rec = {"id": img_id, "prompt": prompt, "seed": seed, "obj_class": obj,
               "requiered_object_num": N,          # 官方 metadata 的拼写，保持一致
               "n_dbscan": n_dbscan,
               "n_used": n_used,
               "zero_cluster_fallback": n_dbscan == 0,
               "obj_num_match": bool(match),
               "skipped_by_official": bool(over9),
               "vanilla_only": bool(a.vanilla_only),
               "has_countgen_img": image is not None,
               "sec": round(t_count, 2),
               "loss": a.loss, "fg_before": round(fg0, 4), "fg_after": round(fg1, 4),
               "l_min": None if fg1 != fg1 else round(l_min(fg1), 4),
               "thresholds": thr_used and {k: round(v, 4) for k, v in thr_used.items()},
               "refine": list(REFINE)}
        REFINE.clear()
        if not any(m["id"] == img_id for m in meta):     # 补跑时别重复写
            meta.append({k: rec[k] for k in
                         ("id", "prompt", "seed", "obj_class", "requiered_object_num")})
        json.dump(meta, open(meta_p, "w"), indent=4)
        with log_p.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()

    n = len(list(out.glob("*_vanilla.png")))
    print(f"\n完成。vanilla {n} 张，用时 {(time.time()-t_all)/60:.1f} 分钟 → {out}")
    print(f"下一步：\n  python {REPO}/count_probe/make_arms.py --src {out} "
          f"--out {out.parent / (out.name + '_arms')}")


if __name__ == "__main__":
    main()
