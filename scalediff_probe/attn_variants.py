"""P0 的两个对照臂：NPA + Query Window Random Shifting，以及 MultiDiffusion。

**不改基线文件。** `help_code/ScaleDiff/SDXL/attn_scalediff_sdxl.py` 一个字
不动，这里只做子类与替换安装（纪律见 BASELINE_PATCHES.md）。

────────────────────────────────────────────────────────────────────────
三个臂是什么、为什么是这三个
────────────────────────────────────────────────────────────────────────
ScaleDiff 自己的 Table 3（4096² SDXL，同管线只换注意力）：

    MultiDiffusion   FID 61.71  KID .0021  IS 19.71  FIDp 38.08  KIDp .0069  ISp 20.85   239s
    NPA (theirs)     FID 61.87  KID .0025  IS 19.56  FIDp 38.89  KIDp .0080  ISp 20.41   113s

**七列质量指标全输，只赢时间。** 原文也承认：*"MultiDiffusion generates
high-quality images and achieves the best scores, but suffers from
substantial computational overhead."* 缺口就是我们要拿回来的东西。

而他们的附录 B.1 里还躺着一个**没启用**的补丁：

> *"minor boundary artifacts can sometimes appear ... Query Window Random
> Shifting is an optional technique ... with minimal computational
> overhead. **Note that this technique was not used during the evaluation
> in this paper.**"*

开源代码里 `grep shift` 零命中 —— 只存在于论文文字。

所以三个臂回答的是**同一个问题的两半**：

    A  npa      基准（他们的评测配置）
    B  shift    NPA + 逐层随机平移的 query 网格
    C  ovl      重叠 query 窗口 + 层内平均（**不是 MultiDiffusion**，见下）

⚠️ 原先写的 "g=C−A 是缺口总量、b/g 是边界占比" 那套读法**已作废** ——
它锚在"C 复现了他们发表的 MultiDiffusion 行"这个前提上，而该前提是错的。
重写后的判据见 p0_run.py 文件头。

────────────────────────────────────────────────────────────────────────
Query Window Random Shifting 的实现（照附录 B.1 的字面）
────────────────────────────────────────────────────────────────────────
原文：*"randomly sample the top and left offsets uniformly from [0, h/2]
and [0, w/2]. The query tensor is then zero-padded by a total of h/2 in
height and w/2 in width using these randomly sampled top-left offsets.
Query patches are subsequently extracted from this enlarged canvas.
After attention computation, regions corresponding to the added padding
are discarded. Since offsets are independently resampled at each layer..."*

这里 h,w 是**原生**窗口尺寸（= window_size），故 padding 总量 = p2 = ws//2。
记原始 latent 网格 H×W（H = height_scale × ws），则：

    画布 (H+p2) × (W+p2)，上侧 pad `top`、下侧 `p2-top`，左右同理；
    query patch 数从 nh×nw 变成 (nh+1)×(nw+1)；
    第 (i,j) 块在**原始**坐标里的起点 = (i*p2 − top, j*p2 − left)，可为负；
    对应 K/V 窗口起点 = clamp(起点 − p4, 0, H − p1)，与基线同一条 clamp。

基线的 `get_kv_view` 用 Python 双重循环建索引；shifting 每层每步都要重建，
那样太慢，所以这里**向量化**重建（几百微秒），不缓存也够快。

**自检（`--selftest`）**：top=left=0 且不 pad 时，本实现必须与基线
`AttnControl` 的索引和输出**逐元素相同**。不同就是实现错了，不是"差不多"。

────────────────────────────────────────────────────────────────────────
`md` 臂 —— **它不是 MultiDiffusion。订正一处实质性错误（2026-08-15）**
────────────────────────────────────────────────────────────────────────
本臂 query 与 K/V 用**同一个** p1×p1 重叠窗口（步长 p1/2），窗口内做完整
自注意力，重叠处按计数平均，窗口数 (2s−1)²。

我原以为这就复现了他们 Table 3 的 MultiDiffusion 行。**错了。**
实测 85s vs NPA 75s = **1.13×**，而论文是 239/113 = **2.11×**。
证据在他们 Table 1（我先前读漏了）：

    Method          Linear         Conv              Cross-Attn     Self-Attn
    Base            s2hwd2         s2hwk2d2          s2hwld         s4h2w2d
    MultiDiffusion  (2s-1)2hwd2    (2s-1)2hwk2d2     (2s-1)2hwld    (2s-1)2h2w2d
    NPA             s2hwd2         s2hwk2d2          s2hwld         s2h2w2d

**MultiDiffusion 的 Linear / Conv / Cross-Attn 三列全按 (2s−1)² 缩放**
—— 它把**整个 UNet 在每个重叠 patch 上各跑一遍**再平均。原文讲 NPA 的
卖点时也说得很清楚：*"eliminates redundant computations caused by
overlapping patches, thereby keeping the computational cost of
non-self-attention layers unchanged"*。
本实现只换了注意力，卷积/线性仍是一次全图前向，故只贵 13%。

**那本臂到底是什么**：「MultiDiffusion 的注意力模式 + NPA 的其余部分」。
还有一处性质要记牢 —— 它给每个 query 的上下文**比 NPA 更少**：

    A npa   query 1024 token，KV 4096  -> 上下文 4x，无重叠
    C 本臂  query 4096 token，KV 4096  -> 上下文 1x，有层内重叠平均

所以它既不是 MultiDiffusion，也不是"上下文更多"那一端；
它测的是**重叠平均**单独值多少钱。目录名沿用 `md`（改名会断续跑），
但报表里一律标为 `ovl-attn`，别让错标传下去。

**顺带记一条失败的检查**：我曾提议用峰值显存判断 MD 有没有装上 ——
没用。峰值由 UNet 别处的激活决定（三臂都是 11.9~12.0GB），注意力的
瞬时张量淹在里面。**真正抓到问题的是单张时长。**

    python scalediff_probe/attn_variants.py --selftest
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

SDXL_DIR = Path(__file__).resolve().parent.parent / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))

from attn_scalediff_sdxl import (AttnControl,                      # noqa: E402
                                 AttnProcessor2_0_local)


# ══════════════════════════════════════════════════════════════════════
# 索引构造（向量化；shift=0 时必须与基线逐元素相同）
# ══════════════════════════════════════════════════════════════════════
def kv_view_shift(H, W, ws, top, left, device):
    """带偏移的 K/V 窗口索引，形状 [(nh+1)(nw+1), p1*p1]。

    top=left=0 且退回不加 pad 的网格时，与基线 `get_kv_view` 相同 ——
    由 selftest 逐元素核对。
    """
    p1, p2, p4 = int(ws), int(ws) // 2, int(ws) // 4
    # 画布上的 query 块起点（原始坐标系，可为负）
    r0 = torch.arange(0, H + p2, p2, device=device) - top
    c0 = torch.arange(0, W + p2, p2, device=device) - left
    r_win = (r0 - p4).clamp(0, H - p1)
    c_win = (c0 - p4).clamp(0, W - p1)
    ar = torch.arange(p1, device=device)
    rows = r_win[:, None] + ar                       # (nr, p1)
    cols = c_win[:, None] + ar                       # (nc, p1)
    idx = rows[:, None, :, None] * W + cols[None, :, None, :]
    return idx.reshape(-1, p1 * p1)                  # i 外 j 内，行主序


def kv_view_plain(H, W, ws, device):
    """无偏移版（基线的等价向量化实现），[nh*nw, p1*p1]。"""
    p1, p2, p4 = int(ws), int(ws) // 2, int(ws) // 4
    r_win = (torch.arange(0, H, p2, device=device) - p4).clamp(0, H - p1)
    c_win = (torch.arange(0, W, p2, device=device) - p4).clamp(0, W - p1)
    ar = torch.arange(p1, device=device)
    rows = r_win[:, None] + ar
    cols = c_win[:, None] + ar
    idx = rows[:, None, :, None] * W + cols[None, :, None, :]
    return idx.reshape(-1, p1 * p1)


def kv_view_ctx(H, W, ws, kv, device):
    """**只把 K/V 窗口放大**，query 切法与基线完全一致。

    这是唯一干净的"上下文"探针。`shift` 只动窗口位置（测边界伪影），
    `ovl-attn` 的上下文反而更少（1x vs 4x，与重叠平均混淆）——
    两个都回答不了"上下文够不够"这个问题。

    窗口以 query 块中心为心：query 块 i 覆盖 [i*p2, i*p2+p2)，中心
    i*p2 + p2/2，故 K/V 起点 = 中心 − kv/2，再 clamp 到边界内。
    **kv = ws 时与基线 `get_kv_view` 逐元素相同**（基线的
    `clamp(i*p2 − p4, ...)` 正是这个式子在 kv=ws=2*p2 时的特例），
    由 selftest ⑦⑧ 核对。
    """
    p2 = int(ws) // 2
    kv = int(kv)
    if kv > H or kv > W:
        kv = min(H, W)                      # 放不下就退到整幅（全局注意力）
    r0 = torch.arange(0, H, p2, device=device) + p2 // 2 - kv // 2
    c0 = torch.arange(0, W, p2, device=device) + p2 // 2 - kv // 2
    r_win = r0.clamp(0, H - kv)
    c_win = c0.clamp(0, W - kv)
    ar = torch.arange(kv, device=device)
    rows = r_win[:, None] + ar
    cols = c_win[:, None] + ar
    idx = rows[:, None, :, None] * W + cols[None, :, None, :]
    return idx.reshape(-1, kv * kv)


def md_view(H, W, ws, device):
    """MultiDiffusion 的重叠窗口起点，步长 p1/2；返回索引与 (nr, nc)。"""
    p1, st = int(ws), int(ws) // 2
    r = torch.arange(0, H - p1 + 1, st, device=device)
    c = torch.arange(0, W - p1 + 1, st, device=device)
    if r.numel() == 0:
        r = torch.zeros(1, dtype=torch.long, device=device)
    if c.numel() == 0:
        c = torch.zeros(1, dtype=torch.long, device=device)
    ar = torch.arange(p1, device=device)
    rows, cols = r[:, None] + ar, c[:, None] + ar
    idx = rows[:, None, :, None] * W + cols[None, :, None, :]
    return idx.reshape(-1, p1 * p1), (r.numel(), c.numel())


# ══════════════════════════════════════════════════════════════════════
# 控制器
# ══════════════════════════════════════════════════════════════════════
class ShiftAttnControl(AttnControl):
    """NPA + Query Window Random Shifting。偏移**每层独立重采样**（附录 B.1）。"""

    def __init__(self, generator=None):
        super().__init__()
        self.generator = generator          # 传入以保证可复现

    def initialize(self, height_scale, width_scale):
        self.height_scale = height_scale
        self.width_scale = width_scale
        self.kv_views = {}                  # shifting 下逐层现算，不预建

    def hw(self, ws):
        return int(self.height_scale * ws), int(self.width_scale * ws)

    def draw(self, ws, device):
        """在 [0, p2] 上均匀取 top/left。含端点，与原文 [0, h/2] 一致。"""
        p2 = int(ws) // 2
        g = self.generator
        t = int(torch.randint(0, p2 + 1, (1,), generator=g, device="cpu"))
        l = int(torch.randint(0, p2 + 1, (1,), generator=g, device="cpu"))
        return t, l


class ShiftProcessor(AttnProcessor2_0_local):
    """只改 NPA 那三行的切块方式，其余（norm/proj/residual）完全走基线。"""

    def patch(self, q, k, v):
        c, ws = self.controller, self.window_size
        H, W = c.hw(ws)
        p2 = int(ws) // 2
        top, left = c.draw(ws, q.device)
        self._last = (top, left, H, W)

        B, nH, L, C = q.shape
        q = q.view(B, nH, H, W, C)
        # pad 顺序从最后一维往前：(C, W, H)
        q = F.pad(q, (0, 0, left, p2 - left, top, p2 - top))
        q = rearrange(q, "B nH (nh p) (nw w) C -> (B nh nw) nH (p w) C",
                      p=p2, w=p2)

        view = kv_view_shift(H, W, ws, top, left, k.device)
        k = self._gather(k, view)
        v = self._gather(v, view)
        return q, k, v

    @staticmethod
    def _gather(kv, view):
        kv = kv[:, :, view, :]                       # B,nH,N,L,C
        B, nH, N, L, C = kv.shape
        return kv.permute(0, 2, 1, 3, 4).reshape(B * N, nH, L, C)

    def unpatch(self, x):
        top, left, H, W = self._last
        p2 = int(self.window_size) // 2
        nh, nw = H // p2 + 1, W // p2 + 1
        x = rearrange(x, "(B nh nw) nH (p w) C -> B nH (nh p) (nw w) C",
                      nh=nh, nw=nw, p=p2, w=p2)
        x = x[:, :, top:top + H, left:left + W, :]   # 丢掉 pad 出来的部分
        return x.reshape(x.shape[0], x.shape[1], H * W, x.shape[-1])


class CtxAttnControl(ShiftAttnControl):
    """NPA，但 K/V 窗口边长 = `units` × p2（p2 = ws//2 = query 块边长）。

    以 p2 为单位而不是以 ws 为单位，是为了留出细粒度退档：
        units=2 -> kv = ws      = 基线（面积 1.0x，selftest ⑧ 守逐字节相同）
        units=3 -> kv = 1.5*ws  （面积 2.25x）  <- 显存吃紧时的退档
        units=4 -> kv = 2*ws    （面积 4.0x）   <- 默认
    """

    def __init__(self, units=4):
        super().__init__(None)
        self.units = units


class CtxProcessor(AttnProcessor2_0_local):
    """query 与基线逐块相同，只有 K/V 窗口变大 —— 单变量。"""

    def patch(self, q, k, v):
        c, ws = self.controller, self.window_size
        H, W = c.hw(ws)
        p2 = int(ws) // 2
        self._last = (H, W)
        q = rearrange(q, "B nH (nh p nw w) C -> (B nh nw) nH (p w) C",
                      nh=H // p2, nw=W // p2, p=p2, w=p2)
        view = kv_view_ctx(H, W, ws, p2 * c.units, k.device)
        return q, ShiftProcessor._gather(k, view), ShiftProcessor._gather(v, view)

    def unpatch(self, x):
        H, W = self._last
        p2 = int(self.window_size) // 2
        return rearrange(x, "(B nh nw) nH (p w) C -> B nH (nh p nw w) C",
                         nh=H // p2, nw=W // p2, p=p2, w=p2)


class MDProcessor(AttnProcessor2_0_local):
    """MultiDiffusion：query 与 K/V 同为 p1×p1 重叠窗口，重叠处按计数平均。"""

    def patch(self, q, k, v):
        c, ws = self.controller, self.window_size
        H, W = int(c.height_scale * ws), int(c.width_scale * ws)
        view, grid = md_view(H, W, ws, q.device)
        self._last = (view, grid, H, W, q.shape)
        return (ShiftProcessor._gather(q, view),
                ShiftProcessor._gather(k, view),
                ShiftProcessor._gather(v, view))

    def unpatch(self, x):
        view, (nr, nc), H, W, qs = self._last
        B, nH, _, C = qs
        N = nr * nc
        x = x.reshape(B, N, nH, -1, C).permute(0, 2, 1, 3, 4)   # B,nH,N,L,C
        # 注：index_add_ 在 CUDA 上走原子加，**不是逐位可复现**的。
        # 对 C 臂（我们复现的对照，不是我们的方法）可以接受；
        # 若将来要拿 md 当主结果，这里要换成确定性的 scatter。
        out = torch.zeros(B, nH, H * W, C, dtype=x.dtype, device=x.device)
        cnt = torch.zeros(1, 1, H * W, 1, dtype=x.dtype, device=x.device)
        flat = view.reshape(-1)
        out.index_add_(2, flat, x.reshape(B, nH, -1, C))
        cnt.index_add_(2, flat, torch.ones(1, 1, flat.numel(), 1,
                                           dtype=x.dtype, device=x.device))
        return out / cnt.clamp(min=1)


def _make_call(cls):
    """把基线 __call__ 里 NPA 那一段换成 patch/unpatch，其余原样复用。"""
    base = AttnProcessor2_0_local.__call__

    def call(self, attn, hidden_states, encoder_hidden_states=None,
             attention_mask=None, temb=None, *args, **kw):
        if not self.controller.active or encoder_hidden_states is not None:
            return base(self, attn, hidden_states, encoder_hidden_states,
                        attention_mask, temb, *args, **kw)
        return _forward(self, attn, hidden_states, temb)

    cls.__call__ = call
    return cls


def _forward(self, attn, hidden_states, temb):
    """基线 __call__ 的自注意力分支，逐行照抄，只把切块换成 self.patch。"""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    ndim = hidden_states.ndim
    if ndim == 4:
        b, ch, hh, ww = hidden_states.shape
        hidden_states = hidden_states.view(b, ch, hh * ww).transpose(1, 2)
    B = hidden_states.shape[0]
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    q = attn.to_q(hidden_states)
    k = attn.to_k(hidden_states)
    v = attn.to_v(hidden_states)
    hd = k.shape[-1] // attn.heads
    q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
    k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
    v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)

    qp, kp, vp = self.patch(q, k, v)
    o = F.scaled_dot_product_attention(qp, kp, vp, dropout_p=0.0,
                                       is_causal=False)
    o = self.unpatch(o)

    o = o.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
    o = attn.to_out[1](attn.to_out[0](o))
    if ndim == 4:
        o = o.transpose(-1, -2).reshape(b, ch, hh, ww)
    if attn.residual_connection:
        o = o + residual
    return o / attn.rescale_output_factor


_make_call(ShiftProcessor)
_make_call(MDProcessor)
_make_call(CtxProcessor)

_WS = {"down_blocks.1": 64, "down_blocks.2": 32, "mid_block": 32,
       "up_blocks.0": 32, "up_blocks.1": 64}


def register(pipe, mode="npa", generator=None):
    """mode: npa（基线，原样调用上游）/ shift / md / ctx<N>。返回 controller。

    ctx<N>：K/V 窗口边长 = N × (window_size//2)。N 默认 4，即 2×ws、面积 4 倍。
    **ctx2 必须与 npa 逐字节相同** —— selftest ⑧ 守这条。
    """
    if mode == "npa":
        from attn_scalediff_sdxl import register_attention_control
        return register_attention_control(pipe)

    if mode.startswith("ctx"):
        # ctx / ctx2 / ctx3 ...，数字是 K/V 窗口的**边长**倍数
        ctrl = CtxAttnControl(int(mode[3:] or 4))
        cls = CtxProcessor
    else:
        ctrl = ShiftAttnControl(generator)
        cls = {"shift": ShiftProcessor, "md": MDProcessor}[mode]
    procs, sizes = {}, []
    for name in pipe.unet.attn_processors:
        ws = next((v for k, v in _WS.items() if name.startswith(k)), None)
        if ws is None:
            raise ValueError(f"unexpected attention name: {name}")
        if ws not in sizes:
            sizes.append(ws)
        procs[name] = (cls(ctrl, ws) if name.endswith("attn1.processor")
                       else pipe.unet.attn_processors[name])
    pipe.unet.set_attn_processor(procs)
    ctrl.set_window_sizes(sizes)
    return ctrl


# ══════════════════════════════════════════════════════════════════════
def selftest():
    """三条，全部逐元素核对，不看"差不多"。"""
    torch.manual_seed(0)
    ok = True

    # ① 向量化的无偏移索引 == 基线 Python 双循环的索引
    for ws, hs, wsx in ((64, 4, 4), (32, 4, 4), (64, 2, 3)):
        c = AttnControl()
        c.set_window_sizes([ws])
        c.initialize(hs, wsx)
        base = c.kv_views[ws]
        mine = kv_view_plain(int(hs * ws), int(wsx * ws), ws, "cpu")
        same = base.shape == mine.shape and torch.equal(base, mine)
        print(f"① kv_view 向量化 == 基线   ws={ws} s=({hs},{wsx})  "
              f"{base.shape} vs {mine.shape}  -> {'一致' if same else '不一致'}")
        ok &= same

    # ② shift 索引在 top=left=0 时，前 nh*nw 块应与无偏移版对齐
    H = W = 256
    ws = 64
    p2 = ws // 2
    s0 = kv_view_shift(H, W, ws, 0, 0, "cpu")
    pl = kv_view_plain(H, W, ws, "cpu")
    nh, nw = H // p2, W // p2
    sub = s0.reshape(nh + 1, nw + 1, -1)[:nh, :nw].reshape(nh * nw, -1)
    same = torch.equal(sub, pl)
    print(f"② shift(top=0,left=0) 的左上 {nh}×{nw} 块 == 无偏移版  "
          f"-> {'一致' if same else '不一致'}")
    ok &= same

    # ③ pad/unpad 是恒等：随机 q 走一遍 patchify+unpatchify 必须原样回来
    class _C(ShiftAttnControl):
        def __init__(self, t, l):
            super().__init__()
            self.t, self.l = t, l
            self.height_scale = self.width_scale = 4

        def draw(self, ws, device):
            return self.t, self.l

    for t, l in ((0, 0), (7, 31), (32, 0), (17, 5)):
        c = _C(t, l)
        p = ShiftProcessor(c, ws)
        q = torch.randn(2, 3, H * W, 8)
        qp, _, _ = p.patch(q, q, q)
        back = p.unpatch(qp)
        same = torch.allclose(back, q, atol=0)
        print(f"③ pad->unpad 恒等  top={t:>3} left={l:>3}  "
              f"q{tuple(q.shape)} -> patch{tuple(qp.shape)}  "
              f"-> {'恒等' if same else '**不恒等**'}")
        ok &= same

    # ④ MultiDiffusion 的窗口数应为 (2s−1)²（他们 Table 1 的 FLOPs 行）
    for s in (2, 4):
        _, (nr, nc) = md_view(s * ws, s * ws, ws, "cpu")
        want = 2 * s - 1
        print(f"④ MD 窗口数  s={s}  {nr}×{nc}  期望 {want}×{want}  "
              f"-> {'对' if (nr, nc) == (want, want) else '**错**'}")
        ok &= (nr, nc) == (want, want)

    # ⑤ MD 的 index_add 平均必须把常数场原样还原（覆盖完整、计数正确）
    view, (nr, nc) = md_view(H, W, ws, "cpu")
    x = torch.ones(1, 1, nr * nc, ws * ws, 1)

    class _M(MDProcessor):
        pass
    m = _M(_C(0, 0), ws)
    m._last = (view, (nr, nc), H, W, (1, 1, H * W, 1))
    out = m.unpatch(x.reshape(nr * nc, 1, ws * ws, 1))
    same = torch.allclose(out, torch.ones_like(out))
    print(f"⑤ MD 重叠平均：常数场还原  max|err|="
          f"{(out-1).abs().max():.2e}  -> {'对' if same else '**错**'}")
    ok &= same

    # ⑦ ctx 的索引在 kv=ws 时必须退化成基线索引
    for ws_, hs, wsx in ((64, 4, 4), (32, 4, 4), (64, 2, 3)):
        Hh, Ww = int(hs * ws_), int(wsx * ws_)
        a_ = kv_view_ctx(Hh, Ww, ws_, ws_, "cpu")
        b_ = kv_view_plain(Hh, Ww, ws_, "cpu")
        same = a_.shape == b_.shape and torch.equal(a_, b_)
        print(f"⑦ ctx(kv=ws) 索引 == 基线   ws={ws_} s=({hs},{wsx})  "
              f"-> {'一致' if same else '不一致'}")
        ok &= same

    # ⑧ ctx1 的**输出**必须与基线 NPA 逐字节相同（守整个前向）
    same = _equiv_check(H, W, ws, ctx_units=2)
    print(f"⑧ ctx2 (kv=ws) 输出 == 基线 NPA 输出   "
          f"-> {'逐字节相同' if same else '**不同，别跑 GPU**'}")
    ok &= same

    # ⑨ 放大后每个 query 看到的 token 数确实变了（防止 mult 被忽略）
    for u_ in (2, 3, 4, 6):
        kv_ = 32 * u_
        v = kv_view_ctx(256, 256, 64, kv_, "cpu")
        print(f"⑨ ctx{u_}  K/V 窗口 {kv_}²  每 query 可见 {v.shape[1]:>6} token"
              f"（基线 4096，面积 {kv_**2/4096:.2f}x）  块数 {v.shape[0]}")
        ok &= v.shape[1] == kv_ ** 2 and v.shape[0] == 64

    # ⑥ **最要紧的一条**：top=left=0 时 shift 臂必须与基线**逐字节相同**。
    #    此时 pad 全在下/右，裁掉后 query 块与 K/V 窗口与基线逐个对齐。
    #    这条覆盖的不只是索引，还有我重写的整个 _forward —— 少写一行
    #    norm、写错一次 transpose，B/C 两个臂就会被悄悄污染而看不出来。
    same = _equiv_check(H, W, ws)
    print(f"⑥ shift(0,0) 输出 == 基线 NPA 输出   "
          f"-> {'逐字节相同' if same else '**不同，别跑 GPU**'}")
    ok &= same

    print("\n" + ("全部通过。" if ok else "**有不通过项，先修再跑 GPU。**"))
    return 0 if ok else 1


def _fake_attn(dim, heads):
    """够 AttnProcessor 用的最小 Attention 替身（两条路径共用同一份权重）。"""
    import torch.nn as nn
    a = nn.Module()
    a.spatial_norm = None
    a.group_norm = None
    a.norm_q = None
    a.norm_k = None
    a.norm_cross = False
    a.heads = heads
    a.to_q = nn.Linear(dim, dim, bias=False)
    a.to_k = nn.Linear(dim, dim, bias=False)
    a.to_v = nn.Linear(dim, dim, bias=False)
    a.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
    a.residual_connection = False
    a.rescale_output_factor = 1.0
    return a.eval()


def _equiv_check(H, W, ws, dim=16, heads=2, ctx_units=None):
    torch.manual_seed(1)
    attn = _fake_attn(dim, heads)
    x = torch.randn(1, H * W, dim)

    base_c = AttnControl()
    base_c.set_window_sizes([ws])
    base_c.initialize(H // ws, W // ws)
    base_c.enable()
    with torch.no_grad():
        y0 = AttnProcessor2_0_local(base_c, ws)(attn, x)

    if ctx_units is not None:
        sc = CtxAttnControl(ctx_units)
        proc = CtxProcessor(sc, ws)
    else:
        class _Zero(ShiftAttnControl):
            def draw(self, ws_, device):
                return 0, 0
        sc = _Zero()
        proc = ShiftProcessor(sc, ws)
    sc.set_window_sizes([ws])
    sc.initialize(H // ws, W // ws)
    sc.enable()
    with torch.no_grad():
        y1 = proc(attn, x)

    if y0.shape != y1.shape:
        print(f"   形状就不同：{tuple(y0.shape)} vs {tuple(y1.shape)}")
        return False
    d = (y0 - y1).abs().max().item()
    if d:
        print(f"   max|diff| = {d:.3e}（要求 0）")
    return d == 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else
             ap.print_help() or 0)
