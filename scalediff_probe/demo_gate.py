"""DemoGate：v1 的门在 DemoFusion 前向结构上的化身。

与 ScaleDiff 版（BlendGate）的差别只有一个：ScaleDiff 放大阶段是
**单次全图前向**，门控图整张用；DemoFusion 放大阶段是 **patch 窗批 +
跨步子网格批**，同一批里每个元素看到画布的不同部分 —— 所以 weights()
必须按"当前批的视图坐标"给每个元素裁自己的那块图。裁剪上下文由
pipeline_demofusion_gated.py 的钩 2/3 在每次 UNet 调用前塞进来。

录制（phase 1）与逐位置混合的数学（BlendCrossAttn）**一行不改**，
原样复用 —— 这正是论文里"同一张图、同一个门，跨管线移植"的字面证据。

批序约定（与管线一致）：latent 批 = repeat_interleave(2)
-> [v1u, v1c, v2u, v2c, ...]，元素 i 的视图 = views[i//2]；
文本批 = cat([neg,pos]*V) 同序，所以 alt 嵌入按视图数平铺即可。
"""

import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_v1 import BlendGate            # noqa: E402


class DemoGate(BlendGate):

    def __init__(self, token_ids, strength=1.0, gate_dilated=True, **kw):
        super().__init__(token_ids, strength=strength, **kw)
        self.gate_dilated = gate_dilated
        self._alt_base = None          # (2, 77, D)，set_alt() 存入
        self.mode = None               # None / 'patch' / 'dilated'
        self._views = None
        self._jit = 0
        self._scale = 1
        self._pads = (0, 0)
        self._canvas = None            # (H, W) 当前 scale 的画布 latent 尺寸
        self._map_canvas = None        # 门控图上采样到画布尺寸的缓存
        # 仪表：门到底有没有在这条管线里工作 —— 用数字回答，不靠看图争
        self.n_blend = 0               # 实际做了逐位置混合的注意力层调用数
        self.n_skip = 0                # 因上下文缺失被跳过的调用数
        self._alt_frac_sum = 0.0       # 累计"背景（拿替代文本）的画面占比"

    # ---- alt 嵌入：基础 (2,77,D)，按当前批的视图数平铺 ----
    def set_alt(self, alt2):
        self._alt_base = alt2

    @property
    def alt(self):
        if self._alt_base is None:
            return None
        if self.mode is None or not self._views:
            return self._alt_base
        return self._alt_base.repeat(len(self._views), 1, 1)

    @alt.setter
    def alt(self, v):                  # 兼容 BlendGate 的 gate.alt = ... 写法
        self._alt_base = v

    # ---- 钩 2/3 塞进来的上下文 ----
    def set_patch_views(self, views, jitter, H, W):
        self.mode = "patch"
        self._views, self._jit, self._canvas = views, jitter, (H, W)

    def set_dilated_views(self, views, scale, h_pad, w_pad, H, W):
        if not self.gate_dilated:
            self.mode = None
            return
        self.mode = "dilated"
        self._views, self._scale = views, scale
        self._pads, self._canvas = (h_pad, w_pad), (H, W)

    def clear_views(self):
        self.mode, self._views = None, None

    # ---- 画布尺寸的门控图（缓存按尺寸失效）----
    def _canvas_map(self, device, dtype):
        H, W = self._canvas
        c = self._map_canvas
        if c is None or c.shape != (H, W) or c.device != device:
            c = F.interpolate(self.map[None, None].to(device=device,
                                                      dtype=torch.float32),
                              size=(H, W), mode="bilinear",
                              align_corners=False)[0, 0]
            self._map_canvas = c
        return c.to(dtype)

    # ---- 核心：按视图逐元素出混合权重 (B, hw, 1) ----
    def weights(self, hw, device, dtype):
        if self.map is None or self.s <= 0 or self.mode is None \
                or not self._views:
            self.n_skip += 1
            return None
        h = w = int(hw ** 0.5)
        if h * w != hw:
            self.n_skip += 1
            return None
        cm = self._canvas_map(device, torch.float32)
        H, W = cm.shape
        crops = []
        if self.mode == "patch":
            for hs, he, ws, we in self._views:
                # 窗坐标在 jitter 补边坐标系 -> 画布坐标系，越界取边缘
                y0, y1 = hs - self._jit, he - self._jit
                x0, x1 = ws - self._jit, we - self._jit
                y0c, y1c = max(y0, 0), min(y1, H)
                x0c, x1c = max(x0, 0), min(x1, W)
                crop = cm[y0c:y1c, x0c:x1c]
                # 越界部分按 replicate 补回，保持窗的几何对应
                pl, pr = x0c - x0, x1 - x1c
                pt, pb = y0c - y0, y1 - y1c
                if pl or pr or pt or pb:
                    crop = F.pad(crop[None, None], (pl, pr, pt, pb),
                                 mode="replicate")[0, 0]
                crops.append(crop)
        else:  # dilated
            h_pad, w_pad = self._pads
            cmp_ = F.pad(cm[None, None], (w_pad, 0, h_pad, 0),
                         mode="replicate")[0, 0]
            for dh, dw in self._views:
                crops.append(cmp_[dh::self._scale, dw::self._scale])
        ms = torch.stack([
            F.interpolate(c[None, None], size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0].reshape(hw)
            for c in crops
        ])                                            # (V, hw)
        ms = ms.repeat_interleave(2, dim=0)[..., None]  # (2V, hw, 1)，u/c 同图
        w_out = 1.0 - self.s * (1.0 - ms)
        self.n_blend += 1
        # 权重 <0.5 的位置 = 主要拿替代（去主体）文本的位置
        self._alt_frac_sum += float((w_out < 0.5).float().mean())
        return w_out.to(device=device, dtype=dtype)

    def report(self):
        """跑完打一行：门是否真的在这条管线里动过手。"""
        n = self.n_blend
        frac = self._alt_frac_sum / max(n, 1)
        return (f"门仪表：混合调用 {n} 次，跳过 {self.n_skip} 次，"
                f"背景（拿替代文本）平均占画面 {frac:.1%}"
                + ("   <- **混合 0 次 = 门根本没工作，先查这个**" if n == 0 else ""))
