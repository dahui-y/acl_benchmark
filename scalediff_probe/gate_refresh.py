"""v1.1：门控图在放大阶段重录 —— 打掉 8 倍上采样这个分辨率瓶颈。

**问题**（我们自己的数据指出来的）：基础阶段最细的 cross-attention 是
64×64，canon 也是 64；到 4096 时画布 latent 是 512×512，这张图被**拉了
8 倍**。实测末样本过渡带宽达 0.86 —— 门在边界上是"半开"的。
清不掉的三条 +1（474/734/956）、v2b 白斑溅到主体上，都指向同一处。

**做法**：在每个放大阶段的**第一步**重录一次。那一刻 latent ≈ 上采样
基图 + 噪声，语义仍是基图的（不会把已经长出来的副本录进去），而注意力
层的分辨率是 256×256（4096 时），**比基础阶段细 4 倍**。
录完 finalize 成新图，之后照常混合。

**为什么只在第一步录**：晚录会把正在形成的副本也当成"主体"录进门控图
—— 那会让门自己给副本发通行证。第一步是唯一安全的窗口。

预注册判据（写在跑之前）：
    R1 过渡带（0.3~0.7 的面积占比）从 v1 的 ~0.5-0.86 降到 < 0.3；
    R2 触发集 delta 均值 <= v1（不许变差），且 474/734/956 至少两条清零；
    R3 主体不被误伤：31 条里 delta = -1 的条数不增加；
    R4 成本 <= v1 × 1.02（录图不额外算，只多一次 finalize）。
判死：R1 过但 R2 不过 -> 分辨率不是残留 +1 的原因，转 v1.2（多变体）；
     R1 不过 -> 重录没拿到更细的图，查注意力层分辨率假设。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_v1 import BlendGate                      # noqa: E402


class RefreshGate(BlendGate):
    """在放大阶段第一步重录一次门控图，canon 提到 256。

    phase 语义扩展：
        1        基础阶段，录图（原样）
        'r'      放大阶段重录窗口：**照常录，不混合**
        2        正常混合
    """

    def __init__(self, token_ids, strength=1.0, canon=64,
                 refresh_canon=256, refresh_steps=1, **kw):
        super().__init__(token_ids, strength=strength, canon=canon, **kw)
        self.refresh_canon = refresh_canon
        self.refresh_steps = refresh_steps
        self._refresh_left = 0
        self.map_base = None          # 基础阶段那张，留档对比
        self.n_refresh = 0

    def begin_refresh(self):
        """放大阶段开始时调用：留档旧图，开一个重录窗口。"""
        if self.refresh_steps <= 0:
            self.phase = 2
            return
        self.map_base = None if self.map is None else self.map.clone()
        self._acc, self._n = None, 0
        self.canon = self.refresh_canon
        self._refresh_left = self.refresh_steps
        self.phase = "r"

    def step_done(self):
        """每个放大步结束时调用；重录窗口用完就 finalize 并转入混合。"""
        if self.phase != "r":
            return
        self._refresh_left -= 1
        if self._refresh_left <= 0:
            if self._acc is not None and self._n:
                self.finalize()
                self.n_refresh += 1
            else:
                # 一次都没录到（注意力层分辨率与假设不符）-> 退回旧图
                self.map = self.map_base
            self.phase = 2

    # phase == 'r' 时 weights() 必须返回 None（不混合，纯录）
    def weights(self, hw, device, dtype):
        if self.phase == "r":
            return None
        return super().weights(hw, device, dtype)

    def band_stats(self):
        """过渡带诊断：返回 (cov, 低权重区, 过渡带) 以及重录前后的对比。"""
        def st(m):
            if m is None:
                return None
            return (float((m > 0.5).float().mean()),
                    float((m < 0.3).float().mean()),
                    float(((m >= 0.3) & (m <= 0.7)).float().mean()))
        return {"before": st(self.map_base), "after": st(self.map),
                "n_refresh": self.n_refresh,
                "map_size": None if self.map is None else tuple(self.map.shape)}
