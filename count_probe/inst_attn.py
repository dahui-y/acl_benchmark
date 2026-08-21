#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方向 d 的两个部件：实例级自注意力遮罩（A）与 `up_52` 亲和损失（B）。零训练。

    立项依据见 DESIGN2.md。一句话：CountGen 从自注意力里**读**出了实例身份
    （`l^up_52` + DBSCAN），却只往回**写**了一个前景/背景的二值约束
    （`loss_utils.py:5` 把布局二值化；`attention_processors.py:45-61` 只处理
    「背景 → blob 并集」）。两个强制机制都不认实例数，而计数的核心正是实例分离。

    ── 遮罩（zero_mask）────────────────────────────────────────────────
    三种模式，前两种是**在位者的两个版本**，第三种才是我们的：

      code   发布版实测行为：背景 token 的**整行**置零。
             因为 `for j in range(0, max_blob_index+1)` 里 j=0 是背景，
             `blob_coordinates` 被填成全 1（见 DESIGN2 §2.0 的合成验证）。
      paper  论文 §3.3 的公式：`S*[i,j]=0 if i∈B and j∈F`，背景仍可看背景。
      —— 以上两种都不区分实例。
      inter  **我们加的**：再置零「blob_a → blob_b」(a≠b)，
             让每个实例只能从自身区域 + 背景聚合，不能与邻居互相拷贝。
             治的是「两个 blob 合并成一个物体」。

    ⚠️ 与在位者保持一致：置零后**不重新归一化**（注意力行不再和为 1）。
       这很可能是无心之失，但改它会同时改变基线行为，故列为消融项而非默认。

    ── 亲和损失（affinity_loss）────────────────────────────────────────
    治的是另一个通道：「一个 blob 内裂出多个物体」。切跨 blob 注意力对它无效，
    因为那是 blob **内部**现象。做法是把 DBSCAN 读的那张矩阵拿来写：

        L_aff = mean_{同 blob}(1 − A_norm) + mean_{异 blob}(A_norm)

    即「让 DBSCAN 恰好读出 k 个簇」的可微版本 —— 一个 blob 裂成两个物体，
    其内部亲和矩阵会出现两个子块，第一项直接罚它。
    归一化沿用原版 `object_layout_loss` 的 min-max，避免重蹈方向 a 查明的
    量纲失配（FINDINGS §2.3：换损失后原版阈值全部失效）。

自检：
    python count_probe/inst_attn.py --self-test
"""

import argparse


def zero_mask(lab, mode="code", inter_blob=False):
    """→ 布尔矩阵 (HW, HW)，True 处的注意力概率要被置零。

    lab: (H, W) 或 (HW,) 的整数标签图，0=背景，1..k=实例。
    """
    import torch
    lab = lab.reshape(-1).long()
    fg = lab > 0
    n = lab.numel()
    if mode == "code":
        m = (~fg).unsqueeze(1).expand(n, n).clone()          # 背景行 → 全部
    elif mode == "paper":
        m = (~fg).unsqueeze(1) & fg.unsqueeze(0)             # 背景行 → 前景列
    elif mode == "off":
        m = torch.zeros((n, n), dtype=torch.bool, device=lab.device)
    else:
        raise ValueError(f"未知 mode={mode}")
    if inter_blob:
        m = m | (fg.unsqueeze(1) & fg.unsqueeze(0)
                 & (lab.unsqueeze(1) != lab.unsqueeze(0)))
    return m


def affinity_loss(self_attn, lab, eps=1e-8):
    """`up_52` 自注意力的实例级亲和损失。

    self_attn: (1, HW, HW) 或 (HW, HW)，存储时已对头平均。
    lab:       (H, W) 整数标签图。
    无前景或只有一个 blob 时，「异 blob」项为空，只保留「同 blob」项。
    """
    import torch
    a = self_attn
    if a.dim() == 3:
        a = a[0]
    lab = lab.reshape(-1).long().to(a.device)
    fg = lab > 0
    zero = a.sum() * 0.0                       # 保持在计算图上，别返回常数
    if not bool(fg.any()):
        return zero
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + eps)             # 与原版同款 min-max
    same = fg.unsqueeze(1) & fg.unsqueeze(0) & (lab.unsqueeze(1) == lab.unsqueeze(0))
    diff = fg.unsqueeze(1) & fg.unsqueeze(0) & (lab.unsqueeze(1) != lab.unsqueeze(0))
    l = zero
    if bool(same.any()):
        l = l + (1.0 - a[same]).mean()
    if bool(diff.any()):
        l = l + a[diff].mean()
    return l


def _self_test():
    import torch
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}: {got}（期望 {want}）")

    # 4×4 布局：blob1 两格、blob2 两格、背景 12 格
    lab = torch.tensor([[0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [2, 2, 0, 0],
                        [0, 0, 0, 0]])
    n, nfg, nbg = 16, 4, 12
    print("zero_mask：")
    chk("code 置零数 = 背景行 × 全列", int(zero_mask(lab, "code").sum()), nbg * n)
    chk("paper 置零数 = 背景行 × 前景列", int(zero_mask(lab, "paper").sum()), nbg * nfg)
    chk("off 置零数", int(zero_mask(lab, "off").sum()), 0)
    # 跨 blob：blob1(2格) × blob2(2格) 双向 = 8
    chk("inter 单独 = 跨 blob 对数",
        int(zero_mask(lab, "off", inter_blob=True).sum()), 2 * 2 * 2)
    chk("paper+inter = 两者之并",
        int(zero_mask(lab, "paper", inter_blob=True).sum()), nbg * nfg + 8)
    # code 已含背景整行，加 inter 只多跨 blob 那 8 个
    chk("code+inter", int(zero_mask(lab, "code", inter_blob=True).sum()), nbg * n + 8)
    # 对角线（自己看自己）不该被 inter 置零
    m = zero_mask(lab, "off", inter_blob=True)
    chk("inter 不碰对角线", int(m.diagonal().sum()), 0)

    # ★ 与发布版实现的**数值等价性**：我们的向量化替换必须与原始循环逐元素相同，
    #   否则「基线」就被我们悄悄改掉了，后面所有对比都不作数。
    print("与发布版 attention_processors.py:40-63 的等价性：")
    torch.manual_seed(0)
    for trial in range(20):
        d = 4
        lb = torch.randint(0, 3, (d, d))
        probs = torch.rand(2, d * d, d * d)
        # —— 原始循环，逐行复刻 —— #
        ref = probs.clone()
        mx = int(lb.max())
        bc = torch.zeros((d, d))
        for j in range(0, mx + 1):
            idx = (lb == j).nonzero(as_tuple=False)
            if len(idx) > 0:
                bc[idx[:, 0], idx[:, 1]] = 1
        bcf = bc.view(-1).bool()
        ref = ref.view(2, d, d, d * d)
        for idx in (lb == 0).nonzero(as_tuple=False):
            ref[:, idx[0], idx[1], bcf] = 0
        ref = ref.reshape(2, d * d, d * d)
        # —— 我们的 —— #
        got = probs.masked_fill(zero_mask(lb, "code").unsqueeze(0), 0.0)
        if not torch.equal(ref, got):
            ok = False
            print(f"  ✗ 第 {trial} 组不等")
            break
    else:
        print("  ✓ 20 组随机布局 × 随机注意力，与原始循环逐元素相同")

    print("affinity_loss：")
    HW = 16
    # 理想矩阵：同 blob 亲和 1、异 blob 0 → 归一化后损失应为 0
    ideal = torch.zeros(HW, HW)
    f = (lab.reshape(-1) > 0)
    same = f.unsqueeze(1) & f.unsqueeze(0) & (lab.reshape(-1).unsqueeze(1) == lab.reshape(-1).unsqueeze(0))
    ideal[same] = 1.0
    v = float(affinity_loss(ideal, lab))
    print(f"  {'✓' if v < 1e-5 else '✗'} 理想亲和矩阵 → {v:.6f}（期望 ≈0）")
    ok &= v < 1e-5
    # 最坏：同 blob 0、异 blob 1 → 两项各 1
    worst = torch.zeros(HW, HW)
    diff = f.unsqueeze(1) & f.unsqueeze(0) & (lab.reshape(-1).unsqueeze(1) != lab.reshape(-1).unsqueeze(0))
    worst[diff] = 1.0
    v = float(affinity_loss(worst, lab))
    print(f"  {'✓' if abs(v - 2.0) < 1e-5 else '✗'} 最坏亲和矩阵 → {v:.6f}（期望 2.0）")
    ok &= abs(v - 2.0) < 1e-5
    # 「一个 blob 裂成两个子块」必须比「一个 blob 一个簇」更差 —— 这是 B 的立身之本
    lab2 = torch.zeros(4, 4, dtype=torch.long)
    lab2[0, :4] = 1                                   # 一个 blob，4 格
    one = torch.zeros(HW, HW)
    idx = (lab2.reshape(-1) == 1).nonzero().flatten()
    one[idx[:, None], idx[None, :]] = 1.0             # 一个紧簇
    two = torch.zeros(HW, HW)
    for g in (idx[:2], idx[2:]):                      # 裂成两个子块
        two[g[:, None], g[None, :]] = 1.0
    v1, v2 = float(affinity_loss(one, lab2)), float(affinity_loss(two, lab2))
    good = v2 > v1
    ok &= good
    print(f"  {'✓' if good else '✗'} 裂成两子块 {v2:.4f} > 单簇 {v1:.4f}（B 的立身之本）")
    # 梯度必须能回传
    x = torch.rand(HW, HW, requires_grad=True)
    affinity_loss(x, lab).backward()
    g = x.grad is not None and float(x.grad.abs().sum()) > 0
    ok &= g
    print(f"  {'✓' if g else '✗'} 梯度非零：{float(x.grad.abs().sum()):.4f}")

    print("\n全部通过 ✓" if ok else "\n有失败项 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_self_test() if a.self_test else ap.print_help())
