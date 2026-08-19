#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实例感知的计数引导目标 —— 我们的方法。纯函数，不依赖管线，可单独测。

    要替换的是 `help_code/make-it-count/utils/loss_utils.py` 的
    `object_layout_loss`。原版做的事：

        foreground_mask = (desired_mask != 0)          # ← N 个标签压成一张 0/1 图
        A = (A - A.min()) / (A.max() - A.min())        # 归一化到 [0,1]
        loss = BCEWithLogits(A, foreground_mask, pos_weight=10)

    三条被我们量出来的根因，逐条对应本文件的一处改动：

    根因 1 —— **损失读不到 N**。`(desired_mask != 0)` 把 1..N 的标签压成二值前景，
        「一个大物体铺满整片」和「N 个分开的物体」得分完全一样（前者甚至更高，
        因为 pos_weight=10 偏袒前景项）。
        测量：|残差|≥2 占 44.9%（n=98），单例 1 个 blob → 多出 6 匹马。
        → 改法：`L_peak` 逐 blob 取峰、并取**最弱的那个**（Attend-and-Excite 的思路），
          N 由此真正进入目标函数；`L_sep` 在相邻 blob 之间挖谷，强制分离。

    根因 2 —— **前景太大**。`postprocess.blob_merger` 取凸包合并，
        前景中位占画面 35.3%、最大 68.3%。一个 blob 装两个实例绰绰有余。
        → 改法：`shrink_mask` 逐 blob 腐蚀，直到总前景占比 ≤ fg_max。
          fg_max 的默认值 0.25 不是拍的，见根因 3。

    根因 3 —— **损失有下界，阈值不可达**。原版把 A 归一化到 [0,1] 后喂
        `binary_cross_entropy_with_logits`，而它内部还会过一次 sigmoid，
        于是"预测值"只能落在 [sigmoid(0), sigmoid(1)] = [0.500, 0.731]，两端都够不到。
        损失因此有一个只取决于前景占比 f 的下界：

            L_min(f) = f·10·(−log 0.731) + (1−f)·(−log 0.5) = 0.693 + 2.439·f

        阈值 {0:1.3, 10:1.2, 20:1.15} 在 f>25% / f>19% 时**数学上不可达**，
        实测 76–85% 的题落在不可达区 → refinement 每次跑满 20 步、把 latent 推远。
        → 改法：本文件的损失**不过 sigmoid**，各项都是 [0,1] 区间上的均值/最大值，
          可以真正降到 0；fg_max 默认取 0.25 正是那条不可达线。

    形式：

        L = L_peak + w_sep·L_sep + w_bg·L_bg + w_conc·L_conc

        L_peak = 1 − min_k ( max_{p∈blob k} A(p) )      每个实例至少要有一个峰，
                                                        盯最弱的那个
        L_sep  = mean_{p∈间隔带} A(p)                    相邻 blob 之间挖谷
        L_bg   = mean_{p∈背景} A(p)                      背景压低
        L_conc = mean_k ( mean_{blob k} A / max_{blob k} A )
                                                        blob 内要像"一个峰"而不是"铺满"，
                                                        直接压"一个 blob 里长两个"。
                                                        默认权重 0 —— 它是最不确定的一项，
                                                        先让主想法单独跑，再作消融打开。

    间隔带的定义：每个 blob 各膨胀一圈，被 **≥2 个** 膨胀区覆盖的格子。
    注意它**不限于背景格**：实测里相邻 blob 常常直接贴在一起（见 horse_num=7
    那张 mask，7 个 blob 是一条马群带被切成的竖条），中间根本没有背景格；
    只算背景会让间隔带为空，恰恰在最该起作用的地方失效。
"""

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


def normalize(A):
    """与原实现同样的 min-max 归一化，保持可比。"""
    A = A.float()
    return (A - A.min()) / (A.max() - A.min() + EPS)


def instance_layout_loss(A, M, w_sep=1.0, w_bg=1.0, w_conc=0.0, w_cov=0.0,
                         dilate=1, tau_fg=None, tau_bg=None, hinge_pow=1.0):
    """A: (H,W) 物体 token 的 cross-attention；M: (H,W) 标签 0..K（0=背景）。

    tau_fg / tau_bg 给了就走 **hinge**：只罚"前景暗于 τ_fg"和"背景亮于 τ_bg"，
    到位即梯度为 0。不给就是无余量版本（第一、二轮用的）。

    为什么需要 hinge —— 这是第二轮**肉眼看图**才发现的：
      原版把 A 归一化到 [0,1] 再喂 BCEWithLogits（内部又过一次 sigmoid），
      预测值只能落在 [0.500, 0.731]，最优解**够不到**。我起初把这当纯缺陷
      （"损失有下界、阈值不可达"），但它同时在当**正则化器**：因为够不到，
      优化永远处在"还在往那边走"的状态，不会真的抵达极端。
      我拆掉它之后，我们的损失有了一个**可达的退化最优解** —— 把 attention
      变成硬 0/1 图。而真实扩散模型的 cross-attention 本是平滑弥散的，
      逼 latent 产出退化注意力就得把它推离自然图像流形。
      实拍结果：`airplane_num=7_seed=149069` 从机场照片变成"白底黑剪影飞机"，
      且 v2（--thresholds max，每次跑满 20 步）比 v1（提前退出）更极端。
      hinge 把最优解从一个极端点改成一个**区间**，到位就松手。
    """
    A = normalize(A)
    K = int(M.max().item())
    if K == 0:                       # 没有 blob，退化成只压背景
        return A.mean()

    masks = torch.stack([(M == k) for k in range(1, K + 1)])       # (K,H,W) bool
    keep = masks.flatten(1).any(1)                                  # 空标签要剔掉
    masks = masks[keep]
    if masks.shape[0] == 0:
        return A.mean()

    # hinge：到位就松手。lo(x)=罚"太暗"，hi(x)=罚"太亮"。
    #
    # hinge_pow 为什么要能调：`relu(x)` 的导数是**阶跃**——满足余量处梯度精确为 0、
    # 不满足处是常数。于是注意力图上的梯度场成了一张 0/1 指示函数，边界就是
    # A=τ 这条等值线。把这种**分片常数、带硬边界**的梯度回传到 latent，
    # 解码出来就是画面被切成若干轴对齐矩形、块间有亮度断层
    # （tune_hinge 的 airplane_num=7_seed=149069 上肉眼可见，而原版 BCE 与
    #   无余量版的梯度都是连续的，没有这个现象）。
    # `relu(x)²` 的导数是 `2·relu(x)`，在边界处**连续地**降到 0：
    # 既保住"满足就不再推"，又去掉梯度场的阶跃。
    def _h(x):
        return x if hinge_pow == 1.0 else x.pow(hinge_pow) if torch.is_tensor(x) \
            else x ** hinge_pow

    lo = ((lambda x: _h(F.relu(tau_fg - x))) if tau_fg is not None
          else (lambda x: 1.0 - x))
    hi = ((lambda x: _h(F.relu(x - tau_bg))) if tau_bg is not None
          else (lambda x: x))

    # L_peak：逐 blob 取峰，盯最弱的那个 —— N 从这里进入目标函数
    peaks = torch.stack([A[m].max() for m in masks])
    L_peak = lo(peaks.min())

    # L_cov：逐 blob 的覆盖度。第一轮实测出来的教训 ——
    #   L_peak 只对每个 blob 的 argmax **一个格子**产生梯度，而原版 BCE 对
    #   整片前景（约 350 格）施压且带 pos_weight=10。梯度密度差两个数量级，
    #   结果是"删多余"那一支从 49.2% 塌到 17.4%：压制多余物体靠的正是那股
    #   稠密的前景/背景压力，我第一版把有用的那一面连同有害的一面一起扔了。
    # 与原版全局前景项的区别：**逐 blob 取平均再对 blob 平均**。
    #   原版的全局平均允许"一个 blob 过饱和补偿另一个空着"，逐 blob 不允许。
    if w_cov > 0:
        L_cov = torch.stack([lo(A[m]).mean() for m in masks]).mean()
    else:
        L_cov = A.sum() * 0

    # 间隔带：各 blob 膨胀一圈后被 ≥2 个覆盖的格子（含前景，理由见文件头）
    d = F.max_pool2d(masks.float().unsqueeze(1), 2 * dilate + 1,
                     stride=1, padding=dilate).squeeze(1)
    band = d.sum(0) >= 2
    L_sep = hi(A[band]).mean() if bool(band.any()) else A.sum() * 0

    bg = (M == 0)
    L_bg = hi(A[bg]).mean() if bool(bg.any()) else A.sum() * 0

    if w_conc > 0:
        conc = torch.stack([A[m].mean() / (A[m].max() + EPS) for m in masks])
        L_conc = conc.mean()
    else:
        L_conc = A.sum() * 0

    return (L_peak + w_cov * L_cov + w_sep * L_sep
            + w_bg * L_bg + w_conc * L_conc)


def shrink_mask(mask, fg_max=0.25, min_blob=4):
    """逐 blob 腐蚀，直到总前景占比 ≤ fg_max。返回 (新 mask, 原占比, 新占比)。

    fg_max 默认 0.25 来自我们自己推的那条线：原版损失的下界是 0.693+2.439·f，
    f>0.25 时 step0 的阈值 1.3 就不可达。虽然本文件的新损失没有这个下界，
    但"一个 blob 大到能装两个实例"这个问题是独立存在的，所以约束保留。

    腐蚀而不是缩放：腐蚀保形状、保连通、保位置（向内收），而缩放会移动重心。
    每个 blob 至少留 min_blob 个格子，不允许腐蚀到消失 —— 消失就等于少一个实例。
    """
    m = np.asarray(mask).astype(int).copy()
    total = m.size
    before = float((m > 0).sum()) / total
    if before <= fg_max:
        return m, before, before
    try:
        from scipy.ndimage import binary_erosion
    except ImportError:
        return m, before, before
    for _ in range(64):                       # 上限只是防死循环
        if float((m > 0).sum()) / total <= fg_max:
            break
        changed = False
        for k in range(1, int(m.max()) + 1):
            b = (m == k)
            if b.sum() <= min_blob:
                continue
            e = binary_erosion(b)
            if e.sum() < min_blob:
                continue
            m[b & ~e] = 0
            changed = True
        if not changed:                       # 全都缩到下限了，再腐蚀就会丢实例
            break
    return m, before, float((m > 0).sum()) / total


def _selftest():
    torch.manual_seed(0)
    H = 32
    # 造一个 3 实例的布局：三个分开的方块
    M = torch.zeros(H, H)
    for k, (r, c) in enumerate([(4, 4), (4, 20), (20, 12)], start=1):
        M[r:r + 5, c:c + 5] = k

    def peaky():
        """理想解：每个 blob 中心一个尖峰，别处接近 0"""
        A = torch.zeros(H, H)
        for r, c in [(6, 6), (6, 22), (22, 14)]:
            A[r, c] = 1.0
        return A

    def flood():
        """病态解：整片前景铺满（原版损失眼里和理想解一样好）"""
        return (M > 0).float()

    def merged():
        """两个实例粘成一个：中间那条也亮"""
        A = flood()
        A[4:9, 9:20] = 1.0
        return A

    def orig_loss(A, M):
        """原版 object_layout_loss（loss_utils.py:4-20）逐行复刻，用于对照。"""
        fg = (M != 0).float()
        A = (A - A.min()) / (A.max() - A.min())
        return F.binary_cross_entropy_with_logits(
            A, fg, pos_weight=torch.tensor(10.))

    print(f"  {'解':<24}{'原版 BCE':>10}{'我们的':>10}")
    vals = {}
    for name, A in (("理想（每 blob 一个尖峰）", peaky()),
                    ("铺满前景", flood()),
                    ("相邻粘连", merged())):
        o, n = orig_loss(A, M).item(), instance_layout_loss(A, M, w_conc=0.5).item()
        vals[name] = (o, n)
        print(f"  {name:<24}{o:>10.4f}{n:>10.4f}")

    ideal, fld = vals["理想（每 blob 一个尖峰）"], vals["铺满前景"]
    # ★ 这两条断言就是本方法存在的理由，写成检查而不是声明
    assert fld[0] < ideal[0], "预期原版损失更偏爱'铺满'"
    assert ideal[1] < fld[1], "我们的损失必须偏爱'理想解'"
    print(f"  ✓ 原版把「铺满」({fld[0]:.4f}) 排在「理想解」({ideal[0]:.4f}) **前面** "
          f"—— 值越小越优，所以它**主动奖励铺满**，不只是看不见 N")
    print(f"  ✓ 我们的把「理想解」({ideal[1]:.4f}) 排在「铺满」({fld[1]:.4f}) 前面")

    # 梯度能回传，且 w_cov 显著提高梯度密度 —— 那是第一轮的病根
    fg = (M > 0)

    def nnz(**kw):
        """返回 (前景里拿到梯度的格数, 背景里拿到梯度的格数)。

        要分开数：背景压力（L_bg）一直是稠密的，第一轮的病根是**前景**那一侧 ——
        L_peak 只对每个 blob 的 argmax 一个格子产生梯度。
        """
        A = torch.rand(H, H, requires_grad=True)
        instance_layout_loss(A, M, **kw).backward()
        assert A.grad is not None and torch.isfinite(A.grad).all()
        g = A.grad != 0
        return int((g & fg).sum()), int((g & ~fg).sum())

    f0, b0 = nnz(w_cov=0.0)
    f1, b1 = nnz(w_cov=1.0)
    n_fg = int(fg.sum())
    # hinge 到位后梯度必须**精确为 0** —— 这正是第二轮塌成剪贴画的解药：
    # 无余量版本会一路把 attention 往硬 0/1 图推，把 latent 顶离自然图像流形。
    # 造一个"够好但不极端"的解：前景 0.85、背景 0.15，两端各留一个锚点格
    # 让 min-max 归一化不会把它直接拉成 0/1（那样两个版本都是 0，测不出差别）。
    def good_not_extreme():
        A = torch.where(M > 0, torch.full((H, H), 0.85), torch.full((H, H), 0.15))
        A[0, 0] = 0.0                       # 归一化锚点：背景里的最暗格
        A[5, 5] = 1.0                       # 归一化锚点：blob 1 里的最亮格
        return A.clone().requires_grad_(True)

    gh = good_not_extreme()
    Lh = instance_layout_loss(gh, M, w_cov=1.0, tau_fg=0.8, tau_bg=0.2)
    Lh.backward()
    gn = good_not_extreme()
    Ln = instance_layout_loss(gn, M, w_cov=1.0)
    Ln.backward()
    print(f"  ✓ 前景 0.85 / 背景 0.15 这个「够好但不极端」的解："
          f"hinge 版 L={float(Lh.detach()):.4f}（梯度 {float(gh.grad.abs().sum()):.4f}），"
          f"无余量版 L={float(Ln.detach()):.4f}（梯度 {float(gn.grad.abs().sum()):.4f}）")
    assert float(Lh.detach()) == 0.0 and float(gh.grad.abs().sum()) == 0.0, \
        "hinge 满足 τ 后损失与梯度都该精确为 0"
    assert float(Ln.detach()) > 0.0, "无余量版会继续往极端推 —— 那正是塌图的来源"

    # hinge 的梯度场：relu 版在背景里只有"0"和"一个常数"两种取值（阶跃），
    # 平方版随"超出余量多少"连续变化。tune_hinge 的图上出现了轴对齐的矩形
    # 断层，怀疑就是阶跃梯度场回传到 latent 的结果。
    bgm = (M == 0)

    def bg_grad_levels(pw):
        A = torch.linspace(0.05, 0.45, H).repeat(H, 1).clone()
        A[M > 0] = 0.9
        A[0, 0], A[5, 5] = 0.0, 1.0
        A = A.requires_grad_(True)
        instance_layout_loss(A, M, w_cov=1.0, tau_fg=0.8, tau_bg=0.2,
                             hinge_pow=pw).backward()
        g = A.grad[bgm].abs()
        return len(torch.unique(torch.round(g[g > 1e-12] * 1e6)))

    l1, l2 = bg_grad_levels(1.0), bg_grad_levels(2.0)
    print(f"  ✓ 背景梯度的不同取值数：hinge_pow=1 → **{l1} 种（阶跃）**，"
          f"hinge_pow=2 → {l2} 种（平滑）")
    assert l1 <= 2 < l2, "平方 hinge 必须把阶跃梯度场变成平滑场"
    print(f"  ✓ 梯度可回传且有限。前景 {n_fg} 格中拿到梯度的："
          f"w_cov=0 → **{f0}**，w_cov=1 → **{f1}**；背景两者都是 {b0}/{b1}")
    assert f0 <= len(set(range(1, 4))) + 8, "w_cov=0 时前景应当几乎没有梯度"
    assert f1 == n_fg, "L_cov 必须让整片前景都拿到梯度"

    # shrink_mask
    big = torch.zeros(H, H)
    big[2:30, 2:30] = 1
    big[10:20, 10:20] = 2
    m2, b, aft = shrink_mask(big.numpy(), fg_max=0.25)
    print(f"  ✓ shrink_mask 前景 {b:.1%} → {aft:.1%}，"
          f"实例数 {int(big.max())} → {len(set(m2.ravel()) - {0})}")
    assert aft <= b and len(set(m2.ravel()) - {0}) == int(big.max())


if __name__ == "__main__":
    print("inst_loss 自测：")
    _selftest()
