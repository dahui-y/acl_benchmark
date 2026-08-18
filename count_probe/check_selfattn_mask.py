#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复算 make-it-count 的 self-attention masking 到底屏蔽了什么。

    `pipeline/attention_processors.py:45` 是：

        for j in range(0, max_blob_index + 1):
            current_blob_mask = self.attnstore.desired_mask == j
            ...
            blob_coordinates[indices[:, 0], indices[:, 1]] = 1

    而 desired_mask 的约定是 **0 = 背景、1..N = 实例**
    （`pipeline/mask_extraction/utils_masks.py: from_channels` 里
      `combined_mask[channels_mask[i] == 1] = i + 1`，其余保持 0）。

    循环从 j=0 起，就把背景也算进 blob_coordinates，于是它在整张 32×32 上
    全为 1。随后第 56–61 行对每个背景位置把
    `attention_probs[:, x, y, blob_coordinates_flat] = 0`，
    等价于**把背景 query 的整行注意力清零** —— 不是"背景不许看前景"，
    而是"背景什么都不许看"。

    这个脚本不依赖 GPU、不依赖 diffusers，只把那段索引逻辑原样搬过来跑一遍，
    好让这条判断可复核而不是"我读代码觉得"。

用法：
    python count_probe/check_selfattn_mask.py
"""

import torch

ATTN_DIM = 32          # 与 pipeline_config.yaml 的 cross_attention_dim 一致


def blob_coordinates(desired_mask, start):
    """原样复刻 attention_processors.py:40-53，只把循环起点参数化。"""
    max_blob_index = int(torch.max(desired_mask).item())
    bc = torch.zeros_like(desired_mask)
    for j in range(start, max_blob_index + 1):
        idx = (desired_mask == j).nonzero(as_tuple=False)
        if len(idx) > 0:
            bc[idx[:, 0], idx[:, 1]] = 1
    return bc.view(-1).bool()


def main():
    # 造一个典型布局：4 个实例，其余是背景（0）
    m = torch.zeros(ATTN_DIM, ATTN_DIM)
    for k, (r, c) in enumerate([(4, 4), (4, 20), (20, 4), (20, 20)], start=1):
        m[r:r + 6, c:c + 6] = k
    bg_frac = (m == 0).float().mean().item()
    print(f"desired_mask: {int(m.max())} 个实例，背景占 {bg_frac*100:.1f}%")

    for start, tag in ((0, "原码 range(0, ...)"), (1, "若改成 range(1, ...)")):
        flat = blob_coordinates(m, start)
        # 复刻 56-61 行：对每个背景 query 位置，把选中的 key 列清零
        attn = torch.ones(1, ATTN_DIM, ATTN_DIM, ATTN_DIM ** 2)
        for idx in (m == 0).nonzero(as_tuple=False):
            attn[:, idx[0], idx[1], flat] = 0
        rows = attn[0].reshape(ATTN_DIM ** 2, -1)
        dead = (rows.sum(1) == 0).float().mean().item()
        print(f"\n{tag}")
        print(f"  blob_coordinates 置 1 的比例 : {flat.float().mean().item():.3f}")
        print(f"  被清零的注意力元素比例       : {(attn == 0).float().mean().item():.3f}")
        print(f"  整行被清零的 query 比例      : {dead:.3f}")

    print("\n判读：原码下 blob_coordinates 恒为全 1，背景 query 的整行注意力被清零。")
    print("      这是 step 0–10、所有 up-block 的 32×32 自注意力上生效的（见 33-38 行的门限）。")
    print("      是不是有意为之要跑了才知道；但它的效果与注释暗示的'背景不看前景'不同。")


if __name__ == "__main__":
    main()
