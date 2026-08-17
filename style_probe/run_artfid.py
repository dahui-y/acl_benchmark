#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跑 StyleSSP 的 eval_artfid.py，但**不改它一行**。

    2026-08-17：直接跑报
        TypeError: sqrtm() got an unexpected keyword argument 'disp'
    SciPy 1.15 弃用、之后移除了 linalg.sqrtm 的 disp 参数，而 ArtFID 的
    compute_frechet_distance 是按老 API 写的：
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    这是**依赖的 API 变更，不是尺子的问题**。但一旦我们去改
    help_code/StyleSSP/evaluation/ 里的文件，「你们是不是动了评测」
    就变成一个说不清的问题 —— 而这条线上我们本来就已经因为
    「测的不是自己以为的东西」死过好几个方向。

    所以补丁打在**调用约定**上，不打在度量上：把 scipy.linalg.sqrtm
    包一层，让它重新接受 disp=False 并返回 (根, 误差估计)，语义与老版一致
    （老版的 errest 是 ‖X·X−A‖_F / ‖A‖_F）。度量代码原样运行。

    对齐目录由 protocol.py --align 造好（eval_artfid 按 sorted() 逐一配对，
    且断言三边张数相等）。

用法：
    python style_probe/run_artfid.py --seed 0
    python style_probe/run_artfid.py --seed 0 --mode art_fid   # 不外推
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "protocol"
EVAL = REPO / "help_code" / "StyleSSP" / "evaluation"


def patch_sqrtm():
    """让 scipy.linalg.sqrtm 重新接受 disp=（新版 SciPy 删了它）。"""
    import scipy.linalg as sla
    import inspect
    if "disp" in inspect.signature(sla.sqrtm).parameters:
        print("scipy.linalg.sqrtm 仍带 disp，无需补丁")
        return
    orig = sla.sqrtm

    def sqrtm(A, disp=True, blocksize=64):
        X = orig(A)
        if disp:
            return X
        # 老版 disp=False 时返回的第二项是相对残差估计
        A = np.asarray(A)
        err = np.linalg.norm(X.dot(X) - A, "fro") / max(
            np.linalg.norm(A, "fro"), np.finfo(float).eps)
        return X, err

    sla.sqrtm = sqrtm
    print(f"已补 scipy.linalg.sqrtm 的 disp 参数（scipy 新版移除；"
          f"只改调用约定，不动度量）")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="art_fid_inf",
                    choices=["art_fid", "art_fid_inf"])
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()

    d = OUT / f"seed{a.seed}"
    for nm in ("sty_x", "cnt_x", "tar"):
        if not (d / nm).exists():
            sys.exit(f"!! 缺 {d/nm} —— 先跑 protocol.py --align --seed {a.seed}")
    n = {nm: len(list((d / nm).glob("*.png"))) for nm in ("sty_x", "cnt_x", "tar")}
    if len(set(n.values())) != 1:
        sys.exit(f"!! 三边张数不等 {n} —— eval_artfid 会断言失败，重跑 --align")
    print(f"seed {a.seed}: 三边各 {n['tar']} 张")

    patch_sqrtm()
    sys.path.insert(0, str(EVAL))
    os.chdir(EVAL)                      # utils.download 用相对路径
    sys.argv = ["eval_artfid.py",
                "--sty", str(d / "sty_x"), "--cnt", str(d / "cnt_x"),
                "--tar", str(d / "tar"), "--mode", a.mode,
                "--batch_size", str(a.batch_size),
                "--num_workers", str(a.num_workers)]
    runpy.run_path(str(EVAL / "eval_artfid.py"), run_name="__main__")


if __name__ == "__main__":
    main()
