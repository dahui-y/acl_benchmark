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
import shutil
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


def _align_dirs(tar, sty, cnt):
    """从 tar 的文件名 {style}__{content}.png 反推对齐目录。

    eval_artfid 按 sorted() 逐一配对且断言三边张数相等，所以要造两个与 tar
    同名的软链目录。**不需要 manifest** —— 配对关系已经编码在文件名里，
    这比再维护一份索引更不容易错。
    """
    pairs = []
    skipped = 0
    for f in sorted(tar.glob("*.png")):
        # matrix.py 把 matrix_*.png 写进了同一个目录。不含 '__' 的不是输出图，
        # 跳过而不是退出 —— 第一版直接 sys.exit，把两个基线挡在门外。
        if "__" not in f.stem:
            skipped += 1
            continue
        s_, c_ = f.stem.split("__", 1)
        pairs.append((f.name, s_, c_))
    if skipped:
        print(f"   跳过 {skipped} 个不含 '__' 的文件（热图等）")
    if not pairs:
        sys.exit(f"!! {tar} 里没有 png")
    sx, cx = tar.parent / f"{tar.name}_sty_x", tar.parent / f"{tar.name}_cnt_x"
    for p_ in (sx, cx):
        if p_.exists():
            shutil.rmtree(p_)
        p_.mkdir(parents=True)

    def _find(d, stem):
        for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
            if (d / f"{stem}{ext}").exists():
                return (d / f"{stem}{ext}").resolve()
        sys.exit(f"!! 在 {d} 里找不到 {stem}.*")

    for name, s_, c_ in pairs:
        (sx / name).symlink_to(_find(sty, s_))
        (cx / name).symlink_to(_find(cnt, c_))
    print(f"对齐 {len(pairs)} 组 → {sx.name} / {cx.name}")
    return tar, sx, cx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    # 任意目录模式：styleid_batch.py 的输出不在 protocol/seedN 布局里。
    # tar 的文件名是 {style}__{content}.png，对齐目录可以从文件名直接反推，
    # 不需要 manifest。
    ap.add_argument("--tar", default=None, help="stylized 目录")
    ap.add_argument("--sty", default=None, help="style 源图目录")
    ap.add_argument("--cnt", default=None, help="content 源图目录")
    ap.add_argument("--mode", default="art_fid_inf",
                    choices=["art_fid", "art_fid_inf"])
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()

    if a.tar:
        tar, sx, cx = _align_dirs(Path(a.tar), Path(a.sty), Path(a.cnt))
        tag = Path(a.tar).name
    else:
        if a.seed is None:
            sys.exit("!! 给 --seed 或给 --tar/--sty/--cnt")
        d = OUT / f"seed{a.seed}"
        tar, sx, cx = d / "tar", d / "sty_x", d / "cnt_x"
        for p_ in (sx, cx, tar):
            if not p_.exists():
                sys.exit(f"!! 缺 {p_} —— 先跑 protocol.py --align --seed {a.seed}")
        tag = f"seed{a.seed}"
    n = {p_.name: len(list(p_.glob("*.png"))) for p_ in (sx, cx, tar)}
    if len(set(n.values())) != 1:
        sys.exit(f"!! 三边张数不等 {n} —— eval_artfid 会断言失败")
    print(f"{tag}: 三边各 {n[tar.name]} 张")

    patch_sqrtm()
    sys.path.insert(0, str(EVAL))
    os.chdir(EVAL)                      # utils.download 用相对路径
    sys.argv = ["eval_artfid.py",
                "--sty", str(sx), "--cnt", str(cx),
                "--tar", str(tar), "--mode", a.mode,
                "--batch_size", str(a.batch_size),
                "--num_workers", str(a.num_workers)]
    runpy.run_path(str(EVAL / "eval_artfid.py"), run_name="__main__")


if __name__ == "__main__":
    main()
