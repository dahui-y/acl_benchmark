#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
装完环境先跑这个。**不占 GPU、不生成任何图**，只回答一句话：能不能开跑。

    为什么值得单独写：这条线上要凑齐的东西有四样在国内都可能卡住
    （spacy 的 460MB wheel、Google Drive 上的 ReLayout 权重、
      torch.hub 要连 GitHub、SDXL 权重），而其中三样**不是在启动时报错，
    是在跑到第 N 张图时才炸**——`torch.hub.load` 写在 relayout 的补物体分支里，
    只有当 DBSCAN 数少了才会被调到。等跑了半小时才发现，就白烧半小时。

用法：
    python count_probe/env_check.py
"""

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIC = REPO / "help_code" / "make-it-count"

OK, WARN, BAD = "  ok ", " 注意 ", " 失败 "
_bad = []


def say(tag, what, detail=""):
    print(f"[{tag}] {what}" + (f"  {detail}" if detail else ""))
    if tag is BAD:
        _bad.append(what)


DIST = {"sklearn": "scikit-learn", "skimage": "scikit-image", "cv2": "opencv-python",
        "spacy_transformers": "spacy-transformers", "huggingface_hub": "huggingface-hub"}


def ver(mod, want=None):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        say(BAD, f"import {mod}", str(e).split("\n")[0][:90])
        return None
    # 先问包元数据，再退回 __version__ —— 有些包压根没有 __version__
    # （spacy-transformers、inflect），click 则把它标成了将要移除的弃用属性。
    try:
        from importlib.metadata import version as _mv
        v = _mv(DIST.get(mod, mod))
    except Exception:
        v = getattr(m, "__version__", "?")
    v = str(v).split("+")[0]         # 剥掉 +cu121 这类本地版本后缀再比
    if want and v != want:
        say(WARN, f"{mod} {v}", f"（他们钉的是 {want}；不等不一定错，但复现出偏差先想这里）")
    else:
        say(OK, f"{mod} {v}")
    return v


def _tuple(v):
    out = []
    for p in str(v).split("."):
        n = "".join(c for c in p if c.isdigit())
        out.append(int(n) if n else 0)
    return tuple(out)


def main():
    # ★ 第一行就把环境打出来。这条线上已经发生过"命令敲在另一个环境里"的事，
    #   而那种错的表现是别处报依赖冲突，不是这里报错，很难顺藤摸回来。
    print(f"环境   {sys.prefix}")
    print(f"python {sys.version.split()[0]}   （建议 3.10；scikit-image 0.23 要 >=3.10）")
    if Path(sys.prefix).name not in ("countgen",):
        print(f"       ⚠️ 环境名不是 countgen —— 确认没敲错 conda activate\n")
    else:
        print()

    print("── 版本 " + "─" * 48)
    tv = ver("torch", "2.1.2")
    ver("torchvision", "0.16.2")
    ver("diffusers", "0.25.0")
    trf = ver("transformers", "4.29.2")
    tok = ver("tokenizers", "0.13.3")
    ver("huggingface_hub", "0.20.1")
    ver("numpy", "1.23.3")
    ver("scipy")
    ver("sklearn")
    ver("skimage")
    ver("cv2")
    sp = ver("spacy", "3.5.2")          # 这一句会连带拉起 typer/click（见下）
    ver("spacy_transformers", "1.2.5")
    ck_v = ver("click")
    ver("inflect")
    ver("ultralytics")
    ver("supervision")

    print("\n── 约束核对 " + "─" * 44)
    # 这两条是 make-it-count 那份 requirements.txt 内部就自相矛盾的地方
    if trf:
        if _tuple(trf) >= (4, 31):
            say(BAD, "transformers 版本过高",
                "spacy-transformers 1.2.5 要求 <4.31.0；requirements.txt 里那条 git 装法会踩这个")
        elif _tuple(trf) < (4, 25, 1):
            say(BAD, "transformers 版本过低", "diffusers 0.25.0 要求 >=4.25.1")
        else:
            say(OK, "transformers 同时满足 spacy-transformers(<4.31) 与 diffusers(>=4.25.1)")
    if tok and _tuple(tok) >= (0, 14):
        say(BAD, "tokenizers >=0.14", f"transformers {trf} 要求 <0.14")
    try:
        from huggingface_hub import cached_download  # noqa: F401
        say(OK, "huggingface_hub.cached_download 还在")
    except Exception:
        say(WARN, "huggingface_hub 里没有 cached_download",
            "若 transformers 报 ImportError，降到 0.19.4")

    # spacy 3.5.2 → typer 0.7.0 → click。typer 0.7 只写了 click<9，pip 会装 8.4.x，
    # 而 click 8.2 之后的参数解析改动 typer 0.7 没适配。spacy/__init__.py 里有
    # `from .cli.info import info`，所以这个冲突连 import spacy 都可能带塌。
    if ck_v and _tuple(ck_v) >= (8, 2):
        say(BAD, f"click {ck_v} 太新", "spacy 3.5.2 的 typer 0.7.0 适配不了；装 click==8.1.7")
    elif ck_v:
        say(OK, f"click {ck_v} 与 typer 0.7.0 相容")

    # torch↔numpy 的桥。版本号对不代表桥是通的：torch 2.1.2 的 C 扩展按 NumPy 1.x
    # 的 ABI 编译，装上 numpy 2.x 时 import torch 照样成功、版本号照样正确，
    # 但 from_numpy 会炸。这条要真的调一次才算数。
    if tv:
        try:
            import numpy as _np
            import torch as _t
            _t.from_numpy(_np.zeros(2, dtype="float32")).numpy()
            say(OK, "torch ↔ numpy 双向转换可用")
        except Exception as e:
            say(BAD, "torch 的 numpy 桥是坏的",
                f"{str(e)[:70]}；多半是 numpy 2.x，装 numpy==1.23.3")

    print("\n── 设备 " + "─" * 48)
    if tv:
        import torch
        if not torch.cuda.is_available():
            say(BAD, "看不到 CUDA 设备")
        else:
            g = torch.cuda.get_device_properties(0)
            gb = g.total_memory / 1024 ** 3
            say(OK if gb >= 20 else WARN, f"{g.name}  {gb:.1f} GB",
                "" if gb >= 20 else "（SDXL 1024² + 带梯度的 refinement，低于 20G 很可能 OOM）")

    print("\n── 资产 " + "─" * 48)
    # 权重不必放在仓库里 —— 用 RELAYOUT_CKPT 指到有空间的盘（如 $SD_OUT 同级）
    ck = Path(os.environ.get(
        "RELAYOUT_CKPT",
        MIC / "pipeline/mask_extraction/relayout_weights/relayout_checkpoint.pth"))
    if ck.exists():
        say(OK, "ReLayout 权重", f"{ck.stat().st_size/1024**2:.0f} MB")
    else:
        say(BAD, "缺 ReLayout 权重",
            f"找的是 {ck}（Google Drive，见 make-it-count README；HF 上无镜像）。"
            f"放别处就 export RELAYOUT_CKPT=/那个/路径")

    y = next((p for p in (Path("yolov9e.pt"), REPO / "yolov9e.pt", MIC / "yolov9e.pt")
              if p.exists()), None)
    if y:
        say(OK, "yolov9e.pt", f"{y}  {y.stat().st_size/1024**2:.0f} MB")
    else:
        say(WARN, "没找到 yolov9e.pt",
            "评测那步才用；ultralytics 会自动下载，服务器不通就先手动 wget")

    try:
        importlib.import_module("en_core_web_trf")
        say(OK, "spacy 模型 en_core_web_trf 已安装")
    except Exception:
        say(BAD, "缺 en_core_web_trf",
            "直接装 wheel（别用 spacy CLI，见 ENV.md）：pip install https://github.com/"
            "explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/"
            "en_core_web_trf-3.5.0-py3-none-any.whl")

    sdxl = os.environ.get("SDXL_PATH")
    if not sdxl:
        say(WARN, "没设 SDXL_PATH", "会去 HF 拉 stabilityai/stable-diffusion-xl-base-1.0")
    elif Path(sdxl).exists():
        has_fp16 = any(Path(sdxl).rglob("*fp16*"))
        say(OK, f"SDXL_PATH={sdxl}",
            "有 fp16 分支" if has_fp16 else "→ 没看到 fp16 文件，跑批要加 --variant \"\"")
    else:
        say(BAD, "SDXL_PATH 指向的目录不存在", sdxl)

    out = os.environ.get("SD_OUT")
    if not out:
        say(BAD, "没设 SD_OUT", "服务器上唯一有空间可写的路径，必须设")
    else:
        p = Path(out)
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".w"
            t.write_text("x")
            t.unlink()
            say(OK, f"SD_OUT={out} 可写")
        except Exception as e:
            say(BAD, "SD_OUT 不可写", str(e)[:80])

    if tv:
        import torch
        hub = Path(torch.hub.get_dir()) / "mateuszbuda_brain-segmentation-pytorch_master"
        if hub.exists():
            say(OK, "torch.hub 已缓存 brain-segmentation-pytorch")
        else:
            say(WARN, "torch.hub 没缓存 brain-segmentation-pytorch",
                "relayout 补物体那条分支会去连 GitHub；连不上就得先把缓存拷过来")

    print("\n" + "─" * 56)
    if _bad:
        print(f"有 {len(_bad)} 项过不去：" + "；".join(_bad))
        sys.exit(1)
    print("可以开跑：python count_probe/countgen_batch.py --limit 5 --out $SD_OUT/count/cocoount")


if __name__ == "__main__":
    main()
