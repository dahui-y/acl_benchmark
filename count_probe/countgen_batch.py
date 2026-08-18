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

os.environ.setdefault("MPLBACKEND", "Agg")      # 无头环境；他们的代码会画图

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
    pipe = SelfCountingSDXLPipeline.from_pretrained(
        cfg["model"]["sdxl_path"], use_safetensors=True,
        torch_dtype=torch.float16, variant=cfg["model"].get("variant") or None,
        local_files_only=cfg["model"]["local_only"],
        use_onnx=False)
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
    ap.add_argument("--skip-over9", action="store_true",
                    help="连 vanilla 都不跑 N>9（完全等同官方行为）")
    ap.add_argument("--no-masks", action="store_true", help="不存 mask 可视化")
    ap.add_argument("--relayout-ckpt", default=os.environ.get("RELAYOUT_CKPT"),
                    help="ReLayout 权重路径。默认用 pipeline_config.yaml 里的相对路径"
                         "（仓库内），但那是个 GB 级文件，不该塞进仓库——"
                         "指到有空间的盘即可，或设环境变量 RELAYOUT_CKPT")
    ap.add_argument("--vanilla-only", action="store_true",
                    help="只跑原版 SDXL + DBSCAN 计数，不做任何修正。"
                         "不需要 ReLayout 权重，一次前向无梯度，快三四倍。"
                         "拿到的是表三四个格子里的前三个（baseline、计数器一致率、"
                         "「计数器说对但实际错」这一桶），只差修正成功率。")
    a = ap.parse_args()

    _patch_caches()
    sys.path.insert(0, str(MIC))
    os.chdir(MIC)                       # 他们的 config 用相对路径

    import torch
    import yaml
    from diffusers.utils.torch_utils import randn_tensor
    from tqdm import tqdm

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

    counter = _patch_counter_probe()
    pipe = _load_pipeline(cfg)
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
            if not a.no_masks:
                show_mask_list([vanilla_masks, correct_mask, object_masks],
                               titles=[f"Vanilla: {n_used}",
                                       f"Corrected: {int(correct_mask.max())}",
                                       f"Postprocess: {int(object_masks.max())}"],
                               save_path=str(out / f"{img_id}_masks.png"))
            # ★ 与官方一致：计数器认为已经对了，就直接输出原版图，不做任何干预
            image = vanilla_img if match else run_counting_pipeline_corrected_masks(
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
               "sec": round(t_count, 2)}
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
