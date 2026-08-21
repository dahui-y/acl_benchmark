#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase C：**纯删除**修正器原型。countgen 环境跑（diffusers 0.25 + ultralytics 都在）。

    机制：不再对 latent 做损失梯度下降（那条路五次否定、评测集净 0），改为
    「检测 → 挑出多余实例 → 该区域重去噪」——让模型自己重新生成那块区域，
    而不是把它往某个方向推。

      1. YOLOv8x 数目标类（环内计数器；立项判据 P(v9e=N|v8x=N)=85.7% 已过 80% 门）
      2. 多出 k 个 → 删分数最低的 k 个框（并列取面积小者；与 relayout 的
         「删最小」同精神，但按检测置信度——「最不像真实实例」优先）
      3. 框膨胀 → 合成一张 mask → diffusers 的 SDXLInpaintPipeline 用 **base
         SDXL 权重**做遮罩重去噪（4 通道 UNet 走逐步 latent 混合，即 RePaint 式；
         零新权重、零训练部件、无梯度图）
      4. 像素级贴回：mask 外用原图原像素（羽化过渡）——不让 VAE 往返污染
         未编辑区域，这是画质红线（与 CountGen 逐张持平）的实现保障
      5. 重检测；仍多 → 再来一轮（上限 --rounds）

    删除的重去噪 prompt（预登记的两个变体，只许在新调参集上比这一次）：
      ctx  从原 prompt 剥掉数量短语，只留场景（"A photo of seven cell phones
           on the road" → "A photo of the road"；无场景后缀则退化为 bg）
      bg   固定 "background"
    两者 negative_prompt 都 = 目标类名（CFG 主动推离该类，防止洞里又长出来）。

    预登记的标定预算（DESIGN.md 同款，超出即调参违纪）：
      strength {0.8, 1.0} × prompt {ctx, bg} = 4 个配置，只跑新调参集，选定后
      评测集只跑一次。其余旋钮（dilate/feather/steps/guidance）用默认值，不扫。

    输出 = 官方布局（{stem}.png / {stem}_vanilla.png / metadata.json /
    counter_log.jsonl），下游 make_arms / yolo_eval / summary 原样可用；
    counter_log 的 n_dbscan 字段写 v8x 的初检数 —— 对下游的语义就是
    「环内计数器读数」，decompose 系工具照常出表。

用法（先在新调参集的 vanilla 上）：
    python count_probe/correct_batch.py \\
        --src $SD_OUT/count/tune2_van --dataset $SD_OUT/count/tune2.json \\
        --out $SD_OUT/count/tune2_del_s10ctx --strength 1.0 --prompt-mode ctx
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from make_arms import coco_name  # noqa: E402  短名→COCO 类名（ball→sports ball 等）


def pick_deletions(boxes, scores, k):
    """删「最不像真实实例」的 k 个：分数最低优先，并列取面积小者。→ 索引。"""
    def area(b):
        return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    order = sorted(range(len(boxes)), key=lambda i: (scores[i], area(boxes[i])))
    return order[:k]


def build_mask(del_boxes, keep_boxes, size, dilate):
    """删除区域 = 膨胀后的待删框之并，再**挖掉要保留的框**（不膨胀）。

    冒烟实测的缺陷：apple N=7、初检 15，删 8 个框后只剩 4 —— 多掉了 3 个。
    密集场景里框彼此相邻，dilate=24 让待删框的 mask 盖住了邻居，重去噪把
    邻居一起抹掉。这不是超参没调好，是 mask 构造没兑现设计意图（「删掉
    恰好 k 个」），属实现缺陷，修它不计入 DESIGN.md 的标定预算。

    挖掉时不给保留框加膨胀：保留框自身的边缘允许被重画（否则删除区域和
    保留区域之间会留一圈无人管的缝），但主体像素受保护。
    """
    from PIL import Image, ImageDraw
    W, H = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x1, y1, x2, y2 in del_boxes:
        d.rectangle([max(x1 - dilate, 0), max(y1 - dilate, 0),
                     min(x2 + dilate, W), min(y2 + dilate, H)], fill=255)
    for x1, y1, x2, y2 in keep_boxes:
        d.rectangle([max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)], fill=0)
    return m


def paste_back(orig, edited, mask, feather):
    """mask 外保留原图原像素（羽化过渡）。VAE 往返不许碰未编辑区。"""
    from PIL import ImageFilter
    import numpy as np
    m = mask.filter(ImageFilter.GaussianBlur(feather))
    a = (np.asarray(m, dtype=np.float32) / 255.0)[..., None]
    out = (np.asarray(edited, dtype=np.float32) * a
           + np.asarray(orig, dtype=np.float32) * (1 - a))
    from PIL import Image
    return Image.fromarray(out.clip(0, 255).astype("uint8"))


def ctx_prompt(prompt, cls):
    """剥掉数量短语，只留场景。找不到场景后缀就退回 'background'。"""
    for kw in (" on ", " in ", " at ", " near "):
        if kw in prompt:
            return "A photo of " + prompt.split(kw, 1)[1], "ctx"
    return "background", "bg(fallback)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="含 {stem}_vanilla.png 的跑批目录")
    ap.add_argument("--dataset", required=True, help="该批的题目 json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sdxl", default=os.environ.get(
        "SDXL_PATH", "stabilityai/stable-diffusion-xl-base-1.0"))
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--weights", default="yolov8x.pt", help="环内计数器")
    # ---- 预登记网格：只有这两个旋钮许动，且只在新调参集上 ----
    ap.add_argument("--strength", type=float, default=1.0, choices=[0.8, 1.0])
    ap.add_argument("--prompt-mode", default="ctx", choices=["ctx", "bg"])
    # ---- 其余全是定死的默认，不扫 ----
    ap.add_argument("--dilate", type=int, default=24)
    ap.add_argument("--feather", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from ultralytics import YOLO
    from diffusers import StableDiffusionXLInpaintPipeline

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(a.dataset))
    if a.limit:
        data = data[:a.limit]

    det = YOLO(a.weights)

    def count_boxes(img, cls):
        r = det(img, verbose=False)[0]
        sel = [(b, s) for b, s, c in zip(r.boxes.xyxy.tolist(),
                                         r.boxes.conf.tolist(),
                                         r.boxes.cls.tolist())
               if r.names[int(c)] == cls]
        return [b for b, _ in sel], [s for _, s in sel]

    kw = dict(torch_dtype=torch.float16, use_safetensors=True,
              local_files_only=a.local_files_only)
    if a.variant:
        kw["variant"] = a.variant
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(a.sdxl, **kw)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"★ 纯删除修正器：strength={a.strength} prompt={a.prompt_mode} "
          f"dilate={a.dilate} feather={a.feather} steps={a.steps} "
          f"guidance={a.guidance} rounds={a.rounds}")

    meta, log_p = [], out / "counter_log.jsonl"
    done = {json.loads(l)["id"] for l in log_p.open()} if log_p.exists() else set()
    t_all = time.time()
    for item in data:
        prompt, seed = item["prompt"], item["seed"]
        N, obj = item["int_number"], item["object"]
        stem = f"{obj}_num={N}_seed={seed}"
        if stem in done:
            continue
        van_p = src / f"{stem}_vanilla.png"
        if not van_p.exists():
            print(f"  !! 缺 {van_p.name}，跳过")
            continue
        cls = coco_name(obj)
        t0 = time.time()
        img = Image.open(van_p).convert("RGB")
        boxes, scores = count_boxes(img, cls)
        n0 = len(boxes)
        trail, deleted, mask_frac, want = [n0], 0, [], []
        cur = img
        if N <= 9 and n0 > N:
            for rnd in range(a.rounds):
                b_now, s_now = (boxes, scores) if rnd == 0 else count_boxes(cur, cls)
                k = len(b_now) - N
                if k <= 0:
                    break
                idx = set(pick_deletions(b_now, s_now, k))
                dele = [b_now[i] for i in idx]
                keep = [b_now[i] for i in range(len(b_now)) if i not in idx]
                mask = build_mask(dele, keep, cur.size, a.dilate)
                pos, tag = (ctx_prompt(prompt, cls) if a.prompt_mode == "ctx"
                            else ("background", "bg"))
                g = torch.Generator("cuda").manual_seed(seed + 1000 * (rnd + 1))
                edited = pipe(prompt=pos, negative_prompt=cls, image=cur,
                              mask_image=mask, strength=a.strength,
                              num_inference_steps=a.steps,
                              guidance_scale=a.guidance, generator=g,
                              height=cur.size[1], width=cur.size[0]).images[0]
                cur = paste_back(cur, edited, mask, a.feather)
                deleted += len(dele)
                import numpy as _np
                mask_frac.append(round(float((_np.asarray(mask) > 127).mean()), 4))
                want.append(k)
                trail.append(len(count_boxes(cur, cls)[0]))
                if trail[-1] <= N:
                    break
        cur.save(out / f"{stem}.png")
        shutil.copy2(van_p, out / f"{stem}_vanilla.png")
        rec = {"id": stem, "prompt": prompt, "seed": seed, "obj_class": obj,
               "requiered_object_num": N,
               "n_dbscan": n0,                    # 语义：环内计数器读数（v8x）
               "n_used": n0,
               "zero_cluster_fallback": False,
               "obj_num_match": bool(n0 == N),
               "skipped_by_official": bool(N > 9),
               "vanilla_only": False, "has_countgen_img": True,
               "sec": round(time.time() - t0, 2),
               "corrector": {"kind": "delete-only", "counter": a.weights,
                             "strength": a.strength, "prompt_mode": a.prompt_mode,
                             "n_trail": trail, "n_deleted_boxes": deleted,
                             # 意图 vs 实际：want[i] 是该轮想删几个，
                             # trail[i]-trail[i+1] 是实际掉了几个。两者背离
                             # 就是 mask 吃到邻居（冒烟里 15→4 而 want=8）。
                             "want": want, "mask_frac": mask_frac}}
        meta.append({k: rec[k] for k in
                     ("id", "prompt", "seed", "obj_class", "requiered_object_num")})
        json.dump(meta, open(out / "metadata.json", "w"), indent=4)
        with log_p.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(list(out.glob("*_vanilla.png")))
    print(f"\n完成。{n} 题，用时 {(time.time()-t_all)/60:.1f} 分钟 → {out}")
    print(f"下一步：\n  python count_probe/make_arms.py --src {out} "
          f"--out {out.parent/(out.name+'_arms')} --dataset {a.dataset}\n"
          f"  python count_probe/yolo_eval.py --arms {out.parent/(out.name+'_arms')}")


if __name__ == "__main__":
    main()
