#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 make-it-count 的输出目录改名分臂，改成它自己的评测脚本能吃的格式。

    为什么需要这一步：**它自己的两个脚本对不上名字。**
      · `pipeline/run_countgen.py:137` 存的是 `{obj}_num={N}_seed={S}.png`
      · `evaluation_script.py:61-70` 要求 `{count}__{class}__{...}.png`
    而且 `run_countgen.py:138` 把 `*_vanilla.png`（原版 SDXL）写进**同一个目录**，
    直接拿去评测会把两个臂混在一起数。

    这里只建软链、不复制像素，产出：

        <out>/vanilla/{N}__{coco_class}__{stem}.png     ← 原版 SDXL 臂
        <out>/countgen/{N}__{coco_class}__{stem}.png    ← CountGen 臂
        <out>/index.json                                ← 逐张的真值与元信息

    这样两条路都能走：
      · 我们自己的 `yolo_eval.py`（读 index.json，多给分档与拆解）
      · 它们的 `evaluation_script.py` **原样不改**（读文件名）——
        这一条是为了「我们没动评测」这句话站得住，和风格线上
        `run_artfid.py` 只补 scipy 调用约定、不碰度量是同一个道理。

    ⚠️ 类名必须是 **YOLO 认识的 COCO 类名**，不是数据集里的简写：
       `ball → sports ball`、`glove → baseball glove`、`phone → cell phone`
       （映射取自 `dataset/create_data_CoCoCount.py` 的 `coco_object_name_dict`）。
       不映射的话 `evaluation_script.py:28` 的类名比对会全 False，
       准确率会假到 0 —— 这是个静默失败，必须在这里挡掉。

用法：
    python count_probe/make_arms.py --src $SD_OUT/count/cocoount \
        --out $SD_OUT/count/cocoount_arms
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIC = REPO / "help_code" / "make-it-count"

# 【一手】dataset/create_data_CoCoCount.py 的 coco_object_name_dict
SHORT2COCO = {"ball": "sports ball", "glove": "baseball glove", "phone": "cell phone"}

# COCO-80 标准类名，顺序即 ultralytics 的 model.names。
# yolo_eval.py 会拿 model.names 断言这份表，所以这里不是"假设"，是"待核对项"。
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def coco_name(obj):
    return SHORT2COCO.get(obj, obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="官方布局的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=str(MIC / "dataset" / "CoCoCount.json"),
                    help="用来取每题的 N / object / prompt（metadata.json 缺字段时的兜底）")
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out).resolve()

    data = {f"{d['object']}_num={d['int_number']}_seed={d['seed']}": d
            for d in json.load(open(a.dataset))}
    log = {}
    lp = src / "counter_log.jsonl"
    if lp.exists():
        for line in lp.open():
            if line.strip():
                r = json.loads(line)
                log[r["id"]] = r
        print(f"读到 counter_log.jsonl：{len(log)} 条（含 DBSCAN 计数器读数）")
    else:
        print("⚠️ 没有 counter_log.jsonl —— 只能出准确率，出不了「上限卡在哪」的拆解。"
              "\n   那份日志只有 count_probe/countgen_batch.py 会写。")

    for d in ("vanilla", "countgen"):
        p = out / d
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    index, missing_cg, non_coco, unknown = [], [], [], []
    stems = sorted({p.name[:-len("_vanilla.png")]
                    for p in src.glob("*_vanilla.png")})
    if not stems:
        raise SystemExit(f"!! {src} 里没有 *_vanilla.png")

    for stem in stems:
        item = data.get(stem)
        if item is None:
            unknown.append(stem)
            continue
        N, obj = item["int_number"], item["object"]
        cname = coco_name(obj)
        if cname not in COCO80:
            non_coco.append((stem, cname))
            continue
        fn = f"{N}__{cname}__{stem}.png"
        (out / "vanilla" / fn).symlink_to(src / f"{stem}_vanilla.png")
        cg = src / f"{stem}.png"
        if cg.exists():
            (out / "countgen" / fn).symlink_to(cg)
        else:
            missing_cg.append(stem)
        r = log.get(stem, {})
        index.append({"file": fn, "stem": stem, "N": N, "object": obj,
                      "coco_class": cname, "prompt": item["prompt"],
                      "seed": item["seed"],
                      "has_countgen": cg.exists(),
                      "n_dbscan": r.get("n_dbscan"),
                      "obj_num_match": r.get("obj_num_match"),
                      "skipped_by_official": r.get("skipped_by_official",
                                                   N > 9)})
    json.dump(index, open(out / "index.json", "w"), ensure_ascii=False, indent=2)

    nv = len(list((out / "vanilla").glob("*.png")))
    nc = len(list((out / "countgen").glob("*.png")))
    print(f"\nvanilla 臂 {nv} 张 → {out/'vanilla'}")
    print(f"countgen 臂 {nc} 张 → {out/'countgen'}")
    if missing_cg:
        over9 = [s for s in missing_cg if data[s]["int_number"] > 9]
        vo = [s for s in missing_cg if log.get(s, {}).get("vanilla_only")]
        print(f"⚠️ {len(missing_cg)} 题没有 CountGen 图，其中 {len(over9)} 题是 N>9"
              f"（官方 run_countgen.py:104 主动跳过，不是失败）")
        if vo:
            print(f"   {len(vo)} 题是 --vanilla-only 跑的（还没拿到 ReLayout 权重）"
                  f"—— 拿到权重后不加该开关重跑一遍即可补上")
        rest = [s for s in missing_cg if data[s]["int_number"] <= 9
                and not log.get(s, {}).get("vanilla_only")]
        if rest:
            print(f"   另有 {len(rest)} 题 N≤9 却缺图 —— 这是**真的没跑成**，例：{rest[:3]}")
    if non_coco:
        print(f"⚠️ {len(non_coco)} 题的类别不在 COCO-80 里，YOLO 评不了，已剔除："
              f"{sorted({c for _, c in non_coco})}")
    if unknown:
        print(f"⚠️ {len(unknown)} 张图在数据集里找不到对应题目（换过数据集？）例：{unknown[:3]}")

    print(f"\n下一步：")
    print(f"  python {REPO}/count_probe/yolo_eval.py --arms {out}")
    print(f"\n（对照用：它们自己的脚本原样跑，应当得到同样的准确率）")
    print(f"  cd {MIC} && python evaluation_script.py "
          f"--images_dir '{out}/countgen' --output_dir <某处>")


if __name__ == "__main__":
    main()
