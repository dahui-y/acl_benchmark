#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
选项 C：跑在位者（StyleID），建立**我们自己的** ArtFID 基线 + 量出抽样噪声底。

────────────────────────────────────────────────────────────────────────
先订正 C 的定义（2026-08-16，摸完数据后）
────────────────────────────────────────────────────────────────────────

  我原本说 C 是「复现 StyleID 的 ArtFID 28.973」。**做不到，而且不该那么定。**

  · 28.973 是 **StyleGallery 自己那一次随机抽样**上测出的 StyleID。
  · StyleID 和 StyleSSP 的仓库**都不发** `data/cnt` 和 `data/sty`，
    两家 README 都只写「从 MS-COCO 和 WikiArt 随机选 40 style + 20 content」。
  · 所以每篇论文的 40×20 都是各自的一次抽样，绝对值本来就不可精确复现。
  · 再加我们的 WikiArt 副本是 **256×256 预缩放**的，而协议要求从原图
    **center-crop 到 512²** —— 绝对值更不可比。

  **C 因此重新定义为：**
    ①  在**我们自己的一次有记录的抽样**上跑 StyleID，得到我们的基线数
    ②  **换 seed 重抽几次，量出抽样之间的方差** ← 这才是「改善要多大才算数」
    ③  看 StyleID 在哪些 (style, content) 组合上崩 —— playbook 第 3 步

  ②是这个脚本存在的主要理由。没有它，后面任何 A/B 差值都读不出意义。
  （同 `std_table.py` 的噪声底纪律。）

  **所有对外报的数必须写明：自抽样，与发表值不可直接比。**

────────────────────────────────────────────────────────────────────────
协议（照 StyleSSP / StyleID，能对齐的都对齐）
────────────────────────────────────────────────────────────────────────

  · 40 style（WikiArt）× 20 content（MS-COCO）= 800 张
  · 512×512
  · ArtFID = (1+LPIPS)·(1+FID)，用 StyleSSP 仓库现成的 `evaluation/eval_artfid.py`
    —— **不自己写指标**（std_table.py 的教训：尺子必须与对手同一把）
  · 方法 = StyleID 式 K/V 注入，直接复用 subspace_look 的 `full` 路径
    （那一列已经在 8 种风格上出过干净结果）

用法：
    source scalediff_probe/env.sh
    python style_probe/protocol.py --check              # 先摸数据，零 GPU
    python style_probe/protocol.py --draw --seed 0      # 抽一次 40×20，写 manifest
    python style_probe/protocol.py --gen  --seed 0      # 800 张，约 1–2 h
    # 然后跑 StyleSSP 的 eval_artfid.py（--gen 结束会打出确切命令）
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "protocol"

N_STYLE, N_CONTENT, RES = 40, 20, 512

# 数据池的候选位置。--check 会逐个探，报告哪些能用。
POOLS = {
    "wikiart": [
        os.environ.get("WIKIART", ""),
        "/openbayes/input/input0/Sim2Struct-1000/temp/jdb/wikiart_ref",
    ],
    "coco": [
        os.environ.get("COCO", ""),
        "/openbayes/input/input0/Sim2Struct-1000/temp/coco",
        "/openbayes/input/input0/Sim2Struct-1000/temp/jdb/coco",
        "/openbayes/input/input0/Sim2Struct-1000/temp/val2017",
    ],
    # 退路：之前 fetch_real_images.py 下的 LAION 真图。**不是 COCO**，
    # 用它就必须在表注里写明「content 用的是 LAION 而非 MS-COCO」。
    # 退路：scalediff_probe/fetch_real_images.py 下的 LAION 真图。
    # **不是 COCO**，用它必须在表注写明。默认输出目录名是 laion_real ——
    # 第一版只写了 real / real_images，漏了这个（那个脚本才是为这个
    # 网络环境写的，链接腐烂、境内可达都考虑过了）。
    "laion_fallback": [
        str(Path(os.environ.get("SD_OUT", "/tmp")) / "laion_real"),
        str(Path(os.environ.get("SD_OUT", "/tmp")) / "real"),
        str(Path(os.environ.get("SD_OUT", "/tmp")) / "real_images"),
    ],
}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_pool(paths):
    for p in paths:
        if not p:
            continue
        d = Path(p)
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.suffix.lower() in EXTS]
        if files:
            return d, files
    return None, []


def cmd_check(args):
    from PIL import Image
    print("═" * 70)
    print("数据摸底")
    print("═" * 70)
    found = {}
    for name, paths in POOLS.items():
        d, files = scan_pool(paths)
        if d is None:
            print(f"\n[{name}] ✗ 都不存在：")
            for p in paths:
                if p:
                    print(f"     {p}")
            continue
        rng = np.random.default_rng(0)
        idx = rng.choice(len(files), min(60, len(files)), replace=False)
        sizes = {}
        for i in idx:
            try:
                sizes[Image.open(files[i]).size] = sizes.get(Image.open(files[i]).size, 0) + 1
            except Exception:
                pass
        mx = max((max(s) for s in sizes), default=0)
        print(f"\n[{name}] ✓ {d}")
        print(f"     {len(files)} 张；抽 {len(idx)} 张看尺寸：")
        for sz, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:4]:
            print(f"       {sz[0]}×{sz[1]}  {n} 张")
        if mx < RES:
            print(f"     ⚠️ 最大边 {mx} < {RES} —— 协议要求从原图 center-crop 到 {RES}²，"
                  f"\n        这里只能上采。**绝对值与发表数不可比**（可用于自比）")
        found[name] = (str(d), len(files), mx)

    print("\n" + "═" * 70)
    ok_s = "wikiart" in found
    ok_c = "coco" in found
    print(f"style  池（WikiArt）: {'✓' if ok_s else '✗ 缺'}")
    print(f"content 池（MS-COCO）: {'✓' if ok_c else '✗ 缺'}"
          + ("" if ok_c else "  ← **这是唯一的阻塞项**"))
    if not ok_c:
        if "laion_fallback" in found:
            print(f"\n  退路：LAION 真图在 {found['laion_fallback'][0]}。")
            print(f"  能跑，但**不是 MS-COCO**，表注必须写明。协议对齐度下降。")
        print(f"\n  要拿 MS-COCO val2017（~1 GB / 5000 张）：")
        print(f"    · 官方 http://images.cocodataset.org/zips/val2017.zip（境内多半不通）")
        print(f"    · 国内镜像 / ModelScope 上有 COCO 的镜像")
        print(f"    · 只需要 20 张 —— 任何能拿到几十张 COCO 图的途径都够")
        print(f"    拿到后：COCO=<目录> python style_probe/protocol.py --check")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "pools.json").write_text(json.dumps(found, ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT/'pools.json'}")


def cmd_draw(args):
    """抽一次 40×20 并落盘。**seed 写进 manifest**，换 seed 就是另一次抽样，
    ②（抽样方差）就靠这个。"""
    from PIL import Image
    sd, sf = scan_pool(POOLS["wikiart"])
    cd, cf = scan_pool(POOLS["coco"])
    if cd is None:
        cd, cf = scan_pool(POOLS["laion_fallback"])
        if cd is None:
            sys.exit("!! content 池缺失，先跑 --check 看怎么办")
        print("!! 用 LAION 退路当 content —— **不是 MS-COCO**，表注必须写明")
    if sd is None:
        sys.exit("!! WikiArt 池缺失")

    rng = np.random.default_rng(args.seed)
    # style 每位艺术家最多取一张，避免 40 张全来自同一人（--scan 那次的教训）
    by_artist = {}
    for f in sf:
        by_artist.setdefault(f.parent.name, []).append(f)
    artists = sorted(by_artist)
    pick_a = rng.permutation(len(artists))[:N_STYLE]
    styles = [sorted(by_artist[artists[i]])[0] for i in sorted(pick_a)]
    contents = [cf[i] for i in sorted(rng.permutation(len(cf))[:N_CONTENT])]

    d = OUT / f"seed{args.seed}"
    (d / "sty").mkdir(parents=True, exist_ok=True)
    (d / "cnt").mkdir(parents=True, exist_ok=True)

    def prep(src, dst):
        """协议是 center-crop 到 512²。源图小于 512 时只能先上采再裁 ——
        这一步会被记进 manifest 的 upscaled 字段，别让它悄悄发生。"""
        im = Image.open(src).convert("RGB")
        w, h = im.size
        up = min(w, h) < RES
        s = RES / min(w, h)
        if s > 1:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
            w, h = im.size
        l, t = (w - RES) // 2, (h - RES) // 2
        im.crop((l, t, l + RES, t + RES)).save(dst)
        return up

    man = {"seed": args.seed, "res": RES, "n_style": len(styles),
           "n_content": len(contents), "style_pool": str(sd),
           "content_pool": str(cd),
           # 来源靠目录里的 SOURCE.txt，**不靠路径猜** —— 用户可以把 COCO=
           # 指向任何目录（比如 teaser 那 5 张），靠路径判断会误报成 MS-COCO。
           "content_source": (cd / "SOURCE.txt").read_text().strip()
                             if (cd / "SOURCE.txt").exists() else "未标注（来源不明）",
           "style": [], "content": []}
    for i, f in enumerate(styles):
        up = prep(f, d / "sty" / f"{i:02d}_{f.parent.name}.png")
        man["style"].append({"i": i, "src": str(f), "artist": f.parent.name, "upscaled": up})
    for i, f in enumerate(contents):
        up = prep(f, d / "cnt" / f"{i:02d}.png")
        man["content"].append({"i": i, "src": str(f), "upscaled": up})
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2))

    n_up = sum(x["upscaled"] for x in man["style"] + man["content"])
    print(f"seed={args.seed}  {len(styles)} style × {len(contents)} content → {d}")
    print(f"  上采过的图：{n_up}/{len(styles)+len(contents)}"
          + ("   ⚠️ 绝对值与发表数不可比" if n_up else ""))
    print(f"  content 来源：{man['content_source'].splitlines()[0]}")
    print(f"  manifest: {d/'manifest.json'}")


def cmd_gen(args):
    """跑 StyleID 式注入，800 张。直接复用 subspace_look 的 full 路径。"""
    import torch
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import subspace_look as S

    d = OUT / f"seed{args.seed}"
    if not (d / "manifest.json").exists():
        sys.exit(f"!! 先跑 --draw --seed {args.seed}")
    man = json.loads((d / "manifest.json").read_text())
    tar = d / "tar"; tar.mkdir(exist_ok=True)

    pipe = S._load(args)
    st = {"mode": "off", "cache": {}, "step": 0, "q_len": args.qlen,
          "P": {}, "arm": "full", "calls": 0, "applied": 0}
    S.install(pipe, st, args.where)

    styles = sorted((d / "sty").glob("*.png"))
    conts = sorted((d / "cnt").glob("*.png"))
    import time
    t0 = time.time(); done = 0
    for si, sp in enumerate(styles):
        st.update(mode="record", cache={}, step=0, calls=0)
        S._record_reference(pipe, st, S._encode(pipe, Image.open(sp).convert("RGB")), args)
        if st["calls"] == 0:
            sys.exit(f"!! 记录阶段没命中 q_len={args.qlen}")
        st["P"] = {}                                  # full 臂不投影
        for ci, cp in enumerate(conts):
            f = tar / f"{sp.stem}__{cp.stem}.png"
            if f.exists():
                done += 1; continue
            st.update(mode="inject", arm="full", step=0)
            S._inject_generate(pipe, st, Image.open(cp).convert("RGB"), args).save(f)
            done += 1
        el = time.time() - t0
        print(f"  [{si+1}/{len(styles)}] {sp.stem}  {done} 张  "
              f"{el/60:.0f}m 已用 / {(len(styles)*len(conts)-done)*el/max(done,1)/60:.0f}m 剩余",
              flush=True)

    print(f"\n{done} 张 → {tar}")
    ev = REPO / "help_code" / "StyleSSP" / "evaluation"
    print(f"\n下一步（ArtFID，用 StyleSSP 现成的尺子，不自己写）：")
    print(f"  # eval_artfid.py 要求 content/style 张数与 stylized 一致，先复制对齐")
    print(f"  cd {ev}")
    print(f"  python eval_artfid.py --sty {d/'sty'} --cnt {d/'cnt'} --tar {tar}")
    src = man.get("content_source", "").splitlines()[0] if man.get("content_source") else "?"
    print(f"\n⚠️ 报数时必须写明：**自抽样 seed={args.seed}"
          f"{'；style 源图经上采' if any(x['upscaled'] for x in man['style']) else ''}"
          f"；content 来源 = {src}"
          f"；与发表值不可直接比。**")
    if len(man["content"]) < 20:
        print(f"⚠️ content 只有 {len(man['content'])} 张（协议是 20）→ "
              f"stylized 只有 {len(man['style'])*len(man['content'])} 张，"
              f"**FID 偏差与方差更大，eval 必须加 --mode art_fid_inf**")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--qlen", type=int, default=4096)
    ap.add_argument("--where", default="up_blocks")
    ap.add_argument("--model", default=os.environ.get(
        "SD15_PATH", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--draw", action="store_true")
    g.add_argument("--gen", action="store_true")
    a = ap.parse_args()
    (cmd_check if a.check else cmd_draw if a.draw else cmd_gen)(a)


if __name__ == "__main__":
    main()
