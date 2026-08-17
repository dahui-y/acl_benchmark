#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取 MS-COCO 的 content 图（只要几十张）。

    照 scalediff_probe/fetch_weights.sh 的路子：**能下** —— 挡住的是我们自己
    在 env.sh 里钉的 HF_HUB_OFFLINE=1，不是网络。hf-mirror.com 在国内是通的。

    ⚠️ 下面这些 repo id 我**没有在这台机器上验证过**。所以脚本的设计是
    「逐个试 + 明确报告哪个成功哪个失败」，而不是赌某一个。
    全失败也不是死路 —— 见 --teaser。

用法：
    conda activate scalediff && source scalediff_probe/env.sh
    python style_probe/fetch_coco.py --hf              # 逐个试 HF 上的 COCO 镜像
    python style_probe/fetch_coco.py --hf --repo <id>  # 试指定的 repo
    python style_probe/fetch_coco.py --teaser          # 退路：用 StyleSSP 论文图里的 5 张
"""

import argparse
import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SD_OUT", "/tmp")) / "style" / "coco"

# HF 上带 COCO 图片的候选数据集。**未经本机验证**，逐个试。
CANDIDATES = [
    ("rafaelpadilla/coco2017", ["data/val*"]),
    ("detection-datasets/coco", ["data/val*"]),
    ("sayakpaul/coco-30-val-2014", None),
    ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", None),
    ("HuggingFaceM4/COCO", ["data/val*"]),
]


def _files(rid):
    """先列仓库文件，**不要猜路径**。
    2026-08-16：第一版给 rafaelpadilla/coco2017 和 HuggingFaceM4/COCO 传了
    `data/val*`，两个都 "Fetching 0 files" —— 模式一个都没匹配上。
    那不是网络问题，是我猜的目录结构不对。"""
    from huggingface_hub import list_repo_files
    return list_repo_files(rid, repo_type="dataset")


def _sizes(rid):
    """{路径: 字节数}。没有 metadata 的返回 0。

    2026-08-17：上一轮 parquet 分支写的是 `pqs[:1]` —— 「下第一个分片」。
    detection-datasets/coco 排序后第一个是 `data/train-00000-of-00040-*.parquet`，
    **485 MB**，而我们只要 20 张图。进度条不动的直接原因就是它太大。
    分片大小差异是**可查的**（siblings 带 lfs.size），不该靠排序碰运气。
    """
    from huggingface_hub import HfApi
    try:
        info = HfApi().dataset_info(rid, files_metadata=True)
    except Exception as e:
        print(f"   (取不到文件大小 {type(e).__name__}，退化为按名字选) ")
        return {}
    out = {}
    for s in info.siblings or []:
        sz = getattr(s, "size", None)
        lfs = getattr(s, "lfs", None)
        if sz is None and lfs is not None:
            sz = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        out[s.rfilename] = sz or 0
    return out


def _pick_parquet(rid, pqs):
    """挑**最小的**分片，同分优先 val/test（协议要的就是 COCO val）。
    返回 (路径, 字节数)。"""
    sz = _sizes(rid)
    def key(f):
        n = f.lower()
        return (0 if ("val" in n or "test" in n) else 1, sz.get(f, 1 << 62), f)
    best = min(pqs, key=key)
    return best, sz.get(best, 0)


def cmd_ls(args):
    _endpoint(args)
    rid = args.repo or "rafaelpadilla/coco2017"
    try:
        fs = _files(rid)
    except Exception as e:
        sys.exit(f"!! 列不出 {rid}: {type(e).__name__}: {str(e)[:200]}")
    print(f"{rid}  共 {len(fs)} 个文件")
    import collections
    ext = collections.Counter(Path(f).suffix.lower() for f in fs)
    print(f"  后缀分布：{dict(ext.most_common(8))}")
    top = collections.Counter(f.split("/")[0] for f in fs)
    print(f"  顶层目录：{dict(top.most_common(8))}")
    print(f"  前 25 个：")
    for f in fs[:25]:
        print(f"    {f}")


def _endpoint(args):
    """2026-08-16 订正：默认走 hf-mirror 是**上一个 session 对 models 的结论**，
    我把它当成对 datasets 也成立的前提，那是错的。

    证据：第一次 `huggingface-cli login` 失败时报的是
      `401 ... for url: https://huggingface.co/api/whoami-v2 (Request ID: ...)`
    —— **那是官方站返回的真实 401，带 Request ID**，说明这台机器能直连
    huggingface.co（走 OpenBayes 的代理）。既然直连通，就不该硬塞一个
    对 dataset LFS 不完整的镜像。

    所以：`--official` 显式清掉 HF_ENDPOINT 走官方站。
    """
    os.environ["HF_HUB_OFFLINE"] = "0"
    # 卡住要能自己死掉。默认 10s 太短会误杀慢代理，30s 足够区分「慢」和「不动」。
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    if args.official:
        os.environ.pop("HF_ENDPOINT", None)
        print("HF_ENDPOINT = (清掉，走官方 huggingface.co)")
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
    print(f"HF_HOME     = {os.environ.get('HF_HOME','(未设)')}")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if os.environ.get(k):
            print(f"{k:12s}= {os.environ[k]}")
    print()


def cmd_hf(args):
    _endpoint(args)
    from huggingface_hub import snapshot_download

    cands = [(args.repo, None)] if args.repo else CANDIDATES
    for rid, _ in cands:
        print(f"── 试 {rid}")
        # ① 先列文件，按实际内容选模式；不再用写死的猜测
        try:
            fs = _files(rid)
        except Exception as e:
            print(f"   ✗ 列文件失败 {type(e).__name__}: {str(e)[:140]}\n")
            continue
        imgs = [f for f in fs if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        pqs = [f for f in fs if Path(f).suffix.lower() in {".parquet", ".arrow"}]
        # zip 也要认 —— nlphuji 那个就是 .zip 装图，第一版把它跳过了
        zips = [f for f in fs if Path(f).suffix.lower() == ".zip"]
        if imgs:
            pats = imgs[:args.n]                    # 散图：只取需要的那几张
            print(f"   {len(fs)} 文件，散图 {len(imgs)} 张 → 只下前 {len(pats)} 张")
        elif zips:
            pats = zips[:1]
            print(f"   {len(fs)} 文件，zip {len(zips)} 个 → 只下 {pats[0]}")
        elif pqs:
            one, nb = _pick_parquet(rid, pqs)       # parquet：挑**最小**的那个分片
            pats = [one]
            mb = f"{nb/1e6:.0f} MB" if nb else "大小未知"
            print(f"   {len(fs)} 文件，parquet {len(pqs)} 个 → 挑最小的 {one}（{mb}）")
            if nb > args.max_mb * 1e6:
                print(f"   ✗ 最小的分片也有 {mb} > --max-mb {args.max_mb}，跳过。"
                      f"为 20 张图下这么大不值。\n")
                continue
        else:
            print(f"   !! 既没有散图也没有 parquet。后缀：",
                  {Path(f).suffix for f in fs[:40]}, "\n")
            continue
        try:
            p = snapshot_download(rid, repo_type="dataset",
                                  allow_patterns=pats, max_workers=4)
        except Exception as e:
            print(f"   ✗ 下载失败 {type(e).__name__}: {str(e)[:200]}")
            print(f"      → 可能是 gated（`hf auth login` 后重试）"
                  f"或 hf-mirror 的 LFS 转发不稳\n")
            continue
        print(f"   ✓ 下到 {p}")
        n = _extract(Path(p), args.n)
        if n:
            print(f"\n拿到 {n} 张 → {OUT}")
            print(f"下一步：COCO={OUT} python style_probe/protocol.py --check")
            return
        print(f"   !! 下下来了但没解出图片，看看 {p} 里是什么\n")
    print("\n全部失败。两条路：")
    print("  · --repo <别的 id> 再试（HF 上 COCO 的镜像不止这几个）")
    print("  · --teaser 用 StyleSSP 论文图里的 5 张真 content 图（见该模式的说明）")


def _extract(root, want):
    """从下载目录里刨出图片。数据集可能是散图，也可能是 parquet / arrow。"""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [f for f in root.rglob("*") if f.suffix.lower() in exts][:want]
    if imgs:
        for i, f in enumerate(sorted(imgs)):
            Image.open(f).convert("RGB").save(OUT / f"content_{i:03d}.png")
        (OUT / "SOURCE.txt").write_text(f"MS-COCO\nfrom {root}\n")
        return len(imgs)
    # zip：解出里面的图（只解需要的那几张，不全解）
    import zipfile
    for z in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist()
                         if Path(n).suffix.lower() in exts and not n.startswith("__")]
                print(f"   zip {z.name} 里有 {len(names)} 张图")
                for i, n in enumerate(sorted(names)[:want]):
                    with zf.open(n) as fh:
                        Image.open(io.BytesIO(fh.read())).convert("RGB").save(
                            OUT / f"content_{i:03d}.png")
                if names:
                    (OUT / "SOURCE.txt").write_text(f"MS-COCO\nfrom {z}\n")
                    return min(len(names), want)
        except Exception as e:
            print(f"   !! 解 {z.name} 失败：{type(e).__name__}: {str(e)[:120]}")
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        return 0
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("   !! 是 parquet 但没装 pyarrow：pip install pyarrow")
        return 0
    got = 0
    for f in pqs:
        t = pq.read_table(f)
        col = next((c for c in t.column_names if c.lower() in
                    ("image", "img", "images")), None)
        if col is None:
            print(f"   !! {f.name} 的列里没有 image：{t.column_names[:8]}")
            continue
        for v in t.column(col).to_pylist():
            b = v.get("bytes") if isinstance(v, dict) else v
            if not isinstance(b, (bytes, bytearray)):
                continue
            Image.open(io.BytesIO(b)).convert("RGB").save(OUT / f"content_{got:03d}.png")
            got += 1
            if got >= want:
                (OUT / "SOURCE.txt").write_text(f"MS-COCO\nfrom {root}\n")
                return got
    if got:
        (OUT / "SOURCE.txt").write_text(f"MS-COCO\nfrom {root}\n")
    return got


def cmd_direct(args):
    """不经 HF：COCO 官方 CDN 上**逐张**取 val2017。

    2026-08-17：`--hf --official` 已经证明网络通（三个 repo 都列出了文件、
    拿到了真实 lfs.size、LFS 传输也起来了）。堵的是**打包方式**——
    HF 上的 COCO 最小 val 分片 404 MB、最小 zip 818 MB，而我们只要 20 张。
    官方 CDN 是按图片编号直接给 JPEG 的，一张约 150 KB。

    ⚠️ 下面这串 id 是 val2017 按编号排序的开头，**我凭记忆写的，没在本机核对过**。
    所以脚本逐个报 200 / 404，够数就停 —— 错一两个不影响，全 404 会立刻看出来
    （那说明 CDN 不通，不是 id 错，因为 id 全错的概率远小于网络不通）。
    """
    import urllib.request
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
           1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761,
           1818, 1993, 2006, 2149, 2153, 2157, 2261, 2299, 2431, 2473,
           2532, 2587, 2592, 2685, 2923, 3156, 3255, 3501, 3553, 3661]
    px = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if px:
        print(f"proxy = {px}")
        op = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": px, "https": px}))
        urllib.request.install_opener(op)
    got, miss = 0, 0
    for cid in ids:
        if got >= args.n:
            break
        url = f"http://images.cocodataset.org/val2017/{cid:012d}.jpg"
        try:
            raw = urllib.request.urlopen(url, timeout=30).read()
        except Exception as e:
            miss += 1
            print(f"   ✗ {cid:012d}  {type(e).__name__}: {str(e)[:90]}")
            if miss >= 5 and got == 0:
                sys.exit("\n!! 前 5 张全失败且一张没成 —— 是 CDN 不通，不是 id 错。"
                         "\n   退路：直接用已经在盘上的 LAION content 集，"
                         "见 protocol.py --draw。")
            continue
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.save(OUT / f"content_{got:03d}.png")
        print(f"   ✓ {cid:012d}  {im.size[0]}×{im.size[1]}  {len(raw)/1e3:.0f} KB")
        got += 1
    if not got:
        sys.exit("!! 一张都没拿到。")
    (OUT / "SOURCE.txt").write_text(
        f"MS-COCO val2017 ({got} images)\n"
        f"from http://images.cocodataset.org/val2017/ (official CDN, per-image)\n"
        f"ids: {ids[:got]}\n")
    print(f"\n拿到 {got} 张（{miss} 张失败）→ {OUT}")
    print(f"下一步：COCO={OUT} python style_probe/protocol.py --check")


def cmd_teaser(args):
    """退路：StyleSSP 论文 teaser（assets/ours.jpg）的**第 0 列就是它自己用的
    content 图** —— 马、芝加哥天际线、人像、埃菲尔铁塔、金门桥，5 张。

    这 5 张是**真货**（论文自己的 content 集里的），比随便找 20 张自然照片
    更贴协议。代价是 n=5 而不是 20：

      · 800 张 → 200 张。**FID 在 200 张上偏差和方差都明显更大。**
        eval_artfid.py 有 `--mode art_fid_inf`（向无穷外推）能缓解，必须用它。
      · 所以这条路**只适合建自比基线**，绝对值更加不可对外比。

    但对 C 的两个目的（自己的基线 + 抽样方差）是够的 —— 而且抽样方差那一项
    本来主要来自 style 侧（40 张），content 侧固定反而让方差读数更干净。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import subspace_look as S
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(S.TEASER).convert("RGB")
    W, H = im.size
    import numpy as np
    g = np.asarray(im.convert("L"), np.float32)
    cw, cx, rc = S.fit_grid(g, S.GRID_C, W, 0)
    ch, cy, rr = S.fit_grid(g, S.GRID_R, H, 1)
    print(f"实测网格：格宽 {cw:.2f} 起点 {cx:+.1f}（残差 {rc:.1f}px）"
          f" / 格高 {ch:.2f} 起点 {cy:+.1f}（残差 {rr:.1f}px）")
    ins = 0.03
    names = ["horse", "skyline", "portrait", "eiffel", "bridge"]
    for r, nm in enumerate(names, start=1):
        x0, y0 = cx + 0 * cw, cy + r * ch
        mx, my = cw * ins, ch * ins
        box = (max(0, round(x0 + mx)), max(0, round(y0 + my)),
               min(W, round(x0 + cw - mx)), min(H, round(y0 + ch - my)))
        im.crop(box).resize((512, 512), Image.LANCZOS).save(
            OUT / f"content_{r-1:03d}_{nm}.png")
        print(f"  {nm}")
    # ★ 这 5 张**不是 MS-COCO**（Brad Pitt 肯定不在 COCO 里）。
    #   靠路径判断来源会误报，所以把来源写成文件，protocol.py 读它。
    (OUT / "SOURCE.txt").write_text(
        "NOT-COCO: StyleSSP teaser (assets/ours.jpg) column 0\n"
        "horse / skyline / portrait(Brad Pitt) / eiffel / bridge\n"
        "这是论文自己配图用的 content 图，不是 MS-COCO。报数时必须写明。\n")
    print(f"\n5 张 → {OUT}")
    print(f"⚠️ **这 5 张不是 MS-COCO**，是 StyleSSP 配图用的 content 图。"
          f"已写入 SOURCE.txt，protocol.py 会读进 manifest。")
    print(f"⚠️ n=5 而非 20：800 张 → 200 张，**FID 偏差与方差都更大**。")
    print(f"   跑 eval_artfid.py 时**必须**加 --mode art_fid_inf。")
    print(f"下一步：COCO={OUT} python style_probe/protocol.py --check")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="取几张")
    ap.add_argument("--repo", default=None, help="指定 HF dataset repo id")
    ap.add_argument("--max-mb", type=int, default=200,
                    help="单个分片超过这个大小就跳过（默认 200MB）")
    ap.add_argument("--official", action="store_true",
                    help="清掉 HF_ENDPOINT，走官方 huggingface.co 而非镜像")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ls", action="store_true", help="只列仓库文件，不下载")
    g.add_argument("--hf", action="store_true")
    g.add_argument("--direct", action="store_true",
                   help="不经 HF，从 COCO 官方 CDN 逐张取 val2017（每张约 150KB）")
    g.add_argument("--teaser", action="store_true")
    a = ap.parse_args()
    (cmd_ls if a.ls else cmd_hf if a.hf else
     cmd_direct if a.direct else cmd_teaser)(a)


if __name__ == "__main__":
    main()
