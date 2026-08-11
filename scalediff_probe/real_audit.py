"""真图下完之后的体检 —— **在烧 GPU 之前**确认参考集是真的。

64% 存活不等于 64% 可用。链接腐烂有两种：

    诚实的：404 / 超时 / 连接断  ->  已经落到失败那一栏，不会污染参考集；
    不诚实的：HTTP 200 返回一张"图片已下架"的占位图，或一个追踪像素
        ->  **它落在 ok 那一栏。**

证据就写在这轮的失败分布里：`too_small_1 / too_small_10 / too_small_80`
各有几条。这些 URL 的元数据里写着正常尺寸（否则 `--min-px 299` 早把它们
筛掉了），下下来却是 1px、10px、80px —— 那不是原图，是站点返回的占位物。
既然存在"下下来是 1px 占位图"的，就一定存在"下下来是 600×400 占位图"的：
**那种尺寸合法，直接混进了参考集**，尺寸检查抓不到。

判据（写在看结果之前）：

  1. **重复**。逐字节 md5 相同、或 32×32 灰度指纹相同的组：
     size >= 2  -> 只保留一张。理由与占位图无关 —— 对 FID 而言重复样本
                   本身就是错的，它把参考分布往点质量上压。
     size >= 5  -> 整组丢弃并打印出来。同一张图在几千条不同 caption 下
                   反复出现，只可能是占位图。
  2. **磁盘上重新量短边**，< 299 的丢弃。缓存分支（`p.exists()` 直接返回）
     绕过了尺寸检查，早期用 256 阈值下的图会漏进来。
  3. **近乎纯色**（灰度标准差 < 5）丢弃：纯白/纯灰底的"无图"占位。

**不做别的过滤。** 每多一条前人没有的过滤，绝对 FID 就更不可比一分。
域名分布、长宽比、mtime 只报告不过滤 —— 报告是给判断用的，不是判据。

  （另一条可用但**故意不用**的诊断：真图与自身 caption 的 CLIP 分数，
   低分几乎必是占位图。不用的理由是它会把参考集往"图文对齐好"的方向偏，
   而 CLIP 正是我们要报的指标之一，沾上就说不清了。）

    python scalediff_probe/real_audit.py            # 只报告
    python scalediff_probe/real_audit.py --apply    # 按上面的判据写 clean.json
"""

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

MIN_SIDE = 299          # 与 fetch_real_images.MIN_SIDE 同值同据
FLAT_STD = 5.0          # 灰度标准差低于此视为纯色
DROP_GROUP = 5          # 指纹相同且组大小 >= 此值 -> 整组丢


def probe(p):
    """解码一张图，返回体检项。**不抛异常** —— 坏文件也是结果之一。"""
    from PIL import Image
    import numpy as np
    rec = {"idx": int(p.stem), "mtime": p.stat().st_mtime,
           "bytes": p.stat().st_size}
    try:
        raw = p.read_bytes()
        im = Image.open(io.BytesIO(raw))
        im.load()
        rgb = im.convert("RGB")
        g = np.asarray(rgb.convert("L"), dtype=np.uint8)
        g32 = np.asarray(Image.fromarray(g).resize((32, 32), Image.BILINEAR))
        rec.update(w=rgb.size[0], h=rgb.size[1],
                   md5=hashlib.md5(raw).hexdigest(),
                   fp=hashlib.md5(g32.tobytes()).hexdigest(),
                   std=float(g.std()), err=None)
    except Exception as e:
        rec.update(err=type(e).__name__)
    return rec


def groups_of(recs, key):
    d = defaultdict(list)
    for r in recs:
        if r.get("err") is None:
            d[r[key]].append(r["idx"])
    return {k: sorted(v) for k, v in d.items() if len(v) > 1}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(root / "laion_real"))
    ap.add_argument("--prompts", default=str(root / "eval_prompts.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--apply", action="store_true",
                    help="按判据写 clean.json（不加则只报告）")
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    a = ap.parse_args()

    d = Path(a.dir)
    files = sorted(x for x in d.glob("*.jpg"))
    if not files:
        sys.exit(f"{d} 里没有 jpg")
    meta = json.loads(Path(a.prompts).read_text())
    items = meta["items"]

    print(f"体检 {len(files)} 张  <-  {d}")
    t0 = time.time()
    recs = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(probe, files), 1):
            recs.append(r)
            if n % 200 == 0 or n == len(files):
                print(f"\r  {n}/{len(files)}  {time.time()-t0:.0f}s",
                      end="", flush=True)
    print()

    bad = [r for r in recs if r["err"]]
    good = [r for r in recs if not r["err"]]
    print(f"\n解码失败 {len(bad)}"
          + ("  " + "  ".join(f"{k}={v}" for k, v in
                              Counter(r['err'] for r in bad).items())
             if bad else ""))

    # ---- 尺寸：磁盘上重新量，不信下载时的判断 ----
    sides = sorted(min(r["w"], r["h"]) for r in good)
    small = [r for r in good if min(r["w"], r["h"]) < a.min_side]
    q = lambda f: sides[min(int(len(sides) * f), len(sides) - 1)]
    print(f"\n短边分位  p0={sides[0]}  p10={q(.1)}  p50={q(.5)}  "
          f"p90={q(.9)}  p100={sides[-1]}")
    print(f"短边 < {a.min_side} 的：{len(small)} 张")
    if small:
        # **看它们落在哪个区间。** 全部落在 [256, 299) -> 正是旧阈值 256
        # 放行的那扇窗，来源是缓存分支；有低于 256 的 -> 另有机制，要查。
        lo = min(min(r["w"], r["h"]) for r in small)
        print("   " + "  ".join(f"{r['idx']}:{r['w']}×{r['h']}"
                                for r in sorted(small, key=lambda r: r["idx"])))
        print(f"   最小短边 {lo}  ->  " + (
            "**全部落在 [256,299)，正是旧阈值 256 放行的那扇窗** —— "
            "来源是缓存分支，删掉重下即可。"
            if lo >= 256 else
            "**有低于 256 的，旧阈值解释不了 —— 另有机制，先查再删。**"))
    if small:
        idxs = sorted(r["idx"] for r in small)
        print(f"   删除命令： cd {d} && rm -f " +
              " ".join(f"{i:05d}.jpg" for i in idxs[:12]) +
              (" ..." if len(idxs) > 12 else ""))
    ar = sorted((max(r["w"], r["h"]) / min(r["w"], r["h"]), r["idx"])
                for r in good)
    print(f"长宽比 p50={ar[len(ar)//2][0]:.2f}  p99={ar[int(len(ar)*.99)][0]:.2f}"
          f"  最极端 {ar[-1][0]:.1f}:1 (idx {ar[-1][1]})   **只报告，不过滤**")

    # ---- 重复：占位图的主证据 ----
    gm = groups_of(good, "md5")
    gf = groups_of(good, "fp")
    dup_md5 = sum(len(v) for v in gm.values())
    dup_fp = sum(len(v) for v in gf.values())
    print(f"\n逐字节重复：{len(gm)} 组 / {dup_md5} 张"
          f"     32×32 指纹重复：{len(gf)} 组 / {dup_fp} 张")
    by_idx = {r["idx"]: r for r in good}
    big = sorted(gf.items(), key=lambda kv: -len(kv[1]))[:10]
    if big:
        print("  最大的几组（组大小 / 尺寸 / 域名 / 示例 caption）：")
        for _, idxs in big:
            r0 = by_idx[idxs[0]]
            doms = Counter(urlparse(items[i].get("url") or "").netloc
                           for i in idxs if i < len(items))
            cap = (items[idxs[0]]["prompt"][:46]
                   if idxs[0] < len(items) else "")
            dom = "  ".join(f"{k}×{v}" for k, v in doms.most_common(2))
            print(f"    {len(idxs):>4}  {r0['w']}×{r0['h']}  {dom:<40} {cap}")
        print(f"  组大小 >= {DROP_GROUP} 的整组丢弃（几乎只可能是占位图）；"
              f"2..{DROP_GROUP-1} 的每组留一张。")

    flat = [r for r in good if r["std"] < FLAT_STD]
    print(f"\n近乎纯色（灰度 std < {FLAT_STD}）：{len(flat)} 张")

    # ---- mtime：识别上一轮候选表遗留的缓存文件 ----
    # 固定的 1 小时门限是个**错的仪器**：探路轮如果就在正式轮前十几分钟跑，
    # 整个跨度不足 1 小时，它必然报 0。所以改成找 mtime 序列里最大的空档 ——
    # 两轮之间的停顿会自己显出来。
    mt = sorted(r["mtime"] for r in recs)
    gaps = [(mt[i + 1] - mt[i], i) for i in range(len(mt) - 1)]
    gap, gi = max(gaps) if gaps else (0, 0)
    print(f"\nmtime 跨度 {(mt[-1]-mt[0])/60:.0f} 分钟；"
          f"最大空档 {gap/60:.1f} 分钟（其前有 {gi+1} 张）")
    cut = mt[gi] if gap > 300 else mt[0] - 1
    stale = [r for r in recs if r["mtime"] <= cut]
    print(f"空档之前的：{len(stale)} 张")

    # **权威判据是候选表指纹，不是 mtime。** mtime 空档分不清"换了候选表"
    # 和"同一张表跑了两趟、中间隔了二十分钟"，两者长得一模一样 ——
    # 它只能提示"这里有一批更早的文件"，不能断定它们属于哪一代表。
    tp = d / "table.json"
    if tp.exists():
        import hashlib
        t = json.loads(tp.read_text())
        now = hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest()
        same = t.get("sha256") == now
        print(f"候选表指纹 {t.get('sha256','?')[:16]}（记于 {t.get('when')}）"
              f"  当前 {now[:16]}  -> " +
              ("**一致**，目录里就是这张表下的图；上面的空档只是两趟之间的停顿。"
               if same else
               "**不一致 —— 目录里混着两代表的文件，先清干净。**"))
        if same:
            stale = []
    else:
        print("  目录里没有 table.json（早于该机制的批次）—— "
              "只能退回 mtime 启发式，它会把两趟之间的停顿也算进去。")
    if stale:
        print("  **可能是上一轮候选表下的缓存**（没有 table.json 时只能这么猜）。"
              "\n  若候选表重取过，下标就整体位移了，它们的 idx 会对应另一条"
              "\n  caption。FID 比分布不比配对，所以不致命；处置是删掉重下"
              "\n  （其余文件命中缓存，只补这些）：")
        print(f"    find {d} -name '*.jpg' ! -newermt '@{cut:.0f}' -delete")

    # ---- split 够不够 ----
    alive_p = d / "alive.json"
    if alive_p.exists():
        alive = json.loads(alive_p.read_text())["alive_idx"]
        c = Counter(items[i].get("split") for i in alive if i < len(items))
        print(f"\nalive 按 split：" + "  ".join(f"{k}={v}" for k, v in c.items()))

    # ---- 判据 ----
    drop, why = set(), {}
    def kill(i, reason):
        if i not in drop:
            drop.add(i); why[reason] = why.get(reason, 0) + 1
    for r in bad:
        kill(r["idx"], "decode_fail")
    for r in small:
        kill(r["idx"], "short_side")
    for r in flat:
        kill(r["idx"], "flat")
    for _, idxs in gf.items():
        if len(idxs) >= DROP_GROUP:
            for i in idxs:
                kill(i, "dup_group_big")
        else:
            for i in idxs[1:]:
                kill(i, "dup_keep_one")
    keep = sorted(r["idx"] for r in recs if r["idx"] not in drop)
    print(f"\n判据执行：丢 {len(drop)} 张  ("
          + "  ".join(f"{k}={v}" for k, v in sorted(why.items(),
                                                    key=lambda kv: -kv[1]))
          + f")   ->  留 {len(keep)} 张")
    ck = Counter(items[i].get("split") for i in keep if i < len(items))
    print("留下的按 split：" + "  ".join(f"{k}={v}" for k, v in ck.items()))
    need = {"eval": 1000, "tune": 200}
    for k, v in need.items():
        got = ck.get(k, 0)
        print(f"  {k}: {got} / 需要 {v}  -> " +
              (f"**够**（{got/v:.1f}x）" if got >= v else "**不够**"))

    if a.apply:
        p = d / "clean.json"
        p.write_text(json.dumps(
            {"dir": str(d), "min_side": a.min_side, "flat_std": FLAT_STD,
             "drop_group": DROP_GROUP, "n_files": len(recs),
             "n_keep": len(keep), "drop_reasons": why,
             "alive_idx": keep}, ensure_ascii=False, indent=1))
        print(f"\n写出 {p}（键名沿用 alive_idx，可直接喂给 base_run.py --alive）")
    else:
        print("\n只报告了。确认无误后加 --apply 写 clean.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
