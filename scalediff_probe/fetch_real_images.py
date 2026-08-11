"""下 LAION 的真图 —— FID 的参考集，也是整条链上风险最高的一环。

ScaleDiff §4.1：*"1,000 image-text pairs … We compute FID, KID, and IS
between generated images and **real images**"*。所以 caption 只是一半，
**另一半是那些 URL 指向的真图**。

为什么这一步必须排在生成基图之前（我第一版排反了）：
  - **下图不用 GPU，生成基图要用 GPU。** 先下图才知道哪些 prompt 的真图
    还活着，只给活着的那些烧 GPU —— 否则一半 GPU 时间给了失效的 prompt。
  - **它是风险最高的一环。** LAION 的 URL 指向全网各站，多年后三到五成
    已失效，而且我们在境内。失败率未知，**先做它，失败了立刻知道**，
    而不是烧完 7 小时 GPU 才发现 FID 没有参考集。

存活率的判据 —— **看乘积，不看区间**：

    候选数 × 存活率 >= 需要数(1200)   ->  够；否则提高 oversample 重取 caption。

    第一版写成"30-60% 就算紧、要提到 5x"，那是个粗糙代理：实测 47% 时
    3600 x 0.47 = 1692 >= 1200，**本来就够**，余量 1.4x。
    真正走不通的门槛是 1200/3600 = 33%，低于它才需要重取；
    低到连 5x 也凑不齐时，退到 §7.2 的备用口径（三行全自己跑、
    只报 A/B 相对变化，表注写明参考集不可得）。

注意：**触发子群不需要真图**（delta 只比 4096 与基图），
所以链接腐烂只影响 FID 那张表，不影响重复指标。

    python scalediff_probe/fetch_real_images.py --split eval --limit 200   # 先探路
    nohup python scalediff_probe/fetch_real_images.py > imgs.log 2>&1 &
"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# **这个阈值第一版取 256，是拍的，而且方向错了（2026-08-10 更正）。**
# 200 条探路里 41 条（20.5%）死在它手上，尺寸是 236/250/240/231/230/220/
# 214/206/204/200 —— 一大批真实照片卡在 256 下面一点点。
#
# 它的用途是排除**占位符**（80x80 图标、"图片不可用"小图），那个量级在
# 100px 以下。而 FID 本来就把所有图缩到 299x299 —— 一张 236px 的真照片
# 在那个尺度上和 300px 的没有区别。所以阈值该定在占位符尺度，不是随手
# 取的整数。
#
# **改动会让存活率变好（47% -> ~68%），但依据独立于这个结果**：
# 重采样目标是 299²，阈值按占位符尺度定。先有依据，后有数字。
MIN_SIDE = 128


def fetch_one(item, out_dir, timeout, min_side):
    """返回 (idx, ok, 原因)。**不抛异常** —— 单条失败不能打断整批。"""
    from PIL import Image
    idx, url = item["idx"], item.get("url")
    p = Path(out_dir) / f"{idx:05d}.jpg"
    if p.exists():
        return idx, True, "cached"
    if not url:
        return idx, False, "no_url"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(20 * 1024 * 1024)          # 上限 20 MB，防超大文件
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if min(im.size) < min_side:
            return idx, False, f"too_small_{min(im.size)}"
        im.save(p, "JPEG", quality=95)
        return idx, True, "ok"
    except Exception as e:
        return idx, False, type(e).__name__


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(root / "eval_prompts.json"))
    ap.add_argument("--out", default=str(root / "laion_real"))
    ap.add_argument("--split", default=None, choices=["tune", "eval"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16,
                    help="并发数。境内出网不稳，太高反而更多超时")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
    a = ap.parse_args()

    meta = json.loads(Path(a.prompts).read_text())
    items = [{**x, "idx": i} for i, x in enumerate(meta["items"])]
    if a.split:
        items = [x for x in items if x.get("split") == a.split]
    if a.limit:
        items = items[:a.limit]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} 条  ->  {out}   并发 {a.workers}  超时 {a.timeout}s")

    t0 = time.time()
    ok, reasons = [], {}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch_one, x, out, a.timeout, a.min_side)
                for x in items]
        for n, f in enumerate(cf.as_completed(futs), 1):
            idx, good, why = f.result()
            reasons[why] = reasons.get(why, 0) + 1
            if good:
                ok.append(idx)
            if n % 25 == 0 or n == len(items):
                el = time.time() - t0
                print(f"\r{n}/{len(items)}  存活 {len(ok)} = "
                      f"{len(ok)/n:.0%}   {el/60:.1f} 分钟  "
                      f"剩约 {el/n*(len(items)-n)/60:.0f} 分钟",
                      end="", flush=True)
    print()

    rate = len(ok) / max(len(items), 1)
    (out / "alive.json").write_text(json.dumps(
        {"prompts": a.prompts, "split": a.split, "n_tried": len(items),
         "n_ok": len(ok), "rate": rate, "reasons": reasons,
         "alive_idx": sorted(ok)}, ensure_ascii=False, indent=1))

    print(f"\n存活 {len(ok)}/{len(items)} = {rate:.1%}")
    print("失败原因分布：" + "  ".join(
        f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
    print(f"名单写入 {out/'alive.json'}")

    n_cand = len(meta["items"])
    need = 1200
    expect = int(n_cand * rate)
    print(f"\n判据（看乘积，不看区间）：候选 {n_cand} x {rate:.1%} "
          f"= 约 {expect} 张存活，需要 {need}")
    if expect >= need:
        print(f"  -> **够**（余量 {expect/need:.1f}x），按原计划走。")
    else:
        need_rate = need / n_cand
        print(f"  -> 不够。需要存活率 >= {need_rate:.0%}，"
              f"或把 --oversample 提到 {need/(n_cand/3*rate):.1f} 重取 caption。")
        if rate < 0.15:
            print("  **存活率过低，重取也难凑齐** -> 退到 §7.2 备用口径：\n"
                  "     三行全部自己跑、只报 A/B 相对变化，表注写明"
                  "'真图参考集不可得，绝对 FID 与发表值不可比'。")
    print("""
下一步（**只给真图活着的那些生成基图**，别给失效的烧 GPU）：
    python scalediff_probe/base_run.py --alive """ + str(out / "alive.json") + """

注意：触发子群不需要真图（delta 只比 4096 与基图），
所以链接腐烂只影响 FID 那张表，不影响重复指标。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
