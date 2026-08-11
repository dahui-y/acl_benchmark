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

存活率决定后面怎么走（判据写在跑之前）：
    >= 60%   3600 条候选足够凑出 1000+200，按原计划；
    30-60%   够但紧，把 oversample 提到 5x 重取 caption；
    < 30%    **参考集这条路走不通** -> 退到 §7.2 的备用口径：
             三行全部自己跑、只报 A/B 相对变化，并在表注写明。

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
MIN_SIDE = 256          # 比这还小的图当作无效（缩略图/占位图）


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

    print("\n判据（跑之前写死）：")
    if rate >= 0.60:
        print("  >=60%  -> 3600 条候选足够凑出 1000+200，按原计划走。")
    elif rate >= 0.30:
        print("  30-60% -> 够但紧。把 fetch_eval_prompts 的 --oversample "
              "提到 5 重取 caption，再跑本脚本。")
    else:
        print("  **<30% -> 参考集这条路走不通。** 退到 §7.2 备用口径：\n"
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
