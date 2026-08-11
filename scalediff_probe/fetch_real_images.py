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

    第一版写成"30-60% 就算紧、要提到 5x"，那是个粗糙代理 —— 该看的是乘积。
    **不够时提高 oversample（多取候选），绝不降低 MIN_SIDE。**
    候选几乎免费，质量标准只有一个。
    只有在候选池本身耗尽（128 片读完仍凑不齐）时，才退到 §7.2 的备用口径
    （三行全自己跑、只报 A/B 相对变化，表注写明参考集不可得）。

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
# **阈值 = 299，理由是"参考图不许被上采样"（2026-08-10 定案）。**
#
# 中途我想把它从 256 降到 128，理由是"FID 反正缩到 299²，236px 和 300px
# 没区别"。**那句话对生成图成立，对真图不成立，而且是朝着数据变多的方向
# 找的说辞：**
#     生成图 4096² -> 299²：降采样，信息足够，无损失；
#     真图   236px -> 299²：**上采样，是插值出来的**，比原生 299px 高频更少。
# Inception 对锐度/纹理敏感，把上采样过的真图混进参考集，会让参考分布
# **系统性偏糊** —— 这个偏差与我们的方法无关，纯粹是数据处理引入的。
#
# 所以正确的原则指向**更严**，不是更松：参考图短边 >= 299，不做上采样。
#
# 数量不够时的解法是**多取候选**，不是降标准：第一个 row group 就有
# 405,224 行（我们只读了 196,608），一共 128 片 —— 候选几乎无限，
# 质量标准只有一个。
MIN_SIDE = 299


def fetch_one(item, out_dir, timeout, min_side):
    """返回 (idx, ok, 原因)。**不抛异常** —— 单条失败不能打断整批。"""
    from PIL import Image
    idx, url = item["idx"], item.get("url")
    p = Path(out_dir) / f"{idx:05d}.jpg"
    if p.exists():
        # **缓存也要过尺寸检查。** 第一版直接 return，于是早期用 256 阈值
        # 下下来的图（短边 256..298）绕过检查留在了参考集里 —— 这正是
        # "阈值不能将就"要防的那件事，从后门溜进来了。
        try:
            im = Image.open(p)
            im.load()
            if min(im.size) >= min_side:
                return idx, True, "cached"
            p.unlink()
        except Exception:
            p.unlink(missing_ok=True)
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
    ap.add_argument("--allow-mixed", action="store_true",
                    help="候选表指纹不符时仍然继续。**默认拒绝** —— "
                         "见下面 table.json 那段")
    a = ap.parse_args()

    meta = json.loads(Path(a.prompts).read_text())
    items = [{**x, "idx": i} for i, x in enumerate(meta["items"])]
    if a.split:
        items = [x for x in items if x.get("split") == a.split]
    if a.limit:
        items = items[:a.limit]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # **候选表指纹。** 文件名是 idx，而 idx 是"在候选表里的位置" ——
    # 候选表一重取，同一个 idx 就指向另一条 caption，目录里于是混着两代文件。
    # 这件事发生过一次（见 §7.1.7），而且当时是靠 mtime 空档这种启发式去猜的。
    # 现在把表的 sha256 钉在目录里：换了表就**拒绝运行**，让它显式，不靠猜。
    import hashlib
    tsha = hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest()
    stamp = out / "table.json"
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old.get("sha256") != tsha and not a.allow_mixed:
            print(f"**候选表变了。** 目录里的图是按\n  {old.get('sha256','?')[:16]}"
                  f"\n下的，当前 {a.prompts} 是\n  {tsha[:16]}\n"
                  f"（记录于 {old.get('when')}，{old.get('n_items')} 条）\n\n"
                  "同一个 idx 在两代表里指向不同 caption，混在一起就说不清了。\n"
                  "二选一：\n"
                  f"  rm -rf {out}          # 全部重下，最干净\n"
                  "  --allow-mixed          # 明知故犯，且必须在论文里交代")
            return 2
    stamp.write_text(json.dumps(
        {"prompts": str(a.prompts), "sha256": tsha, "n_items": len(meta["items"]),
         "when": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1))

    print(f"{len(items)} 条  ->  {out}   并发 {a.workers}  超时 {a.timeout}s")
    print(f"候选表指纹 {tsha[:16]}")

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
        print(f"  -> 不够。**提高 --oversample 多取候选，不要降 MIN_SIDE**"
              f"（建议 {need/(n_cand/3*max(rate,1e-6)):.1f}）。"
              f"候选几乎免费，质量标准只有一个。")
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
