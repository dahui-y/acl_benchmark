"""取评测用的 LAION image-text pair（第三条腿的输入）—— 照 ScaleDiff 的原样做。

ScaleDiff §4.1 Evaluation 原文：

    we randomly sample 1,000 image-text pairs from the LAION-5B dataset and
    generate one image per prompt using each method. We compute FID, KID,
    and IS between generated images and real images.

所以参考集就是 LAION 那 1000 张真图本身，不是 COCO。我之前提的
"COCO vs LAION 二选一"是我自己造出来的两难 —— parquet 里 TEXT 和 URL
在同一行，取 caption 生成、取 URL 下真图，就是他们那套。

唯一的真实工程问题是**链接腐烂**（LAION 的 URL 指向全网各站，多年后
相当一部分已失效）。解法是超采样：多取几倍的行，留前 1000 个下成功的。
这是工程细节，不是设计分叉。

保留的两条不可比因素（写在用它之前，不是事后解释）：
  - 采不到他们那 1000 条 —— 抽样方差；
  - relaion 是 2023-12 下架后的安全过滤重发，分布与原始 LAION-5B 微移。
判据不变：我们自己复现的 ScaleDiff 行是锚，落在发表值附近则其余
baseline 可引用发表数字，落得远则三行全自己跑、只报 A/B 相对变化。

关键约束：那些 parquet 分片是给 20 亿行用的，单片就上 GB，而我们只要 1000
条 caption。所以**不下整片** —— parquet 的 footer 里有 row group 索引，
用 HTTP Range 只读 footer + 第一个 row group，几 MB 就够。

顺带把 URL 列也存下来：真图参考集如果最后要走 LAION 而不是 COCO，
那一列就是入口（见 §7.2 的可比性讨论）。

    python scalediff_probe/fetch_eval_prompts.py                  # 默认 relaion
    python scalediff_probe/fetch_eval_prompts.py --repo laion/laion2B-en-aesthetic
"""

import argparse
import io
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfnet import pick_endpoint                        # noqa: E402


class HttpRangeFile(io.RawIOBase):
    """只读、可 seek 的 HTTP 文件对象。pyarrow 靠 seek 读 footer。"""

    def __init__(self, url, size=None):
        self.url, self._pos = url, 0
        self._size = size if size is not None else self._head_size()

    def _head_size(self):
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            if n is None:
                raise RuntimeError("服务端不给 Content-Length，没法做 range 读")
            return int(n)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = (off if whence == 0 else
                     self._pos + off if whence == 1 else self._size + off)
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        for attempt in range(4):                        # 端点间歇性掉线，重试
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    buf = r.read()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  range 读失败({type(e).__name__})，重试 {attempt+1}/3")
        self._pos += len(buf)
        return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="laion/relaion2B-en-research-safe",
                    help="下架后的官方重发；比 laion2B-en-aesthetic 更该用")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="LAION 链接腐烂严重；多存这么多倍的候选行，"
                         "下图时留前 n 个下成功的")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=4,
                    help="太短的 caption 生不出场景，且和我们的 prompt 差太远")
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ep = pick_endpoint()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(endpoint=ep)
    info = api.repo_info(a.repo, repo_type="dataset")
    shards = sorted(s.rfilename for s in (info.siblings or [])
                    if s.rfilename.endswith(".parquet"))
    if not shards:
        print(f"{a.repo} 里没有 parquet"); return 1
    url = f"{ep}/datasets/{a.repo}/resolve/main/{shards[0]}"
    print(f"分片 {shards[0]}（共 {len(shards)} 片，只读第一片的第一个 row group）")

    f = HttpRangeFile(url)
    print(f"  整片 {f._size / 2**20:.0f} MB —— 不下载，只 range 读")
    pf = pq.ParquetFile(f)
    cols = pf.schema_arrow.names
    print(f"  列: {cols}")

    tcol = next((c for c in cols if c.upper() in ("TEXT", "CAPTION")), None)
    ucol = next((c for c in cols if c.upper() == "URL"), None)
    if tcol is None:
        print(f"  找不到 caption 列，实际列见上"); return 1

    want = [c for c in (tcol, ucol) if c]
    tbl = pf.read_row_group(0, columns=want)
    print(f"  第一个 row group {tbl.num_rows} 行")

    texts = tbl.column(tcol).to_pylist()
    urls = tbl.column(ucol).to_pylist() if ucol else [None] * len(texts)

    seen, pool = set(), []
    for t, u in zip(texts, urls):
        if not t:
            continue
        t = " ".join(t.split())
        w = len(t.split())
        if not (a.min_words <= w <= a.max_words):
            continue
        k = t.lower()
        if k in seen:                                   # 去重：LAION 里重复很多
            continue
        seen.add(k)
        pool.append({"prompt": t, "url": u})
    need = int(a.n * a.oversample)
    print(f"  过滤+去重后 {len(pool)} 条可用（需要 {need} = {a.n}×{a.oversample} 超采样）")
    if len(pool) < need:
        print(f"  **不足 {need} 条** —— 多读一个 row group（read_row_group(1)）再来")
        return 1

    random.Random(a.seed).shuffle(pool)
    picked = pool[:need]

    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    out = Path(a.out) if a.out else root / "eval_prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "repo": a.repo, "shard": shards[0], "row_group": 0,
        "seed": a.seed, "n": a.n, "oversample": a.oversample,
        "protocol": "ScaleDiff §4.1: 1000 LAION-5B image-text pairs; "
                    "FID/KID/IS vs the real images of those same pairs",
        "filter": {"min_words": a.min_words, "max_words": a.max_words,
                   "dedup": "lowercase exact"},
        "items": picked,
    }, ensure_ascii=False, indent=1))
    print(f"\n写出 {out}   {len(picked)} 条")
    for it in picked[:5]:
        print(f"  - {it['prompt'][:90]}")

    print("""
可比性提醒（写在用它之前）：
  这是 relaion（2023-12 下架后的安全过滤重发），分布与 ScaleDiff /
  DemoFusion 当年用的原始 LAION-5B 不完全相同；而且我们也不可能采到
  和他们一样的那 1000 条。所以**绝对 FID 与发表值的偏差，一部分来自
  参考集而非方法**。判据仍按 §7.2 事前定的：我们自己复现的 ScaleDiff
  行是锚 —— 它落在发表值附近，则其余 baseline 可引用发表数字；
  落得远，就三行全部自己跑、只报 A/B 相对变化，并在表注里写明。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
