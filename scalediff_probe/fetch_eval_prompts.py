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
import re
import sys
import time
import urllib.request
from pathlib import Path

# ---- 词法筛：哪些 caption 明确声明了可数的视觉物体 ----
#
# **第一版几乎全错，实测 5 个例子错 4 个：**
#   [10 mustang] "10 Great Mustang Movies to Watch"    -> 是 10 部电影
#   [6 bedroom]  "Richmond 6 Piece Bedroom Set"        -> 是 6 件套
#   [1 world]    "Named One Of World's Most Liveable"  -> "one of" 是部分格
# 原因：① 部分格 one of 不是计数；② 量词（piece/pcs/pack/set）后面的名词
# 不是被数的东西；③ 中心词抽错（取了 Mustang，真中心词是 Movies）。
#
# **换判据，不是补规则：能数的只有检测器能检的东西。**
# 中心词必须落在 COCO-80 里 —— 这个词表非任意且有引用：GenEval 的 counting
# 任务用 Mask2Former(COCO)，CountGen 的评测脚本用 YOLOv9e(COCO)。
# 再加单复数一致（n>1 要复数、n=1 要单数）与量词/部分格排除。
#
# 代价：子集会小很多，所以计数指标**另建一个集合**（--count-n），
# 不与随机 1000 条混用 —— 那 1000 条保持无筛选，才和 ScaleDiff 的
# FID/KID/IS 协议可比。CountGen 造 CoCoCount 也是这个思路。

COCO80 = """person bicycle car motorcycle airplane bus train truck boat
bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack
umbrella handbag tie suitcase frisbee snowboard kite skateboard surfboard
bottle cup fork knife spoon bowl banana apple sandwich orange broccoli
carrot pizza donut cake chair couch bed toilet tv laptop mouse remote
keyboard microwave oven toaster sink refrigerator book clock vase
scissors toothbrush""".split()

_IRREG = {"person": "people", "mouse": "mice", "knife": "knives",
          "sandwich": "sandwiches", "bus": "buses", "scissors": "scissors",
          "sheep": "sheep", "broccoli": "broccoli"}


def _plural(w):
    if w in _IRREG:
        return _IRREG[w]
    return w + ("es" if w.endswith(("s", "x", "ch", "sh")) else "s")


SING2CLS = {c: c for c in COCO80}
PLUR2CLS = {_plural(c): c for c in SING2CLS}

# COCO 只有 "person"，但 caption 里写的是 man/woman/child/surfer…
# 不补这些会丢掉大量真实的计数声明（实测 "three little children" 被拒）。
# 只收**明确指人**的词，不收职业泛称之外的模糊词。
_PERSON_S = ["man", "woman", "boy", "girl", "child", "kid", "lady", "guy",
             "baby", "person", "surfer", "hiker", "skier", "rider",
             "player", "worker", "soldier", "dancer", "runner"]
_PERSON_P = {"men": "person", "women": "person", "children": "person",
             "people": "person", "babies": "person", "ladies": "person",
             "persons": "person"}
SING2CLS.update({w: "person" for w in _PERSON_S})
PLUR2CLS.update(_PERSON_P)
PLUR2CLS.update({w + "s": "person" for w in _PERSON_S
                 if w not in ("man", "woman", "child", "person", "lady",
                              "baby")})

# "a single / lone / solitary / sole X" —— 这**正是我们的律针对的句式**
# （孤独主体 + 大场景），必须收，等价于基数 1。
SINGLE_WORDS = {"single", "lone", "solitary", "sole", "only"}

NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# 量词：数字修饰的是它，不是后面的名词。"6 Piece Bedroom Set" 死在这里。
MEASURE = {"piece", "pieces", "pcs", "pc", "pack", "packs", "set", "sets",
           "pair", "pairs", "box", "boxes", "count", "ct", "pk", "lot",
           "bundle", "kit", "inch", "inches", "cm", "mm", "ft", "oz", "lb",
           "kg", "ml", "gb", "mb", "way", "tier", "star", "seat", "door",
           "speed", "layer", "color", "colors", "size", "sizes", "row",
           "pcs/set", "in", "of"}

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMTOK = re.compile(r"^([1-9]|10)$")


# 候选 caption 源，按"与这条线实际使用的评测集的贴近程度"排序。
#
# **证据分三层，强度不同（2026-08-10 第二次更正）：**
#   PixelRush（同线，training-free 高分辨率）原文：
#       "1000 prompts randomly sampled from the LAION/LAION2B aesthetic dataset"
#       —— 明确，且该文开篇即称 "follow the experimental settings of prior methods"
#   ScaleDiff §4.1 原文：
#       "1,000 image-text pairs from the LAION-5B dataset [38]"
#       —— **伞名**，没说子集。
#   DemoFusion：**未能直接核实**（PDF 取不下来，检索未给出原句）。
#
# 包含关系： laion2B-en-aesthetic ⊂ laion2B-en ⊂ LAION-5B
# 所以 ScaleDiff 写 "LAION-5B"、实际用 aesthetic 子集**并不矛盾**，只是用了伞名。
# 而这条线上唯一明确写出子集的那篇写的是 aesthetic。
#
# 我第一版把 relaion 排在首位，漏洞在于：只顺着"原始下架 -> 官方重发"找替代品，
# 没去查**这条线实际用的是哪个子集**。relaion 现在降为退路——它是原始
# LAION-5B 的安全过滤重发（relaion-safe ⊂ relaion-research ⊂ LAION-5B），
# 分布上离 aesthetic 更远。
#
# **不猜哪个门控** —— 逐个真的 HEAD 一下分片，取第一个能下的。
CANDIDATES = [
    "laion/laion2B-en-aesthetic",         # 这条线实际用的（PixelRush 明确写出）
    "laion/relaion2B-en-research",        # 退路：官方重发，最接近原始 LAION-5B
    "laion/relaion2B-en-research-safe",   # 上者真子集，多一层 NSFW 过滤
    "laion/laion-coco",                   # 合成 caption，分布差最远
]

GATED_HELP = """
**所有候选都下不了 —— 这是门控（gating），不是网络。**
判断依据：repo_info 成功（元数据公开）而 resolve 返回 401/403（内容需授权）。
LAION 在 2023-12 下架重发后，这些集合都要登录 + 同意条款。

三步解决：
  1. 按 CANDIDATES 顺序去对应的数据集页面接受条款：
        https://huggingface.co/datasets/laion/laion2B-en-aesthetic  （首选）
        https://huggingface.co/datasets/laion/relaion2B-en-research （退路）
     relaion 系列是 gated access，**要填机构信息 + 同意条款**，
     需审核，不是点一下就通过；
  2. 在 https://huggingface.co/settings/tokens 建一个 read token；
  3. 在服务器上任选其一：
        huggingface-cli login          # 交互粘贴 token
        export HF_TOKEN=hf_xxxxx       # 或直接给环境变量
  然后重跑本脚本。

若无法取得授权，退路（要在骨架 §7.2 里记下口径变化）：
  改用非门控的 caption 源（--repo 指定），并在论文中说明 prompt 分布
  与 ScaleDiff 的 LAION-5B 采样不同 —— 那时绝对 FID 不可比，
  只报 A/B 相对变化。
"""


def declared_cardinality(text, window=3):
    """返回 (基数, COCO 类名) 或 None。

    四条全部满足才采信：
      1. 有数量词（英文数词或 1-10 的数字），且**不是 "one of"**；
      2. 其后 window 个词内出现 COCO-80 名词；
      3. 中间不出现量词（piece/pack/set/...）；
      4. 单复数与基数一致（n=1 单数，n>1 复数）—— 不一致就整条不采信，
         因为那通常说明数字修饰的是别的东西（"6 Piece Bedroom"）。
    """
    toks = _TOKEN.findall(text.lower())
    for i, t in enumerate(toks):
        if t in NUMS:
            n = NUMS[t]
        elif _NUMTOK.match(t):
            n = int(t)
        elif t in SINGLE_WORDS:
            n = 1
        else:
            continue
        if n == 1 and i + 1 < len(toks) and toks[i + 1] == "of":
            continue                                # "one of" 是部分格
        for j in range(i + 1, min(i + 1 + window, len(toks))):
            w = toks[j]
            if w in MEASURE:
                break                               # 数字修饰的是量词
            if n == 1 and w in SING2CLS:
                return 1, SING2CLS[w]
            if n > 1 and w in PLUR2CLS:
                return n, PLUR2CLS[w]
            if (n == 1 and w in PLUR2CLS) or (n > 1 and w in SING2CLS):
                break                               # 单复数不一致，不采信
    return None


# 单次 HTTP range 请求的上限。8 MB 在这条间歇性链路上实测稳定；
# 调大会回到 RemoteDisconnected，调小则请求数太多。
CHUNK = 8 * 1024 * 1024

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfnet import pick_endpoint                        # noqa: E402


def auth_headers():
    """HF 的 token。**门控数据集必须带它** —— relaion 等在 2023 下架重发后
    需要登录 + 同意条款，未授权时 resolve 返回 401（而 repo_info 仍然成功，
    因为元数据是公开的，这一点很容易误判成网络问题）。

    token 来源：huggingface-cli login 写入的缓存，或 HF_TOKEN 环境变量。
    """
    try:
        from huggingface_hub import get_token
        t = get_token()
    except Exception:
        t = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {t}"} if t else {}


def probe_shard(url, hdrs):
    """HEAD 一下这个分片。返回 (ok, 状态码或异常名)。"""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=hdrs)
        with urllib.request.urlopen(req, timeout=30) as r:
            return True, r.status
    except urllib.error.HTTPError as e:
        return False, e.code
    except Exception as e:
        return False, type(e).__name__


class HttpRangeFile(io.RawIOBase):
    """只读、可 seek 的 HTTP 文件对象。pyarrow 靠 seek 读 footer。"""

    def __init__(self, url, size=None, headers=None):
        self.url, self._pos = url, 0
        self.headers = headers or {}
        self._size = size if size is not None else self._head_size()

    def _head_size(self):
        req = urllib.request.Request(self.url, method="HEAD",
                                     headers=self.headers)
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

    def _chunk(self, start, end):
        """取 [start, end] 这一小段，带退避重试。"""
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    self.url, headers={**self.headers,
                                       "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 2 ** attempt
                print(f"\n  range [{start}-{end}] 失败({type(e).__name__})，"
                      f"{wait}s 后重试 {attempt+1}/5", flush=True)
                time.sleep(wait)

    def read(self, n=-1):
        """**必须分块。** pyarrow 读一个 column chunk 可能一次要几百 MB，
        当成单个 HTTP range 发出去，连接撑不住就 RemoteDisconnected，
        而重试重发同样的巨大请求，必然继续失败（实测就是这么挂的）。
        切成 CHUNK 大小的小段，掉线只损失一块，且可以退避重试。
        """
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size) - 1
        out, cur, total = [], self._pos, end - self._pos + 1
        while cur <= end:
            stop = min(cur + CHUNK - 1, end)
            out.append(self._chunk(cur, stop))
            cur = stop + 1
            if total > CHUNK:
                print(f"\r  取数据 {(cur - self._pos) / 2**20:6.1f} / "
                      f"{total / 2**20:.1f} MB", end="", flush=True)
        if total > CHUNK:
            print()
        buf = b"".join(out)
        self._pos += len(buf)
        return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="不指定则按 CANDIDATES 顺序逐个探测，取第一个能下载的")
    ap.add_argument("--n", type=int, default=1000, help="eval split 的大小")
    ap.add_argument("--tune", type=int, default=200,
                    help="**不相交的调参 split**：门阈值 τ、检测工作点等一切"
                         "还需要标定的东西只许在这上面定。取数时就切开、"
                         "写进 JSON —— 数据落地后再切会有'看过才切'的嫌疑。")
    ap.add_argument("--count-n", type=int, default=400,
                    help="**独立的计数集大小**（tune+eval 合计，同样超采样）。"
                         "计数指标不与随机 1000 条混用：那 1000 条必须保持"
                         "无筛选才和 ScaleDiff 的 FID 协议可比，而计数只在"
                         "'计数有定义'的 prompt 上才有意义。CountGen 造"
                         "CoCoCount 也是这个思路。")
    ap.add_argument("--max-batches", type=int, default=200,
                    help="最多扫这么多批（每批 8192 行）来凑够计数集")
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="LAION 存的是图片 URL 不是图片，多年后三到五成已失效。"
                         "ScaleDiff 要 image-text pair（caption 生成、真图算 FID），"
                         "所以按 (n + tune) 的这么多倍攒候选，下图时留下成功的。")
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
    hdrs = auth_headers()
    print(f"HF token: {'有' if hdrs else '**无**（门控数据集会 401）'}")

    # **repo_info 成功 ≠ 能下载。** 门控仓库的元数据公开、内容需授权，
    # 所以必须逐个真的 HEAD 一下分片，取第一个能下的。
    cands = [a.repo] if a.repo else CANDIDATES
    repo = url = shards = None
    for cand in cands:
        try:
            info = api.repo_info(cand, repo_type="dataset")
        except Exception as e:
            print(f"  {cand:<45} repo_info 失败 {type(e).__name__}")
            continue
        sh = sorted(x.rfilename for x in (info.siblings or [])
                    if x.rfilename.endswith(".parquet"))
        if not sh:
            print(f"  {cand:<45} 没有 parquet")
            continue
        u = f"{ep}/datasets/{cand}/resolve/main/{sh[0]}"
        ok, code = probe_shard(u, hdrs)
        print(f"  {cand:<45} {len(sh):>3} 片  分片可下载: "
              + ("是" if ok else f"否({code})"))
        if ok:
            repo, url, shards = cand, u, sh
            break
    if repo is None:
        print(GATED_HELP)
        return 1
    print(f"\n用 {repo}   分片 {shards[0]}"
          f"（共 {len(shards)} 片，只读第一片的第一个 row group）")

    f = HttpRangeFile(url, headers=hdrs)
    print(f"  整片 {f._size / 2**20:.0f} MB —— 不下载，只 range 读")
    pf = pq.ParquetFile(f)
    cols = pf.schema_arrow.names
    print(f"  列: {cols}")

    tcol = next((c for c in cols if c.upper() in ("TEXT", "CAPTION")), None)
    ucol = next((c for c in cols if c.upper() == "URL"), None)
    if tcol is None:
        print(f"  找不到 caption 列，实际列见上"); return 1

    want = [c for c in (tcol, ucol) if c]
    rg = pf.metadata.row_group(0)
    print(f"  第一个 row group {rg.num_rows} 行 / "
          f"{rg.total_byte_size / 2**20:.0f} MB（压缩前）")

    # **流式读，够了就停。** read_row_group(0) 会把整个 row group 拉下来
    # （这些分片单片 3.4 GB，一个 row group 就几百 MB），而我们只要 3000 条
    # caption —— 用 iter_batches 边读边筛，攒够立刻 break。
    # **两个 split 都要超采样**：need 只按 eval 那 1000 算是漏了 tune 的 200，
    # 实际倍率会变成 2.5x 而不是写好的 3x。
    need = int((a.n + a.tune) * a.oversample)
    count_need = int(a.count_n * a.oversample)
    seen, pool, cpool = set(), [], []
    nread = nbatch = 0
    for batch in pf.iter_batches(batch_size=8192, columns=want):
        texts = batch.column(tcol).to_pylist()
        urls = (batch.column(ucol).to_pylist() if ucol
                else [None] * len(texts))
        nread += len(texts)
        nbatch += 1
        for t, u in zip(texts, urls):
            if not t:
                continue
            t = " ".join(t.split())
            w = len(t.split())
            if not (a.min_words <= w <= a.max_words):
                continue
            k = t.lower()
            if k in seen:                               # 去重：LAION 里重复很多
                continue
            seen.add(k)
            card = declared_cardinality(t)
            rec = {"prompt": t, "url": u,
                   "card": card[0] if card else None,
                   "subject": card[1] if card else None}
            if len(pool) < need:
                pool.append(rec)
            # **计数集独立收集**，不从 pool 里挑 —— 从 pool 里挑会让
            # 随机 1000 条和计数集重叠，两个集合的独立性就没了。
            if card and len(cpool) < count_need:
                cpool.append(rec)
        print(f"\r  已读 {nread} 行 -> 随机池 {len(pool)}/{need}   "
              f"计数池 {len(cpool)}/{count_need}", end="", flush=True)
        if len(pool) >= need and len(cpool) >= count_need:
            break
        if nbatch >= a.max_batches:
            print(f"\n  扫到 {a.max_batches} 批上限就停了")
            break
    print()

    rng = random.Random(a.seed)
    rng.shuffle(pool)
    rng.shuffle(cpool)

    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    out = Path(a.out) if a.out else root / "eval_prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def split_and_write(items, n_eval, n_tune, path, what):
        """**先切 split 再谈别的。** 切分在取数时完成、写进 JSON，
        早于任何标定 —— 数据落地后再切会有'看过才切'的嫌疑。"""
        r = n_tune / (n_tune + n_eval)
        k = int(len(items) * r)
        for i, it in enumerate(items):
            it["split"] = "tune" if i < k else "eval"
        path.write_text(json.dumps({
            "repo": repo, "shard": shards[0], "row_group": 0,
            "seed": a.seed, "n_eval": n_eval, "n_tune": n_tune,
            "oversample": a.oversample, "what": what,
            "protocol": "ScaleDiff §4.1: 1000 LAION image-text pairs; "
                        "FID/KID/IS vs the real images of those same pairs",
            "split_rule": "取数时按 tune/(tune+eval) 比例切，早于任何标定",
            "filter": {"min_words": a.min_words, "max_words": a.max_words,
                       "dedup": "lowercase exact"},
            "items": items,
        }, ensure_ascii=False, indent=1))
        print(f"  {what:<10} -> {path.name}   {len(items)} 条候选"
              f"（tune {k} / eval {len(items)-k}，目标 {n_tune}/{n_eval}）")

    print()
    split_and_write(pool, a.n, a.tune, out, "随机集")
    split_and_write(cpool, int(a.count_n * a.n / (a.n + a.tune)),
                    int(a.count_n * a.tune / (a.n + a.tune)),
                    out.parent / "count_prompts.json", "计数集")
    print("  **eval split 在方法冻结前一次都不许回看。**")

    print("\n随机集样例（FID/KID/IS/CLIP 用这个，不筛选）：")
    for it in pool[:4]:
        print(f"  - {it['prompt'][:88]}")
    print("计数集样例（计数指标只用这个）：")
    for it in cpool[:8]:
        print(f"  - [{it['card']} {it['subject']}] {it['prompt'][:74]}")

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
