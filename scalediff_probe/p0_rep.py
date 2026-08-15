"""P0 四臂的 Rep⁺/Dmg⁻ —— 检验"两个失效是同一个病"这条假设。

────────────────────────────────────────────────────────────────────────
这一步要回答什么
────────────────────────────────────────────────────────────────────────
§10.10 第三层（目前仍是**假设**）：分解切断长程上下文，模型用"局部看着
合理"的东西填空 —— 区域空旷时填出**副本**（物体重复），区域有纹理时填出
**糊或胡编**（patch 指标变差）。**同一个病，两个症状。**

可证伪判据（写死）：若真是一个病，一个只改窗口几何的机制应当**同时**动
KIDp 和 Rep⁺。

    同时动   -> 假设升级为证据。这正好补上"效应量小"这个短板 ——
               headline 不再是"KIDp 改善千分之二"，而是"我们指出了分块
               范式的税，并用一个机制同时消掉它的两个症状"。
    只动一个 -> **第三层作废**，退回一二层（仍是一篇把帕累托前沿往上推
               的论文，弱一档，但不死）。

图是现成的（p0 正在生成），所以这条**不额外花生成算力**，只花计数的时间。

────────────────────────────────────────────────────────────────────────
为什么不直接用 `vlm_count.py --armdelta`
────────────────────────────────────────────────────────────────────────
三处对不上：
1. 它要一个目录里一份含 `arm` 字段的 manifest；p0 是**四个目录四份**。
2. 它用**按目录**的 `vlm_subjects.json`。同一个 idx 在不同臂被解析成
   不同主体词，臂间就不可比 —— 这个漂移 §（base_audit）已经踩过一次
   （157 在一处是 'person'(6)、另一处是 'team'(2)）。`do_delta` 后来
   改成全局缓存，`do_armdelta` **没跟上**。这里一律走全局缓存。
3. 它对每个臂都数一遍基图。而 p0 的 1024 基图**四臂逐字节相同**
   （已用 md5 验证，管线在基础阶段 `attnController.disable()`），
   基图只需数一次，省四分之三的计数。

────────────────────────────────────────────────────────────────────────
**先过这一关：LAION caption 抽不抽得出可数主体**
────────────────────────────────────────────────────────────────────────
`vlm_count.py` 文件头明写：*"LAION alt-text 太脏，那批不走 VLM 计数。"*
而 p0 的 prompt 正是 LAION caption。**所以这条线未必走得通，必须先验。**

    --peek      零 GPU。打印全部 caption，按启发式标出可疑的
                （太短 / 纯商品码 / 含尺寸像素串 / 无实义名词）。
                **先人眼扫一遍**，再决定花不花 GPU。
    --subjects  GPU，约 2 分钟。抽主体并落全局缓存，打印供作者核对。
                预注册门槛：**可用主体 >= 60%**，否则这条线判死，
                重复那半边不进论文（只留 patch 三列）。
    （默认）     GPU。数 base(1024) 与 hi(4096 降采样回 1024)，
                出 Rep⁺/Dmg⁻ 表 + 对 npa 的配对比较。

VLM 占 ~16GB，**跟 SDXL 生成抢不了显存**，必须等 p0 跑完再动 GPU 那两档。

    python scalediff_probe/p0_rep.py --peek          # 现在就能跑
    python scalediff_probe/p0_rep.py --subjects      # p0 跑完后
    python scalediff_probe/p0_rep.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ("npa", "shift", "md", "ctx")
LABEL = {"md": "ovl-attn"}          # 目录名沿用 md，报表标真名（§10.15）

# 主体词里出现这些，说明抽出来的是"图片"本身而不是画面里的东西
BAD_SUBJ = {"image", "photo", "picture", "background", "scene", "view",
            "art", "design", "wallpaper", "illustration", "none", "n/a",
            "product", "item", "thing", "object"}
# caption 的脏模式（LAION alt-text 常见）
_PX = re.compile(r"\b\d{3,4}\s?[x×]\s?\d{3,4}\b", re.I)
_CODE = re.compile(r"\b[A-Z0-9]{2,}[-_/][A-Z0-9-_/]{2,}\b")
_URLY = re.compile(r"(https?://|www\.|\.com|\.jpg|\.png)", re.I)
_BOILER = re.compile(
    r"(click to enlarge|zoom|stock photo|royalty[- ]free|shutterstock|"
    r"alamy|getty|free download|for sale|buy now|add to cart)", re.I)


def caption_flags(t):
    """标出可疑，**不过滤** —— 判断权留给人眼（与 real_audit 同一条纪律）。"""
    f = []
    w = t.split()
    if len(w) < 3:
        f.append("太短")
    if _PX.search(t):
        f.append("像素串")
    if _CODE.search(t):
        f.append("商品码")
    if _URLY.search(t):
        f.append("URL/文件名")
    if _BOILER.search(t):
        f.append("图库套话")
    if t.isupper() and len(w) > 1:
        f.append("全大写")
    return f


def load_plan(p0):
    f = Path(p0) / "plan.json"
    if not f.exists():
        sys.exit(f"没有 {f} —— 先跑 p0_run.py（哪怕只 --plan）")
    o = json.loads(f.read_text())
    return o["idx"], o["prompt"]


def do_peek(idxs, pmap):
    print(f"p0 名单 {len(idxs)} 条（LAION caption）。"
          f"**零 GPU，只做人眼预筛。**\n")
    dirty = 0
    for i in idxs:
        t = pmap[str(i)]
        fl = caption_flags(t)
        if fl:
            dirty += 1
        print(f"[{i:>5}] {'⚠ ' + '/'.join(fl) if fl else '  ':<22} {t[:96]}")
    print(f"\n可疑 {dirty}/{len(idxs)} = {dirty/len(idxs):.0%}")
    print("""
判读（写在看之前）：
  可疑 <= 20%  -> LAION caption 够干净，去跑 --subjects
  20~40%       -> 边缘。跑 --subjects，但把可用率门槛当硬闸
  > 40%        -> **这条线当场判死**，重复那半边不进论文，
                  §10.10 第三层保持"未验证的假设"，只留 patch 三列。
注意这只是**启发式**，最终判据是 --subjects 抽出来的主体词能不能用。""")
    return 0


def load_vlm():
    from vlm_count import VLM
    return VLM()


def do_subjects(idxs, pmap, cache_p):
    v = load_vlm()
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    t0 = time.time()
    todo = [i for i in idxs if str(i) not in cache]
    print(f"抽主体：{len(todo)} 条待抽（缓存已有 {len(idxs)-len(todo)}）")
    for n, i in enumerate(todo, 1):
        cache[str(i)] = v.subject_of(pmap[str(i)])
        if n % 20 == 0:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False))
            print(f"\r  {n}/{len(todo)}  {(time.time()-t0)/60:.1f} 分钟",
                  end="", flush=True)
    cache_p.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"\n全局主体缓存 -> {cache_p}\n")

    good = []
    for i in idxs:
        s = (cache.get(str(i)) or "").strip()
        bad = (not s) or len(s.split()) > 2 or s in BAD_SUBJ \
            or any(w in BAD_SUBJ for w in s.split())
        print(f"[{i:>5}] {'✗' if bad else '✓'} {s:<22} <- {pmap[str(i)][:70]}")
        if not bad:
            good.append(i)
    r = len(good) / len(idxs)
    print(f"\n可用主体 {len(good)}/{len(idxs)} = {r:.0%}"
          f"   （门槛 60%，写在跑之前）")
    print("**过了** -> 去掉 --subjects 直接跑计数。" if r >= 0.6 else
          "**没过 -> 这条线判死**：重复那半边不进论文，§10.10 第三层"
          "保持未验证假设，只留 patch 三列。")
    (cache_p.parent / "p0_rep_usable.json").write_text(
        json.dumps({"usable_idx": good, "rate": r}, ensure_ascii=False))
    return 0 if r >= 0.6 else 1


def do_count(p0, idxs, cache_p, limit):
    from vlm_count import load_img
    up = cache_p.parent / "p0_rep_usable.json"
    if not up.exists():
        sys.exit("先跑 --subjects（它写出可用 idx 名单与主体缓存）")
    usable = set(json.loads(up.read_text())["usable_idx"])
    subj = json.loads(cache_p.read_text())

    have = {}
    for a in ARMS:
        m = Path(p0) / a / "manifest.jsonl"
        if m.exists():
            have[a] = {json.loads(l)["idx"]: json.loads(l)
                       for l in m.open() if l.strip()}
    if not have:
        sys.exit(f"{p0} 下没有任何臂的 manifest")
    common = sorted(usable.intersection(*[set(h) for h in have.values()]))
    if limit:
        common = common[:limit]
    print(f"臂 {list(have)}   四臂齐全且主体可用的 idx：{len(common)}")
    if not common:
        return 1

    v = load_vlm()
    outp = Path(p0) / "vlm_rep.jsonl"
    done = {(json.loads(l)["idx"], json.loads(l)["arm"]): json.loads(l)
            for l in outp.open() if l.strip()} if outp.exists() else {}
    base = {}                       # 基图四臂逐字节相同 -> 只数一次
    for k, r in done.items():
        if r.get("n_base") is not None:
            base[k[0]] = r["n_base"]

    t0 = time.time()
    todo = [(i, a) for i in common for a in have if (i, a) not in done]
    print(f"待数 {len(todo)} 个 (idx, arm)"
          f"（基图只数一次，省掉约 {len(common)*(len(have)-1)} 次）")
    with outp.open("a") as f:
        for n, (i, a) in enumerate(todo, 1):
            d = Path(p0) / a
            fl = have[a][i]["files"]
            if i not in base:
                f_lo = fl.get("1024") or fl.get(1024)
                base[i], _ = v.count(load_img(d / f_lo), subj[str(i)])
            fh = max((k for k in fl if str(k).isdigit()), key=lambda k: int(k))
            n_h, _ = v.count(load_img(d / fl[fh]), subj[str(i)])   # 降到 1024
            nb = base[i]
            rec = {"idx": i, "arm": a, "subject": subj[str(i)],
                   "n_base": nb, "n_hi_dn": n_h,
                   "delta": (n_h - nb) if None not in (nb, n_h) else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[(i, a)] = rec
            el = (time.time() - t0) / 60
            print(f"\r  {n}/{len(todo)}  {el:.1f} 分钟  "
                  f"剩约 {el/n*(len(todo)-n):.0f} 分钟", end="", flush=True)
    print()
    report(done, common)
    return 0


def report(done, common):
    """Rep⁺/Dmg⁻ 分解（§3.14d 锁定）+ 对 npa 的配对差。

    为什么不用带符号的均值：负 delta（主体被破坏）会抵消正 delta（重复），
    两个方向相反的失效在一个数里互相掩盖。**两列都要改善才算赢。**
    """
    import numpy as np
    arms = [a for a in ARMS if any(k[1] == a for k in done)]
    per = {a: {i: done[(i, a)]["delta"] for i in common
               if (i, a) in done and done[(i, a)]["delta"] is not None}
           for a in arms}
    ok = sorted(set.intersection(*[set(per[a]) for a in arms])) if arms else []
    print(f"\n有效配对 idx：{len(ok)}")
    if not ok:
        return

    def stat(a):
        d = np.array([per[a][i] for i in ok], float)
        return np.maximum(d, 0).mean(), np.maximum(-d, 0).mean()

    print(f"\n{'臂':<14}{'Rep+ 重复':>12}{'Dmg- 误伤':>12}{'总误差':>10}")
    print("-" * 48)
    for a in arms:
        r, g = stat(a)
        print(f"{LABEL.get(a,a):<14}{r:>12.3f}{g:>12.3f}{r+g:>10.3f}")

    if "npa" not in arms:
        return
    print(f"\n对 npa 的配对差（同一批 {len(ok)} 个 idx，bootstrap 1000 次）")
    print(f"{'臂':<14}{'ΔRep+':>10}{'2σ':>9}{'ΔDmg-':>10}{'2σ':>9}{'判':>8}")
    print("-" * 60)
    rng = np.random.default_rng(0)
    base_d = np.array([per["npa"][i] for i in ok], float)
    for a in arms:
        if a == "npa":
            continue
        d = np.array([per[a][i] for i in ok], float)
        f_r = lambda x, y: np.maximum(x, 0).mean() - np.maximum(y, 0).mean()
        f_g = lambda x, y: np.maximum(-x, 0).mean() - np.maximum(-y, 0).mean()
        gr, gg = f_r(d, base_d), f_g(d, base_d)
        br, bg = [], []
        for _ in range(1000):
            s = rng.integers(0, len(ok), len(ok))
            br.append(f_r(d[s], base_d[s]))
            bg.append(f_g(d[s], base_d[s]))
        sr, sg = 2 * np.std(br), 2 * np.std(bg)
        win = (gr < -sr) and (gg <= sg)      # 重复降了、误伤没变差
        vd = "✅赢" if win else ("❌反" if gr > sr else "…不定")
        print(f"{LABEL.get(a,a):<14}{gr:>+10.3f}{sr:>9.3f}"
              f"{gg:>+10.3f}{sg:>9.3f}{vd:>8}")
    print("""
判据（§10.10 第三层，写在看数字之前）：
  某臂**同时**在 KIDp（std_table 的配对表）和 Rep+（这里）上赢
      -> "两个失效是一个病"从假设变证据，进论文主线；
  只赢 KIDp 不赢 Rep+，或反之
      -> **第三层作废**，退回一二层：只报 patch 三列，重复那半边降为附录。
  Dmg- 变差超过 2σ
      -> 该机制在修重复的同时破坏主体，与 §3.14e 的强度旋钮同型，不要。""")


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=str(root / "p0"))
    ap.add_argument("--peek", action="store_true", help="零 GPU，人眼预筛")
    ap.add_argument("--subjects", action="store_true", help="抽主体 + 可用率闸")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    idxs, pmap = load_plan(a.p0)
    # 全局主体缓存 —— 与 do_delta 同一份，绝不按目录各存（会漂，见文件头）
    cache_p = root / "vlm_subjects_global.json"

    if a.peek:
        return do_peek(idxs, pmap)
    if a.subjects:
        return do_subjects(idxs, pmap, cache_p)
    return do_count(a.p0, idxs, cache_p, a.limit)


if __name__ == "__main__":
    sys.exit(main())
