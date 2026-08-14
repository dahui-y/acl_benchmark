"""残留名词自检：门有没有把重复压力**转嫁**到替代文本剩下的物体上？

由 E4 移植轮的 331 暴露（背景条件字面是 "helicopters"），但问题不在
DemoFusion —— **自家 ScaleDiff 上一样存在**：34 条触发集里，263 的背景
拿到 "the Washington Monument"、583 拿到 "the net of a tennis court"、
1527 拿到 "a door"。我们此前只数主体（statue/chair/cat），**从没数过
这些残留名词**，所以 +0.61 -> +0.10 那份成绩单可能虚高。

做法：对每条样本，取 v1 的替代文本（parti_v1 manifest 的 head +
strip_subject），用 VLM 抽出其中的主要可数名词，再在**基线臂和 v1 臂的
4096 输出**上各数一遍。看差值：
    v1 明显多  -> 压力转嫁坐实 -> v1.1（摘掉所有可数名词）成为必要改进
    持平/更少  -> 转嫁不成立，成绩单如实

    python scalediff_probe/resid_count.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_phrases import strip_subject                # noqa: E402
from vlm_count import VlmCounter, load_img               # noqa: E402

# 纯场景/不可数词：抽出来也没有计数意义，跳过（判为"替代文本干净"）
SCENE = {"mountain", "sky", "beach", "top", "background", "landscape",
         "scene", "view", "field", "ground", "water", "air", "sunset",
         "switchback", "road", "farm", "loch", "silhouette", "moon"}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "parti_hi"))
    ap.add_argument("--v1", default=str(root / "parti_v1"))
    ap.add_argument("--res", type=int, default=4096)
    a = ap.parse_args()

    B, V = Path(a.base), Path(a.v1)
    brows = {json.loads(l)["idx"]: json.loads(l)
             for l in (B / "manifest.jsonl").open()}
    vrows = {json.loads(l)["idx"]: json.loads(l)
             for l in (V / "manifest.jsonl").open()}
    idxs = sorted(set(brows) & set(vrows))
    print(f"两臂共有 {len(idxs)} 条")

    v = VlmCounter()
    outp = V / "vlm_resid.jsonl"
    done = {json.loads(l)["idx"]: json.loads(l)
            for l in outp.open()} if outp.exists() else {}

    def img(d, r):
        f = r["files"].get(str(a.res)) or r["files"].get(a.res)
        return load_img(d / f) if f else None

    mean = lambda x: sum(x) / max(len(x), 1)
    with outp.open("a") as f:
        for i in idxs:
            if i in done:
                continue
            vr = vrows[i]
            head = vr.get("head")
            alt = vr.get("removed") and strip_subject(vr["prompt"], head)[0]
            if not alt:
                continue
            resid = v.subject_of(alt)
            base_w = (resid or "").split()[-1] if resid else ""
            if not resid or base_w in SCENE:
                rec = {"idx": i, "alt": alt, "resid": resid,
                       "skipped": "替代文本无可数名词"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[i] = rec
                print(f"  [{i:>4}] alt={alt[:36]!r:<38} -> {resid!r} 跳过（纯场景）")
                continue
            ib, iv = img(B, brows[i]), img(V, vr)
            if ib is None or iv is None:
                continue
            nb, _ = v.count(ib, resid)
            nv, _ = v.count(iv, resid)
            rec = {"idx": i, "alt": alt, "resid": resid,
                   "n_base": nb, "n_v1": nv,
                   "d": (nv - nb) if None not in (nb, nv) else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[i] = rec
            print(f"  [{i:>4}] {resid:<18} base {nb} -> v1 {nv}"
                  f"   {'**增加**' if rec['d'] and rec['d'] > 0 else ''}")

    recs = [x for x in done.values() if x.get("d") is not None]
    skipped = [x for x in done.values() if x.get("skipped")]
    if not recs:
        print("没有可比条目")
        return 0
    ds = [x["d"] for x in recs]
    up = [x for x in recs if x["d"] > 0]
    print(f"\n== 残留名词自检 ==\n"
          f"  可比 {len(recs)} 条（另有 {len(skipped)} 条替代文本纯场景，跳过）\n"
          f"  差值均值 {mean(ds):+.2f}   增加的 {len(up)} 条   "
          f"减少的 {sum(1 for d in ds if d < 0)} 条")
    for x in sorted(up, key=lambda x: -x["d"])[:10]:
        print(f"    [{x['idx']:>4}] {x['resid']:<16} "
              f"{x['n_base']} -> {x['n_v1']}  (+{x['d']})   alt={x['alt'][:40]!r}")
    print("\n判读：均值明显 >0 且集中在特定样本 -> **压力转嫁坐实**，"
          "\n     v1.1（替代文本摘掉所有可数名词，只留场景词）成为必要改进，"
          "\n     且当前 34 条成绩单需附这一列如实披露；"
          "\n     均值 ≈0 -> 转嫁不成立，成绩单照旧。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
