"""demo_e4 的臂间计数：**每个 tag 可以数多个物体**。

为什么要数多个：331 暴露了一个真问题 —— 门把主体（Statue）从背景条件里
摘掉后，替代文本剩下的是 "helicopters"，于是背景被告知"画直升机"。
只数 statue 会显示"我们赢了"，只有把 **helicopter 也数一遍**，才知道
重复压力是不是被转嫁了。这是方法的诚实自检，不是可选项。

    python scalediff_probe/demo_count.py --spec 00331:"Statue of Liberty",helicopter
    python scalediff_probe/demo_count.py            # 用内置 spec
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_count import VlmCounter, load_img              # noqa: E402

# tag -> 要数的物体清单（主体 + 替代文本里残留的物体）
SPEC = {
    "smoke331": ["Statue of Liberty", "helicopter"],
    "00331": ["Statue of Liberty", "helicopter"],
    "01059": ["pickup truck"],
    "01527": ["cat", "door"],
    "00715": ["house"],
    "00738": ["truck", "geyser"],
    "00583": ["rocking chair", "tennis net"],
}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(root / "demo_e4"))
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--spec", nargs="*", default=None,
                    help='形如 00331:"Statue of Liberty",helicopter')
    a = ap.parse_args()

    spec = dict(SPEC)
    for s in (a.spec or []):
        tag, objs = s.split(":", 1)
        spec[tag] = [o.strip().strip('"') for o in objs.split(",")]

    d = Path(a.dir)
    rows = [json.loads(l) for l in (d / "manifest.jsonl").open()]
    rows = [r for r in rows if r["size"] >= a.res]
    if not rows:
        sys.exit(f"{d} 里没有 >= {a.res} 的行")

    v = VlmCounter()
    outp = d / f"vlm_count_{a.res}.jsonl"
    done = {(x["tag"], x["arm"], x["obj"]): x
            for x in map(json.loads, outp.open())} if outp.exists() else {}

    with outp.open("a") as f:
        for r in rows:
            objs = spec.get(r["tag"])
            if not objs:
                print(f"[{r['tag']}] 没有 spec，跳过（用 --spec 指定要数什么）")
                continue
            fn = r["files"].get(str(a.res)) or r["files"].get(a.res)
            if not fn:
                continue
            im = load_img(d / fn)
            for obj in objs:
                key = (r["tag"], r["arm"], obj)
                if key in done:
                    continue
                n, raw = v.count(im, obj)
                rec = {"tag": r["tag"], "arm": r["arm"], "obj": obj,
                       "n": n, "raw": raw}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[key] = rec
                print(f"  [{r['tag']}] {r['arm']:<5} {obj:<20} {n}")

    # 汇总：同 tag 同物体，各臂并排；标出"转嫁"信号
    print(f"\n{'tag':<10}{'object':<22}", end="")
    arms = sorted({k[1] for k in done})
    for arm in arms:
        print(f"{arm:>10}", end="")
    print("   判读")
    for tag in sorted({k[0] for k in done}):
        for obj in sorted({k[2] for k in done if k[0] == tag}):
            vals = {arm: done.get((tag, arm, obj), {}).get("n") for arm in arms}
            print(f"{tag:<10}{obj:<22}", end="")
            for arm in arms:
                print(f"{str(vals.get(arm, '-')):>10}", end="")
            b, g = vals.get("base"), vals.get("v1")
            if None in (b, g):
                note = ""
            elif g < b:
                note = "  门减少了它 ✓"
            elif g == b:
                note = "  持平"
            else:
                note = "  **门增加了它 —— 重复压力转嫁的证据**"
            print(note)
    print("\n判读要点：主体（被门摘掉的那个）应减少；替代文本里残留的物体"
          "\n若显著增加，说明压力被转嫁 -> v1.1 需摘掉所有可数名词。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
