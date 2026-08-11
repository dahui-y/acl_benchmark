"""空视野比例在 LAION 基图上恒为 0 —— 先定位机制，再改，别猜。

现象：`trigger_select --split tune --limit 20` 出来 p0..p90 全是 0.00，
最大也只有 0.12。真实语料不可能每张图的主体都铺满 16 块，**这是仪器失效**。

嫌疑落在我上一步的设计上：**把整条 caption 喂给 GroundingDINO**。
它是开放词表的，喂一句话就会去 ground 句子里的**每一个**名词短语，
于是有两条完全不同的机制都能把空视野比例压到 0：

  (a) **整图框**。caption 描述的是整个场景（"a traditional easter brunch
      menu"），GroundingDINO 很可能回一个覆盖近乎全图的框。
      一个整图框就填满全部 16 块，evf 直接为 0。
      —— 这条与分块无关，关掉分块也一样。

  (b) **背景名词**。caption 里含 road / sky / wall / table 这类**背景**名词,
      每一块里都能匹配上点什么。而 `detect()` 还会分块再跑一遍
      （tile = width/4，即 256px），**每块都单独送进检测器**，
      于是每块都拿得到框。
      —— 这条关掉分块就会缓解。

两条的修法完全不同，所以必须先分清：

    整图一遍就 evf=0            -> (a)：问题在"框住的是场景不是主体"
    整图一遍 evf 正常、分块后为 0 -> (b)：问题在"分块把背景也数成了内容"
    两条都有                    -> 两处都要改

**更根本的一点（这才是我的概念错误）**：律里的变量是"**不含主体**的视野数"，
不是"什么都没有的视野数"。马路、天空、草地不是主体。把整条 caption 拿去
ground，等于把"主体覆盖率"换成了"任何东西的覆盖率"，后者恒等于 1。

    python scalediff_probe/trigger_debug.py --n 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trigger import empty_view_fraction        # noqa: E402


def head_phrase(caption, preps=("of", "on", "in", "at", "with", "for",
                                "from", "by", "and", "to", "over", "under")):
    """粗糙地取第一个名词短语：截到第一个介词/逗号之前。

    **这不是最终方案**，只是第三个对照组，用来看"只 ground 主体短语"
    能不能把 evf 救回来。真要用得换成正经的名词短语分析器。
    """
    words = caption.replace(",", " , ").split()
    out = []
    for w in words:
        lw = w.lower().strip(".,")
        if lw in preps or lw == ",":
            break
        out.append(w)
        if len(out) >= 6:
            break
    return " ".join(out) if out else caption[:40]


def show(det, im, text, R, label):
    """整图一遍（不分块），把每个框的面积占比打出来。"""
    b, s = det._one(im, text)
    W, H = im.size
    areas = sorted(((x[2] - x[0]) * (x[3] - x[1]) / (W * H)) for x in b)
    evf = empty_view_fraction(b, W, H, R)
    print(f"    {label:<22} 框 {len(b):>3}  evf {evf:.2f}  "
          f"面积占比 " + ("  ".join(f"{v:.2f}" for v in areas[-6:]) or "(无)"))
    return evf, len(b), (max(areas) if areas else 0.0)


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--box-thr", type=float, default=0.30)
    a = ap.parse_args()

    base = Path(a.base)
    rows = [json.loads(l) for l in (base / "manifest.jsonl").open()][:a.n]
    from PIL import Image
    from count_objects import Detector
    det = Detector(box_thr=a.box_thr)

    agg = {"full_whole": [], "head_whole": [], "full_tiled": []}
    for r in rows:
        im = Image.open(base / r["file"]).convert("RGB")
        hp = head_phrase(r["prompt"])
        print(f"\n[{r['idx']}] {r['prompt'][:72]}")
        print(f"    主体短语猜测: 「{hp}」")
        e1, n1, m1 = show(det, im, r["prompt"], a.R, "整图 x 全 caption")
        e2, n2, m2 = show(det, im, hp, a.R, "整图 x 主体短语")
        b, _ = det.detect(im, r["prompt"])
        e3 = empty_view_fraction(b, im.width, im.height, a.R)
        print(f"    {'分块 x 全 caption':<22} 框 {len(b):>3}  evf {e3:.2f}"
              f"   <- trigger_select 现在用的就是这个")
        agg["full_whole"].append((e1, m1))
        agg["head_whole"].append((e2, m2))
        agg["full_tiled"].append((e3, 0))

    def avg(k, i=0):
        v = [x[i] for x in agg[k]]
        return sum(v) / max(len(v), 1)

    print("\n" + "=" * 62)
    print(f"平均 evf   整图x全caption {avg('full_whole'):.2f}   "
          f"整图x主体短语 {avg('head_whole'):.2f}   "
          f"分块x全caption {avg('full_tiled'):.2f}")
    print(f"平均最大框面积占比   全caption {avg('full_whole',1):.2f}   "
          f"主体短语 {avg('head_whole',1):.2f}")
    print("""
判读：
  整图x全caption 的 evf 就已经 ~0，且最大框面积占比接近 1
      -> **(a) 整图框**：ground 的是场景不是主体。分块不是主因。
  整图x全caption 的 evf 正常，分块后掉到 0
      -> **(b) 分块把背景也数成了内容**。
  整图x主体短语 的 evf 明显高于 整图x全caption
      -> 方向对了：**只 ground 主体**能把这个量救回来，
         接下来要解决的是"怎么从任意 caption 稳定地取出主体短语"。
  三个都 ~0
      -> 与 caption 无关，去查 empty_view_fraction 的 5% 交集判据
         和 box_thr 本身。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
