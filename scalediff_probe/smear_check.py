"""拖影 = 成不了形的重复？把律从图像级下放到 tile 级验一次。

观察（09_crowd s77 的 (2048,1024) 裁块）：
  基图  一条光滑的水泥匝道，表面干净；
  基线  匝道上覆盖一大片**有方向性的条纹拖影**；
  我们  匝道干净很多，卡车清晰成形，拖影减少但未消失。
基图没有、4096 有 -> 外推阶段生成的；而且**有方向性**。

假设（要验的）：那是**被结构引导压住的重复**。
  受限视野里只有光滑水泥，却收到完整 prompt "…rush hour, many cars"，
  于是想画车；但 Structure Guidance 每步把低频拉回平坦的匝道参考
  （refine(latents_LFM, pred_x0, ...)），车的轮廓属于低频，被锁死。
  画不出完整的车 -> 只在高频留下车的痕迹 = 一团有方向的拖影。

  结构锁定弱 -> 完整幻影（多一只鹰）；锁定强 -> 拖影（幻影成不了形）。
  **同一个机制的两种外观。物体计数器只数得到前者。**

判别量：**各向异性**。真实纹理（沥青颗粒、树冠）各向同性；拖影不是。
  用结构张量：对 tile 的高通残差算 J=[[<gx²>,<gxgy>],[<gxgy>,<gy²>]]，
  各向异性 = (λ1-λ2)/(λ1+λ2)。0 = 完全各向同性，1 = 纯方向性。

律下放到 tile 级：基图上这个 tile 里有没有主体框，就是律的自变量本身。
**23040 个 tile 比 90 张图的统计力强两个量级。**

预注册的预测（写在跑之前）：
  P1  基图无主体的 tile，4096 上的高频增益 > 有主体的 tile
      （没有东西的地方被凭空填了更多）
  P2  基图无主体的 tile，各向异性 > 有主体的 tile（拖影集中在那里）
  P3  我们这版在"基图无主体"的 tile 上，高频增益与各向异性都低于基线
      （抑制了那个驱动力）
  三条全过 -> 假设成立，拖影写成"重复的亚临床形态"，进正文；
  P3 不过   -> 拖影与我们的介入无关，退回 limitation 一句话。

    python scalediff_probe/smear_check.py
    python scalediff_probe/smear_check.py --cat lone crowd
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector                     # noqa: E402
from detail_check import hf_ratio                      # noqa: E402

TILE = 256
LO, HI = 0.25, 0.5


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def files(rec):
    return (rec["files"][str(min(int(k) for k in rec["files"]))],
            rec["files"][str(max(int(k) for k in rec["files"]))])


def aniso(g):
    """结构张量各向异性。0 = 各向同性纹理，1 = 纯方向性拖影。

    先减掉局部均值（去低频），否则大尺度亮度梯度会主导方向。
    """
    g = g - g.mean()
    gy, gx = np.gradient(g)
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    tr = jxx + jyy
    if tr < 1e-12:
        return 0.0
    d = np.sqrt(max((jxx - jyy) ** 2 + 4 * jxy ** 2, 0.0))
    return float(d / tr)


def grids(img, tile):
    g = np.asarray(img.convert("L"), dtype=np.float64)
    H, W = g.shape
    ny, nx = H // tile, W // tile
    hf = np.zeros((ny, nx)); an = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            t = g[i*tile:(i+1)*tile, j*tile:(j+1)*tile]
            hf[i, j] = hf_ratio(t, LO, HI)
            an[i, j] = aniso(t)
    return hf, an


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--cat", nargs="*", default=None)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MA, MB = mani(a.base), mani(a.new)
    keys = sorted(set(A) & set(B) & set(MA) & set(MB))
    if a.cat:
        keys = [k for k in keys if A[k]["cat"] in a.cat]

    det = Detector()
    # 累积器：[有主体, 无主体] x 各量
    acc = {(s, q): [] for s in ("subj", "nosubj")
           for q in ("gain_a", "gain_b", "an_a", "an_b", "hf_base")}
    nrow = 0
    for k in keys:
        lo_f, hi_f = files(MA[k])
        _, hi_b = files(MB[k])
        base = Image.open(Path(a.base) / lo_f).convert("RGB")
        boxes, _ = det.detect(base, A[k]["subject"])
        ia = Image.open(Path(a.base) / hi_f).convert("RGB")
        ib = Image.open(Path(a.new) / hi_b).convert("RGB")
        scale = ia.width / base.width

        hb, _ = grids(base.resize(ia.size, Image.BICUBIC), a.tile)
        ha, aa = grids(ia, a.tile)
        hbb, ab = grids(ib, a.tile)

        ny, nx = ha.shape
        # tile 里有没有主体（用基图框，×scale 到 4096 坐标）
        has = np.zeros((ny, nx), bool)
        for bx in boxes:
            x0, y0, x1, y1 = [float(v) * scale for v in bx]
            for i in range(ny):
                for j in range(nx):
                    tx0, ty0 = j * a.tile, i * a.tile
                    ix = max(0, min(x1, tx0 + a.tile) - max(x0, tx0))
                    iy = max(0, min(y1, ty0 + a.tile) - max(y0, ty0))
                    if ix * iy > 0.05 * a.tile * a.tile:
                        has[i, j] = True

        for i in range(ny):
            for j in range(nx):
                s = "subj" if has[i, j] else "nosubj"
                acc[(s, "gain_a")].append(ha[i, j] - hb[i, j])
                acc[(s, "gain_b")].append(hbb[i, j] - hb[i, j])
                acc[(s, "an_a")].append(aa[i, j])
                acc[(s, "an_b")].append(ab[i, j])
                acc[(s, "hf_base")].append(hb[i, j])
        nrow += 1
        print(f"\r{nrow}/{len(keys)}  {k[0]:02d}_{A[k]['cat']}_s{k[1]}    ",
              end="", flush=True)
        del ia, ib, base
    print()

    def m(s, q):
        v = acc[(s, q)]
        return float(np.mean(v)) if v else float("nan")

    n_s, n_n = len(acc[("subj", "an_a")]), len(acc[("nosubj", "an_a")])
    print(f"\ntile 总数 {n_s + n_n}   有主体 {n_s}   无主体 {n_n}\n")
    print(f"{'':<14}{'高频增益 基线':>14}{'高频增益 我们':>14}"
          f"{'各向异性 基线':>14}{'各向异性 我们':>14}")
    print("-" * 72)
    for s, name in (("subj", "基图有主体"), ("nosubj", "基图无主体")):
        print(f"{name:<14}{m(s,'gain_a'):>14.4f}{m(s,'gain_b'):>14.4f}"
              f"{m(s,'an_a'):>14.4f}{m(s,'an_b'):>14.4f}")

    p1 = m("nosubj", "gain_a") > m("subj", "gain_a")
    p2 = m("nosubj", "an_a") > m("subj", "an_a")
    p3 = (m("nosubj", "gain_b") < m("nosubj", "gain_a")
          and m("nosubj", "an_b") < m("nosubj", "an_a"))
    print(f"\n预注册预测：")
    print(f"  P1 无主体 tile 的高频增益更大：{'过' if p1 else '**没过**'}"
          f"  ({m('nosubj','gain_a'):.4f} vs {m('subj','gain_a'):.4f})")
    print(f"  P2 无主体 tile 各向异性更高：{'过' if p2 else '**没过**'}"
          f"  ({m('nosubj','an_a'):.4f} vs {m('subj','an_a'):.4f})")
    print(f"  P3 我们在无主体 tile 上两项都更低：{'过' if p3 else '**没过**'}"
          f"  (增益 {m('nosubj','gain_b'):.4f} vs {m('nosubj','gain_a'):.4f}，"
          f"各向异性 {m('nosubj','an_b'):.4f} vs {m('nosubj','an_a'):.4f})")
    print("""
读法：
  三条全过 -> 拖影是**重复的亚临床形态**：受限视野里没有主体却收到完整
     prompt，想画物体但低频被 Structure Guidance 锁住，成不了形，只在
     高频留下方向性痕迹。同一机制两种外观，**物体计数器只数得到完整的那种**。
     这条进正文，并且把律从图像级推到 tile 级（23040 个样本）。
  P3 没过 -> 拖影与我们的介入无关，退回 limitation 一句话，不要硬讲。""")


if __name__ == "__main__":
    main()
