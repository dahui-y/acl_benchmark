"""窗口接缝探针：直接测失效本身，不测它的代理。零 GPU。

────────────────────────────────────────────────────────────────────────
为什么要这个
────────────────────────────────────────────────────────────────────────
2026-08-15 四臂中途看图：**肉眼看不出 npa 与 ctx 的质量差别**。
但看图时意识到，四个臂对"窗口接缝"有**互不相同**的预测：

  **`ctx` 根本不改 query 网格**（只放大 K/V），所以若接缝来自 query
  分块，ctx 修不掉；能修的是 `shift`（网格逐层随机化）与 `ovl-attn`
  （重叠平均）。

而边界位置是**算出来的，不是找出来的** —— 先预测后验证，不是事后找规律。

────────────────────────────────────────────────────────────────────────
边界周期的推导（4096² / SDXL / ScaleDiff NPA）
────────────────────────────────────────────────────────────────────────
VAE /8：4096 图 -> 512×512 latent，故 height_scale s = 4。

  down_blocks.1 / up_blocks.1：特征图 = latent/2 = 256。ws=64 -> h=4×64=256 ✓
      query 块边长 p2 = 32 特征；每特征 = 2 latent = **16 px** -> **512 px**
  down_blocks.2 / mid / up_blocks.0：特征图 = latent/4 = 128。ws=32 -> h=128 ✓
      p2 = 16 特征；每特征 = 4 latent = **32 px**            -> **512 px**

两级折算到像素空间同为 512 -> 4096² 上是 8×8 网格，边界在 511|512 之间。

────────────────────────────────────────────────────────────────────────
v1 在真实图上失效，v2 怎么修的（2026-08-15，同日）
────────────────────────────────────────────────────────────────────────
**v1**：|梯度| 沿另一轴平均成一维剖面 -> 按 512 折叠 -> 看相位 511 是否凸起。
合成 1/f 平稳纹理上工作正常（无接缝时对比度 0.016，注入接缝后峰值精确
落在 511）。

**真实图上崩了**：四个臂的折叠对比度全是 0.38~0.41（合成噪声地板的
二十几倍），而峰值位置随机（命中率 0~8%，纯偶然是 0.4%）。
病因：**真实照片非平稳** —— 画面中间有主体，中间那几列梯度系统性高，
折叠只平均 8 个位置，压不住这种慢变趋势。

> **我的阳性对照有缺陷**：只做了"合成图 ± 接缝"，
> **漏了"非平稳 + 无接缝"这一档**，而恰恰是它会暴露这个失效模式。
> 教训：阴性对照必须覆盖真实数据的主要干扰源，不能只用干净合成图。

（中途还试过频域谐波梳，也不行：窗口里只有 8 个周期，频率分辨率太差，
谱线被低频内容淹掉，注入 sd=8 的接缝都测不出来。已弃。）

**v2 两处改动**：
1. 折叠前用移动中位数（窗宽 3×period）**去趋势**，扣掉慢变部分；
2. 统计量换成稳健 z 分数 `z = (剖面 − median) / (1.4826·MAD)`
   （去趋势后剖面近似零均值，"峰值/中位数"没意义了）。

**另加一层不依赖任何假设的对照：多周期扫描。**
同时在 384/448/480/**512**/544/576/640 上算。非平稳性对所有周期一视同仁，
真接缝只应在 512 上冒出来。**512 不特殊 = 没有 query 网格接缝。**
这条还能证伪我自己的推导：若 512 算错了，别的周期会亮。

    python scalediff_probe/seam_probe.py --selftest    # 四档对照
    python scalediff_probe/seam_probe.py --limit 12
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ARMS = ("npa", "shift", "md", "ctx")
LABEL = {"md": "ovl-attn"}
PERIOD = 512          # 推导值，见文件头 —— **不是拟合出来的**
SCAN = [384, 448, 480, 512, 544, 576, 640]


def gray(p):
    if isinstance(p, (str, Path)):
        return np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    return np.asarray(p, dtype=np.float32)


def _movmed(x, w):
    """移动中位数，边缘按 edge 填充。用来扣掉慢变趋势。"""
    w = w | 1
    h = w // 2
    xp = np.pad(x, h, mode="edge")
    v = np.lib.stride_tricks.sliding_window_view(xp, w)
    return np.median(v, axis=-1)


def phase_profile(a, axis, period=PERIOD, detrend=True):
    """|梯度| 沿另一轴平均 -> 去趋势 -> 按 period 折叠。"""
    g = np.abs(np.diff(a, axis=axis))
    prof = g.mean(axis=1 - axis).astype(np.float64)
    if detrend:
        prof = prof - _movmed(prof, 3 * period)
    n = (len(prof) // period) * period
    if n < period:
        return None
    return prof[:n].reshape(-1, period).mean(0)


def seam_stats(a, period=PERIOD, detrend=True):
    """稳健 z 分数。**不假设相位**：同时报 z_max 及其位置、与 z_pred。

    预测相位 = period − 1（块边界在 P−1|P 之间，np.diff 把索引前移半格）。
    """
    out = {}
    for ax, name in ((1, "x"), (0, "y")):
        pf = phase_profile(a, ax, period, detrend)
        if pf is None:
            out[name] = None
            continue
        med = float(np.median(pf))
        mad = float(np.median(np.abs(pf - med))) * 1.4826
        if mad <= 0:
            out[name] = None
            continue
        z = (pf - med) / mad
        j = int(np.argmax(z))
        out[name] = {"z_max": float(z[j]), "phase": j,
                     "dist": int(min((j - (period - 1)) % period,
                                     ((period - 1) - j) % period)),
                     "z_pred": float(z[period - 1])}
    return out


def z_both(a, period=PERIOD):
    st = seam_stats(a, period)
    if not (st["x"] and st["y"]):
        return None
    return (st["x"]["z_pred"] + st["y"]["z_pred"]) / 2


# ══════════════════════════════════════════════════════════════════════
def _texture(m, rng, aniso=False):
    """1/f 纹理。aniso=True 再乘一个平滑包络，模拟真实照片的非平稳。"""
    f = rng.normal(size=(m, m))
    F = np.fft.fftshift(np.fft.fft2(f))
    yy, xx = np.mgrid[-m // 2:m // 2, -m // 2:m // 2]
    r = np.sqrt(xx ** 2 + yy ** 2) + 1
    b = np.real(np.fft.ifft2(np.fft.ifftshift(F / r)))
    b = 128 + 60 * b / (b.std() + 1e-9)
    if aniso:
        # 中间一个"主体"：局部对比度显著更高 —— 这正是压垮 v1 的东西
        g = np.exp(-(((xx / (m * 0.18)) ** 2 + (yy / (m * 0.18)) ** 2)))
        b = 128 + (b - 128) * (0.35 + 2.2 * g)
    return np.clip(b, 0, 255)


def _seam(b, period, sd, rng):
    nb = b.shape[0] // period
    off = rng.normal(0, sd, (nb, nb))
    return np.clip(b + np.kron(off, np.ones((period, period))), 0, 255)


def selftest(m=4096, period=PERIOD):
    """四档对照。**第三档是 v1 缺的那一档，也是 v2 存在的理由。**"""
    rng = np.random.default_rng(1)
    flat = _texture(m, np.random.default_rng(0))
    aniso = _texture(m, np.random.default_rng(0), aniso=True)

    cases = [
        ("① 平稳，无接缝", flat, False),
        ("② 平稳，接缝 sd=8", _seam(flat, period, 8, rng), True),
        ("③ **非平稳**，无接缝", aniso, False),
        ("④ 非平稳，接缝 sd=8", _seam(aniso, period, 8, rng), True),
    ]
    print(f"对照，{m}²，周期 {period}\n")
    print(f"{'档':<24}{'z_pred':>9}{'z_max':>8}{'phase':>7}{'命中':>6}")
    print("-" * 56)
    ok, got = True, []
    for name, img, want in cases:
        st = seam_stats(img, period)["x"]
        print(f"{name:<24}{st['z_pred']:>9.2f}{st['z_max']:>8.2f}"
              f"{st['phase']:>7d}{'✓' if st['dist'] <= 2 else '✗':>6}")
        got.append((name, st, want))
    print()
    for name, st, want in got:
        good = (st["z_pred"] > 4 and st["dist"] <= 2) if want \
            else abs(st["z_pred"]) < 3
        ok &= good
        print(f"  {'✅' if good else '❌'} {name}："
              + ("应测出接缝且落在预测相位" if want
                 else "应无假阳性（|z_pred| < 3）"))

    print("\n多周期扫描（应当只有 512 亮）：")
    print(f"{'档':<24}" + "".join(f"{p:>8}" for p in SCAN))
    for name, img, _ in cases:
        print(f"{name:<24}"
              + "".join(f"{z_both(img, p):>8.2f}" for p in SCAN))
    print("\n" + ("**对照全过。**" if ok else
                  "**没过 —— 探针不灵，真实图上的读数不算数。**"))
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════
def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=str(root / "p0"))
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--periods", type=int, nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    p0 = Path(a.p0)
    have = {}
    for m in ARMS:
        fs = ({int(p.stem.split("_")[0]): p
               for p in (p0 / m).glob(f"*_{a.res}.png")}
              if (p0 / m).exists() else {})
        if fs:
            have[m] = fs
    if len(have) < 2:
        sys.exit(f"{p0} 下可比的臂不足（找到 {list(have)}）")
    common = sorted(set.intersection(*[set(v) for v in have.values()]))
    if a.idx:
        common = [i for i in common if i in a.idx]
    common = common[:a.limit]
    periods = sorted(set((a.periods or SCAN) + [PERIOD]))
    print(f"臂 {list(have)}   共同 idx {len(common)} 张")
    print(f"多周期扫描 {periods} —— **512 是推导出的预测值，其余是对照**。"
          f"\n非平稳性对所有周期一视同仁；真接缝只应在 512 上冒出来。\n")
    if not common:
        return 1

    res = {m: {p: [] for p in periods} for m in have}
    hit = {m: [] for m in have}
    for k, i in enumerate(common, 1):
        for m in have:
            g = gray(have[m][i])
            for p in periods:
                v = z_both(g, p)
                if v is not None:
                    res[m][p].append(v)
            st = seam_stats(g, PERIOD)
            if st["x"] and st["y"]:
                hit[m] += [st["x"]["dist"] <= 2, st["y"]["dist"] <= 2]
        print(f"\r  {k}/{len(common)}", end="", flush=True)
    print("\n")

    print(f"{'臂':<12}" + "".join(
        f"{('*' if p == PERIOD else '') + str(p):>9}" for p in periods)
        + f"{'峰值命中':>10}")
    print("-" * (12 + 9 * len(periods) + 10))
    for m in have:
        print(f"{LABEL.get(m,m):<12}"
              + "".join(f"{np.mean(res[m][p]):>9.2f}" for p in periods)
              + f"{np.mean(hit[m]):>9.0%}")
    print(f"\n  * = 由 NPA 的 query 块尺寸推导出的预测周期"
          f"（命中的偶然率 = {5/PERIOD:.1%}）")

    print(f"\n512 相对对照周期的突出度（512 的 z − 其余的中位数）")
    verdict = {}
    for m in have:
        others = [np.mean(res[m][p]) for p in periods if p != PERIOD]
        d = np.mean(res[m][PERIOD]) - np.median(others)
        verdict[m] = d
        print(f"  {LABEL.get(m,m):<12}{d:>+7.2f}"
              + ("   <- 512 明显突出，接缝存在" if d > 1.0 else
                 "   <- 512 不特殊，**该臂没有可测的 query 网格接缝**"))

    base_m = "npa" if "npa" in res else list(res)[0]
    print(f"\n对 {base_m} 的配对差 @ {PERIOD}（bootstrap 1000 次）")
    rng = np.random.default_rng(0)
    b = np.array(res[base_m][PERIOD])
    for m in have:
        if m == base_m:
            continue
        v = np.array(res[m][PERIOD])
        n = min(len(v), len(b))
        d = v[:n] - b[:n]
        bs = [d[rng.integers(0, n, n)].mean() for _ in range(1000)]
        s2 = 2 * np.std(bs)
        vd = "✅ 接缝减弱" if d.mean() < -s2 else (
            "❌ 接缝加重" if d.mean() > s2 else "…不定")
        print(f"  {LABEL.get(m,m):<12}Δ={d.mean():>+7.2f}  2σ={s2:.2f}   {vd}")

    print(f"""
判读（写在看数字之前）：
  **先看 npa 的突出度那一行。**
    <= 1.0 -> 512 不特殊 = NPA 上**没有可测的 query 网格接缝**。
              "边界伪影"这条当场否掉，§10.9（他们那个没启用的开关）
              连带降级为无关紧要，`shift` 臂失去意义。
    >  1.0 -> 接缝存在且位置与推导一致，再看四条预测：
                shift    应消失  （网格逐层随机化）
                ovl-attn 应减弱  （重叠平均）
                ctx      应仍在  （只放大 K/V，网格不变）
              四条全中 -> 分解的代价被拆成两个可分离机制，各有对症干预。
              不成模式 -> 我对机制的理解错了，**重新想，不许事后编解释**。""")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"periods": periods, "idx": common,
             "z": {m: {str(p): list(map(float, v)) for p, v in r.items()}
                   for m, r in res.items()},
             "prominence": {m: float(v) for m, v in verdict.items()}},
            ensure_ascii=False, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
