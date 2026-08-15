"""窗口接缝探针：直接测失效本身，不测它的代理。零 GPU。

────────────────────────────────────────────────────────────────────────
为什么要这个
────────────────────────────────────────────────────────────────────────
2026-08-15，四臂中途看图：**肉眼看不出 npa 与 ctx 的质量差别**（4 张随机
样本 + 一张裁块）。按跑之前写死的判读，这一档是坏消息。

但看图时意识到一件事，它同时是问题和机会：

  **`ctx` 根本不改 query 网格**（它只放大 K/V）。所以若接缝来自 query
  分块，ctx 修不掉；能修的是 `shift`（网格逐层随机化）与 `ovl-attn`
  （重叠平均）。三个臂对"接缝"这一项有**互不相同的预测** ——
  这比"KIDp 差 0.0002"可证伪得多。

而窗口边界的位置是**算出来的，不是找出来的**（见下），所以这不是事后
找规律，是先预测再验证。

────────────────────────────────────────────────────────────────────────
边界周期怎么算出来的（4096² / SDXL / ScaleDiff NPA）
────────────────────────────────────────────────────────────────────────
VAE 下采样 8 倍：4096 图 -> 512×512 latent，故 height_scale s = 4。

  down_blocks.1 / up_blocks.1：特征图 = latent/2 = 256。ws=64 -> h=4×64=256 ✓
      query 块边长 p2 = 32 个特征；每特征 = 2 latent = **16 px**
      -> 块边长 32 × 16 = **512 px**
  down_blocks.2 / mid / up_blocks.0：特征图 = latent/4 = 128。ws=32 -> h=128 ✓
      p2 = 16 个特征；每特征 = 4 latent = **32 px**
      -> 16 × 32 = **512 px**

两个层级**折算到像素空间都是 512**（因为 ws 随层级一起缩）。
所以 4096² 上是 8×8 的 query 网格，边界在 x, y = 0, 512, 1024, ..., 3584。

────────────────────────────────────────────────────────────────────────
怎么测
────────────────────────────────────────────────────────────────────────
横向梯度 |∂I/∂x| 对所有行取平均 -> 长 4096 的一维剖面；再按周期 P=512
折叠平均 -> 长 512 的相位剖面。有系统性接缝，相位 0 处出尖峰；没有则平。

报的数是 **seam contrast**：

    C = (相位 0 附近 ±1 px 的均值) / (相位剖面的中位数) − 1

C ≈ 0 表示没有接缝；C > 0 表示边界处梯度系统性偏高。
纵向同理，两个方向都报。

**阳性对照（必跑）**：给干净图人工加一条 512 px 周期的弱接缝，探针必须
测得出来，且强度随注入幅度单调。测不出来就是探针不灵，不是图没接缝 ——
这条纪律与 §8.9 系列（尺子先报量程再报读数）一致。

    python scalediff_probe/seam_probe.py --selftest        # 阳性对照
    python scalediff_probe/seam_probe.py --idx 0 1 2 3 4
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
PERIOD = 512          # 见文件头的推导，**不是拟合出来的**


def gray(p_or_arr):
    if isinstance(p_or_arr, (str, Path)):
        im = Image.open(p_or_arr).convert("L")
        a = np.asarray(im, dtype=np.float32)
    else:
        a = np.asarray(p_or_arr, dtype=np.float32)
    return a


def phase_profile(a, axis, period=PERIOD):
    """沿 axis 求 |梯度| 的平均剖面，再按 period 折叠。"""
    g = np.abs(np.diff(a, axis=axis))
    prof = g.mean(axis=1 - axis)                 # 对另一轴平均
    n = (len(prof) // period) * period
    if n < period:
        return None
    return prof[:n].reshape(-1, period).mean(0)


def seam_stats(a, period=PERIOD):
    """**不假设相位，把它测出来。**

    第一版取相位 0 附近 ±1 共三个 bin 求平均 —— 接缝只落在其中一个 bin，
    被稀释 3 倍，阳性对照因此没过。现在报三个数，两件事分开查：

        C_max  = max(相位剖面) / median − 1      有没有周期结构
        phase  = argmax 的位置                   在不在预测的位置上
        C_pred = 剖面[predicted] / median − 1     预测位置上的强度

    预测位置：query 块覆盖 [k*P, (k+1)*P)，边界在 P−1 与 P 之间，
    而 np.diff 把索引前移半格，故 **predicted phase = P − 1**。

    C_max 大**且** phase ≈ P−1  -> 接缝坐实（结构与位置双中）
    C_max 大但 phase 在别处      -> 是别的周期性，不是 query 网格
    C_max ≈ 0                    -> 没有周期结构
    """
    out = {}
    for ax, name in ((1, "x"), (0, "y")):
        pf = phase_profile(a, ax, period)
        if pf is None:
            out[name] = None
            continue
        med = float(np.median(pf))
        if med <= 0:
            out[name] = None
            continue
        j = int(np.argmax(pf))
        out[name] = {"c_max": float(pf.max() / med - 1),
                     "phase": j,
                     # 与预测位置的环形距离
                     "dist": int(min((j - (period - 1)) % period,
                                     ((period - 1) - j) % period)),
                     "c_pred": float(pf[period - 1] / med - 1)}
    return out


def seam_contrast(a, period=PERIOD):
    """兼容旧调用：只返回预测位置上的强度。"""
    st = seam_stats(a, period)
    return {k: (v["c_pred"] if v else None) for k, v in st.items()}


def selftest(n=4096, period=PERIOD):
    """阳性对照。**接缝模型要像真的**：真实的拼接接缝不是"某一列偏亮"，
    而是**相邻块由不同的去噪轨迹生成、块间统计量不连续**。所以这里给
    每个块独立的 DC 偏移（第一版用"给一列加常数"，太弱，amp=4 时探针
    仍读不出来 —— 那不是图没接缝，是模型不像，作废重做）。

    同时报**原始 peak / median**，让读者看得到量纲，而不只看比值。
    """
    rng = np.random.default_rng(0)
    m = n
    f = rng.normal(size=(m, m))
    F = np.fft.fftshift(np.fft.fft2(f))
    yy, xx = np.mgrid[-m // 2:m // 2, -m // 2:m // 2]
    r = np.sqrt(xx ** 2 + yy ** 2) + 1
    base = np.real(np.fft.ifft2(np.fft.ifftshift(F / r)))
    base = 128 + 60 * base / (base.std() + 1e-9)
    base = np.clip(base, 0, 255)
    nb = m // period
    print(f"阳性对照：{m}² 的 1/f 纹理底图（std≈{base.std():.0f}），"
          f"周期 {period} px，{nb}×{nb} 块\n")
    print(f"{'块间 DC 偏移 std':>16}{'C_max(x)':>10}{'phase':>8}"
          f"{'|dist|':>8}{'C_pred(x)':>10}{'C_pred(y)':>10}")
    print(f"{'':>16}{'':>10}{'':>8}{'预测 511':>8}")
    print("-" * 62)
    prev, ok, c0 = None, True, None
    for sd in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        a = base.copy()
        if sd:
            off = rng.normal(0, sd, (nb, nb))
            a = a + np.kron(off, np.ones((period, period)))
        st = seam_stats(a, period)["x"]
        print(f"{sd:>16.1f}{st['c_max']:>10.4f}{st['phase']:>8d}"
              f"{st['dist']:>8d}{st['c_pred']:>10.4f}"
              f"{seam_stats(a, period)['y']['c_pred']:>10.4f}")
        c = {"x": st["c_pred"], "y": seam_stats(a, period)["y"]["c_pred"]}
        if sd == 0.0:
            c0 = c["x"]
            ok &= abs(c["x"]) < 0.02 and abs(c["y"]) < 0.02
        else:
            if prev is not None:
                ok &= c["x"] > prev
            prev = c["x"]
    detect = prev is not None and prev > 0.05
    ok &= detect
    print(f"""
判读：
  零偏移时 C≈0（实测 {c0:+.4f}，门槛 |C|<0.02）  -> 无假阳性
  C 随偏移单调上升                                -> 有方向性
  最大偏移下 C > 0.05（实测 {prev:+.4f}）          -> 灵敏度足够
""")
    print("**阳性对照过。**" if ok else
          "**阳性对照没过 —— 探针不灵，真实图上的任何读数都不算数。**")
    return 0 if ok else 1


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=str(root / "p0"))
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--period", type=int, default=PERIOD)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    p0 = Path(a.p0)
    have = {}
    for m in ARMS:
        d = p0 / m
        if not d.exists():
            continue
        fs = {int(p.stem.split("_")[0]): p
              for p in d.glob(f"*_{a.res}.png")}
        if fs:
            have[m] = fs
    if len(have) < 2:
        sys.exit(f"{p0} 下可比的臂不足（找到 {list(have)}）")
    common = sorted(set.intersection(*[set(v) for v in have.values()]))
    if a.idx:
        common = [i for i in common if i in a.idx]
    common = common[:a.limit]
    print(f"臂 {list(have)}   共同 idx {len(common)} 张   "
          f"周期 {a.period} px（推导值，见文件头）\n")
    if not common:
        return 1

    res = {m: [] for m in have}
    for k, i in enumerate(common, 1):
        for m in have:
            st = seam_stats(gray(have[m][i]), a.period)
            if st["x"] and st["y"]:
                res[m].append({
                    "idx": i,
                    "c_pred": (st["x"]["c_pred"] + st["y"]["c_pred"]) / 2,
                    "c_max": (st["x"]["c_max"] + st["y"]["c_max"]) / 2,
                    # 两个方向的峰值是否都落在预测相位附近（容差 2 px）
                    "hit": int(st["x"]["dist"] <= 2) + int(st["y"]["dist"] <= 2)})
        print(f"\r  {k}/{len(common)}", end="", flush=True)
    print()

    print(f"\n{'臂':<12}{'C_pred':>10}{'C_max':>10}{'峰值命中率':>12}"
          f"   （命中 = argmax 落在预测相位 {a.period-1} 的 ±2 内）")
    print("-" * 62)
    base_m = "npa" if "npa" in res else list(res)[0]
    for m in have:
        r = res[m]
        if not r:
            continue
        cp = np.mean([x["c_pred"] for x in r])
        cm = np.mean([x["c_max"] for x in r])
        hit = sum(x["hit"] for x in r) / (2 * len(r))
        print(f"{LABEL.get(m,m):<12}{cp:>10.4f}{cm:>10.4f}{hit:>11.0%}")
    print(f"\n  参照：合成无接缝纹理上 C_max 的噪声地板 ≈ 0.016，"
          f"命中率应 ≈ {2/ a.period:.1%}（纯偶然）")

    print(f"\n对 {base_m} 的配对差（同一批 {len(common)} 张，bootstrap 1000 次）")
    rng = np.random.default_rng(0)
    bmap = {x["idx"]: x["c_pred"] for x in res[base_m]}
    for m in have:
        if m == base_m:
            continue
        pairs = [(bmap[x["idx"]], x["c_pred"]) for x in res[m]
                 if x["idx"] in bmap]
        if len(pairs) < 3:
            continue
        d = np.array([q - b for b, q in pairs])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        s2 = 2 * np.std(bs)
        vd = "✅ 接缝减弱" if d.mean() < -s2 else (
            "❌ 接缝加重" if d.mean() > s2 else "…不定")
        print(f"  {LABEL.get(m,m):<12}Δ C_pred = {d.mean():>+9.4f}  "
              f"2σ={s2:.4f}   {vd}   n={len(d)}")

    print(f"""
预测（写在看数字之前，由各臂改了什么直接推出）：
  npa       基准 —— 若"边界伪影"假设成立，它应当**有**可测的接缝
  shift     query 网格逐层随机化 -> 接缝应**消失**
  ovl-attn  重叠 query + 层内平均 -> 接缝应**减弱**
  ctx       **只放大 K/V，query 网格不变** -> 接缝应**仍在**

  四条全中 -> 我们把"分解的代价"拆成了两个可分离的机制，且各有对症的
      干预。这是一张能进论文的图，比 KIDp 差 0.0002 有说服力得多。
  npa 本身就没有接缝（C≈0）-> **"边界伪影"这条当场否掉**，
      §10.9（他们那个没启用的开关）连带降级为无关紧要。
  预测不成模式（比如 ctx 也减弱）-> 说明我对机制的理解是错的，
      要重新想，不要事后编解释。""")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"period": a.period, "idx": common,
             "per_arm": {m: {k: list(map(float, v)) for k, v in r.items()}
                         for m, r in res.items()}}, ensure_ascii=False,
            indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
