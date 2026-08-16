"""调度诊断：SDXL 上真正控制 patch 三列的旋钮是哪个。零方法设计，纯诊断。

────────────────────────────────────────────────────────────────────────
为什么换到这里看
────────────────────────────────────────────────────────────────────────
P0 判死了"注意力窗口几何"那条线（§10.20）。事后从我们自己的测量能看出
为什么：**自注意力只占 SDXL UNet 总计算的约 6.5%**（由 ovl-attn 把注意力
FLOPs 提到 3.06× 而总时长只涨 13% 反推）。93.5% 的网络与窗口划分无关，
卷积把影响稀释掉了。**我们把力气全用在了 6.5% 的那部分。**

那 patch 三列到底被什么控制？把他们**自己的两张消融表**换算成 KIDp：

| 旋钮 | ΔFIDp | ΔKIDp | 出处 |
|---|---|---|---|
| **LFM 开/关** | −2.06 | **−0.0007** | Table 3 |
| **τ 300→700** | −1.73 | **−0.0013** | Table 4 |
| 注意力 NPA→MD | −0.81 | −0.0011 | Table 3 |
| SG 开/关 | −0.05 | −0.0001 | Table 3 |
| 我们实测的注意力几何 | ~0 | −0.0002 | P0 |

两条读出来的事实：
1. **SG 对 patch 三列几乎无作用**（−0.0001）—— 它是治重复/全局结构的；
2. patch 那条线上真正的旋钮是 **LFM** 和 **τ**。

────────────────────────────────────────────────────────────────────────
读代码发现的、比"扫一个超参"更值钱的事
────────────────────────────────────────────────────────────────────────
`pipeline_scalediff_sdxl.py` 里 `refine()` 被调用两次，**共用同一个
`scale_factor`**：

    latents_LFM = refine(latents_LU, latents_RU, 1,      scale_factor=sf)  # LFM
    pred_x0     = refine(latents_LFM, pred_x0, scale=s,  scale_factor=sf)  # SG

`scale_factor` 是**频率分割点**（`h = int(H*sf)`，即保留到 1/8 空间频率）。
LFM 用它决定"低频取自 Z_LU、高频取自 Z_RU"的分界；SG 用它决定"把
pred_x0 的哪一段频率拉向参考"。

> **两件不同的事，共用一个分割点，纯粹是实现上的便利，且从未被消融过。**
> 他们消融了 τ（Table 4）、消融了注意力（Table 3）、消融了 LFM/SG 的
> **开关**，**唯独没有消融这个分割点该设多少**，更没试过把两处解耦。

两个调用点可以**可靠区分**：LFM 传位置参数 `1`，SG 传关键字 `scale=`。
所以解耦不需要改基线文件，包一层 `refine` 即可（BASELINE_PATCHES.md）。

────────────────────────────────────────────────────────────────────────
配置（6 个要跑；基线已有）
────────────────────────────────────────────────────────────────────────
`p0/npa` 那批**正好就是默认配置**（sf=0.125, restart=0.4, NPA），
直接当基线，不重跑。

    名字        sf_lfm   sf_sg   restart_ratio
    base        0.125    0.125   0.4     <- 复用 p0/npa
    lfm_lo      0.0625   0.125   0.4
    lfm_hi      0.25     0.125   0.4
    lfm_xhi     0.5      0.125   0.4
    sg_lo       0.125    0.0625  0.4
    sg_hi       0.125    0.5     0.4
    tau_hi      0.125    0.125   0.7     <- 对应他们 Table 4 的 τ=700

restart_ratio ↔ τ 的换算：`restart_step = int(steps*(1−r))`，50 步、
r=0.4 -> 第 30 步 -> τ≈400 ✓ 与他们一致；τ=700 <-> **r=0.7**。
Table 4 里 τ=700 的 patch 三列最好（FIDp 37.65 / KIDp .0064），
而他们**选了对 patch 更差的 τ=400**，因为那样全局 FID 最好 ——
又一次拿局部换全局。

**一个免费的加成**：管线一次返回 1024/2048/4096 三档，所以每张图同时给出
2048² 和 4096² 的读数，**不花额外算力**就能回答那个关键问题：

> **最优分割点会不会随放大倍数 s 移动？** 他们全程用一个固定值。
> 若最优点随 s 移动，"自适应频率分割"就是一个方法，且**零额外开销**。

────────────────────────────────────────────────────────────────────────
预注册判据（写在跑之前）
────────────────────────────────────────────────────────────────────────
n=60，配对 σ_d ≈ 0.00029（由 P0 实测外推）。

**多重比较修正 —— 这一条我初稿漏了，补在看数字之前。**
6 个配置同时对 base 比，每个用 2σ（约 5%）判，则"至少一个假阳性"的概率
是 1 − 0.95⁶ ≈ **26%**。即：六个旋钮全无效时，我们也有四分之一的概率
看到"某配置显著变好"然后兴冲冲往下走。
Bonferroni 把族错误率控在 5% -> 单个检验用 **2.6σ_d ≈ 0.00074**。

  ① 有没有杠杆：某配置在 4096² 上同时满足
       (a) ΔKIDp < 0（方向对）
       (b) |ΔKIDp| > **2.6σ_d**（Bonferroni 修正后的显著性）
       否 -> **所有旋钮都无杠杆，SDXL 这个 backbone 上确实没有 headroom，
              方向判死。** 那时再转 DiT 是有依据的转，不是碰运气。
       是 -> 进 ②

  **已知风险（写在前面）**：`tau_hi` 只走 r=0.4->0.7，而他们 Table 4 的
  −0.0013 是 τ=300->700 的**全程**；我们这一步约半程，预期 −0.0006~−0.0009，
  **正好压在门槛上**，且那是 2048² 的数。tau_hi 出"不定"**不代表 τ 无用**，
  只代表这一步跨得不够大 —— 届时补一个 r=0.85 的点（约 1.25 h）即可，
  不用重做全套。
  ② 最优点是否随分辨率移动：比较 2048² 与 4096² 上的最优配置
       相同 -> 只是"他们超参没调最优"，**这不是论文**，方向判死。
       不同 -> **自适应分割成立**，方法候选，进设计。
  ③ 附带读数：sg_lo / sg_hi 若与 base 无差别，则坐实"SG 与 patch 三列
       无关"（他们 Table 3 已暗示），解耦的价值就全在 LFM 一侧。

**判死也是有效结果** —— 它把"SDXL 上还有没有戏"彻底关掉。

    python scalediff_probe/sched_probe.py --plan       # 只看名单，不烧 GPU
    nohup python scalediff_probe/sched_probe.py > sched.log 2>&1 &
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NEGATIVE                                    # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"

# name -> (sf_lfm, sf_sg, restart_ratio)。base 复用 p0/npa，不在此表。
CONFIGS = {
    "lfm_lo":  (0.0625, 0.125,  0.4),
    "lfm_hi":  (0.25,   0.125,  0.4),
    "lfm_xhi": (0.5,    0.125,  0.4),
    "sg_lo":   (0.125,  0.0625, 0.4),
    "sg_hi":   (0.125,  0.5,    0.4),
    "tau_hi":  (0.125,  0.125,  0.7),
}
BASE = (0.125, 0.125, 0.4)


def needs_fp16_variant():
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    for s in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
              / "snapshots").glob("*"):
        u = s / "unet"
        if not (u / "diffusion_pytorch_model.safetensors").exists() \
                and list(u.glob("*.fp16.safetensors")):
            return True
    return False


def done_of(d):
    m = Path(d) / "manifest.jsonl"
    return ({json.loads(l)["idx"] for l in m.open() if l.strip()}
            if m.exists() else set())


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(root / "sched"))
    ap.add_argument("--p0", default=str(root / "p0"),
                    help="从这里借 plan.json；p0/npa 即基线配置")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    planf = Path(a.p0) / "plan.json"
    if not planf.exists():
        sys.exit(f"没有 {planf} —— 基线名单从 p0 借，先确认它在")
    plan = json.loads(planf.read_text())
    idxs = plan["idx"][:a.n]
    pmap = plan["prompt"]
    base_dir = Path(a.p0) / "npa"
    have_base = done_of(base_dir)
    miss = [i for i in idxs if i not in have_base]

    out = Path(a.out)
    dirs = {c: out / c for c in a.configs}
    done = {c: done_of(d) for c, d in dirs.items()}
    todo = [(i, c) for i in idxs for c in a.configs if i not in done[c]]

    print(f"\n名单：借自 {planf}，前 {len(idxs)} 条（tune split）")
    print(f"基线 = {base_dir}（p0/npa，默认配置 sf=0.125/0.125, r=0.4）"
          f"  已有 {len(have_base)} 张，覆盖名单 {len(idxs)-len(miss)}/{len(idxs)}")
    if miss:
        print(f"  ⚠ 基线缺 {len(miss)} 条：{miss[:8]}{'...' if len(miss)>8 else ''}"
              f"\n    p0 还在跑就等它补齐，或把 --n 调小到已覆盖的范围")
    print(f"\n{'配置':<10}{'sf_lfm':>9}{'sf_sg':>8}{'restart':>9}{'已有':>7}")
    print("-" * 45)
    print(f"{'base':<10}{BASE[0]:>9}{BASE[1]:>8}{BASE[2]:>9}"
          f"{len(have_base & set(idxs)):>7}   <- 复用，不跑")
    for c in a.configs:
        s1, s2, r = CONFIGS[c]
        print(f"{c:<10}{s1:>9}{s2:>8}{r:>9}{len(done[c] & set(idxs)):>7}")
    print(f"\n待跑 {len(todo)} 张，按 75s/张估 **{len(todo)*75/3600:.1f} 小时**")
    print("""
预注册判据（写在跑之前）：
  **多重比较**：6 个配置同时比 base，2σ 判会有 ~26% 概率至少一个假阳性。
  Bonferroni 修正 -> 单检验用 **2.6σ_d ≈ 0.00074**（n=60）。
  ① 某配置 4096² 上 ΔKIDp < 0 且 |ΔKIDp| > 2.6σ_d
       否 -> 所有旋钮都无杠杆，**SDXL 上没有 headroom，方向判死**；
             那时转 DiT 才是有依据的转。
       是 -> 进 ②
     已知风险：tau_hi 只走半程（r 0.4->0.7），预期效应压在门槛上，
     出"不定"不代表 τ 无用，补 r=0.85 一个点即可。
  ② 2048² 与 4096² 的最优配置是否相同
       相同 -> 只是"超参没调最优"，**不是论文**，判死。
       不同 -> **自适应频率分割**成立，方法候选。
  ③ sg_lo/sg_hi 与 base 无差别 -> 坐实"SG 与 patch 三列无关"，
       解耦的价值全在 LFM 一侧。""")
    if a.plan:
        print("\n--plan：没有加载模型。")
        return 0
    if not todo:
        print("\n没有要跑的了。打分：")
        print(f"  python scalediff_probe/std_table.py --patch-only "
              f"--real-split tune --res 4096 --boot 200 \\\n"
              f"      --arms \"base={base_dir}\" "
              + " ".join(f'"{c}={dirs[c]}"' for c in a.configs))
        return 0

    import torch
    import pipeline_scalediff_sdxl as PS
    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline

    # ── LFM 与 SG 的分割点解耦 ──────────────────────────────────────
    # 基线文件一个字不动。两个调用点靠**传参方式**区分，可靠：
    #   LFM：refine(x_ref, x_pred, 1, scale_factor=sf)      -> scale 是位置参数
    #   SG ：refine(x_ref, x_pred, scale=s, scale_factor=sf) -> scale 是关键字
    _orig_refine = PS.refine
    SF = {"lfm": BASE[0], "sg": BASE[1]}
    seen = {"lfm": 0, "sg": 0}

    def refine_split(x_ref, x_pred, *args, scale=None, scale_factor=None,
                     **kw):
        is_lfm = len(args) > 0
        sc = args[0] if is_lfm else scale
        seen["lfm" if is_lfm else "sg"] += 1
        return _orig_refine(x_ref, x_pred, sc,
                            scale_factor=SF["lfm" if is_lfm else "sg"])
    PS.refine = refine_split

    kw = {"torch_dtype": torch.float16}
    if needs_fp16_variant():
        kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    mf = {c: (dirs[c] / "manifest.jsonl").open("a") for c in a.configs}
    t_all = time.time()
    try:
        for n, (idx, cfg) in enumerate(todo, 1):
            s1, s2, r = CONFIGS[cfg]
            SF["lfm"], SF["sg"] = s1, s2
            seen["lfm"] = seen["sg"] = 0
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                images = pipe(
                    pmap[str(idx)], negative_prompt=NEGATIVE,
                    height=1024, width=1024,
                    generator=torch.Generator(device="cuda").manual_seed(a.seed),
                    num_inference_steps=a.steps, guidance_scale=7.5,
                    restart_ratio=r, scale_factor=s1,   # 管线自己的值被上面覆盖
                    upsample_stage=a.stage)
            except torch.cuda.OutOfMemoryError:
                print(f"\n[{cfg}] {idx} OOM，跳过")
                torch.cuda.empty_cache()
                continue
            dt = time.time() - t0
            if n == 1:
                # 一次性自检：两个调用点都被走到，且次数合理
                exp_sg = a.stage * int(a.steps * r)
                print(f"\n自检：LFM 调用 {seen['lfm']} 次（应 = 放大级数 "
                      f"{a.stage}），SG 调用 {seen['sg']} 次（应 ≈ "
                      f"{exp_sg}）—— 解耦包装确实生效\n")
            paths = {}
            for im in images:
                p = dirs[cfg] / f"{idx:05d}_{im.width}.png"
                im.save(p)
                paths[im.width] = p.name
            mf[cfg].write(json.dumps({
                "idx": idx, "config": cfg, "sf_lfm": s1, "sf_sg": s2,
                "restart_ratio": r, "prompt": pmap[str(idx)], "seed": a.seed,
                "files": paths, "sec": round(dt, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }, ensure_ascii=False) + "\n")
            mf[cfg].flush()
            el = time.time() - t_all
            print(f"\r{n}/{len(todo)}  {el/3600:.1f}h 已用  "
                  f"剩约 {el/n*(len(todo)-n)/3600:.1f}h  "
                  f"[{cfg}] {idx} {dt:.0f}s", end="", flush=True)
    finally:
        for f in mf.values():
            f.close()
        PS.refine = _orig_refine

    print(f"\n\n完成，用时 {(time.time()-t_all)/3600:.2f} 小时")
    print(f"""
打分（4096² 与 2048² 各跑一次 —— ② 那条判据要两个分辨率）：
  for R in 4096 2048; do
    CUDA_VISIBLE_DEVICES="" python scalediff_probe/std_table.py \\
        --patch-only --real-split tune --res $R --boot 200 \\
        --arms "base={base_dir}" """
          + " ".join(f'"{c}={dirs[c]}"' for c in a.configs) + """
  done

**--arms 第一个必须是 base** —— 配对表以第一个臂为基准。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
