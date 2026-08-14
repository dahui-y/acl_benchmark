"""门控图探针：找出"哪些层 / 哪些时间步"才给出有结构的主体图。

**为什么需要它**（v1.1 重录实验的失败指出来的）：
band_stats 显示 263/474 的过渡带 = **1.00**、低权重区 = **0.00** ——
整张门控图没有任何位置被判为背景，**门从来只是"半开"**。
病根不是分辨率（重录到 256×256 毫无改善），是 `record` 把
**~70 个 attn2 层 × 50 个时间步**一锅平均，把结构洗平了：
归一化后 r 几乎处处落在均值 ±10% 内 -> map 全挤在 [0.3, 0.7]。

**这个探针只跑基础阶段**（upsample_stage=0，~8 秒/张），一次前向
同时累积所有 (层分辨率 × 时间步桶) 的分立版本，跑完直接报出每种
配置的对比度。选配置用**诊断集**（§6.9 允许：脚手架就是干这个的），
选定后才上触发集验证。

判据（选哪个配置）：
    对比度 = 低权重区(<0.3) 与 高权重区(>0.7) 的面积都 > 10%，
    且过渡带 < 40%；在多条 prompt 上稳定。

    python scalediff_probe/gate_map_probe.py --idx 0 2 4 8 20
    python scalediff_probe/gate_map_probe.py --idx 0 --vis   # 出可视化
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
SDXL_DIR = REPO / "help_code" / "ScaleDiff" / "SDXL"
sys.path.insert(0, str(SDXL_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PROMPTS, NEGATIVE                        # noqa: E402
from subject_phrases import HEADS, strip_subject             # noqa: E402
from method_v0 import subject_token_ids                      # noqa: E402
from method_v1 import BlendCrossAttn                         # noqa: E402

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
CANON = 64


class ProbeGate:
    """只录不混，按 (注意力边长, 时间步桶) 分立累积。"""

    def __init__(self, token_ids, n_steps=50, n_buckets=5, n_content=None):
        self.tok = token_ids
        # n_content = prompt 的真实 token 数（含 BOS/EOS）。CLIP 的 77 长
        # 序列里其余全是 padding，而 padding/EOS 常吸走大量注意力且空间
        # 均匀 —— 用完整 softmax 当分母，主体信号被均匀底噪稀释。
        self.n_content = n_content
        self.phase = 1
        self.alt = None
        self.map = None
        self.s = 0.0                      # 永不混合
        self.step = 0
        self.n_steps, self.n_buckets = n_steps, n_buckets
        self.acc = defaultdict(lambda: [None, 0])
        self.per_layer = defaultdict(lambda: [None, 0])
        self.single = []          # (层名, 边长, 步桶, 单图 max/p50)

    def bucket(self):
        return min(self.step * self.n_buckets // max(self.n_steps, 1),
                   self.n_buckets - 1)

    def record(self, probs, hw, name=None):
        """name = 具体层名。**先平均再量对比度分不开两种情况**：
        单图很锐但彼此不一致（平均抵消） vs 单图本来就平。
        所以这里同时记 (a) 逐层名的累积图 和 (b) 平均**之前**每张图的
        对比度 —— 后者是判决 (a)/(b) 的唯一依据。"""
        if not self.tok:
            return
        h = int(hw ** 0.5)
        if h * h != hw:
            return
        name = getattr(self, "_cur_name", None)
        sub = probs[-1, :, :, self.tok].sum(-1).mean(0).float()      # (hw,)
        sigs = {"raw": sub}
        if self.n_content:
            tot = probs[-1, :, :, :self.n_content].sum(-1).mean(0).float()
            sigs["content"] = sub / (tot + 1e-8)
        for sig, v in sigs.items():
            m = F.interpolate(v.reshape(1, 1, h, h), (CANON, CANON),
                              mode="bilinear", align_corners=False)[0, 0]
            if sig == "raw":
                # 平均之前的单图对比度（judge (a) vs (b) 的关键）
                self.single.append((name or f"res{h}", h, self.bucket(),
                                    contrast(m)[1]))
                if name:
                    a = self.per_layer[name]
                    a[0] = m if a[0] is None else a[0] + m
                    a[1] += 1
            for key in ((h, self.bucket(), sig), (h, "all", sig),
                        ("all", self.bucket(), sig), ("all", "all", sig)):
                a = self.acc[key]
                a[0] = m if a[0] is None else a[0] + m
                a[1] += 1

    def weights(self, hw, device, dtype):
        return None

    def finalize(self):
        pass


def norm_linear(m, k=1.0, width=0.5):
    r = m / (m.mean() + 1e-8)
    return ((r - k) / width + 0.5).clamp(0, 1)


def norm_otsu(m):
    """Otsu 自动阈值：让图自己说主体占多少，不预设面积。
    阈值两侧各留 5% 动态范围做过渡带。"""
    f = m.flatten()
    lo_, hi_ = f.min(), f.max()
    bins = 64
    hist = torch.histc(f, bins=bins, min=float(lo_), max=float(hi_))
    prob = hist / hist.sum()
    centers = torch.linspace(float(lo_), float(hi_), bins,
                             device=m.device, dtype=m.dtype)
    w0 = torch.cumsum(prob, 0)
    mu = torch.cumsum(prob * centers, 0)
    muT = mu[-1]
    denom = (w0 * (1 - w0)).clamp_min(1e-8)
    var_b = (muT * w0 - mu) ** 2 / denom
    # 退化 bin 屏蔽：w0 逼近 0/1 时 denom->0 会把 var_b 顶上天，
    # 在极度偏斜的直方图（小主体图就是）上会选到边界。实测踩过。
    var_b[(w0 < 0.02) | (w0 > 0.98)] = -1.0
    thr = centers[int(torch.argmax(var_b))]
    half = 0.05 * (hi_ - lo_)
    return ((m - (thr - half)) / (2 * half + 1e-8)).clamp(0, 1)


def norm_quantile(m, lo=0.55, hi=0.85):
    """分位数归一化：**直接规定**多少面积落在背景/主体侧，
    与 r 本身平不平无关 —— 这是对"图太平"最直接的解药。"""
    f = m.flatten()
    a = torch.quantile(f, lo)
    b = torch.quantile(f, hi)
    return ((m - a) / (b - a + 1e-8)).clamp(0, 1)


def contrast(m):
    """与归一化**无关**的原始对比度。

    用 **p99/p50 与 max/p50**，不用 p95 —— 小主体（lone 类真实占比
    ~2%）在 p95 处还落在背景里，比值会被读成 1.00（自检踩过）。"""
    f = m.flatten().float()
    p99 = torch.quantile(f, 0.99)
    p50 = torch.quantile(f, 0.50)
    return float(p99 / (p50 + 1e-8)), float(f.max() / (p50 + 1e-8))


def band(m):
    return (float((m > 0.7).float().mean()),
            float((m < 0.3).float().mean()),
            float(((m >= 0.3) & (m <= 0.7)).float().mean()))


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default=str(root / "gate_probe"))
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()

    from pipeline_scalediff_sdxl import CustomStableDiffusionXLPipeline
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    kw = {"torch_dtype": torch.float16}
    for s_ in (hub / "models--stabilityai--stable-diffusion-xl-base-1.0"
               / "snapshots").glob("*"):
        if not (s_ / "unet" / "diffusion_pytorch_model.safetensors").exists() \
                and list((s_ / "unet").glob("*.fp16.safetensors")):
            kw["variant"] = "fp16"
    pipe = CustomStableDiffusionXLPipeline.from_pretrained(CKPT, **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    orig_step = pipe.noise_pred_step
    allres, gates = {}, []
    for idx in a.idx:
        cat, subj, prompt = PROMPTS[idx]
        head = HEADS[idx]
        if not head:
            print(f"[{idx}] 无主体，跳过")
            continue
        tids = subject_token_ids(pipe, prompt, head)
        n_content = int(pipe.tokenizer(prompt, truncation=True,
                                       max_length=77).input_ids.__len__())
        gate = ProbeGate(tids, n_steps=a.steps, n_content=n_content)
        print(f"  内容 token 数 {n_content} / 77（其余是 padding）")
        procs = dict(pipe.unet.attn_processors)
        for k in procs:
            if k.endswith("attn2.processor"):
                procs[k] = NamedCrossAttn(gate, k.replace(".processor", ""))
        pipe.unet.set_attn_processor(procs)

        def patched(latents, t, *args, _o=orig_step, _g=gate, **kw2):
            r = _o(latents, t, *args, **kw2)
            _g.step += 1
            return r
        pipe.noise_pred_step = patched

        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        pipe(prompt, negative_prompt=NEGATIVE, height=1024, width=1024,
             generator=torch.Generator(device="cuda").manual_seed(a.seed),
             num_inference_steps=a.steps, guidance_scale=7.5,
             restart_ratio=0.4, scale_factor=0.125, upsample_stage=0)
        pipe.noise_pred_step = orig_step

        maps = {}
        for key, (acc, n) in gate.acc.items():
            if acc is None or not n:
                continue
            maps[key] = acc / n
        allres[idx] = maps
        gates.append(gate)
        print(f"\n[{idx}] {cat}/{subj}  tok={tids}  "
              f"层分辨率 {sorted({k[0] for k in maps if k[0] != 'all'})}")

    # ---- 判决 (a)/(b)：平均前 vs 平均后 ----
    print("\n" + "=" * 74)
    print("判决：单图对比度（平均**之前**） vs 该层平均后的对比度")
    print("  单图高、平均后低 -> (a) 各层空间不一致，平均抵消 -> 选层可救")
    print("  单图也低         -> (b) 交叉注意力本身就弥散 -> 换信号/改叙述")
    sing = [c for g in gates for c in g.single]
    if sing:
        import statistics as st
        vals = [c[3] for c in sing]
        print(f"\n  单图 max/p50：n={len(vals)}  中位 {st.median(vals):.2f}  "
              f"p90 {sorted(vals)[int(.9*len(vals))]:.2f}  最大 {max(vals):.2f}")
        by = defaultdict(list)
        for nm, h, b, v in sing:
            by[(h, b)].append(v)
        print(f"\n  {'边长':>6}{'步桶':>6}{'单图 max/p50 中位':>18}{'最大':>8}")
        for k in sorted(by, key=lambda k: (k[0], str(k[1]))):
            v = by[k]
            print(f"{k[0]:>6}{str(k[1]):>6}{st.median(v):>18.2f}{max(v):>8.2f}")
    # 逐层名：哪一层的平均图最锐
    pl = defaultdict(list)
    for g in gates:
        for nm, (acc, n) in g.per_layer.items():
            if acc is not None and n:
                pl[nm].append(contrast(acc / n)[1])
    if pl:
        rank = sorted(((sum(v) / len(v), nm) for nm, v in pl.items()),
                      reverse=True)
        print(f"\n  最锐的 8 个层（该层自身平均后的 max/p50）：")
        for v, nm in rank[:8]:
            print(f"    {v:>6.2f}  {nm}")
        print(f"  最平的 3 个：" + ", ".join(f"{v:.2f} {nm.split('.')[-3:][0]}"
                                              for v, nm in rank[-3:]))
    print("=" * 74)

    # ---- 汇总：每个配置在所有 prompt 上的平均对比度 ----
    keys = sorted({k for m in allres.values() for k in m},
                  key=lambda k: (str(k[0]), str(k[1])))
    print(f"\n{'层':>5}{'步桶':>5}{'信号':>9}{'  归一化':>10}"
          f"{'p99/p50':>9}{'max/p50':>9}{'主体>.7':>9}{'背景<.3':>9}{'过渡带':>8}{'跨度':>8}  判读")
    print("-" * 97)
    best = []
    for key in keys:
        cfgs = [("rel_w.5", lambda m: norm_linear(m, width=0.5)),
                ("rel_w.2", lambda m: norm_linear(m, width=0.2)),
                ("rel_w.1", lambda m: norm_linear(m, width=0.1)),
                ("rel_w.05", lambda m: norm_linear(m, width=0.05)),
                ("otsu", norm_otsu),
                ("quantile", norm_quantile)]
        for nm, fn in cfgs:
            bs = [band(fn(allres[i][key])) for i in allres if key in allres[i]]
            if not bs:
                continue
            hi = sum(b[0] for b in bs) / len(bs)
            lo = sum(b[1] for b in bs) / len(bs)
            tb = sum(b[2] for b in bs) / len(bs)
            ok = hi > 0.10 and lo > 0.10 and tb < 0.40
            # **自适应性判据（决定性）**：主体覆盖必须随图变化 ——
            # lone 类该小、portrait 类该大。固定分位数会全给同一个数。
            covs = [band(fn(allres[i][key]))[0] for i in allres if key in allres[i]]
            spread = (max(covs) - min(covs)) if len(covs) > 1 else 0.0
            adapt = spread > 0.15
            if ok and adapt:
                best.append((tb, key, nm, hi, lo, spread))
            cs = [contrast(allres[i][key]) for i in allres if key in allres[i]]
            cr = sum(c[0] for c in cs) / max(len(cs), 1)
            cmx = sum(c[1] for c in cs) / max(len(cs), 1)
            print(f"{str(key[0]):>5}{str(key[1]):>5}{str(key[2]):>9}{nm:>10}"
                  f"{cr:>9.2f}{cmx:>9.2f}{hi:>9.1%}{lo:>9.1%}{tb:>8.1%}{spread:>8.1%}  "
                  f"{'**合格**' if ok and adapt else ('带宽OK不自适应' if ok else '')}")
    print("\n判据：主体>0.7 与 背景<0.3 都 >10%、过渡带 <40%，"
          "**且覆盖跨度 >15%（主体大小必须自适应 —— 固定分位数会失败）**。")
    # 逐样本覆盖率：自适应性一眼可见
    print(f"\n逐样本主体覆盖（挑几个配置对照，lone 应小 / portrait 应大）：")
    show = [((32, "all", "raw"), "raw+rel_w.1",
             lambda m: norm_linear(m, width=0.1)),
            ((32, "all", "content"), "content+rel_w.1",
             lambda m: norm_linear(m, width=0.1)),
            ((32, "all", "content"), "content+rel_w.05",
             lambda m: norm_linear(m, width=0.05))]
    print(f"{'配置':>16}" + "".join(f"{('idx'+str(i)):>10}" for i in allres))
    for key, nm, fn in show:
        row = [band(fn(allres[i][key]))[0] if key in allres[i] else float('nan')
               for i in allres]
        print(f"{nm:>16}" + "".join(f"{v:>10.1%}" for v in row))
    if best:
        best.sort()
        tb, key, nm, hi, lo, sp = best[0]
        print(f"\n**推荐配置：层={key[0]} 步桶={key[1]} 信号={key[2]} 归一化={nm}"
              f"（过渡带 {tb:.1%}，主体 {hi:.1%}，背景 {lo:.1%}，"
              f"覆盖跨度 {sp:.1%}）**")
    else:
        print("**没有配置合格** —— 主体图本身可能就不带足够对比度，"
              "要换信号（如 self-attention 聚类 / 多 token 对比）。")

    if a.vis and allres:
        from PIL import Image
        import numpy as np
        idx0 = list(allres)[0]
        ks = [k for k in keys if k in allres[idx0] and k[1] == "all"][:12]
        C = 128
        sheet = Image.new("L", (len(ks) * C, 2 * C), 255)
        for j, key in enumerate(ks):
            for r_, fn in enumerate((norm_linear, norm_quantile)):
                m = fn(allres[idx0][key]).cpu().numpy()
                im = Image.fromarray((m * 255).astype(np.uint8)).resize((C, C))
                sheet.paste(im, (j * C, r_ * C))
        p = out / f"maps_{idx0}.png"
        sheet.save(p)
        print(f"\n可视化 {p}（上行 linear，下行 quantile，列 = {ks}）")
    json.dump({str(i): {str(k): [band(norm_linear(v, width=0.1)), contrast(v)]
                        for k, v in m.items()} for i, m in allres.items()},
              (out / "band_stats.json").open("w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
