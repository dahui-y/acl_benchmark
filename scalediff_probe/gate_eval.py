"""适用性门的离线评估：v1.1 = 门开取 v1，门关取基线。零 GPU 生成。

为什么修法是门而不是更好的摘除器：

    leak_check 的阴性结果（预注册判据没过，方向还反了）说明文本相似度
    探不到 "holding -> 有手" 这种蕴含式泄漏。而回头看 trigger 的数据，
    出回归的 23/26/27 的空视野比例几乎全在 0%~31% —— **这些行本来就不该
    被干预**。骨架 §1.2.1 早已把适用性门推出来了：主体铺满画面时不存在
    "没有主体的邻域"，没有要抑制的东西。门一关，泄漏没有作用的机会。
    摘除质量只在门开的行上要紧，而那些行（lone）的摘除本来就干净。

门的定义（预注册，跑之前写死）：

    介入 <=> 基图空视野比例 > 0.5 且 基图检出主体数 >= 1

    只依赖基图 + 检测器，推理期可得（基图生成 ~7s 里加 ~2s 检测）。
    基图检不出主体时不介入 —— 那是基图自己的失败，不是外推失败，
    安全默认是保持原版。

离线合成的合法性：
    门只依赖基图，而基图两个 arm 逐字节相同（90/90 已验证）；
    门关的行如果真的重跑，走的是还原后的原版处理器 —— 这条代码路径
    已经由 empty 十五行的逐字节一致（15/15）背书。
    所以 v1.1 的表可以直接由 v1 表 + 基线表按门拼出来，不用再烧 GPU。

预注册的预测：
    P1  lone 十五行全部门开（否则门把主结果杀了，阈值就得重想）
    P2  23/26/27 在出回归的那些 seed 上门关 -> 回归从表里消失
    P3  每个 seed 上 v1.1 的全体 MAE <= v1 的
    P4  门的阈值扫描（0.3 / 0.5 / 0.75）不改变 P1-P3 的结论

    python scalediff_probe/gate_eval.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_objects import Detector                    # noqa: E402
from trigger import empty_view_fraction               # noqa: E402


def load(d):
    return {(r["idx"], r["seed"]): r
            for r in json.loads((Path(d) / "counts.json").read_text())}


def mani(d):
    return {(json.loads(l)["idx"], json.loads(l)["seed"]): json.loads(l)
            for l in (Path(d) / "manifest.jsonl").open()}


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "batch"))
    ap.add_argument("--new", default=str(root / "method_batch_s1"))
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.3, 0.75],
                    help="第一个是主阈值，其余进敏感度表")
    ap.add_argument("--R", type=int, default=4)
    a = ap.parse_args()

    A, B = load(a.base), load(a.new)
    MB = mani(a.new)
    keys = sorted(set(A) & set(B))
    seeds = sorted({k[1] for k in keys})

    det = Detector()
    evf, nbox = {}, {}
    MA = mani(a.base)
    for k in keys:
        f1 = MA[k]["files"].get("1024")
        im = Image.open(Path(a.base) / f1).convert("RGB")
        b, _ = det.detect(im, A[k]["subject"])
        nbox[k] = len(b)
        evf[k] = empty_view_fraction(b, im.width, im.height, a.R)

    def gate(k, thr):
        # v1 本来就没干预的行（head=None 等），门取什么都一样，按未干预算
        if not MB[k].get("applied", True):
            return False
        return evf[k] > thr and nbox[k] >= 1

    thr0 = a.thr[0]
    print(f"主阈值 thr={thr0}   R={a.R}\n")
    print(f"{'idx':<5}{'seed':>6}{'cat':<11}{'空视野':>8}{'框数':>5}"
          f"{'门':>4}{'base e':>8}{'v1 e':>6}{'v1.1 e':>8}")
    print("-" * 66)
    for k in keys:
        if A[k].get("excess") is None:
            # **crowd / texture（无声明基数）此前被整行跳过，于是门在这两类
            # 上的行为从未被看到** —— 而它们恰恰是门要保护的情形（主体铺满
            # 画面 -> 空视野比例低 -> 应当关门、逐字节不动）。
            # 没有 excess 就不参与 MAE，但门的开关必须打印出来。
            g = gate(k, thr0)
            print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<11}{evf[k]:>7.0%}{nbox[k]:>5}"
                  f"{'开' if g else '关':>4}{'':>8}{'':>6}{'':>8}"
                  f"  (无基数，不进 MAE)")
            continue
        g = gate(k, thr0)
        e_v11 = B[k]["excess"] if g else A[k]["excess"]
        flag = ""
        if k[0] in (23, 26, 27) and not g and abs(B[k]["excess"]) > abs(A[k]["excess"]):
            flag = "  << 回归被门消掉"
        print(f"{k[0]:<5}{k[1]:>6}{A[k]['cat']:<11}{evf[k]:>7.0%}{nbox[k]:>5}"
              f"{'开' if g else '关':>4}{A[k]['excess']:>+8}{B[k]['excess']:>+6}"
              f"{e_v11:>+8}{flag}")

    # 预测核对
    print("\n预注册的预测：")
    lone_keys = [k for k in keys if A[k]["cat"] == "lone"]
    p1 = all(gate(k, thr0) for k in lone_keys)
    print(f"  P1 lone 全部门开: {sum(gate(k, thr0) for k in lone_keys)}"
          f"/{len(lone_keys)}   " + ("过" if p1 else "**没过 —— 门杀了主结果**"))

    reg = [k for k in keys if k[0] in (23, 26, 27)
           and A[k].get("excess") is not None
           and abs(B[k]["excess"]) > abs(A[k]["excess"])]
    killed = [k for k in reg if not gate(k, thr0)]
    print(f"  P2 出回归的 23/26/27 行被门关掉: {len(killed)}/{len(reg)}   "
          + ("过" if len(killed) == len(reg) else "**部分留存，看明细**"))

    import statistics as st
    print(f"\n  P3 全体 MAE（有基数的行）：")
    print(f"     {'seed':>8}{'基线':>8}{'v1':>8}{'v1.1':>8}")
    ok3 = True
    v11_all, v1_all = [], []
    for sd in seeds:
        ks = [k for k in keys if k[1] == sd and A[k].get("excess") is not None]
        m_a = sum(abs(A[k]["excess"]) for k in ks) / len(ks)
        m_b = sum(abs(B[k]["excess"]) for k in ks) / len(ks)
        m_c = sum(abs((B[k] if gate(k, thr0) else A[k])["excess"])
                  for k in ks) / len(ks)
        v1_all.append(m_b); v11_all.append(m_c)
        ok3 &= m_c <= m_b + 1e-9
        print(f"     {sd:>8}{m_a:>8.2f}{m_b:>8.2f}{m_c:>8.2f}")
    print(f"     {'mean±std':>8}"
          f"{'':>8}"
          f"{sum(v1_all)/len(v1_all):>8.2f}"
          f"{sum(v11_all)/len(v11_all):>8.2f}"
          f"   (v1 std {st.stdev(v1_all):.2f} -> v1.1 std {st.stdev(v11_all):.2f})")
    print("     " + ("P3 过：每个 seed 上 v1.1 <= v1" if ok3
                     else "**P3 没过：有 seed 上门反而变差**"))

    print(f"\n  P4 阈值敏感度：")
    for thr in a.thr:
        lg = sum(gate(k, thr) for k in lone_keys)
        ks = [k for k in keys if A[k].get("excess") is not None]
        m = sum(abs((B[k] if gate(k, thr) else A[k])["excess"]) for k in ks) / len(ks)
        print(f"     thr={thr:<5} lone 门开 {lg}/{len(lone_keys)}   全体 MAE {m:.2f}")

    print("\n门的代价：推理期在基图上跑一次检测器，~2s / 张（75s 的 +2.7%）。")
    print("合成的合法性：门只依赖基图（90/90 逐字节同），门关行走原版路径"
          "（15/15 逐字节同背书）。")


if __name__ == "__main__":
    main()
