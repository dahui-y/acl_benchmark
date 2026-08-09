"""泄漏检查：去主体 prompt 里还残留多少主体语义。

诊断（三个 seed 的批跑之后）：23/26/27 三处回归是同一个缺陷 ——
strip_subject 只删名词短语，留下了暗示主体的词：

    23  剩 "a close-up of holding a ceramic cup..."   holding 暗示手
    26  剩 "crossing a wide river, side view"         crossing 暗示桥
    27  剩 "a wooden table, roman numerals"           roman numerals 是表盘

而 lone 五条摘得干净，那一格才降了 68%。规则试图用词法做语义判断，
所以会一直漏。修法要把判据从"删对了没有"换成**"删干净了没有"** ——
用文本编码器量 sim(去主体 prompt, 主体词)，高就是没删干净。

这个脚本先验证诊断（不改方法，不跑生成）：

预注册的判据（写在跑之前）：
    ① 已知泄漏的 23 / 26 / 27，其 sim(去主体, 主体) 排名应当在
       lone 五条（00-04）之前 —— 即泄漏行的相似度**系统性更高**；
    ② lone 五条应当聚在低相似度端。

    过 -> 诊断成立，可以把这个相似度做成方法里的自动泄漏门
          （高于阈值 -> 继续删或退回不干预），那是闭环不是又一条规则。
    不过 -> 文本相似度探不到这种泄漏，修法得另想（比如用去主体 prompt
          单独生成一张小图数主体，贵但直接）。

用的是缓存里那个 open_clip 文本塔（和 clip_score 同一个），不用 GPU 生成。

    python scalediff_probe/leak_check.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_score import load_clip                     # noqa: E402
from prompts import PROMPTS                          # noqa: E402
from subject_phrases import HEADS, strip_subject     # noqa: E402

# 已知泄漏行（三个 seed 的批跑里反复出现回归的那三条）
KNOWN_LEAKY = {23, 26, 27}


@torch.no_grad()
def text_emb(clip, texts):
    """只用文本塔。返回归一化后的 (N, D)。"""
    if hasattr(clip, "tok"):                          # OpenClip
        tt = clip.tok(texts).to("cuda")
        e = clip.m.encode_text(tt)
    else:                                             # HFClip
        inp = clip.p(text=texts, return_tensors="pt",
                     padding=True, truncation=True).to("cuda")
        e = clip.m.get_text_features(**inp)
    return e / e.norm(dim=-1, keepdim=True)


def main():
    clip = load_clip()

    rows = []
    for i, ((cat, subj, prompt), head) in enumerate(zip(PROMPTS, HEADS)):
        if head is None:
            continue
        alt, removed = strip_subject(prompt, head)
        rows.append((i, cat, head, subj, prompt, alt))

    # sim(去主体 prompt, 主体词)。主体词用检测器用的那个 subject 名词
    # （person/boat/bird/...），不用 head —— head 是 prompt 里的词，
    # 和 alt 的词面重叠会干扰；subject 是独立给出的类别词。
    alts = [r[5] for r in rows]
    subs = [f"a photo of a {r[3]}" for r in rows]
    fulls = [r[4] for r in rows]

    ea = text_emb(clip, alts)
    es = text_emb(clip, subs)
    ef = text_emb(clip, fulls)

    leak = (ea * es).sum(-1)                          # 去主体 vs 主体
    ceil = (ef * es).sum(-1)                          # 原 prompt vs 主体（上界参照）
    # 归一残留：去掉不同主体词本身基线相似度的差异
    resid = leak / ceil.clamp(min=1e-6)

    print(f"\n{'idx':<5}{'cat':<11}{'head':<13}{'sim(去,主)':>12}"
          f"{'sim(原,主)':>12}{'残留比':>9}")
    print("-" * 64)
    order = sorted(range(len(rows)), key=lambda k: -float(resid[k]))
    for k in order:
        i, cat, head, subj, _, alt = rows[k]
        mark = "  << 已知泄漏" if i in KNOWN_LEAKY else (
               "  (lone)" if cat == "lone" else "")
        print(f"{i:<5}{cat:<11}{head:<13}{float(leak[k]):>12.4f}"
              f"{float(ceil[k]):>12.4f}{float(resid[k]):>9.3f}{mark}")

    # 判据 ①：已知泄漏行的残留比排名
    ranks = {rows[k][0]: pos for pos, k in enumerate(order)}
    lone_pos = [ranks[r[0]] for r in rows if r[1] == "lone"]
    leaky_pos = [ranks[i] for i in KNOWN_LEAKY if i in ranks]
    n = len(rows)
    print(f"\n判据①  已知泄漏行的排名（0 = 残留最高，共 {n} 行）: "
          f"{sorted(leaky_pos)}")
    print(f"        lone 五条的排名: {sorted(lone_pos)}")
    ok1 = max(leaky_pos) < min(lone_pos) if leaky_pos and lone_pos else False
    strict = all(p < min(lone_pos) for p in leaky_pos)
    print("        " + ("-> 过：泄漏行全部排在 lone 之前，诊断成立" if strict else
                        "-> 部分重叠，看具体间隔再定" ))

    lv = [float(resid[k]) for k in range(n) if rows[k][0] in KNOWN_LEAKY]
    ov = [float(resid[k]) for k in range(n) if rows[k][1] == "lone"]
    if lv and ov:
        gap = sum(lv)/len(lv) - sum(ov)/len(ov)
        print(f"\n残留比：已知泄漏 {sum(lv)/len(lv):.3f}   "
              f"lone {sum(ov)/len(ov):.3f}   间隔 {gap:+.3f}")
        if strict and gap > 0.1:
            print("-> 泄漏行系统性更高，可以把残留比做成自动泄漏门。")
        else:
            print("-> **判据没过（间隔为负或两簇重叠）：文本相似度探不到这种泄漏，"
                  "泄漏门这条路不通。**\n   按预注册的备选：要么用去主体 prompt "
                  "生成小图数主体（贵但直接），\n   要么靠适用性门让这些行根本不被干预"
                  "（gate_eval.py）。")


if __name__ == "__main__":
    main()
