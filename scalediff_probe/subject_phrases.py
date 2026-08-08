"""自动从 prompt 里摘掉主体短语，得到 method_v1 要的第二套嵌入。

为什么必须自动：v1 那一张图用的 PROMPT_NOSUBJ 是手写的。手写能证明机制，
不能当方法 —— 审稿人第一句就会问"每条 prompt 都要人再写一遍？"。
批跑 30 条也没法手写着来。

规则（不依赖任何 NLP 包，确定性，同输入同输出）：

    给定主体中心词 head，
    向左吃掉它的修饰语，遇到介词/逗号/句首停；
        例外：遇到 "of" 且它前面是数量词（flock / colony / dozens ...）
        就跨过去，把 "a flock of" 一起吃掉。
    向右最多看 4 个词找一个介词，找到就把介词一起吃掉
        （"standing on" 里的 "on" 归主体短语，后面的景物短语自然接上）；
        4 个词内没有介词就不向右延伸。
    最后 tidy() 收拾悬空的介词和逗号。

    "a photograph of a lone hiker standing on a rocky ridge, ..."
        -> 摘掉 "a lone hiker standing on"
        -> "a photograph of a rocky ridge, ..."
    这与 v1 手写的那一句**逐字相同**，所以自动化不会改变已经拿到的那个结果。

HEADS 是 prompt 集里每条的主体中心词。它不是"标注"——是从 prompt 字符串
本身读出来的语法信息，换成任意现成的名词短语分析器是一行替换；这里不引依赖，
是因为服务器连不上外网。empty 那五条没有可数主体，值为 None，
v1 在这几条上自动退化为原版（subject_token_ids 返回空 -> 不装门控）。

    python scalediff_probe/subject_phrases.py        # 打印全部 30 条，先过目
"""

PREPS = {
    "of", "on", "in", "at", "above", "across", "over", "under", "below",
    "behind", "beside", "near", "between", "through", "along", "around",
    "from", "to", "with", "against", "onto", "into", "by", "beneath",
}

# 只有这些词后面的 "of" 才跨得过去。"a close-up portrait of an elderly
# fisherman" 里的 "of" 不能跨 —— 跨了会把 "portrait" 也吃掉。
QUANT = {
    "dozens", "hundreds", "thousands", "flock", "colony", "group", "herd",
    "pile", "couple", "pair", "row", "rows", "stack", "field", "wall",
    "handful", "cluster", "swarm", "crowd", "lot",
}

# prompt 自己声明的主体个数。**这是免费的真值标签，它写在输入里。**
#
# 为什么需要它：`delta = 高分辨率计数 - 基图计数` 要跨分辨率比较，而
# scale_check 实测两档之间还有 2.00 个物体的残余漂移（tile=width/4 已经把
# 它从 4.25 压下来了，但压不到 0）—— 基图被系统性少数，delta 被系统性做高。
# 换成 `excess = 高分辨率计数 - 基数`，参照物是 prompt 而不是另一张图，
# 尺度偏差整个消失。
#
# "a lone hiker" / "a single surfer" / "one astronaut" -> 1
# "an empty snow field, no people"                     -> 0
# "two hands"                                          -> 2
# crowd / texture 的 prompt 没说数量 -> None，不进主指标
CARD = [
    1, 1, 1, 1, 1,                    # lone: lone / single / one / a ... boat / a single eagle
    None, None, None, None, None,     # crowd: 没给数量
    0, 0, 0, 0, 0,                    # empty: no people / nothing else / no landmarks
    None, None, None, None, None,     # texture: thousands of ... 没给确切数量
    1, 1, 1, 2, 1,                    # portrait: 一个渔夫 / 一个女子 / 一只眼 / 两只手 / 一张猫脸
    1, 1, 1, 1, 1,                    # structure: 单个建筑物 / 桥 / 表 / 键盘 / 楼梯
]

# (idx, head)。None = prompt 里没有可数主体
HEADS = [
    "hiker", "surfer", "astronaut", "boat", "eagle",                 # lone
    "shoppers", "audience", "penguins", "sheep", "cars",             # crowd
    None, None, None, None, None,                                    # empty
    "wildflowers", "bookshelves", "windows", "leaves", "rocks",      # texture
    "fisherman", "woman", "eye", "hands", "face",                    # portrait
    "facade", "bridge", "watch", "keyboard", "staircase",            # structure
]


def _base(w):
    return w.strip(",.;:").lower()


def _ends_clause(w):
    return w.endswith(",") or w.endswith(".") or w.endswith(";")


def _join(left, right):
    """在【摘除的接缝处】收拾悬空介词，不碰句子别处。

    全局清理是错的：[00] 的 "snow mountains behind, golden hour" 和 [28] 的
    "directly above, full frame" 里，behind / above 是景物词，只是碰巧也是
    介词。只有接缝左边那一个才可能是被摘剩下的。
    """
    # 接缝左边是介词、右边已经是子句边界 -> 那个介词是被摘剩下的
    while left and _base(left[-1]) in PREPS and (not right or right[0].startswith(",")):
        left = left[:-1]
    s = " ".join(left + right)
    s = s.replace(" ,", ",").replace(",,", ",")
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.strip().strip(",").strip()
    while s.lower().startswith(("and ", "of ")):
        s = s.split(" ", 1)[1]
    return s


def strip_subject(prompt, head):
    """返回 (去主体 prompt, 被摘掉的那一段)。head 为 None 时原样返回。"""
    if not head:
        return prompt, ""
    words = prompt.split()
    hi = next((i for i, w in enumerate(words) if _base(w) == head.lower()), -1)
    if hi < 0:
        return prompt, ""

    # 向左
    lo = hi
    i = hi - 1
    while i >= 0:
        w = words[i]
        if _ends_clause(w):
            break
        b = _base(w)
        if b == "of":
            if i - 1 >= 0 and not _ends_clause(words[i - 1]) \
                    and _base(words[i - 1]) in QUANT:
                lo, i = i - 1, i - 2      # 跨过 "<quant> of"
                continue
            break
        if b in PREPS:
            break
        lo, i = i, i - 1

    # 向右：4 个词内找介词，找到才延伸
    hj = hi
    if not _ends_clause(words[hi]):
        for j in range(hi + 1, min(hi + 5, len(words))):
            if _base(words[j]) in PREPS:
                hj = j
                break
            if _ends_clause(words[j]):
                break

    removed = " ".join(words[lo:hj + 1])
    right = words[hj + 1:]
    # 摘掉的最后一个词自带逗号（"...autumn leaves," ）-> 逗号是句子的，留下
    if _ends_clause(words[hj]) and right:
        right = [","] + right
    return _join(words[:lo], right), removed


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompts import PROMPTS

    assert len(HEADS) == len(PROMPTS), (len(HEADS), len(PROMPTS))
    for i, ((cat, subj, p), head) in enumerate(zip(PROMPTS, HEADS)):
        alt, removed = strip_subject(p, head)
        flag = "  << 原样（无主体）" if head is None else ""
        print(f"[{i:02d}] {cat:<10} head={str(head):<13} 摘掉: {removed!r}{flag}")
        print(f"     原:  {p}")
        print(f"     去:  {alt}\n")
