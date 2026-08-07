"""批跑用的 prompt 集。30 条，六类，每类 5 条。

为什么要分类而不是随便抽 30 条：seed 77 那一张给了一个假设——幻影出现在
"基图信息不足、但语义上适合放主体"的区域。要验证或推翻它，输入必须在
"有没有主体""主体多不多""背景空不空"这几个维度上拉开。随机抽 LAION
prompt 做不到这一点，而且这一步的目的是找失效，不是报指标。

subject 字段是给检测器用的名词。留空表示 prompt 里没有可数的主体——
那一类恰恰是最要紧的对照：**基图里一个人都没有，高分辨率会不会凭空造出人**。
"""

PROMPTS = [
    # ---- lone: 单主体，大面积空背景。幻影最容易看见的一类 ----
    ("lone", "person",
     "a photograph of a lone hiker standing on a rocky ridge, vast forested valley "
     "and distant snow mountains behind, golden hour"),
    ("lone", "person",
     "a single surfer paddling out on a wide empty ocean at dawn, long horizon"),
    ("lone", "person",
     "one astronaut walking across a vast red desert plain, distant mesas, clear sky"),
    ("lone", "boat",
     "a small wooden fishing boat alone on a huge calm lake, mist, mountains far behind"),
    ("lone", "bird",
     "a single eagle soaring high above an enormous canyon, empty sky"),

    # ---- crowd: 已经有很多同类物体。多一个看不出来，但"该有几个"会崩 ----
    ("crowd", "person",
     "a busy street market at dusk, dozens of shoppers between stalls, warm lanterns"),
    ("crowd", "person",
     "a packed concert audience seen from the stage, hands raised, stage lights"),
    ("crowd", "penguin",
     "a colony of penguins on an antarctic shore, hundreds of birds, overcast"),
    ("crowd", "sheep",
     "a large flock of sheep spread across a green hillside, stone walls"),
    ("crowd", "car",
     "an aerial view of a highway interchange at rush hour, many cars"),

    # ---- empty: prompt 里【没有】可数主体。检测器应当在基图上数到 0 ----
    ("empty", "person",
     "an empty snow field under heavy overcast, no people, no structures, flat light"),
    ("empty", "person",
     "a vast sand dune landscape at noon, wind ripples, nothing else"),
    ("empty", "person",
     "the surface of a calm sea reaching the horizon, uniform grey sky"),
    ("empty", "person",
     "a dense uniform pine forest canopy seen from above, no clearings"),
    ("empty", "person",
     "a smooth glacier surface with faint blue crevasses, no landmarks"),

    # ---- texture: 密集重复纹理。这一类看的是"重复"而不是"多一个物体" ----
    ("texture", "flower",
     "a field of wildflowers stretching to the horizon, thousands of blooms"),
    ("texture", "book",
     "a library wall of bookshelves floor to ceiling, thousands of spines"),
    ("texture", "window",
     "the glass facade of a skyscraper filling the frame, thousands of windows"),
    ("texture", "leaf",
     "a close forest floor covered in fallen autumn leaves, full frame"),
    ("texture", "rock",
     "a scree slope of loose grey rocks, full frame, overcast light"),

    # ---- portrait: ScaleDiff 自己点名 "inconsistent local content when
    #      generating sharp close-up images" ----
    ("portrait", "person",
     "a close-up portrait of an elderly fisherman, weathered skin, soft window light"),
    ("portrait", "person",
     "a studio headshot of a young woman with freckles, plain grey backdrop"),
    ("portrait", "eye",
     "an extreme close-up of a human eye, iris texture, eyelashes"),
    ("portrait", "hand",
     "a close-up of two hands holding a ceramic cup, shallow depth of field"),
    ("portrait", "cat",
     "a close-up of a tabby cat face, whiskers sharp, blurred garden behind"),

    # ---- structure: 有强规则结构，畸变一眼可见 ----
    ("structure", "building",
     "a symmetric baroque cathedral facade, straight-on view, blue hour"),
    ("structure", "bridge",
     "a long steel truss bridge crossing a wide river, side view"),
    ("structure", "clock",
     "an antique pocket watch open on a wooden table, roman numerals"),
    ("structure", "keyboard",
     "a mechanical keyboard photographed from directly above, full frame"),
    ("structure", "staircase",
     "a spiral staircase seen from directly below, concentric railings"),
]

NEGATIVE = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
