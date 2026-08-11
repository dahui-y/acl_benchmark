"""量一下 caption 被 CLIP 截断的程度 —— 它决定触发变量该拿哪段文本去 ground。

base_run 的日志里跳出：

    Token indices sequence length is longer than the specified maximum (95 > 77)
    truncated: ['z 3 gw ( posted by sophie kerr on 1 1 may. 2 0 1 6 )']

被截掉的是网页样板（图片 ID、发布者、日期），而且 `1 1 may. 2 0 1 6` 每个
字符各占一个 token —— 所以一条 60 词以内的 caption 也能撑到 95 token。

**两面要分开：**

  · **对生成无害，不用管。** SDXL 永远在 77 token 处截断，ScaleDiff /
    AccDiffusion 拿同样的 LAION caption 也是同样截断。这是协议的一部分，
    不是我们引入的偏差。
  · **对触发变量有害。** `trigger_select` 把整条 caption 喂给 GroundingDINO，
    而**超出 77 token 的那段根本没参与生成** —— 图里不可能有它描述的东西。
    拿没条件过图像的文本去 ground 图像，多出来的框全是噪声，
    而这个量对假阳性框恰恰最敏感（见 §5.4 里 `[10]` 那个反例）。

原则一句话：**ground 的文本应当恰好等于条件过图像的那段。**

    python scalediff_probe/caption_audit.py --split tune
"""

import argparse
import json
import os
import sys
from pathlib import Path

CKPT = "stabilityai/stable-diffusion-xl-base-1.0"
MAXLEN = 77


def load_tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(CKPT, subfolder="tokenizer")


def conditioned_text(tok, caption, maxlen=MAXLEN):
    """截到**真正条件过图像**的那段文本。

    CLIP 的 77 含首尾各一个特殊 token，所以正文只有 75 个。
    解码回字符串是为了喂给 GroundingDINO（它要的是文本不是 id）。
    """
    ids = tok(caption, add_special_tokens=False)["input_ids"]
    if len(ids) <= maxlen - 2:
        return caption, len(ids), False
    return tok.decode(ids[:maxlen - 2]), len(ids), True


def main():
    root = Path(os.environ.get("SD_OUT", "./scalediff_out"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(root / "laion_base"))
    ap.add_argument("--split", default=None, choices=["tune", "eval"])
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (Path(a.base) / "manifest.jsonl").open()]
    if a.split:
        rows = [r for r in rows if r.get("split") == a.split]
    if not rows:
        sys.exit("没有符合条件的记录")

    tok = load_tokenizer()
    lens, cut = [], []
    for r in rows:
        _, n, trunc = conditioned_text(tok, r["prompt"])
        lens.append(n)
        if trunc:
            cut.append((n, r["idx"], r["prompt"]))

    lens.sort()
    q = lambda f: lens[min(int(len(lens) * f), len(lens) - 1)]
    print(f"{len(rows)} 条  token 数分位："
          f"p0={lens[0]}  p50={q(.5)}  p90={q(.9)}  p99={q(.99)}  p100={lens[-1]}")
    print(f"**超过 {MAXLEN-2} 个正文 token（被截断）：{len(cut)} 条 = "
          f"{len(cut)/len(rows):.1%}**")

    if cut:
        cut.sort(reverse=True)
        print(f"\n最长的 {min(a.show, len(cut))} 条（看被截掉的是不是样板文本）：")
        for n, idx, p in cut[:a.show]:
            keep, _, _ = conditioned_text(tok, p)
            print(f"  [{idx}] {n} tok")
            print(f"      条件过图的： {keep[:96]}")
            print(f"      被截掉的：   ...{p[len(keep):][:96]}")

    print(f"""
读法：
  被截掉的多是图片 ID / 发布者 / 日期 / 站名  -> 截断是**好事**，
      它把样板噪声挡在了条件之外；触发变量照做即可（ground 前 {MAXLEN-2} 个 token）。
  被截掉的是场景描述的实质部分            -> 说明 --max-words 60 放得太宽,
      **要记进局限**：那部分内容 prompt 里有、图里没有。
  截断比例很低（< 5%）                    -> 影响有限，改法仍然照做，成本为零。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
