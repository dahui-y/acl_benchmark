"""Pick the OSCBench prompts for the HunyuanVideo-1.5 problem check.

The question this run answers is narrow: does HunyuanVideo-1.5, on our card and
at 480p, actually show the state-change failure OSCBench reports at 720p? Until
that is seen here, everything downstream is planning on top of someone else's
table.

The selection is stratified by OSCBench's OWN diagnosis rather than at random,
so every band it says exists is represented and disagreement with its Figure 5
is visible:

  weak    peeling / coating / squeezing / crushing / mashing  -- their worst
  mid     chopping / slicing                                  -- their Figure 4
                                                                 example family
  strong  rolling / melting / browning                        -- their best

All three scenario splits (regular / novel / compositional) are covered because
the novel split is where they report the hardest degradation, and a check that
skipped it could pass while the real problem lives there.

Selection is deterministic: first match per (verb, split) slot in file order.
No randomness, so the subset is reproducible from the repo alone.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (band, verb, split, how many)
PLAN = [
    ("weak", "peeling", "regular", 2), ("weak", "peeling", "novel", 1),
    ("weak", "peeling", "compositional", 1),
    ("weak", "coating", "regular", 2), ("weak", "coating", "novel", 2),
    ("weak", "squeezing", "regular", 1), ("weak", "crushing", "regular", 1),
    ("weak", "mashing", "regular", 1), ("weak", "mashing", "novel", 1),
    ("mid", "chopping", "regular", 1), ("mid", "chopping", "novel", 1),
    ("mid", "slicing", "regular", 1), ("mid", "slicing", "compositional", 1),
    ("strong", "rolling", "regular", 2), ("strong", "rolling", "novel", 1),
    ("strong", "rolling", "compositional", 1),
    ("strong", "melting", "regular", 2), ("strong", "browning", "regular", 2),
]


def gerunds(line):
    hits = re.findall(r"\b(?:is|are|then|and|,)\s+(\w+ing)\b", line)
    return set(hits)


def main():
    out = Path(__file__).parent / "osc_prompts.jsonl"
    rows, used = [], set()
    for band, verb, split, k in PLAN:
        lines = (ROOT / "prompt_splits" / f"prompts_{split}.txt").read_text().splitlines()
        taken = 0
        for line in lines:
            line = line.strip()
            if not line or line in used or verb not in gerunds(line):
                continue
            rows.append({"id": f"osc{len(rows):03d}", "band": band,
                         "verb": verb, "split": split, "prompt": line})
            used.add(line)
            taken += 1
            if taken == k:
                break
        if taken < k:
            print(f"WARNING: wanted {k} x {verb}/{split}, found {taken}")

    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = {b: sum(1 for r in rows if r["band"] == b) for b in ("weak", "mid", "strong")}
    print(f"{len(rows)} prompts -> {out}   (weak {n['weak']} / mid {n['mid']} "
          f"/ strong {n['strong']})")
    for r in rows:
        print(f"  {r['id']}  {r['band']:<7}{r['verb']:<10}{r['split']:<14}{r['prompt']}")


if __name__ == "__main__":
    main()
