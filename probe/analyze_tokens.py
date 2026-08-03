"""Redo the salience measurement on token sequences instead of pooled vectors.

Why this exists: every number in `analyze.py` is a statement about a mean-pooled
vector, but diffusion models condition on the full token sequence through
cross-attention. Aspect markers ("has", "tried", "about to") are localised in a
few tokens. Mean pooling dilutes them across the whole sentence; cross-attention
does not have to. So the pooled result could understate how visible aspect is,
and the headline finding would be an artefact of the measurement rather than a
property of the encoder.

Two token-level measures, both anchored on the same `filler` control so they are
directly comparable to the pooled SNS:

  chamfer   Symmetric nearest-neighbour distance between the two token sets.
            Position-agnostic, so it handles the differing sentence lengths
            without needing an alignment.

  novelty   For each token of the variant, 1 - (best cosine to any reference
            token). A token that has no counterpart in the reference scores
            high. `max_novelty` is the single most distinctive token -- which is
            exactly what cross-attention is free to attend to. If the aspect
            conditions do not beat the filler control here either, then the
            pooled result was not a pooling artefact.

Padding is recovered from the stored states: encode.py zeroes padded positions,
so a position with ~zero norm is padding.

Usage:
    python analyze_tokens.py --emb emb_umt5xxl_tok.npz --stimuli stimuli.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ASPECTS = ["prog", "perf", "result", "prospective", "failed", "atelic"]
CONTROLS = ["paraphrase_min", "paraphrase", "other_verb"]
REFERENCE = "prog"
ANCHOR = "filler"


def real_tokens(seq, eps=1e-4):
    """Drop padded positions. encode.py zeroes them, so norm ~ 0 marks padding."""
    norms = np.linalg.norm(seq, axis=-1)
    kept = seq[norms > eps]
    return kept if len(kept) else seq[:1]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def similarity_matrix(a, b):
    return unit(a.astype(np.float32)) @ unit(b.astype(np.float32)).T


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance between two token sets."""
    sim = similarity_matrix(a, b)
    return float(0.5 * ((1 - sim.max(axis=1)).mean() + (1 - sim.max(axis=0)).mean()))


def novelty(variant, reference):
    """Per-token distance to its best match in the reference.

    High values mark tokens the reference has no counterpart for -- the tokens
    cross-attention could latch onto.
    """
    sim = similarity_matrix(variant, reference)
    return 1 - sim.max(axis=1)


def load(emb_path, stimuli_path):
    data = np.load(emb_path, allow_pickle=True)
    if "tokens" not in data.files:
        raise SystemExit("this npz has no token states; re-run encode.py with --save-tokens")
    table = {}
    for seq, item, cond in zip(data["tokens"], data["item_ids"], data["conditions"]):
        table[(int(item), str(cond))] = real_tokens(seq)
    stimuli = [json.loads(l) for l in Path(stimuli_path).read_text().splitlines() if l.strip()]
    return data, table, stimuli


def measure(table, stimuli):
    rows = {}
    for cond in ASPECTS + CONTROLS + [ANCHOR]:
        if cond == REFERENCE:
            continue
        ch, mx, mean_nov = [], [], []
        for rec in stimuli:
            item = rec["item_id"]
            ref = table.get((item, REFERENCE))
            var = table.get((item, cond))
            if ref is None or var is None:
                continue
            ch.append(chamfer(ref, var))
            nov = novelty(var, ref)
            mx.append(float(nov.max()))
            mean_nov.append(float(nov.mean()))
        if ch:
            rows[cond] = {"chamfer": np.array(ch), "max_novelty": np.array(mx),
                          "mean_novelty": np.array(mean_nov)}
    return rows


def report(rows, pooled_sns=None):
    anchor = rows.get(ANCHOR)
    if anchor is None:
        raise SystemExit("filler anchor missing from the stimuli")

    print("\nTOKEN-LEVEL SALIENCE  — anchored on the same `filler` control")
    header = (f"{'condition':<15} {'chamfer':>9} {'SNS_tok':>9} {'maxNov':>8} "
              f"{'SNS_nov':>9} {'p(vs fill)':>11}")
    if pooled_sns:
        header += f" {'SNS_pooled':>11} {'shift':>7}"
    print(header)

    out = {}
    for cond in ASPECTS + CONTROLS:
        if cond not in rows or cond == REFERENCE:
            continue
        ch = rows[cond]["chamfer"]
        mx = rows[cond]["max_novelty"]
        sns_tok = float(np.mean(ch / np.where(anchor["chamfer"] < 1e-9, np.nan, anchor["chamfer"])))
        sns_nov = float(np.mean(mx / np.where(anchor["max_novelty"] < 1e-9, np.nan,
                                              anchor["max_novelty"])))
        try:
            pval = float(wilcoxon(ch, anchor["chamfer"]).pvalue)
        except ValueError:
            pval = float("nan")

        line = (f"{cond:<15} {ch.mean():>9.4f} {sns_tok:>9.2f} {mx.mean():>8.4f} "
                f"{sns_nov:>9.2f} {pval:>11.2e}")
        if pooled_sns and cond in pooled_sns:
            p = pooled_sns[cond]
            line += f" {p:>11.2f} {sns_tok - p:>+7.2f}"
        marker = "   <- control" if cond in CONTROLS else ""
        print(line + marker)
        out[cond] = {"chamfer": float(ch.mean()), "sns_token": sns_tok,
                     "max_novelty": float(mx.mean()), "sns_novelty": sns_nov,
                     "p_vs_filler": pval}

    aspect = [out[c]["sns_token"] for c in ASPECTS if c in out]
    below = sum(1 for v in aspect if v < 1.0)
    mean_tok = float(np.mean(aspect))
    print(f"\n  mean token-level SNS over {len(aspect)} aspect conditions: {mean_tok:.2f}")
    print(f"  conditions below the filler anchor: {below}/{len(aspect)}")

    print("\n" + "=" * 74)
    if below >= len(aspect) - 1 and mean_tok < 1.0:
        verdict = "POOLED RESULT SURVIVES"
        print(f"VERDICT: {verdict}")
        print("The finding is not a pooling artefact. Even when the sentence is compared")
        print("token by token -- the representation cross-attention actually consumes --")
        print("aspect moves the conditioning less than inert filler words do.")
    else:
        verdict = "POOLED RESULT DOES NOT SURVIVE"
        print(f"VERDICT: {verdict}")
        print("At token level the aspect conditions are as distinctive as, or more than,")
        print("the filler control. The pooled measurement understated aspect by diluting")
        print("markers across the sentence.")
        print("  -> The headline claim must be rewritten. Cross-attention can see these")
        print("     tokens; low pooled salience does not mean the generator lacks the signal.")
    print("=" * 74)
    return out, mean_tok, below, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True, help="npz written with --save-tokens")
    ap.add_argument("--stimuli", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--pooled-result", type=Path, default=None,
                    help="result json from analyze.py, to print the pooled vs token shift")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data, table, stimuli = load(args.emb, args.stimuli)
    print(f"model={data['model']}  items={len(stimuli)}  (token-level, last layer)")

    pooled_sns = None
    if args.pooled_result and args.pooled_result.exists():
        pooled = json.loads(args.pooled_result.read_text())
        pooled_sns = {k: v["sns"] for k, v in pooled.get("geometry", {}).items() if "sns" in v}

    rows = measure(table, stimuli)
    out, mean_tok, below, verdict = report(rows, pooled_sns)

    if args.out:
        args.out.write_text(json.dumps({
            "model": str(data["model"]), "level": "token", "verdict": verdict,
            "mean_sns_token": mean_tok, "conditions_below_filler": f"{below}/{len(ASPECTS) - 1}",
            "conditions": out,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
