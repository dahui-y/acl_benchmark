"""Analyse the encoder embeddings and return the verdict that selects the paper's framing.

Two measures, in the order of what they settle.

(a) Geometry -- how far does changing the aspect move the conditioning vector?

    Raw cosine distance is not interpretable on its own: it mostly tracks how
    many tokens changed. Running this on t5-base made that concrete. Swapping
    the verb ("crushing" -> "rolling") is a one-token edit and moved the vector
    LESS than the aspect conditions, which rewrite several tokens, so anchoring
    a ratio on that control produced nonsense.

    The anchor is therefore `filler`: the same event, the same progressive
    aspect, plus a few semantically inert tokens. It buys distance through
    surface change alone. The reported statistic is

        SNS = d(prog, aspect_condition) / d(prog, filler)

    SNS < 1 means changing whether the event culminates moves the conditioning
    vector *less* than appending filler words does.

    Also reported: `paraphrase_min` (one-token synonym, the tight noise floor),
    `paraphrase` (looser floor), `other_verb` (different event), and the
    correlation between distance and token edit distance.

(b) Linear probe -- is aspect recoverable at all from the pooled vector, with
    whole verbs (and separately whole objects) held out so the probe cannot key
    on lexical identity? Compared against a label-permutation floor and a
    bag-of-words ceiling.

    Read this one carefully. T5-style encoders preserve token identity, and the
    aspect conditions carry distinct surface cues ("has", "about to", "tried"),
    so high accuracy is expected and only tells you the information is present.
    Accuracy far BELOW the bag-of-words ceiling is the informative case: it means
    the encoder is attenuating a distinction its input carried.

(c) Layer localisation -- where along the stack the distinction appears or is lost.

Usage:
    python analyze.py --emb emb_umt5xxl.npz --stimuli stimuli.jsonl --all-layers
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Single source of truth for the condition set, so adding an axis to the suite
# does not silently leave it unmeasured here.
from build_stimuli import (  # noqa: E402
    ASPECT_CONDITIONS,
    PHASE_CONDITIONS,
    TELICITY_CONDITIONS,
)

AXES = [("aspect", ASPECT_CONDITIONS), ("telicity", TELICITY_CONDITIONS),
        ("phase", PHASE_CONDITIONS)]
TARGETS = [c for _, conds in AXES for c in conds]
CONTROLS = ["paraphrase_min", "paraphrase", "other_verb"]
REFERENCE = "prog"
ANCHOR = "filler"


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float(1.0 - (a * b).sum(-1))


def token_edit_distance(s1, s2):
    """Normalised token-level Levenshtein."""
    a, b = s1.lower().split(), s2.lower().split()
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def index_embeddings(data, layer_key):
    table = {}
    for vec, item, cond in zip(data[layer_key], data["item_ids"], data["conditions"]):
        table[(int(item), str(cond))] = vec
    return table


def distances(table, stimuli):
    """-> {(item_id, condition): cosine distance from the progressive reference}"""
    out = {}
    for rec in stimuli:
        item = rec["item_id"]
        ref = table.get((item, REFERENCE))
        if ref is None:
            continue
        for cond in rec["texts"]:
            if cond != REFERENCE and (item, cond) in table:
                out[(item, cond)] = cosine(ref, table[(item, cond)])
    return out


def surface_check(dists, stimuli):
    d, e = [], []
    for rec in stimuli:
        texts = rec["texts"]
        for cond, text in texts.items():
            key = (rec["item_id"], cond)
            if key in dists:
                d.append(dists[key])
                e.append(token_edit_distance(texts[REFERENCE], text))
    r = float(np.corrcoef(e, d)[0, 1])
    print(f"\n  surface check: cosine distance vs token edit distance  r={r:.3f}")
    print("  -> raw distances carry a surface-overlap component; that is why the anchor is")
    print("     `filler` (inert tokens, no aspect change) rather than a raw threshold.")
    return r


def summarise_geometry(dists, stimuli):
    print("\n(a) GEOMETRY  — distance from the progressive reference")
    print("    anchor = filler (same aspect + inert tokens);  SNS = d_cond / d_filler")
    print(f"{'condition':<15} {'d_cond':>9} {'d_filler':>9} {'SNS':>7} {'p(vs fill)':>11} {'edit':>6}")

    summary = {}
    ordered = ([(name, c) for name, conds in AXES for c in conds]
               + [("control", c) for c in CONTROLS])
    current_axis = None
    for axis_name, cond in ordered:
        if cond == REFERENCE:
            continue
        if axis_name != current_axis:
            print(f"  -- {axis_name} --")
            current_axis = axis_name
        d_c, d_f, edits = [], [], []
        for rec in stimuli:
            key, anchor_key = (rec["item_id"], cond), (rec["item_id"], ANCHOR)
            if key in dists and anchor_key in dists:
                d_c.append(dists[key])
                d_f.append(dists[anchor_key])
                edits.append(token_edit_distance(rec["texts"][REFERENCE], rec["texts"][cond]))
        if not d_c:
            continue
        d_c, d_f = np.array(d_c), np.array(d_f)
        sns = float(np.mean(d_c / np.where(d_f < 1e-9, np.nan, d_f)))
        try:
            pval = float(wilcoxon(d_c, d_f).pvalue)
        except ValueError:
            pval = float("nan")

        marker = "   <- control" if axis_name == "control" else ""
        print(f"{cond:<15} {d_c.mean():>9.4f} {d_f.mean():>9.4f} {sns:>7.2f} {pval:>11.2e} "
              f"{np.mean(edits):>6.3f}{marker}")
        summary[cond] = {"d_cond": float(d_c.mean()), "d_filler": float(d_f.mean()),
                         "sns": sns, "p_vs_filler": pval, "mean_edit": float(np.mean(edits))}

    print()
    for name, conds in AXES:
        vals = [summary[c]["sns"] for c in conds if c in summary]
        if vals:
            print(f"  {name:<9} mean SNS {np.mean(vals):.2f}   "
                  f"below filler {sum(1 for v in vals if v < 1.0)}/{len(vals)}")

    aspect_sns = [summary[c]["sns"] for c in TARGETS if c in summary]
    overall = float(np.mean(aspect_sns))
    below = sum(1 for v in aspect_sns if v < 1.0)
    print(f"\n  overall: mean SNS {overall:.2f} over {len(aspect_sns)} target conditions, "
          f"{below} below the filler anchor")
    return summary, overall, below, len(aspect_sns)


# Pairs whose two members differ by one grammatical feature. Measuring these
# directly matters: two conditions can both sit near the noise floor relative to
# the progressive reference and still be far from *each other*. Only the direct
# distance answers whether the encoder separates them.
DIRECT_CONTRASTS = [
    # cross-telicity: should be large if the encoder represents telicity
    ("atelic", "telic_plural", "CROSS telicity: bare vs definite plural"),
    ("atelic", "telic_numeral", "CROSS telicity: bare plural vs numeral"),
    ("atelic_some", "telic_plural", "CROSS telicity: some vs the"),
    # within-telicity: small either way. If these match the cross pairs, the
    # effect is determiner-blindness in general, not telicity in particular.
    ("telic_plural", "telic_numeral", "WITHIN telic: the vs three"),
    ("atelic", "atelic_some", "WITHIN atelic: bare vs some"),
    # other axes, for scale
    ("perf", "result", "aspect: perfective vs resultant, differs by 'has'"),
    ("prog", "prospective", "aspect: does the event happen at all"),
    ("phase_begin", "phase_finish", "phase: onset vs culmination"),
]


def direct_contrasts(dists_pairwise, stimuli):
    """Distance within each minimal pair, read against the synonym-swap floor."""
    print("\n(a2) DIRECT CONTRASTS  — distance between the two members of a pair")
    print("     floor = d(prog, paraphrase_min), a reword with no meaning change")
    print(f"{'pair':<28} {'d_pair':>9} {'floor':>9} {'ratio':>7} {'p':>11}  note")

    floor = np.array([dists_pairwise[(r["item_id"], REFERENCE, "paraphrase_min")]
                      for r in stimuli
                      if (r["item_id"], REFERENCE, "paraphrase_min") in dists_pairwise])
    out = {}
    for a, b, note in DIRECT_CONTRASTS:
        vals = [dists_pairwise[(r["item_id"], a, b)] for r in stimuli
                if (r["item_id"], a, b) in dists_pairwise]
        if not vals or not len(floor):
            continue
        vals = np.array(vals)
        n = min(len(vals), len(floor))
        ratio = float(vals.mean() / floor.mean()) if floor.mean() > 1e-9 else float("nan")
        try:
            pval = float(wilcoxon(vals[:n], floor[:n]).pvalue)
        except ValueError:
            pval = float("nan")
        print(f"{a + ' vs ' + b:<28} {vals.mean():>9.4f} {floor.mean():>9.4f} "
              f"{ratio:>7.2f} {pval:>11.2e}  {note}")
        out[f"{a}|{b}"] = {"d_pair": float(vals.mean()), "floor": float(floor.mean()),
                           "ratio": ratio, "p": pval, "note": note}
    print("     ratio ~ 1 means the encoder separates the pair no more than a synonym swap")
    return out


def pairwise_distances(table, stimuli, wanted):
    out = {}
    for rec in stimuli:
        item = rec["item_id"]
        for a, b in wanted:
            va, vb = table.get((item, a)), table.get((item, b))
            if va is not None and vb is not None:
                out[(item, a, b)] = cosine(va, vb)
    return out


def probe(table, stimuli, group_key, seed=0, with_null=True):
    """with_null=False skips the label-permutation baseline.

    The null is only needed once per model; refitting it at every layer during
    the sweep doubles the cost of the expensive part for no extra information.
    """
    X, y, groups = [], [], []
    for rec in stimuli:
        for cond in TARGETS:
            key = (rec["item_id"], cond)
            if key in table:
                X.append(table[key])
                y.append(cond)
                groups.append(rec[group_key])
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    rng = np.random.default_rng(seed)
    accs, chance = [], []
    for train, test in cv.split(X, y, groups):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train]), y[train])
        accs.append(clf.score(scaler.transform(X[test]), y[test]))
        if with_null:
            null = LogisticRegression(max_iter=2000)
            null.fit(scaler.transform(X[train]), rng.permutation(y[train]))
            chance.append(null.score(scaler.transform(X[test]), y[test]))
    return (float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(chance)) if chance else float("nan"))


def bow_ceiling(stimuli, group_key):
    texts, y, groups = [], [], []
    for rec in stimuli:
        for cond in TARGETS:
            texts.append(rec["texts"][cond])
            y.append(cond)
            groups.append(rec[group_key])
    y, groups = np.array(y), np.array(groups)
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    accs = [
        LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te])
        for tr, te in cv.split(X, y, groups)
    ]
    return float(np.mean(accs))


def verdict(mean_sns, below, n_aspects, probe_acc, chance, bow):
    decodable = probe_acc > chance + 0.15
    low_magnitude = mean_sns < 1.0

    print("\n" + "=" * 74)
    if decodable and low_magnitude:
        head = "INFORMATION PRESENT BUT LOW-MAGNITUDE"
        print(f"VERDICT: {head}")
        print("Aspect is recoverable from the conditioning vector, yet changing it moves that")
        print(f"vector less than appending inert words does ({below}/{n_aspects} conditions below 1.0).")
        print("  -> The generator receives the signal at a magnitude it can easily ignore.")
        print("  -> Original framing holds. Add the mechanistic section, and the training-free")
        print("     difference-vector intervention is the natural follow-up.")
    elif not decodable:
        head = "ENCODER IS THE BOTTLENECK"
        print(f"VERDICT: {head}")
        print("Aspect is not linearly recoverable from the conditioning vector.")
        print("  -> Reframe from semantics to architecture: the encoder discards the")
        print("     distinction before the generator ever sees it.")
        print("  -> Still publishable; the claim becomes one about encoder capacity.")
    else:
        head = "ASPECT IS SALIENT IN THE ENCODER"
        print(f"VERDICT: {head}")
        print("Aspect both decodes and moves the vector more than inert tokens do.")
        print("  -> If generation still ignores it, the loss is downstream of conditioning.")
        print("     That is a sharper result than expected, but check the generation side")
        print("     before committing to the framing.")

    if bow is not None and probe_acc < bow - 0.15:
        print(f"\n  NOTE: probe {probe_acc:.3f} sits well below the bag-of-words ceiling {bow:.3f}.")
        print("  The surface form carries the distinction and the encoder attenuates it.")
    elif decodable:
        print(f"\n  NOTE: probe {probe_acc:.3f} is near the bag-of-words ceiling {bow:.3f}. Expected —")
        print("  the encoder preserves token identity and the conditions carry distinct cues.")
        print("  Decodability alone is weak evidence; the magnitude column is what matters.")
    print("=" * 74)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--layer", default=None, help="layer key, e.g. layer_12; default = last")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = np.load(args.emb, allow_pickle=True)
    stimuli = [json.loads(l) for l in args.stimuli.read_text().splitlines() if l.strip()]
    layer_keys = sorted((k for k in data.files if k.startswith("layer_")),
                        key=lambda k: int(k.split("_")[1]))
    layer = args.layer or layer_keys[-1]
    print(f"model={data['model']}  layer={layer}  items={len(stimuli)}")

    table = index_embeddings(data, layer)
    dists = distances(table, stimuli)
    r = surface_check(dists, stimuli)
    geo, mean_sns, below, n_aspects = summarise_geometry(dists, stimuli)

    wanted = [(a, b) for a, b, _ in DIRECT_CONTRASTS] + [(REFERENCE, "paraphrase_min")]
    contrasts = direct_contrasts(pairwise_distances(table, stimuli, wanted), stimuli)

    n_classes = len(TARGETS)
    print(f"\n(b) LINEAR PROBE  — {n_classes}-way condition classification, "
          f"chance = {1 / n_classes:.3f}")
    results = {}
    for group_key, label in (("gerund", "held-out verbs"), ("noun", "held-out objects")):
        acc, sd, chance = probe(table, stimuli, group_key)
        bow = bow_ceiling(stimuli, group_key)
        print(f"  {label:<17} acc={acc:.3f} (+-{sd:.3f})   permuted={chance:.3f}   "
              f"bag-of-words={bow:.3f}")
        results[group_key] = {"acc": acc, "sd": sd, "permuted": chance, "bow": bow}

    if args.all_layers and len(layer_keys) > 1:
        print("\n(c) LAYER LOCALISATION  — probe accuracy, held-out verbs")
        print(f"     ({len(layer_keys)} layers, CPU-bound; a few minutes for a 4096-dim encoder)")
        for key in layer_keys:
            acc, _, _ = probe(index_embeddings(data, key), stimuli, "gerund", with_null=False)
            print(f"  {key:<10} {acc:.3f}  {'#' * int(round(acc * 40))}")
            results.setdefault("layers", {})[key] = acc

    head = verdict(mean_sns, below, n_aspects, results["gerund"]["acc"],
                   results["gerund"]["permuted"], results["gerund"]["bow"])

    if args.out:
        args.out.write_text(json.dumps({
            "model": str(data["model"]), "layer": layer, "verdict": head,
            "mean_sns": mean_sns, "conditions_below_filler": f"{below}/{n_aspects}",
            "distance_vs_edit_r": r, "geometry": geo,
            "direct_contrasts": contrasts, "probe": results,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
