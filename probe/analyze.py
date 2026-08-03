"""Analyse the encoder embeddings and return the verdict that selects the paper's framing.

Three measures, in order of what they settle:

(a) Geometry. Cosine distance from the progressive reference to each other aspect
    condition, bracketed by controls. Reported as the Aspect Salience Index,
    ASI = (d_aspect - d_floor) / (d_other_verb - d_floor).
    ASI near 0 means aspect moves the embedding no more than a meaning-preserving
    reword does; ASI near 1 means it moves it as much as swapping the event.

    The floor is `paraphrase_min`, a single-token synonym substitution. The looser
    `paraphrase` control (whole scene phrase fronted) is reported alongside but is
    NOT used as the floor: it moves many tokens, which would inflate the floor and
    push ASI down -- toward the result we expect. The tight floor keeps the test
    conservative against our own hypothesis.

(b) Linear probe. Can aspect be recovered linearly from the pooled vector, with
    whole verbs (and separately whole objects) held out so the probe cannot key
    on lexical identity? Compared against a label-permutation floor and against a
    bag-of-words ceiling on the raw sentences. Scoring far below bag-of-words
    means the encoder is discarding a distinction its input carried.

(c) Layer localisation. Where along the stack the distinction appears or is lost.

Usage:
    python analyze.py --emb emb_umt5xxl.npz --stimuli stimuli.jsonl
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

ASPECTS = ["prog", "perf", "result", "prospective", "failed", "atelic"]
REFERENCE = "prog"


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - (a * b).sum(-1)


def token_edit_distance(s1, s2):
    """Normalised token-level Levenshtein, to show distances are not just surface churn."""
    a, b = s1.lower().split(), s2.lower().split()
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def index_embeddings(data, layer_key):
    vectors = data[layer_key]
    item_ids, conditions = data["item_ids"], data["conditions"]
    table = {}
    for vec, item, cond in zip(vectors, item_ids, conditions):
        table[(int(item), str(cond))] = vec
    return table


def geometry(table, stimuli):
    rows = []
    for rec in stimuli:
        item = rec["item_id"]
        ref = table.get((item, REFERENCE))
        if ref is None:
            continue
        texts = rec["texts"]
        d_floor = float(cosine(ref, table[(item, "paraphrase_min")]))
        d_para_loose = float(cosine(ref, table[(item, "paraphrase")]))
        d_verb = float(cosine(ref, table[(item, "other_verb")]))
        for cond in ASPECTS:
            if cond == REFERENCE:
                continue
            d = float(cosine(ref, table[(item, cond)]))
            span = d_verb - d_floor
            rows.append({
                "item_id": item,
                "condition": cond,
                "d_aspect": d,
                "d_floor": d_floor,
                "d_para_loose": d_para_loose,
                "d_other_verb": d_verb,
                "asi": (d - d_floor) / span if abs(span) > 1e-9 else np.nan,
                "edit_aspect": token_edit_distance(texts[REFERENCE], texts[cond]),
                "edit_floor": token_edit_distance(texts[REFERENCE], texts["paraphrase_min"]),
            })
    return rows


def summarise_geometry(rows):
    print("\n(a) GEOMETRY  — distance from the progressive reference")
    print("    floor = paraphrase_min (one-token synonym); d_loose shown for reference only")
    print(f"{'condition':<13} {'d_aspect':>9} {'d_floor':>9} {'d_loose':>9} {'d_verb':>9} "
          f"{'ASI':>7} {'p(vs fl)':>10} {'edit_a':>7} {'edit_f':>7}")
    summary = {}
    for cond in ASPECTS:
        if cond == REFERENCE:
            continue
        sel = [r for r in rows if r["condition"] == cond]
        if not sel:
            continue
        d_a = np.array([r["d_aspect"] for r in sel])
        d_p = np.array([r["d_floor"] for r in sel])
        d_l = np.array([r["d_para_loose"] for r in sel])
        d_v = np.array([r["d_other_verb"] for r in sel])
        asi = np.nanmean([r["asi"] for r in sel])
        try:
            pval = wilcoxon(d_a, d_p).pvalue
        except ValueError:  # identical vectors
            pval = float("nan")
        e_a = np.mean([r["edit_aspect"] for r in sel])
        e_f = np.mean([r["edit_floor"] for r in sel])
        print(f"{cond:<13} {d_a.mean():>9.4f} {d_p.mean():>9.4f} {d_l.mean():>9.4f} "
              f"{d_v.mean():>9.4f} {asi:>7.3f} {pval:>10.2e} {e_a:>7.3f} {e_f:>7.3f}")
        summary[cond] = {"d_aspect": float(d_a.mean()), "d_floor": float(d_p.mean()),
                         "d_para_loose": float(d_l.mean()), "d_other_verb": float(d_v.mean()),
                         "asi": float(asi), "p": float(pval)}
    overall = float(np.nanmean([r["asi"] for r in rows]))
    print(f"\n  mean ASI over all aspect conditions: {overall:.3f}"
          "   (0 = no more than a synonym swap, 1 = as salient as changing the verb)")
    return summary, overall


def probe(table, stimuli, group_key, seed=0):
    X, y, groups = [], [], []
    for rec in stimuli:
        for cond in ASPECTS:
            key = (rec["item_id"], cond)
            if key not in table:
                continue
            X.append(table[key])
            y.append(cond)
            groups.append(rec[group_key])
    X, y, groups = np.array(X), np.array(y), np.array(groups)

    n_splits = min(5, len(set(groups)))
    cv = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(seed)
    accs, chance = [], []
    for train, test in cv.split(X, y, groups):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[train]), y[train])
        accs.append(clf.score(scaler.transform(X[test]), y[test]))

        shuffled = rng.permutation(y[train])
        null = LogisticRegression(max_iter=2000, C=1.0)
        null.fit(scaler.transform(X[train]), shuffled)
        chance.append(null.score(scaler.transform(X[test]), y[test]))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(chance))


def bow_ceiling(stimuli, group_key, seed=0):
    texts, y, groups = [], [], []
    for rec in stimuli:
        for cond in ASPECTS:
            texts.append(rec["texts"][cond])
            y.append(cond)
            groups.append(rec[group_key])
    y, groups = np.array(y), np.array(groups)
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)

    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    accs = []
    for train, test in cv.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[train], y[train])
        accs.append(clf.score(X[test], y[test]))
    return float(np.mean(accs))


def verdict(overall_asi, probe_acc, chance, bow):
    print("\n" + "=" * 72)
    if probe_acc > chance + 0.15 and overall_asi < 0.25:
        head = "INFORMATION PRESENT BUT LOW-MAGNITUDE"
        body = ("Aspect is linearly decodable from the conditioning vector, yet it barely "
                "moves the embedding relative to a meaning-preserving reordering.\n"
                "  -> The generator has the signal and does not use it.\n"
                "  -> Original framing holds; you also get a mechanistic localisation section,\n"
                "     and the training-free difference-vector intervention is worth trying.")
    elif probe_acc <= chance + 0.15:
        head = "ENCODER IS THE BOTTLENECK"
        body = ("Aspect is not linearly recoverable from the conditioning vector.\n"
                "  -> Reframe from semantics to architecture: the text encoder discards the\n"
                "     distinction before the generator ever sees it.\n"
                "  -> Still publishable, but the claim is about encoder capacity, not about\n"
                "     the generator ignoring language.")
    else:
        head = "ASPECT IS SALIENT IN THE ENCODER"
        body = ("Aspect both decodes well and moves the embedding substantially.\n"
                "  -> If generation still ignores it, the loss is downstream of conditioning;\n"
                "     that is a sharper result than expected. Check the generation side before\n"
                "     committing to the framing.")
    print(f"VERDICT: {head}")
    print(body)
    if bow is not None and probe_acc < bow - 0.15:
        print(f"\n  NOTE: probe {probe_acc:.3f} is well below the bag-of-words ceiling {bow:.3f}.\n"
              "  The surface form carries the distinction and the encoder is attenuating it.")
    print("=" * 72)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--stimuli", type=Path, default=Path(__file__).parent / "stimuli.jsonl")
    ap.add_argument("--layer", default=None, help="layer key, e.g. layer_12; default = last")
    ap.add_argument("--all-layers", action="store_true", help="run the probe at every layer")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = np.load(args.emb, allow_pickle=True)
    stimuli = [json.loads(l) for l in args.stimuli.read_text().splitlines() if l.strip()]
    layer_keys = sorted((k for k in data.files if k.startswith("layer_")),
                        key=lambda k: int(k.split("_")[1]))
    layer = args.layer or layer_keys[-1]
    print(f"model={data['model']}  layer={layer}  items={len(stimuli)}")

    table = index_embeddings(data, layer)
    rows = geometry(table, stimuli)
    geo_summary, overall_asi = summarise_geometry(rows)

    print("\n(b) LINEAR PROBE  — 6-way aspect classification, chance = 0.167")
    results = {}
    for group_key in ("gerund", "noun"):
        acc, sd, chance = probe(table, stimuli, group_key)
        bow = bow_ceiling(stimuli, group_key)
        label = "held-out verbs" if group_key == "gerund" else "held-out objects"
        print(f"  {label:<16} acc={acc:.3f} (+-{sd:.3f})   permuted={chance:.3f}   "
              f"bag-of-words={bow:.3f}")
        results[group_key] = {"acc": acc, "sd": sd, "permuted": chance, "bow": bow}

    if args.all_layers and len(layer_keys) > 1:
        print("\n(c) LAYER LOCALISATION  — probe accuracy with held-out verbs")
        for key in layer_keys:
            acc, _, chance = probe(index_embeddings(data, key), stimuli, "gerund")
            bar = "#" * int(round(acc * 40))
            print(f"  {key:<10} {acc:.3f}  {bar}")
            results.setdefault("layers", {})[key] = acc

    head = verdict(overall_asi, results["gerund"]["acc"], results["gerund"]["permuted"],
                   results["gerund"]["bow"])

    if args.out:
        args.out.write_text(json.dumps({
            "model": str(data["model"]), "layer": layer, "verdict": head,
            "mean_asi": overall_asi, "geometry": geo_summary, "probe": results,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
