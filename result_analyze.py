import os
import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr
from typing import Dict, Any, List

DIM_KEYS = ["1a", "1b", "2a", "3a", "3b", "4a", "5a", "5b"]
survey_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]



def normalize_video_key(k: str) -> str:
    k = (k or "").strip()
    if k.lower().endswith(".mp4"):
        k = k[:-4]
    return k


def parse_human_score(v):
    """Human JSON stores score as string like '4' or 'NA'."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.upper() == "NA":
            return None
        try:
            iv = int(float(s))
            return iv if 1 <= iv <= 5 else None
        except Exception:
            return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if 1 <= iv <= 5 else None
    return None


def parse_mllm_score(v):
    """MLLM JSON stores score as int, but we also tolerate str/float."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # iv = int(v)
        # return iv if 1 <= iv <= 5 else None
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.upper() == "NA":
            return None
        try:
            iv = int(float(s))
            return iv if 1 <= iv <= 5 else None
        except Exception:
            return None
    return None

def compute_dim_means(dim2values):
    means = {}
    counts = {}
    for d in DIM_KEYS:
        vals = dim2values.get(d, [])
        counts[d] = len(vals)
        means[d] = float(np.mean(vals)) if vals else np.nan
    return means, counts

def load_human_scores_by_index(base_dir="."):
    human_scores_merged = defaultdict(dict)
    folder_stats = {}

    for i in survey_ids:
        json_path = os.path.join(
            base_dir,
            f"evaluations-{i}-mean3.json",
        )
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dim2values = {d: [] for d in DIM_KEYS}
        n_videos = 0

        for raw_vk, item in (data or {}).items():
            n_videos += 1
            vk = normalize_video_key(raw_vk)
            evals = item.get("evaluations", {})

            for d in DIM_KEYS:
                sc = parse_human_score(evals.get(d))
                if sc is None:
                    continue
                dim2values[d].append(sc)
                human_scores_merged[vk][d] = sc

        means, counts = compute_dim_means(dim2values)
        folder_stats[i] = {
            "means": means,
            "counts": counts,
            "n_videos": n_videos,
        }

    return human_scores_merged, folder_stats



def load_mllm_scores(mllm_json_path):
    with open(mllm_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mllm_scores = defaultdict(dict)
    dim2values = {d: [] for d in DIM_KEYS}

    for raw_vk, item in (data or {}).items():
        vk = normalize_video_key(raw_vk)
        evals = item.get("evaluation", {})

        for d in DIM_KEYS:
            node = evals.get(d)
            sc = parse_mllm_score(node.get("score")) if isinstance(node, dict) else parse_mllm_score(node)
            if sc is None:
                continue
            mllm_scores[vk][d] = sc
            dim2values[d].append(sc)

    means, counts = compute_dim_means(dim2values)
    return mllm_scores, means, counts


def correlation():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/Users/hanxianjing/proj/Video benchmark/human_annotation", help="Base directory containing results-human-feedback-survey-*")
    ap.add_argument("--mllm_json", default="/Users/hanxianjing/proj/Video benchmark/phy-world-bench/batch/correct_evidence2/batch_correct_evidence2_840_fixNA.json", help="Path to MLLM evaluation json")
    ap.add_argument("--out_csv", default="correlation_results.csv", help="Output CSV filename")
    args = ap.parse_args()

    human_scores, folder_stats = load_human_scores_by_index(base_dir=args.base_dir)
    mllm_scores, mllm_means, mllm_counts = load_mllm_scores(args.mllm_json)

    human_set = set(human_scores.keys())
    mllm_set = set(mllm_scores.keys())
    matched_videos = sorted(human_set & mllm_set)

    print("\n========== Alignment ==========")
    print(f"Matched videos (intersection): {len(matched_videos)}")
    print("================================\n")

    rows = []
    for dim in DIM_KEYS:
        xs, ys = [], []
        skipped = 0

        for vk in matched_videos:
            hs = human_scores.get(vk, {}).get(dim)
            ms = mllm_scores.get(vk, {}).get(dim)
            if ms is None and hs is not None:
                print(vk)
            if hs is None or ms is None:
                skipped += 1
                continue
            xs.append(hs)
            ys.append(ms)

        # ys_rescaled = rescale(ys, xs)
        tau, tau_p = kendalltau(xs, ys)
        rho, rho_p = spearmanr(xs, ys)

        rows.append({
            "dimension": dim,
            "n_pairs": len(xs),
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "skipped_pairs_due_to_missing_or_NA": skipped,
        })

    # Print summary
    print("Per-dimension Kendall’s τ and Spearman’s ρ (across videos)")
    print("dim | n | tau | rho")
    print("----|---|------|------")
    for r in rows:
        print(f"{r['dimension']:>3} | {r['n_pairs']:>3} | {r['kendall_tau']:> .3f} \t {r['spearman_rho']:> .3f}")

    taus = np.array([r["kendall_tau"] for r in rows], dtype=float)
    rhos = np.array([r["spearman_rho"] for r in rows], dtype=float)
    tau_mean_macro = float(np.nanmean(taus)) if np.isfinite(np.nanmean(taus)) else np.nan
    rho_mean_macro = float(np.nanmean(rhos)) if np.isfinite(np.nanmean(rhos)) else np.nan
    print(f"Mean|  -  | {tau_mean_macro:> .3f} \t {rho_mean_macro:> .3f}\n")

    header = "pair\t" + "\t".join([f"{d:>4}" for d in DIM_KEYS])
    print(header)

    pairs = [(survey_ids[i], survey_ids[i + 1]) for i in range(0, len(survey_ids), 2)]

    all_dim_values = {d: [] for d in DIM_KEYS}

    for a, b in pairs:
        sa = folder_stats.get(a)
        sb = folder_stats.get(b)

        if sa is None and sb is None:
            print(f"[WARN] Missing both surveys: {a} and {b}")
            continue

        merged_means = {}
        merged_counts = {}

        for d in DIM_KEYS:
            ca = (sa["counts"].get(d, 0) if sa else 0)
            cb = (sb["counts"].get(d, 0) if sb else 0)
            ma = (sa["means"].get(d, np.nan) if sa else np.nan)
            mb = (sb["means"].get(d, np.nan) if sb else np.nan)

            ctot = ca + cb
            merged_counts[d] = ctot

            if ctot == 0:
                merged_means[d] = np.nan
            else:
                merged_means[d] = (ma * ca + mb * cb) / ctot

            if merged_counts[d] > 0 and not np.isnan(merged_means[d]):
                all_dim_values[d].extend([merged_means[d]] * merged_counts[d])

        pair_name = f"{a:>2}-{b:<2}"
        line = f"{pair_name:>4}\t" + "\t".join([f"{merged_means[d]:>4.2f}" for d in DIM_KEYS])
        print(line)

    all_means = {
        d: (np.mean(all_dim_values[d]) if len(all_dim_values[d]) > 0 else np.nan)
        for d in DIM_KEYS
    }

    all_line = " Mean\t" + "\t".join([f"{all_means[d]:>4.2f}" for d in DIM_KEYS])
    print(all_line)

    line = " MLLM\t" + "\t".join([f"{mllm_means[d]:>4.2f}" for d in DIM_KEYS])
    print(line)

correlation()