"""
06_inference.py — Scalable Matching Inference on Test Set

Loads trained LightGBM model, scores all candidate pairs,
applies tuned threshold, and writes matching_results.tsv.

Run from student_resource/:
  python code/business_entity_resolution/src/06_inference.py
  python code/business_entity_resolution/src/06_inference.py --threshold 0.55
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import normalize_name, normalize_address, get_name_tokens, get_address_tokens, make_ngrams
from features import compute_pair_features_batch, FEATURE_NAMES

BASE = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE / "code" / "business_entity_resolution" / "models"
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def make_record(row: dict) -> dict:
    n = normalize_name(row.get("business_name", ""))
    a = normalize_address(row.get("business_address", ""))
    return {
        **row,
        "norm_name": n,
        "norm_addr": a,
        "name_tokens": get_name_tokens(n),
        "addr_tokens": get_address_tokens(a),
        "name_trigrams": make_ngrams(n, 3),
        "addr_trigrams": make_ngrams(a, 3),
    }


def load_threshold(default: float) -> float:
    thresh_path = MODEL_DIR / "threshold.txt"
    if thresh_path.exists():
        t = float(thresh_path.read_text().strip())
        print(f"Loaded threshold from file: {t}")
        return t
    return default


def run_inference(candidates_path: str, threshold: float = None):
    t0 = time.time()

    # Load model
    print("Loading model...")
    model = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
    with open(MODEL_DIR / "tfidf_vectorizers.pkl", "rb") as f:
        vecs = pickle.load(f)
    vec_name, vec_addr = vecs["vec_name"], vecs["vec_addr"]

    if threshold is None:
        threshold = load_threshold(0.5)
    print(f"Decision threshold: {threshold}")

    # Load test data
    folder = BASE / "dataset/test"
    s1 = pd.read_csv(folder / "test_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / "test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / "test_source3.tsv", sep="\t", dtype=str).fillna("")
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    print(f"Test: S1={len(s1):,}, pool={len(pool):,}")

    # Build lookup dicts
    print("Building lookup dicts...")
    pool_lookup = {}
    for _, row in pool.iterrows():
        pool_lookup[row["entity_id"]] = row.to_dict()
    s1_lookup = {}
    for _, row in s1.iterrows():
        s1_lookup[row["entity_id"]] = row.to_dict()

    del pool
    gc.collect()

    # Load candidate pairs
    cand_df = pd.read_csv(candidates_path, sep="\t", dtype=str).fillna("")
    print(f"Candidate pairs: {len(cand_df):,} rows")

    results = {}
    total_pairs_scored = 0
    BATCH = 2000

    print("\nRunning inference...")
    for idx, (_, row) in enumerate(cand_df.iterrows()):
        s1_id = row["source1_entity_id"]
        cand_str = row.get("candidate_entity_ids", "").strip()

        if not cand_str:
            results[s1_id] = []
            continue

        cand_ids = cand_str.split(",")
        s1_r = s1_lookup.get(s1_id)
        if not s1_r:
            results[s1_id] = []
            continue

        s1_rec = make_record(s1_r)

        # Score in batches
        scored = []
        for start in range(0, len(cand_ids), BATCH):
            batch_ids = cand_ids[start:start + BATCH]
            s1_batch = []
            cand_batch = []
            valid_ids = []

            for cid in batch_ids:
                cr = pool_lookup.get(cid)
                if cr:
                    s1_batch.append(s1_rec)
                    cand_batch.append(make_record(cr))
                    valid_ids.append(cid)

            if not cand_batch:
                continue

            X = compute_pair_features_batch(s1_batch, cand_batch, vec_name, vec_addr)
            probs = model.predict(X)
            total_pairs_scored += len(valid_ids)

            for cid, prob in zip(valid_ids, probs):
                if prob >= threshold:
                    scored.append((cid, float(prob)))

        # Sort by prob desc, deduplicate
        scored.sort(key=lambda x: x[1], reverse=True)
        seen, matches = set(), []
        for cid, _ in scored:
            if cid not in seen:
                seen.add(cid)
                matches.append(cid)
        results[s1_id] = matches

        if (idx + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(cand_df) - idx - 1) / rate
            print(f"  {idx+1:,}/{len(cand_df):,} entities ({rate:.0f}/s, ETA {eta/60:.1f}min)")

    # Ensure ALL S1 entities appear
    for s1_id in s1["entity_id"]:
        if s1_id not in results:
            results[s1_id] = []

    # Build output
    print("\nBuilding output TSV...")
    rows = []
    for s1_id in s1["entity_id"]:
        matches = results.get(s1_id, [])
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        })
    out_df = pd.DataFrame(rows)

    # Stats
    n_match = out_df["matched_entity_ids"].str.strip().astype(bool).sum()
    n_sing = len(out_df) - n_match
    total_m = out_df["matched_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    ).sum()

    print(f"\nResults:")
    print(f"  Total S1 entities: {len(out_df):,}")
    print(f"  Predicted singletons: {n_sing:,} ({100*n_sing/len(out_df):.1f}%)")
    print(f"  Entities with matches: {n_match:,}")
    print(f"  Total matches predicted: {total_m:,}")
    print(f"  Total pairs scored: {total_pairs_scored:,}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    out_path = OUTPUT_DIR / "matching_results.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (auto-loaded from training if not set)")
    args = parser.parse_args()
    run_inference(args.candidates, args.threshold)
