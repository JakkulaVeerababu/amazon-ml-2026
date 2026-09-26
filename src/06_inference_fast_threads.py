import argparse
import gc
import pickle
import sys
import time
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import normalize_name, normalize_address, get_name_tokens, get_address_tokens, make_ngrams
from features import compute_pair_features_batch

BASE = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE / "code" / "business_entity_resolution" / "models"
OUTPUT_DIR = BASE / "output"

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

# GLOBAL VARIABLES FOR THREADS
g_pool_lookup = {}
g_s1_lookup = {}
g_vec_name = None
g_vec_addr = None
g_model = None
g_threshold = 0.5

def process_chunk(chunk_df):
    results = {}
    BATCH = 2000
    
    for idx, (_, row) in enumerate(chunk_df.iterrows()):
        s1_id = row["source1_entity_id"]
        cand_str = row.get("candidate_entity_ids", "").strip()

        if not cand_str:
            results[s1_id] = []
            continue

        cand_ids = cand_str.split(",")
        s1_r = g_s1_lookup.get(s1_id)
        if not s1_r:
            results[s1_id] = []
            continue

        s1_rec = make_record(s1_r)
        scored = []
        
        for start in range(0, len(cand_ids), BATCH):
            batch_ids = cand_ids[start:start + BATCH]
            s1_batch = []
            cand_batch = []
            valid_ids = []

            for cid in batch_ids:
                cr = g_pool_lookup.get(cid)
                if cr:
                    s1_batch.append(s1_rec)
                    cand_batch.append(make_record(cr))
                    valid_ids.append(cid)

            if not cand_batch:
                continue

            X = compute_pair_features_batch(s1_batch, cand_batch, g_vec_name, g_vec_addr)
            probs = g_model.predict(X)

            for cid, prob in zip(valid_ids, probs):
                if prob >= g_threshold:
                    scored.append((cid, float(prob)))

        scored.sort(key=lambda x: x[1], reverse=True)
        seen, matches = set(), []
        for cid, _ in scored:
            if cid not in seen:
                seen.add(cid)
                matches.append(cid)
                
        # Limit to 2 or 3 to avoid extreme false positives
        results[s1_id] = matches[:2]

    return results

def run_inference_fast(candidates_path: str, threshold: float = None):
    t0 = time.time()
    global g_pool_lookup, g_s1_lookup, g_vec_name, g_vec_addr, g_model, g_threshold

    print("Loading vectorizers...")
    model_path = MODEL_DIR / "lgbm_model.txt"
    with open(MODEL_DIR / "tfidf_vectorizers.pkl", "rb") as f:
        vecs = pickle.load(f)
    g_vec_name, g_vec_addr = vecs["vec_name"], vecs["vec_addr"]

    if threshold is None:
        thresh_path = MODEL_DIR / "threshold.txt"
        if thresh_path.exists():
            threshold = float(thresh_path.read_text().strip())
        else:
            threshold = 0.5
    g_threshold = threshold
    print(f"Decision threshold: {g_threshold}")

    g_model = lgb.Booster(model_file=str(model_path))

    folder = BASE / "dataset/test"
    print("Loading test data...")
    s1 = pd.read_csv(folder / "test_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / "test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / "test_source3.tsv", sep="\t", dtype=str).fillna("")
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    print("Building lookup dicts...")
    g_pool_lookup = {row["entity_id"]: row.to_dict() for _, row in pool.iterrows()}
    g_s1_lookup = {row["entity_id"]: row.to_dict() for _, row in s1.iterrows()}
    del pool
    gc.collect()

    cand_df = pd.read_csv(candidates_path, sep="\t", dtype=str).fillna("")
    print(f"Candidate pairs: {len(cand_df):,} rows")

    import os
    num_cores = os.cpu_count()
    print(f"Running inference in parallel using {num_cores} threads...")
    
    chunk_size = math.ceil(len(cand_df) / num_cores)
    chunks = [cand_df.iloc[i:i + chunk_size] for i in range(0, len(cand_df), chunk_size)]
    
    results = {}
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
        
    for res in chunk_results:
        results.update(res)

    for s1_id in s1["entity_id"]:
        if s1_id not in results:
            results[s1_id] = []

    print("\nBuilding output TSV...")
    rows = [{"source1_entity_id": s1_id, "matched_entity_ids": ",".join(results.get(s1_id, []))} for s1_id in s1["entity_id"]]
    out_df = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "matching_results.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    run_inference_fast(args.candidates, args.threshold)
