"""
Fast scoring: For each S1 entity, compute REAL TF-IDF cosine similarity 
against its blocking candidates and pick the top-1 by actual score.
No ML model. Pure vectorized math. Runs in ~20 minutes.
"""
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize

BASE = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE / "code" / "business_entity_resolution" / "models"
OUTPUT_DIR = BASE / "output"

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import normalize_name, normalize_address

def run():
    t0 = time.time()

    print("Loading vectorizers...")
    with open(MODEL_DIR / "tfidf_vectorizers.pkl", "rb") as f:
        vecs = pickle.load(f)
    vec_name = vecs["vec_name"]
    vec_addr = vecs["vec_addr"]

    folder = BASE / "dataset/test"
    print("Loading test data...")
    s1 = pd.read_csv(folder / "test_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / "test_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / "test_source3.tsv", sep="\t", dtype=str).fillna("")
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    print("Normalizing...")
    s1["norm_name"] = s1["business_name"].apply(normalize_name)
    s1["norm_addr"] = s1["business_address"].apply(normalize_address)
    pool["norm_name"] = pool["business_name"].apply(normalize_name)
    pool["norm_addr"] = pool["business_address"].apply(normalize_address)

    # Build fast lookups
    pool_name_lookup = dict(zip(pool["entity_id"], pool["norm_name"]))
    pool_addr_lookup = dict(zip(pool["entity_id"], pool["norm_addr"]))
    s1_name_lookup = dict(zip(s1["entity_id"], s1["norm_name"]))
    s1_addr_lookup = dict(zip(s1["entity_id"], s1["norm_addr"]))
    del pool
    gc.collect()

    print("Loading candidate pairs...")
    cand_df = pd.read_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", dtype=str).fillna("")
    print(f"  {len(cand_df):,} rows")

    results_top1 = {}
    results_top2 = {}
    BATCH = 500  # Process 500 S1 entities at a time

    n = len(cand_df)
    for batch_start in range(0, n, BATCH):
        batch_end = min(batch_start + BATCH, n)
        batch = cand_df.iloc[batch_start:batch_end]

        for _, row in batch.iterrows():
            s1_id = row["source1_entity_id"]
            cand_str = row.get("candidate_entity_ids", "").strip()

            if not cand_str:
                results_top1[s1_id] = ""
                results_top2[s1_id] = ""
                continue

            cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
            # Filter to candidates that exist in pool
            cand_ids = [c for c in cand_ids if c in pool_name_lookup]

            if not cand_ids:
                results_top1[s1_id] = ""
                results_top2[s1_id] = ""
                continue

            s1_name = s1_name_lookup.get(s1_id, "")
            s1_addr = s1_addr_lookup.get(s1_id, "")
            cand_names = [pool_name_lookup[c] for c in cand_ids]
            cand_addrs = [pool_addr_lookup[c] for c in cand_ids]

            # Compute TF-IDF cosine similarity for names
            s1_name_vec = sk_normalize(vec_name.transform([s1_name]))
            cand_name_vecs = sk_normalize(vec_name.transform(cand_names))
            name_scores = (s1_name_vec @ cand_name_vecs.T).toarray().flatten()

            # Compute TF-IDF cosine similarity for addresses
            if s1_addr and any(cand_addrs):
                s1_addr_vec = sk_normalize(vec_addr.transform([s1_addr]))
                cand_addr_vecs = sk_normalize(vec_addr.transform(cand_addrs))
                addr_scores = (s1_addr_vec @ cand_addr_vecs.T).toarray().flatten()
            else:
                addr_scores = np.zeros(len(cand_ids))

            # Combined score: 70% name + 30% address
            combined = 0.7 * name_scores + 0.3 * addr_scores

            # Sort by score descending
            ranked_idx = np.argsort(combined)[::-1]
            ranked_ids = [cand_ids[i] for i in ranked_idx]

            results_top1[s1_id] = ranked_ids[0] if ranked_ids else ""
            results_top2[s1_id] = ",".join(ranked_ids[:2]) if ranked_ids else ""

        if (batch_start // BATCH) % 100 == 0:
            elapsed = (time.time() - t0) / 60
            pct = 100 * batch_end / n
            eta = elapsed / max(pct, 0.01) * (100 - pct)
            print(f"  Progress: {batch_end:,}/{n:,} ({pct:.1f}%) | Elapsed: {elapsed:.1f}m | ETA: {eta:.1f}m")

    # Build output
    print("\nBuilding output files...")
    rows_top1, rows_top2 = [], []
    for s1_id in s1["entity_id"]:
        rows_top1.append({"source1_entity_id": s1_id, "matched_entity_ids": results_top1.get(s1_id, "")})
        rows_top2.append({"source1_entity_id": s1_id, "matched_entity_ids": results_top2.get(s1_id, "")})

    pd.DataFrame(rows_top1).to_csv(OUTPUT_DIR / "ranked_top1.tsv", sep="\t", index=False)
    pd.DataFrame(rows_top2).to_csv(OUTPUT_DIR / "ranked_top2.tsv", sep="\t", index=False)

    non_empty = sum(1 for r in rows_top1 if r["matched_entity_ids"])
    print(f"ranked_top1.tsv: {non_empty:,}/{len(s1):,} entities have a match")
    print(f"Saved: output/ranked_top1.tsv, output/ranked_top2.tsv")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    run()
