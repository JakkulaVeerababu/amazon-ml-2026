import argparse
import gc
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import normalize_name, normalize_address, get_name_tokens, get_address_tokens, make_ngrams
from features import enrich_df, build_tfidf_vectorizers, compute_pair_features_batch, FEATURE_NAMES

# Import the fast tfidf blocking logic from 03_blocking
from importlib import import_module
blocking = import_module("03_blocking")

BASE = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE / "code" / "business_entity_resolution" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_train():
    folder = BASE / "dataset/train"
    s1 = pd.read_csv(folder / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / "train_source3.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(folder / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3, gt

def build_gt_lookup(gt: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        ms = row["matched_entity_ids"].strip()
        lookup[sid] = set(ms.split(",")) if ms else set()
    return lookup

def make_record(row_dict: dict) -> dict:
    n = row_dict.get("norm_name") or normalize_name(row_dict.get("business_name", ""))
    a = row_dict.get("norm_addr") or normalize_address(row_dict.get("business_address", ""))
    return {
        **row_dict,
        "norm_name": n,
        "norm_addr": a,
        "name_tokens": get_name_tokens(n),
        "addr_tokens": get_address_tokens(a),
        "name_trigrams": make_ngrams(n, 3),
        "addr_trigrams": make_ngrams(a, 3),
    }

def main(sample_size: int = 10000, seed: int = 42):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    print("Loading training data...")
    s1, s2, s3, gt = load_train()
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    gt_lookup = build_gt_lookup(gt)
    print("Normalizing text...")
    s1["norm_name"] = s1["business_name"].apply(normalize_name)
    s1["norm_addr"] = s1["business_address"].apply(normalize_address)
    pool["norm_name"] = pool["business_name"].apply(normalize_name)
    pool["norm_addr"] = pool["business_address"].apply(normalize_address)

    # We only sample a small subset of S1, but we need it to be from a single country to make blocking fast
    country = "US"
    s1_us = s1[s1["country"] == country].reset_index(drop=True)
    pool_us = pool[pool["country"] == country].reset_index(drop=True)
    
    sample_size_actual = min(sample_size, len(s1_us))
    s1_sample = s1_us.sample(n=sample_size_actual, random_state=seed).reset_index(drop=True)
    print(f"Sampled {len(s1_sample)} {country} entities for training")

    # Fit TF-IDF on the full corpus for feature extraction
    print("Fitting TF-IDF vectorizers on full corpus...")
    vec_name, vec_addr = build_tfidf_vectorizers(s1, pool)
    
    vec_path = MODEL_DIR / "tfidf_vectorizers.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump({"vec_name": vec_name, "vec_addr": vec_addr}, f)
    
    # Run blocking on the sampled S1 to get HARD negatives
    print("\nRunning TF-IDF blocking to get hard negatives...")
    cands = blocking.tfidf_blocking_chunked(
        s1_sample, pool_us, top_k=30, min_sim=0.08,
        s1_chunk=5000, pool_chunk=500000
    )

    print("Building pairs...")
    s1_recs, cand_recs, labels, groups = [], [], [], []
    pool_by_id = pool_us.set_index("entity_id").to_dict("index")
    
    group_idx = 0
    for idx, row in s1_sample.iterrows():
        sid = row["entity_id"]
        true_matches = gt_lookup.get(sid, set())
        hard_candidates = cands.get(sid, set())
        
        # Include all true matches
        all_candidates = true_matches | hard_candidates
        
        s1_r = make_record(row.to_dict())
        
        for cid in all_candidates:
            if cid in pool_by_id:
                s1_recs.append(s1_r)
                cand_recs.append(make_record(pool_by_id[cid]))
                labels.append(1 if cid in true_matches else 0)
                groups.append(group_idx)
        group_idx += 1

    labels = np.array(labels, dtype=np.int32)
    groups = np.array(groups)
    print(f"Pairs: {len(labels):,} (pos={labels.sum():,}, neg={(labels==0).sum():,})")

    # Train/Val split
    unique_groups = np.unique(groups)
    val_groups = rng.choice(unique_groups, size=int(0.15 * len(unique_groups)), replace=False)
    val_mask = np.isin(groups, val_groups)
    train_mask = ~val_mask

    print("Extracting features...")
    def compute_features_batched(s1_recs, cand_recs, vec_name, vec_addr, batch=5000):
        X_parts = []
        N = len(s1_recs)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            X_part = compute_pair_features_batch(
                s1_recs[start:end], cand_recs[start:end],
                tfidf_name=vec_name, tfidf_addr=vec_addr
            )
            X_parts.append(X_part)
        return np.vstack(X_parts)

    X_train = compute_features_batched(
        [r for r, m in zip(s1_recs, train_mask) if m],
        [r for r, m in zip(cand_recs, train_mask) if m],
        vec_name, vec_addr
    )
    y_train = labels[train_mask]

    X_val = compute_features_batched(
        [r for r, m in zip(s1_recs, val_mask) if m],
        [r for r, m in zip(cand_recs, val_mask) if m],
        vec_name, vec_addr
    )
    y_val = labels[val_mask]

    print("\nTraining LightGBM...")
    pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "n_jobs": -1,
        "seed": seed,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )

    val_probs = model.predict(X_val)
    # Find best F0.5
    thresholds = np.arange(0.1, 0.95, 0.02)
    best_t, best_f05 = 0.5, 0.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        if f05 > best_f05:
            best_f05, best_t = f05, t
            
    print(f"  Best threshold: {best_t:.2f} -> Val F_0.5 ~= {best_f05:.4f}")

    thresh_path = MODEL_DIR / "threshold.txt"
    thresh_path.write_text(str(best_t))
    model_path = MODEL_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
