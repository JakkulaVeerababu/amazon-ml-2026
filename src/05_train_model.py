"""
05_train_model.py — Train LightGBM Matching Classifier (Scalable)

For 2.2M S1 entities: we sample a representative training subset.
- Positive pairs: from ground truth (on sampled S1 subset)
- Negative pairs: hard negatives from blocking candidates + random

Run from student_resource/:
  python code/business_entity_resolution/src/05_train_model.py \
      --sample-size 200000 --neg-ratio 5
"""

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
from preprocess_utils import (
    normalize_name, normalize_address, get_name_tokens, get_address_tokens, make_ngrams
)
from features import enrich_df, build_tfidf_vectorizers, compute_pair_features_batch, FEATURE_NAMES

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
    # Reuse pre-computed normalisations if present (avoids redundant work)
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


def build_pairs(
    s1_sample: pd.DataFrame,
    pool: pd.DataFrame,
    gt_lookup: dict,
    neg_ratio: int,
    rng: np.random.Generator,
) -> tuple:
    """Build labeled training pairs."""
    print("  Building country index...")
    pool_by_country = pool.groupby("country")["entity_id"].apply(list).to_dict()

    # We will gather all the pool IDs we actually need to avoid a 10-million row memory explosion
    all_pool_ids = pool["entity_id"].values
    
    s1_eids     = s1_sample["entity_id"].values
    s1_countries = s1_sample["country"].values
    s1_names    = s1_sample["business_name"].values
    s1_addrs    = s1_sample["business_address"].values
    s1_nnames   = s1_sample["norm_name"].values
    s1_naddrs   = s1_sample["norm_addr"].values

    import random
    print(f"  Sampling candidates for {len(s1_sample):,} S1 entities...")
    needed_pool_ids = set()
    s1_tasks = []

    for idx in range(len(s1_sample)):
        sid     = s1_eids[idx]
        country = s1_countries[idx]
        true_matches = gt_lookup.get(sid, set())

        s1_r = make_record({
            "entity_id": sid,
            "business_name": s1_names[idx],
            "business_address": s1_addrs[idx],
            "country": country,
            "norm_name": s1_nnames[idx],
            "norm_addr": s1_naddrs[idx],
        })
        
        country_pool = pool_by_country.get(country, all_pool_ids)
        n_neg = max(1, neg_ratio * max(1, len(true_matches)))
        
        chosen = []
        if len(country_pool) > 0:
            pool_size = len(country_pool)
            chosen_raw = random.sample(country_pool, min(n_neg + 10, pool_size))
            chosen = [x for x in chosen_raw if x not in true_matches][:n_neg]
            
        needed_pool_ids.update(true_matches)
        needed_pool_ids.update(chosen)
        s1_tasks.append((s1_r, true_matches, chosen))

    print(f"  Extracting {len(needed_pool_ids):,} required pool rows into memory...")
    pool_subset = pool[pool["entity_id"].isin(needed_pool_ids)]
    pool_by_id = pool_subset.set_index("entity_id").to_dict("index")

    print("  Building pair feature lists...")
    s1_recs, cand_recs, labels, groups = [], [], [], []
    group_idx = 0
    for s1_r, true_matches, chosen in s1_tasks:
        for match_id in true_matches:
            if match_id in pool_by_id:
                s1_recs.append(s1_r)
                cand_recs.append(make_record(pool_by_id[match_id]))
                labels.append(1)
                groups.append(group_idx)
                
        for neg_id in chosen:
            if neg_id in pool_by_id:
                s1_recs.append(s1_r)
                cand_recs.append(make_record(pool_by_id[neg_id]))
                labels.append(0)
                groups.append(group_idx)
                
        group_idx += 1

    return s1_recs, cand_recs, np.array(labels, dtype=np.int32), np.array(groups)


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
        if (start // batch) % 20 == 0:
            print(f"    Features: {end:,}/{N:,}")
    return np.vstack(X_parts)


def f05_threshold_search(probs, y_true, thresholds=None):
    """Find best threshold for F_0.5 on a validation set."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.02)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        if f05 > best_f:
            best_f, best_t = f05, t
    return best_t, best_f


def main(sample_size: int = 200000, neg_ratio: int = 5, seed: int = 42):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    print(f"Loading training data...")
    s1, s2, s3, gt = load_train()
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    gt_lookup = build_gt_lookup(gt)
    print(f"  S1={len(s1):,}, pool={len(pool):,}")

    # Add norm_name / norm_addr columns needed by build_tfidf_vectorizers and build_pairs
    print("Normalizing text fields...")
    s1["norm_name"] = s1["business_name"].apply(normalize_name)
    s1["norm_addr"] = s1["business_address"].apply(normalize_address)
    pool["norm_name"] = pool["business_name"].apply(normalize_name)
    pool["norm_addr"] = pool["business_address"].apply(normalize_address)
    gc.collect()

    # Sample S1 for training (simple random sample preserves distribution)
    sample_size_actual = min(sample_size, len(s1))
    s1_sample = s1.sample(n=sample_size_actual, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {len(s1_sample):,} S1 entities for training")

    # Fit TF-IDF on full corpus (norm_name/norm_addr now present)
    print("Fitting TF-IDF vectorizers on full corpus...")
    vec_name, vec_addr = build_tfidf_vectorizers(s1, pool)
    gc.collect()

    # Save vectorizers early
    vec_path = MODEL_DIR / "tfidf_vectorizers.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump({"vec_name": vec_name, "vec_addr": vec_addr}, f)
    print(f"Vectorizers saved: {vec_path}")

    # Build training pairs
    print(f"\nBuilding training pairs (neg_ratio={neg_ratio})...")
    s1_recs, cand_recs, y, groups = build_pairs(s1_sample, pool, gt_lookup, neg_ratio, rng)
    print(f"  Pairs: {len(y):,} (pos={y.sum():,}, neg={(y==0).sum():,})")

    # Train/val split by group
    unique_groups = np.unique(groups)
    val_groups = rng.choice(unique_groups, size=int(0.15 * len(unique_groups)), replace=False)
    val_mask = np.isin(groups, val_groups)
    train_mask = ~val_mask

    X_train = compute_features_batched(
        [r for r, m in zip(s1_recs, train_mask) if m],
        [r for r, m in zip(cand_recs, train_mask) if m],
        vec_name, vec_addr
    )
    y_train = y[train_mask]

    X_val = compute_features_batched(
        [r for r, m in zip(s1_recs, val_mask) if m],
        [r for r, m in zip(cand_recs, val_mask) if m],
        vec_name, vec_addr
    )
    y_val = y[val_mask]

    print(f"\nTrain: {len(y_train):,}, Val: {len(y_val):,}")

    # Train LightGBM
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
        "lambda_l1": 0.1,
        "lambda_l2": 0.5,
        "scale_pos_weight": pos_weight,
        "verbose": -1,
        "n_jobs": -1,
        "seed": seed,
        "max_bin": 255,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    # Threshold tuning on validation set
    print("\nTuning decision threshold...")
    val_probs = model.predict(X_val)
    best_t, best_f05 = f05_threshold_search(val_probs, y_val)
    print(f"  Best threshold: {best_t:.2f} -> Val F_0.5 ~= {best_f05:.4f}")

    # Save threshold
    thresh_path = MODEL_DIR / "threshold.txt"
    thresh_path.write_text(str(best_t))

    # Feature importance
    print("\nTop features by gain:")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )
    for fname, imp in importances[:10]:
        print(f"  {fname}: {imp:.0f}")

    # Save model
    model_path = MODEL_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    print(f"Training time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200000,
                        help="Number of S1 entities to sample for training")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Negative to positive ratio per entity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.sample_size, args.neg_ratio, args.seed)
