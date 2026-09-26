"""
03_blocking.py — Scalable Candidate Generation for 10M+ Records

Architecture for 2.2M S1 × 10.3M pool:
  1. Country partitioning (US: 1.3M S1 × 6.2M pool, India: 0.9M S1 × 4.1M pool)
  2. TF-IDF sparse matrix → chunked cosine similarity (batched, memory-safe)
  3. Token inverted index (trigram) as supplementary recall
  4. Address rescue for entities with zero name candidates

Run from student_resource/:
  python code/business_entity_resolution/src/03_blocking.py --split train
  python code/business_entity_resolution/src/03_blocking.py --split test
"""

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import (
    normalize_name, normalize_address,
    get_name_tokens, get_address_tokens, make_ngrams
)

BASE = Path(__file__).resolve().parents[3]
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Tunable knobs ────────────────────────────────────────────────────────────
TFIDF_TOP_K     = 30    # TF-IDF top-K per S1 entity (name)
TFIDF_MIN_SIM   = 0.05  # Minimum cosine similarity to keep a candidate
TFIDF_S1_CHUNK  = 5000   # S1 batch size for dual-chunked TF-IDF
TFIDF_POOL_CHUNK = 500_000  # Pool sub-chunk size
TRIGRAM_TOP_K   = 20    # Trigram inverted index top-K
ADDR_MIN_TOKENS = 2     # Min shared address tokens for rescue pass
MAX_CANDIDATES  = 80    # Hard cap per S1 entity
# ────────────────────────────────────────────────────────────────────────────


def load_split(split: str):
    prefix = split  # train or test
    folder = BASE / f"dataset/{split}"
    s1 = pd.read_csv(folder / f"{prefix}_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(folder / f"{prefix}_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(folder / f"{prefix}_source3.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    return df


def tfidf_blocking_chunked(
    s1: pd.DataFrame, pool: pd.DataFrame,
    top_k: int = TFIDF_TOP_K,
    min_sim: float = TFIDF_MIN_SIM,
    s1_chunk: int = 5000,
    pool_chunk: int = 500_000,
) -> dict:
    """
    Dual-chunked TF-IDF cosine similarity blocking.
    S1 is processed in chunks of s1_chunk rows.
    Pool is processed in sub-chunks of pool_chunk rows.
    A running top-K heap per S1 entity tracks best candidates across all pool sub-chunks.
    This never materialises the full (s1_chunk × pool_size) dense matrix.
    """
    from sklearn.preprocessing import normalize as sk_normalize
    import heapq

    print(f"      Fitting TF-IDF on {len(s1)+len(pool)} records...")
    all_texts = list(s1["norm_name"].values) + list(pool["norm_name"].values)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.1,
        sublinear_tf=True,
    )
    vectorizer.fit(all_texts)
    del all_texts
    gc.collect()

    # Pre-normalise and transform pool in sub-chunks to avoid one giant matrix
    print(f"      Transforming pool ({len(pool):,} records) in sub-chunks...")
    pool_ids = np.array(pool["entity_id"].values)
    pool_names = pool["norm_name"].values
    s1_ids = np.array(s1["entity_id"].values)
    s1_names = s1["norm_name"].values

    # We'll iterate S1 in chunks, and for each S1-chunk scan all pool sub-chunks.
    # Per S1 entity we keep a min-heap of size top_k: (sim, pool_idx)
    n_s1 = len(s1)
    n_s1_chunks = (n_s1 + s1_chunk - 1) // s1_chunk
    n_pool_chunks = (len(pool) + pool_chunk - 1) // pool_chunk

    candidates = {}

    for s1_ci in range(n_s1_chunks):
        s1_start = s1_ci * s1_chunk
        s1_end = min(s1_start + s1_chunk, n_s1)
        batch_size = s1_end - s1_start

        # Transform + L2-normalise this S1 chunk (sparse)
        s1_mat = sk_normalize(
            vectorizer.transform(s1_names[s1_start:s1_end]), norm="l2"
        )  # shape: (batch_size, vocab)

        # Running top-K: list of (min-heap) per S1 entity in this chunk
        # heap entries: (-sim, pool_global_idx)  — negative so max becomes min-heap pop
        heaps = [[] for _ in range(batch_size)]

        for p_ci in range(n_pool_chunks):
            p_start = p_ci * pool_chunk
            p_end = min(p_start + pool_chunk, len(pool))

            pool_mat = sk_normalize(
                vectorizer.transform(pool_names[p_start:p_end]), norm="l2"
            )  # shape: (p_end-p_start, vocab)

            try:
                import cupyx.scipy.sparse as cxs
                # GPU Sparse dot product
                s1_gpu = cxs.csr_matrix(s1_mat)
                pool_gpu = cxs.csr_matrix(pool_mat.T)
                sim_gpu = s1_gpu.dot(pool_gpu)
                sim_csr = sim_gpu.get()
                del s1_gpu, pool_gpu, sim_gpu
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                # CPU Sparse dot product -> CSR format
                sim_csr = s1_mat @ pool_mat.T
            # Process CSR matrix row by row using numpy vectorization
            indptr = sim_csr.indptr
            indices = sim_csr.indices
            data = sim_csr.data

            for i in range(batch_size):
                row_start = indptr[i]
                row_end = indptr[i+1]
                if row_start == row_end:
                    continue
                
                r_data = data[row_start:row_end]
                r_indices = indices[row_start:row_end]
                
                # Filter by min_sim
                valid_mask = r_data >= min_sim
                if not valid_mask.any():
                    continue
                
                v_data = r_data[valid_mask]
                v_indices = r_indices[valid_mask]
                
                # Top-K
                if len(v_data) > top_k:
                    # argpartition is O(N)
                    top_idx = np.argpartition(v_data, -top_k)[-top_k:]
                    v_data = v_data[top_idx]
                    v_indices = v_indices[top_idx]
                
                # Push to heap (now only doing max top_k pushes instead of thousands)
                heap = heaps[i]
                for val, col_i in zip(v_data, v_indices):
                    global_idx = p_start + col_i
                    if len(heap) < top_k:
                        heapq.heappush(heap, (val, global_idx))
                    elif val > heap[0][0]:
                        heapq.heapreplace(heap, (val, global_idx))

            del sim_csr

            gc.collect()

        # Extract candidates from heaps
        for i in range(batch_size):
            sid = s1_ids[s1_start + i]
            candidates[sid] = {pool_ids[idx] for _, idx in heaps[i]}

        if (s1_ci + 1) % 20 == 0 or s1_ci == 0:
            print(f"      TF-IDF: {s1_end:,}/{n_s1:,} S1 done")

    gc.collect()
    return candidates


def trigram_blocking(
    s1: pd.DataFrame, pool: pd.DataFrame,
    top_k: int = TRIGRAM_TOP_K,
) -> dict:
    """
    Character trigram inverted index blocking.
    Returns: dict {s1_id → set of pool_ids}
    """
    print(f"      Building trigram index for {len(pool)} pool records...")
    inv_idx = defaultdict(list)
    pool_ids = pool["entity_id"].values

    for i, name in enumerate(pool["norm_name"].values):
        tgrams = make_ngrams(name, 3)
        for tg in tgrams:
            inv_idx[tg].append(i)

    print(f"      Trigram index size: {len(inv_idx)} grams")

    candidates = {}
    for j, (sid, name) in enumerate(zip(s1["entity_id"].values, s1["norm_name"].values)):
        counter = defaultdict(int)
        for tg in make_ngrams(name, 3):
            for pool_idx in inv_idx.get(tg, []):
                counter[pool_idx] += 1
        if not counter:
            candidates[sid] = set()
            continue
        top_idxs = sorted(counter, key=lambda x: counter[x], reverse=True)[:top_k]
        candidates[sid] = set(pool_ids[i] for i in top_idxs)

        if (j + 1) % 100000 == 0:
            print(f"      Trigram: {j+1}/{len(s1)} done")

    return candidates


def address_rescue(
    s1: pd.DataFrame, pool: pd.DataFrame,
    existing: dict,
    min_overlap: int = ADDR_MIN_TOKENS,
) -> dict:
    """
    Address token rescue: add candidates sharing ≥ min_overlap address tokens.
    Only for S1 entities with few/no name candidates.
    """
    # Build address token inverted index
    inv_idx = defaultdict(list)
    pool_ids = pool["entity_id"].values

    for i, addr in enumerate(pool["norm_addr"].values):
        tokens = get_address_tokens(addr)
        for t in tokens:
            if len(t) >= 3:
                inv_idx[t].append(i)

    for sid, addr in zip(s1["entity_id"].values, s1["norm_addr"].values):
        # Only rescue entities with fewer than 5 candidates
        if len(existing.get(sid, set())) >= 5:
            continue
        tokens = get_address_tokens(addr)
        counter = defaultdict(int)
        for t in tokens:
            if len(t) >= 3:
                for pool_idx in inv_idx.get(t, []):
                    counter[pool_idx] += 1
        new_cands = {pool_ids[i] for i, cnt in counter.items() if cnt >= min_overlap}
        existing.setdefault(sid, set()).update(new_cands)

    return existing


def process_country_group(
    s1_grp: pd.DataFrame, pool_grp: pd.DataFrame, country: str
) -> dict:
    """Full blocking pipeline for one country group."""
    if pool_grp.empty:
        return {eid: set() for eid in s1_grp["entity_id"]}

    print(f"\n  [{country}] S1={len(s1_grp):,}, pool={len(pool_grp):,}")

    print(f"    Running TF-IDF blocking (dual-chunked: S1×{TFIDF_S1_CHUNK}, pool×{TFIDF_POOL_CHUNK})...")

    # Memory-safe dual-chunked TF-IDF (never densifies full sim matrix)
    if len(pool_grp) > 500000:
        cands = tfidf_blocking_chunked(
            s1_grp, pool_grp, top_k=TFIDF_TOP_K, min_sim=0.08,
            s1_chunk=TFIDF_S1_CHUNK, pool_chunk=TFIDF_POOL_CHUNK,
        )
    else:
        cands = tfidf_blocking_chunked(
            s1_grp, pool_grp, top_k=TFIDF_TOP_K, min_sim=TFIDF_MIN_SIM,
            s1_chunk=TFIDF_S1_CHUNK, pool_chunk=TFIDF_POOL_CHUNK,
        )

    # Trigram supplementary recall
    # print(f"    Running trigram blocking...")
    # tg_cands = trigram_blocking(s1_grp, pool_grp, top_k=TRIGRAM_TOP_K)
    # for sid in s1_grp["entity_id"]:
    #     cands.setdefault(sid, set()).update(tg_cands.get(sid, set()))

    # Address rescue for entities with few candidates
    n_empty = sum(1 for c in cands.values() if len(c) < 5)
    # if n_empty > 0:
    #     print(f"    Address rescue for {n_empty} entities with < 5 candidates...")
    #     cands = address_rescue(s1_grp, pool_grp, cands)

    # Cap per entity
    for sid in cands:
        if len(cands[sid]) > MAX_CANDIDATES:
            cands[sid] = set(list(cands[sid])[:MAX_CANDIDATES])

    # Stats
    n_nonempty = sum(1 for c in cands.values() if c)
    total = sum(len(c) for c in cands.values())
    print(f"    -> {n_nonempty}/{len(s1_grp)} entities have candidates, {total:,} total pairs")

    return cands


def run_blocking(split: str):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"BLOCKING - {split.upper()} split")
    print(f"{'='*60}")

    s1, s2, s3 = load_split(split)
    print(f"Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")

    print("Normalizing text...")
    s1 = normalize_df(s1)
    s2 = normalize_df(s2)
    s3 = normalize_df(s3)

    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    all_candidates = {eid: set() for eid in s1["entity_id"]}

    # Country-partitioned blocking
    countries = sorted(set(s1["country"].unique()) | set(pool["country"].unique()))
    print(f"\nCountries to process: {countries}")

    for country in countries:
        s1_grp = s1[s1["country"] == country].reset_index(drop=True)
        pool_grp = pool[pool["country"] == country].reset_index(drop=True)
        if s1_grp.empty:
            continue

        cands = process_country_group(s1_grp, pool_grp, country)
        for eid, ids in cands.items():
            all_candidates[eid].update(ids)

        del cands, s1_grp, pool_grp
        gc.collect()

    # Cross-country rescue for entities still with zero candidates
    no_cand_eids = [eid for eid, c in all_candidates.items() if not c]
    if no_cand_eids:
        print(f"\nCross-country rescue for {len(no_cand_eids):,} entities...")
        s1_rescue = s1[s1["entity_id"].isin(set(no_cand_eids))].reset_index(drop=True)
        # Try all pool
        tg_cands = trigram_blocking(s1_rescue, pool, top_k=15)
        for eid, ids in tg_cands.items():
            all_candidates[eid].update(ids)

    # Build output
    print("\nBuilding output TSV...")
    rows = []
    for eid in s1["entity_id"]:
        cand_list = sorted(all_candidates.get(eid, set()))
        rows.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(cand_list),
        })
    result_df = pd.DataFrame(rows)

    # Stats
    n_nonempty = result_df["candidate_entity_ids"].str.strip().astype(bool).sum()
    total = result_df["candidate_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    ).sum()

    print(f"\n{'='*60}")
    print(f"BLOCKING COMPLETE ({split})")
    print(f"  S1 entities: {len(result_df):,}")
    print(f"  With candidates: {n_nonempty:,} ({100*n_nonempty/len(result_df):.1f}%)")
    print(f"  Total candidate pairs: {total:,}")
    print(f"  Avg per entity: {total/len(result_df):.1f}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")

    out_path = OUTPUT_DIR / "candidate_pairs.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args()
    run_blocking(args.split)
