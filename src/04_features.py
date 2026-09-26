"""
04_features.py — Feature Engineering for Candidate Pairs
Computes pairwise similarity features between a Source 1 entity and a candidate.

Features computed:
  - Name: TF-IDF cosine, Jaccard token, trigram Jaccard, Jaro-Winkler,
           Levenshtein ratio, token overlap coefficient
  - Address: Same set of metrics
  - Country: Exact match boolean
  - Cross: Name + address combined token overlap
  - Length: Ratio of name lengths and address lengths
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance as rfz_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_utils import (
    normalize_name, normalize_address,
    get_name_tokens, get_address_tokens,
    make_ngrams, jaccard, token_overlap
)

FEATURE_NAMES = [
    "name_tfidf_cos",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_trigram_jaccard",
    "name_prefix_match",       # 1 if first word matches
    "addr_tfidf_cos",
    "addr_jaro_winkler",
    "addr_levenshtein_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_trigram_jaccard",
    "country_exact",
    "name_len_ratio",
    "addr_len_ratio",
    "both_has_addr",
    "cross_token_overlap",     # name∩addr tokens across both records
]


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return 1.0 if a == 0 else 0.0
    return min(a, b) / max(a, b)


def compute_pair_features_batch(
    s1_records: list[dict],
    candidate_records: list[dict],
    tfidf_name: TfidfVectorizer = None,
    tfidf_addr: TfidfVectorizer = None,
) -> np.ndarray:
    """
    Compute features for a batch of (s1, candidate) pairs.
    s1_records and candidate_records are lists of dicts with keys:
      entity_id, business_name, business_address, country,
      norm_name, norm_addr, name_tokens, addr_tokens, name_trigrams, addr_trigrams

    Returns: np.ndarray of shape (N, len(FEATURE_NAMES))
    """
    N = len(s1_records)
    X = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)

    for i, (s1r, candr) in enumerate(zip(s1_records, candidate_records)):
        n1, n2 = s1r["norm_name"], candr["norm_name"]
        a1, a2 = s1r["norm_addr"], candr["norm_addr"]
        nt1, nt2 = s1r["name_tokens"], candr["name_tokens"]
        at1, at2 = s1r["addr_tokens"], candr["addr_tokens"]
        ng1, ng2 = s1r["name_trigrams"], candr["name_trigrams"]
        ag1, ag2 = s1r["addr_trigrams"], candr["addr_trigrams"]

        # --- Name features ---
        # tfidf_cos computed separately if vectorizer provided; placeholder here
        X[i, 0] = 0.0  # filled below
        X[i, 1] = fuzz.WRatio(n1, n2) / 100.0
        X[i, 2] = fuzz.ratio(n1, n2) / 100.0
        X[i, 3] = jaccard(nt1, nt2)
        X[i, 4] = token_overlap(nt1, nt2)
        X[i, 5] = jaccard(ng1, ng2)
        # prefix: first word match
        n1_words = n1.split()
        n2_words = n2.split()
        X[i, 6] = 1.0 if (n1_words and n2_words and n1_words[0] == n2_words[0]) else 0.0

        # --- Address features ---
        X[i, 7] = 0.0  # filled below
        X[i, 8] = fuzz.WRatio(a1, a2) / 100.0 if a1 and a2 else 0.0
        X[i, 9] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        X[i, 10] = jaccard(at1, at2)
        X[i, 11] = token_overlap(at1, at2)
        X[i, 12] = jaccard(ag1, ag2)

        # --- Country ---
        X[i, 13] = 1.0 if s1r["country"] == candr["country"] else 0.0

        # --- Length ratios ---
        X[i, 14] = _safe_ratio(len(n1), len(n2))
        X[i, 15] = _safe_ratio(len(a1), len(a2))

        # --- Both have address? ---
        X[i, 16] = 1.0 if (a1 and a2) else 0.0

        # --- Cross token overlap: union of name+addr tokens ---
        all1 = nt1 | at1
        all2 = nt2 | at2
        X[i, 17] = token_overlap(all1, all2)

    # TF-IDF cosine (batch matrix multiply — much faster)
    if tfidf_name is not None and N > 0:
        n1_texts = [r["norm_name"] for r in s1_records]
        n2_texts = [r["norm_name"] for r in candidate_records]
        m1 = tfidf_name.transform(n1_texts)
        m2 = tfidf_name.transform(n2_texts)
        # Element-wise cosine (pair-wise, not all-pairs)
        dot = np.array(m1.multiply(m2).sum(axis=1)).flatten()
        n1_norm = np.sqrt(np.array(m1.power(2).sum(axis=1)).flatten())
        n2_norm = np.sqrt(np.array(m2.power(2).sum(axis=1)).flatten())
        denom = n1_norm * n2_norm
        denom[denom == 0] = 1e-9
        X[:, 0] = dot / denom

    if tfidf_addr is not None and N > 0:
        a1_texts = [r["norm_addr"] for r in s1_records]
        a2_texts = [r["norm_addr"] for r in candidate_records]
        m1 = tfidf_addr.transform(a1_texts)
        m2 = tfidf_addr.transform(a2_texts)
        dot = np.array(m1.multiply(m2).sum(axis=1)).flatten()
        n1_norm = np.sqrt(np.array(m1.power(2).sum(axis=1)).flatten())
        n2_norm = np.sqrt(np.array(m2.power(2).sum(axis=1)).flatten())
        denom = n1_norm * n2_norm
        denom[denom == 0] = 1e-9
        X[:, 7] = dot / denom

    return X


def build_tfidf_vectorizers(train_s1: pd.DataFrame, train_pool: pd.DataFrame):
    """Fit TF-IDF vectorizers on combined name+address corpora."""
    all_names = list(train_s1["norm_name"]) + list(train_pool["norm_name"])
    all_addrs = list(train_s1["norm_addr"]) + list(train_pool["norm_addr"])

    vec_name = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
    ).fit(all_names)

    vec_addr = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
    ).fit(all_addrs)

    return vec_name, vec_addr


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add precomputed normalization columns to a DataFrame."""
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["name_tokens"] = df["norm_name"].apply(get_name_tokens)
    df["addr_tokens"] = df["norm_addr"].apply(get_address_tokens)
    df["name_trigrams"] = df["norm_name"].apply(lambda x: make_ngrams(x, 3))
    df["addr_trigrams"] = df["norm_addr"].apply(lambda x: make_ngrams(x, 3))
    return df


if __name__ == "__main__":
    # Smoke test
    r1 = {
        "norm_name": "amazon inc",
        "norm_addr": "123 main st seattle wa",
        "country": "US",
        "name_tokens": {"amazon", "inc"},
        "addr_tokens": {"123", "main", "st", "seattle", "wa"},
        "name_trigrams": make_ngrams("amazon inc", 3),
        "addr_trigrams": make_ngrams("123 main st seattle wa", 3),
    }
    r2 = {
        "norm_name": "amazon corp",
        "norm_addr": "123 main street seattle washington",
        "country": "US",
        "name_tokens": {"amazon", "corp"},
        "addr_tokens": {"123", "main", "st", "seattle", "washington"},
        "name_trigrams": make_ngrams("amazon corp", 3),
        "addr_trigrams": make_ngrams("123 main street seattle washington", 3),
    }
    X = compute_pair_features_batch([r1], [r2])
    print("Features:", dict(zip(FEATURE_NAMES, X[0])))
