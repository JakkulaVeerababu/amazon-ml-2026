# Business Entity Resolution Pipeline
## Amazon ML Challenge 2026 — Team Submission

---

## Overview

A two-stage ML pipeline for cross-source business entity resolution:
1. **Blocking** — fast candidate generation (TF-IDF + trigram + address rescue)
2. **Matching** — LightGBM classifier on 18 pairwise similarity features

---

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocess_utils.py   # Text normalization (names, addresses)
│   ├── features.py           # 18 pairwise similarity features
│   ├── 01_eda.py             # Exploratory data analysis
│   ├── 02_preprocess.py      # Preprocessing smoke test
│   ├── 03_blocking.py        # Candidate generation stage
│   ├── 04_features.py        # Feature engineering module
│   ├── 05_train_model.py     # LightGBM training
│   ├── 06_inference.py       # Inference + output generation
│   ├── 07_validate_local.py  # Local F_0.5 scorer
│   └── 08_run_pipeline.py    # Master pipeline runner
├── models/                   # Saved model artifacts (auto-created)
├── requirements.txt
└── README.md (this file)
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas, numpy, scikit-learn
- lightgbm
- rapidfuzz (fast string similarity)
- sentence-transformers (for Phase 2 embeddings)
- torch, transformers

---

## Reproducing Results (End-to-End)

Run all commands from the `student_resource/` directory.

### Step 1 — Blocking (Candidate Generation)

```bash
python code/business_entity_resolution/src/03_blocking.py --split test
```

Produces: `output/candidate_pairs.tsv`

### Step 2 — Train the Matching Model

```bash
python code/business_entity_resolution/src/05_train_model.py
```

Produces: `code/business_entity_resolution/models/lgbm_model.txt`
          `code/business_entity_resolution/models/tfidf_vectorizers.pkl`

### Step 3 — Run Inference

```bash
python code/business_entity_resolution/src/06_inference.py \
    --candidates output/candidate_pairs.tsv \
    --threshold 0.5
```

Produces: `output/matching_results.tsv`

### Step 4 — Validate Format

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### One-Command Run (all steps)

```bash
python code/business_entity_resolution/src/08_run_pipeline.py --mode full
```

---

## Pipeline Details

### Blocking Strategy
- **Country partitioning**: Only compare records within the same country
- **TF-IDF cosine blocking**: Top-50 candidates per S1 entity using 1-2 gram TF-IDF on normalized names
- **Character trigram inverted index**: Top-30 additional candidates via trigram overlap on names
- **Address rescue pass**: Additional candidates sharing ≥2 address tokens
- **Cross-country rescue**: For entities with zero candidates after country-partitioned blocking
- Cap: 100 candidates max per S1 entity

### Feature Engineering (18 features)
| Group | Features |
|-------|---------|
| Name | TF-IDF cosine, Jaro-Winkler, Levenshtein ratio, token Jaccard, token overlap, trigram Jaccard, prefix word match |
| Address | TF-IDF cosine, Jaro-Winkler, Levenshtein ratio, token Jaccard, token overlap, trigram Jaccard |
| Structural | Country exact match, name length ratio, address length ratio, both-have-address flag, cross-field token overlap |

### Matching Model
- **Algorithm**: LightGBM binary classifier
- **Training**: Positive pairs from ground truth + 5× sampled hard negatives (same-country preferred)
- **Objective**: Binary cross-entropy with class-weight balancing
- **Threshold**: Tuned on held-out validation split to maximize F_0.5

### Threshold Selection
Since F_0.5 is precision-heavy (penalizes false positives 2×), we tune the threshold to favor precision over recall. Recommended range: 0.45–0.65.

---

## Local Validation

```bash
python code/business_entity_resolution/src/07_validate_local.py \
    --predictions output/matching_results_val.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv
```

---

## Notes

- All source code is MIT/Apache 2.0 compatible
- No external data lookups or APIs are used
- France (unseen country in test set) is handled via open-set country partitioning
- Models are ≤8B parameters (LightGBM is parameter-free in that sense; embedding models used are sub-500M)
