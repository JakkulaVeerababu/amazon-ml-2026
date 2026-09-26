"""
01_eda.py — Exploratory Data Analysis
Loads all train data and prints key statistics to guide pipeline design.
Run from student_resource/: python code/business_entity_resolution/src/01_eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / "dataset"


def load_data():
    s1 = pd.read_csv(BASE / "train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(BASE / "train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(BASE / "train/train_source3.tsv", sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(BASE / "train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    return s1, s2, s3, gt


def main():
    s1, s2, s3, gt = load_data()

    print("=" * 60)
    print("SHAPES")
    print("=" * 60)
    for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3), ("GroundTruth", gt)]:
        print(f"  {name}: {df.shape}")

    print()
    print("=" * 60)
    print("SAMPLE ROWS (Source1)")
    print("=" * 60)
    print(s1.head(5).to_string())

    print()
    print("=" * 60)
    print("COUNTRY DISTRIBUTIONS")
    print("=" * 60)
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(f"  {name}: {df['country'].value_counts().to_dict()}")

    print()
    print("=" * 60)
    print("GROUND TRUTH STATS")
    print("=" * 60)
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    )
    print(f"  Total S1 entities in GT: {len(gt)}")
    print(f"  Singletons (0 matches): {(gt['n_matches']==0).sum()} ({100*(gt['n_matches']==0).mean():.1f}%)")
    print(f"  Has ≥1 match: {(gt['n_matches']>0).sum()} ({100*(gt['n_matches']>0).mean():.1f}%)")
    print(f"  Max matches for one entity: {gt['n_matches'].max()}")
    print(f"  Avg matches (non-singleton): {gt[gt['n_matches']>0]['n_matches'].mean():.2f}")
    print()
    print("  Match count distribution (top 15):")
    print(gt["n_matches"].value_counts().sort_index().head(15).to_string())

    # Check which sources appear in matches
    all_matches = ",".join(gt["matched_entity_ids"].values)
    s2_count = all_matches.count("S2-")
    s3_count = all_matches.count("S3-")
    print(f"\n  S2 IDs in ground truth: {s2_count}")
    print(f"  S3 IDs in ground truth: {s3_count}")

    print()
    print("=" * 60)
    print("FIELD COMPLETENESS")
    print("=" * 60)
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        for col in ["business_name", "business_address", "country"]:
            empty = (df[col] == "").sum()
            print(f"  {name}.{col}: {empty} empty ({100*empty/len(df):.1f}%)")

    print()
    print("=" * 60)
    print("NAME LENGTH STATS")
    print("=" * 60)
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        lengths = df["business_name"].str.len()
        print(f"  {name}: min={lengths.min()}, median={lengths.median():.0f}, max={lengths.max()}")

    print()
    print("EDA complete.")


if __name__ == "__main__":
    main()
