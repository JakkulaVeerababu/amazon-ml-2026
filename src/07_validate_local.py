"""
07_validate_local.py — Local F_0.5 Validation on a Train/Val Split
Holds out a portion of training data to estimate leaderboard score.

Run from student_resource/:
  python code/business_entity_resolution/src/07_validate_local.py \
      --predictions output/matching_results_val.tsv \
      --ground-truth dataset/train/train_ground_truth.tsv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def f05_per_entity(y_pred_set: set, y_true_set: set) -> float:
    """Compute F_0.5 for a single entity."""
    if not y_true_set and not y_pred_set:
        return 1.0  # Singleton correctly identified
    if not y_true_set and y_pred_set:
        return 0.0  # False positive on singleton
    if not y_pred_set:
        # No prediction for a non-singleton
        # Precision=1 (vacuously), Recall=0 → F0.5=0
        return 0.0

    tp = len(y_pred_set & y_true_set)
    precision = tp / len(y_pred_set) if y_pred_set else 1.0
    recall = tp / len(y_true_set) if y_true_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def compute_macro_f05(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """
    Compute macro-averaged F_0.5.
    predictions: DataFrame with [source1_entity_id, matched_entity_ids]
    ground_truth: DataFrame with [source1_entity_id, matched_entity_ids]
    """
    gt_lookup = {}
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = str(row.get("matched_entity_ids", "")).strip()
        if matches_str and matches_str != "nan":
            gt_lookup[s1_id] = set(matches_str.split(","))
        else:
            gt_lookup[s1_id] = set()

    pred_lookup = {}
    for _, row in predictions.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = str(row.get("matched_entity_ids", "")).strip()
        if matches_str and matches_str != "nan":
            pred_lookup[s1_id] = set(matches_str.split(","))
        else:
            pred_lookup[s1_id] = set()

    scores = []
    precision_list = []
    recall_list = []
    singleton_correct = 0
    singleton_total = 0

    for s1_id, y_true in gt_lookup.items():
        y_pred = pred_lookup.get(s1_id, set())
        score = f05_per_entity(y_pred, y_true)
        scores.append(score)

        if not y_true:
            singleton_total += 1
            if not y_pred:
                singleton_correct += 1

        if y_true:
            tp = len(y_pred & y_true)
            precision_list.append(tp / len(y_pred) if y_pred else 1.0)
            recall_list.append(tp / len(y_true))

    return {
        "macro_f05": float(np.mean(scores)),
        "n_entities": len(scores),
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else 0.0,
        "singleton_total": singleton_total,
        "avg_precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "avg_recall": float(np.mean(recall_list)) if recall_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="output/matching_results.tsv",
        help="Path to predictions TSV (source1_entity_id, matched_entity_ids)"
    )
    parser.add_argument(
        "--ground-truth",
        default="dataset/train/train_ground_truth.tsv",
        help="Path to ground truth TSV"
    )
    args = parser.parse_args()

    print(f"Loading predictions: {args.predictions}")
    preds = pd.read_csv(args.predictions, sep="\t", dtype=str).fillna("")

    print(f"Loading ground truth: {args.ground_truth}")
    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna("")

    # Only score entities that appear in ground truth
    gt_ids = set(gt["source1_entity_id"])
    pred_in_gt = preds[preds["source1_entity_id"].isin(gt_ids)]
    print(f"  GT entities: {len(gt_ids)}, predictions covering GT: {len(pred_in_gt)}")

    metrics = compute_macro_f05(pred_in_gt, gt)

    print(f"\n{'='*50}")
    print(f"LOCAL VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  Macro F_0.5 Score:      {metrics['macro_f05']:.4f}")
    print(f"  Avg Precision:          {metrics['avg_precision']:.4f}")
    print(f"  Avg Recall:             {metrics['avg_recall']:.4f}")
    print(f"  Singleton Accuracy:     {metrics['singleton_accuracy']:.4f} ({metrics['singleton_correct']}/{metrics['singleton_total']})" if 'singleton_correct' in metrics else f"  Singleton Accuracy:     {metrics['singleton_accuracy']:.4f}")
    print(f"  Total entities scored:  {metrics['n_entities']}")
    print(f"{'='*50}")

    return metrics


if __name__ == "__main__":
    main()
