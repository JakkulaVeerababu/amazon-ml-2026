"""
08_run_pipeline.py — Master Pipeline Runner
Runs the full end-to-end pipeline:
  1. Blocking (candidate generation)
  2. Training (if model not cached)
  3. Inference
  4. Validation (optional, on train split)

Run from student_resource/:
  python code/business_entity_resolution/src/08_run_pipeline.py --mode full
  python code/business_entity_resolution/src/08_run_pipeline.py --mode inference-only
  python code/business_entity_resolution/src/08_run_pipeline.py --mode blocking-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent
BASE = SRC.parents[3]


def run(cmd: str):
    print(f"\n>>> {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=str(BASE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"ERROR: Command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"    Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "blocking-only", "train-only", "inference-only"],
        default="full",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Pipeline mode: {args.mode}")
    print(f"Base directory: {BASE}")

    py = sys.executable

    if args.mode in ("full", "blocking-only"):
        run(f"{py} {SRC}/03_blocking.py --split {args.split}")

    if args.mode in ("full", "train-only"):
        run(f"{py} {SRC}/05_train_model.py")

    if args.mode in ("full", "inference-only"):
        run(f"{py} {SRC}/06_inference.py --candidates output/candidate_pairs.tsv --threshold {args.threshold}")

    if args.mode == "full":
        # Run format validator
        run(
            f"{py} utils/validate_submission.py "
            f"--matching output/matching_results.tsv "
            f"--candidate output/candidate_pairs.tsv "
            f"--test-dir dataset/test"
        )

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
