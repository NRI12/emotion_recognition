"""Run all predefined pipelines sequentially.

Each pipeline is a (feature_extraction, model) pair.  Results from every
run are appended to outputs/runs/all_results.csv by train.py, so after
this script finishes you can call compare.py to rank them.

Usage:
    python run_pipelines.py

    # Skip certain pipelines (comma-separated indices)
    python run_pipelines.py --skip 4,5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List


PIPELINES: List[Dict[str, str]] = [
    # Classical ML
    {"name": "mfcc_random_forest",       "feature_extraction": "mfcc",   "model": "random_forest"},
    {"name": "mfcc_svm",                 "feature_extraction": "mfcc",   "model": "svm"},
    {"name": "mfcc_logistic_regression", "feature_extraction": "mfcc",   "model": "logistic_regression"},
    # Deep learning
    {"name": "mfcc_mlp",                 "feature_extraction": "mfcc",   "model": "mlp"},
    {"name": "melspec_cnn",              "feature_extraction": "melspec", "model": "cnn"},
    {"name": "logmel_cnn",               "feature_extraction": "logmel",  "model": "cnn"},
]


def _run(pipeline: Dict[str, str]) -> int:
    cmd = [
        sys.executable, "train.py",
        f"feature_extraction={pipeline['feature_extraction']}",
        f"model={pipeline['model']}",
        f"pipeline_name={pipeline['name']}",
        f"hydra.run.dir=outputs/runs/{pipeline['name']}",
    ]
    sep = "=" * 60
    print(f"\n{sep}\nPipeline: {pipeline['name']}\n{sep}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated 0-based indices of pipelines to skip",
    )
    args = parser.parse_args()
    skip = {int(i) for i in args.skip.split(",") if i.strip()}

    failed: List[str] = []
    for idx, pipeline in enumerate(PIPELINES):
        if idx in skip:
            print(f"Skipping [{idx}] {pipeline['name']}")
            continue
        rc = _run(pipeline)
        if rc != 0:
            failed.append(pipeline["name"])

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Failed pipelines: {failed}")
        sys.exit(1)
    else:
        print("All pipelines completed successfully.")
        print("Run `python compare.py` to rank results.")


if __name__ == "__main__":
    main()
