"""Aggregate and rank all experiment results.

Reads every all_results.csv under outputs/runs/, merges them, ranks by
test/f1, and writes the final table to outputs/comparison.csv.

Usage:
    python compare.py
    python compare.py --metric test/acc
    python compare.py --results_dir outputs/runs
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "**", "all_results.csv")
    files = glob.glob(pattern, recursive=True)

    # Also check flat results file at top of results_dir
    top = os.path.join(results_dir, "all_results.csv")
    if os.path.isfile(top) and top not in files:
        files.append(top)

    if not files:
        print(f"No result files found under '{results_dir}'.")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"Loaded {len(df)} runs from {len(files)} file(s).")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/runs")
    parser.add_argument("--metric", default="test/f1")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        return

    if args.metric not in df.columns:
        print(f"Metric '{args.metric}' not found. Available columns: {list(df.columns)}")
        return

    ranked = df.sort_values(args.metric, ascending=False).reset_index(drop=True)
    ranked.index += 1  # 1-based rank

    display_cols = [
        "pipeline_name", "model_type", "feature_method",
        "test/acc", "test/f1", "val/acc", "val/f1",
        "train_time_sec", "timestamp",
    ]
    display_cols = [c for c in display_cols if c in ranked.columns]

    print(f"\n=== Model Comparison — ranked by {args.metric} ===\n")
    print(ranked[display_cols].to_string())

    out_path = "outputs/comparison.csv"
    os.makedirs("outputs", exist_ok=True)
    ranked.to_csv(out_path, index_label="rank")
    print(f"\nFull comparison saved -> {out_path}")


if __name__ == "__main__":
    main()
