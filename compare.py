"""Aggregate and rank all experiment results.

Reads every all_results.csv under outputs/runs/, merges them, ranks by the
chosen metric, and writes outputs/comparison.csv.

Augmentation comparison (--compare-aug)
-----------------------------------------
When enabled, pipelines that have both a base run and an augmented run
(_aug suffix) are shown side-by-side with delta columns so you can see
the effect of augmentation at a glance:

    pipeline_base | test/f1 | test/f1_aug | Δf1  | test/acc | test/acc_aug | Δacc

Usage:
    python compare.py                         # rank all runs by test/f1
    python compare.py --metric test/acc
    python compare.py --compare-aug           # side-by-side aug comparison
    python compare.py --compare-aug --metric val/f1
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "**", "all_results.csv")
    files = glob.glob(pattern, recursive=True)

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


# ---------------------------------------------------------------------------
# Standard ranking table
# ---------------------------------------------------------------------------

def show_ranked(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        return df

    ranked = df.sort_values(metric, ascending=False).reset_index(drop=True)
    ranked.index += 1

    display_cols = [
        "pipeline_name", "model_type", "feature_method", "augmented",
        "test/acc", "test/f1", "val/acc", "val/f1",
        "train_time_sec", "timestamp",
    ]
    display_cols = [c for c in display_cols if c in ranked.columns]

    print(f"\n=== Model Comparison — ranked by {metric} ===\n")
    print(ranked[display_cols].to_string())
    return ranked


# ---------------------------------------------------------------------------
# Side-by-side augmentation comparison
# ---------------------------------------------------------------------------

def show_aug_comparison(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot aug vs no-aug into one row per base pipeline with delta columns."""

    if "augmented" not in df.columns:
        print("[warn] 'augmented' column not found — run pipelines first with --aug-mode both.")
        return pd.DataFrame()

    # Normalise augmented column to bool
    df = df.copy()
    df["augmented"] = df["augmented"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Derive base name: strip _aug suffix
    df["base_name"] = df["pipeline_name"].str.replace(r"_aug$", "", regex=True)

    base_df = df[~df["augmented"]].copy()
    aug_df  = df[ df["augmented"]].copy()

    if aug_df.empty:
        print("[warn] No augmented runs found. Run with --aug-mode aug or --aug-mode both first.")
        return pd.DataFrame()

    keep_cols = ["base_name", "model_type", "feature_method",
                 "test/acc", "test/f1", "val/acc", "val/f1", "train_time_sec"]

    # Take the latest run per base_name (in case of reruns)
    def latest(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if "timestamp" in frame.columns:
            return (frame.sort_values("timestamp")
                         .groupby(group_col, as_index=False)
                         .last())
        return frame.groupby(group_col, as_index=False).last()

    base_latest = latest(base_df, "base_name")
    aug_latest  = latest(aug_df,  "base_name")

    merged = base_latest[[c for c in keep_cols if c in base_latest.columns]].merge(
        aug_latest[["base_name", "test/acc", "test/f1", "val/acc", "val/f1",
                    "train_time_sec"]],
        on="base_name",
        suffixes=("", "_aug"),
        how="outer",
    )

    # Delta columns (aug − base); only for numeric metric pairs
    for col in ("test/f1", "test/acc", "val/f1", "val/acc"):
        aug_col = f"{col}_aug"
        if col in merged.columns and aug_col in merged.columns:
            merged[f"Δ{col.split('/')[1]}"] = (
                merged[aug_col].sub(merged[col]).round(4)
            )

    # Sort by the base metric
    sort_col = metric if metric in merged.columns else "test/f1"
    merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)
    merged.index += 1

    display_cols = [
        "base_name", "model_type", "feature_method",
        "test/f1", "test/f1_aug", "Δf1",
        "test/acc", "test/acc_aug", "Δacc",
        "val/f1",  "val/f1_aug",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]

    print(f"\n=== Augmentation Comparison — sorted by {sort_col} (base) ===")
    print("  Positive Δ = augmentation helped\n")
    print(merged[display_cols].to_string())
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare emotion recognition experiment results.")
    parser.add_argument("--results_dir", default="outputs/runs")
    parser.add_argument("--metric", default="test/f1",
        help="Metric to sort by (default: test/f1).")
    parser.add_argument("--compare-aug", action="store_true",
        help="Show side-by-side aug vs no-aug comparison table.")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    if df.empty:
        return

    os.makedirs("outputs", exist_ok=True)

    # ── standard ranking ──────────────────────────────────────────────────
    ranked = show_ranked(df, args.metric)
    out_all = "outputs/comparison.csv"
    ranked.to_csv(out_all, index_label="rank")
    print(f"\nFull comparison saved → {out_all}")

    # ── augmentation side-by-side ─────────────────────────────────────────
    if args.compare_aug:
        aug_table = show_aug_comparison(df, args.metric)
        if not aug_table.empty:
            out_aug = "outputs/comparison_aug.csv"
            aug_table.to_csv(out_aug, index_label="rank")
            print(f"Aug comparison saved  → {out_aug}")


if __name__ == "__main__":
    main()
