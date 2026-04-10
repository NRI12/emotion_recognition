"""One-time script to migrate all_results.csv to the current schema.

Run once after updating RESULT_COLUMNS:
    python fix_csv.py
"""
import glob, os
from src.evaluation.metrics import _migrate_csv, RESULT_COLUMNS

paths = glob.glob("outputs/runs/**/all_results.csv", recursive=True)
top = "outputs/runs/all_results.csv"
if os.path.isfile(top) and top not in paths:
    paths.append(top)

if not paths:
    print("No CSV files found.")
else:
    for p in paths:
        print(f"Migrating: {p}")
        _migrate_csv(p)
    print(f"\nDone. Current schema: {RESULT_COLUMNS}")
