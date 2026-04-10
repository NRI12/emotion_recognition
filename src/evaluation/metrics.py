from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Fixed column order for the shared results CSV.
# Every run writes exactly these columns; missing values become empty string.
RESULT_COLUMNS = [
    "pipeline_name",
    "model_type",
    "feature_method",
    "augmented",
    "train_time_sec",
    "val/acc",
    "val/f1",
    "test/acc",
    "test/f1",
    "timestamp",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute accuracy, macro-F1, and confusion matrix."""
    sep = "/" if prefix else ""
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm  = confusion_matrix(y_true, y_pred).tolist()
    return {
        f"{prefix}{sep}acc": round(acc, 4),
        f"{prefix}{sep}f1":  round(f1,  4),
        f"{prefix}{sep}confusion_matrix": cm,
    }


def _migrate_csv(output_path: str) -> None:
    """Rewrite CSV with the current RESULT_COLUMNS schema.

    Called automatically when the existing file has a different header
    (e.g. a column was added).  Missing columns are backfilled with "".
    Rows whose column count matches the *old* header are realigned; rows
    that already have the new column count are left as-is.
    """
    with open(output_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        old_fields = reader.fieldnames or []
        rows = list(reader)

    # Nothing to do if schema already matches
    if old_fields == RESULT_COLUMNS:
        return

    # Backfill any column that is in RESULT_COLUMNS but missing from old header
    for row in rows:
        for col in RESULT_COLUMNS:
            if col not in row:
                row[col] = ""
        # If the row was written with the NEW schema but the file header was
        # still OLD, the last field will contain an extra comma-joined value.
        # Example: test/f1 == "0.6649,2026-04-10T11:23:17"
        # We detect this by checking if any field contains a comma.
        for col in list(row.keys()):
            val = row.get(col, "")
            if isinstance(val, str) and "," in val and col not in ("pipeline_name",):
                parts = val.split(",", 1)
                # Assign each part to the column it actually belongs to
                row[col] = parts[0].strip()
                # The second part is the timestamp that got merged
                if "timestamp" in RESULT_COLUMNS and len(parts) > 1:
                    row["timestamp"] = parts[1].strip()

    tmp_path = output_path + ".migrating"
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=RESULT_COLUMNS,
            restval="",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, output_path)
    added = [c for c in RESULT_COLUMNS if c not in old_fields]
    print(f"[csv-migrate] Schema updated — added columns: {added}  ({output_path})")


def save_run_results(results: Dict[str, Any], output_path: str) -> None:
    """Append one run's metrics to the shared CSV using the current schema.

    - If the file exists with an older schema, it is migrated automatically
      before the new row is appended.
    - Unknown keys in `results` are silently ignored (extrasaction='ignore').
    - Missing columns get an empty string (restval='').
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    row: Dict[str, Any] = {
        k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))
    }
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    file_exists = os.path.isfile(output_path)

    # Auto-migrate if the file exists but has a different header
    if file_exists:
        with open(output_path, newline="", encoding="utf-8") as fh:
            existing_header = (csv.DictReader(fh).fieldnames or [])
        if existing_header != RESULT_COLUMNS:
            _migrate_csv(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=RESULT_COLUMNS,
            restval="",
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
