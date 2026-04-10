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


def save_run_results(results: Dict[str, Any], output_path: str) -> None:
    """Append one run's metrics to the shared CSV using a fixed schema.

    - Unknown keys in `results` are silently ignored (extrasaction='ignore').
    - Missing columns get an empty string (restval='').
    - The header is written once on file creation; subsequent runs just append rows.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    row: Dict[str, Any] = {
        k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))
    }
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=RESULT_COLUMNS,
            restval="",           # fill missing columns with ""
            extrasaction="ignore", # drop keys not in RESULT_COLUMNS
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
