from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute accuracy, macro-F1, and confusion matrix.

    Keys are prefixed with '{prefix}/' when prefix is non-empty.
    """
    sep = "/" if prefix else ""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        f"{prefix}{sep}acc": round(acc, 4),
        f"{prefix}{sep}f1": round(f1, 4),
        f"{prefix}{sep}confusion_matrix": cm,
    }


def save_run_results(results: Dict[str, Any], output_path: str) -> None:
    """Append a single run's scalar metrics to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Keep only scalar-serializable values
    row: Dict[str, Any] = {
        k: v
        for k, v in results.items()
        if isinstance(v, (int, float, str, bool))
    }
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
