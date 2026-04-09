"""Tune hyperparameters for all (or selected) pipelines sequentially.

Each pipeline calls tune.py via subprocess, which runs an Optuna study and
saves best params to outputs/best_params/<model>_<feature>.json.

Usage
-----
    # Tune all pipelines (default n_trials from configs/tuning/default.yaml)
    python run_tuning.py

    # Tune specific models by alias
    python run_tuning.py --models rf,svm,cnn

    # Override number of trials
    python run_tuning.py --n_trials 50

    # Tune only classical models, 20 trials each
    python run_tuning.py --models rf,svm,lr --n_trials 20

After tuning, apply best params and train:
    python run_pipelines.py --models rf        # uses default config
    # Manually override with best params from outputs/best_params/*.json

Model aliases:
    rf         → mfcc + random_forest
    svm        → mfcc + svm
    lr         → mfcc + logistic_regression
    mlp        → mfcc + mlp
    cnn        → melspec + cnn
    logmel_cnn → logmel + cnn
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List


TUNE_PIPELINES: List[Dict[str, str]] = [
    # Classical ML
    {"name": "rf",         "feature_extraction": "mfcc",    "model": "random_forest"},
    {"name": "svm",        "feature_extraction": "mfcc",    "model": "svm"},
    {"name": "lr",         "feature_extraction": "mfcc",    "model": "logistic_regression"},
    # Deep learning
    {"name": "mlp",        "feature_extraction": "mfcc",    "model": "mlp"},
    {"name": "cnn",        "feature_extraction": "melspec",  "model": "cnn"},
    {"name": "logmel_cnn", "feature_extraction": "logmel",   "model": "cnn"},
]

_ALIASES: Dict[str, str] = {p["name"]: p["name"] for p in TUNE_PIPELINES}


def _run(pipeline: Dict[str, str], n_trials: int) -> int:
    cmd = [
        sys.executable, "tune.py",
        f"feature_extraction={pipeline['feature_extraction']}",
        f"model={pipeline['model']}",
        f"tuning.n_trials={n_trials}",
        f"hydra.run.dir=outputs/tuning/{pipeline['name']}",
    ]
    sep = "=" * 60
    print(f"\n{sep}\nTuning: {pipeline['name']}  ({n_trials} trials)\n{sep}")
    return subprocess.run(cmd).returncode


def _select(models_arg: str) -> List[Dict[str, str]]:
    names = {m.strip() for m in models_arg.split(",")}
    selected = [p for p in TUNE_PIPELINES if p["name"] in names]
    missing = names - {p["name"] for p in selected}
    if missing:
        valid = ", ".join(p["name"] for p in TUNE_PIPELINES)
        print(f"[warn] Unknown model(s): {missing}. Valid: {valid}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for all (or selected) pipelines."
    )
    parser.add_argument(
        "--models", default="",
        help="Comma-separated aliases to tune (e.g. rf,svm,cnn). Omit to tune all.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=None,
        help="Number of Optuna trials per model (overrides configs/tuning/default.yaml).",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated 0-based indices to skip (ignored when --models is set).",
    )
    args = parser.parse_args()

    if args.models:
        pipelines = _select(args.models)
    else:
        skip = {int(i) for i in args.skip.split(",") if i.strip()}
        pipelines = [p for i, p in enumerate(TUNE_PIPELINES) if i not in skip]

    if not pipelines:
        print("No pipelines selected. Exiting.")
        sys.exit(0)

    # Determine n_trials: CLI arg > config default (read lazily from yaml)
    if args.n_trials is not None:
        n_trials = args.n_trials
    else:
        try:
            import yaml
            with open("configs/tuning/default.yaml") as f:
                n_trials = yaml.safe_load(f).get("n_trials", 30)
        except Exception:
            n_trials = 30

    print(f"Tuning {len(pipelines)} pipeline(s) × {n_trials} trials each.")
    print(f"Best params saved to: outputs/best_params/<name>.json")

    failed: List[str] = []
    for pipeline in pipelines:
        rc = _run(pipeline, n_trials)
        if rc != 0:
            failed.append(pipeline["name"])

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All tuning jobs completed.")
        print("Review results:  ls outputs/best_params/")
        print("Train with best: python run_pipelines.py --models rf,svm,cnn")


if __name__ == "__main__":
    main()
