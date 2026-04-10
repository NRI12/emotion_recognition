"""Re-evaluate all (or selected) trained models and update the results CSV.

Loads existing checkpoints — no retraining.

Usage
-----
    # Re-eval all pipelines that have a saved model
    python run_eval.py

    # Re-eval specific models
    python run_eval.py --models rf,svm,mlp

    # Re-eval with a specific checkpoint path (single model)
    python run_eval.py --models mlp --ckpt outputs/checkpoints/mfcc_mlp/best.ckpt

Model aliases  →  (feature_extraction, model, pipeline_name, checkpoint)
    rf         →  mfcc  + random_forest       → artifacts/models/mfcc_random_forest.pkl
    svm        →  mfcc  + svm                 → artifacts/models/mfcc_svm.pkl
    lr         →  mfcc  + logistic_regression → artifacts/models/mfcc_logistic_regression.pkl
    mlp        →  mfcc  + mlp                 → outputs/checkpoints/mfcc_mlp/best.ckpt
    cnn        →  melspec + cnn               → outputs/checkpoints/melspec_cnn/best.ckpt
    logmel_cnn →  logmel  + cnn               → outputs/checkpoints/logmel_cnn/best.ckpt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List


EVAL_PIPELINES: List[Dict[str, str]] = [
    {"name": "rf",         "feature_extraction": "mfcc",    "model": "random_forest",       "pipeline_name": "mfcc_random_forest"},
    {"name": "svm",        "feature_extraction": "mfcc",    "model": "svm",                 "pipeline_name": "mfcc_svm"},
    {"name": "lr",         "feature_extraction": "mfcc",    "model": "logistic_regression", "pipeline_name": "mfcc_logistic_regression"},
    {"name": "mlp",        "feature_extraction": "mfcc",    "model": "mlp",                 "pipeline_name": "mfcc_mlp"},
    {"name": "cnn",        "feature_extraction": "melspec", "model": "cnn",                 "pipeline_name": "melspec_cnn"},
    {"name": "logmel_cnn", "feature_extraction": "logmel",  "model": "cnn",                 "pipeline_name": "logmel_cnn"},
]


def _run(pipeline: Dict[str, str], ckpt: str = "") -> int:
    cmd = [
        sys.executable, "eval.py",
        f"feature_extraction={pipeline['feature_extraction']}",
        f"model={pipeline['model']}",
        f"pipeline_name={pipeline['pipeline_name']}",
        f"hydra.run.dir=outputs/eval/{pipeline['pipeline_name']}",
    ]
    if ckpt:
        cmd.append(f"+eval.ckpt_path={ckpt}")

    sep = "=" * 60
    print(f"\n{sep}\nEval: {pipeline['pipeline_name']}\n{sep}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained models and update results CSV."
    )
    parser.add_argument(
        "--models", default="",
        help="Comma-separated aliases (e.g. rf,svm,mlp). Omit to eval all.",
    )
    parser.add_argument(
        "--ckpt", default="",
        help="Checkpoint path override (only used when --models selects exactly 1 DL model).",
    )
    args = parser.parse_args()

    if args.models:
        names     = {m.strip() for m in args.models.split(",")}
        pipelines = [p for p in EVAL_PIPELINES if p["name"] in names]
        missing   = names - {p["name"] for p in pipelines}
        if missing:
            valid = ", ".join(p["name"] for p in EVAL_PIPELINES)
            print(f"[warn] Unknown: {missing}. Valid: {valid}")
    else:
        pipelines = EVAL_PIPELINES

    if not pipelines:
        print("No pipelines selected.")
        sys.exit(0)

    failed: List[str] = []
    for pipeline in pipelines:
        ckpt = args.ckpt if len(pipelines) == 1 else ""
        rc = _run(pipeline, ckpt)
        if rc != 0:
            failed.append(pipeline["pipeline_name"])

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All evaluations done.")
        print("Run `python compare.py` to see updated rankings.")


if __name__ == "__main__":
    main()
