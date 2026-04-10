"""Run predefined pipelines sequentially, skipping completed ones.

A pipeline is considered **done** when its artifact already exists:
  - Classical ML  → artifacts/models/<name>.pkl
  - Deep learning → outputs/checkpoints/<name>/best.ckpt

Use --force to re-run everything regardless of existing artifacts.

Usage
-----
    # Run all pipelines (skip completed)
    python run_pipelines.py

    # Run specific models by alias (comma-separated)
    python run_pipelines.py --models rf,svm,cnn

    # Force re-run even if artifacts exist
    python run_pipelines.py --force

    # Skip by 0-based index
    python run_pipelines.py --skip 4,5

Available model aliases:
    rf          → mfcc_random_forest
    svm         → mfcc_svm
    lr          → mfcc_logistic_regression
    mlp         → mfcc_mlp
    cnn         → melspec_cnn
    logmel_cnn  → logmel_cnn
    hubert_rf   → hubert_random_forest
    hubert_svm  → hubert_svm
    hubert_lr   → hubert_logistic_regression
    hubert_mlp  → hubert_mlp
    hubert_cnn    → hubert_cnn    (temporal mode)
    hubert_lstm   → hubert_lstm
    hubert_bilstm → hubert_bilstm
    wavlm_rf    → wavlm_random_forest
    wavlm_svm   → wavlm_svm
    wavlm_lr    → wavlm_logistic_regression
    wavlm_mlp   → wavlm_mlp
    wavlm_cnn     → wavlm_cnn    (temporal mode)
    wavlm_lstm    → wavlm_lstm
    wavlm_bilstm  → wavlm_bilstm
    mfcc_lstm     → mfcc_lstm
    mfcc_bilstm   → mfcc_bilstm
    logmel_lstm   → logmel_lstm
    logmel_bilstm → logmel_bilstm
    melspec_lstm  → melspec_lstm
    melspec_bilstm→ melspec_bilstm
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Each entry may include:
#   name            : unique identifier, used for artifact paths and logging
#   feature_extraction : Hydra config name under configs/feature_extraction/
#   model           : Hydra config name under configs/model/
#   preprocessing   : (optional) Hydra config name under configs/preprocessing/
#   overrides       : (optional) list of extra "key=value" strings passed to train.py
# ---------------------------------------------------------------------------

PIPELINES: List[Dict] = [
    # ── Classical ML (MFCC) ───────────────────────────────────────────────
    {"name": "mfcc_random_forest",       "feature_extraction": "mfcc", "model": "random_forest"},
    {"name": "mfcc_svm",                 "feature_extraction": "mfcc", "model": "svm"},
    {"name": "mfcc_logistic_regression", "feature_extraction": "mfcc", "model": "logistic_regression"},
    # ── Deep learning (spectral) ──────────────────────────────────────────
    {"name": "mfcc_mlp",                 "feature_extraction": "mfcc",    "model": "mlp"},
    {"name": "melspec_cnn",              "feature_extraction": "melspec",  "model": "cnn"},
    {"name": "logmel_cnn",               "feature_extraction": "logmel",   "model": "cnn"},
    # ── HuBERT (flat → RF / SVM / LR / MLP) ──────────────────────────────
    {"name": "hubert_random_forest",       "feature_extraction": "hubert", "model": "random_forest",       "preprocessing": "ssl"},
    {"name": "hubert_svm",                 "feature_extraction": "hubert", "model": "svm",                  "preprocessing": "ssl"},
    {"name": "hubert_logistic_regression", "feature_extraction": "hubert", "model": "logistic_regression",  "preprocessing": "ssl"},
    {"name": "hubert_mlp",                 "feature_extraction": "hubert", "model": "mlp",                  "preprocessing": "ssl"},
    # ── HuBERT temporal → CNN ─────────────────────────────────────────────
    {"name": "hubert_cnn", "feature_extraction": "hubert", "model": "cnn", "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    # ── WavLM (flat → RF / SVM / LR / MLP) ───────────────────────────────
    {"name": "wavlm_random_forest",       "feature_extraction": "wavlm", "model": "random_forest",       "preprocessing": "ssl"},
    {"name": "wavlm_svm",                 "feature_extraction": "wavlm", "model": "svm",                  "preprocessing": "ssl"},
    {"name": "wavlm_logistic_regression", "feature_extraction": "wavlm", "model": "logistic_regression",  "preprocessing": "ssl"},
    {"name": "wavlm_mlp",                 "feature_extraction": "wavlm", "model": "mlp",                  "preprocessing": "ssl"},
    # ── WavLM temporal → CNN ──────────────────────────────────────────────
    {"name": "wavlm_cnn", "feature_extraction": "wavlm", "model": "cnn", "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    # ── LSTM / BiLSTM (spectral) ──────────────────────────────────────────
    # MFCC default is flat; override to temporal for RNN input
    {"name": "mfcc_lstm",    "feature_extraction": "mfcc",    "model": "lstm",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "mfcc_bilstm",  "feature_extraction": "mfcc",    "model": "bilstm",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "logmel_lstm",  "feature_extraction": "logmel",  "model": "lstm"},
    {"name": "logmel_bilstm","feature_extraction": "logmel",  "model": "bilstm"},
    {"name": "melspec_lstm", "feature_extraction": "melspec", "model": "lstm"},
    {"name": "melspec_bilstm","feature_extraction": "melspec","model": "bilstm"},
    # ── LSTM / BiLSTM (HuBERT) ───────────────────────────────────────────
    {"name": "hubert_lstm",   "feature_extraction": "hubert", "model": "lstm",   "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "hubert_bilstm", "feature_extraction": "hubert", "model": "bilstm", "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    # ── LSTM / BiLSTM (WavLM) ────────────────────────────────────────────
    {"name": "wavlm_lstm",    "feature_extraction": "wavlm",  "model": "lstm",   "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "wavlm_bilstm",  "feature_extraction": "wavlm",  "model": "bilstm", "preprocessing": "ssl",
     "overrides": ["feature_extraction.output_mode=temporal"]},
]

_ALIASES: Dict[str, str] = {
    # MFCC
    "rf":         "mfcc_random_forest",
    "svm":        "mfcc_svm",
    "lr":         "mfcc_logistic_regression",
    "mlp":        "mfcc_mlp",
    "cnn":        "melspec_cnn",
    "logmel_cnn": "logmel_cnn",
    # HuBERT
    "hubert_rf":  "hubert_random_forest",
    "hubert_svm": "hubert_svm",
    "hubert_lr":  "hubert_logistic_regression",
    "hubert_mlp": "hubert_mlp",
    "hubert_cnn": "hubert_cnn",
    # WavLM
    "wavlm_rf":     "wavlm_random_forest",
    "wavlm_svm":    "wavlm_svm",
    "wavlm_lr":     "wavlm_logistic_regression",
    "wavlm_mlp":    "wavlm_mlp",
    "wavlm_cnn":    "wavlm_cnn",
    "wavlm_lstm":   "wavlm_lstm",
    "wavlm_bilstm": "wavlm_bilstm",
    # LSTM / BiLSTM (spectral)
    "mfcc_lstm":     "mfcc_lstm",
    "mfcc_bilstm":   "mfcc_bilstm",
    "logmel_lstm":   "logmel_lstm",
    "logmel_bilstm": "logmel_bilstm",
    "melspec_lstm":  "melspec_lstm",
    "melspec_bilstm":"melspec_bilstm",
    # LSTM / BiLSTM (HuBERT)
    "hubert_lstm":   "hubert_lstm",
    "hubert_bilstm": "hubert_bilstm",
}

_CLASSICAL_MODELS = {"random_forest", "svm", "logistic_regression"}


# ---------------------------------------------------------------------------
# Skip-if-done logic
# ---------------------------------------------------------------------------

def _artifact_path(pipeline: Dict) -> Path:
    """Return the file whose existence marks this pipeline as complete."""
    name = pipeline["name"]
    if pipeline["model"] in _CLASSICAL_MODELS:
        return Path("artifacts/models") / f"{name}.pkl"
    else:
        return Path("outputs/checkpoints") / name / "best.ckpt"


def _is_done(pipeline: Dict) -> bool:
    return _artifact_path(pipeline).exists()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(pipeline: Dict) -> int:
    cmd = [
        sys.executable, "train.py",
        f"feature_extraction={pipeline['feature_extraction']}",
        f"model={pipeline['model']}",
        f"pipeline_name={pipeline['name']}",
        f"hydra.run.dir=outputs/runs/{pipeline['name']}",
    ]
    if "preprocessing" in pipeline:
        cmd.append(f"preprocessing={pipeline['preprocessing']}")
    cmd.extend(pipeline.get("overrides", []))

    sep = "=" * 60
    print(f"\n{sep}\nPipeline : {pipeline['name']}\nArtifact : {_artifact_path(pipeline)}\n{sep}")
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _select_pipelines(models_arg: str) -> List[Dict]:
    names = {_ALIASES.get(m.strip(), m.strip()) for m in models_arg.split(",")}
    selected = [p for p in PIPELINES if p["name"] in names]
    missing = names - {p["name"] for p in selected}
    if missing:
        valid = ", ".join(sorted(_ALIASES)) + ", " + ", ".join(p["name"] for p in PIPELINES)
        print(f"[warn] Unknown model(s): {missing}\nValid options: {valid}")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predefined training pipelines, skipping completed ones."
    )
    parser.add_argument(
        "--models", default="",
        help="Comma-separated aliases to run (e.g. rf,svm,hubert_svm). Omit for all.",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated 0-based indices to skip (ignored when --models is set).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all selected pipelines even if artifacts already exist.",
    )
    args = parser.parse_args()

    # --- select pipelines ---
    if args.models:
        pipelines = _select_pipelines(args.models)
    else:
        skip_idx = {int(i) for i in args.skip.split(",") if i.strip()}
        pipelines = [p for i, p in enumerate(PIPELINES) if i not in skip_idx]

    if not pipelines:
        print("No pipelines selected. Exiting.")
        sys.exit(0)

    # --- show status and filter ---
    print(f"\n{'Pipeline':<35} {'Artifact':<45} {'Status'}")
    print("-" * 90)
    to_run: List[Dict] = []
    for p in pipelines:
        artifact = _artifact_path(p)
        done = _is_done(p) and not args.force
        status = "SKIP (done)" if done else ("FORCE" if _is_done(p) else "PENDING")
        print(f"  {p['name']:<33} {str(artifact):<45} {status}")
        if not done:
            to_run.append(p)
    print()

    if not to_run:
        print("All selected pipelines already have artifacts. Use --force to re-run.")
        sys.exit(0)

    print(f"{len(to_run)} pipeline(s) to run, {len(pipelines) - len(to_run)} skipped.\n")

    # --- run ---
    failed: List[str] = []
    for pipeline in to_run:
        rc = _run(pipeline)
        if rc != 0:
            failed.append(pipeline["name"])
            print(f"[warn] Pipeline '{pipeline['name']}' exited with code {rc}. Continuing...")

    # --- summary ---
    print(f"\n{'=' * 60}")
    if failed:
        print(f"Failed  : {failed}")
        sys.exit(1)
    else:
        print("All pipelines completed successfully.")
        print("Run `python compare.py` to rank results.")


if __name__ == "__main__":
    main()
