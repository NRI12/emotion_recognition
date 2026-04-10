"""Run predefined pipelines sequentially, skipping completed ones.

A pipeline is considered **done** when its artifact already exists:
  - Classical ML  → artifacts/models/<name>.pkl
  - Deep learning → outputs/checkpoints/<name>/best.ckpt

Augmentation variants (--aug-mode)
------------------------------------
SpecAugment is only meaningful for temporal (2-D) features, so augmented
variants are generated automatically for CNN / LSTM / BiLSTM pipelines.
Each augmented run gets an "_aug" suffix on its name and artifact path,
and writes augmented=True to the results CSV.

  --aug-mode none  : run only base pipelines (no augmentation)  [default]
  --aug-mode aug   : run only augmented variants
  --aug-mode both  : run base AND augmented variants

Usage
-----
    python run_pipelines.py                          # all base pipelines
    python run_pipelines.py --aug-mode both          # base + aug variants
    python run_pipelines.py --aug-mode aug           # aug variants only
    python run_pipelines.py --models cnn,logmel_cnn --aug-mode both
    python run_pipelines.py --force                  # ignore existing artifacts

Available aliases (base pipelines):
    # Classical ML (MFCC only)
    rf              → mfcc_random_forest
    svm             → mfcc_svm
    lr              → mfcc_logistic_regression
    # Spectral deep learning
    mlp             → mfcc_mlp
    cnn             → melspec_cnn
    logmel_cnn      → logmel_cnn
    mfcc_lstm       → mfcc_lstm
    mfcc_bilstm     → mfcc_bilstm
    logmel_lstm     → logmel_lstm
    logmel_bilstm   → logmel_bilstm
    melspec_lstm    → melspec_lstm
    melspec_bilstm  → melspec_bilstm

Augmented aliases are the same name with "_aug" appended, e.g. "cnn_aug".
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Each entry may include:
#   name              : unique identifier → artifact path + CSV row
#   feature_extraction: Hydra config under configs/feature_extraction/
#   model             : Hydra config under configs/model/
#   preprocessing     : (optional) Hydra config under configs/preprocessing/
#   overrides         : (optional) extra "key=value" strings for train.py
# ---------------------------------------------------------------------------

PIPELINES: List[Dict] = [
    # ── Classical ML (MFCC only) ──────────────────────────────────────────
    {"name": "mfcc_random_forest",       "feature_extraction": "mfcc", "model": "random_forest"},
    {"name": "mfcc_svm",                 "feature_extraction": "mfcc", "model": "svm"},
    {"name": "mfcc_logistic_regression", "feature_extraction": "mfcc", "model": "logistic_regression"},
    # ── Spectral deep learning ────────────────────────────────────────────
    {"name": "mfcc_mlp",     "feature_extraction": "mfcc",    "model": "mlp"},
    {"name": "melspec_cnn",  "feature_extraction": "melspec",  "model": "cnn"},
    {"name": "logmel_cnn",   "feature_extraction": "logmel",   "model": "cnn"},
    # ── LSTM / BiLSTM (spectral) ──────────────────────────────────────────
    {"name": "mfcc_lstm",     "feature_extraction": "mfcc",    "model": "lstm",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "mfcc_bilstm",   "feature_extraction": "mfcc",    "model": "bilstm",
     "overrides": ["feature_extraction.output_mode=temporal"]},
    {"name": "logmel_lstm",   "feature_extraction": "logmel",  "model": "lstm"},
    {"name": "logmel_bilstm", "feature_extraction": "logmel",  "model": "bilstm"},
    {"name": "melspec_lstm",  "feature_extraction": "melspec", "model": "lstm"},
    {"name": "melspec_bilstm","feature_extraction": "melspec", "model": "bilstm"},
]

_ALIASES: Dict[str, str] = {p["name"]: p["name"] for p in PIPELINES}
_ALIASES.update({
    "rf": "mfcc_random_forest", "svm": "mfcc_svm", "lr": "mfcc_logistic_regression",
    "mlp": "mfcc_mlp", "cnn": "melspec_cnn",
})

# Models that can benefit from augmentation:
#   - cnn / lstm / bilstm : spectrogram-level + waveform-level aug
#   - mlp                 : waveform-level aug only (flat features, no spec aug)
# Classical models (RF/SVM/LR) are excluded — sklearn doesn't support on-the-fly aug.
_AUGMENTABLE_MODELS = {"cnn", "lstm", "bilstm", "mlp"}
_CLASSICAL_MODELS   = {"random_forest", "svm", "logistic_regression"}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _make_aug_variant(p: Dict) -> Dict:
    """Return a copy of pipeline p with augmentation enabled and _aug name suffix."""
    aug = dict(p)
    aug["name"] = p["name"] + "_aug"
    aug["overrides"] = list(p.get("overrides", [])) + ["augmentation.enabled=true"]
    return aug


def _build_run_list(base: List[Dict], aug_mode: str) -> List[Dict]:
    """Expand base pipeline list according to aug_mode."""
    result: List[Dict] = []
    for p in base:
        supports_aug = p["model"] in _AUGMENTABLE_MODELS
        if aug_mode in ("none", "both"):
            result.append(p)
        if aug_mode in ("aug", "both") and supports_aug:
            result.append(_make_aug_variant(p))
    return result


# ---------------------------------------------------------------------------
# Artifact / done logic
# ---------------------------------------------------------------------------

def _artifact_path(pipeline: Dict) -> Path:
    """File whose existence marks this pipeline as complete."""
    name = pipeline["name"]
    if pipeline["model"] in _CLASSICAL_MODELS:
        return Path("artifacts/models") / f"{name}.pkl"
    return Path("outputs/checkpoints") / name / "best.ckpt"


def _is_done(pipeline: Dict) -> bool:
    return _artifact_path(pipeline).exists()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(pipeline: Dict) -> int:
    name = pipeline["name"]
    cmd = [
        sys.executable, "train.py",
        f"feature_extraction={pipeline['feature_extraction']}",
        f"model={pipeline['model']}",
        f"pipeline_name={name}",
        f"hydra.run.dir=outputs/runs/{name}",
    ]
    if "preprocessing" in pipeline:
        cmd.append(f"preprocessing={pipeline['preprocessing']}")
    cmd.extend(pipeline.get("overrides", []))

    aug_tag = " [AUG]" if name.endswith("_aug") else ""
    sep = "=" * 64
    print(f"\n{sep}\nPipeline : {name}{aug_tag}\nArtifact : {_artifact_path(pipeline)}\n{sep}")
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def _resolve_names(models_arg: str, all_pipelines: List[Dict]) -> List[Dict]:
    """Resolve comma-separated aliases/names to pipeline dicts."""
    # lookup: full pipeline name → pipeline dict  (never overwritten by aliases)
    lookup = {p["name"]: p for p in all_pipelines}
    result: List[Dict] = []
    missing = []
    for token in models_arg.split(","):
        token = token.strip()
        # 1) try short alias → full name, 2) fall back to token as full name
        resolved = _ALIASES.get(token, token)
        if resolved in lookup:
            result.append(lookup[resolved])
        else:
            missing.append(token)
    if missing:
        valid = ", ".join(sorted(_ALIASES))
        print(f"[warn] Unknown alias(es): {missing}\nValid short aliases: {valid}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run emotion recognition pipelines, skipping completed ones."
    )
    parser.add_argument("--models", default="",
        help="Comma-separated pipeline aliases to run. Omit for all.")
    parser.add_argument("--skip", default="",
        help="0-based indices to skip (ignored when --models is set).")
    parser.add_argument("--force", action="store_true",
        help="Re-run even if artifacts already exist.")
    parser.add_argument(
        "--aug-mode", choices=["none", "aug", "both"], default="none",
        help=(
            "Augmentation variants to include:\n"
            "  none → base pipelines only (default)\n"
            "  aug  → augmented variants only\n"
            "  both → base + augmented variants"
        ),
    )
    args = parser.parse_args()

    # ── 1. select base pipelines ──────────────────────────────────────────
    if args.models:
        base = _resolve_names(args.models, PIPELINES)
    else:
        skip_idx = {int(i) for i in args.skip.split(",") if i.strip()}
        base = [p for i, p in enumerate(PIPELINES) if i not in skip_idx]

    if not base:
        print("No pipelines selected. Exiting.")
        sys.exit(0)

    # ── 2. expand with aug variants ───────────────────────────────────────
    pipelines = _build_run_list(base, args.aug_mode)

    # ── 3. show status table ──────────────────────────────────────────────
    col_w = (36, 48, 12)
    header = f"  {'Pipeline':<{col_w[0]}} {'Artifact':<{col_w[1]}} {'Status'}"
    print(f"\n{header}")
    print("-" * (sum(col_w) + 4))

    to_run: List[Dict] = []
    for p in pipelines:
        artifact = _artifact_path(p)
        already_done = _is_done(p)
        skip = already_done and not args.force
        if skip:
            status = "SKIP (done)"
        elif already_done:
            status = "FORCE"
        else:
            status = "PENDING"
        aug_marker = " *" if p["name"].endswith("_aug") else "  "
        print(f"{aug_marker}{p['name']:<{col_w[0]}} {str(artifact):<{col_w[1]}} {status}")
        if not skip:
            to_run.append(p)

    if args.aug_mode != "none":
        print("  (* = augmented variant)")
    print()

    if not to_run:
        print("All selected pipelines already have artifacts. Use --force to re-run.")
        sys.exit(0)

    n_aug  = sum(1 for p in to_run if p["name"].endswith("_aug"))
    n_base = len(to_run) - n_aug
    print(f"To run: {len(to_run)} pipeline(s)  ({n_base} base, {n_aug} augmented)\n")

    # ── 4. run ────────────────────────────────────────────────────────────
    failed: List[str] = []
    for pipeline in to_run:
        rc = _run(pipeline)
        if rc != 0:
            failed.append(pipeline["name"])
            print(f"[warn] '{pipeline['name']}' exited with code {rc}. Continuing...")

    # ── 5. summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    if failed:
        print(f"Failed : {failed}")
        sys.exit(1)
    else:
        print("All pipelines completed successfully.")
        print("Run `python compare.py` to rank and compare results.")


if __name__ == "__main__":
    main()
