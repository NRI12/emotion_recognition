#!/usr/bin/env bash
# Run all valid model × feature_extraction combinations sequentially.
# Classical + MLP use flat MFCC features; CNN uses 2D melspec / logmel.
set -e

run() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    shift
    python train.py "$@"
}

# ── Classical ────────────────────────────────────────────────────────
run "random_forest + mfcc"       model=random_forest  feature_extraction=mfcc    pipeline_name=rf_mfcc
run "svm + mfcc"                 model=svm            feature_extraction=mfcc    pipeline_name=svm_mfcc
run "logistic_regression + mfcc" model=logistic_regression feature_extraction=mfcc pipeline_name=lr_mfcc

# ── MLP ──────────────────────────────────────────────────────────────
run "mlp + mfcc"                 model=mlp            feature_extraction=mfcc    pipeline_name=mlp_mfcc

# ── CNN ──────────────────────────────────────────────────────────────
run "cnn + melspec"              model=cnn            feature_extraction=melspec  pipeline_name=cnn_melspec
run "cnn + logmel"               model=cnn            feature_extraction=logmel   pipeline_name=cnn_logmel

echo ""
echo "All pipelines done. Results in outputs/results.csv"
