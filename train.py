"""Main training entry point.

Usage examples:
    # Train with defaults (MFCC + RandomForest)
    python train.py

    # MFCC + MLP
    python train.py feature_extraction=mfcc model=mlp pipeline_name=mfcc_mlp

    # Log-mel + CNN with SpecAugment
    python train.py feature_extraction=logmel model=cnn augmentation.enabled=true

    # MFCC with delta coefficients + MLP
    python train.py feature_extraction=mfcc feature_extraction.use_delta=true model=mlp

    # Override individual params
    python train.py model=svm model.C=10.0 training.extraction_workers=24
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure a usable temp directory exists before PyTorch is imported.
_tmp = os.environ.get("TMPDIR") or "/tmp"
if not os.path.isdir(_tmp):
    _tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
    os.makedirs(_tmp, exist_ok=True)
    os.environ["TMPDIR"] = _tmp
    tempfile.tempdir = _tmp

import torch
# PyTorch 2.6 changed weights_only default to True, but Lightning checkpoints
# produced by save_hyperparameters() embed omegaconf/typing objects not in the
# safe-globals list. Patch torch.load itself so the fix covers every call site.
_orig_torch_load = torch.load
def _torch_load_unsafe(f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kw):
    return _orig_torch_load(f, map_location=map_location, pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kw)
torch.load = _torch_load_unsafe

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.data.download import ensure_dataset
from src.features.extractor import FeatureExtractor
from src.utils.logger import ExperimentLogger

_CLASSICAL = {"random_forest", "svm", "logistic_regression"}
_DL        = {"mlp", "cnn", "lstm", "bilstm"}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    ensure_dataset(cfg.data)

    extractor  = FeatureExtractor(cfg.feature_extraction, cfg.preprocessing)
    exp_logger = ExperimentLogger(cfg)
    model_type = cfg.model.type

    t0 = time.time()

    if model_type in _CLASSICAL:
        from src.pipelines.classical import ClassicalPipeline
        results = ClassicalPipeline(cfg, extractor).run()

    elif model_type in _DL:
        from src.pipelines.deep import DeepLearningPipeline
        results = DeepLearningPipeline(cfg, extractor, exp_logger).run()

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Expected one of {_CLASSICAL | _DL}"
        )

    elapsed = time.time() - t0
    exp_logger.log_run(results, train_time=elapsed)


if __name__ == "__main__":
    main()
