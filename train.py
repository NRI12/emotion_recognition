"""Main training entry point.

Usage examples:
    # Train with defaults (MFCC + RandomForest)
    python train.py

    # MFCC + MLP
    python train.py feature_extraction=mfcc model=mlp pipeline_name=mfcc_mlp

    # MelSpec + CNN
    python train.py feature_extraction=melspec model=cnn pipeline_name=melspec_cnn

    # Override individual params
    python train.py model=svm model.C=10.0
"""
from __future__ import annotations

import os
import time

# Windows: prevent OpenMP duplicate-library crash when numpy and torch
# both ship their own libiomp5md.dll.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
# PyTorch 2.6 changed weights_only default to True, but Lightning checkpoints
# produced by save_hyperparameters() embed omegaconf/typing objects not in the
# safe-globals list. Patch torch.load itself so the fix covers every call site
# regardless of how Lightning imports it internally.
_orig_torch_load = torch.load
def _torch_load_unsafe(f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kw):
    return _orig_torch_load(f, map_location=map_location, pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kw)
torch.load = _torch_load_unsafe

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.data.datamodule import EmotionDataModule
from src.data.download import ensure_dataset
from src.evaluation.metrics import save_run_results
from src.features.extractor import FeatureExtractor
from src.models.classical import train_classical_model
from src.models.cnn import CNNModule
from src.models.mlp import MLPModule
from src.utils.logger import ExperimentLogger

_CLASSICAL = {"random_forest", "svm", "logistic_regression"}
_DL = {"mlp", "cnn"}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    # Ensure dataset exists before any ML code runs.
    # Downloads automatically when data.auto_download=true and csv_path is missing.
    ensure_dataset(cfg.data)

    exp_logger = ExperimentLogger(cfg)
    extractor = FeatureExtractor(cfg.feature_extraction, cfg.preprocessing)
    model_type: str = cfg.model.type

    t0 = time.time()

    # ------------------------------------------------------------------
    # Classical ML path
    # ------------------------------------------------------------------
    if model_type in _CLASSICAL:
        results = train_classical_model(cfg, extractor)
        elapsed = time.time() - t0
        exp_logger.log_run(results, train_time=elapsed)
        return

    # ------------------------------------------------------------------
    # Deep learning path
    # ------------------------------------------------------------------
    if model_type not in _DL:
        raise ValueError(f"Unknown model type: '{model_type}'. Expected one of {_CLASSICAL | _DL}")

    datamodule = EmotionDataModule(cfg, extractor)
    datamodule.setup()
    num_classes = datamodule.num_classes

    if model_type == "mlp":
        model = MLPModule(cfg, input_dim=extractor.get_feature_dim(), num_classes=num_classes)
    else:  # cnn
        model = CNNModule(cfg, num_classes=num_classes)

    ckpt_dir = os.path.join("outputs", "checkpoints", cfg.get("pipeline_name", model_type))
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=exp_logger.get_lightning_logger(),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val/f1",
                patience=cfg.training.get("patience", 10),
                mode="max",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val/f1",
                mode="max",
                save_top_k=1,
                dirpath=ckpt_dir,
                filename="best",
            ),
        ],
        deterministic=True,
        accelerator="auto",
    )

    trainer.fit(model, datamodule)
    test_results = trainer.test(model, datamodule, ckpt_path="best")
    results = test_results[0] if test_results else {}

    elapsed = time.time() - t0
    exp_logger.log_run(results, train_time=elapsed)


if __name__ == "__main__":
    main()
