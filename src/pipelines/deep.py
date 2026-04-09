"""Deep learning training pipeline (MLP / CNN).

Orchestrates: DataModule setup → model init → Lightning Trainer → test.
SpecAugment is wired into the training DataLoader when augmentation is enabled.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from src.data.datamodule import EmotionDataModule
from src.pipelines.base import BasePipeline
from src.utils.logger import ExperimentLogger


class DeepLearningPipeline(BasePipeline):
    """Lightning-based training pipeline for MLP and CNN models."""

    def __init__(
        self,
        cfg: DictConfig,
        extractor,
        exp_logger: Optional[ExperimentLogger] = None,
    ) -> None:
        super().__init__(cfg, extractor)
        self.exp_logger = exp_logger

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        # Build SpecAugment if enabled (CNN / temporal features only)
        augment = None
        aug_cfg = cfg.get("augmentation", None)
        if aug_cfg and aug_cfg.get("enabled", False) and not self.extractor.is_flat:
            from src.features.transforms import SpecAugment
            augment = SpecAugment.from_cfg(aug_cfg.spec_augment)
            print("SpecAugment enabled.")

        datamodule = EmotionDataModule(cfg, self.extractor, augment=augment)
        datamodule.setup()

        model_type: str = cfg.model.type
        num_classes: int = datamodule.num_classes

        if model_type == "mlp":
            from src.models.mlp import MLPModule
            model = MLPModule(
                cfg,
                input_dim=self.extractor.get_feature_dim(),
                num_classes=num_classes,
            )
        elif model_type == "cnn":
            from src.models.cnn import CNNModule
            model = CNNModule(
                cfg,
                num_classes=num_classes,
                in_chans=self.extractor.get_output_channels(),
            )
        else:
            raise ValueError(f"Unknown DL model type: {model_type!r}")

        lightning_logger = (
            self.exp_logger.get_lightning_logger() if self.exp_logger else True
        )
        ckpt_dir = os.path.join(
            "outputs", "checkpoints", cfg.get("pipeline_name", model_type)
        )

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            logger=lightning_logger,
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

        t0 = time.time()
        trainer.fit(model, datamodule)
        test_results = trainer.test(model, datamodule, ckpt_path="best")
        train_time = time.time() - t0

        results = test_results[0] if test_results else {}
        results["train_time"] = train_time
        return results
