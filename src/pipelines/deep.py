"""Deep learning training pipeline (MLP / CNN).

Orchestrates: DataModule setup → model init → Lightning Trainer → test.
SpecAugment is wired into the training DataLoader when augmentation is enabled.
"""
from __future__ import annotations

import os
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

        # Build augmentation pipeline when enabled.
        #
        # Temporal models (CNN / LSTM / BiLSTM):
        #   → spectrogram-level aug (SpecAugment, GaussianNoise, TimeShift, …)
        #     applied on-the-fly after cache load
        #   → waveform-level aug (noise, speed, pitch) applied before extraction,
        #     bypassing the feature cache
        #
        # Flat models (MLP):
        #   → spectrogram-level aug is meaningless on 1-D vectors → skipped
        #   → waveform-level aug IS meaningful: changes raw audio before MFCC
        #     extraction so the model sees different features every epoch
        #   → AugmentPipeline built with spec transforms disabled automatically
        augment = None
        aug_cfg = cfg.get("augmentation", None)
        if aug_cfg and aug_cfg.get("enabled", False):
            _WAV_KEYS  = ("waveform_noise", "speed_perturbation", "pitch_shift")
            _SPEC_KEYS = ("spec_augment", "gaussian_noise", "time_shift", "random_erasing")

            wav_enabled  = [k for k in _WAV_KEYS  if aug_cfg.get(k, {}).get("enabled", False)]
            spec_enabled = [k for k in _SPEC_KEYS if aug_cfg.get(k, {}).get("enabled", False)]

            # For flat-feature models only waveform aug makes sense
            if self.extractor.is_flat and not wav_enabled:
                print("[aug] Flat features + no waveform aug enabled → skipping augmentation.")
            else:
                # Temporarily suppress spec transforms for flat-feature models
                _aug_cfg = aug_cfg
                if self.extractor.is_flat:
                    from omegaconf import OmegaConf
                    _aug_cfg = OmegaConf.merge(
                        aug_cfg,
                        {k: {"enabled": False} for k in _SPEC_KEYS},
                    )
                from src.features.transforms import AugmentPipeline
                augment = AugmentPipeline.from_cfg(
                    _aug_cfg, sample_rate=cfg.preprocessing.sample_rate
                )
                active_spec = spec_enabled if not self.extractor.is_flat else []
                print(f"Augmentation enabled — spec: {active_spec or ['none']}  "
                      f"waveform: {wav_enabled or ['none']}")

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
        elif model_type in ("lstm", "bilstm"):
            from src.models.rnn import RNNModule
            model = RNNModule(
                cfg,
                input_size=self.extractor.get_temporal_input_size(),
                num_classes=num_classes,
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

        trainer.fit(model, datamodule)
        # train_time is measured by train.py around the entire pipeline.run() call.

        # Retrieve best checkpoint path from the ModelCheckpoint callback.
        ckpt_callback = next(
            c for c in trainer.callbacks
            if isinstance(c, pl.callbacks.ModelCheckpoint)
        )
        best_ckpt = ckpt_callback.best_model_path

        # Evaluate val and test on the best checkpoint using fresh trainer instances.
        # Using separate trainers avoids any residual state from the fit loop and
        # ensures both metric sets come from the same (best) model weights.
        _eval_kw = dict(
            accelerator="auto",
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        val_results  = pl.Trainer(**_eval_kw).validate(model, datamodule, ckpt_path=best_ckpt)
        test_results = pl.Trainer(**_eval_kw).test(model, datamodule, ckpt_path=best_ckpt)

        val_acc  = round(float(val_results[0].get("val/acc",   0.0)), 4) if val_results  else 0.0
        val_f1   = round(float(val_results[0].get("val/f1",    0.0)), 4) if val_results  else 0.0
        test_acc = round(float(test_results[0].get("test/acc", 0.0)), 4) if test_results else 0.0
        test_f1  = round(float(test_results[0].get("test/f1",  0.0)), 4) if test_results else 0.0

        print(f"val_acc={val_acc}  val_f1={val_f1}")
        print(f"test_acc={test_acc}  test_f1={test_f1}")
        return {
            "val/acc":  val_acc,
            "val/f1":   val_f1,
            "test/acc": test_acc,
            "test/f1":  test_f1,
        }
