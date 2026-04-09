from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data.dataset import (
    AudioPreprocessor,
    EmotionDataset,
    load_dataframe,
    split_dataframe,
)


class EmotionDataModule(pl.LightningDataModule):
    """Lightning DataModule: handles splits, datasets, and dataloaders.

    Args:
        cfg:               Full Hydra config.
        feature_extractor: FeatureExtractor instance.
        augment:           Optional nn.Module applied to training samples only
                           (e.g. SpecAugment). Pass None to disable.
    """

    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor,
        augment: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.preprocessor = AudioPreprocessor(cfg.preprocessing)
        self.batch_size: int = cfg.training.batch_size
        self.num_workers: int = cfg.training.get("num_workers", 0)
        self.persistent_workers: bool = (
            cfg.training.get("persistent_workers", False) and self.num_workers > 0
        )
        # populated by setup()
        self.num_classes: int = 0
        self.label_encoder = None

    def setup(self, stage: str = None) -> None:
        df, self.label_encoder = load_dataframe(self.cfg.data)
        self.num_classes = len(self.label_encoder.classes_)

        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=self.cfg.data.train_ratio,
            val_ratio=self.cfg.data.val_ratio,
            stratify=self.cfg.data.stratify,
            seed=self.cfg.seed,
        )

        # Feature cache: skip recomputation on repeated runs / across epochs.
        cache = None
        if self.cfg.data.get("use_feature_cache", True):
            from src.features.cache import FeatureCache
            cache_dir = self.cfg.data.get("feature_cache_dir", "data/processed")
            cache = FeatureCache(cache_dir, self.feature_extractor, self.preprocessor)
            print(f"Feature cache: {cache.cache_dir}")

        self.train_ds = EmotionDataset(
            train_df, self.preprocessor, self.feature_extractor, self.label_encoder,
            cache=cache, training=True, augment=self.augment,
        )
        self.val_ds = EmotionDataset(
            val_df, self.preprocessor, self.feature_extractor, self.label_encoder,
            cache=cache, training=False,
        )
        self.test_ds = EmotionDataset(
            test_df, self.preprocessor, self.feature_extractor, self.label_encoder,
            cache=cache, training=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )
