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
    get_or_create_splits,
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

        cache_dir = self.cfg.data.get("feature_cache_dir", "data/processed")
        train_df, val_df, test_df = get_or_create_splits(
            df, self.cfg.data, self.cfg.seed, cache_dir=cache_dir
        )

        # Feature cache strategy:
        #
        # Spectral features (MFCC / logmel / melspec):
        #   → disk cache (.npy per sample) — fast reads from epoch 2 onward.
        #
        # SSL features (HuBERT / WavLM):
        #   → in-memory cache (RAM, float16) — no disk writes, extracted once
        #     then reused every epoch.  num_workers is forced to 0 so the
        #     shared dict lives in the main process (subprocess workers would
        #     each get an isolated copy, defeating the purpose).
        is_ssl = self.feature_extractor.method in self.feature_extractor.SSL_METHODS
        cache = None

        if is_ssl:
            from src.features.cache import InMemoryCache
            cache = InMemoryCache(self.feature_extractor, self.preprocessor)
            # Force single-process DataLoader: CUDA cannot be used in forked
            # subprocesses, and shared-memory dict requires main process.
            self.num_workers       = 0
            self.persistent_workers = False
            print(f"[SSL] In-memory cache enabled — num_workers forced to 0. "
                  f"HuBERT/WavLM extracts on GPU, results kept in RAM (float16).")
        elif self.cfg.data.get("use_feature_cache", True):
            from src.features.cache import FeatureCache
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
