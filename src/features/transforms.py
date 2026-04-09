"""Spectrogram augmentation transforms.

SpecAugment applies frequency and time masking to (C, F, T) spectrograms
during training to improve model generalisation.

Reference: Park et al. 2019 — SpecAugment: A Simple Data Augmentation
           Method for Automatic Speech Recognition (arxiv.org/abs/1904.08779)

Usage
-----
    augment = SpecAugment.from_cfg(cfg.augmentation.spec_augment)
    x = augment(x)   # x: (C, F, T)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T
from omegaconf import DictConfig


class SpecAugment(nn.Module):
    """Apply frequency and time masking to a (C, F, T) spectrogram tensor.

    Args:
        freq_mask_param: Maximum size of each frequency mask (bins).
        time_mask_param: Maximum size of each time mask (frames).
        n_freq_masks:    Number of frequency masks applied per sample.
        n_time_masks:    Number of time masks applied per sample.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param) for _ in range(n_time_masks)]
        )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "SpecAugment":
        return cls(
            freq_mask_param=cfg.get("freq_mask_param", 15),
            time_mask_param=cfg.get("time_mask_param", 35),
            n_freq_masks=cfg.get("n_freq_masks", 2),
            n_time_masks=cfg.get("n_time_masks", 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a (C, F, T) tensor."""
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x
