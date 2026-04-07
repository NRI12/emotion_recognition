from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from omegaconf import DictConfig


class AudioPreprocessor:
    """Loads and preprocesses raw audio: resample, normalize, fixed-length trim/pad.

    Uses soundfile for loading (works on all platforms without extra codec deps)
    and torchaudio.transforms.Resample for high-quality resampling.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.sample_rate: int = cfg.sample_rate
        self.duration: float = cfg.duration
        self.normalize: bool = cfg.normalize
        self.num_samples: int = int(self.sample_rate * self.duration)

    def load(self, path: str) -> torch.Tensor:
        wav_np, sr = sf.read(path, always_2d=False, dtype="float32")
        # soundfile: mono -> (N,), stereo -> (N, C)
        waveform = torch.from_numpy(wav_np)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # (1, N)
        else:
            waveform = waveform.T              # (C, N)
        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample
        if sr != self.sample_rate:
            waveform = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        # RMS normalize
        if self.normalize:
            rms = waveform.pow(2).mean().sqrt()
            if rms > 1e-8:
                waveform = waveform / (rms + 1e-8)
        # Trim or pad to fixed length
        waveform = self._fix_length(waveform)
        return waveform  # (1, num_samples)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        if n > self.num_samples:
            waveform = waveform[..., : self.num_samples]
        elif n < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - n))
        return waveform


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion recognition.

    Each item is (features, label) where features are produced by the
    supplied FeatureExtractor on-the-fly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: AudioPreprocessor,
        feature_extractor,
        label_encoder: LabelEncoder,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(self.label_encoder.transform([row["emotion"]])[0])
        waveform = self.preprocessor.load(row["path"])
        features = self.feature_extractor.extract(waveform)
        return features, label


def load_dataframe(cfg: DictConfig) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load CSV, resolve audio paths, and fit a LabelEncoder."""
    df = pd.read_csv(cfg.csv_path)
    assert "path" in df.columns, "CSV must contain a 'path' column"
    assert "emotion" in df.columns, "CSV must contain an 'emotion' column"

    # Resolve audio paths according to path_mode
    path_mode: str = cfg.get("path_mode", "relative")
    audio_dir: str = cfg.get("audio_dir", "")

    if path_mode == "basename":
        # Extract filename only and join with audio_dir
        df["path"] = df["path"].apply(
            lambda p: os.path.join(audio_dir, os.path.basename(p))
        )
    elif path_mode == "relative" and audio_dir:
        # Prepend audio_dir to relative paths
        df["path"] = df["path"].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(audio_dir, p)
        )
    # path_mode == "full": use paths as-is

    le = LabelEncoder()
    le.fit(df["emotion"])
    return df, le


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    stratify: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train / val / test subsets."""
    strat = df["emotion"] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        stratify=strat,
        random_state=seed,
    )
    val_size = val_ratio / (1.0 - train_ratio)
    strat_temp = temp_df["emotion"] if stratify else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_size,
        stratify=strat_temp,
        random_state=seed,
    )
    return train_df, val_df, test_df
