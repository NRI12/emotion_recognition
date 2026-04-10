from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
    supplied FeatureExtractor on-the-fly, or loaded from a FeatureCache
    when one is provided (avoids recomputation across epochs and runs).

    When training=True and augment is provided, the augmentation transform
    is applied after loading features (cached features are pre-augmentation,
    so augmentation varies every epoch).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: AudioPreprocessor,
        feature_extractor,
        label_encoder: LabelEncoder,
        cache=None,           # Optional[FeatureCache]
        training: bool = False,
        augment=None,         # Optional[nn.Module] — applied only when training=True
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self._cache = cache
        self.training = training
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(self.label_encoder.transform([row["emotion"]])[0])

        # Waveform augmentation requires extracting features on-the-fly
        # (bypassing the cache) because the waveform is modified before extraction.
        use_waveform_aug = (
            self.training
            and self.augment is not None
            and getattr(self.augment, "has_waveform_aug", False)
        )

        try:
            if use_waveform_aug:
                # Bypass cache: load waveform → augment → extract
                waveform = self.preprocessor.load(row["path"])
                waveform = self.augment.augment_waveform(waveform)
                features = self.feature_extractor.extract(waveform)
            elif self._cache is not None:
                features = self._cache.get(row["path"])
            else:
                waveform = self.preprocessor.load(row["path"])
                features = self.feature_extractor.extract(waveform)
        except Exception as exc:  # noqa: BLE001
            # Corrupt / missing audio — return a zero tensor so the DataLoader
            # batch stays intact.  One bad sample has negligible impact on training.
            print(f"[warn] zero-pad bad sample [{idx}] {row['path']}: {exc}")
            ref = self.df.iloc[0]
            try:
                waveform = self.preprocessor.load(ref["path"])
                features = torch.zeros_like(self.feature_extractor.extract(waveform))
            except Exception:
                features = torch.zeros(1)

        # Spectrogram augmentation applied on-the-fly after features are ready —
        # only during training and only for 2-D tensors (C, F, T).
        if self.training and self.augment is not None and features.ndim == 3:
            features = self.augment(features)
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


# ---------------------------------------------------------------------------
# Split caching (R5) — ensures identical splits across all callers
# ---------------------------------------------------------------------------

def _csv_fingerprint(csv_path: str) -> str:
    """MD5 of the CSV file content (first 12 hex chars)."""
    h = hashlib.md5()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def get_or_create_splits(
    df: pd.DataFrame,
    data_cfg: DictConfig,
    seed: int,
    cache_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with deterministic caching.

    The cache key is a hash of: CSV content fingerprint + split ratios + seed.
    If a matching cache file exists the same index lists are reused, guaranteeing
    identical splits across train / eval / tune runs that share the same config.
    """
    key_data = {
        "csv_fingerprint": _csv_fingerprint(data_cfg.csv_path),
        "train_ratio":     float(data_cfg.train_ratio),
        "val_ratio":       float(data_cfg.val_ratio),
        "stratify":        bool(data_cfg.stratify),
        "seed":            int(seed),
    }
    key_hash = hashlib.md5(
        json.dumps(key_data, sort_keys=True).encode()
    ).hexdigest()[:8]

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir_path / f"splits_{key_hash}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        train_df = df.loc[cached["train"]].reset_index(drop=True)
        val_df   = df.loc[cached["val"]].reset_index(drop=True)
        test_df  = df.loc[cached["test"]].reset_index(drop=True)
        print(f"Split cache loaded  [{cache_path.name}]"
              f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
        return train_df, val_df, test_df

    # Compute split and persist index lists so every future caller gets the same rows.
    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        stratify=data_cfg.stratify,
        seed=seed,
    )
    payload = {
        "train": train_df.index.tolist(),
        "val":   val_df.index.tolist(),
        "test":  test_df.index.tolist(),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Split cache saved   [{cache_path.name}]"
          f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    # reset_index so downstream code gets a clean 0-based index
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
