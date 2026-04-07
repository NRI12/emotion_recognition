from __future__ import annotations

import numpy as np
import torch
import librosa
from omegaconf import DictConfig


class FeatureExtractor:
    """Converts a preprocessed waveform into model-ready features.

    Flat methods (mfcc, chroma, spectral_contrast) return a 1-D tensor
    of shape (n_features,) suitable for classical ML and MLP.

    2-D methods (melspec, logmel) return a tensor of shape
    (1, n_mels, n_frames) suitable for CNN models.
    """

    FLAT_METHODS = {"mfcc", "chroma", "spectral_contrast"}
    SPEC_METHODS = {"melspec", "logmel"}

    def __init__(self, feat_cfg: DictConfig, prep_cfg: DictConfig) -> None:
        self.cfg = feat_cfg
        self.method: str = feat_cfg.method
        self.sample_rate: int = prep_cfg.sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from a (1, num_samples) waveform tensor."""
        wav = waveform.squeeze().numpy().astype(np.float32)
        if self.method == "mfcc":
            return self._mfcc(wav)
        elif self.method == "melspec":
            return self._melspec(wav)
        elif self.method == "logmel":
            return self._logmel(wav)
        elif self.method == "chroma":
            return self._chroma(wav)
        elif self.method == "spectral_contrast":
            return self._spectral_contrast(wav)
        else:
            raise ValueError(f"Unknown feature method: {self.method}")

    def get_feature_dim(self) -> int:
        """Return the flat feature dimension (only valid for FLAT_METHODS)."""
        if self.method == "mfcc":
            return self.cfg.n_mfcc * 2
        elif self.method == "chroma":
            return 12 * 2
        elif self.method == "spectral_contrast":
            return 7 * 2
        else:
            raise ValueError(
                f"Method '{self.method}' produces 2-D features; "
                "use get_feature_dim() only for flat methods."
            )

    @property
    def is_flat(self) -> bool:
        return self.method in self.FLAT_METHODS

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _mfcc(self, wav: np.ndarray) -> torch.Tensor:
        mfcc = librosa.feature.mfcc(
            y=wav,
            sr=self.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.get("n_fft", 2048),
            hop_length=self.cfg.get("hop_length", 512),
        )
        features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return torch.tensor(features, dtype=torch.float32)

    def _melspec(self, wav: np.ndarray) -> torch.Tensor:
        spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.get("n_fft", 2048),
            hop_length=self.cfg.get("hop_length", 512),
            fmax=self.cfg.get("fmax", 8000),
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = self._minmax(spec)
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)

    def _logmel(self, wav: np.ndarray) -> torch.Tensor:
        spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.get("n_fft", 2048),
            hop_length=self.cfg.get("hop_length", 512),
            fmax=self.cfg.get("fmax", 8000),
        )
        log_spec = np.log(spec + 1e-9)
        log_spec = self._minmax(log_spec)
        return torch.tensor(log_spec, dtype=torch.float32).unsqueeze(0)

    def _chroma(self, wav: np.ndarray) -> torch.Tensor:
        chroma = librosa.feature.chroma_stft(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.cfg.get("n_fft", 2048),
            hop_length=self.cfg.get("hop_length", 512),
        )
        features = np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])
        return torch.tensor(features, dtype=torch.float32)

    def _spectral_contrast(self, wav: np.ndarray) -> torch.Tensor:
        contrast = librosa.feature.spectral_contrast(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.cfg.get("n_fft", 2048),
            hop_length=self.cfg.get("hop_length", 512),
        )
        features = np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])
        return torch.tensor(features, dtype=torch.float32)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo > 1e-8:
            return (arr - lo) / (hi - lo)
        return arr - lo
