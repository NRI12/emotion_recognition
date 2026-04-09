"""Unified torchaudio-based feature extractor.

Design decisions vs the old librosa extractor:
  - torchaudio transforms run as tensor ops → GPU-compatible, no Python overhead
    per sample after the first call; picklable for DataLoader/ProcessPoolExecutor.
  - Delta / delta-delta coefficients expand the channel dimension (C = 1 → 2 → 3),
    giving models velocity and acceleration information along time.
  - output_mode='flat'     → (D,) 1-D tensor for classical ML / MLP.
                             Aggregates via mean + std over the time axis.
  - output_mode='temporal' → (C, F, T) tensor for CNN.
  - normalize auto-selects:
      flat     → 'none'       (sklearn StandardScaler handles cross-sample norm)
      temporal → 'per_sample' (instance norm per channel, zero-mean unit-var)
"""
from __future__ import annotations

import torch
import torchaudio
import torchaudio.functional as AF
from omegaconf import DictConfig


class FeatureExtractor:
    """Unified torchaudio-based feature extractor.

    Supported methods
    -----------------
    mfcc              MFCC (optional Δ / ΔΔ), flat or temporal
    logmel            Log-mel spectrogram, temporal (CNN-ready)
    melspec           Mel spectrogram (dB scale), temporal (CNN-ready)
    chroma            Chroma features (librosa fallback), flat
    spectral_contrast Spectral contrast (librosa fallback), flat
    """

    FLAT_METHODS = {"mfcc", "chroma", "spectral_contrast"}
    SPEC_METHODS = {"logmel", "melspec"}

    def __init__(self, feat_cfg: DictConfig, prep_cfg: DictConfig) -> None:
        self.cfg = feat_cfg
        self.method: str = feat_cfg.method
        self.sample_rate: int = prep_cfg.sample_rate
        self.use_delta: bool = feat_cfg.get("use_delta", False)
        self.use_delta_delta: bool = feat_cfg.get("use_delta_delta", False)

        # output_mode: auto-detect from method if not explicitly set
        _default_mode = "flat" if self.method in self.FLAT_METHODS else "temporal"
        self.output_mode: str = feat_cfg.get("output_mode", _default_mode)

        # normalize: auto-detect from output_mode if not explicitly set
        _default_norm = "none" if self.output_mode == "flat" else "per_sample"
        self.normalize: str = feat_cfg.get("normalize", _default_norm)

        self._transform = self._build_transform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from a (1, T) waveform tensor.

        Returns
        -------
        flat mode     (D,)       – classical ML / MLP input
        temporal mode (C, F, T)  – CNN input
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        features = self._transform(waveform)  # (1, F, T)

        # Log compression / dB scaling
        if self.method == "logmel":
            features = torch.log(features + 1e-9)
        elif self.method == "melspec":
            features = 10.0 * torch.log10(features.clamp(min=1e-10))

        # Expand channel dimension with delta / delta-delta
        if self.use_delta:
            delta = AF.compute_deltas(features)           # (1, F, T)
            if self.use_delta_delta:
                delta2 = AF.compute_deltas(delta)
                features = torch.cat([features, delta, delta2], dim=0)  # (3, F, T)
            else:
                features = torch.cat([features, delta], dim=0)          # (2, F, T)

        if self.normalize == "per_sample":
            features = _instance_norm(features)

        if self.output_mode == "flat":
            return _flatten(features)

        return features  # (C, F, T)

    def get_feature_dim(self) -> int:
        """Flat output dimension. Valid only when output_mode='flat'."""
        if self.output_mode != "flat":
            raise ValueError(
                f"output_mode='{self.output_mode}' — "
                "get_feature_dim() is only valid for output_mode='flat'"
            )
        C = self.get_output_channels()
        if self.method == "mfcc":
            return C * self.cfg.n_mfcc * 2
        elif self.method in ("logmel", "melspec"):
            return C * self.cfg.n_mels * 2
        elif self.method == "chroma":
            return 12 * 2          # 12 chroma bins; delta not supported for librosa methods
        elif self.method == "spectral_contrast":
            return 7 * 2           # 7 sub-bands
        raise ValueError(f"Unknown method: {self.method!r}")

    def get_output_channels(self) -> int:
        """Number of channels C in the (C, F, T) tensor representation."""
        return 1 + int(self.use_delta) + int(self.use_delta and self.use_delta_delta)

    @property
    def is_flat(self) -> bool:
        return self.output_mode == "flat"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_transform(self):
        n_fft = self.cfg.get("n_fft", 2048)
        hop_length = self.cfg.get("hop_length", 512)

        if self.method == "mfcc":
            return torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.cfg.n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": self.cfg.get("n_mels", 128),
                    "f_max": self.cfg.get("fmax", 8000.0),
                },
            )
        elif self.method in ("logmel", "melspec"):
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.cfg.n_mels,
                f_max=self.cfg.get("fmax", 8000.0),
            )
        elif self.method == "chroma":
            return _LibrosaTransform("chroma", self.sample_rate, n_fft, hop_length)
        elif self.method == "spectral_contrast":
            return _LibrosaTransform("spectral_contrast", self.sample_rate, n_fft, hop_length)
        raise ValueError(f"Unknown feature method: {self.method!r}")


# ---------------------------------------------------------------------------
# Module-level helpers (no self → picklable, safe in ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _instance_norm(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-variance per channel (statistics over F × T dims)."""
    mu = x.mean(dim=(-2, -1), keepdim=True)      # (C, 1, 1)
    sigma = x.std(dim=(-2, -1), keepdim=True)     # (C, 1, 1)
    return (x - mu) / (sigma + 1e-8)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Aggregate time axis via mean + std → 1-D tensor.

    Input  (C, F, T)
    Output (2 · C · F,)  layout: [all_means ..., all_stds ...]
    """
    mean = x.mean(dim=-1)   # (C, F)
    std  = x.std(dim=-1)    # (C, F)
    return torch.cat([mean.flatten(), std.flatten()])


class _LibrosaTransform:
    """Callable wrapper for librosa methods without a torchaudio equivalent."""

    def __init__(self, method: str, sr: int, n_fft: int, hop_length: int) -> None:
        self._method = method
        self._sr = sr
        self._n_fft = n_fft
        self._hop_length = hop_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        import librosa
        wav = waveform.squeeze().numpy().astype("float32")
        if self._method == "chroma":
            feat = librosa.feature.chroma_stft(
                y=wav, sr=self._sr, n_fft=self._n_fft, hop_length=self._hop_length
            )
        else:
            feat = librosa.feature.spectral_contrast(
                y=wav, sr=self._sr, n_fft=self._n_fft, hop_length=self._hop_length
            )
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
