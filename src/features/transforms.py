"""Spectrogram and waveform augmentation transforms.

All spectrogram augmentations operate on (C, F, T) tensors and are applied
on-the-fly AFTER the feature cache is loaded — the cache itself is never
modified.  This means every epoch sees a different augmented view of each
cached spectrogram.

Waveform augmentations (GaussianNoiseWaveform, SpeedPerturbation,
PitchShift) operate on raw (1, N) waveforms BEFORE feature extraction.
Because they change the signal at the waveform level they BYPASS the feature
cache and require on-the-fly extraction, which is slower but gives richer
diversity.  See EmotionDataset for how the bypass is wired.

Public API
----------
    # Spectrogram-level
    SpecAugment          — frequency + time masking (Park et al. 2019)
    GaussianNoise        — additive Gaussian noise on spectrogram
    TimeShift            — cyclic shift along the time axis
    RandomErasing        — zero out a random rectangular patch
    Mixup                — blends two samples; wired at the batch level in
                           the Lightning module (not used here directly)

    # Waveform-level
    GaussianNoiseWaveform — additive white noise on raw audio
    SpeedPerturbation     — resample to random speed (stretches / squeezes)
    PitchShift            — raise / lower pitch without changing tempo

    # Composite
    AugmentPipeline      — chains any subset of the above from config
"""
from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig


# ===========================================================================
# Spectrogram-level augmentations  (C, F, T) → (C, F, T)
# ===========================================================================

class SpecAugment(nn.Module):
    """Frequency and time masking (Park et al. 2019, arxiv 1904.08779).

    Args:
        freq_mask_param: max frequency-bin mask width
        time_mask_param: max time-frame mask width
        n_freq_masks:    number of frequency masks
        n_time_masks:    number of time masks
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
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x


class GaussianNoise(nn.Module):
    """Add Gaussian noise to a spectrogram — simulates microphone noise.

    Args:
        std: standard deviation of the noise relative to the signal RMS.
             A small value (0.01–0.05) blends naturally with the signal.
    """

    def __init__(self, std: float = 0.02) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class TimeShift(nn.Module):
    """Cyclically shift the spectrogram along the time axis.

    A random number of frames in [−max_shift, +max_shift] are rolled;
    the wrapped-around portion appears at the opposite end.
    This is equivalent to randomly shifting the start point of the audio.

    Args:
        max_fraction: maximum shift as a fraction of total time frames.
                      E.g. 0.1 allows shifting up to 10 % of the sequence.
    """

    def __init__(self, max_fraction: float = 0.1) -> None:
        super().__init__()
        self.max_fraction = max_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_len = x.shape[-1]
        max_shift = max(1, int(T_len * self.max_fraction))
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(x, shifts=shift, dims=-1)


class RandomErasing(nn.Module):
    """Zero out a random rectangular region of the spectrogram.

    Similar to Cutout (DeVries & Taylor 2017) adapted for spectrograms.
    The erased region has random position, width (time), and height (freq).

    Args:
        max_freq_fraction: max height of erased rectangle as fraction of F.
        max_time_fraction: max width of erased rectangle as fraction of T.
        p:                 probability of applying the transform per sample.
    """

    def __init__(
        self,
        max_freq_fraction: float = 0.2,
        max_time_fraction: float = 0.2,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_freq_fraction = max_freq_fraction
        self.max_time_fraction = max_time_fraction
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        _, F, T_len = x.shape
        h = random.randint(1, max(1, int(F * self.max_freq_fraction)))
        w = random.randint(1, max(1, int(T_len * self.max_time_fraction)))
        f0 = random.randint(0, F - h)
        t0 = random.randint(0, T_len - w)
        x = x.clone()
        x[:, f0:f0 + h, t0:t0 + w] = 0.0
        return x


# ===========================================================================
# Waveform-level augmentations  (1, N) → (1, N)
# These bypass the feature cache — see EmotionDataset.
# ===========================================================================

class GaussianNoiseWaveform(nn.Module):
    """Add white Gaussian noise to the raw waveform.

    Args:
        min_snr_db: minimum signal-to-noise ratio in dB (noisier).
        max_snr_db: maximum signal-to-noise ratio in dB (cleaner).
        p:          probability of applying per sample.
    """

    def __init__(
        self, min_snr_db: float = 10.0, max_snr_db: float = 40.0, p: float = 0.5
    ) -> None:
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / (snr_linear + 1e-9)
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise


class SpeedPerturbation(nn.Module):
    """Random speed perturbation via resampling.

    Stretches or compresses the audio by resampling at a rate slightly
    above or below the original, then clips/pads back to the original
    length.  Equivalent to changing speaking rate while keeping pitch.

    Args:
        min_rate: lower bound of the speed factor (e.g. 0.9 = 10 % slower).
        max_rate: upper bound of the speed factor (e.g. 1.1 = 10 % faster).
        sample_rate: original sample rate (used for resampler).
        p: probability of applying per sample.
    """

    def __init__(
        self,
        min_rate: float = 0.9,
        max_rate: float = 1.1,
        sample_rate: int = 16000,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.sample_rate = sample_rate
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        rate = random.uniform(self.min_rate, self.max_rate)
        orig_len = waveform.shape[-1]
        new_sr = int(self.sample_rate * rate)
        resampled = torchaudio.functional.resample(waveform, new_sr, self.sample_rate)
        # Clip or zero-pad back to the original length
        n = resampled.shape[-1]
        if n >= orig_len:
            return resampled[..., :orig_len]
        return torch.nn.functional.pad(resampled, (0, orig_len - n))


class PitchShift(nn.Module):
    """Randomly shift pitch by ±n semitones without changing tempo.

    Uses torchaudio.functional.pitch_shift (requires SoX effects chain /
    librosa backend).  Falls back to identity if the backend is unavailable.

    Args:
        max_semitones: maximum pitch shift in semitones (applied ± randomly).
        sample_rate:   audio sample rate.
        p:             probability of applying per sample.
    """

    def __init__(
        self, max_semitones: float = 2.0, sample_rate: int = 16000, p: float = 0.5
    ) -> None:
        super().__init__()
        self.max_semitones = max_semitones
        self.sample_rate = sample_rate
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return waveform
        n_steps = random.uniform(-self.max_semitones, self.max_semitones)
        try:
            return torchaudio.functional.pitch_shift(
                waveform, self.sample_rate, n_steps=n_steps
            )
        except Exception:
            return waveform  # graceful fallback


# ===========================================================================
# Composite pipeline
# ===========================================================================

class AugmentPipeline(nn.Module):
    """Chain multiple augmentation modules in sequence.

    Built from Hydra config so every combination can be enabled/disabled
    without code changes.

    Spectrogram transforms are applied as a chain on (C, F, T) tensors.
    Waveform transforms are separate (accessed via .waveform_aug) and
    applied BEFORE feature extraction in EmotionDataset.

    Config layout (see configs/augmentation/default.yaml):
        spec_augment.enabled
        gaussian_noise.enabled
        time_shift.enabled
        random_erasing.enabled
        waveform_noise.enabled
        speed_perturbation.enabled
        pitch_shift.enabled
    """

    def __init__(
        self,
        spec_transforms: Optional[List[nn.Module]] = None,
        waveform_transforms: Optional[List[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.spec_aug = nn.Sequential(*spec_transforms) if spec_transforms else None
        # Store waveform transforms as a plain list (they accept tensors directly)
        self._waveform_transforms: List[nn.Module] = waveform_transforms or []

    # ------------------------------------------------------------------
    # Spectrogram forward — called from EmotionDataset after cache load
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectrogram augmentations. x: (C, F, T)"""
        if self.spec_aug is not None:
            x = self.spec_aug(x)
        return x

    # ------------------------------------------------------------------
    # Waveform augmentation — called from EmotionDataset before extraction
    # ------------------------------------------------------------------

    @property
    def has_waveform_aug(self) -> bool:
        return len(self._waveform_transforms) > 0

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply waveform augmentations. waveform: (1, N)"""
        for transform in self._waveform_transforms:
            waveform = transform(waveform)
        return waveform

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_cfg(cls, aug_cfg: DictConfig, sample_rate: int) -> "AugmentPipeline":
        """Build pipeline from augmentation config block."""
        spec_transforms: List[nn.Module] = []
        waveform_transforms: List[nn.Module] = []

        # ── Spectrogram augmentations ─────────────────────────────────
        sa_cfg = aug_cfg.get("spec_augment", {})
        if sa_cfg.get("enabled", True):
            spec_transforms.append(SpecAugment.from_cfg(sa_cfg))

        gn_cfg = aug_cfg.get("gaussian_noise", {})
        if gn_cfg.get("enabled", False):
            spec_transforms.append(GaussianNoise(std=gn_cfg.get("std", 0.02)))

        ts_cfg = aug_cfg.get("time_shift", {})
        if ts_cfg.get("enabled", False):
            spec_transforms.append(TimeShift(max_fraction=ts_cfg.get("max_fraction", 0.1)))

        re_cfg = aug_cfg.get("random_erasing", {})
        if re_cfg.get("enabled", False):
            spec_transforms.append(RandomErasing(
                max_freq_fraction=re_cfg.get("max_freq_fraction", 0.2),
                max_time_fraction=re_cfg.get("max_time_fraction", 0.2),
                p=re_cfg.get("p", 0.5),
            ))

        # ── Waveform augmentations ────────────────────────────────────
        wn_cfg = aug_cfg.get("waveform_noise", {})
        if wn_cfg.get("enabled", False):
            waveform_transforms.append(GaussianNoiseWaveform(
                min_snr_db=wn_cfg.get("min_snr_db", 10.0),
                max_snr_db=wn_cfg.get("max_snr_db", 40.0),
                p=wn_cfg.get("p", 0.5),
            ))

        sp_cfg = aug_cfg.get("speed_perturbation", {})
        if sp_cfg.get("enabled", False):
            waveform_transforms.append(SpeedPerturbation(
                min_rate=sp_cfg.get("min_rate", 0.9),
                max_rate=sp_cfg.get("max_rate", 1.1),
                sample_rate=sample_rate,
                p=sp_cfg.get("p", 0.5),
            ))

        ps_cfg = aug_cfg.get("pitch_shift", {})
        if ps_cfg.get("enabled", False):
            waveform_transforms.append(PitchShift(
                max_semitones=ps_cfg.get("max_semitones", 2.0),
                sample_rate=sample_rate,
                p=ps_cfg.get("p", 0.5),
            ))

        return cls(spec_transforms, waveform_transforms)
