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

SSL methods (hubert, wavlm):
  - Uses HuggingFace transformers; models are downloaded on first use.
  - Input is resampled to 16 kHz internally (models' native rate).
  - Output shape: (1, hidden_size, T)  — matches (C, F, T) convention.
  - flat mode aggregates over T via mean + std → (2 * hidden_size,).
  - Delta / delta-delta are not applied to SSL embeddings.
  - Model loading is lazy and thread-safe (one load per extractor instance).
"""
from __future__ import annotations

import threading

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
    hubert            HuBERT embeddings (HuggingFace), flat or temporal
    wavlm             WavLM embeddings (HuggingFace), flat or temporal
    """

    FLAT_METHODS = {"mfcc", "chroma", "spectral_contrast"}
    SPEC_METHODS = {"logmel", "melspec"}
    SSL_METHODS  = {"hubert", "wavlm"}

    def __init__(self, feat_cfg: DictConfig, prep_cfg: DictConfig) -> None:
        self.cfg = feat_cfg
        self.method: str = feat_cfg.method
        self.sample_rate: int = prep_cfg.sample_rate

        # Delta / ΔΔ not applicable to SSL embeddings
        self.use_delta: bool = feat_cfg.get("use_delta", False) if self.method not in self.SSL_METHODS else False
        self.use_delta_delta: bool = feat_cfg.get("use_delta_delta", False) if self.method not in self.SSL_METHODS else False

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

        # Log compression / dB scaling (spectral methods only)
        if self.method == "logmel":
            features = torch.log(features + 1e-9)
        elif self.method == "melspec":
            features = 10.0 * torch.log10(features.clamp(min=1e-10))

        # Expand channel dimension with delta / delta-delta (not for SSL)
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
        elif self.method in self.SSL_METHODS:
            return self.cfg.hidden_size * 2  # mean + std over time axis
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
        elif self.method in self.SSL_METHODS:
            return _SSLTransform(
                method=self.method,
                model_name=self.cfg.model_name,
                layer=self.cfg.get("layer", -1),
                input_sr=self.sample_rate,
                device=self.cfg.get("device", "cpu"),
            )
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


class _SSLTransform:
    """HuBERT / WavLM feature extractor via HuggingFace transformers.

    Resamples input to 16 kHz (models' native rate) if needed.
    Returns (1, hidden_size, T) — compatible with the (C, F, T) convention.

    Model and processor are loaded lazily on the first call and cached on the
    instance, so each FeatureExtractor pays the loading cost only once.
    Thread-safe: a lock guards the lazy-load critical section.

    Parameters
    ----------
    method     : "hubert" or "wavlm"
    model_name : HuggingFace model ID, e.g. "facebook/hubert-base-ls960"
    layer      : transformer layer index to extract from.
                 -1 → last_hidden_state (fastest, no all-layer storage).
                 0 → embedding output, 1..N → transformer layer outputs.
    input_sr   : sample rate of the incoming waveform (from AudioPreprocessor)
    device     : torch device string, e.g. "cpu" or "cuda"
    """

    TARGET_SR = 16_000

    def __init__(
        self,
        method: str,
        model_name: str,
        layer: int,
        input_sr: int,
        device: str = "cpu",
    ) -> None:
        self._method = method
        self._model_name = model_name
        self._layer = layer
        self._device = device
        self._resampler = (
            torchaudio.transforms.Resample(input_sr, self.TARGET_SR)
            if input_sr != self.TARGET_SR
            else None
        )
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy loading (thread-safe double-checked locking)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoFeatureExtractor, AutoModel
            self._processor = AutoFeatureExtractor.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract SSL embeddings from a (1, N) waveform tensor.

        Returns
        -------
        torch.Tensor of shape (1, hidden_size, T) on CPU.
        """
        self._load()

        if self._resampler is not None:
            waveform = self._resampler(waveform)

        wav_np = waveform.squeeze(0).numpy()  # (N,)

        inputs = self._processor(
            wav_np,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        need_all = self._layer != -1
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=need_all)

        if self._layer == -1:
            hidden = outputs.last_hidden_state   # (1, T, H)
        else:
            hidden = outputs.hidden_states[self._layer]  # (1, T, H)

        # (1, T, H) → (1, H, T) to match (C, F, T) convention
        return hidden.squeeze(0).T.unsqueeze(0).cpu()  # (1, H, T)
