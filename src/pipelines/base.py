from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.features.extractor import FeatureExtractor


class BasePipeline(ABC):
    """Abstract base for all training pipelines."""

    def __init__(self, cfg: DictConfig, extractor: FeatureExtractor) -> None:
        self.cfg = cfg
        self.extractor = extractor

    @abstractmethod
    def run(self) -> dict:
        """Execute the full train → val → test pipeline. Returns metrics dict."""
        ...
