from __future__ import annotations

import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ExperimentLogger:
    """Logs run results to CSV and optionally Weights & Biases."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.use_wandb: bool = cfg.logging.get("use_wandb", False)
        self.run_dir: str = cfg.logging.get("run_dir", "outputs/runs")
        self.results_csv: str = os.path.join(self.run_dir, "all_results.csv")
        self._wandb_run = None

        if self.use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        try:
            import wandb

            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)

            self._wandb_run = wandb.init(
                project=self.cfg.logging.get("project", "emotion_recognition"),
                name=self.cfg.get("pipeline_name", None),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                reinit=True,
            )
        except ImportError:
            print("wandb not installed; skipping W&B logging.")
            self.use_wandb = False

    def log_run(self, results: Dict[str, Any], train_time: float) -> None:
        """Persist run metadata + metrics to the shared CSV (and W&B if enabled)."""
        from src.evaluation.metrics import save_run_results

        # Build row with explicit, fixed keys so the CSV schema never shifts.
        row: Dict[str, Any] = {
            "pipeline_name":  self.cfg.get("pipeline_name", "default"),
            "model_type":     self.cfg.model.type,
            "feature_method": self.cfg.feature_extraction.method,
            "train_time_sec": round(train_time, 2),
            "val/acc":  results.get("val/acc",  ""),
            "val/f1":   results.get("val/f1",   ""),
            "test/acc": results.get("test/acc", ""),
            "test/f1":  results.get("test/f1",  ""),
        }

        save_run_results(row, self.results_csv)
        print(f"Results saved -> {self.results_csv}")

        if self.use_wandb and self._wandb_run:
            self._wandb_run.log(row)
            self._wandb_run.finish()

    def get_lightning_logger(self):
        """Return a PL logger appropriate for the current config."""
        if self.use_wandb:
            try:
                from pytorch_lightning.loggers import WandbLogger

                return WandbLogger(
                    project=self.cfg.logging.get("project", "emotion_recognition"),
                    log_model=False,
                )
            except ImportError:
                pass

        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(
            save_dir=self.run_dir,
            name=self.cfg.get("pipeline_name", "default"),
        )
