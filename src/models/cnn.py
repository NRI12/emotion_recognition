from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy, F1Score
from omegaconf import DictConfig


class CNNModule(pl.LightningModule):
    """Lightning module using a timm CNN backbone on single-channel spectrogram input."""

    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            num_classes=num_classes,
            in_chans=1,  # single-channel spectrogram
        )
        self.criterion = nn.CrossEntropyLoss()
        self._build_metrics(num_classes)

    def _build_metrics(self, n: int) -> None:
        kw = dict(task="multiclass", num_classes=n)
        self.train_acc = Accuracy(**kw)
        self.val_acc = Accuracy(**kw)
        self.test_acc = Accuracy(**kw)
        self.val_f1 = F1Score(**kw, average="macro")
        self.test_f1 = F1Score(**kw, average="macro")

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, logits.argmax(dim=-1), y

    def training_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, _):
        _, preds, y = self._step(batch)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test/acc", self.test_acc)
        self.log("test/f1", self.test_f1)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.get("weight_decay", 1e-4),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.training.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sched}
