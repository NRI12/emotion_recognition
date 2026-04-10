"""LSTM and BiLSTM Lightning modules for emotion recognition.

Input convention
----------------
Accepts temporal tensors of shape (batch, C, F, T) — the same format produced
by FeatureExtractor in output_mode='temporal' and cached by FeatureCache.

Inside forward() the tensor is reshaped to (batch, T, C*F) so the RNN sees
one feature vector per time step.  For SSL features (HuBERT / WavLM) this
becomes (batch, T, hidden_size), which is the most natural use case.

Architecture
------------
    Input (batch, T, input_size)
        │
    [optional LayerNorm]
        │
    LSTM (num_layers, hidden_size, bidirectional)
        │
    Last hidden state — (batch, hidden_size * directions)
        │
    Dropout
        │
    Linear → num_classes

The last hidden state is taken from the final LSTM layer.  For BiLSTM the
forward and backward states are concatenated, giving 2 × hidden_size features.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Core encoder (shared by LSTM and BiLSTM modules)
# ---------------------------------------------------------------------------

class _LSTMEncoder(nn.Module):
    """Multi-layer LSTM with optional input LayerNorm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.input_norm = nn.LayerNorm(input_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, input_size)  →  (batch, out_size)"""
        x = self.input_norm(x)
        _, (h_n, _) = self.rnn(x)
        # h_n: (num_layers * directions, batch, hidden_size)
        if self.bidirectional:
            # concat last forward state and last backward state
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*H)
        else:
            h = h_n[-1]                                 # (batch, H)
        return h


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class RNNModule(pl.LightningModule):
    """Lightning module wrapping an LSTM or BiLSTM encoder.

    Parameters
    ----------
    cfg        : full Hydra config (accesses cfg.model.* and cfg.training.*)
    input_size : C × F — flattened features per time step
                 (from FeatureExtractor.get_temporal_input_size())
    num_classes: number of emotion classes
    """

    def __init__(self, cfg: DictConfig, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        bidirectional: bool = cfg.model.get("bidirectional", False)
        self.encoder = _LSTMEncoder(
            input_size=input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.model.dropout),
            nn.Linear(self.encoder.out_size, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self._build_metrics(num_classes)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _build_metrics(self, n: int) -> None:
        kw = dict(task="multiclass", num_classes=n)
        self.train_acc = Accuracy(**kw)
        self.val_acc   = Accuracy(**kw)
        self.test_acc  = Accuracy(**kw)
        self.val_f1    = F1Score(**kw, average="macro")
        self.test_f1   = F1Score(**kw, average="macro")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, C, F, T)  →  logits (batch, num_classes)"""
        B, C, F, T = x.shape
        # (batch, C, F, T) → (batch, T, C*F)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        return self.head(self.encoder(x))

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, logits.argmax(dim=-1), y

    def training_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  self.val_acc, prog_bar=True)
        self.log("val/f1",   self.val_f1,  prog_bar=True)

    def test_step(self, batch, _):
        _, preds, y = self._step(batch)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test/acc", self.test_acc)
        self.log("test/f1",  self.test_f1)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.get("weight_decay", 1e-4),
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", patience=5, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/f1"}}
