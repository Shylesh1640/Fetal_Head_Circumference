"""
Lightning Module + DataModule
==============================
Wraps the Dense U-Net model, hybrid Dice+BCE loss, and the full metric
suite (Dice, IoU, Precision, Recall, Accuracy, F1, Loss) into a single
LightningModule, with separate metric states for train / val / test so
nothing leaks across stages or epochs.
"""

from __future__ import annotations

from typing import Any

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dense_unet import build_dense_unet
from losses import HybridDiceBCELoss
from metrics_utils import build_metric_collection
from dataset import build_datasets


class DenseUNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        scheduler_patience: int = 5,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_dense_unet(model_config)
        self.criterion = HybridDiceBCELoss(bce_weight=bce_weight, dice_weight=dice_weight)

        self.train_metrics = build_metric_collection("train_", threshold=threshold)
        self.val_metrics = build_metric_collection("val_", threshold=threshold)
        self.test_metrics = build_metric_collection("test_", threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---------------------------------------------------------------- #
    # Shared step logic
    # ---------------------------------------------------------------- #
    def _shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        probs = torch.sigmoid(logits)

        metrics = {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]
        metrics.update(probs, masks.int())

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ---------------------------------------------------------------- #
    # Epoch-end metric logging (compute + log + reset)
    # ---------------------------------------------------------------- #
    def on_train_epoch_end(self):
        self._log_and_reset(self.train_metrics)

    def on_validation_epoch_end(self):
        self._log_and_reset(self.val_metrics)

    def on_test_epoch_end(self):
        self._log_and_reset(self.test_metrics)

    def _log_and_reset(self, metric_collection):
        results = metric_collection.compute()
        self.log_dict(results, prog_bar=True, sync_dist=True)
        metric_collection.reset()

    # ---------------------------------------------------------------- #
    # Optimizer
    # ---------------------------------------------------------------- #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.hparams.scheduler_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }


class HC18DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None):
        if self.train_ds is None:
            self.train_ds, self.val_ds, self.test_ds = build_datasets(
                data_root=self.hparams.data_root,
                image_size=self.hparams.image_size,
                val_frac=self.hparams.val_frac,
                test_frac=self.hparams.test_frac,
                seed=self.hparams.seed,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
