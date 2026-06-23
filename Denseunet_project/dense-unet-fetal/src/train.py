"""
train.py
========
Entry point for training / validating / testing the Dense U-Net on HC18.

Example (Lightning AI Studio terminal):

    python src/train.py \
        --data_root /teamspace/studios/this_studio/data/hc18 \
        --image_size 256 \
        --batch_size 8 \
        --max_epochs 100 \
        --gpus 1

After training finishes, the script automatically runs `trainer.test(...)`
on the held-out test split using the best checkpoint (by val_loss), and
prints a clean summary table of Dice / IoU / Precision / Recall / Accuracy
/ F1 / Loss for train, val and test.
"""

from __future__ import annotations

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from lightning_module import DenseUNetLightningModule, HC18DataModule


def parse_args():
    p = argparse.ArgumentParser(description="Train Dense U-Net on HC18")

    # Data
    p.add_argument("--data_root", type=str, required=True,
                    help="Path to HC18 root folder (containing 'training_set/').")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)

    # Model
    p.add_argument("--growth_rate", type=int, default=12)
    p.add_argument("--first_conv_channels", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--down_blocks", type=int, nargs="+", default=[4, 4, 4, 4, 4])
    p.add_argument("--up_blocks", type=int, nargs="+", default=[4, 4, 4, 4, 4])
    p.add_argument("--bottleneck_layers", type=int, default=4)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--bce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)

    # Trainer
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs (0 for CPU).")
    p.add_argument("--precision", type=str, default="16-mixed",
                    help="'32-true', '16-mixed', or 'bf16-mixed'.")
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--run_name", type=str, default="dense_unet_hc18")

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    model_config = {
        "in_channels": 1,
        "num_classes": 1,
        "first_conv_channels": args.first_conv_channels,
        "down_blocks": tuple(args.down_blocks),
        "up_blocks": tuple(args.up_blocks),
        "bottleneck_layers": args.bottleneck_layers,
        "growth_rate": args.growth_rate,
        "dropout": args.dropout,
    }

    datamodule = HC18DataModule(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    lightning_model = DenseUNetLightningModule(
        model_config=model_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="{epoch:02d}-{val_loss:.4f}-{val_Dice:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stop_patience)
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger(save_dir=args.output_dir, name="logs", version=args.run_name)
    loggers = [csv_logger]
    try:
        tb_logger = TensorBoardLogger(save_dir=args.output_dir, name="tb_logs", version=args.run_name)
        loggers.append(tb_logger)
    except ModuleNotFoundError:
        print("[warn] tensorboard not installed; continuing with CSVLogger only. "
              "Run `pip install tensorboard` to enable TensorBoard logging.")

    accelerator = "gpu" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu"
    devices = args.gpus if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision if accelerator == "gpu" else "32-true",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb],
        logger=loggers,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(lightning_model, datamodule=datamodule)

    print("\n" + "=" * 70)
    print("Training complete. Running final evaluation on all splits using")
    print("the best checkpoint (lowest val_loss) ...")
    print("=" * 70 + "\n")

    best_ckpt = checkpoint_cb.best_model_path or None

    # Re-run validation + test with the best checkpoint for the final report.
    val_results = trainer.validate(lightning_model, datamodule=datamodule, ckpt_path=best_ckpt)
    test_results = trainer.test(lightning_model, datamodule=datamodule, ckpt_path=best_ckpt)

    print_summary(val_results, test_results, best_ckpt)


def print_summary(val_results, test_results, best_ckpt):
    def fmt(results):
        if not results:
            return {}
        return {k: f"{v:.4f}" for k, v in results[0].items()}

    print(f"Best checkpoint: {best_ckpt}\n")
    print("Validation metrics:")
    for k, v in fmt(val_results).items():
        print(f"  {k:>20s}: {v}")
    print("\nTest metrics:")
    for k, v in fmt(test_results).items():
        print(f"  {k:>20s}: {v}")


if __name__ == "__main__":
    main()
