"""
model.py
Builds the proposed architecture: U-Net encoder-decoder with a MiT-B2
(Mix Vision Transformer / SegFormer) encoder, pretrained on ImageNet.

Uses the `segmentation-models-pytorch` (SMP) library, which provides
ready-made, well-tested U-Net decoders paired with 800+ encoders including
the full SegFormer "mit_b0..b5" family (requires `timm`).

Reference: https://github.com/qubvel-org/segmentation_models.pytorch
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config as cfg


def build_model(
    encoder_name: str = cfg.ENCODER_NAME,
    encoder_weights: str = cfg.ENCODER_WEIGHTS,
    in_channels: int = cfg.IN_CHANNELS,
    classes: int = cfg.NUM_CLASSES,
) -> nn.Module:
    """Returns a U-Net (SMP) with the given encoder. Output is RAW LOGITS
    (activation=None) — apply sigmoid manually at inference time / let
    BCEWithLogitsLoss apply it internally during training.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick sanity check
    model = build_model()
    dummy = torch.randn(2, cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    out = model(dummy)
    print(f"Model: U-Net + {cfg.ENCODER_NAME}")
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
