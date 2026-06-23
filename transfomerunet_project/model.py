"""
model.py
U-Net decoder + MiT-B2 (SegFormer / Mix Vision Transformer B2) encoder,
built via the `segmentation_models_pytorch` (SMP) library.

SMP is the canonical, actively-maintained open-source implementation that
exposes `mit_b0`...`mit_b5` SegFormer encoders (backed by `timm`) as drop-in
U-Net encoders. This is the exact architecture combination described in the
paper: U-Net (spatial/skip-connection decoder) + MiT-B2 (ImageNet-pretrained
transformer encoder).

Repo: https://github.com/qubvel-org/segmentation_models.pytorch
"""

import segmentation_models_pytorch as smp
import config


def build_model() -> "smp.Unet":
    """
    Builds U-Net with a MiT-B2 transformer encoder, pretrained on ImageNet.

    Note: activation=None -> model returns raw logits. Sigmoid is applied
    explicitly inside the loss function and metric computation, which is the
    numerically-stable convention (avoids float overflow issues that can
    occur from applying sigmoid twice or from BCE-on-probabilities).
    """
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,        # "mit_b2"
        encoder_weights=config.ENCODER_WEIGHTS,  # "imagenet"
        in_channels=config.IN_CHANNELS,           # 3
        classes=config.NUM_CLASSES,                # 1 (binary)
        activation=config.ACTIVATION,              # None -> raw logits
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: forward pass with a dummy batch.
    import torch

    m = build_model()
    x = torch.randn(2, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Total params: {n_params:,}")
    print(f"Trainable params: {n_trainable:,}")
