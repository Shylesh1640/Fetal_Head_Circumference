"""
model.py
--------
MS-UNet: U-Net decoder with a Mix Vision Transformer (MiT-B2) encoder,
built on top of Segmentation Models PyTorch (SMP).

This matches the architecture described in Section III-B of the paper:
    - MiT-B2 encoder (SegFormer backbone), ImageNet-pretrained, 4 encoding
      stages producing features at 64x64 -> ... -> deep bottleneck
    - U-Net decoder with skip connections, BatchNorm + ReLU refinement at
      each upsampling stage
    - 1x1 convolution + sigmoid segmentation head -> output mask
      of shape (1, H, W)

Usage
-----
    from model import build_model
    model = build_model()                 # mit_b2 + imagenet weights
    logits = model(x)                     # x: [B, 3, H, W] -> [B, 1, H, W] (raw logits)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import config


def build_model(
    encoder_name: str = config.ENCODER_NAME,
    encoder_weights: str = config.ENCODER_WEIGHTS,
    in_channels: int = config.IN_CHANNELS,
    classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """
    Build the U-Net + MiT-B2 segmentation model.

    The model returns RAW LOGITS (activation=None). Apply torch.sigmoid()
    externally when you need probabilities (this is what BCEWithLogitsLoss
    and our hybrid loss expect, for numerical stability).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


class MSUNet(nn.Module):
    """
    Thin wrapper around the SMP Unet+MiT-B2 model. Having an explicit class
    (rather than using the smp.Unet object directly) makes it easy to:
      - register Grad-CAM hooks on a named encoder stage,
      - extend the model later (e.g. add a classification head),
      - keep a single import point (`from model import MSUNet`).
    """

    def __init__(
        self,
        encoder_name: str = config.ENCODER_NAME,
        encoder_weights: str = config.ENCODER_WEIGHTS,
        in_channels: int = config.IN_CHANNELS,
        classes: int = config.NUM_CLASSES,
    ):
        super().__init__()
        self.net = build_model(encoder_name, encoder_weights, in_channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def encoder(self):
        return self.net.encoder

    @property
    def decoder(self):
        return self.net.decoder

    @property
    def segmentation_head(self):
        return self.net.segmentation_head


if __name__ == "__main__":
    # Quick sanity check
    model = MSUNet()
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        out = model(x)
    print("Input :", x.shape)
    print("Output:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params/1e6:.2f}M")
