"""
Model zoo. Every model takes a 3x256x256 image and outputs a 1x256x256
sigmoid probability mask, so they all drop into the same train/eval loop.

Install once:
    pip install segmentation-models-pytorch timm --break-system-packages
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(name: str):
    name = name.lower()

    if name == "unet_baseline":
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation="sigmoid")

    if name == "attention_unet":
        # scSE = concurrent spatial + channel squeeze-excitation attention
        # applied at every decoder stage -> "Attention U-Net" equivalent
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         decoder_attention_type="scse",
                         in_channels=3, classes=1, activation="sigmoid")

    if name == "dilated_unet":
        # DeepLabV3+ = atrous/dilated convolutions + multiscale ASPP context
        return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet",
                                  in_channels=3, classes=1, activation="sigmoid")

    if name == "dense_unet":
        return smp.Unet(encoder_name="densenet121", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation="sigmoid")

    if name == "ms_unet":
        # U-Net++ : nested, multi-scale dense skip pathways
        return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet",
                                 in_channels=3, classes=1, activation="sigmoid")

    if name == "transformer_unet":
        # paper's own model: U-Net decoder + MiT-B2 transformer encoder
        return smp.Unet(encoder_name="mit_b2", encoder_weights="imagenet",
                         in_channels=3, classes=1, activation="sigmoid")

    if name == "segformer":
        return smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet",
                              in_channels=3, classes=1, activation="sigmoid")

    raise ValueError(f"Unknown model name: {name}")


MODEL_NAMES = [
    "unet_baseline",
    "attention_unet",
    "dilated_unet",
    "dense_unet",
    "ms_unet",
    "transformer_unet",
    "segformer",
]


class DualEncoderFusion(nn.Module):
    """
    Hybrid model: fuses bottleneck features from two of the trained
    single-encoder models above, then decodes with a fresh U-Net decoder.

    Use this AFTER you've identified your top-2 single models from the
    leaderboard (e.g. transformer_unet + attention_unet).
    """

    def __init__(self, encoder_a: nn.Module, encoder_b: nn.Module,
                 feat_channels_a: int, feat_channels_b: int, out_channels=256):
        super().__init__()
        self.encoder_a = encoder_a          # e.g. smp model's .encoder
        self.encoder_b = encoder_b
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels_a + feat_channels_b, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # lightweight decoder back up to 256x256, 1 channel
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat_a = self.encoder_a(x)[-1]
        feat_b = self.encoder_b(x)[-1]
        if feat_a.shape[-2:] != feat_b.shape[-2:]:
            feat_b = nn.functional.interpolate(feat_b, size=feat_a.shape[-2:],
                                                mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([feat_a, feat_b], dim=1))
        out = self.decoder(fused)
        out = nn.functional.interpolate(out, size=x.shape[-2:],
                                         mode="bilinear", align_corners=False)
        return out
