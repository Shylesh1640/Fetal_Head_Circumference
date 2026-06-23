"""
Dense U-Net for Binary Medical Image Segmentation
===================================================
Architecture family: FC-DenseNet / "Tiramisu"-style Dense U-Net
(Jégou et al., "The One Hundred Layers Tiramisu", CVPR-W 2017),
which is the standard reference architecture behind the term
"Dense U-Net" for 2D biomedical segmentation (as opposed to the
3D hybrid H-DenseUNet used for CT volumes).

Design:
  - Encoder: a stack of DenseBlocks, each followed by a TransitionDown
    (BN -> ReLU -> 1x1 Conv -> Dropout -> 2x2 MaxPool).
  - Bottleneck: one DenseBlock at the lowest resolution.
  - Decoder: a stack of TransitionUp (2x2 transposed conv) + DenseBlock,
    where the input to every decoder DenseBlock is the concatenation of
    the upsampled feature map and the corresponding encoder skip
    connection (classic U-Net skip, but operating on densely-connected
    feature blocks instead of plain conv blocks).
  - Every DenseLayer inside a DenseBlock follows the original DenseNet
    recipe: BN -> ReLU -> 3x3 Conv -> Dropout, and its output is
    concatenated (not summed) to its block's running feature map, which
    is what makes the block "dense" (every layer sees every previous
    layer's output within the block).

This file is self-contained and only depends on torch / torch.nn.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """Single densely-connected layer: BN -> ReLU -> 3x3 Conv -> Dropout.

    The output (growth_rate channels) is meant to be concatenated to the
    block's running feature map by the caller (DenseBlock).
    """

    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        out = self.drop(out)
        return out


class DenseBlock(nn.Module):
    """A block of `n_layers` DenseLayers with dense (concatenative) connectivity.

    If `upsample_block` is True, the block returns ONLY the concatenation of
    the newly produced layers (used in the decoder, to control channel
    growth as recommended by the Tiramisu paper). Otherwise it returns the
    concatenation of the input together with all produced layers (used in
    the encoder, where we want full feature reuse before transition-down).
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        n_layers: int,
        dropout: float = 0.2,
        upsample_block: bool = False,
    ):
        super().__init__()
        self.upsample_block = upsample_block
        self.layers = nn.ModuleList()
        cur_channels = in_channels
        for _ in range(n_layers):
            self.layers.append(DenseLayer(cur_channels, growth_rate, dropout))
            cur_channels += growth_rate

        # Output channel count exposed for the parent module to wire up.
        self.out_channels_full = cur_channels  # input + all new features
        self.out_channels_new = growth_rate * n_layers  # only new features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        if self.upsample_block:
            # only the freshly produced feature maps (skip the original input)
            return torch.cat(features[1:], dim=1)
        return torch.cat(features, dim=1)


class TransitionDown(nn.Module):
    """BN -> ReLU -> 1x1 Conv -> Dropout -> 2x2 MaxPool (halves H, W)."""

    def __init__(self, in_channels: int, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.bn(x)))
        x = self.drop(x)
        x = self.pool(x)
        return x


class TransitionUp(nn.Module):
    """2x2 stride-2 transposed convolution that doubles H, W."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convT = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.convT(x)
        # Guard against off-by-one size mismatches from odd input resolutions.
        if out.shape[-2:] != skip.shape[-2:]:
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([out, skip], dim=1)


class DenseUNet(nn.Module):
    """Dense U-Net for binary segmentation.

    Args:
        in_channels: number of input image channels (1 for grayscale ultrasound).
        num_classes: number of output channels (1 for binary segmentation; raw
            logits are returned, apply sigmoid externally for probabilities).
        first_conv_channels: channels produced by the initial stem 3x3 conv.
        down_blocks: number of DenseLayers in each of the encoder DenseBlocks,
            read left-to-right from the input towards the bottleneck.
        up_blocks: number of DenseLayers in each of the decoder DenseBlocks,
            read from the bottleneck towards the output.
        bottleneck_layers: number of DenseLayers in the bottleneck DenseBlock.
        growth_rate: number of feature maps added per DenseLayer.
        dropout: dropout probability used after every conv in the network.

    Default configuration is a compact 5-level Dense U-Net (~9M params) that
    is comfortable to train on a single consumer/cloud GPU (e.g. T4/A10/RTX),
    matching the depth of the standard 256x256 U-Net used for HC18.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        first_conv_channels: int = 48,
        down_blocks: tuple = (4, 4, 4, 4, 4),
        up_blocks: tuple = (4, 4, 4, 4, 4),
        bottleneck_layers: int = 4,
        growth_rate: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(down_blocks) == len(up_blocks), "down_blocks and up_blocks must match in length"
        self.n_levels = len(down_blocks)

        # ---------------- Stem ----------------
        self.stem = nn.Conv2d(in_channels, first_conv_channels, kernel_size=3, padding=1, bias=False)

        # ---------------- Encoder ----------------
        self.encoder_blocks = nn.ModuleList()
        self.transition_downs = nn.ModuleList()
        skip_channels = []  # channel count of each skip connection, shallow -> deep
        cur_channels = first_conv_channels
        for n_layers in down_blocks:
            block = DenseBlock(cur_channels, growth_rate, n_layers, dropout, upsample_block=False)
            self.encoder_blocks.append(block)
            cur_channels = block.out_channels_full
            skip_channels.append(cur_channels)
            self.transition_downs.append(TransitionDown(cur_channels, dropout))

        # ---------------- Bottleneck ----------------
        self.bottleneck = DenseBlock(
            cur_channels, growth_rate, bottleneck_layers, dropout, upsample_block=True
        )
        cur_channels = self.bottleneck.out_channels_new

        # ---------------- Decoder ----------------
        self.transition_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i, n_layers in enumerate(up_blocks):
            skip_ch = skip_channels[-(i + 1)]
            self.transition_ups.append(TransitionUp(cur_channels, cur_channels))
            concat_channels = cur_channels + skip_ch
            is_last = i == len(up_blocks) - 1
            block = DenseBlock(
                concat_channels, growth_rate, n_layers, dropout, upsample_block=not is_last
            )
            self.decoder_blocks.append(block)
            cur_channels = block.out_channels_new if not is_last else block.out_channels_full

        # ---------------- Head ----------------
        self.head = nn.Conv2d(cur_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips = []
        for block, down in zip(self.encoder_blocks, self.transition_downs):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for i, (up, block) in enumerate(zip(self.transition_ups, self.decoder_blocks)):
            skip = skips[-(i + 1)]
            x = up(x, skip)
            x = block(x)

        logits = self.head(x)
        return logits


def build_dense_unet(config: dict | None = None) -> DenseUNet:
    """Factory used by the Lightning module so the architecture can be
    configured purely from a YAML / dict config without importing torch
    types elsewhere."""
    config = config or {}
    return DenseUNet(
        in_channels=config.get("in_channels", 1),
        num_classes=config.get("num_classes", 1),
        first_conv_channels=config.get("first_conv_channels", 48),
        down_blocks=tuple(config.get("down_blocks", (4, 4, 4, 4, 4))),
        up_blocks=tuple(config.get("up_blocks", (4, 4, 4, 4, 4))),
        bottleneck_layers=config.get("bottleneck_layers", 4),
        growth_rate=config.get("growth_rate", 12),
        dropout=config.get("dropout", 0.2),
    )


if __name__ == "__main__":
    # Quick sanity check: forward pass + parameter count.
    model = build_dense_unet({"in_channels": 1, "num_classes": 1})
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.2f} M)")
