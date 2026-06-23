"""
Attention U-Net (Oktay et al., 2018 - "Attention U-Net: Learning Where to Look for the Pancreas")

Architecture follows the canonical public PyTorch implementation structure
(conv_block / up_conv / Attention_block / AttU_Net), the most widely used and
cited reference implementation of this paper:
    https://github.com/LeeJunHyun/Image_Segmentation

Channel progression: 64 -> 128 -> 256 -> 512 -> 1024 (5 encoder stages),
mirrored decoder with attention-gated skip connections.
"""

import torch
import torch.nn as nn


class conv_block(nn.Module):
    """Two consecutive (Conv3x3 -> BatchNorm -> ReLU) blocks."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """Upsample (x2) -> Conv3x3 -> BatchNorm -> ReLU."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    """
    Additive attention gate.
    g  : gating signal from the coarser (decoder) scale
    x  : skip-connection feature map from the encoder
    F_g, F_l : number of channels of g and x respectively
    F_int    : number of intermediate channels
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttU_Net(nn.Module):
    """
    Full Attention U-Net.

    img_ch     : number of input channels (1 for grayscale ultrasound, 3 if you
                 replicate the channel to feed ImageNet-style pipelines)
    output_ch  : number of output channels (1 for binary fetal-head mask)
    """

    def __init__(self, img_ch: int = 1, output_ch: int = 1, base_ch: int = 64):
        super().__init__()

        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=c1)
        self.Conv2 = conv_block(ch_in=c1, ch_out=c2)
        self.Conv3 = conv_block(ch_in=c2, ch_out=c3)
        self.Conv4 = conv_block(ch_in=c3, ch_out=c4)
        self.Conv5 = conv_block(ch_in=c4, ch_out=c5)

        # Decoder + attention gates
        self.Up5 = up_conv(ch_in=c5, ch_out=c4)
        self.Att5 = Attention_block(F_g=c4, F_l=c4, F_int=c3)
        self.Up_conv5 = conv_block(ch_in=c5, ch_out=c4)

        self.Up4 = up_conv(ch_in=c4, ch_out=c3)
        self.Att4 = Attention_block(F_g=c3, F_l=c3, F_int=c2)
        self.Up_conv4 = conv_block(ch_in=c4, ch_out=c3)

        self.Up3 = up_conv(ch_in=c3, ch_out=c2)
        self.Att3 = Attention_block(F_g=c2, F_l=c2, F_int=c1)
        self.Up_conv3 = conv_block(ch_in=c3, ch_out=c2)

        self.Up2 = up_conv(ch_in=c2, ch_out=c1)
        self.Att2 = Attention_block(F_g=c1, F_l=c1, F_int=base_ch // 2)
        self.Up_conv2 = conv_block(ch_in=c2, ch_out=c1)

        self.Conv_1x1 = nn.Conv2d(c1, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding + attention-gated skip connections
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        return out  # raw logits — apply sigmoid in loss / metrics, not here


if __name__ == "__main__":
    # quick shape sanity check
    model = AttU_Net(img_ch=1, output_ch=1)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("Output shape:", y.shape)
    print(f"Total parameters: {n_params:,}")
