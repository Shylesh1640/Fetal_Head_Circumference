"""
Model Library for Fetal Head Segmentation Template
====================================================
Contains: nnUNet, ResAttnUNet, nnUNet_ResAttn, nnUNet_MSFFBridge_ResAttn,
          UNet_MiTB2, UNet_Hybrid, msunet

NOTE ON PROVENANCE
-----------------------------
Only nnU-Net (github.com/MIC-DKFZ/nnUNet) has a single official repo, and it's a
full self-configuring 3D pipeline/framework -- not a pluggable nn.Module, so it
can't be dropped into a notebook cell as-is. The compound names below
(ResAttn, MSFFBridge, Hybrid, msunet, MiTB2) are combinations of well-known
building blocks from multiple papers, not one canonical repo each:
  - nnU-Net conv block style: Conv -> InstanceNorm -> LeakyReLU (x2)  [Isensee et al. 2021]
  - Attention Gate on skip connections                                [Oktay et al., Attention U-Net, 2018]
  - Residual blocks in encoder/decoder                                 [He et al. ResNet, 2016 / ResUNet]
  - Multi-Scale Feature Fusion (MSFF) bridge = dilated-conv ASPP-style
    fusion at the bottleneck                                          [common pattern, e.g. DeepLab ASPP]
  - MiT-B2 encoder = SegFormer's Mix Transformer encoder, used here via
    segmentation_models_pytorch (smp), which supports it directly       [Xie et al., SegFormer, 2021]
  - "Hybrid" = CNN encoder/decoder with a Transformer self-attention
    block fused at the bottleneck (CNN-Transformer hybrid pattern,
    e.g. TransUNet-style)                                              [Chen et al., TransUNet, 2021]
  - msunet = Multi-Scale U-Net: same-resolution encoder fed with an
    image pyramid (downsampled copies of the input at each encoder
    stage), a common "MS-UNet" pattern in several papers

These are implemented from scratch against the cited designs (not copied from
any single repo) and adapted so every architecture satisfies the template's
contract: input (B,1,256,256) -> output (B,1,256,256) raw logits (no sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ConvINReLU(nn.Module):
    """nnU-Net style block: Conv -> InstanceNorm -> LeakyReLU, x2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Residual version of ConvINReLU (He et al. residual pattern)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return self.act(out + residual)


class AttentionGate(nn.Module):
    """Attention Gate from Oktay et al., Attention U-Net (2018)."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.InstanceNorm2d(inter_ch, affine=True))
        self.W_x = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.InstanceNorm2d(inter_ch, affine=True))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.InstanceNorm2d(1, affine=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return skip * psi


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, block=ConvINReLU):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, block=ConvINReLU, attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attention = AttentionGate(out_ch, out_ch, out_ch // 2) if attention else None
        self.conv = block(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.attention is not None:
            skip = self.attention(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class MSFFBridge(nn.Module):
    """Multi-Scale Feature Fusion bridge (ASPP-style dilated fusion) at the bottleneck."""
    def __init__(self, ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch // 4, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        return self.fuse(torch.cat(feats, dim=1)) + x


class TransformerBottleneck(nn.Module):
    """Lightweight self-attention block for CNN-Transformer hybrids (TransUNet-style)."""
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(ch, ch * 2), nn.GELU(), nn.Linear(ch * 2, ch))
        self.norm2 = nn.LayerNorm(ch)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        t = self.norm(tokens)
        attn_out, _ = self.attn(t, t, t)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# 1) nnUNet  -- practical 2D adaptation of nnU-Net's plain-block design
# ---------------------------------------------------------------------------
class nnUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ConvINReLU(in_ch, chs[0])
        self.d1 = Down(chs[0], chs[1])
        self.d2 = Down(chs[1], chs[2])
        self.d3 = Down(chs[2], chs[3])
        self.d4 = Down(chs[3], chs[4])
        self.u1 = Up(chs[4], chs[3])
        self.u2 = Up(chs[3], chs[2])
        self.u3 = Up(chs[2], chs[1])
        self.u4 = Up(chs[1], chs[0])
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 2) ResAttnUNet -- residual blocks + attention gates on skip connections
# ---------------------------------------------------------------------------
class ResAttnUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ResConvBlock(in_ch, chs[0])
        self.d1 = Down(chs[0], chs[1], block=ResConvBlock)
        self.d2 = Down(chs[1], chs[2], block=ResConvBlock)
        self.d3 = Down(chs[2], chs[3], block=ResConvBlock)
        self.d4 = Down(chs[3], chs[4], block=ResConvBlock)
        self.u1 = Up(chs[4], chs[3], block=ResConvBlock, attention=True)
        self.u2 = Up(chs[3], chs[2], block=ResConvBlock, attention=True)
        self.u3 = Up(chs[2], chs[1], block=ResConvBlock, attention=True)
        self.u4 = Up(chs[1], chs[0], block=ResConvBlock, attention=True)
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 3) nnUNet_ResAttn -- nnU-Net-style backbone + residual + attention gates
# ---------------------------------------------------------------------------
class nnUNet_ResAttn(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ResConvBlock(in_ch, chs[0])
        self.d1 = Down(chs[0], chs[1], block=ResConvBlock)
        self.d2 = Down(chs[1], chs[2], block=ResConvBlock)
        self.d3 = Down(chs[2], chs[3], block=ResConvBlock)
        self.d4 = Down(chs[3], chs[4], block=ResConvBlock)
        self.u1 = Up(chs[4], chs[3], block=ResConvBlock, attention=True)
        self.u2 = Up(chs[3], chs[2], block=ResConvBlock, attention=True)
        self.u3 = Up(chs[2], chs[1], block=ResConvBlock, attention=True)
        self.u4 = Up(chs[1], chs[0], block=ResConvBlock, attention=True)
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 4) nnUNet_MSFFBridge_ResAttn -- above + MSFF bridge at the bottleneck
# ---------------------------------------------------------------------------
class nnUNet_MSFFBridge_ResAttn(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ResConvBlock(in_ch, chs[0])
        self.d1 = Down(chs[0], chs[1], block=ResConvBlock)
        self.d2 = Down(chs[1], chs[2], block=ResConvBlock)
        self.d3 = Down(chs[2], chs[3], block=ResConvBlock)
        self.d4 = Down(chs[3], chs[4], block=ResConvBlock)
        self.bridge = MSFFBridge(chs[4])
        self.u1 = Up(chs[4], chs[3], block=ResConvBlock, attention=True)
        self.u2 = Up(chs[3], chs[2], block=ResConvBlock, attention=True)
        self.u3 = Up(chs[2], chs[1], block=ResConvBlock, attention=True)
        self.u4 = Up(chs[1], chs[0], block=ResConvBlock, attention=True)
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x5 = self.bridge(x5)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 5) UNet_MiTB2 -- U-Net with SegFormer's MiT-B2 transformer encoder,
#    via segmentation_models_pytorch (already a dependency in the template).
# ---------------------------------------------------------------------------
def build_UNet_MiTB2(in_ch=1, out_ch=1):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=out_ch,
        activation=None,
    )


# ---------------------------------------------------------------------------
# 6) UNet_Hybrid -- CNN encoder/decoder with a Transformer self-attention
#    block fused at the bottleneck (TransUNet-style hybrid)
# ---------------------------------------------------------------------------
class UNet_Hybrid(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ConvINReLU(in_ch, chs[0])
        self.d1 = Down(chs[0], chs[1])
        self.d2 = Down(chs[1], chs[2])
        self.d3 = Down(chs[2], chs[3])
        self.d4 = Down(chs[3], chs[4])
        self.trans = TransformerBottleneck(chs[4], num_heads=4)
        self.u1 = Up(chs[4], chs[3])
        self.u2 = Up(chs[3], chs[2])
        self.u3 = Up(chs[2], chs[1])
        self.u4 = Up(chs[1], chs[0])
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x5 = self.trans(x5)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 7) msunet -- Multi-Scale U-Net: encoder fed an image pyramid, fusing a
#    downsampled copy of the raw input at every encoder stage.
# ---------------------------------------------------------------------------
class msunet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = ConvINReLU(in_ch, chs[0])
        # fuse-convs merge the downsampled raw image into each stage
        self.fuse1 = nn.Conv2d(chs[0] + in_ch, chs[0], 1)
        self.fuse2 = nn.Conv2d(chs[1] + in_ch, chs[1], 1)
        self.fuse3 = nn.Conv2d(chs[2] + in_ch, chs[2], 1)
        self.fuse4 = nn.Conv2d(chs[3] + in_ch, chs[3], 1)
        self.pool = nn.MaxPool2d(2)
        self.d1 = ConvINReLU(chs[0], chs[1])
        self.d2 = ConvINReLU(chs[1], chs[2])
        self.d3 = ConvINReLU(chs[2], chs[3])
        self.d4 = ConvINReLU(chs[3], chs[4])
        self.u1 = Up(chs[4], chs[3])
        self.u2 = Up(chs[3], chs[2])
        self.u3 = Up(chs[2], chs[1])
        self.u4 = Up(chs[1], chs[0])
        self.outc = nn.Conv2d(chs[0], out_ch, 1)

    def _fuse(self, feat, raw, fuse_conv):
        raw_ds = F.interpolate(raw, size=feat.shape[2:], mode='bilinear', align_corners=False)
        return fuse_conv(torch.cat([feat, raw_ds], dim=1))

    def forward(self, x):
        x1 = self._fuse(self.inc(x), x, self.fuse1)
        x2 = self._fuse(self.d1(self.pool(x1)), x, self.fuse2)
        x3 = self._fuse(self.d2(self.pool(x2)), x, self.fuse3)
        x4 = self._fuse(self.d3(self.pool(x3)), x, self.fuse4)
        x5 = self.d4(self.pool(x4))
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Registry so the template can select a model by name (see notes below)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "nnUNet": nnUNet,
    "ResAttnUNet": ResAttnUNet,
    "nnUNet_ResAttn": nnUNet_ResAttn,
    "nnUNet_MSFFBridge_ResAttn": nnUNet_MSFFBridge_ResAttn,
    "UNet_MiTB2": build_UNet_MiTB2,      # special-cased: needs no base= arg
    "UNet_Hybrid": UNet_Hybrid,
    "msunet": msunet,
}


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    for name, cls in MODEL_REGISTRY.items():
        try:
            m = cls()
            y = m(x)
            assert y.shape == (2, 1, 256, 256), f"{name}: bad shape {y.shape}"
            print(f"{name:30s} OK  params={sum(p.numel() for p in m.parameters()):,}")
        except Exception as e:
            print(f"{name:30s} FAILED: {e}")
