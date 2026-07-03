"""Multi-encoder, late-fusion 3D U-Net for missing-modality brain tumor segmentation.

Architecture rationale (see /workspace/TASK.yaml context.plan methodology review):
this experiment must share ONE architecture with the sibling experiments
(main_latent_imputation, baseline_umm_gan) so the imputation *method* is the only
thing that varies. A "standard" single-encoder early-fusion 3D U-Net cannot host a
latent-space imputation step, because there is no per-modality latent to impute.
So all siblings use this shape: one encoder branch per modality (T1n, T1c, T2w,
T2f), each producing multi-scale features; features are fused (channel-concat +
1x1x1 conv) at every scale; a single shared decoder consumes the fused features.

- baseline_zero_fill (this experiment): the missing modality's *input volume* is
  zero-filled, then passed through its normal encoder branch like any other
  modality.
- baseline_umm_gan (sibling): the missing modality's input volume is replaced by
  UMM-GAN pixel synthesis, then passed through the same encoder branch.
- main_latent_imputation (sibling): the missing modality's *encoder output*
  (rather than its input) is replaced by a deterministic function of the other
  encoders' latents.

Only the construction of the missing branch's contribution differs; the encoder
weights, fusion, and decoder are architecturally identical across all three.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """(Conv3d -> InstanceNorm3d -> LeakyReLU) x2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoder(nn.Module):
    """Single-modality encoder producing multi-scale features.

    5 scales: level0 (full res) .. level3, bottleneck (1/16 res).
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        c0, c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.enc0 = ConvBlock3D(in_ch, c0)
        self.down0 = nn.Conv3d(c0, c0, kernel_size=2, stride=2)
        self.enc1 = ConvBlock3D(c0, c1)
        self.down1 = nn.Conv3d(c1, c1, kernel_size=2, stride=2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.down2 = nn.Conv3d(c2, c2, kernel_size=2, stride=2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.down3 = nn.Conv3d(c3, c3, kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3D(c3, c4)
        self.out_channels = [c0, c1, c2, c3, c4]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.enc0(x)
        f1 = self.enc1(self.down0(f0))
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        f4 = self.bottleneck(self.down3(f3))
        return [f0, f1, f2, f3, f4]


class FusionBlock(nn.Module):
    """Channel-concat fusion across modality encoders + 1x1x1 conv projection."""

    def __init__(self, per_modality_ch: int, n_modalities: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(per_modality_ch * n_modalities, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(feats, dim=1))


class UpBlock3D(nn.Module):
    """Transposed-conv upsample + skip concat + ConvBlock3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiEncoderUNet3D(nn.Module):
    """Multi-encoder, late-fusion 3D U-Net.

    Input: dict/list of `n_modalities` single-channel volumes (B, 1, D, H, W)
    each, in a fixed modality order. A missing modality is represented by
    whatever the caller passes for that slot (zeros for the zero-fill
    baseline) — the network itself is agnostic to how that slot was filled.

    Output: (B, n_classes, D, H, W) raw logits (sigmoid applied in the loss/
    metric, not here, for numerical stability with BCEWithLogits).
    """

    def __init__(self, n_modalities: int = 4, base_ch: int = 16, n_classes: int = 3):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList(
            [ModalityEncoder(in_ch=1, base_ch=base_ch) for _ in range(n_modalities)]
        )
        ch = self.encoders[0].out_channels  # [c0, c1, c2, c3, c4]
        self.fuse0 = FusionBlock(ch[0], n_modalities, ch[0])
        self.fuse1 = FusionBlock(ch[1], n_modalities, ch[1])
        self.fuse2 = FusionBlock(ch[2], n_modalities, ch[2])
        self.fuse3 = FusionBlock(ch[3], n_modalities, ch[3])
        self.fuse4 = FusionBlock(ch[4], n_modalities, ch[4])

        self.up3 = UpBlock3D(ch[4], ch[3], ch[3])
        self.up2 = UpBlock3D(ch[3], ch[2], ch[2])
        self.up1 = UpBlock3D(ch[2], ch[1], ch[1])
        self.up0 = UpBlock3D(ch[1], ch[0], ch[0])
        self.head = nn.Conv3d(ch[0], n_classes, kernel_size=1)

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        assert len(modalities) == self.n_modalities
        per_scale: list[list[torch.Tensor]] = [[], [], [], [], []]
        for m_idx, vol in enumerate(modalities):
            feats = self.encoders[m_idx](vol)
            for s in range(5):
                per_scale[s].append(feats[s])

        s0 = self.fuse0(per_scale[0])
        s1 = self.fuse1(per_scale[1])
        s2 = self.fuse2(per_scale[2])
        s3 = self.fuse3(per_scale[3])
        s4 = self.fuse4(per_scale[4])

        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)
        return self.head(d0)


if __name__ == "__main__":
    net = MultiEncoderUNet3D(n_modalities=4, base_ch=16, n_classes=3)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    x = [torch.randn(1, 1, 96, 96, 96) for _ in range(4)]
    y = net(x)
    print("output shape:", y.shape)
