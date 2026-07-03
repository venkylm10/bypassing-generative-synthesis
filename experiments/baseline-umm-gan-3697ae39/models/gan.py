"""UMM-GAN reimplementation (Zhang et al., 2023, arXiv:2304.05340) for the
specific missing-modality scenario this experiment needs: synthesize the
missing T1ce volume from the 3 available modalities (T1n, T2w, T2f).

Scoping note (see code/README.md and report.md for the full rationale): the
original paper's generator is "unified" in the sense that one model handles
*arbitrary* missing-modality combinations via a modality-availability mask.
Reproducing that fully (training on randomly sampled missing-modality subsets)
is out of scope for this baseline, whose role is specifically "synthesize the
missing T1ce modality" (TASK.yaml context.plan.this_experiment.role). We
therefore reimplement the paper's core idea — an adversarially-trained
encoder-decoder image-to-image generator conditioned on the available
modalities, with a PatchGAN discriminator — fixed to the T1n+T2w+T2f -> T1ce
direction. This is a 3D pix2pix-style GAN, matching the spirit of the
methodology_hints reimplementation (github.com/Anas-github-acc/unified-mm_imputation)
without inheriting a mask-conditioned multi-target head we do not need here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = GConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UMMGenerator3D(nn.Module):
    """Encoder-decoder image synthesis generator: (T1n, T2w, T2f) -> T1ce.

    Single-encoder early-fusion (channel concat of the 3 available modalities
    at input) U-Net, tanh output (targets are z-scored so an unbounded linear
    head is fine too, but tanh + scale keeps GAN training numerically stable
    — targets are clamped to [-3, 3] std and rescaled to [-1, 1] by the
    dataset before training, and predictions are rescaled back at use time).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 24, out_ch: int = 1):
        super().__init__()
        c0, c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.enc0 = GConvBlock(in_ch, c0)
        self.down0 = nn.Conv3d(c0, c0, kernel_size=2, stride=2)
        self.enc1 = GConvBlock(c0, c1)
        self.down1 = nn.Conv3d(c1, c1, kernel_size=2, stride=2)
        self.enc2 = GConvBlock(c1, c2)
        self.down2 = nn.Conv3d(c2, c2, kernel_size=2, stride=2)
        self.enc3 = GConvBlock(c2, c3)
        self.down3 = nn.Conv3d(c3, c3, kernel_size=2, stride=2)
        self.bottleneck = GConvBlock(c3, c4)

        self.up3 = GUpBlock(c4, c3, c3)
        self.up2 = GUpBlock(c3, c2, c2)
        self.up1 = GUpBlock(c2, c1, c1)
        self.up0 = GUpBlock(c1, c0, c0)
        self.head = nn.Sequential(nn.Conv3d(c0, out_ch, kernel_size=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.enc0(x)
        f1 = self.enc1(self.down0(f0))
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        f4 = self.bottleneck(self.down3(f3))
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        d0 = self.up0(d1, f0)
        return self.head(d0)


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator. Input: concat(condition[3ch], candidate[1ch]).

    Outputs a raw (pre-sigmoid) score map; trained with LSGAN (MSE) loss.
    """

    def __init__(self, in_ch: int = 4, base_ch: int = 32):
        super().__init__()

        def block(ci, co, stride=2, norm=True):
            layers = [nn.Conv3d(ci, co, kernel_size=4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm3d(co, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base_ch, norm=False),
            *block(base_ch, base_ch * 2),
            *block(base_ch * 2, base_ch * 4),
            *block(base_ch * 4, base_ch * 4, stride=1),
            nn.Conv3d(base_ch * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3D gradient-difference loss (finite differences along D,H,W), L1."""
    def grad(t):
        gd = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        gh = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        gw = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return gd, gh, gw

    pd, ph, pw = grad(pred)
    td, th, tw = grad(target)
    return (pd - td).abs().mean() + (ph - th).abs().mean() + (pw - tw).abs().mean()


if __name__ == "__main__":
    g = UMMGenerator3D(in_ch=3, base_ch=24, out_ch=1)
    d = PatchDiscriminator3D(in_ch=4, base_ch=32)
    x = torch.randn(1, 3, 96, 96, 96)
    y = g(x)
    print("G out:", y.shape, sum(p.numel() for p in g.parameters()) / 1e6, "M params")
    score = d(torch.cat([x, y], dim=1))
    print("D out:", score.shape, sum(p.numel() for p in d.parameters()) / 1e6, "M params")
