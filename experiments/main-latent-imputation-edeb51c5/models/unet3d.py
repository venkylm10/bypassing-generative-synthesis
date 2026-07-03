"""Multi-encoder, late-fusion 3D U-Net with a deterministic latent-imputation
branch for a missing modality.

Each modality (t1n, t1c, t2w, t2f) has its own independent encoder pathway.
At every downsampling scale, the four per-modality feature maps are fused
(channel concat + 1x1x1 conv) into a single skip-connection tensor consumed
by a shared decoder. When a modality is "missing", instead of running its
real encoder we substitute, at every scale, the Normalized Mean Algorithm
(NMA) of the *available* modalities' feature maps at that same scale:
each available feature map is instance-normalized (per-sample, per-channel
zero mean / unit std over spatial dims) and the normalized maps are averaged.
This requires no learned imputation network and no ground-truth access to
the missing modality (TASK.yaml.context.method: "a deterministic latent
imputation layer ... computed directly on the available modalities' latent
embeddings").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


class ModalityEncoder(nn.Module):
    """Independent per-modality encoder producing a 4-scale feature pyramid."""

    def __init__(self, in_ch=1, widths=(16, 32, 64, 128)):
        super().__init__()
        self.stages = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.stages.append(conv_block(prev, w))
            prev = w
        self.downs = nn.ModuleList([nn.Conv3d(w, w, 3, stride=2, padding=1) for w in widths[:-1]])

    def forward(self, x):
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return feats  # list of 4 tensors, decreasing spatial res, increasing channels


def normalized_mean_impute(feats):
    """Normalized Mean Algorithm: z-score each available feature map per
    (sample, channel) over spatial dims, then average across modalities.
    `feats`: list of (B, C, D, H, W) tensors from available modalities at one scale.
    """
    normed = []
    for f in feats:
        mean = f.mean(dim=(2, 3, 4), keepdim=True)
        std = f.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        normed.append((f - mean) / std)
    return torch.stack(normed, dim=0).mean(dim=0)


class FusionUNet3D(nn.Module):
    """Multi-encoder late-fusion 3D U-Net.

    forward(volumes, missing) where `volumes` is a dict modality->tensor(B,1,D,H,W)
    (the missing modality's tensor is ignored/may be absent) and `missing` is
    the modality name to impute via NMA (or None to use all 4 real encoders,
    e.g. for the GAN-imputed / zero-fill comparison pipelines).
    """

    def __init__(self, widths=(16, 32, 64, 128), bottleneck_extra=256, n_classes=4):
        super().__init__()
        self.widths = widths
        self.encoders = nn.ModuleDict({m: ModalityEncoder(1, widths) for m in MODALITIES})
        # fusion convs: one per scale, concat 4*C_s -> C_s
        self.fusions = nn.ModuleList([nn.Conv3d(4 * w, w, 1) for w in widths])
        # one extra downsample (deepest encoder scale -> bottleneck scale) before the
        # bottleneck conv, so the decoder has exactly len(widths) upsampling steps
        self.bottleneck_down = nn.Conv3d(widths[-1], widths[-1], 3, stride=2, padding=1)
        self.bottleneck = conv_block(widths[-1], bottleneck_extra)

        up_widths = list(widths[::-1])  # 128, 64, 32, 16
        in_ch = bottleneck_extra
        self.up_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for w in up_widths:
            self.up_samples.append(nn.ConvTranspose3d(in_ch, w, 2, stride=2))
            self.up_convs.append(conv_block(w + w, w))  # skip has width w (fused)
            in_ch = w
        self.out_conv = nn.Conv3d(up_widths[-1], n_classes, 1)

    def forward(self, volumes, missing=None):
        per_modal_feats = {}
        for m in MODALITIES:
            if m == missing:
                continue
            per_modal_feats[m] = self.encoders[m](volumes[m])

        n_scales = len(self.widths)
        fused_skips = []
        for s in range(n_scales):
            scale_feats = []
            for m in MODALITIES:
                if m == missing:
                    continue
                scale_feats.append(per_modal_feats[m][s])
            if missing is not None:
                imputed = normalized_mean_impute(scale_feats)
                # reinsert in canonical modality order for a fixed fusion-conv input layout
                ordered = []
                for m in MODALITIES:
                    ordered.append(imputed if m == missing else per_modal_feats[m][s])
            else:
                ordered = [per_modal_feats[m][s] for m in MODALITIES]
            fused = self.fusions[s](torch.cat(ordered, dim=1))
            fused_skips.append(fused)

        x = self.bottleneck_down(fused_skips[-1])
        x = self.bottleneck(x)
        # decoder consumes skips from deepest-but-one scale outward
        skips_for_decoder = fused_skips[::-1]  # scale index n-1 .. 0
        for i, (up, conv) in enumerate(zip(self.up_samples, self.up_convs)):
            x = up(x)
            skip = skips_for_decoder[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return self.out_conv(x)
