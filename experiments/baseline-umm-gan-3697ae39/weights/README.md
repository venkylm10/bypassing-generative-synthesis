# Weights excluded from GitHub

`last.pt` and `best.pt` (U-Net segmentation checkpoints, ~467MB each) are
excluded from this push — GitHub's hard file-size limit is 100MB. Both are
present as deliverables at `/workspace/output/code/weights/` and mirrored to
`/workspace/output/unet_gan_imputed_weights/` on the experiment pod/GCS.
Each is a complete checkpoint (model, optimizer, scheduler, RNG states,
epoch, best_metric, patience_counter, history) per the checkpoint contract
in CLAUDE.md.
