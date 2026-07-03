"""Copy the primary (seed 0) checkpoint into the platform-required
`latent_imputed_unet_weights/` deliverable path."""
import os
import shutil

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))

src_dir = os.path.join(CODE_DIR, "weights_seed0")
dst_dir = os.path.join(OUTPUT_DIR, "latent_imputed_unet_weights")
os.makedirs(dst_dir, exist_ok=True)

for fname in ["best.pt", "last.pt"]:
    src = os.path.join(src_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_dir, fname))
        print(f"copied {src} -> {dst_dir}/{fname}")
    else:
        print(f"WARNING: {src} not found")
