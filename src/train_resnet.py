"""F6 — ResNet-50 feature extractor baseline (comparison vs LDM encoder).

Same protocol as train.py (same FF++ splits, same frames, SAME MLP head),
but the feature extractor is a frozen ImageNet ResNet-50 instead of the
Stable Diffusion VAE encoder. This is the controlled comparison the supervisor
asked for: only the upstream extractor changes, so any difference in AUC is
attributable to the extractor, not the classifier or the data.

Run from the project root on blutch:
    python src/train_resnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split
from src.model import ResNetEncoder
# Reuse the shared pipeline from train.py: encoding cache, training loop, config.
from src.train import (
    DEVICE,
    MANIPULATION,
    N_FRAMES_PER_VIDEO,
    N_VIDEOS_PER_CLASS_TRAIN,
    N_VIDEOS_PER_CLASS_VAL,
    FFPP_ROOT,
    SEED,
    encode_or_load,
    train_classifier,
)


def main() -> None:
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    # --- Same FF++ splits as the LDM variant ------------------------------
    print("Building datasets from official FF++ splits...")
    train_paths, train_labels = build_ffds_split(
        FFPP_ROOT, "train.json",
        n_videos_per_class=N_VIDEOS_PER_CLASS_TRAIN,
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        manipulation=MANIPULATION,
        seed=SEED,
    )
    val_paths, val_labels = build_ffds_split(
        FFPP_ROOT, "val.json",
        n_videos_per_class=N_VIDEOS_PER_CLASS_VAL,
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        manipulation=MANIPULATION,
        seed=SEED,
    )
    train_ffds = FFDS(train_paths, train_labels)
    val_ffds = FFDS(val_paths, val_labels)
    print(f"  train: {len(train_ffds)} samples | val: {len(val_ffds)} samples")

    # --- Extract ResNet-50 features once (cached separately from LDM) -----
    print("Loading frozen ResNet-50 feature extractor...")
    encoder = ResNetEncoder().to(DEVICE)
    tag = f"resnet_{MANIPULATION}_{N_FRAMES_PER_VIDEO}f_seed{SEED}"
    print("Extracting train features (or loading from cache)...")
    train_ds = encode_or_load(train_ffds, encoder, DEVICE, f"train_{tag}")
    print("Extracting val features (or loading from cache)...")
    val_ds = encode_or_load(val_ffds, encoder, DEVICE, f"val_{tag}")

    # --- Train the SAME MLP head, only input_dim differs (2048) -----------
    train_classifier(
        train_ds, val_ds,
        input_dim=ResNetEncoder.OUTPUT_DIM,   # 2048
        ckpt_name="best_resnet",
        curves_name="curves_resnet",
    )


if __name__ == "__main__":
    main()
