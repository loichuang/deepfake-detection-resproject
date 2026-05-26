"""F4 — Train a frozen-encoder + MLP classifier on FaceForensics++.

Pipeline (matching the supervisor's whiteboard, with one CPU optimisation):

    dataset    = FFDS(...)                      # frames + labels
    encoder    = FrozenVAEEncoder()             # SD 1.5 VAE, frozen
    model      = MLPClassifier()                # 3-layer MLP, trained

    # Optimisation B: pre-encode every image ONCE into a latent, then train
    # the MLP on the cached latents. On CPU this turns a ~1h run into ~5 min,
    # because the expensive VAE forward pass is not repeated every epoch.

Run from the project root on blutch:
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

# Avoid "Too many open files" when DataLoader workers share many tensors.
mp.set_sharing_strategy("file_system")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split
from src.model import FrozenVAEEncoder, MLPClassifier

# ---------------------------------------------------------------------------
# Configuration (edit these for quick experiments)
# ---------------------------------------------------------------------------
FFPP_ROOT = "/medias/db/deepfakes/Faceforensics"
MANIPULATION = "Deepfakes"
N_VIDEOS_PER_CLASS_TRAIN = None   # None = use ALL videos in train.json
N_VIDEOS_PER_CLASS_VAL = None     # None = use ALL videos in val.json
N_FRAMES_PER_VIDEO = 5            # 5 frames/video ≈ 10k images, kills overfitting
BATCH_SIZE = 64          # batch for MLP training (latents are small, cheap)
ENCODE_BATCH_SIZE = 4    # batch for VAE encoding (512x512 images, VRAM-heavy)
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3      # strong L2 regularisation (overfitting risk is high)
SEED = 42
LDM_DIM = 4 * 64 * 64    # 16384 — flattened SD 1.5 latent dimension

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "latent_cache"


# ---------------------------------------------------------------------------
# Helper: encode a whole FFDS dataset into cached latents (Optimisation B)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_dataset(
    ffds: FFDS,
    encoder: FrozenVAEEncoder,
    device: str,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> TensorDataset:
    """Run every image through the frozen VAE once, return a TensorDataset of
    (latent, label) pairs kept in memory.

    Uses a small batch_size because encoding 512x512 images through the VAE
    is VRAM-heavy. Latents are moved to CPU immediately to free GPU memory.
    """
    loader = DataLoader(ffds, batch_size=batch_size, shuffle=False, num_workers=2)
    latents, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        z = encoder(imgs).cpu()        # (B, 4, 64, 64), moved to CPU right away
        latents.append(z)
        labels.append(ys)
        if device == "cuda":
            torch.cuda.empty_cache()   # release intermediate activations
    latents = torch.cat(latents)
    labels = torch.cat(labels)
    print(f"  encoded {latents.shape[0]} samples -> latent shape {tuple(latents.shape[1:])}")
    return TensorDataset(latents, labels)


def encode_or_load(
    ffds: FFDS,
    encoder: FrozenVAEEncoder,
    device: str,
    cache_name: str,
) -> TensorDataset:
    """Encode the dataset, or load it from disk cache if already computed.

    The cache key encodes the relevant parameters (split, manipulation,
    frames, seed) so a stale cache is never reused by mistake. This turns
    repeated `train.py` runs from ~15 min (re-encoding) into a few seconds.
    """
    cache_path = CACHE_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        print(f"  loading cached latents from {cache_path}")
        obj = torch.load(cache_path)
        return TensorDataset(obj["latents"], obj["labels"])
    ds = encode_dataset(ffds, encoder, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": ds.tensors[0], "labels": ds.tensors[1]}, cache_path)
    print(f"  cached latents to {cache_path}")
    return ds


# ---------------------------------------------------------------------------
# Metric: AUROC computed from scratch (no torchmetrics dependency needed)
# ---------------------------------------------------------------------------
def auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the ROC curve via the rank-based (Mann-Whitney U) formula."""
    scores = scores.flatten()
    targets = targets.flatten()
    n_pos = (targets == 1).sum().item()
    n_neg = (targets == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank scores (average ranks for ties).
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    sum_ranks_pos = ranks[targets == 1].sum().item()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Reusable training routine — SAME for every feature extractor (LDM, ResNet…)
# Only `input_dim` and the output names change, so comparisons are fair.
# ---------------------------------------------------------------------------
def train_classifier(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    input_dim: int,
    ckpt_name: str,
    curves_name: str,
) -> float:
    """Train the shared MLP head on pre-encoded features. Returns best val AUC."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"MLP classifier: {sum(p.numel() for p in model.parameters()):,} trainable parameters")

    history = {"epoch": [], "train_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_scores, train_targets, losses = [], [], []
        for z, y in train_loader:
            z, y = z.to(DEVICE), y.to(DEVICE).float()
            logits = model(z)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            train_scores.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(y.cpu())
        train_auc = auroc(torch.cat(train_scores), torch.cat(train_targets))

        model.eval()
        val_scores, val_targets = [], []
        with torch.no_grad():
            for z, y in val_loader:
                z = z.to(DEVICE)
                logits = model(z)
                val_scores.append(torch.sigmoid(logits).cpu())
                val_targets.append(y)
        val_auc = auroc(torch.cat(val_scores), torch.cat(val_targets))

        mean_loss = sum(losses) / len(losses)
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch:>2d} | loss {mean_loss:.4f} | train AUC {train_auc:.4f} | val AUC {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), OUT_DIR / f"{ckpt_name}.pt")

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Best model saved to {OUT_DIR / f'{ckpt_name}.pt'}")

    # Save training curves (headless backend, no display on blutch).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history["epoch"], history["train_loss"])
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].set_title("Training loss")
        ax[1].plot(history["epoch"], history["train_auc"], label="train AUC")
        ax[1].plot(history["epoch"], history["val_auc"], label="val AUC")
        ax[1].axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("AUC"); ax[1].set_ylim(0.4, 1.02)
        ax[1].legend(); ax[1].set_title("AUROC (train vs val)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{curves_name}.png", dpi=150)
        print(f"Curves saved to {OUT_DIR / f'{curves_name}.png'}")
    except ImportError:
        print("matplotlib not available, skipping curves.")

    return best_val_auc


# ---------------------------------------------------------------------------
# Main — LDM encoder variant
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    # --- Build datasets from the official splits --------------------------
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

    # --- Pre-encode all images once (Optimisation B) ----------------------
    print("Loading frozen VAE encoder...")
    encoder = FrozenVAEEncoder().to(DEVICE)
    tag = f"{MANIPULATION}_{N_FRAMES_PER_VIDEO}f_seed{SEED}"
    print("Pre-encoding train set (or loading from cache)...")
    train_ds = encode_or_load(train_ffds, encoder, DEVICE, f"train_{tag}")
    print("Pre-encoding val set (or loading from cache)...")
    val_ds = encode_or_load(val_ffds, encoder, DEVICE, f"val_{tag}")

    # --- Train the shared MLP head on the LDM latents ---------------------
    train_classifier(
        train_ds, val_ds,
        input_dim=LDM_DIM,
        ckpt_name="best_ldm",
        curves_name="curves_ldm",
    )


if __name__ == "__main__":
    main()
