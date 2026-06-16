"""Fine-tuning du VAE encoder SD 1.4 + MLP sur FaceForensics++.

Ablation frozen vs fine-tuned : même backbone SD 1.4, même MLP, même
protocole que train.py, SAUF que l'encodeur VAE est entraînable.

Différences clés par rapport à train.py :
  - TrainableVAEEncoder au lieu de FrozenVAEEncoder.
  - Pas de cache latent (les latents changent à chaque epoch).
  - Two-group optimizer : LR_ENCODER (1e-5) pour le VAE, LR_MLP (1e-4) pour le MLP.
  - Batch size réduit (4) à cause des gradients VAE en VRAM.

Run depuis la racine du projet sur blutch :
    python src/train_ldm_finetune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split
from src.model import TrainableVAEEncoder, MLPClassifier
from src.train import auroc   # réutilise la même fonction AUROC, pas de re-training

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FFPP_ROOT            = "/medias/db/deepfakes/Faceforensics"
MANIPULATION         = "Deepfakes"
N_FRAMES_PER_VIDEO   = 5
BATCH_SIZE           = 4        # VAE encoder gradients très coûteux en VRAM à 512×512
EPOCHS               = 20
LR_ENCODER           = 1e-5    # LR bas pour le backbone (éviter catastrophic forgetting)
LR_MLP               = 1e-4    # LR normal pour la tête
WEIGHT_DECAY         = 1e-3
SEED                 = 42
LDM_DIM              = 4 * 64 * 64   # 16384

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Régime : fine-tuning (encodeur VAE SD 1.4 entraînable)")

    # --- Datasets -----------------------------------------------------------
    print("Building datasets from official FF++ splits...")
    train_paths, train_labels = build_ffds_split(
        FFPP_ROOT, "train.json",
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        manipulation=MANIPULATION,
        seed=SEED,
    )
    val_paths, val_labels = build_ffds_split(
        FFPP_ROOT, "val.json",
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        manipulation=MANIPULATION,
        seed=SEED,
    )
    train_loader = DataLoader(
        FFDS(train_paths, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        FFDS(val_paths, val_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    print(f"  train: {len(train_loader.dataset)} samples | val: {len(val_loader.dataset)} samples")

    # --- Modèles ------------------------------------------------------------
    print("Loading TrainableVAEEncoder (SD 1.4)...")
    encoder = TrainableVAEEncoder().to(DEVICE)
    mlp     = MLPClassifier(input_dim=LDM_DIM).to(DEVICE)
    print(f"  VAE params     : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  MLP params     : {sum(p.numel() for p in mlp.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()

    # Two-group optimizer : LR bas sur l'encodeur pour ne pas détruire les
    # représentations pré-entraînées, LR normal sur le MLP.
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": LR_ENCODER},
        {"params": mlp.parameters(),     "lr": LR_MLP},
    ], weight_decay=WEIGHT_DECAY)

    # AMP : réduit la VRAM ~2× et accélère les opérations matricielles sur GPU.
    use_amp = (DEVICE == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    best_val_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ----------------------------------------------------------
        encoder.train()
        mlp.train()
        train_scores, train_targets, losses = [], [], []

        for imgs, ys in train_loader:
            imgs = imgs.to(DEVICE)
            ys   = ys.to(DEVICE).float()

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                z      = encoder(imgs).flatten(1)   # (B, 16384)
                logits = mlp(z)
                loss   = criterion(logits, ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            train_scores.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(ys.cpu())

        train_auc = auroc(torch.cat(train_scores), torch.cat(train_targets))

        # --- VAL ------------------------------------------------------------
        encoder.eval()
        mlp.eval()
        val_scores, val_targets = [], []

        with torch.no_grad():
            for imgs, ys in val_loader:
                imgs = imgs.to(DEVICE)
                with autocast(enabled=use_amp):
                    z      = encoder(imgs).flatten(1)
                    logits = mlp(z)
                val_scores.append(torch.sigmoid(logits).cpu())
                val_targets.append(ys)

        val_auc = auroc(torch.cat(val_scores), torch.cat(val_targets))
        mean_loss = sum(losses) / len(losses)

        print(
            f"Epoch {epoch:>2d}/{EPOCHS} | "
            f"loss {mean_loss:.4f} | "
            f"train AUC {train_auc:.4f} | "
            f"val AUC {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "mlp_state_dict":     mlp.state_dict(),
                },
                OUT_DIR / "best_ldm_finetune.pt",
            )
            print(f"  → best saved (val AUC {val_auc:.4f})")

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint : {OUT_DIR / 'best_ldm_finetune.pt'}")


if __name__ == "__main__":
    main()
