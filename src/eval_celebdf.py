"""F5 — Cross-dataset generalization: FF++-trained model evaluated on Celeb-DF-v2.

Loads the MLP trained on FaceForensics++ (results/best_mlp.pt) and applies it,
WITHOUT any retraining, to the official Celeb-DF-v2 test set. The gap between
the in-domain FF++ AUC and this cross-dataset AUC measures how well the model
generalizes to deepfakes it has never seen during training.

Run from the project root on blutch:
    python src/eval_celebdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_celebdf_test_split
from src.model import FrozenVAEEncoder, TrainableVAEEncoder, ResNetEncoder, MLPClassifier
from src.train import auroc, encode_dataset, LDM_DIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENCODER_TYPE = "ldm"              # "ldm", "ldm_finetune", ou "resnet"
CELEBDF_ROOT = "/medias/db/deepfakes/Celeb-DF-v2"
N_FRAMES_PER_VIDEO = 5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_encoder_and_model():
    """Return (encoder, input_dim, checkpoint_path) for ENCODER_TYPE."""
    if ENCODER_TYPE == "ldm":
        return FrozenVAEEncoder(), LDM_DIM, RESULTS_DIR / "best_ldm.pt"
    elif ENCODER_TYPE == "ldm_finetune":
        return TrainableVAEEncoder(), LDM_DIM, RESULTS_DIR / "best_ldm_finetune.pt"
    elif ENCODER_TYPE == "resnet":
        return ResNetEncoder(), ResNetEncoder.OUTPUT_DIM, RESULTS_DIR / "best_resnet.pt"
    raise ValueError(f"Unknown ENCODER_TYPE: {ENCODER_TYPE}")


def main() -> None:
    print(f"Device: {DEVICE} | encoder: {ENCODER_TYPE}")
    encoder, input_dim, checkpoint = build_encoder_and_model()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train the {ENCODER_TYPE} model first."
        )

    # --- Build the official Celeb-DF-v2 test set --------------------------
    print("Building Celeb-DF-v2 official test split...")
    paths, labels = build_celebdf_test_split(
        celebdf_root=CELEBDF_ROOT,
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        seed=SEED,
    )
    ffds = FFDS(paths, labels)
    print(f"  test samples: {len(ffds)} (real: {labels.count(0)}, fake: {labels.count(1)})")

    # --- Charger le checkpoint (format différent pour ldm_finetune) -------
    encoder = encoder.to(DEVICE)
    ckpt = torch.load(checkpoint, map_location=DEVICE)

    if ENCODER_TYPE == "ldm_finetune":
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded encoder weights from {checkpoint}")

    encoder.eval()
    print(f"Encoding Celeb-DF test set with {ENCODER_TYPE} encoder...")
    test_ds = encode_dataset(ffds, encoder, DEVICE)

    # --- Load the FF++-trained MLP ----------------------------------------
    model = MLPClassifier(input_dim=input_dim).to(DEVICE)
    if ENCODER_TYPE == "ldm_finetune":
        model.load_state_dict(ckpt["mlp_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded FF++-trained MLP from {checkpoint}")

    # --- Inference ---------------------------------------------------------
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    scores, targets = [], []
    with torch.no_grad():
        for z, y in loader:
            z = z.to(DEVICE)
            logits = model(z)
            scores.append(torch.sigmoid(logits).cpu())
            targets.append(y)
    scores = torch.cat(scores)
    targets = torch.cat(targets)

    # --- Metrics -----------------------------------------------------------
    test_auc = auroc(scores, targets)
    preds = (scores > 0.5).float()
    accuracy = (preds == targets).float().mean().item()

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    print()
    print("=" * 55)
    print("CROSS-DATASET RESULTS — FF++ model on Celeb-DF-v2")
    print("=" * 55)
    print(f"AUC        : {test_auc:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Confusion  : TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print()
    print("Interpretation: compare this AUC to your in-domain FF++ test AUC.")
    print("A large drop is the expected (and publishable) cross-dataset gap.")


if __name__ == "__main__":
    main()
