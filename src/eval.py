"""F4/F5 — Evaluate a trained MLP on a held-out FF++ split.

Loads the best checkpoint saved by train.py and reports AUC, accuracy and a
confusion matrix on a split the model has NEVER seen during training.

By default we evaluate on the official `test.json` split (in-domain FF++).
Later, for true cross-dataset generalization (F5), we will point this at
Celeb-DF-v2 instead.

Run from the project root on blutch:
    python src/eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split
from src.model import FrozenVAEEncoder, TrainableVAEEncoder, ResNetEncoder, MLPClassifier
# Reuse helpers from train.py. Importing does NOT trigger training,
# because train.py guards its entry point with `if __name__ == "__main__"`.
from src.train import auroc, encode_dataset, LDM_DIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENCODER_TYPE = "ldm"              # "ldm", "ldm_finetune", ou "resnet"
FFPP_ROOT = "/medias/db/deepfakes/Faceforensics"
MANIPULATION = "Deepfakes"
SPLIT = "test.json"               # in-domain test set
N_VIDEOS_PER_CLASS = None         # None = use ALL videos in the split
N_FRAMES_PER_VIDEO = 5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
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

    # --- Build the held-out test set --------------------------------------
    print(f"Building test split ({SPLIT})...")
    paths, labels = build_ffds_split(
        FFPP_ROOT, SPLIT,
        n_videos_per_class=N_VIDEOS_PER_CLASS,
        n_frames_per_video=N_FRAMES_PER_VIDEO,
        manipulation=MANIPULATION,
        seed=SEED,
    )
    ffds = FFDS(paths, labels)
    print(f"  test samples: {len(ffds)} (real: {labels.count(0)}, fake: {labels.count(1)})")

    # --- Charger le checkpoint (format différent pour ldm_finetune) -------
    encoder = encoder.to(DEVICE)
    ckpt = torch.load(checkpoint, map_location=DEVICE)

    if ENCODER_TYPE == "ldm_finetune":
        # Le checkpoint contient encoder + MLP sous forme de dict
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded encoder weights from {checkpoint}")

    encoder.eval()
    print(f"Encoding test set with {ENCODER_TYPE} encoder...")
    test_ds = encode_dataset(ffds, encoder, DEVICE)

    # --- Load the trained MLP ---------------------------------------------
    model = MLPClassifier(input_dim=input_dim).to(DEVICE)
    if ENCODER_TYPE == "ldm_finetune":
        model.load_state_dict(ckpt["mlp_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded MLP from {checkpoint}")

    # --- Inference ---------------------------------------------------------
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
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

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    print()
    print("=" * 50)
    print(f"TEST RESULTS — in-domain FF++ ({MANIPULATION}, {SPLIT})")
    print("=" * 50)
    print(f"AUC        : {test_auc:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Confusion  : TP={tp}  TN={tn}  FP={fp}  FN={fn}")


if __name__ == "__main__":
    main()
