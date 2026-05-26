"""End-to-end comparison: LDM encoder vs ResNet-50 feature extractor.

For each extractor, this script:
  1. encodes the FF++ train/val/test splits and the Celeb-DF-v2 test set
     (cached on disk, so re-runs are fast),
  2. trains the SAME MLP head (only input_dim differs),
  3. evaluates on FF++ test (in-domain) and Celeb-DF (cross-dataset),

then prints and saves a single comparison table. This directly answers the
supervisor's question: does the diffusion encoder capture manipulation traces
better or worse than a standard ResNet?

Run from the project root on blutch:
    python scripts/run_comparison.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split, build_celebdf_test_split
from src.model import FrozenVAEEncoder, ResNetEncoder, MLPClassifier
from src.train import (
    DEVICE, FFPP_ROOT, MANIPULATION, N_FRAMES_PER_VIDEO,
    N_VIDEOS_PER_CLASS_TRAIN, N_VIDEOS_PER_CLASS_VAL, SEED, OUT_DIR,
    auroc, encode_or_load, train_classifier,
)

CELEBDF_ROOT = "/medias/db/deepfakes/Celeb-DF-v2"

# Each extractor: (constructor, feature dimension fed to the MLP)
EXTRACTORS = {
    "ldm":    (FrozenVAEEncoder, 4 * 64 * 64),   # 16384
    "resnet": (ResNetEncoder,    ResNetEncoder.OUTPUT_DIM),  # 2048
}


@torch.no_grad()
def evaluate(model: MLPClassifier, encoded_ds, device: str) -> float:
    """Return AUC of `model` on a pre-encoded TensorDataset."""
    loader = DataLoader(encoded_ds, batch_size=64, shuffle=False)
    scores, targets = [], []
    model.eval()
    for z, y in loader:
        scores.append(torch.sigmoid(model(z.to(device))).cpu())
        targets.append(y)
    return auroc(torch.cat(scores), torch.cat(targets))


def main() -> None:
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}\n")

    # --- Build the FFDS splits ONCE (paths don't depend on the extractor) -
    print("Building FF++ splits + Celeb-DF test set...")
    train_ffds = FFDS(*build_ffds_split(
        FFPP_ROOT, "train.json", n_videos_per_class=N_VIDEOS_PER_CLASS_TRAIN,
        n_frames_per_video=N_FRAMES_PER_VIDEO, manipulation=MANIPULATION, seed=SEED))
    val_ffds = FFDS(*build_ffds_split(
        FFPP_ROOT, "val.json", n_videos_per_class=N_VIDEOS_PER_CLASS_VAL,
        n_frames_per_video=N_FRAMES_PER_VIDEO, manipulation=MANIPULATION, seed=SEED))
    test_ffds = FFDS(*build_ffds_split(
        FFPP_ROOT, "test.json", n_videos_per_class=None,
        n_frames_per_video=N_FRAMES_PER_VIDEO, manipulation=MANIPULATION, seed=SEED))
    celeb_ffds = FFDS(*build_celebdf_test_split(
        CELEBDF_ROOT, n_frames_per_video=N_FRAMES_PER_VIDEO, seed=SEED))
    print(f"  train={len(train_ffds)} val={len(val_ffds)} "
          f"ffpp_test={len(test_ffds)} celebdf_test={len(celeb_ffds)}\n")

    results: dict[str, dict[str, float]] = {}

    for name, (EncoderCls, dim) in EXTRACTORS.items():
        print("=" * 60)
        print(f"EXTRACTOR: {name}  (feature dim = {dim})")
        print("=" * 60)
        encoder = EncoderCls().to(DEVICE)
        tag = f"{name}_{MANIPULATION}_{N_FRAMES_PER_VIDEO}f_seed{SEED}"

        train_ds = encode_or_load(train_ffds, encoder, DEVICE, f"train_{tag}")
        val_ds = encode_or_load(val_ffds, encoder, DEVICE, f"val_{tag}")
        test_ds = encode_or_load(test_ffds, encoder, DEVICE, f"test_{tag}")
        celeb_ds = encode_or_load(celeb_ffds, encoder, DEVICE, f"celebdf_{tag}")

        # Train the shared MLP head.
        train_classifier(train_ds, val_ds, input_dim=dim,
                          ckpt_name=f"best_{name}", curves_name=f"curves_{name}")

        # Reload the BEST checkpoint and evaluate.
        model = MLPClassifier(input_dim=dim).to(DEVICE)
        model.load_state_dict(torch.load(OUT_DIR / f"best_{name}.pt", map_location=DEVICE))
        ffpp_auc = evaluate(model, test_ds, DEVICE)
        celeb_auc = evaluate(model, celeb_ds, DEVICE)
        results[name] = {"ffpp_test": ffpp_auc, "celebdf": celeb_auc}
        print(f"\n[{name}] FF++ test AUC = {ffpp_auc:.4f} | Celeb-DF AUC = {celeb_auc:.4f}\n")

    # --- Final comparison table -------------------------------------------
    header = f"{'Extractor':<12}{'FF++ test AUC':>16}{'Celeb-DF AUC':>16}"
    lines = ["=" * len(header), "COMPARISON — LDM encoder vs ResNet-50", "=" * len(header),
             header, "-" * len(header)]
    for name, r in results.items():
        lines.append(f"{name:<12}{r['ffpp_test']:>16.4f}{r['celebdf']:>16.4f}")
    table = "\n".join(lines)
    print("\n" + table)

    out_path = OUT_DIR / "comparison.txt"
    out_path.write_text(table + "\n")
    print(f"\nSaved comparison table to {out_path}")


if __name__ == "__main__":
    main()
