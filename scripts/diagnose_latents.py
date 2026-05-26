"""Diagnostic — why does the FF++ model collapse to ~0.5 AUC on Celeb-DF?

Runs three checks to distinguish a genuine domain gap from a preprocessing /
distribution problem:

  1. Latent distribution stats (mean / std / L2 norm) — FF++ vs Celeb-DF.
     If they differ a lot, the MLP receives out-of-distribution inputs on
     Celeb-DF and a normalization step would help.

  2. Centroid distance real<->fake in each dataset.
     Tells us whether ANY linear signal separates the two classes.

  3. A quick IN-DOMAIN linear probe trained on Celeb-DF itself.
     - high AUC  -> Celeb-DF latents ARE separable; the problem is purely the
                    FF++ -> Celeb-DF transfer (domain gap). Fix with multi-
                    manipulation training / augmentation / VIB.
     - ~0.5 AUC  -> latents are not separable in-domain either: investigate
                    preprocessing or the encoder's behaviour on Celeb-DF.

Run from the project root on blutch:
    python scripts/diagnose_latents.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_celebdf_test_split
from src.model import FrozenVAEEncoder
from src.train import auroc, encode_dataset, CACHE_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CELEBDF_ROOT = "/medias/db/deepfakes/Celeb-DF-v2"
FFPP_CACHE = CACHE_DIR / "train_Deepfakes_5f_seed42.pt"


def print_stats(latents: torch.Tensor, name: str) -> None:
    flat = latents.flatten(1)
    print(f"[{name}]  N={flat.shape[0]}  dim={flat.shape[1]}")
    print(f"    mean    : {flat.mean():+.4f}")
    print(f"    std     : {flat.std():.4f}")
    print(f"    min/max : {flat.min():+.3f} / {flat.max():+.3f}")
    print(f"    L2 norm : {flat.norm(dim=1).mean():.2f} (mean per sample)")


def centroid_distance(latents: torch.Tensor, labels: torch.Tensor, name: str) -> None:
    flat = latents.flatten(1)
    real_c = flat[labels == 0].mean(0)
    fake_c = flat[labels == 1].mean(0)
    print(f"    [{name}] real<->fake centroid distance: {(real_c - fake_c).norm():.2f}")


def quick_linear_probe(latents: torch.Tensor, labels: torch.Tensor, epochs: int = 60) -> float:
    """Train a linear probe in-domain (80/20 split) and return validation AUC."""
    flat = latents.flatten(1).to(DEVICE)
    y = labels.float().to(DEVICE)
    n = flat.shape[0]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)
    n_tr = int(0.8 * n)
    tr, va = idx[:n_tr], idx[n_tr:]

    probe = torch.nn.Linear(flat.shape[1], 1).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = crit(probe(flat[tr]).squeeze(-1), y[tr])
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(probe(flat[va]).squeeze(-1)).cpu()
    return auroc(val_scores, y[va].cpu())


def main() -> None:
    print(f"Device: {DEVICE}\n")

    # --- 1. FF++ latents (from training cache) ----------------------------
    print("=" * 60)
    print("CHECK 1+2 — distribution & separability")
    print("=" * 60)
    if not FFPP_CACHE.exists():
        raise FileNotFoundError(
            f"FF++ cache not found: {FFPP_CACHE}. Run `python src/train.py` first."
        )
    obj = torch.load(FFPP_CACHE)
    ffpp_lat, ffpp_lab = obj["latents"], obj["labels"]
    print_stats(ffpp_lat, "FF++ train")
    centroid_distance(ffpp_lat, ffpp_lab, "FF++")

    # --- Celeb-DF latents (encode a sample) -------------------------------
    print()
    paths, labels = build_celebdf_test_split(CELEBDF_ROOT, n_frames_per_video=5, seed=42)
    celeb_ffds = FFDS(paths, labels)
    encoder = FrozenVAEEncoder().to(DEVICE)
    print(f"Encoding {len(celeb_ffds)} Celeb-DF frames...")
    celeb_ds = encode_dataset(celeb_ffds, encoder, DEVICE)
    celeb_lat, celeb_lab = celeb_ds.tensors
    print_stats(celeb_lat, "Celeb-DF test")
    centroid_distance(celeb_lat, celeb_lab, "Celeb-DF")

    # --- 3. In-domain separability test -----------------------------------
    print()
    print("=" * 60)
    print("CHECK 3 — in-domain linear probe (separability ceiling)")
    print("=" * 60)
    ffpp_probe_auc = quick_linear_probe(ffpp_lat, ffpp_lab)
    celeb_probe_auc = quick_linear_probe(celeb_lat, celeb_lab)
    print(f"FF++   in-domain linear-probe AUC : {ffpp_probe_auc:.4f}")
    print(f"Celeb  in-domain linear-probe AUC : {celeb_probe_auc:.4f}")

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if celeb_probe_auc > 0.75:
        print("Celeb-DF latents ARE separable in-domain.")
        print("=> The 0.53 cross-dataset AUC is a genuine DOMAIN GAP.")
        print("   Fix: multi-manipulation training, augmentation, or VIB.")
    elif celeb_probe_auc < 0.6:
        print("Celeb-DF latents are NOT separable even in-domain.")
        print("=> Suspect a preprocessing/distribution issue, not just a gap.")
        print("   Check frame resolution, normalization, encoder behaviour.")
    else:
        print("Celeb-DF latents are weakly separable in-domain.")
        print("=> Mixed: partly domain gap, partly limited signal.")


if __name__ == "__main__":
    main()
