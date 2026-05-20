"""Quick standalone test of FFDS + split helpers on the real FF++ data.

Run from the project root on blutch:
    python scripts/test_dataset.py

Validates:
  1. The official split JSON is parsed correctly.
  2. Video paths resolve on disk and one frame per video is sampled.
  3. The FFDS Dataset loads a sample with the expected shape/range.
  4. The sampling is reproducible (same seed -> same paths).
"""

import sys
from pathlib import Path

# Make `src` importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split

FFPP_ROOT = "/medias/db/deepfakes/Faceforensics"
N_PER_CLASS = 10

print("=" * 60)
print("STEP 1 — Build a tiny split (10 videos per class)")
print("=" * 60)
paths, labels = build_ffds_split(
    ffpp_root=FFPP_ROOT,
    split_json="train.json",
    n_videos_per_class=N_PER_CLASS,
    seed=42,
)
print(f"Total samples : {len(paths)}")
print(f"  real (label 0): {labels.count(0)}")
print(f"  fake (label 1): {labels.count(1)}")
print(f"First real path : {paths[0]}")
print(f"First fake path : {paths[-1]}")

print()
print("=" * 60)
print("STEP 2 — Load one sample through the FFDS Dataset")
print("=" * 60)
ds = FFDS(paths, labels)
tensor, label = ds[0]
print(f"len(ds)      : {len(ds)}")
print(f"tensor shape : {tuple(tensor.shape)}   (expected (3, 512, 512))")
print(f"tensor dtype : {tensor.dtype}")
print(f"tensor range : [{tensor.min():.3f}, {tensor.max():.3f}]   (expected ~[-1, 1])")
print(f"label        : {label.item()}")

print()
print("=" * 60)
print("STEP 3 — Reproducibility check (same seed -> same paths)")
print("=" * 60)
paths2, _ = build_ffds_split(
    ffpp_root=FFPP_ROOT,
    split_json="train.json",
    n_videos_per_class=N_PER_CLASS,
    seed=42,
)
print(f"Reproducible : {'OK' if paths == paths2 else 'FAIL'}")

print()
print("All checks done.")
