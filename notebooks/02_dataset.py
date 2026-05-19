"""F3 — Build a small latent dataset for binary deepfake-like classification.

Strategy (AEROBLADE-style proxy, Ricker et al. CVPR 2024)
---------------------------------------------------------
Real latents : z_real = E(x)              for x = natural face image
Fake latents : z_fake = E(D(E(x)))        same image passed through the VAE

The label y in {0, 1} indicates whether the latent comes from a never-encoded
image (y=0) or from an image that went through one VAE round-trip (y=1).
This is a working proxy for true deepfake detection until FF++ is available.

Outputs
-------
- data/latents/real/<i>.pt    — 200 tensors of shape (4, 64, 64)
- data/latents/fake/<i>.pt    — 200 tensors of shape (4, 64, 64)
- data/manifest.csv           — columns: path, label
"""

# %% [markdown]
# # Cell 0 — Imports and device

# %%
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
LATENTS_DIR = DATA_DIR / "latents"
(LATENTS_DIR / "real").mkdir(parents=True, exist_ok=True)
(LATENTS_DIR / "fake").mkdir(parents=True, exist_ok=True)

print(f"Device      : {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir    : {DATA_DIR}")


# %% [markdown]
# # Cell 1 — Download 200 CelebA face images via HuggingFace `datasets`
#
# The `huggan/CelebA-faces` dataset hosts ~30k aligned face crops from CelebA.
# We stream only the first 200, no need to download the entire archive.

# %%
from datasets import load_dataset

N_IMAGES = 200

print("Loading huggan/CelebA-faces (streaming first 200 images)...")
ds = load_dataset("huggan/CelebA-faces", split=f"train[:{N_IMAGES}]")
print(f"Loaded {len(ds)} images. First sample keys: {list(ds[0].keys())}")
ds[0]["image"]  # display the first face inline


# %% [markdown]
# # Cell 2 — Load the SD 1.5 VAE (same one as F1)

# %%
from diffusers import AutoencoderKL

SCALING_FACTOR = 0.18215
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
vae.eval()
print(f"VAE loaded on {DEVICE}")


# %% [markdown]
# # Cell 3 — Helper functions for image <-> tensor conversion
#
# Same convention as F1: PIL image in [0, 255] → tensor (1, 3, 512, 512) in [-1, 1].

# %%
def image_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    """PIL Image -> tensor (1, 3, size, size) in [-1, 1] on DEVICE."""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    """Inverse of image_to_tensor for visualisation. Assumes (1, 3, H, W)."""
    arr = ((x.clamp(-1, 1)[0].cpu().permute(1, 2, 0) + 1) * 127.5).numpy().astype(np.uint8)
    return Image.fromarray(arr)


# %% [markdown]
# # Cell 4 — Encode the 200 real images into real latents
#
# For each natural image x:    z_real = E(x) * scaling_factor
# Save as data/latents/real/<i>.pt

# %%
print(f"Encoding {N_IMAGES} real images...")
for i, sample in enumerate(tqdm(ds)):
    x = image_to_tensor(sample["image"])
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean * SCALING_FACTOR
    torch.save(z[0].cpu(), LATENTS_DIR / "real" / f"{i:04d}.pt")
print(f"Real latents saved to {LATENTS_DIR / 'real'}")


# %% [markdown]
# # Cell 5 — Produce the fake latents via encode → decode → encode
#
# For each natural image x:
#     z_intermediate = E(x) * s              # first encoding
#     x_reconstructed = D(z_intermediate / s) # decode back to image
#     z_fake = E(x_reconstructed) * s        # re-encode the reconstruction
#
# z_fake is the latent of an image that has been through the VAE once.

# %%
print(f"Generating {N_IMAGES} fake latents (encode -> decode -> encode)...")
for i, sample in enumerate(tqdm(ds)):
    x = image_to_tensor(sample["image"])
    with torch.no_grad():
        z_intermediate = vae.encode(x).latent_dist.mean * SCALING_FACTOR
        x_reconstructed = vae.decode(z_intermediate / SCALING_FACTOR).sample
        z_fake = vae.encode(x_reconstructed).latent_dist.mean * SCALING_FACTOR
    torch.save(z_fake[0].cpu(), LATENTS_DIR / "fake" / f"{i:04d}.pt")
print(f"Fake latents saved to {LATENTS_DIR / 'fake'}")


# %% [markdown]
# # Cell 6 — Build the manifest CSV
#
# Format expected by F4 (and FF++ later): columns `path, label`.
# label = 0 for real, 1 for fake.

# %%
manifest_path = DATA_DIR / "manifest.csv"
rows = []
for f in sorted((LATENTS_DIR / "real").glob("*.pt")):
    rows.append((str(f.relative_to(PROJECT_ROOT)), 0))
for f in sorted((LATENTS_DIR / "fake").glob("*.pt")):
    rows.append((str(f.relative_to(PROJECT_ROOT)), 1))

with open(manifest_path, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["path", "label"])
    writer.writerows(rows)

print(f"Manifest written: {manifest_path}")
print(f"Total entries  : {len(rows)} (real: {sum(1 for _, l in rows if l == 0)}, "
      f"fake: {sum(1 for _, l in rows if l == 1)})")


# %% [markdown]
# # Cell 7 — Visual sanity check : one real vs one fake reconstruction
#
# Show the original image and its VAE reconstruction side by side.
# The reconstruction is what generated the "fake" latent.

# %%
sample = ds[0]
x = image_to_tensor(sample["image"])
with torch.no_grad():
    z = vae.encode(x).latent_dist.mean * SCALING_FACTOR
    x_reconstructed = vae.decode(z / SCALING_FACTOR).sample

img_original = tensor_to_image(x)
img_reconstructed = tensor_to_image(x_reconstructed)

comparison = Image.new("RGB", (img_original.width * 2, img_original.height))
comparison.paste(img_original, (0, 0))
comparison.paste(img_reconstructed, (img_original.width, 0))
comparison


# %% [markdown]
# # F3 — Summary
#
# What we just built:
# - 200 real latents from natural CelebA faces.
# - 200 fake latents from the same faces, after one VAE round-trip
#   (the AEROBLADE-style proxy for deepfake detection).
# - A manifest.csv that pairs each .pt file with its binary label.
#
# The hypothesis we're going to test in F4 is:
# the linear projection of `flatten(z)` is enough to separate the two
# populations, i.e. the VAE leaves a detectable signature.
# Spoiler: it should — Ricker et al. observed AUC > 0.99 on similar setups.
# If our F4 confirms this, the pipeline is sane; we can then swap the data
# for the real FF++ deepfakes when access returns.
