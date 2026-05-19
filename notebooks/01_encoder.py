# %%
"""F1 — First contact with the Stable Diffusion 1.5 VAE encoder.

Pedagogical goal
----------------
Understand concretely what `ℰ` does: take an image (3 RGB channels, 512x512
pixels) and project it into a latent z₀ (4 channels, 64x64 values). Verify
that we can decode the latent back to an image with `𝒟`.

How to run
----------
In VSCode, click "Run Cell" above each `# %%` block, or use Shift+Enter.
Each cell produces a visible output (a print, a shape, an image).

Code attribution
----------------
The core encoding pattern in cell 4 is adapted from Wang & Kalogeiton (2024),
file `DiffusionImplicitDetection/data_preparation/inversion.py`, lines 210-222.
Their original code:

    latents = self.model.vae.encode(image)["latent_dist"].mean
    latents = latents * 0.18215

Requirements
------------
    pip install -r ../requirements.txt
"""

# %% [markdown]
# # Cell 0 — Imports and device selection
#
# `torch.backends.mps` exposes the GPU of Apple Silicon Macs (M1, M2, M3).
# On a Linux GPU machine like blutch, the equivalent is `cuda`. The fallback
# is `cpu`, which works but is roughly 20x slower for the VAE.

# %%
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device in use   : {DEVICE}")
print(f"PyTorch version : {torch.__version__}")


# %% [markdown]
# # Cell 1 — Load a local image
#
# We load a face image from the local `data/` folder. To get this image,
# we ran the following command once in the terminal (curl handles SSL
# certificates via the system store, bypassing Python's urllib issues):
#
#     curl -o data/sample.jpg \
#         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/in_paint/celeba_hq_256.png"

# %%
IMG_PATH = "../data/sample.jpg"

img = Image.open(IMG_PATH).convert("RGB").resize((512, 512))
print(f"Image loaded: size={img.size}, mode={img.mode}")
img  # In a VSCode notebook cell, this displays the image inline


# %% [markdown]
# # Cell 2 — Preprocess the image
#
# The VAE expects a tensor of shape `(B, 3, H, W)` with values in `[-1, 1]`
# (Stable Diffusion convention). Our PIL image is in `[0, 255]`, so we:
#   1. cast to float32 numpy
#   2. divide by 127.5 and subtract 1 → range [-1, 1]
#   3. permute axes (H, W, C) → (C, H, W) and add the batch dim

# %%

# %%
arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 512, 512)
x = x.to(DEVICE)

print(f"Image tensor shape   : {tuple(x.shape)}")
print(f"Min / max values     : {x.min().item():.3f} / {x.max().item():.3f}")
print(f"Mean / std           : {x.mean().item():.3f} / {x.std().item():.3f}")


# %%

# %%

# %% [markdown]
# # Cell 3 — Load the Stable Diffusion 1.5 VAE
#
# `AutoencoderKL` is the diffusers class that bundles both the encoder
# `ℰ` (method `.encode`) and the decoder `𝒟` (method `.decode`). The
# first call downloads ~335 MB from HuggingFace; subsequent calls hit
# the local cache (`~/.cache/huggingface/`).

# %%

# %%
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
vae.eval()  # inference mode: disables dropout and switches BatchNorm

n_params = sum(p.numel() for p in vae.parameters())
print(f"VAE loaded: {type(vae).__name__}")
print(f"Total parameters    : {n_params:,}")
print(f"  encoder params    : {sum(p.numel() for p in vae.encoder.parameters()):,}")
print(f"  decoder params    : {sum(p.numel() for p in vae.decoder.parameters()):,}")


# %%

# %% [markdown]
# # Cell 4 — Encode the image to the latent z₀
#
# `vae.encode(x)` returns an `AutoencoderKLOutput` object containing a
# `DiagonalGaussianDistribution`. We extract its mean `μ(x)` via
# `.latent_dist.mean`.
#
# IMPORTANT: we multiply by 0.18215. This is the canonical scaling factor
# of Stable Diffusion 1.5, calibrated so that the latent variance is ~1
# (useful for the UNet, which we don't use here, but the literature
# always applies it).
#
# This is the exact operation from `inversion.py:220-221` in the
# Wang & Kalogeiton repository — we adopt their convention verbatim.

# %%

# %%
SCALING_FACTOR = 0.18215

with torch.no_grad():
    z = vae.encode(x).latent_dist.mean * SCALING_FACTOR

print(f"Latent z₀ shape      : {tuple(z.shape)}")
print(f"Total dimensions     : {z.numel()} values per image")
print(f"Range                : {z.min().item():+.3f} to {z.max().item():+.3f}")
print(f"Mean / std           : {z.mean().item():+.3f} / {z.std().item():.3f}")


# %%

# %% [markdown]
# # Cell 5 — Decode the latent and measure reconstruction quality
#
# `vae.decode(z / SCALING_FACTOR)` applies the decoder network.
# We divide by the scaling factor to undo the multiplication done at
# encoding (otherwise the decoder sees values outside its training range
# and reconstruction is poor).

# %%

# %%
with torch.no_grad():
    x_hat = vae.decode(z / SCALING_FACTOR).sample

# Convert [-1, 1] → [0, 255] uint8 for display
x_hat_clamped = x_hat.clamp(-1, 1)
arr_hat = ((x_hat_clamped[0].cpu().permute(1, 2, 0) + 1) * 127.5).numpy().astype(np.uint8)
img_reconstructed = Image.fromarray(arr_hat)

# Reconstruction error in [-1, 1] pixel space
mse = torch.nn.functional.mse_loss(x_hat_clamped, x).item()
psnr = 10 * np.log10(4.0 / mse)  # signal range = 2, so max² = 4
print(f"Reconstruction MSE   : {mse:.5f}")
print(f"Estimated PSNR       : {psnr:.2f} dB")
img_reconstructed


# %%

# %% [markdown]
# # Cell 6 — Side-by-side comparison (original vs reconstruction)

# %%

# %%
comparison = Image.new("RGB", (img.width * 2, img.height))
comparison.paste(img, (0, 0))
comparison.paste(img_reconstructed, (img.width, 0))
comparison


# %%

# %% [markdown]
# # F1 — Summary
#
# What we just did:
# - Loaded the pre-trained SD 1.5 VAE.
# - Encoded a 512×512×3 image into a latent of shape 4×64×64.
# - This projection compresses information: from 786 432 values
#   (3×512×512) down to 16 384 values (4×64×64), a ~48x reduction.
# - Verified that the decoder reconstructs an image very close to the
#   original (PSNR ~25 dB), which confirms that z₀ preserves the bulk
#   of the visual information.
#
# It is exactly this preserved information in z₀ that we will exploit to
# discriminate real from fake images. Our research hypothesis H₁ can now
# be restated more precisely: "the subset of information preserved in z₀
# already carries the deepfake signature".
#
# Next step: F3 — build a small latent dataset (real + fake) and use it
# to train a binary classifier.

# %%
