"""Model components for encoder-only deepfake detection.

Two pieces, matching the supervisor's whiteboard:
  - FrozenVAEEncoder : the SD 1.5 VAE encoder, frozen (no gradients).
                       Maps an image (B, 3, 512, 512) to a latent (B, 4, 64, 64).
  - MLPClassifier    : a 3-layer MLP on the flattened latent -> binary logit.

Only the MLP is trained; the encoder is a fixed feature extractor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

# Canonical Stable Diffusion 1.5 latent scaling factor.
SCALING_FACTOR = 0.18215


class FrozenVAEEncoder(nn.Module):
    """Stable Diffusion 1.4 VAE encoder, frozen.

    We load the VAE from `CompVis/stable-diffusion-v1-4` (subfolder "vae"),
    which is EXACTLY the encoder used by Wang & Kalogeiton (2024) in their
    `gen_latent.py` (model == "sd1.4"). This guarantees strict methodological
    fidelity to the reference paper: same encoder weights, no ambiguity.

    All parameters have requires_grad=False and the module is kept in eval()
    mode, so no gradient ever flows back into the VAE. The forward pass is
    wrapped in torch.no_grad() to save memory.
    """

    def __init__(
        self,
        pretrained_name: str = "CompVis/stable-diffusion-v1-4",
        subfolder: str = "vae",
    ) -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_name, subfolder=subfolder)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image (B, 3, 512, 512) in [-1, 1] -> latent (B, 4, 64, 64)."""
        return self.vae.encode(x).latent_dist.mean * SCALING_FACTOR

    def train(self, mode: bool = True) -> "FrozenVAEEncoder":
        # Keep the VAE in eval() regardless of the parent's train/eval state.
        super().train(mode)
        self.vae.eval()
        return self


class ResNetEncoder(nn.Module):
    """ImageNet-pretrained ResNet-50, frozen, used as a feature extractor.

    Returns the 2048-dim global-average-pooled vector (the representation just
    before ResNet's classification head). This is the COMPARISON BASELINE
    against the LDM encoder: the downstream MLP is identical, only the feature
    extractor changes — so we can isolate whether the diffusion encoder
    captures manipulation traces better or worse than a standard ResNet.

    Input convention: images arrive as (B, 3, 512, 512) in [-1, 1] (the same
    FFDS output used by the VAE). We convert them to ImageNet normalization and
    resize to 224x224 (ResNet's native resolution) before extracting features.
    """

    OUTPUT_DIM = 2048

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Drop the final FC layer; keep everything up to the global avg pool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()

        # ImageNet normalization constants (registered as buffers so .to(device) moves them).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image (B, 3, 512, 512) in [-1, 1] -> features (B, 2048)."""
        x = (x + 1.0) / 2.0                                    # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std                         # ImageNet normalize
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        feat = self.features(x)                                # (B, 2048, 1, 1)
        return feat.flatten(1)                                 # (B, 2048)

    def train(self, mode: bool = True) -> "ResNetEncoder":
        super().train(mode)
        self.features.eval()
        return self


class MLPClassifier(nn.Module):
    """3-layer MLP head on a flattened feature vector.

    Architecture (identical to the comparison baseline used by the teammate,
    so that the LDM-vs-ResNet comparison is fair — only the upstream extractor
    differs):

        flatten(x) [input_dim]
          -> Linear(512) -> ReLU -> Dropout
          -> Linear(128) -> ReLU
          -> Linear(1)   -> logit

    The same head serves both feature extractors:
      - LDM encoder latent : input_dim = 4 * 64 * 64 = 16384
      - ResNet-50 features  : input_dim = 2048

    The output is a single raw logit per sample (apply sigmoid for a
    probability). Used with BCEWithLogitsLoss downstream.

    Parameters
    ----------
    input_dim : int
        Size of the flattened feature vector fed to the MLP.
    dropout : float
        Dropout probability after the first hidden activation (default 0.3).
    """

    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # accepts (B, C, H, W) or (B, D)
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature tensor -> logit (B,)."""
        return self.net(x).squeeze(-1)
