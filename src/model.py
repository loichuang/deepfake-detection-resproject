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


class MLPClassifier(nn.Module):
    """3-layer MLP head on the flattened latent.

    Architecture:
        flatten(z) [in_channels * spatial^2]   (16384 for 4x64x64)
          -> Linear -> BatchNorm -> ReLU -> Dropout      (layer 1)
          -> Linear -> BatchNorm -> ReLU -> Dropout      (layer 2)
          -> Linear -> logit                             (layer 3, output)

    The output is a single real-valued logit per sample (not yet a probability;
    apply sigmoid for that). We use BCEWithLogitsLoss downstream, which expects
    raw logits for numerical stability.

    Parameters
    ----------
    in_channels : int
        Latent channel count (4 for SD 1.5).
    spatial : int
        Latent spatial size (64 for a 512x512 input image).
    hidden : tuple[int, int]
        Sizes of the two hidden layers. Default (512, 128).
    dropout : float
        Dropout probability after each hidden activation. High by default
        (0.3) because the input is very high-dimensional relative to the
        number of training samples.
    """

    def __init__(
        self,
        in_channels: int = 4,
        spatial: int = 64,
        hidden: tuple[int, int] = (512, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_features = in_channels * spatial * spatial  # 16384
        h1, h2 = hidden

        self.net = nn.Sequential(
            nn.Flatten(),               # (B, 4, 64, 64) -> (B, 16384)
            nn.Linear(in_features, h1),  # layer 1
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),           # layer 2
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),            # layer 3 (output logit)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Latent (B, 4, 64, 64) -> logit (B,)."""
        return self.net(z).squeeze(-1)
