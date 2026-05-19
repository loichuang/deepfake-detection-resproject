"""FaceForensics++ frame-level Dataset.

This implementation follows the structure sketched by the project supervisor
on the whiteboard: a PyTorch Dataset that lists image paths + labels and
returns (preprocessed_tensor, label) tuples ready for training.

Each item is a single frame extracted from FF++ (already pre-cropped on the
EURECOM blutch cluster under `c23/frames/`). Aggregation back to video level
will happen later (feature F7).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FFDS(Dataset):
    """FaceForensics frame Dataset.

    Parameters
    ----------
    list_of_paths : Sequence[str | Path]
        Absolute paths to image files (PNG/JPG faces extracted from videos).
    list_of_labels : Sequence[int]
        Binary labels parallel to list_of_paths.
        Convention: 0 = real, 1 = fake (manipulated).
    image_size : int, default 512
        Target resolution for the VAE input. SD 1.5 expects 512x512.
    """

    def __init__(
        self,
        list_of_paths: Sequence[str | Path],
        list_of_labels: Sequence[int],
        image_size: int = 512,
    ) -> None:
        assert len(list_of_paths) == len(list_of_labels), (
            "list_of_paths and list_of_labels must have the same length."
        )
        self.list_of_paths = [str(p) for p in list_of_paths]
        self.list_of_labels = [int(lab) for lab in list_of_labels]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.list_of_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.list_of_paths[idx]

        # WARNING: cv2.imread returns BGR, not RGB. Forgetting to convert
        # swaps red and blue channels and severely degrades the VAE output.
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to the VAE's expected input resolution.
        img_resized = cv2.resize(
            img_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # Normalise pixel values to [-1, 1] (Stable Diffusion convention).
        img_normalised = img_resized.astype(np.float32) / 127.5 - 1.0

        # Reshape (H, W, C) -> (C, H, W) for PyTorch convolutions.
        tensor = torch.from_numpy(img_normalised).permute(2, 0, 1).contiguous()

        label = torch.tensor(self.list_of_labels[idx], dtype=torch.float32)
        return tensor, label


# ---------------------------------------------------------------------------
# Helpers — parse the official Rössler 2019 splits and sample frames.
# ---------------------------------------------------------------------------


def list_video_dirs_from_split(
    ffpp_root: str | Path,
    split_json: str,
    manipulation: str = "Deepfakes",
    compression: str = "c23",
) -> tuple[list[Path], list[Path]]:
    """Parse a Rössler 2019 split file and return (real, fake) video folders.

    Each split JSON contains a list of identity pairs ``[[A, B], ...]``.
    For each pair four video folders exist on disk:

    - real  :  ``original_sequences/youtube/{c}/frames/{A}``  and  ``.../{B}``
    - fake  :  ``manipulated_sequences/{manip}/{c}/frames/{A}_{B}``
                                            and       ``.../{B}_{A}``

    The two returned lists are NOT paired index-by-index: they are independent
    collections of video folders for each class.
    """
    ffpp_root = Path(ffpp_root)
    pairs = json.loads((ffpp_root / split_json).read_text())

    real_root = ffpp_root / "original_sequences" / "youtube" / compression / "frames"
    fake_root = ffpp_root / "manipulated_sequences" / manipulation / compression / "frames"

    real_dirs: list[Path] = []
    fake_dirs: list[Path] = []
    for a, b in pairs:
        for vid in (a, b):
            candidate = real_root / vid
            if candidate.exists():
                real_dirs.append(candidate)
        for vid in (f"{a}_{b}", f"{b}_{a}"):
            candidate = fake_root / vid
            if candidate.exists():
                fake_dirs.append(candidate)
    return real_dirs, fake_dirs


def sample_one_frame_per_video(
    video_dirs: Sequence[Path],
    rng: random.Random,
) -> list[Path]:
    """For each video folder, pick a single frame at random.

    Frames are expected to be ``.png`` files inside ``video_dir``.
    Folders containing zero frames are silently skipped.
    """
    chosen: list[Path] = []
    for video_dir in video_dirs:
        frames = sorted(video_dir.glob("*.png"))
        if frames:
            chosen.append(rng.choice(frames))
    return chosen


def build_ffds_split(
    ffpp_root: str | Path,
    split_json: str,
    n_videos_per_class: int = 100,
    manipulation: str = "Deepfakes",
    compression: str = "c23",
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """End-to-end helper: from an official split file to (paths, labels).

    Reads the split JSON, limits the number of videos per class for tractable
    experiments, samples exactly one frame per video, and concatenates the
    result into the (paths, labels) pair expected by ``FFDS``.

    Parameters
    ----------
    n_videos_per_class : int
        Maximum number of real videos AND maximum number of fake videos kept.
        Default 100 yields a fast tractable dataset for a first experiment.
    seed : int
        Random seed for reproducibility (used for both video sub-sampling
        and frame selection inside each video).
    """
    rng = random.Random(seed)
    real_dirs, fake_dirs = list_video_dirs_from_split(
        ffpp_root, split_json, manipulation, compression
    )

    real_dirs = rng.sample(real_dirs, k=min(n_videos_per_class, len(real_dirs)))
    fake_dirs = rng.sample(fake_dirs, k=min(n_videos_per_class, len(fake_dirs)))

    real_paths = sample_one_frame_per_video(real_dirs, rng)
    fake_paths = sample_one_frame_per_video(fake_dirs, rng)

    paths: list[str] = [str(p) for p in real_paths] + [str(p) for p in fake_paths]
    labels: list[int] = [0] * len(real_paths) + [1] * len(fake_paths)
    return paths, labels
