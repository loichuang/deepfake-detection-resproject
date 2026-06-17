"""FaceForensics++ frame-level Dataset.

This implementation follows the structure sketched by the project supervisor
on the whiteboard: a PyTorch Dataset that lists image paths + labels and
returns (preprocessed_tensor, label) tuples ready for training.

Each item is a single frame extracted from FF++ (already pre-cropped on the
EURECOM blutch cluster under `c23/frames/`). Aggregation back to video level
will happen later (feature F7).
"""


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


def sample_n_frames_per_video(
    video_dirs: Sequence[Path],
    rng: random.Random,
    n_frames: int = 5,
) -> list[Path]:
    """For each video folder, pick ``n_frames`` distinct frames at random.

    Frames are expected to be ``.png`` files inside ``video_dir``.
    If a folder has fewer than ``n_frames`` frames, all of them are taken.
    Folders containing zero frames are silently skipped.

    Sampling several frames per video (instead of one) is the main lever
    against overfitting: it multiplies the dataset size while keeping the
    identity-disjoint split guarantee (all frames of a video share its split).
    """
    chosen: list[Path] = []
    for video_dir in video_dirs:
        frames = sorted(video_dir.glob("*.png"))
        if not frames:
            continue
        k = min(n_frames, len(frames))
        chosen.extend(rng.sample(frames, k=k))
    return chosen


def build_ffds_split(
    ffpp_root: str | Path,
    split_json: str,
    n_videos_per_class: int | None = None,
    n_frames_per_video: int = 5,
    manipulation: str = "Deepfakes",
    compression: str = "c23",
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """End-to-end helper: from an official split file to (paths, labels).

    Reads the split JSON, optionally limits the number of videos per class,
    samples ``n_frames_per_video`` frames per video, and concatenates the
    result into the (paths, labels) pair expected by ``FFDS``.

    Parameters
    ----------
    n_videos_per_class : int | None
        Maximum number of real videos AND fake videos kept. ``None`` (default)
        means use ALL videos in the split. Set to a small int for quick tests.
    n_frames_per_video : int
        Number of distinct random frames sampled per video (default 5). The
        primary lever against overfitting — more frames means more samples.
    seed : int
        Random seed for reproducibility (used for both video sub-sampling
        and frame selection inside each video).
    """
    rng = random.Random(seed)
    real_dirs, fake_dirs = list_video_dirs_from_split(
        ffpp_root, split_json, manipulation, compression
    )

    if n_videos_per_class is not None:
        real_dirs = rng.sample(real_dirs, k=min(n_videos_per_class, len(real_dirs)))
        fake_dirs = rng.sample(fake_dirs, k=min(n_videos_per_class, len(fake_dirs)))

    real_paths = sample_n_frames_per_video(real_dirs, rng, n_frames_per_video)
    fake_paths = sample_n_frames_per_video(fake_dirs, rng, n_frames_per_video)

    paths: list[str] = [str(p) for p in real_paths] + [str(p) for p in fake_paths]
    labels: list[int] = [0] * len(real_paths) + [1] * len(fake_paths)
    return paths, labels


def build_celebdf_test_split(
    celebdf_root: str | Path,
    n_frames_per_video: int = 5,
    testing_list: str = "List_of_testing_videos.txt",
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Build (paths, labels) for the OFFICIAL Celeb-DF-v2 test set.

    Reads ``List_of_testing_videos.txt`` to know which videos belong to the
    official test set, then samples frames from the pre-cropped face images.

    The label is derived from the FOLDER name, NOT from the file's own label
    column. Celeb-DF uses the opposite convention (1 = real, 0 = fake) in that
    file, so we ignore it and rely on the directory:
        - "Celeb-synthesis"            -> fake (label 1)
        - "Celeb-real" / "YouTube-real" -> real (label 0)
    This is robust regardless of the txt convention.

    Pre-cropped frames live at:  ``<category>/images/<video_id>/<frame>.png``
    while the txt lists paths as: ``<category>/<video_id>.mp4``.
    """
    celebdf_root = Path(celebdf_root)
    rng = random.Random(seed)
    lines = (celebdf_root / testing_list).read_text().strip().splitlines()

    paths: list[str] = []
    labels: list[int] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        _, rel_path = parts                      # ignore the txt label column
        category = rel_path.split("/")[0]        # e.g. "YouTube-real"
        video_id = Path(rel_path).stem           # e.g. "00170" or "id0_id1_0000"
        label = 1 if "synthesis" in category.lower() else 0

        frames_dir = celebdf_root / category / "images" / video_id
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            continue
        k = min(n_frames_per_video, len(frames))
        for f in rng.sample(frames, k=k):
            paths.append(str(f))
            labels.append(label)
    return paths, labels
