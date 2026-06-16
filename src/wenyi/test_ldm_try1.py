"""Wenyi — Test TRY 1 : FF++ in-domain + Celeb-DF cross-dataset.

Charge le checkpoint results/wenyi_ldm_try1.pt (ou le .pth original de Wenyi).

Run depuis la racine du projet :
    python src/wenyi/test_ldm_try1.py

Pour utiliser le checkpoint original de Wenyi (best_ffds_model_ldm_try_1.pth),
modifier CHECKPOINT_PATH ci-dessous.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/medias/db/ImagingSecurity_misc/zengw/hf_cache"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

from src.wenyi.train_ldm_try1 import FFDS, ThreeLayerMLP, evaluate_model
from diffusers import StableDiffusionPipeline

RESULTS_DIR     = Path(__file__).resolve().parent.parent.parent / "results"
CHECKPOINT_PATH = str(RESULTS_DIR / "wenyi_ldm_try1.pt")
FFPP_ROOT       = "/medias/db/deepfakes/Faceforensics/"
CELEBDF_ROOT    = "/medias/db/deepfakes/Celeb-DF-v2/"


# ==========================================
# Dataset Celeb-DF (ALL videos, comme Wenyi)
# ==========================================
class CelebDFDataset(Dataset):
    def __init__(self, root_dir=CELEBDF_ROOT, num_frames=3):
        self.image_paths = []
        self.labels = []

        for label, subdir in [(1, "Celeb-synthesis"), (0, "Celeb-real"), (0, "YouTube-real")]:
            img_root = os.path.join(root_dir, subdir, "images")
            if not os.path.exists(img_root):
                continue
            for vid_name in sorted(os.listdir(img_root)):
                vid_path = os.path.join(img_root, vid_name)
                if os.path.isdir(vid_path):
                    frames = sorted([f for f in os.listdir(vid_path)
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                    if frames:
                        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                        for idx in indices:
                            self.image_paths.append(os.path.join(vid_path, frames[idx]))
                            self.labels.append(label)

        print(f"Celeb-DF-v2 Dataset loaded. Total frames: {len(self.image_paths)}")
        self.mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img_cropped = self.mtcnn(img)
        if img_cropped is None:
            fallback = transforms.Compose([
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = fallback(img)
        else:
            img_tensor = img_cropped
        return img_tensor, torch.tensor([self.labels[idx]], dtype=torch.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========Using device: {device}==========")

    print("Loading FFDS TEST Dataset...")
    ffds_loader = DataLoader(
        FFDS(split='test', num_frames=3), batch_size=32, num_workers=2, shuffle=False
    )
    print("Loading Celeb-DF-v2 Dataset...")
    celeb_dataset = CelebDFDataset(num_frames=3)
    if len(celeb_dataset) == 0:
        print("ERROR: No Celeb-DF images loaded!")
        exit()
    celeb_loader = DataLoader(celeb_dataset, batch_size=32, num_workers=2, shuffle=False)

    print("Loading LDM (Stable Diffusion v1.5)...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    model = ThreeLayerMLP(input_dim=3136).to(device)
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    print("\nSTARTING FF++ IN-DOMAIN EVALUATION...")
    ffds_loss, ffds_acc, ffds_auc = evaluate_model(pipeline, model, ffds_loader, criterion, device)

    print("\nSTARTING CELEB-DF CROSS-DATASET EVALUATION...")
    celeb_loss, celeb_acc, celeb_auc = evaluate_model(pipeline, model, celeb_loader, criterion, device)

    print(f"\n==========FINAL RESULTS (TRY 1)==========")
    print(f"[FF++ In-Domain]  Loss: {ffds_loss:.4f} | Acc: {ffds_acc:.4f} | AUC: {ffds_auc:.4f}")
    print(f"[Celeb-DF Cross]  Loss: {celeb_loss:.4f} | Acc: {celeb_acc:.4f} | AUC: {celeb_auc:.4f}")
    print("=========================================")
