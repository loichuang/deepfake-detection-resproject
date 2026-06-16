"""Wenyi — Test TRY 2 : FF++ in-domain + Celeb-DF cross-dataset.

Charge le checkpoint results/wenyi_ldm_try2.pt (ou le .pth original de Wenyi).

Run depuis la racine du projet :
    python src/wenyi/test_ldm_try2.py
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

from src.wenyi.train_ldm_try2 import FFDS, ThreeLayerMLP, evaluate_model, extract_ldm_features
from diffusers import StableDiffusionPipeline, DDIMScheduler

RESULTS_DIR     = Path(__file__).resolve().parent.parent.parent / "results"
CHECKPOINT_PATH = str(RESULTS_DIR / "wenyi_ldm_try2.pt")
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
        self.mtcnn = MTCNN(image_size=512, margin=40, keep_all=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img_cropped = self.mtcnn(img)
        if img_cropped is None:
            fallback = transforms.Compose([
                transforms.Resize(512), transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        FFDS(split='test', num_frames=3), batch_size=4, num_workers=2, shuffle=False
    )
    print("Loading Celeb-DF-v2 Dataset...")
    celeb_dataset = CelebDFDataset(num_frames=3)
    if len(celeb_dataset) == 0:
        print("ERROR: No Celeb-DF images loaded!")
        exit()
    celeb_loader = DataLoader(celeb_dataset, batch_size=4, num_workers=2, shuffle=False)

    print("Loading LDM (Stable Diffusion v1.5)...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=20, device=device)
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    text_inputs = pipeline.tokenizer(
        [""], padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        base_empty_embedding = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]

    model = ThreeLayerMLP(input_dim=16384).to(device)
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    print("\nSTARTING FF++ IN-DOMAIN EVALUATION...")
    ffds_loss, ffds_acc, ffds_auc = evaluate_model(
        pipeline, model, ffds_loader, base_empty_embedding, criterion, device
    )
    print("\nSTARTING CELEB-DF CROSS-DATASET EVALUATION...")
    celeb_loss, celeb_acc, celeb_auc = evaluate_model(
        pipeline, model, celeb_loader, base_empty_embedding, criterion, device
    )

    print(f"\n==========FINAL RESULTS (TRY 2)==========")
    print(f"[FF++ In-Domain]  Loss: {ffds_loss:.4f} | Acc: {ffds_acc:.4f} | AUC: {ffds_auc:.4f}")
    print(f"[Celeb-DF Cross]  Loss: {celeb_loss:.4f} | Acc: {celeb_acc:.4f} | AUC: {celeb_auc:.4f}")
    print("=========================================")
