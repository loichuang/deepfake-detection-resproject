"""Wenyi — TRY 1 : LDM résidu, 1 step PNDM, 224×224, zero-embedding.

Pipeline :
    image (224×224) → VAE encode → z
    → add_noise(z, t=100) → UNet 1 step → z_hat
    → feature = (z - z_hat)  [signé, 3136 dims]
    → MLP → real/fake

Résultats obtenus : FF++ AUC ~0.56, Celeb-DF AUC ~0.50 (proche du hasard).
Checkpoint sauvé : results/wenyi_ldm_try1.pt

Run depuis la racine du projet :
    python src/wenyi/train_ldm_try1.py
"""

import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "/medias/db/ImagingSecurity_misc/zengw/hf_cache"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
from sklearn.metrics import roc_auc_score
from diffusers import StableDiffusionPipeline

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = str(RESULTS_DIR / "wenyi_ldm_try1.pt")

FFPP_ROOT = "/medias/db/deepfakes/Faceforensics/"


# ==========================================
# 1. Dataset
# ==========================================
class FFDS(Dataset):
    def __init__(self, root_dir=FFPP_ROOT, split='train', num_frames=3):
        self.image_paths = []
        self.labels = []

        fake_dir = os.path.join(root_dir, "manipulated_sequences/Deepfakes/c23/frames/")
        real_dir = os.path.join(root_dir, "original_sequences/youtube/c23/frames/")

        split_file = os.path.join(root_dir, f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Cannot find official split file: {split_file}")

        with open(split_file, 'r') as f:
            split_data = json.load(f)

        self.valid_ids = set()
        for item in split_data:
            if isinstance(item, list):
                self.valid_ids.update(item)
            else:
                self.valid_ids.add(item)

        def sample_frames(directory, label):
            if not os.path.exists(directory):
                print(f"[Warning] Path does not exist: {directory}")
                return
            for vid_name in os.listdir(directory):
                if not any(v_id in vid_name for v_id in self.valid_ids):
                    continue
                vid_path = os.path.join(directory, vid_name)
                if os.path.isdir(vid_path):
                    frames = sorted(os.listdir(vid_path))
                    if len(frames) > 0:
                        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                        for idx in indices:
                            self.image_paths.append(os.path.join(vid_path, frames[idx]))
                            self.labels.append(label)

        sample_frames(fake_dir, label=1)
        sample_frames(real_dir, label=0)
        print(f"{split.upper()} Dataset loaded. Total frames: {len(self.image_paths)}")

        self.mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        img_cropped = self.mtcnn(img)
        if img_cropped is None:
            fallback = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = fallback(img)
        else:
            img_tensor = img_cropped
        return img_tensor, torch.tensor([label], dtype=torch.float32)


# ==========================================
# 2. MLP (input_dim=3136 : 4×28×28 pour 224×224)
# ==========================================
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim=3136):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# ==========================================
# 3. Feature extractor TRY 1
#    z → add_noise(t=100) → UNet 1 step → z_hat → diff = z - z_hat
# ==========================================
def extract_ldm_features(pipeline, images, device):
    with torch.no_grad():
        images = images.to(device, dtype=torch.float16)
        z = pipeline.vae.encode(images).latent_dist.sample() * 0.18215
        t = torch.tensor([100] * images.shape[0], device=device)
        noise = torch.randn_like(z)
        z_noisy = pipeline.scheduler.add_noise(z, noise, t)
        # Zero-embedding (pas de CLIP réel — différence clé vs TRY 2)
        encoder_hidden_states = torch.zeros(
            (images.shape[0], 77, 768), device=device, dtype=torch.float16
        )
        noise_pred = pipeline.unet(z_noisy, t, encoder_hidden_states=encoder_hidden_states).sample
        alpha_t = pipeline.scheduler.alphas_cumprod[t[0]]
        z_hat = (z_noisy - (1 - alpha_t) ** 0.5 * noise_pred) / alpha_t ** 0.5
        diff = z - z_hat
        return diff.view(diff.size(0), -1).float()


# ==========================================
# 4. Evaluation
# ==========================================
def evaluate_model(pipeline, model, dataloader, criterion, device):
    model.eval()
    running_loss, all_labels, all_preds = 0.0, [], []
    running_corrects, total_samples = 0, 0
    with torch.no_grad():
        for crop, label in dataloader:
            crop, label = crop.to(device), label.to(device)
            latent = extract_ldm_features(pipeline, crop, device)
            pred = model(latent)
            loss = criterion(pred, label)
            running_loss += loss.item()
            probs = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            labels_np = label.detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels_np)
            running_corrects += np.sum((probs > 0.5).astype(float) == labels_np)
            total_samples += len(labels_np)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return running_loss / len(dataloader), running_corrects / total_samples, auc


# ==========================================
# 5. Training loop
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========Using device: {device}==========")

    train_dataset = FFDS(split='train', num_frames=3)
    val_dataset   = FFDS(split='val',   num_frames=3)
    train_loader  = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=32, num_workers=2, shuffle=False)

    print("\nLoading Latent Diffusion Model (Stable Diffusion v1.5)...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    print("LDM loaded and successfully FROZEN!")

    model     = ThreeLayerMLP(input_dim=3136).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_auc = 0.0
    for epoch in range(10):
        model.train()
        train_scores, train_labels_all, losses = [], [], []
        print(f"\n==========Epoch {epoch+1}/10==========")
        for batch_idx, (crop, label) in enumerate(train_loader):
            crop, label = crop.to(device), label.to(device)
            optimizer.zero_grad()
            latent = extract_ldm_features(pipeline, crop, device)
            pred = model(latent)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            probs = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            labels_np = label.detach().cpu().numpy().flatten()
            train_scores.extend(probs)
            train_labels_all.extend(labels_np)
            if batch_idx % 20 == 0:
                try:
                    batch_auc = roc_auc_score(labels_np, probs)
                except ValueError:
                    batch_auc = 0.5
                print(f"Train Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - AUC: {batch_auc:.4f}")

        train_auc = roc_auc_score(train_labels_all, train_scores)
        print("Evaluating on Validation Set...")
        val_loss, val_acc, val_auc = evaluate_model(pipeline, model, val_loader, criterion, device)
        print(f"==========Epoch {epoch+1} Summary==========")
        print(f"Train Loss: {sum(losses)/len(losses):.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
            print(f"※ Best saved → {CHECKPOINT_PATH} (val AUC {val_auc:.4f})")
