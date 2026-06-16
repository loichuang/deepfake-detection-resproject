"""Wenyi — TRY 2 : LDM résidu, DDIM 20 steps, 512×512, CLIP embedding réel.

Pipeline :
    image (512×512) → VAE encode → z (mean, déterministe)
    → add_noise(z, t_start=timesteps[10]) → DDIM denoise 20 steps → z_hat
    → feature = |z - z_hat|  [absolu, 16384 dims]
    → MLP → real/fake

Résultats obtenus : FF++ AUC ~0.79, Celeb-DF AUC ~0.60.
Checkpoint sauvé : results/wenyi_ldm_try2.pt

Run depuis la racine du projet :
    python src/wenyi/train_ldm_try2.py
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
from diffusers import StableDiffusionPipeline, DDIMScheduler

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = str(RESULTS_DIR / "wenyi_ldm_try2.pt")

FFPP_ROOT = "/medias/db/deepfakes/Faceforensics/"


# ==========================================
# 1. Dataset avec cache MTCNN 512×512
# ==========================================
class FFDS(Dataset):
    def __init__(self, root_dir=FFPP_ROOT, split='train', num_frames=3):
        self.image_paths = []
        self.labels = []

        fake_dir = os.path.join(root_dir, "manipulated_sequences/Deepfakes/c23/frames/")
        real_dir = os.path.join(root_dir, "original_sequences/youtube/c23/frames/")

        self.cache_dir = os.path.join(root_dir, f"mtcnn_cache_512_pt/{split}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_paths = []

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
                return
            for vid_name in os.listdir(directory):
                vid_base = vid_name.split('.')[0]
                parts = vid_base.split('_')
                # Strict ID matching (évite la fuite entre splits)
                if not all(p in self.valid_ids for p in parts):
                    continue
                vid_path = os.path.join(directory, vid_name)
                if os.path.isdir(vid_path):
                    frames = sorted(os.listdir(vid_path))
                    if len(frames) > 0:
                        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                        for idx in indices:
                            frame_name = frames[idx]
                            self.image_paths.append(os.path.join(vid_path, frame_name))
                            self.cache_paths.append(
                                os.path.join(self.cache_dir, f"{vid_base}_{frame_name}.pt")
                            )
                            self.labels.append(label)

        sample_frames(fake_dir, label=1)
        sample_frames(real_dir, label=0)
        print(f"{split.upper()} Dataset loaded. Total frames: {len(self.image_paths)}")

        self.mtcnn = MTCNN(image_size=512, margin=40, keep_all=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cache_path = self.cache_paths[idx]
        if os.path.exists(cache_path):
            img_tensor = torch.load(cache_path)
        else:
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
            torch.save(img_tensor, cache_path)
        return img_tensor, torch.tensor([self.labels[idx]], dtype=torch.float32)


# ==========================================
# 2. MLP (input_dim=16384 : 4×64×64 pour 512×512)
# ==========================================
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim=16384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(512, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# ==========================================
# 3. Feature extractor TRY 2
#    z (mean) → add_noise(t_start) → DDIM denoise → z_hat → |z - z_hat|
# ==========================================
def extract_ldm_features(pipeline, images, base_empty_embedding, device, start_step=10):
    with torch.no_grad():
        images = images.to(device, dtype=torch.float16)
        batch_size = images.shape[0]
        current_text_embeds = base_empty_embedding.repeat(batch_size, 1, 1)

        # Encodage déterministe (mean, pas sample)
        z = pipeline.vae.encode(images).latent_dist.mean * pipeline.vae.config.scaling_factor
        t_start = pipeline.scheduler.timesteps[start_step]

        noise = torch.randn_like(z)
        z_noisy = pipeline.scheduler.add_noise(
            z, noise, torch.tensor([t_start] * batch_size, device=device)
        )
        z_hat = z_noisy

        # Débruitage DDIM complet depuis start_step
        for t in pipeline.scheduler.timesteps[start_step:]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise_pred = pipeline.unet(
                z_hat, t_batch, encoder_hidden_states=current_text_embeds
            ).sample
            z_hat = pipeline.scheduler.step(noise_pred, t, z_hat).prev_sample

        # Résidu absolu (différence clé vs TRY 1 qui est signé)
        features = torch.abs(z - z_hat)
        return features.view(features.size(0), -1).float()


# ==========================================
# 4. Evaluation
# ==========================================
def evaluate_model(pipeline, model, dataloader, base_empty_embedding, criterion, device):
    model.eval()
    running_loss, all_labels, all_preds = 0.0, [], []
    running_corrects, total_samples = 0, 0
    with torch.no_grad():
        for crop, label in dataloader:
            crop, label = crop.to(device), label.to(device)
            latent = extract_ldm_features(pipeline, crop, base_empty_embedding, device)
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
    train_loader  = DataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=4, num_workers=2, shuffle=False)

    print("\nLoading Latent Diffusion Model (Stable Diffusion v1.5)...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=20, device=device)
    print(f"DDIM Timesteps: {pipeline.scheduler.timesteps.tolist()}")

    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    print("LDM loaded and successfully FROZEN.")

    # CLIP text encoder avec prompt vide (différence clé vs TRY 1)
    text_inputs = pipeline.tokenizer(
        [""], padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        base_empty_embedding = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]

    model     = ThreeLayerMLP(input_dim=16384).to(device)
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
            latent = extract_ldm_features(pipeline, crop, base_empty_embedding, device)
            pred = model(latent)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            probs = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            labels_np = label.detach().cpu().numpy().flatten()
            train_scores.extend(probs)
            train_labels_all.extend(labels_np)
            if batch_idx % 10 == 0:
                try:
                    batch_auc = roc_auc_score(labels_np, probs)
                except ValueError:
                    batch_auc = 0.5
                print(f"Train Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - AUC: {batch_auc:.4f}")

        train_auc = roc_auc_score(train_labels_all, train_scores)
        print("Evaluating on Validation Set...")
        val_loss, val_acc, val_auc = evaluate_model(
            pipeline, model, val_loader, base_empty_embedding, criterion, device
        )
        print(f"==========Epoch {epoch+1} Summary==========")
        print(f"Train Loss: {sum(losses)/len(losses):.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
            print(f"※ Best saved → {CHECKPOINT_PATH} (val AUC {val_auc:.4f})")
