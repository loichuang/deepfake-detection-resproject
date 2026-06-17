"""Wenyi — Test ResNet fine-tuné : FF++ in-domain + Celeb-DF cross-dataset.

Charge le checkpoint results/wenyi_resnet_finetune.pt.

Run depuis la racine du projet :
    python src/wenyi/test_resnet_finetune.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

from src.wenyi.train_resnet_finetune import FFDS, ThreeLayerMLP, evaluate_model

RESULTS_DIR     = Path(__file__).resolve().parent.parent.parent / "results"
CHECKPOINT_PATH = str(RESULTS_DIR / "wenyi_resnet_finetune.pt")
FFPP_ROOT       = "/medias/db/deepfakes/Faceforensics/"
CELEBDF_ROOT    = "/medias/db/deepfakes/Celeb-DF-v2/"


# ==========================================
# Dataset Celeb-DF
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


def evaluate_and_save(encoder, model, dataloader, criterion, device, scores_path, labels_path):
    """Évalue et sauvegarde les scores bruts par frame en .npy pour les courbes ROC."""
    encoder.eval(); model.eval()
    running_loss, all_labels, all_preds = 0.0, [], []
    running_corrects, total_samples = 0, 0
    with torch.no_grad():
        for crop, label in dataloader:
            crop, label = crop.to(device), label.to(device)
            features = encoder(crop)
            pred = model(features)
            loss = criterion(pred, label)
            running_loss += loss.item()
            probs = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            labels_np = label.detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels_np)
            running_corrects += np.sum((probs > 0.5).astype(float) == labels_np)
            total_samples += len(labels_np)
    scores_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    np.save(scores_path, scores_arr)
    np.save(labels_path, labels_arr)
    print(f"  Scores sauvegardés : {scores_path}")
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels_arr, scores_arr) if len(set(all_labels)) > 1 else 0.5
    return running_loss / len(dataloader), running_corrects / total_samples, auc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========Using device: {device}==========")

    print("Loading FFDS TEST Dataset...")
    ffds_loader = DataLoader(
        FFDS(split='test', num_frames=3), batch_size=64, num_workers=4, shuffle=False
    )
    print("Loading Celeb-DF-v2 Dataset...")
    celeb_dataset = CelebDFDataset(num_frames=2)
    if len(celeb_dataset) == 0:
        print("ERROR: No Celeb-DF images loaded!")
        exit()
    celeb_loader = DataLoader(celeb_dataset, batch_size=64, num_workers=4, shuffle=False)

    # Charger encoder + MLP
    encoder = models.resnet50(pretrained=False)
    encoder.fc = nn.Identity()
    encoder = encoder.to(device)
    model = ThreeLayerMLP(input_dim=2048).to(device)

    print(f"Loading weights from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])

    criterion = nn.BCEWithLogitsLoss()

    print("\nSTARTING FF++ IN-DOMAIN EVALUATION...")
    ffds_loss, ffds_acc, ffds_auc = evaluate_and_save(
        encoder, model, ffds_loader, criterion, device,
        scores_path=str(RESULTS_DIR / "wenyi_resnet_scores_ffpp.npy"),
        labels_path=str(RESULTS_DIR / "wenyi_resnet_labels_ffpp.npy"),
    )

    print("\nSTARTING CELEB-DF CROSS-DATASET EVALUATION...")
    celeb_loss, celeb_acc, celeb_auc = evaluate_and_save(
        encoder, model, celeb_loader, criterion, device,
        scores_path=str(RESULTS_DIR / "wenyi_resnet_scores_celeb.npy"),
        labels_path=str(RESULTS_DIR / "wenyi_resnet_labels_celeb.npy"),
    )

    print(f"\n==========FINAL RESULTS (ResNet fine-tuné)==========")
    print(f"[FF++ In-Domain]  Loss: {ffds_loss:.4f} | Acc: {ffds_acc:.4f} | AUC: {ffds_auc:.4f}")
    print(f"[Celeb-DF Cross]  Loss: {celeb_loss:.4f} | Acc: {celeb_acc:.4f} | AUC: {celeb_auc:.4f}")
    print("====================================================")
