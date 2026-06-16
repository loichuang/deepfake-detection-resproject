# Latent DeepFake Detection

**EURECOM — Spring 2026 — Semester Research Project**
Supervised by Alexandre Libourel.

---

## Research question

The VAE encoder of Stable Diffusion 1.4 projects an image into a latent representation `z`. We test whether `z` alone — without any diffusion step, without fine-tuning — is sufficient to discriminate real images from deepfakes.

> **H₁** — A frozen LDM encoder, used as a fixed feature extractor with a lightweight MLP head, outperforms a frozen ResNet-50 baseline on the same task.

This is a deliberate simplification of Wang & Kalogeiton (ECCV 2024), who use the full classifier-free guidance residual across multiple denoising steps. If H₁ holds, it shows that the diffusion process itself contributes little to detection — the discriminative signal is already in the encoder.

---

## Results

### Main ablation — frozen vs. fine-tuned × backbone

| Backbone | Frozen | Fine-tuned |
|---|---|---|
| ResNet-50 | AUC 0.78 / 0.54 | AUC 0.97 / 0.70 |
| LDM encoder (SD 1.4) | **AUC 0.87 / 0.52** | AUC ? / ? *(in progress)* |

Format: `FF++ in-domain AUC / Celeb-DF cross-dataset AUC`.
Frozen LDM confidence intervals (95% video-level cluster bootstrap): FF++ **0.87 [0.835, 0.903]**, Celeb-DF **0.52 [0.471, 0.559]**.
Frozen ResNet: FF++ **0.78 [0.734, 0.822]**, Celeb-DF **0.54 [0.490, 0.577]**.
In-domain CIs are disjoint → LDM superiority is statistically significant.

### Ablation on diffusion depth (Wenyi Zeng)

| Feature | Resolution | FF++ AUC | Celeb-DF AUC |
|---|---|---|---|
| `z` only (encoder, ours) | 512×512 | **0.87** | 0.52 |
| `\|z − ẑ\|` DDIM 20 steps (TRY 2) | 512×512 | 0.79 | **0.60** |
| `z − ẑ` PNDM 1 step (TRY 1) | 224×224 | 0.56 | 0.50 |

The encoder-only approach dominates in-domain. Adding the diffusion residual marginally improves cross-dataset generalization at ~20× inference cost.

---

## Project structure

```
.
├── src/
│   ├── dataset.py              # FFDS + Celeb-DF dataset classes
│   ├── model.py                # FrozenVAEEncoder, TrainableVAEEncoder,
│   │                           #   ResNetEncoder, MLPClassifier
│   ├── train.py                # Training — frozen LDM encoder
│   ├── train_resnet.py         # Training — frozen ResNet baseline
│   ├── train_ldm_finetune.py   # Training — fine-tunable LDM encoder
│   ├── eval.py                 # Evaluation on FF++ test split
│   │                           #   (ENCODER_TYPE: ldm | resnet | ldm_finetune)
│   ├── eval_celebdf.py         # Cross-dataset evaluation on Celeb-DF-v2
│   └── wenyi/                  # Wenyi Zeng's parallel experiments
│       ├── train_ldm_try1.py   # LDM residual, 1 step PNDM, 224×224
│       ├── train_ldm_try2.py   # LDM residual, DDIM 20 steps, 512×512
│       ├── train_resnet_finetune.py  # ResNet fine-tuned end-to-end
│       ├── test_ldm_try1.py
│       ├── test_ldm_try2.py
│       └── test_resnet_finetune.py
├── scripts/
│   ├── make_figures.py         # Bootstrap CI + ROC curves → results/*.png
│   ├── run_comparison.py       # Side-by-side LDM vs ResNet evaluation
│   └── diagnose_latents.py     # Latent space diagnostic (linear probe)
├── notebooks/                  # Exploratory notebooks (jupytext .py sources)
├── presentation/
│   ├── midterm_defense.tex     # Beamer slides (metropolis theme)
│   └── midterm_defense.pdf     # Compiled PDF
├── results/                    # Figures, tables, checkpoints (*.pt gitignored)
├── requirements-blutch.txt     # Python dependencies for blutch cluster
└── .gitignore
```

---

## How to run

All commands from the project root on the **blutch cluster**.

### 1. Training

```bash
# Frozen LDM encoder (main experiment)
python src/train.py

# Frozen ResNet baseline
python src/train_resnet.py

# Fine-tunable LDM encoder (ablation)
python src/train_ldm_finetune.py

# Wenyi — LDM residual DDIM 20 steps
python src/wenyi/train_ldm_try2.py
```

### 2. Evaluation

Edit `ENCODER_TYPE` in the eval scripts (`"ldm"`, `"resnet"`, or `"ldm_finetune"`), then:

```bash
# FF++ in-domain
python src/eval.py

# Celeb-DF cross-dataset
python src/eval_celebdf.py

# Wenyi's models
python src/wenyi/test_ldm_try2.py
python src/wenyi/test_resnet_finetune.py
```

### 3. Figures (bootstrap CI + ROC curves)

```bash
python scripts/make_figures.py
```

Outputs to `results/`: `roc_curves.png`, `auc_bars_ci.png`, `comparison_ci.txt`.

---

## Datasets

| Dataset | Path (blutch) | Usage |
|---|---|---|
| FaceForensics++ (c23, DeepFakes) | `/medias/db/deepfakes/Faceforensics/` | Train / val / test (official Rössler 2019 splits) |
| Celeb-DF-v2 | `/medias/db/deepfakes/Celeb-DF-v2/` | Cross-dataset evaluation only |

Identity-disjoint splits: no identity appears in both train and test sets.
Sampling: 5 frames per video (evenly spaced). Evaluation metric: AUC with 95% video-level cluster bootstrap CI (2000 iterations).

---

## References

- Wang & Kalogeiton, *Exposing the Fakes: Leveraging Diffusion Models for Deepfake Detection*, ECCV 2024.
- Ricker et al., *AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error*, CVPR 2024.
- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
- Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020.
- Yan et al., *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*, NeurIPS 2023.
