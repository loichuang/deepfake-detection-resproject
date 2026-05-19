# Latent_DeepFake_Project

**EURECOM — Spring 2026 — Semester Research Project**
*Leveraging diffusion models for deepfake detection — encoder-only variant.*

## Research question

The encoder `ℰ` of Stable Diffusion 1.5 projects an input image into a latent representation `z₀`. We test the following hypothesis:

> **H₁** — `z₀` alone (without any diffusion step, without classifier-free guidance) is sufficient to discriminate real images from deepfakes.

This is a deliberate simplification of Wang & Kalogeiton (ECCV 2024) who use the full classifier-free guidance residual on top of the encoder. If H₁ holds, it would show that the diffusion process itself contributes little to detection — a scientifically informative result.

## Roadmap — 8 features

1. **F1** — manipulate the VAE encoder: load it, encode an image, observe `z₀`, decode it back.
2. **F2** — face detection and cropping. *Skipped: pre-extracted faces are already provided on the EURECOM blutch cluster under `c23/frames/`.*
3. **F3** — build a minimal latent dataset (real + fake).
4. **F4** — train a linear classifier on top of `z₀` and measure AUC.
5. **F5** — cross-dataset evaluation FF++ → Celeb-DF-v2.
6. **F6** — more expressive heads: MLP, ResNet.
7. **F7** — video-level aggregation of frame-level predictions.
8. **F8** — secondary objective: Variational Information Bottleneck or encoder + decoder.

Each feature corresponds to a standalone notebook in `notebooks/`.

## Datasets

- **FaceForensics++** — `/medias/db/deepfakes/Faceforensics/` (blutch). Pre-cropped faces in `c23/frames/`. Official Rössler 2019 splits in `train.json` / `val.json` / `test.json`.
- **Celeb-DF-v2** — `/medias/db/deepfakes/Celeb-DF-v2/` (blutch). Used for cross-dataset evaluation only.

## References

- [1] Yan et al., *DeepfakeBench*, NeurIPS 2023.
- [2] Wang & Kalogeiton, *Exposing the Fakes: The Case Against Real Images*, ECCV 2024. Reference repo at `../DiffusionImplicitDetection/`.
- [3] Ricker et al., *AEROBLADE*, CVPR 2024.
- [4] Rössler et al., *FaceForensics++*, ICCV 2019.
- [5] Li et al., *Celeb-DF*, CVPR 2020.
