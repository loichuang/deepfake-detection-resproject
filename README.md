# encoder-only

**EURECOM — Spring 2026 — Semester Research Project**
*Leverage diffusion models to detect deepfakes — encoder-only variant.*

## Question de recherche

L'encodeur `ℰ` de Stable Diffusion 1.5 projette une image dans un latent `z₀`. On teste l'hypothèse :

> **H₁** — `z₀` seul (sans étape de diffusion, sans CFG) suffit à séparer images réelles et fakes.

## Plan en 8 features

1. **F1** — manipuler l'encoder VAE : charger, encoder, décoder une image, observer `z₀`.
2. **F2** — détecter et cropper un visage (à confirmer : peut-être déjà fait sur blutch).
3. **F3** — construire un mini-dataset de latents (real + fake).
4. **F4** — entraîner un classifieur linéaire et mesurer l'AUC.
5. **F5** — évaluer en cross-dataset FF++ → Celeb-DF-v2.
6. **F6** — variantes de tête : MLP, ResNet.
7. **F7** — agrégation video-level des prédictions.
8. **F8** — objectif secondaire : VIB ou encoder + decoder.

Chaque feature donne lieu à un notebook indépendant dans `notebooks/`.
