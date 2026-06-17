"""Génère toutes les figures de comparaison — 6 modèles, progression en 3 étapes.

Prérequis : avoir lancé les scripts de test qui sauvegardent les .npy
  python3 src/eval.py --encoder ldm           → via inférence dans ce script
  python3 src/eval.py --encoder resnet         → idem
  python3 src/eval.py --encoder ldm_finetune   → idem
  python3 src/wenyi/test_ldm_try1.py          → results/wenyi_try1_scores_*.npy
  python3 src/wenyi/test_ldm_try2.py          → results/wenyi_try2_scores_*.npy
  python3 src/wenyi/test_resnet_finetune.py   → results/wenyi_resnet_scores_*.npy

Outputs (dans results/) :
  roc_fig1_frozen.png        — Figure 1 : LDM figé + ResNet figé
  roc_fig2_denoising.png     — Figure 2 : + Wenyi PNDM 1 step + DDIM 20 steps
  roc_fig3_finetune.png      — Figure 3 : + LDM fine-tuné + Wenyi ResNet FT
  auc_bars_all.png           — Bar chart 6 modèles avec IC 95 %
  comparison_all.txt         — Tableau texte

Usage (depuis la racine sur blutch) :
    python3 scripts/make_figures_all.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FFDS, build_ffds_split, build_celebdf_test_split
from src.model import FrozenVAEEncoder, TrainableVAEEncoder, ResNetEncoder, MLPClassifier
from src.train import encode_dataset, LDM_DIM, FFPP_ROOT, MANIPULATION, SEED

CELEBDF_ROOT = "/medias/db/deepfakes/Celeb-DF-v2"
N_FRAMES     = 5
N_BOOT       = 2000
RESULTS_DIR  = Path(__file__).resolve().parent.parent / "results"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Palette & métadonnées ────────────────────────────────────────────────────
NAVY = "#1E2761"

MODELS = {
    # name : (label, couleur, groupe)
    "ldm":           ("LDM encoder (frozen)",      "#2A9D8F", "frozen"),
    "resnet":        ("ResNet-50 (frozen)",         "#9AA0B4", "frozen"),
    "wenyi_try1":    ("PNDM 1-step (Try 1)",        "#457B9D", "denoising"),
    "wenyi_try2":    ("DDIM 20-step (Try 2)",       "#1D3557", "denoising"),
    "ldm_finetune":  ("LDM fine-tuned (SD 1.4)",   "#E76F51", "finetune"),
    "wenyi_resnet":  ("ResNet-50 fine-tuned",       "#A8DADC", "finetune"),
}

# Groupes progressifs pour les 3 figures
GROUPS = [
    ("frozen",    "Frozen encoders"),
    ("denoising", "Denoising residual"),
    ("finetune",  "Fine-tuning"),
]


# ─── Métriques (numpy, sans sklearn) ──────────────────────────────────────────
def auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0: return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def roc_curve_np(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    l = labels[order].astype(float)
    P, N = l.sum(), len(l) - l.sum()
    tpr = np.concatenate([[0.0], np.cumsum(l) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - l) / N])
    return fpr, tpr


def bootstrap_ci(scores, labels, video_ids, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, v in enumerate(video_ids):
        groups.setdefault(v, []).append(i)
    vids = list(groups.values())
    aucs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(vids), len(vids))
        idx = np.concatenate([vids[c] for c in chosen])
        a = auc_np(scores[idx], labels[idx])
        if not np.isnan(a):
            aucs.append(a)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def video_ids_from_paths(paths):
    return [Path(p).parent.name for p in paths]


# ─── Inférence pour nos 3 modèles ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, ds) -> np.ndarray:
    model.eval()
    out = []
    for z, _ in DataLoader(ds, batch_size=64, shuffle=False):
        out.append(torch.sigmoid(model(z.to(DEVICE))).cpu())
    return torch.cat(out).numpy().ravel()


def load_our_model(name):
    if name == "ldm":
        encoder = FrozenVAEEncoder().to(DEVICE)
        mlp = MLPClassifier(input_dim=LDM_DIM).to(DEVICE)
        ckpt = torch.load(RESULTS_DIR / "best_ldm.pt", map_location=DEVICE)
        mlp.load_state_dict(ckpt)
    elif name == "ldm_finetune":
        encoder = TrainableVAEEncoder().to(DEVICE)
        mlp = MLPClassifier(input_dim=LDM_DIM).to(DEVICE)
        ckpt = torch.load(RESULTS_DIR / "best_ldm_finetune.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        mlp.load_state_dict(ckpt["mlp_state_dict"])
    elif name == "resnet":
        encoder = ResNetEncoder().to(DEVICE)
        mlp = MLPClassifier(input_dim=ResNetEncoder.OUTPUT_DIM).to(DEVICE)
        ckpt = torch.load(RESULTS_DIR / "best_resnet.pt", map_location=DEVICE)
        mlp.load_state_dict(ckpt)
    encoder.eval(); mlp.eval()
    return encoder, mlp


# ─── Chargement scores .npy (Wenyi) ───────────────────────────────────────────
WENYI_NPY = {
    "wenyi_try1":   ("wenyi_try1_scores_ffpp.npy",   "wenyi_try1_labels_ffpp.npy",
                     "wenyi_try1_scores_celeb.npy",  "wenyi_try1_labels_celeb.npy"),
    "wenyi_try2":   ("wenyi_try2_scores_ffpp.npy",   "wenyi_try2_labels_ffpp.npy",
                     "wenyi_try2_scores_celeb.npy",  "wenyi_try2_labels_celeb.npy"),
    "wenyi_resnet": ("wenyi_resnet_scores_ffpp.npy", "wenyi_resnet_labels_ffpp.npy",
                     "wenyi_resnet_scores_celeb.npy","wenyi_resnet_labels_celeb.npy"),
}


# ─── Figure helper ────────────────────────────────────────────────────────────
def plot_roc(ax, results_so_far, dname, title):
    for name, r in results_so_far.items():
        if dname not in r:
            continue
        d = r[dname]
        label_str = MODELS[name][0]
        ax.plot(d["fpr"], d["tpr"], color=MODELS[name][1], lw=2.2,
                label=f"{label_str}  (AUC {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#C9CDD8", lw=1, ls="--")
    ax.set_title(title, color=NAVY, fontsize=12, pad=6)
    ax.set_xlabel("FPR", color="#33384D", fontsize=10)
    ax.set_ylabel("TPR", color="#33384D", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(color="#E2E8F0", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color("#C9CDD8")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}\n")

    # — Datasets FF++ et Celeb-DF —
    ffpp_paths, ffpp_labels = build_ffds_split(
        FFPP_ROOT, "test.json", n_videos_per_class=None,
        n_frames_per_video=N_FRAMES, manipulation=MANIPULATION, seed=SEED)
    celeb_paths, celeb_labels = build_celebdf_test_split(
        CELEBDF_ROOT, n_frames_per_video=N_FRAMES, seed=SEED)
    ffpp_vids  = video_ids_from_paths(ffpp_paths)
    celeb_vids = video_ids_from_paths(celeb_paths)

    datasets = {
        "FF++":     (FFDS(ffpp_paths, ffpp_labels),   np.array(ffpp_labels),  ffpp_vids),
        "Celeb-DF": (FFDS(celeb_paths, celeb_labels), np.array(celeb_labels), celeb_vids),
    }

    all_results = {}   # name → dname → {auc, lo, hi, fpr, tpr}

    # — Nos 3 modèles avec checkpoints —
    for name in ["ldm", "resnet", "ldm_finetune"]:
        ckpt_map = {"ldm": "best_ldm.pt", "resnet": "best_resnet.pt",
                    "ldm_finetune": "best_ldm_finetune.pt"}
        if not (RESULTS_DIR / ckpt_map[name]).exists():
            print(f"⚠ {ckpt_map[name]} absent — ignoré")
            continue
        print(f"[{MODELS[name][0]}]")
        encoder, mlp = load_our_model(name)
        all_results[name] = {}
        for dname, (ffds_obj, labels, vids) in datasets.items():
            enc_ds = encode_dataset(ffds_obj, encoder, DEVICE)
            scores = predict(mlp, enc_ds)
            auc    = auc_np(scores, labels)
            lo, hi = bootstrap_ci(scores, labels, vids)
            fpr, tpr = roc_curve_np(scores, labels)
            all_results[name][dname] = dict(auc=auc, lo=lo, hi=hi, fpr=fpr, tpr=tpr)
            print(f"  {dname:9s}  AUC {auc:.4f}  [{lo:.4f}, {hi:.4f}]")

    # — Wenyi : chargement des .npy —
    for name, (sf, lf, sc, lc) in WENYI_NPY.items():
        sf_path = RESULTS_DIR / sf; lf_path = RESULTS_DIR / lf
        sc_path = RESULTS_DIR / sc; lc_path = RESULTS_DIR / lc
        if not sf_path.exists():
            print(f"⚠ {sf} absent — {MODELS[name][0]} ignoré (relancer le test script)")
            continue
        print(f"[{MODELS[name][0]}]")
        all_results[name] = {}
        for dname, (sp, lp, vids_key) in [
            ("FF++",     (sf_path, lf_path, ffpp_vids)),
            ("Celeb-DF", (sc_path, lc_path, celeb_vids)),
        ]:
            if not sp.exists():
                continue
            scores = np.load(sp)
            labels = np.load(lp)
            # Pour Wenyi on utilise les video_ids des datasets officiels si taille compatible,
            # sinon on crée des video_ids fictifs (1 frame = 1 vidéo) pour le bootstrap.
            if len(scores) == len(vids_key):
                vids = vids_key
            else:
                vids = [str(i) for i in range(len(scores))]
            auc = auc_np(scores, labels)
            lo, hi = bootstrap_ci(scores, labels, vids)
            fpr, tpr = roc_curve_np(scores, labels)
            all_results[name][dname] = dict(auc=auc, lo=lo, hi=hi, fpr=fpr, tpr=tpr)
            print(f"  {dname:9s}  AUC {auc:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ─── 3 figures ROC progressives ───────────────────────────────────────────
    cumulative = {}
    for k, (group_key, group_label) in enumerate(GROUPS):
        # Ajoute les modèles de ce groupe
        for name, (label, color, grp) in MODELS.items():
            if grp == group_key and name in all_results:
                cumulative[name] = all_results[name]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, dname, subtitle in zip(
            axes,
            ["FF++", "Celeb-DF"],
            ["FF++ — in-domain", "Celeb-DF — cross-dataset"]
        ):
            plot_roc(ax, cumulative, dname, subtitle)

        # Légende du groupe courant dans le titre
        groups_shown = " + ".join(GROUPS[j][1] for j in range(k + 1))
        fig.suptitle(f"ROC curves — {groups_shown}", color=NAVY, fontsize=13, y=1.01)
        fig.tight_layout()
        out_path = RESULTS_DIR / f"roc_fig{k+1}_{group_key}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    # ─── Bar chart 6 modèles ──────────────────────────────────────────────────
    dsets = ["FF++", "Celeb-DF"]
    model_order = list(MODELS.keys())
    x = np.arange(len(dsets))
    n = len(model_order)
    width = 0.11
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for k, name in enumerate(model_order):
        label, color, _ = MODELS[name]
        aucs, errs_lo, errs_hi, valid_x = [], [], [], []
        for i, dname in enumerate(dsets):
            if name not in all_results or dname not in all_results[name]:
                continue
            r = all_results[name][dname]
            aucs.append(r["auc"])
            errs_lo.append(r["auc"] - r["lo"])
            errs_hi.append(r["hi"] - r["auc"])
            valid_x.append(x[i] + offsets[k])

        if not aucs:
            continue

        ax.bar(valid_x, aucs, width * 0.85, color=color, label=label,
               yerr=[errs_lo, errs_hi], capsize=4,
               error_kw=dict(ecolor="#33384D", lw=1.2))
        for xi, a in zip(valid_x, aucs):
            ax.text(xi, a + 0.022, f"{a:.3f}", ha="center", fontsize=7,
                    color=NAVY, fontweight="bold", rotation=90)

    ax.axhline(0.5, color="#C9CDD8", lw=1, ls=":", zorder=0)
    ax.text(0.98, 0.505, "chance", color="#9AA0B4", fontsize=8.5,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(["FF++ (in-domain)", "Celeb-DF (cross-dataset)"], color=NAVY, fontsize=11)
    ax.set_ylabel("AUC", color="#33384D", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    ax.grid(axis="y", color="#E2E8F0", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color("#C9CDD8")
    ax.set_title("AUC — all models (95% bootstrap CI, video-level)",
                 color=NAVY, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "auc_bars_all.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / 'auc_bars_all.png'}")

    # ─── Tableau texte ────────────────────────────────────────────────────────
    lines = ["AUC — comparaison tous modèles (IC 95% bootstrap video-level)", "=" * 65]
    for name in model_order:
        label = MODELS[name][0]
        for dname in dsets:
            if name not in all_results or dname not in all_results[name]:
                lines.append(f"{label:35s} {dname:10s}  N/A")
                continue
            r = all_results[name][dname]
            lines.append(f"{label:35s} {dname:10s}  {r['auc']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]")
    (RESULTS_DIR / "comparison_all.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_DIR / 'comparison_all.txt'}")

    print("\n✓ Toutes les figures générées.")


if __name__ == "__main__":
    main()
