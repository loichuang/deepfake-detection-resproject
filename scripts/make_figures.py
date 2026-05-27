"""Generate presentation figures + bootstrap confidence intervals.

For each trained model (LDM, ResNet), on each test set (FF++ in-domain,
Celeb-DF cross-dataset):
  - load the best checkpoint,
  - reuse the cached latents (fast, no re-encoding),
  - compute predicted scores,
  - estimate AUC with a 95% confidence interval via a VIDEO-LEVEL bootstrap
    (frames of the same video are resampled together, since they are not
    independent),
  - draw ROC curves and a bar chart with error bars.

Outputs (in results/):
  - roc_curves.png       ROC for FF++ and Celeb-DF, LDM vs ResNet
  - auc_bars_ci.png      bar chart with 95% CI error bars
  - comparison_ci.txt    text table: AUC [low, high]

Run from the project root on blutch:
    python scripts/make_figures.py
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
from src.model import FrozenVAEEncoder, ResNetEncoder, MLPClassifier
from src.train import (
    DEVICE, FFPP_ROOT, MANIPULATION, N_FRAMES_PER_VIDEO, SEED, OUT_DIR,
    encode_or_load,
)

CELEBDF_ROOT = "/medias/db/deepfakes/Celeb-DF-v2"
N_BOOT = 2000

# Sober palette, matching the slides.
COL = {"ldm": "#2A9D8F", "resnet": "#9AA0B4"}
NAVY = "#1E2761"
EXTRACTORS = {
    "ldm":    (FrozenVAEEncoder, 4 * 64 * 64, "best_ldm.pt",   "LDM encoder"),
    "resnet": (ResNetEncoder,    2048,        "best_resnet.pt", "ResNet-50"),
}


# ----- metrics (numpy, no sklearn dependency) -----
def auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-based (Mann-Whitney U) formula."""
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N)


def roc_curve_np(scores: np.ndarray, labels: np.ndarray):
    """Return (fpr, tpr) arrays for the ROC curve."""
    order = np.argsort(-scores)
    l = labels[order].astype(float)
    P = l.sum()
    N = len(l) - P
    tpr = np.concatenate([[0.0], np.cumsum(l) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - l) / N])
    return fpr, tpr


def bootstrap_ci_video(scores, labels, video_ids, n_boot=N_BOOT, seed=0):
    """95% CI on the AUC via video-level (cluster) bootstrap."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(video_ids):
        groups.setdefault(v, []).append(i)
    vids = list(groups.values())
    aucs = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(vids), size=len(vids))
        idx = np.concatenate([vids[c] for c in chosen])
        a = auc_np(scores[idx], labels[idx])
        if not np.isnan(a):
            aucs.append(a)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


@torch.no_grad()
def predict(model, ds) -> np.ndarray:
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    out = []
    model.eval()
    for z, _ in loader:
        out.append(torch.sigmoid(model(z.to(DEVICE))).cpu())
    return torch.cat(out).numpy()


def video_ids_from_paths(paths):
    """Frame path .../<video_id>/<frame>.png -> video_id (parent folder)."""
    return [Path(p).parent.name for p in paths]


def main() -> None:
    print(f"Device: {DEVICE}")

    # Build the two test sets once (paths kept for video-level bootstrap).
    ffpp_paths, ffpp_labels = build_ffds_split(
        FFPP_ROOT, "test.json", n_videos_per_class=None,
        n_frames_per_video=N_FRAMES_PER_VIDEO, manipulation=MANIPULATION, seed=SEED)
    celeb_paths, celeb_labels = build_celebdf_test_split(
        CELEBDF_ROOT, n_frames_per_video=N_FRAMES_PER_VIDEO, seed=SEED)

    datasets = {
        "FF++ test": (FFDS(ffpp_paths, ffpp_labels), np.array(ffpp_labels), video_ids_from_paths(ffpp_paths)),
        "Celeb-DF":  (FFDS(celeb_paths, celeb_labels), np.array(celeb_labels), video_ids_from_paths(celeb_paths)),
    }

    # results[ext][dataset] = {"scores", "labels", "auc", "lo", "hi", "fpr", "tpr"}
    results = {}
    for name, (EncCls, dim, ckpt, _label) in EXTRACTORS.items():
        ckpt_path = OUT_DIR / ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} missing. Run run_comparison.py first.")
        encoder = EncCls().to(DEVICE)
        model = MLPClassifier(input_dim=dim).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        tag = f"{name}_{MANIPULATION}_{N_FRAMES_PER_VIDEO}f_seed{SEED}"

        results[name] = {}
        for dname, (ffds, labels, vids) in datasets.items():
            cache_key = ("test" if dname == "FF++ test" else "celebdf") + f"_{tag}"
            ds = encode_or_load(ffds, encoder, DEVICE, cache_key)
            scores = predict(model, ds)
            auc = auc_np(scores, labels)
            lo, hi = bootstrap_ci_video(scores, labels, vids)
            fpr, tpr = roc_curve_np(scores, labels)
            results[name][dname] = dict(auc=auc, lo=lo, hi=hi, fpr=fpr, tpr=tpr)
            print(f"[{name:6s}] {dname:9s}  AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    OUT_DIR.mkdir(exist_ok=True)

    # ---- Figure 1: ROC curves (FF++ | Celeb-DF) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, dname in zip(axes, ["FF++ test", "Celeb-DF"]):
        for name, (_e, _d, _c, label) in EXTRACTORS.items():
            r = results[name][dname]
            ax.plot(r["fpr"], r["tpr"], color=COL[name], lw=2.2,
                    label=f"{label}  (AUC {r['auc']:.2f})")
        ax.plot([0, 1], [0, 1], color="#C9CDD8", lw=1, ls="--")
        ax.set_title(f"{dname}" + ("  —  in-domain" if dname == "FF++ test" else "  —  cross-dataset"),
                     color=NAVY, fontsize=12)
        ax.set_xlabel("False positive rate", color="#33384D", fontsize=10)
        ax.set_ylabel("True positive rate", color="#33384D", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.grid(color="#E2E8F0", lw=0.5)
        for spine in ax.spines.values():
            spine.set_color("#C9CDD8")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=160)
    print(f"Saved {OUT_DIR / 'roc_curves.png'}")

    # ---- Figure 2: bar chart with 95% CI error bars ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    dsets = ["FF++ test", "Celeb-DF"]
    x = np.arange(len(dsets))
    width = 0.36
    for k, (name, (_e, _d, _c, label)) in enumerate(EXTRACTORS.items()):
        aucs = [results[name][d]["auc"] for d in dsets]
        los = [results[name][d]["auc"] - results[name][d]["lo"] for d in dsets]
        his = [results[name][d]["hi"] - results[name][d]["auc"] for d in dsets]
        ax.bar(x + (k - 0.5) * width, aucs, width, color=COL[name], label=label,
               yerr=[los, his], capsize=5, error_kw=dict(ecolor="#33384D", lw=1))
        for xi, a in zip(x + (k - 0.5) * width, aucs):
            ax.text(xi, a + 0.03, f"{a:.2f}", ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.axhline(0.5, color="#C9CDD8", lw=1, ls=":")
    ax.text(1.45, 0.515, "chance", color="#9AA0B4", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(dsets, color=NAVY, fontsize=11)
    ax.set_ylabel("AUC", color="#33384D"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(axis="y", color="#E2E8F0", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color("#C9CDD8")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auc_bars_ci.png", dpi=160)
    print(f"Saved {OUT_DIR / 'auc_bars_ci.png'}")

    # ---- Text table ----
    lines = ["AUC with 95% video-level bootstrap CI", "=" * 50]
    for name, (_e, _d, _c, label) in EXTRACTORS.items():
        for d in dsets:
            r = results[name][d]
            lines.append(f"{label:12s} {d:10s}  {r['auc']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")
    (OUT_DIR / "comparison_ci.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_DIR / 'comparison_ci.txt'}")


if __name__ == "__main__":
    main()
