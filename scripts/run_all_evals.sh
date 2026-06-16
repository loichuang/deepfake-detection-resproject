#!/usr/bin/env bash
# =============================================================================
# run_all_evals.sh — Lance toutes les évaluations du projet sur blutch.
#
# Usage (depuis la racine du projet) :
#   bash scripts/run_all_evals.sh
#
# Ce que fait ce script :
#   1. Vérifie que les checkpoints existent ; sinon, lance l'entraînement.
#   2. Évalue chaque modèle sur FF++ (in-domain) et Celeb-DF-v2 (cross-dataset).
#   3. Affiche un tableau récapitulatif en fin de script.
#
# Modèles couverts :
#   - LDM figé           (best_ldm.pt)
#   - ResNet-50 figé     (best_resnet.pt)
#   - LDM fine-tuné      (best_ldm_finetune.pt)   ← entraîné ici si absent
#   - Wenyi TRY 1        (wenyi_ldm_try1.pt)
#   - Wenyi TRY 2        (wenyi_ldm_try2.pt)
#   - Wenyi ResNet FT    (wenyi_resnet_finetune.pt)
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RESULTS="$ROOT/results"
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"

# Timestamp pour les logs
TS=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo " Latent DeepFake Detection — Full Evaluation Suite"
echo " $(date)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_logged() {
    local label="$1"; shift
    local log="$LOG_DIR/${label}_${TS}.log"
    echo "→ [$label] Démarrage..."
    if python "$@" 2>&1 | tee "$log"; then
        echo "  ✓ [$label] Terminé → log: $log"
    else
        echo "  ✗ [$label] ERREUR — voir $log"
        return 1
    fi
}

# Extrait l'AUC depuis la sortie d'un script d'évaluation
extract_auc() {
    grep -Eo "AUC\s*:\s*[0-9]+\.[0-9]+" "$1" | tail -1 | grep -Eo "[0-9]+\.[0-9]+"
}

# ---------------------------------------------------------------------------
# 1. ENTRAÎNEMENT — uniquement les checkpoints manquants
# ---------------------------------------------------------------------------
echo "── PHASE 1 : Entraînement (checkpoints manquants seulement) ──"
echo ""

if [ ! -f "$RESULTS/best_ldm.pt" ]; then
    echo "  best_ldm.pt absent → lancement de src/train.py"
    run_logged "train_ldm" src/train.py
else
    echo "  ✓ best_ldm.pt existant — entraînement ignoré"
fi

if [ ! -f "$RESULTS/best_resnet.pt" ]; then
    echo "  best_resnet.pt absent → lancement de src/train_resnet.py"
    run_logged "train_resnet" src/train_resnet.py
else
    echo "  ✓ best_resnet.pt existant — entraînement ignoré"
fi

if [ ! -f "$RESULTS/best_ldm_finetune.pt" ]; then
    echo "  best_ldm_finetune.pt absent → lancement de src/train_ldm_finetune.py"
    run_logged "train_ldm_finetune" src/train_ldm_finetune.py
else
    echo "  ✓ best_ldm_finetune.pt existant — entraînement ignoré"
fi

# Wenyi checkpoints : on les suppose déjà présents (entraînés par Wenyi).
# Si absent, décommenter les lignes ci-dessous.
# [ ! -f "$RESULTS/wenyi_ldm_try1.pt" ] && run_logged "wenyi_try1_train" src/wenyi/train_ldm_try1.py
# [ ! -f "$RESULTS/wenyi_ldm_try2.pt" ] && run_logged "wenyi_try2_train" src/wenyi/train_ldm_try2.py
# [ ! -f "$RESULTS/wenyi_resnet_finetune.pt" ] && run_logged "wenyi_resnet_train" src/wenyi/train_resnet_finetune.py

echo ""
echo "── PHASE 2 : Évaluations ──"
echo ""

# ---------------------------------------------------------------------------
# 2a. LDM figé — FF++ + Celeb-DF
# ---------------------------------------------------------------------------
LOG_LDM_FFPP="$LOG_DIR/eval_ldm_ffpp_${TS}.log"
LOG_LDM_CELEB="$LOG_DIR/eval_ldm_celebdf_${TS}.log"

echo "→ [LDM figé] FF++ in-domain..."
python src/eval.py --encoder ldm 2>&1 | tee "$LOG_LDM_FFPP"
echo "→ [LDM figé] Celeb-DF cross-dataset..."
python src/eval_celebdf.py --encoder ldm 2>&1 | tee "$LOG_LDM_CELEB"

# ---------------------------------------------------------------------------
# 2b. ResNet figé — FF++ + Celeb-DF
# ---------------------------------------------------------------------------
LOG_RN_FFPP="$LOG_DIR/eval_resnet_ffpp_${TS}.log"
LOG_RN_CELEB="$LOG_DIR/eval_resnet_celebdf_${TS}.log"

echo "→ [ResNet figé] FF++ in-domain..."
python src/eval.py --encoder resnet 2>&1 | tee "$LOG_RN_FFPP"
echo "→ [ResNet figé] Celeb-DF cross-dataset..."
python src/eval_celebdf.py --encoder resnet 2>&1 | tee "$LOG_RN_CELEB"

# ---------------------------------------------------------------------------
# 2c. LDM fine-tuné — FF++ + Celeb-DF
# ---------------------------------------------------------------------------
LOG_LDM_FT_FFPP="$LOG_DIR/eval_ldm_finetune_ffpp_${TS}.log"
LOG_LDM_FT_CELEB="$LOG_DIR/eval_ldm_finetune_celebdf_${TS}.log"

echo "→ [LDM fine-tuné] FF++ in-domain..."
python src/eval.py --encoder ldm_finetune 2>&1 | tee "$LOG_LDM_FT_FFPP"
echo "→ [LDM fine-tuné] Celeb-DF cross-dataset..."
python src/eval_celebdf.py --encoder ldm_finetune 2>&1 | tee "$LOG_LDM_FT_CELEB"

# ---------------------------------------------------------------------------
# 2d. Wenyi TRY 1
# ---------------------------------------------------------------------------
LOG_W1="$LOG_DIR/eval_wenyi_try1_${TS}.log"
echo "→ [Wenyi TRY 1] FF++ + Celeb-DF..."
python src/wenyi/test_ldm_try1.py 2>&1 | tee "$LOG_W1"

# ---------------------------------------------------------------------------
# 2e. Wenyi TRY 2
# ---------------------------------------------------------------------------
LOG_W2="$LOG_DIR/eval_wenyi_try2_${TS}.log"
echo "→ [Wenyi TRY 2] FF++ + Celeb-DF..."
python src/wenyi/test_ldm_try2.py 2>&1 | tee "$LOG_W2"

# ---------------------------------------------------------------------------
# 2f. Wenyi ResNet fine-tuné
# ---------------------------------------------------------------------------
LOG_WR="$LOG_DIR/eval_wenyi_resnet_${TS}.log"
echo "→ [Wenyi ResNet FT] FF++ + Celeb-DF..."
python src/wenyi/test_resnet_finetune.py 2>&1 | tee "$LOG_WR"

# ---------------------------------------------------------------------------
# 3. RÉSUMÉ
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " RÉSUMÉ DES RÉSULTATS"
echo "============================================================"
echo ""

# Extraction AUC depuis les logs
auc_val() {
    grep -Eo "AUC\s*:\s*[0-9]+\.[0-9]+" "$1" 2>/dev/null | tail -1 | grep -Eo "[0-9]+\.[0-9]+" || echo "N/A"
}

# Wenyi: format différent "AUC: 0.xxxx"
wenyi_auc_ffpp() {
    grep "FF.. In-Domain\|In-Domain" "$1" 2>/dev/null | grep -Eo "[0-9]+\.[0-9]+" | tail -1 || echo "N/A"
}
wenyi_auc_celeb() {
    grep "Celeb.*Cross\|Cross" "$1" 2>/dev/null | grep -Eo "[0-9]+\.[0-9]+" | tail -1 || echo "N/A"
}

printf "%-30s %-12s %-12s\n" "Modèle" "FF++ AUC" "Celeb-DF AUC"
printf "%-30s %-12s %-12s\n" "------------------------------" "------------" "------------"
printf "%-30s %-12s %-12s\n" "LDM figé (SD 1.4)"     "$(auc_val $LOG_LDM_FFPP)"    "$(auc_val $LOG_LDM_CELEB)"
printf "%-30s %-12s %-12s\n" "ResNet-50 figé"         "$(auc_val $LOG_RN_FFPP)"     "$(auc_val $LOG_RN_CELEB)"
printf "%-30s %-12s %-12s\n" "LDM fine-tuné (SD 1.4)" "$(auc_val $LOG_LDM_FT_FFPP)" "$(auc_val $LOG_LDM_FT_CELEB)"
printf "%-30s %-12s %-12s\n" "Wenyi — PNDM 1 step"    "$(wenyi_auc_ffpp $LOG_W1)"  "$(wenyi_auc_celeb $LOG_W1)"
printf "%-30s %-12s %-12s\n" "Wenyi — DDIM 20 steps"  "$(wenyi_auc_ffpp $LOG_W2)"  "$(wenyi_auc_celeb $LOG_W2)"
printf "%-30s %-12s %-12s\n" "Wenyi — ResNet FT"      "$(wenyi_auc_ffpp $LOG_WR)"  "$(wenyi_auc_celeb $LOG_WR)"

echo ""
echo "Logs complets : $LOG_DIR/"
echo "============================================================"
