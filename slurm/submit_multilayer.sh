#!/bin/bash
#SBATCH --job-name=multilayer
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# "Scanner > brain" audit: per-layer frozen-encoder GAP embeddings for THREE
# encoders (brain-pretrained / CT-pretrained / random-init), then the
# scanner-vs-clinical decodability matrix. ~20-30 min/model on one A100.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export PHENO_CSV=${PHENO_CSV:-Phenotypic_V1_0b.csv}   # ABIDE-II: ABIDEII_Composite_Phenotypic.csv
OUTDIR=${OUTDIR:-$SCRATCH}                            # per-cohort dir avoids mixing
mkdir -p "$OUTDIR"

run_model () {
  local tag=$1 ckpt=$2
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "[skip] $tag: checkpoint not found at $ckpt"; return
  fi
  echo "=== extracting $tag (ckpt=${ckpt:-RANDOM-INIT}) ==="
  if [ -z "$ckpt" ]; then unset FROZEN_CKPT; else export FROZEN_CKPT="$ckpt"; fi
  export MULTILAYER_NPZ="$OUTDIR/multilayer_${tag}.npz"
  python -u extract_multilayer.py
}

run_model brainseg "$HOME/models/brainseg_ukb.pt"    # brain-pretrained (UK Biobank 41k)
run_model ctssl    "$HOME/models/model_swinvit.pt"   # CT-pretrained MONAI SSL
run_model random   ""                                # random-init baseline

echo "=== scanner-dominance matrix ==="
export GLOB="$OUTDIR/multilayer_*.npz"
python -u scanner_dominance.py                        # -> scanner_dominance_results.json
