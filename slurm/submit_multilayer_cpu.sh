#!/bin/bash
#SBATCH --job-name=multilayer_cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

# Per-layer frozen-encoder extraction on CPU (GPU queue is backed up ~31h; CPU
# schedules in minutes). Same embeddings as GPU, just slower. Uses SUBJ_PER_SITE
# to keep it tractable. Set ABIDE_ROOT / PHENO_CSV / OUTDIR for the cohort.

module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide2}
export PHENO_CSV=${PHENO_CSV:-ABIDEII_Composite_Phenotypic.csv}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-40}            # cap per site for speed
OUTDIR=${OUTDIR:-$SCRATCH/multilayer_abide2}
mkdir -p "$OUTDIR"

run_model () {
  local tag=$1 ckpt=$2
  if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
    echo "[skip] $tag: checkpoint not found at $ckpt"; return
  fi
  echo "=== extracting $tag (ckpt=${ckpt:-RANDOM-INIT}) ==="
  if [ -z "$ckpt" ]; then unset FROZEN_CKPT; else export FROZEN_CKPT="$ckpt"; fi
  export MULTILAYER_NPZ="$OUTDIR/multilayer_${tag}.npz"
  python -u src/extract_multilayer.py
}

run_model brainseg "$HOME/models/brainseg_ukb.pt"
run_model ctssl    "$HOME/models/model_swinvit.pt"
run_model random   ""

echo "=== scanner-dominance matrix ==="
export GLOB="$OUTDIR/multilayer_*.npz"
python -u src/scanner_dominance.py
