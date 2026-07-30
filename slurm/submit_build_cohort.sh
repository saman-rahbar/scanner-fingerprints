#!/bin/bash
#SBATCH --job-name=confound_cohort
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Extract frozen Swin-UNETR-encoder embeddings for ABIDE T1w (site+age confounds,
# sex as the downstream task) -> cohort_cache.npz, then run the confound audit.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export FROZEN_CKPT=${FROZEN_CKPT:-}          # optional MONAI SSL weights; empty -> init encoder
export COHORT_NPZ=${COHORT_NPZ:-$SCRATCH/cohort_cache.npz}

python src/build_cohort.py                        # -> $COHORT_NPZ
echo "== audit =="
python src/confound_audit.py                      # reads $COHORT_NPZ -> audit_results.json
