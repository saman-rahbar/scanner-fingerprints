#!/bin/bash
#SBATCH --job-name=segdice
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# Segmentation-Dice headline: frozen BrainSegFounder encoder + trained decoder on
# ABIDE silver labels; leave-one-site-out cross-site Dice BEFORE vs AFTER the
# mid-forward confound-subspace removal, with per-site sign test.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export SILVER_ROOT=${SILVER_ROOT:-$SCRATCH/abide_silver}
export FROZEN_CKPT=${FROZEN_CKPT:-$HOME/models/brainseg_ukb.pt}
export N_CLASSES=${N_CLASSES:-14}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-40}     # fast first-look; scale up once signal seen
export EPOCHS=${EPOCHS:-20}
export LR=${LR:-1e-4}
export INLP_ITERS=${INLP_ITERS:-2,4,8}

python -u src/segdice_intervention.py            # -> segdice_results.json
