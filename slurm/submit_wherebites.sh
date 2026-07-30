#!/bin/bash
#SBATCH --job-name=wherebites
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# "Where does the confound bite?" -- few-site training + ALL-SCALE mid-forward
# confound projection + per-region Dice. Single train/test split (train on
# TRAIN_SITES, deploy on the rest), so ~1h, not 6-fold LOSO.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export SILVER_ROOT=${SILVER_ROOT:-$SCRATCH/abide_silver}
export FROZEN_CKPT=${FROZEN_CKPT:-$HOME/models/brainseg_ukb.pt}
export N_CLASSES=${N_CLASSES:-14}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-60}
export EPOCHS=${EPOCHS:-40}
export LR=${LR:-1e-4}
export INLP_ITERS=${INLP_ITERS:-8}
export SCALES=${SCALES:-0,1,2,3,4}            # all Swin scales; '4' = bottleneck-only
export TRAIN_SITES=${TRAIN_SITES:-}           # empty -> 2 largest sites

python -u segdice_wherebites.py               # -> segdice_wherebites_results.json
