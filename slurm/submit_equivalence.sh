#!/bin/bash
#SBATCH --job-name=equivalence
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Paired non-inferiority + TOST equivalence for random-vs-pretrained
# site decodability. CPU-only, reuses the multilayer_*.npz caches.

module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export GLOB=${GLOB:-$SCRATCH/multilayer_*.npz}   # ABIDE-II: $SCRATCH/multilayer_abide2/multilayer_*.npz
export N_REP=${N_REP:-200}
export DEEP_LAYERS=${DEEP_LAYERS:-3,4}
export EPS=${EPS:-0.05}

python -u intrinsic_equivalence.py               # -> intrinsic_equivalence_results.json
