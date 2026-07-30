#!/bin/bash
#SBATCH --job-name=dominance_ci
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# Scanner-dominance matrix WITH confidence intervals (repeated holdouts) + a
# nonlinear MLP probe. CPU-only, reuses the multilayer_*.npz caches.

module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}    # for the ASD-label join
export GLOB=${GLOB:-$SCRATCH/multilayer_*.npz}
export N_REP=${N_REP:-50}

python -u scanner_dominance_ci.py                    # -> scanner_dominance_ci_results.json
