#!/bin/bash
#SBATCH --job-name=popraw
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# (1) population-adjusted site decodability + (2) raw-voxel baseline. CPU.
module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"
export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}

echo "=== population-adjusted site decodability ==="
GLOB="$SCRATCH/multilayer_*.npz" python -u src/population_adjust.py
echo "=== raw-voxel baseline ==="
python -u src/raw_voxel_baseline.py
