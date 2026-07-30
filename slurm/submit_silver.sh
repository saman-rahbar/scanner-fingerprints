#!/bin/bash
#SBATCH --job-name=silver_synthseg
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

# Generate SynthSeg silver labels for ABIDE T1w on CPU (GPU queue is jammed; CPU
# nodes are near-empty). SynthSeg runs on CPU with --cpu; ABIDE T1s are clean 3T
# research scans so standard (non-robust) SynthSeg is sufficient and ~2x faster.
# Requires FreeSurfer (ships mri_synthseg).

set -e
# FreeSurfer on the cluster is license-gated: place your (free) license at
# ~/.licenses/freesurfer.lic first, else the module refuses to load.
export FS_LICENSE=${FS_LICENSE:-$HOME/.licenses/freesurfer.lic}
module load StdEnv/2023 freesurfer/8.2.0-1 2>/dev/null || module load freesurfer
source "$EBROOTFREESURFER/FreeSurferEnv.sh"
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export SILVER_ROOT=${SILVER_ROOT:-$SCRATCH/abide_silver}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-60}       # match segdice cap so labels cover it
export SYNTHSEG_FLAGS=${SYNTHSEG_FLAGS:---cpu --threads ${SLURM_CPUS_PER_TASK:-8}}

python -u src/make_silver_labels.py
