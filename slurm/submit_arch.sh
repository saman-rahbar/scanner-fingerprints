#!/bin/bash
#SBATCH --job-name=arch_control
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

# Generality control: is the intrinsic site fingerprint SwinUNETR-specific? Extract
# per-stage embeddings from RANDOM-INIT encoders of two other architectures (a
# non-hierarchical ViT and a pure-CNN 3-D ResNet), then run the decodability matrix.
# No checkpoints needed (random init). ~30-40 min/arch on one A100.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-60}
OUTDIR=${OUTDIR:-$SCRATCH/multilayer_arch}
mkdir -p "$OUTDIR"

for arch in vit resnet; do
  echo "=== random-init $arch ==="
  ARCH=$arch MULTILAYER_NPZ="$OUTDIR/multilayer_${arch}.npz" python -u src/extract_multilayer.py
done

echo "=== decodability matrix (random-init ViT + ResNet) ==="
GLOB="$OUTDIR/multilayer_*.npz" python -u src/scanner_dominance.py
