#!/bin/bash
#SBATCH --job-name=multiseed_intrinsic
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

# Seed-stability control for the "intrinsic" claim: for each architecture, build
# SEVERAL random-init encoders (different weight-init seeds), read the deep-layer
# GAP embedding, and decode acquisition site. If deep-layer site decodability is
# stable across seeds, the fingerprint is a property of the architecture rather
# than of one lucky initialization. Each volume is preprocessed once and reused
# across seeds, so 3 seeds cost ~1x the data-loading time. ~1.5-2 h total on 1 A100.

module load StdEnv/2023 gcc/12.3 cuda/12.2 cudnn/8.9 arrow/21.0.0 python/3.11
cd "${SLURM_SUBMIT_DIR}"
source "${VENV:-./venv}/bin/activate"

export ABIDE_ROOT=${ABIDE_ROOT:-$HOME/data/abide}
export SUBJ_PER_SITE=${SUBJ_PER_SITE:-60}
export SEEDS=${SEEDS:-0,1,2}
OUTDIR=${OUTDIR:-$SCRATCH/multiseed_intrinsic}
mkdir -p "$OUTDIR"

for arch in swin vit resnet; do
  echo "=== multi-seed intrinsic: $arch (seeds=$SEEDS) ==="
  ARCH=$arch MULTISEED_JSON="$OUTDIR/multiseed_${arch}.json" \
    python -u multiseed_intrinsic.py
done

echo "=== SUMMARY ==="
for arch in swin vit resnet; do
  python - "$OUTDIR/multiseed_${arch}.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"{d['arch']:7s}  L4 site decodability = {d['L4_mean']:.3f} +/- "
          f"{d['L4_std']:.3f}  (range {d['L4_range']:.3f}, seeds {d['seeds']})")
except Exception as e:
    print(f"{sys.argv[1]}: {e}")
PY
done
