#!/bin/bash
#SBATCH --job-name=fmri_arm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Pass the allocation at submission rather than baking it in:
#   sbatch --account=<your-allocation> slurm/submit_fmri_arm.sh

# ---------------------------------------------------------------------------
# The fMRI arm: build the embeddings, then probe them.
#
# No GPU. Nothing here trains or runs a pretrained network: the raw arm is a
# correlation matrix and the random arm is a fixed random convolution, both
# numpy. The probing is scikit-learn. Asking for an A100 to run logistic
# regressions would sit in the GPU queue for hours to do work that eight CPU
# cores finish in minutes.
#
# TWO-STEP, and the order matters. Compute nodes have no route to the internet,
# so the download cannot happen here. Run this once on a LOGIN node first:
#
#   python src/fmri_arm.py --abide-root "$ABIDE_ROOT" --atlas cc200
#
# That writes multilayer_fmri-*.npz and caches the ROI timeseries. This job then
# re-runs it (a no-op against the cache, so it is cheap and keeps the job
# self-contained) and runs the probe.
# ---------------------------------------------------------------------------

set -euo pipefail

module load StdEnv/2023 gcc/12.3 arrow/21.0.0 python/3.11

# scipy-stack supplies scikit-learn, numpy and pandas on this cluster, so the
# job does not depend on a virtualenv existing. If one is present it wins, since
# it may pin versions the analysis was developed against.
module load scipy-stack 2>/dev/null || true
if [ -n "${VENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
elif [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# Fail here with a readable message rather than three minutes in, inside a
# probe, with an ImportError.
python - <<'DEPS' || exit 1
import sys
missing = []
for mod in ("numpy", "pandas", "sklearn"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit(f"missing dependencies: {missing}. "
             "module load scipy-stack, or set VENV to an environment that has them.")
print("dependencies OK")
DEPS

# Find the scripts rather than assume a layout. Working copies differ in where
# the analysis lives, and hard-coding one path fails at import time after the
# job has already waited in the queue.
SRCDIR=""
for cand in "$PWD/src" "$PWD"; do
    if [ -f "$cand/scanner_dominance.py" ]; then SRCDIR="$cand"; break; fi
done
if [ -z "$SRCDIR" ]; then
    for cand in "$PWD"/*/; do
        if [ -f "$cand/scanner_dominance.py" ]; then SRCDIR="${cand%/}"; break; fi
    done
fi
if [ -z "$SRCDIR" ]; then
    SRCDIR="$(dirname "$(find "$PWD" -name scanner_dominance.py -not -path '*/venv/*' 2>/dev/null | head -1)")"
fi
if [ ! -f "$SRCDIR/scanner_dominance.py" ]; then
    echo "Cannot find scanner_dominance.py under $PWD"; exit 1
fi
echo "using scripts in: $SRCDIR"
export PYTHONPATH="$SRCDIR:${PYTHONPATH:-}"

: "${ABIDE_ROOT:?set ABIDE_ROOT to the directory holding the phenotype CSV}"
ATLAS="${ATLAS:-cc200}"

echo "=============================================================="
echo "STAGE 1  build fMRI embeddings (cached; no network needed)"
echo "=============================================================="
python "$SRCDIR/fmri_arm.py" \
    --abide-root "$ABIDE_ROOT" \
    --atlas "$ATLAS" \
    --n-per-site "${N_PER_SITE:-0}" \
    --seeds 0 1 2

echo
echo "=============================================================="
echo "STAGE 2  probe them with the existing pipeline, unchanged"
echo "=============================================================="
# Same probe, same chance correction, same repeated holdouts as the structural
# audit. That is the point: any difference between the modalities is a
# difference in the data, not in how it was measured.
GLOB="multilayer_fmri-*.npz" python "$SRCDIR/scanner_dominance.py"

echo
echo "=============================================================="
echo "DONE.  Compare these decodabilities against the structural ones."
echo "  - if site dominates here too, the pitfall is not modality-specific"
echo "  - if the random arm matches the pretrained arm, pretraining added"
echo "    nothing to what the recording already carried"
echo "=============================================================="
