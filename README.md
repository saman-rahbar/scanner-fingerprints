# Frozen Brain-MRI Foundation Models Are Site Fingerprints — code

Reproduction code for the paper. The toolkit audits what frozen brain-MRI
foundation-model embeddings encode (acquisition site vs. clinical signal), across
network depth and across encoders, and evaluates two post-hoc removals
(iterative null-space projection and ComBat) on a global readout and on dense
segmentation.

Everything is self-contained Python. Every analysis script ships a synthetic
self-test (`--sandbox`) that runs in seconds with no data, model, or GPU, so you
can verify the method end-to-end before wiring in real data.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The segmentation experiments additionally use SynthSeg for silver labels, which
ships with FreeSurfer 7.3+ (`mri_synthseg`); install FreeSurfer separately if you
run those.

## Quick check (no data needed)

```bash
for s in confound_audit combat_baseline scanner_dominance scanner_dominance_ci \
         global_readout intrinsic_equivalence segdice_intervention segdice_wherebites; do
  python src/$s.py --sandbox
done
```

## Data

All datasets are public. T1-weighted MRI is downloaded from the FCP-INDI S3
bucket (no credentials):

- **ABIDE-I** — `s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/` + `Phenotypic_V1_0b.csv`
- **ABIDE-II** — `s3://fcp-indi/data/Projects/ABIDE2/RawData/`; the composite
  phenotype CSV is obtained from the ABIDE-II site. Helper: `slurm/fetch_abide2.sh`.

Point `ABIDE_ROOT` at the cohort directory and `PHENO_CSV` at its phenotype file.

## Frozen encoders

Three SwinUNETR-family encoders are audited via `FROZEN_CKPT` (unset = random init):

- **brain-pretrained** — a publicly released brain-MRI SSL SwinUNETR checkpoint
- **CT-pretrained** — the MONAI self-supervised SwinUNETR checkpoint
- **random-init** — the same architecture, fixed seed, no checkpoint

## Reproduce the results

**1. Per-layer embeddings** (global-average-pooled features at all five Swin scales):

```bash
ABIDE_ROOT=/path/to/abide FROZEN_CKPT=/path/to/brain_ssl.pt \
  MULTILAYER_NPZ=out/multilayer_brainseg.npz python src/extract_multilayer.py
# repeat with the CT checkpoint and with FROZEN_CKPT unset (random)
```

**2. Decodability matrix with confidence intervals** (site vs. clinical, linear + MLP):

```bash
GLOB="out/multilayer_*.npz" ABIDE_ROOT=/path/to/abide python src/scanner_dominance_ci.py
```

**3. Intrinsic test** — paired non-inferiority + TOST equivalence (random vs. pretrained):

```bash
GLOB="out/multilayer_*.npz" python src/intrinsic_equivalence.py
```

Cross-architecture control (is the fingerprint SwinUNETR-specific?): extract
random-init ViT / ResNet with `ARCH=vit` or `ARCH=resnet`, then run the matrix:
```bash
ARCH=vit ABIDE_ROOT=/path/to/abide MULTILAYER_NPZ=out/multilayer_vit.npz python src/extract_multilayer.py
```

**4. Global readout** — leave-one-site-out clinical classification, raw vs. INLP vs. ComBat:

```bash
COHORT_NPZ=out/cohort.npz ABIDE_ROOT=/path/to/abide python src/global_readout.py
```

**5. Segmentation intervention** — frozen encoder + trained decoder, mid-forward
projection; and the few-site / all-scale / matched-rank random-direction control:

```bash
# silver labels first (needs FreeSurfer)
ABIDE_ROOT=/path/to/abide SILVER_ROOT=out/silver python src/make_silver_labels.py
# leave-one-site-out cross-site Dice, before vs. after removal
ABIDE_ROOT=/path/to/abide SILVER_ROOT=out/silver FROZEN_CKPT=/path/to/brain_ssl.pt \
  python src/segdice_intervention.py
# few-site regime + all-scale + random-direction control
... python src/segdice_wherebites.py
```

**6. Figures:**

```bash
python src/make_figures.py    # writes figures/*.pdf and *.png
```

## Cluster jobs

`slurm/` holds example SLURM submission scripts. Set `--account`, the module
loads, and `VENV` for your environment. Compute nodes without internet should
prefetch data and checkpoints on a login node first.

## Layout

```
src/
  confound_audit.py        core probes, INLP, orthogonal projection
  combat_baseline.py       ComBat (empirical-Bayes) + INLP-vs-ComBat comparison
  build_cohort.py          ABIDE T1w -> frozen-encoder embeddings adapter
  extract_multilayer.py    per-layer GAP extraction for one encoder
  scanner_dominance.py     site-vs-clinical decodability matrix
  scanner_dominance_ci.py  + confidence intervals and nonlinear (MLP) probe
  intrinsic_equivalence.py paired / TOST test for random-vs-pretrained
  arch_encoders.py         random-init ViT / ResNet encoders (cross-architecture control)
  global_readout.py        LOSO clinical classification, raw/INLP/ComBat
  segdice_intervention.py  frozen encoder + trained decoder, mid-forward projection
  segdice_wherebites.py    few-site + all-scale + random-direction control
  make_silver_labels.py    SynthSeg silver labels
  make_figures.py          paper figures
slurm/                     example job scripts
README.md  requirements.txt  LICENSE
```

## Configuration

Scripts are configured by environment variables (`ABIDE_ROOT`, `PHENO_CSV`,
`FROZEN_CKPT`, `MULTILAYER_NPZ`, `GLOB`, `N_REP`, `SUBJ_PER_SITE`, `EPOCHS`,
`INLP_ITERS`, `IMG_SIZE`, `SPACING_MM`, ...); each script documents its own in the
module docstring. Absent data or checkpoints raise loudly rather than fabricate.
