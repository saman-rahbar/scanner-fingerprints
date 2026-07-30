#!/bin/bash
# Download ABIDE-II T1w (+ phenotype if reachable) from the public fcp-indi S3
# bucket. Confirmed layout: RawData/<SITE>/sub-<ID>/ses-*/anat/*_T1w.nii.gz .
# Run on a LOGIN node (internet). Site always derives from the folder path, so
# the extraction works even if the phenotype download fails (clinical columns
# just fill in when the CSV is present).
set -u
DEST=${DEST:-$HOME/data/abide2}
BIDS=${BIDS:-s3://fcp-indi/data/Projects/ABIDE2/RawData}
mkdir -p "$DEST"

# a broad multi-scanner spread (skip longitudinal _Long sets)
SITES=${SITES:-"ABIDEII-BNI_1 ABIDEII-EMC_1 ABIDEII-ETHZ_1 ABIDEII-GU_1 \
ABIDEII-IP_1 ABIDEII-IU_1 ABIDEII-KKI_1 ABIDEII-NYU_1 ABIDEII-NYU_2 \
ABIDEII-OHSU_1 ABIDEII-SDSU_1 ABIDEII-TCD_1 ABIDEII-UCD_1 ABIDEII-UCLA_1 \
ABIDEII-USM_1"}

echo "===== phenotype (optional) ====="
for url in \
  "https://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Composite_Phenotypic.csv" \
  "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/ABIDEII_Composite_Phenotypic.csv"; do
  if wget -q "$url" -O "$DEST/ABIDEII_Composite_Phenotypic.csv" \
        && [ -s "$DEST/ABIDEII_Composite_Phenotypic.csv" ]; then
    echo "[ok] phenotype from $url"; break
  fi
done
[ -s "$DEST/ABIDEII_Composite_Phenotypic.csv" ] || \
  echo "[warn] no phenotype; site-only replication still works (site from path)"

for site in $SITES; do
  echo "===== syncing $site (T1w only) ====="
  aws s3 sync --no-sign-request "$BIDS/$site/" "$DEST/$site/" \
      --exclude "*" --include "*_T1w.nii.gz" || echo "[warn] $site sync failed"
done

echo
echo "[done] ABIDE-II under $DEST"
echo "T1w files: $(find "$DEST" -name '*_T1w.nii.gz' 2>/dev/null | wc -l)"
