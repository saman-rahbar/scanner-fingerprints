#!/usr/bin/env python
"""make_silver_labels.py -- generate SynthSeg silver segmentations for ABIDE T1w.

SynthSeg (Billot et al., MedIA 2023) is contrast- and resolution-agnostic and is
deliberately robust to scanner/site, so the silver LABELS are not themselves the
confound we are auditing. Ships with FreeSurfer 7.3+ as `mri_synthseg`.

We build input/output path lists and call mri_synthseg ONCE (loads the model a
single time) rather than per-file. Output naming matches segdice_intervention.py:
  $SILVER_ROOT/sub-<SID>_synthseg.nii.gz

Env:
  ABIDE_ROOT   -- ABIDE root with <SITE>/sub-*/anat/*_T1w.nii.gz  (required)
  SILVER_ROOT  -- output dir for silver labels                    (required)
  SUBJ_PER_SITE-- optional cap per site (match the seg run's cap; default: all)
  SYNTHSEG_CMD -- override the mri_synthseg executable (default: 'mri_synthseg')
  SYNTHSEG_FLAGS -- extra flags, e.g. '--robust --cpu' (default: '')
"""
import os
import re
import sys
import glob
import subprocess


def main():
    root = os.environ.get("ABIDE_ROOT")
    out_root = os.environ.get("SILVER_ROOT")
    if not root or not os.path.isdir(root):
        sys.exit(f"ABIDE_ROOT missing/invalid: {root!r}")
    if not out_root:
        sys.exit("SILVER_ROOT not set")
    os.makedirs(out_root, exist_ok=True)
    cmd = os.environ.get("SYNTHSEG_CMD", "mri_synthseg")
    flags = os.environ.get("SYNTHSEG_FLAGS", "").split()
    cap = int(os.environ.get("SUBJ_PER_SITE", "0"))

    # discover T1s, grouped by site (site = the path component after ABIDE_ROOT)
    by_site = {}
    for t1 in glob.glob(os.path.join(root, "**", "*_T1w.nii.gz"), recursive=True):
        m = re.search(r"sub-0*(\d+)", os.path.basename(t1))
        if not m:
            continue
        rel = os.path.relpath(t1, root)
        site = rel.split(os.sep)[0]
        by_site.setdefault(site, []).append((int(m.group(1)), t1))

    inputs, outputs = [], []
    for site in sorted(by_site):
        rows = sorted(by_site[site])
        if cap > 0:
            rows = rows[:cap]
        for sid, t1 in rows:
            dst = os.path.join(out_root, f"sub-{sid}_synthseg.nii.gz")
            if os.path.exists(dst):
                continue                      # idempotent: skip done
            inputs.append(t1); outputs.append(dst)

    print(f"[silver] {len(inputs)} volumes to segment across {len(by_site)} sites "
          f"(cap {cap or 'none'}/site)")
    if not inputs:
        print("[silver] nothing to do (all present)"); return 0

    lst_dir = os.path.join(out_root, "_lists"); os.makedirs(lst_dir, exist_ok=True)
    in_txt = os.path.join(lst_dir, "inputs.txt"); out_txt = os.path.join(lst_dir, "outputs.txt")
    with open(in_txt, "w") as f:
        f.write("\n".join(inputs) + "\n")
    with open(out_txt, "w") as f:
        f.write("\n".join(outputs) + "\n")

    full = [cmd, "--i", in_txt, "--o", out_txt] + flags
    print("[silver] running:", " ".join(full))
    r = subprocess.run(full)
    if r.returncode != 0:
        sys.exit(f"mri_synthseg failed (rc={r.returncode}). Is FreeSurfer loaded? "
                 f"(module load freesurfer; source $FREESURFER_HOME/SetUpFreeSurfer.sh)")
    done = sum(os.path.exists(o) for o in outputs)
    print(f"[silver] wrote {done}/{len(outputs)} silver labels to {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
