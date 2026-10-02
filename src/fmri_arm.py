#!/usr/bin/env python
"""The fMRI arm of the audit: same cohort, same probes, a temporal signal.

The structural audit showed that acquisition site is decodable at ~0.9 from a
frozen brain-MRI encoder, that a randomly initialised encoder does just as well,
and that the raw voxels do better still. This runs the identical test on
resting-state fMRI from the *same ABIDE subjects at the same sites*, so the two
modalities can be compared without changing cohort, site definition, or probe.

Why this modality and not another. Structural MRI is not a timeseries, and a
reviewer at a biosignal venue can say so in one line. Resting-state fMRI is a
temporal signal from the same scanners and the same people, so it tests whether
the pitfall is a property of brain imaging or a property of that one modality,
and it does so without any new cohort to defend.

Why the preprocessed ROI timeseries rather than raw 4-D volumes. ABIDE
Preprocessed already publishes per-subject ROI timeseries as small text files.
Downloading those is minutes rather than days, needs no fMRI preprocessing of
our own, and removes preprocessing choices as an explanation for whatever we
find. The cost is that the pipeline's own choices are baked in, which is why the
pipeline and strategy are recorded in the output.

Three representations are written, each in the npz layout the existing probe
already reads (emb_L*, site, age, sex, subject_id), so scanner_dominance.py runs
on them unchanged:

  raw       vectorised Fisher-z functional connectivity. This is the
            raw-input control: no encoder at all.
  random    a randomly initialised temporal encoder over the ROI timeseries,
            pooled per stage. This is the negative control that decides whether
            pretraining contributed anything.

            IMPORTANT: to be a fair control this must be the *same architecture*
            as whatever pretrained model you compare against, at random weights.
            The generic convolutional stack below is a stand-in so the pipeline
            runs end to end; replace it with the pretrained model's own class
            once that model is chosen. A control built from a different
            architecture answers a different question.
  <ckpt>    optional: any encoder you can load, for the pretrained arm.

Run on a LOGIN node the first time, since it downloads.

  python fmri_arm.py --out-dir . --atlas cc200 --n-per-site 60
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

#: ABIDE Preprocessed lives here. The path encodes the pipeline and the nuisance
#: strategy, both of which change what survives in the signal, so they are
#: recorded in the output rather than assumed.
BASE = ("https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
        "Outputs/{pipeline}/{strategy}/rois_{atlas}/{file_id}_rois_{atlas}.1D")

#: Global signal regression removes a component that carries both artefact and
#: signal, and it is exactly the kind of step that could mask a site effect. The
#: default keeps it in, so the audit is not quietly cleaning up its own subject.
DEFAULT_STRATEGY = "filt_noglobal"


def load_pheno(abide_root: str, pheno_csv: str):
    """Subject table: FILE_ID, site, age, sex, diagnosis.

    Subjects whose FILE_ID is "no_filename" have no preprocessed derivative and
    are dropped here rather than failing one at a time during download.
    """
    import pandas as pd
    csv = pheno_csv if os.path.isabs(pheno_csv) else os.path.join(abide_root, pheno_csv)
    if not os.path.exists(csv):
        sys.exit(f"phenotype table not found: {csv}")
    df = pd.read_csv(csv, encoding="latin-1")
    df.columns = df.columns.str.strip()
    need = {"FILE_ID", "SITE_ID", "SUB_ID", "AGE_AT_SCAN", "SEX", "DX_GROUP"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"phenotype table missing columns: {sorted(missing)}")
    df = df[df["FILE_ID"].astype(str) != "no_filename"]
    return df


def fetch_roi_series(file_id, atlas, pipeline, strategy, cache: Path):
    """One subject's ROI timeseries, cached on disk.

    Returns None when the derivative does not exist for this subject, which is
    common and not an error: coverage differs by pipeline and atlas.
    """
    dest = cache / f"{file_id}_rois_{atlas}.1D"
    if not dest.exists():
        url = BASE.format(pipeline=pipeline, strategy=strategy,
                          atlas=atlas, file_id=file_id)
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
    try:
        arr = np.loadtxt(dest, skiprows=0)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < 30 or arr.shape[1] < 2:
        return None
    return arr.astype(np.float64)          # [T, n_rois]


def valid_rois(series: list, min_frac: float = 0.99):
    """A single ROI set shared by the subjects that are kept.

    The set has to be shared. Dropping dead parcels per subject and then
    trimming the vectorised correlations to a common length looks equivalent and
    is not: element i would index a different ROI pair for different subjects.
    The count of dead parcels also varies with coverage and field of view, which
    vary by site, so per-subject dropping can create site decodability out of
    nothing. That is the exact artifact this audit exists to detect, so it must
    not be introduced by the audit itself.

    Requiring every subject to have every parcel is too strict in practice: a
    handful of subjects with poor coverage can eliminate most parcels and throw
    away the majority of the features. Instead keep parcels alive in at least
    ``min_frac`` of subjects, then drop the subjects that are still missing one.
    Both counts are returned so the trade can be reported rather than hidden.
    """
    n_roi = min(s.shape[1] for s in series)
    alive = np.stack([np.std(s[:, :n_roi], axis=0) > 0 for s in series])  # [N, R]
    frac = alive.mean(axis=0)
    keep_roi = np.where(frac >= min_frac)[0]
    if len(keep_roi) < 2:
        return keep_roi, np.arange(len(series))
    keep_sub = np.where(alive[:, keep_roi].all(axis=1))[0]
    return keep_roi, keep_sub


def connectivity(ts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Vectorised Fisher-z correlation over a fixed, shared ROI set.

    This is the raw-input control for this modality: it involves no learned
    parameters, so any site decodability it carries is a property of the
    recording rather than of a model.
    """
    ts = ts[:, keep]
    if ts.shape[1] < 2:
        return None
    c = np.corrcoef(ts, rowvar=False)
    c = np.clip(c, -0.999999, 0.999999)
    z = np.arctanh(c)
    v = z[np.triu_indices_from(z, k=1)]
    return v if np.isfinite(v).all() else None


class RandomTemporalEncoder:
    """A randomly initialised 1-D convolutional encoder over ROI timeseries.

    Deliberately untrained. The point of this object is to answer one question:
    how much of the decodability that a pretrained encoder shows was already
    available from an architecture that learned nothing? Five stages so the
    output matches the five Swin depths the structural audit reports, letting
    the two modalities be read on the same axis.

    Channel-mixing is random and fixed by seed, and the pooling is a global mean
    over time, so each stage yields one vector per subject exactly as the
    structural pipeline does.
    """

    def __init__(self, n_rois: int, seed: int = 0, widths=(64, 96, 192, 384, 768)):
        rng = np.random.RandomState(seed)
        self.widths = widths
        self.kernels, self.mixes = [], []
        fan_in = n_rois
        for w in widths:
            # He-style scaling keeps activations from collapsing or exploding
            # across five stages, which would make the control uninformative for
            # a reason that has nothing to do with the data.
            self.kernels.append(rng.randn(3, fan_in, w).astype(np.float32)
                                * np.sqrt(2.0 / (3 * fan_in)))
            fan_in = w

    def __call__(self, ts: np.ndarray):
        """Returns one pooled embedding per stage."""
        x = ts.astype(np.float32)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        out = []
        for k in self.kernels:
            T, C = x.shape
            if T < 3:
                return None
            # valid 1-D convolution, stride 2, then ReLU
            win = np.lib.stride_tricks.sliding_window_view(x, 3, axis=0)  # [T-2,C,3]
            win = win.transpose(0, 2, 1)                                   # [T-2,3,C]
            y = np.tensordot(win, k, axes=([1, 2], [0, 1]))                # [T-2,W]
            y = np.maximum(y, 0.0)[::2]
            if y.shape[0] < 3:
                y = np.maximum(y, 0.0)
            x = y
            out.append(x.mean(axis=0))
        return out


def permutation_null(E, labels, n_perm: int = 20, seed: int = 0):
    """Decodability floor for this feature matrix at this sample size.

    A linear probe on many dimensions and few subjects can separate labels that
    carry no information at all, and a reported decodability means nothing until
    that floor is known. Shuffling the labels and re-probing gives it directly.

    Measured on synthetic noise this floor is about 0.01 to 0.02 with a maximum
    near 0.08, for dimensionalities from 768 to 30000 and 250 to 546 subjects,
    so the L2 penalty is holding. Report it anyway: it costs seconds and it is
    the difference between a number and an interpretable number.
    """
    from confound_audit import probe_decodability
    rng = np.random.RandomState(seed)
    vals = [float(probe_decodability(E, rng.permutation(labels), s))
            for s in range(n_perm)]
    return float(np.mean(vals)), float(np.max(vals))


class RandomFCEncoder:
    """A randomly initialised encoder over the connectivity vector.

    This is the negative control that belongs to this modality. The temporal
    convolution above pools over time, which discards the inter-ROI correlation
    structure that carries site, so it answers a question about that
    architecture rather than about pretraining. fMRI models are normally fed
    connectivity or region timeseries with attention across regions, and the
    honest control takes the same input the pretrained model takes and differs
    only in that its weights were never trained.

    Random projections with a nonlinearity, stacked, so each stage yields one
    vector per subject and the depth axis lines up with the structural audit.

    Width is held constant across stages on purpose. An encoder that narrows
    with depth loses information at every stage by construction, so its
    decodability falls for a reason that has nothing to do with representation:
    the first version of this class went 768 to 64 and produced a tidy
    decreasing curve that was pure compression. The structural encoder widens
    with depth, so a narrowing control plotted on the same axis would invite
    exactly the wrong comparison.
    """

    def __init__(self, in_dim: int, seed: int = 0, widths=(768, 768, 768, 768, 768)):
        rng = np.random.RandomState(seed)
        self.W = []
        fan_in = in_dim
        for w in widths:
            self.W.append((rng.randn(fan_in, w) * np.sqrt(2.0 / fan_in)).astype(np.float32))
            fan_in = w

    def __call__(self, v: np.ndarray):
        x = v.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        out = []
        for W in self.W:
            x = np.maximum(x @ W, 0.0)
            out.append(x.copy())
        return out


def build(args):
    import pandas as pd
    df = load_pheno(args.abide_root, args.pheno_csv)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Match the structural audit's site handling: optionally cap per site so a
    # few large sites cannot dominate the k-way problem.
    rows = []
    for site, grp in df.groupby("SITE_ID"):
        g = grp if not args.n_per_site else grp.head(args.n_per_site)
        rows.extend(g.itertuples())

    series, sites, ages, sexes, ids = [], [], [], [], []
    for i, r in enumerate(rows):
        ts = fetch_roi_series(str(r.FILE_ID), args.atlas, args.pipeline,
                              args.strategy, cache)
        if ts is None:
            continue
        series.append(ts)
        sites.append(str(r.SITE_ID))
        ages.append(float(r.AGE_AT_SCAN) if np.isfinite(r.AGE_AT_SCAN) else np.nan)
        sexes.append(float(r.SEX))
        # Same id convention as the structural pipeline, so the phenotype join
        # and the diagnosis lookup in scanner_dominance.py work unchanged.
        ids.append(f"{r.SITE_ID}/sub-{int(r.SUB_ID)}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(rows)} scanned, {len(series)} usable", flush=True)

    if len(series) < 40:
        sys.exit(f"only {len(series)} usable subjects; refusing degenerate output")
    print(f"[cohort] {len(series)} subjects across {len(set(sites))} sites")

    meta = dict(site=np.array(sites), age=np.array(ages, float),
                sex=np.array(sexes, float),
                subject_id=np.array(ids, dtype=object))
    outdir = Path(args.out_dir)

    # --- raw control: connectivity, no encoder ------------------------------
    rois, subs = valid_rois(series, min_frac=args.roi_min_frac)
    n_roi_total = min(s.shape[1] for s in series)
    print(f"[raw] {len(rois)}/{n_roi_total} ROIs alive in >={args.roi_min_frac:.0%} "
          f"of subjects; {len(subs)}/{len(series)} subjects retain all of them")
    if len(rois) < 2:
        sys.exit("no usable shared ROI set; try --roi-min-frac 0.95 or --atlas ho")
    fc, keep = [], []
    for j in subs:
        v = connectivity(series[j], rois)
        if v is not None:
            fc.append(v)
            keep.append(j)
    if fc:
        F = np.stack(fc)   # same ROI pairs, same order, every subject
        m = {k: v[keep] for k, v in meta.items()}
        np.savez(outdir / f"multilayer_fmri-raw-{args.atlas}.npz",
                 emb_L0=F, **m)
        print(f"[raw]    wrote multilayer_fmri-raw-{args.atlas}.npz  "
              f"{F.shape[0]} subjects, dim {F.shape[1]}")

    # --- negative control on the representation a model actually consumes ---
    if fc:
        for seed in args.seeds:
            enc = RandomFCEncoder(in_dim=F.shape[1], seed=seed)
            stages = None
            for v in F:
                e = enc(v)
                if stages is None:
                    stages = [[] for _ in e]
                for l, s in enumerate(e):
                    stages[l].append(s)
            data = {f"emb_L{l}": np.stack(stages[l]) for l in range(len(stages))}
            data.update({k: v[keep] for k, v in meta.items()})
            np.savez(outdir / f"multilayer_fmri-randfc-s{seed}.npz", **data)
            print(f"[randfc] seed {seed}: wrote multilayer_fmri-randfc-s{seed}.npz  "
                  f"{F.shape[0]} subjects")

    # --- negative control: random-init temporal encoder ---------------------
    for seed in args.seeds:
        enc = RandomTemporalEncoder(n_rois=series[0].shape[1], seed=seed)
        stages, keep = None, []
        for j, ts in enumerate(series):
            if ts.shape[1] != series[0].shape[1]:
                continue
            e = enc(ts)
            if e is None or not all(np.isfinite(v).all() for v in e):
                continue
            if stages is None:
                stages = [[] for _ in e]
            for l, v in enumerate(e):
                stages[l].append(v)
            keep.append(j)
        if not stages or len(keep) < 40:
            print(f"[random] seed {seed}: too few usable subjects, skipped")
            continue
        data = {f"emb_L{l}": np.stack(stages[l]) for l in range(len(stages))}
        data.update({k: v[keep] for k, v in meta.items()})
        np.savez(outdir / f"multilayer_fmri-random-s{seed}.npz", **data)
        dims = ",".join(str(np.stack(stages[l]).shape[1]) for l in range(len(stages)))
        print(f"[random] seed {seed}: wrote multilayer_fmri-random-s{seed}.npz  "
              f"{len(keep)} subjects, dims [{dims}]")

    # The probe lives beside this file, wherever that is. Printing a fixed
    # "src/..." path sent the reader to a directory that does not exist in
    # every working copy.
    here = Path(__file__).resolve().parent
    print("\nNow run the existing probe over these, unchanged:")
    print(f'  GLOB="multilayer_fmri-*.npz" python {here}/scanner_dominance.py')
    print("  (or submit slurm/submit_fmri_arm.sh, which locates it itself)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abide-root", default=os.environ.get("ABIDE_ROOT"),
                    help="directory holding the phenotype CSV")
    ap.add_argument("--pheno-csv", default=os.environ.get("PHENO_CSV",
                                                          "Phenotypic_V1_0b.csv"))
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--cache-dir", default="abide_fmri_rois")
    ap.add_argument("--atlas", default="cc200",
                    help="cc200, ho, aal, dosenbach160, ...")
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--roi-min-frac", type=float, default=0.99,
                    help="keep parcels alive in at least this fraction of "
                         "subjects, then drop subjects missing any of them")
    ap.add_argument("--n-per-site", type=int, default=0,
                    help="cap subjects per site; 0 keeps all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="random-encoder seeds; more than one shows the control "
                         "is not a property of one lucky initialisation")
    args = ap.parse_args()
    if not args.abide_root:
        sys.exit("set --abide-root or ABIDE_ROOT")
    return build(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
