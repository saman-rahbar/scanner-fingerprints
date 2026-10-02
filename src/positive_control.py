#!/usr/bin/env python
"""Does pretraining ever beat random initialisation? The control's own control.

The audit's central move is a negative control: if a randomly initialised
encoder decodes site as well as a pretrained one, the pretrained number was not
telling us about pretraining. That argument only carries weight if the control
is capable of coming out the other way. A control that fires on every target,
whatever the target, is not evidence about site; it is evidence that the
comparison is too blunt to distinguish anything.

So this asks the opposite question of the same numbers. Across every target and
every depth already measured, is there anywhere the pretrained encoders exceed
the random one by a margin larger than the noise in the estimate? A single
convincing cell is enough: it shows the comparison has power, and that the
site result is a finding rather than an artifact of the method.

Reads the JSON that scanner_dominance.py already writes, so it costs nothing to
run and cannot disagree with the reported tables.

  python positive_control.py scanner_dominance_results.json

The honest outcomes are two, and both are reported plainly:

  pretraining wins somewhere   the negative control discriminates, and the site
                               result stands as a claim about site.
  pretraining never wins       state it. Either these checkpoints add nothing
                               on any measured target, which is itself worth
                               reporting, or the targets are too weak to detect
                               an advantage, which limits what the negative
                               control can support. Do not report the site
                               result as though this question had been settled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Substrings that identify an encoder as untrained. Everything else is treated
#: as pretrained, which is the conservative direction: mislabelling a random
#: encoder as pretrained can only make a pretraining advantage harder to find.
RANDOM_MARKERS = ("random", "rand", "randfc", "init")

#: Targets worth asking the question of. Site is deliberately excluded: it is
#: the thing the paper argues pretraining does *not* drive, so including it here
#: would be circular.
TARGETS = ("sex", "age_r2", "asd")

#: A difference smaller than this is inside the repeated-holdout spread reported
#: alongside the main tables (+/- 0.03 to 0.05), so it is not evidence of
#: anything. Set deliberately at the top of that range.
MARGIN = 0.05


def is_random(tag: str) -> bool:
    return any(m in tag.lower() for m in RANDOM_MARKERS)


def load(paths):
    merged = {}
    for p in paths:
        f = Path(p)
        if not f.is_file():
            sys.exit(f"not found: {p}")
        merged.update(json.loads(f.read_text()))
    return merged


def _value(row: dict, key: str):
    """A target's score from either result format: scanner_dominance writes
    plain floats, scanner_dominance_ci writes {"linear": {key: {"mean": ...}}}."""
    if "linear" in row:
        v = row["linear"].get(key)
        return v.get("mean") if isinstance(v, dict) else v
    return row.get(key)


def compare(results: dict):
    randoms = {t: r for t, r in results.items() if is_random(t)}
    trained = {t: r for t, r in results.items() if not is_random(t)}
    if not randoms:
        sys.exit("no randomly initialised encoder in these results; the "
                 "positive control needs one to compare against")
    if not trained:
        sys.exit("no pretrained encoder in these results")

    rows = []
    for ttag, tres in sorted(trained.items()):
        for layer, trow in tres.get("layers", {}).items():
            # Best random encoder at this layer, so the comparison is against
            # the strongest untrained baseline rather than a lucky weak seed.
            best_rand = {}
            for rres in randoms.values():
                rrow = rres.get("layers", {}).get(layer)
                if not rrow:
                    continue
                for k in TARGETS:
                    v = _value(rrow, k)
                    if v is not None and v == v:          # not NaN
                        best_rand[k] = max(best_rand.get(k, 0.0), float(v))
            for k in TARGETS:
                tv, rv = _value(trow, k), best_rand.get(k)
                if tv is None or rv is None or tv != tv:
                    continue
                rows.append((ttag, layer, k, float(tv), rv, float(tv) - rv))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+",
                    help="scanner_dominance_results.json (one or more)")
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    rows = compare(load(args.results))
    if not rows:
        sys.exit("no comparable target/layer cells found")

    wins = [r for r in rows if r[5] >= args.margin]
    rows.sort(key=lambda r: -r[5])

    print(f"\n{'encoder':22s} {'layer':7s} {'target':7s} "
          f"{'pretrained':>10s} {'best random':>12s} {'delta':>7s}")
    print("-" * 72)
    for tag, layer, k, tv, rv, d in rows[:12]:
        flag = "  <- pretraining wins" if d >= args.margin else ""
        print(f"{tag[:22]:22s} {layer:7s} {k:7s} {tv:10.3f} {rv:12.3f} "
              f"{d:+7.3f}{flag}")

    print(f"\n  cells compared: {len(rows)}   "
          f"pretraining ahead by >= {args.margin}: {len(wins)}")
    if wins:
        best = rows[0]
        print(f"\n  Pretraining is ahead in {len(wins)} of {len(rows)} cells. Largest "
              f"advantage: {best[0]} at {best[1]} on {best[2]}, {best[3]:.3f} "
              f"against {best[4]:.3f} ({best[5]:+.3f}).")
        print("  Check this on repeated splits before relying on it: on one split, "
              "clinical probes on a cohort of this size can show advantages that "
              "do not survive averaging.")
    else:
        print("\n  No cell where pretraining is ahead by the margin.")
        print("  Report this rather than omitting it. Pretraining does not "
              "measurably beat random initialisation on any measured target at "
              "any depth, which either says these checkpoints add nothing here "
              "or that the available targets are too weak to detect it. Both "
              "readings limit what the negative control alone can support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
