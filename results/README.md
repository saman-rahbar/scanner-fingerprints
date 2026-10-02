# Results

Summary scores behind the paper's figures and tables. They contain no images
and nothing that identifies a person: only per-model, per-layer scores.

| File | What it holds |
|---|---|
| `abide1_dominance_50splits_1p0mm.json` | ABIDE-I, 546 people, 1.0 mm run. Site, sex, age and diagnosis scores for each model and layer, as the mean and 90% interval over 50 splits, for the linear probe and the small neural-network probe. Source of Figure 2 and the resolution table. |
| `abide1_dominance_single_split.json` | The same models and targets on one split. Used to show how much a single split can mislead (paper Section 4.4). |
| `fmri_dominance.json` | Resting-state fMRI, ABIDE, six matched sites. The raw connectivity input and two untrained models over three seeds, one split each. Source of Figure 5b and the fMRI table. |

Scores for site, sex and diagnosis are balanced accuracy corrected for chance
(0 is guessing, 1 is perfect). Age is R². Produced by `src/scanner_dominance_ci.py`
and `src/scanner_dominance.py`.
