# Held-out unit variability overshadows architectural differences in clustered machine-learning benchmarks

Code and result tables for the manuscript of the same name: a measurement, on
four corpora from three disciplines, of how much of a held-out score belongs to
the model and how much belongs to which units happened to be withheld.

## Reproducing the paper

```bash
pip install -r requirements.txt
python verify_claims.py
```

That recomputes **all 185 quantitative claims in the manuscript** from the
released result tables and prints one line per claim. It runs on CPU in a few
minutes, reads no images and loads no model weights. A claim whose input table
is missing is reported as PENDING and the script exits non-zero, so a number
cannot enter the manuscript before it can be recomputed.

Two other entry points:

```bash
python scripts/analyse/cross_audit.py --root . \
    --manuscript manuscript_ncs_submission.md --supplementary supplementary.md
python scripts/analyse/make_figures_clean.py --root . --out figures/
```

The first checks the manuscript against the supplementary information for
unresolved cross-references, for quantities stated in more than one place with
different values, for leftover drafting markers and for the journal's length
limits. The second regenerates all five figures from the same summary tables
the text quotes, so a figure cannot disagree with the prose.

Retraining is a separate matter and is not part of the above. The 512 model
runs need GPUs and several days; the scripts are in `scripts/clean/` and the
split manifests they consume are released, but nothing in the reproduction
path depends on them.

## What the measurement found

| Dataset | Modality | Task | Unit | Units | Models | Overstatement | Unit share | Model share |
|---|---|---|---|---|---|---|---|---|
| Durian | images | detection, 6 classes | farm | 8 | 6 | 39.3–55.8% | 82.4% | 0.0% |
| GWHD 2021 | images | detection, 1 class | session | 47 | 2 | 22.5–29.1% | 68.0% | 0.3% |
| BreaKHis | images | classification | patient | 81 | 2 | 8.1–9.2% | 88.1% | 0.0% |
| UCI HAR | inertial | classification | subject | 30 | 2 | 3.7–4.5% | 73.3% | 2.7% |

*Overstatement* is how much an item-level random split inflates the reported
figure relative to a unit-disjoint one. *Unit share* and *model share* are
variance components of the reported figure. Ten model configurations across
five architecture families; 512 training runs under the clean protocol.

Four findings:

- **Splitting by item rather than by unit overstates the figure by 3.7 to
  55.8%**, in an order that follows the coefficient of variation across units.
  Within an architecture family the overstatement is similar; across families
  it is not — five one-stage durian detectors overstate by 47.8–55.8% and a
  two-stage Faster R-CNN by 39.3%.
- **The corrected figure is not stable either.** Scored one unit at a time and
  averaged over each dataset's models, the eight durian farms run 0.094–0.356,
  the 47 wheat sessions 0.085–0.803, the 81 patients 0.053–1.000 and the 30
  subjects 0.750–0.999. The aggregate reports none of that.
- **Which unit is scored accounts for 68–88% of the variance** against 0.0–2.7%
  for the model. Across all 57 subsets of the six durian configurations the
  unit share stays between 70% and 93% and the model share never exceeds 1.1%,
  so the result is not an artefact of comparing similar models.
- **Four of eighteen model pairs are resolved** by a bootstrap of the paired
  difference over units; among the image datasets none of the four crosses an
  architecture family.

## The evaluation protocol

Under leave-one-unit-out as commonly implemented, the withheld unit is also the
training loop's validation set: scored every epoch, triggering early stopping,
selecting which weights are kept. It never enters a gradient, but it enters
model selection.

Every image model here was retrained with the inner validation set drawn only
from training data, so the withheld units enter no training decision. The
earlier, contaminated results are kept for comparison — the difference between
the two protocols raises the measured overstatement by up to 14.7 points, and
on one corpus it reverses which of two detectors wins.

| Directory | Protocol |
|---|---|
| `results_clean/` | clean — withheld units in no training decision |
| `results_durian/`, `results_gwhd/`, `results_breakhis/` | contaminated, retained for the protocol comparison |
| `results_har/` | no correction needed; the perceptron runs a fixed number of iterations and the forest has no early stopping |

## Layout

```
verify_claims.py                  185 claims, recomputed from the tables
unitcheck.py                      standard library only; does a split leak?
results_clean/summary_*.csv       every number the manuscript quotes
scripts/analyse/                  recomputation, figures, audits
scripts/clean/                    the clean-protocol training scripts
docs/manuscript_ncs.md            working copy, with TODO markers
manuscript_ncs_submission.md      submission copy, markers removed
supplementary.md                  17 tables, 1 figure, 4 notes
docs/review_log_2026-09-14.md     what four rounds of review changed
docs/checking.md                  how this manuscript is checked
docs/superseded/                  earlier framings and the old protocol
```

## unitcheck

```bash
python unitcheck.py assignments.csv --unit farm --split split
```

Takes a table of item-to-unit assignments and, where available, per-unit
scores. Reports how many units straddle the split boundary, how many units each
side rests on, and the dispersion across them. Standard library only; reads no
images and no weights. Under the item-level protocols used in this paper it
finds every unit straddling, on all four datasets.

## What the results do not license

The full list is Supplementary Note 4. The short version: eight farms is few,
and the durian resampling analyses draw from that pool; the variance components
carry wide bootstrap intervals, and resampling units does not remove the
dependence from overlapping training sets; the separation of measurement noise
from between-unit heterogeneity is conditional on the within-unit correlation
and, on the wearable corpus, on a value adjacent windows are unlikely to
satisfy; the training-unit dose-response predates the protocol change and is
preliminary; the model rosters are two capacities of one family on two of the
four corpora; and every result is an offline evaluation.

Three errors found in our own pipeline are reported in the paper rather than
quietly repaired — a filename join that placed 8.9% of the durian corpus under
the wrong farm, a lesion-size claim true of per-farm medians and false of the
images, and the checkpoint-selection problem above. Two were found by the
scripts in this repository rather than by inspection.

## Data

The durian corpus is at https://doi.org/10.5281/zenodo.22030622 (CC BY-NC 4.0):
images at 640 × 640 with coordinates stripped, content-based farm attribution,
tree identifiers, and all split manifests. GWHD 2021, BreaKHis and UCI HAR are
public; the scripts that reproduce our processing of them are here.

This repository is archived at https://doi.org/10.5281/zenodo.22764830.

## Citation

Lin, D. S. Held-out unit variability overshadows architectural differences in
clustered machine-learning benchmarks (manuscript in preparation, 2026).
