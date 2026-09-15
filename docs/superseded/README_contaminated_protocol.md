# Evaluation-unit sampling can dominate reported machine-learning performance

Code and result tables for the manuscript of the same name: a measurement, on
four datasets, of how much of a held-out score belongs to the model and how
much belongs to which units happened to be sampled.

```
                modality   task            unit       units  overstatement
Durian          images     detection       farm           8  37.4-44.3%
GWHD 2021       images     detection       session       47  19.6-22.4%
BreaKHis        images     classification  patient       81   8.0- 8.7%
UCI HAR         inertial   classification  subject       30   3.7- 4.5%
```

Eight architectures, ten dataset-model combinations, 439 training runs.

Five findings:

- **The correction costs 3.7% to 44.3%**, ordered across datasets the same way
  the between-unit dispersion is, and nearly invariant to the model within a
  dataset. Near the ceiling the error rate is the honest statement: on
  histopathology it rises 13.5-fold, on activity recognition 4.2-fold.
- **The corrected figure is not stable either.** Scored one unit at a time,
  every dataset shows wide and practically consequential variation: 58 durian
  bursts run 0.000-0.995, 47 GWHD sessions 0.173-0.767, 81 BreaKHis patients
  0.062-1.000, 30 HAR subjects 0.756-0.998. The aggregate reports none of it.
- **The held-out sample accounts for 88.9% of the variance** in the reported
  durian figure against 1.1% for architecture and 4.7% for seed, over four
  models spanning three capacities and one family change. It is 78.7% against
  0.4% on histopathology and 56.1% against 14.6% on activity recognition,
  where the two models share only their input features. GWHD reverses: seed
  61.2%, sample 38.8%, and that is reported rather than set aside.
- **Training units and evaluation units obey different rules.** With training
  set size fixed, adding *training* units narrows the draw-to-draw spread
  three- to six-fold and may or may not raise the mean (+61.5% on durian,
  +6.6% on HAR, no trend on GWHD). Adding *evaluation* units never moves the
  mean at all and narrows the spread as the inverse square root. Training-unit
  diversity buys model quality; evaluation-unit diversity buys measurement
  precision. Neither substitutes for the other.
- **A held-out set that picks the checkpoint has stopped being held out.**
  Under leave-one-unit-out the fold's validation set is the withheld unit
  itself. Re-validating final-epoch weights on the seven dataset-model
  combinations where they survived raises the overstatement by 2.4 to 17.3
  points, in the same order as the partitioning effect. Every figure in the
  paper is therefore the conservative one.

And one negative result that matters most for deployment: **nothing we
recorded predicts which unit will fail** — not eight capture covariates, not
the amount of data per unit, and not the model's own confidence, which
correlates with per-farm score at r = +0.05 over eight farms.

Durian dataset: https://doi.org/10.5281/zenodo.22030622
This repository: https://doi.org/10.5281/zenodo.22031684

---

## The errors this repository also documents

Three faults were found inside this project's own pipeline. Each is an
instance of the mechanism the paper is about, which is why they are in the
manuscript's main text rather than a footnote.

**A filename join misattributed 8.9% of the durian pool.** Two field visits
produced overlapping camera counters; the second import received `(1)`
suffixes; the resize step that produced the 640-pixel annotation set dropped
them. Fifty of 560 images carried another photograph's farm, 49 of them
labelled farm 0 when taken at farm 6. The fold that withheld farm 0 had
trained on farm 6, so 29.7% of its held-out set came from a farm it had
trained on; it scored 0.394, the highest of five. Found by hashing image
content against the originals (`scripts/prepare/match_by_phash.py`,
`audit_pool_join.py`), not by inspection. Attribution is now content-based
(`rejoin_by_content.py`); the same procedure recovered 303 images whose
filenames the annotation platform had replaced, raising the pool from 560 to
827 images and from five usable farms to eight. Under the corrected
attribution that farm scores 0.260.

**A focal-length claim was true of per-farm medians and false of images.** The
2.22 mm ultra-wide lens appears at *every* farm, from 2% to 65% of a farm's
images, and within a single farm it changes annotated Leaf_rot box area 1.8-
to 4.0-fold. The unstratified lesion-scale ratio of 75-fold falls to 15-fold
within one lens. The check that should have caught this was testing the
median.

**The checkpoint was chosen on the unit it was then scored on.** Measured
rather than only declared: `checkpoint_bias.py` and
`scripts/analyse/checkpoint_bias_bkh.py` re-validate the final-epoch weights,
which no validation signal selected. 200 re-validations, no retraining.

A fourth claim was withdrawn outright. A correlation between lesion size and
fold score (r = +0.54 on five farms) does not survive on eight (r = +0.29);
`verify_claims.py` fails if it returns.

---

## The tool

`unitcheck.py` runs the audit in this paper on any dataset with a unit
identifier. Standard library only, no model, no images.

```bash
# before training: does an item-level split leak?
python unitcheck.py --assignment items.csv --unit patient --split fold

# after training: what does the reported figure hide?
python unitcheck.py --assignment items.csv --unit patient --split fold \
    --scores per_patient.csv --score-unit patient --metric accuracy \
    --n-col n_images

# contrast the two partitions directly
python unitcheck.py --assignment items.csv --unit patient \
    --split random_fold --grouped-split patient_fold
```

It ends with the two numbers this paper argues should accompany any aggregate
score: how many independent units contributed, and the dispersion across
withheld units. The first is answerable before a single model is trained.

On this repository's own tables:

```bash
python unitcheck.py --assignment results_durian/split_assignment.csv \
    --unit farm --split split_random --grouped-split split_farm_fold0
```

reports that 8 of 8 farms straddle the item-level boundary and none straddles
the farm-level one.

## Reproduce every number

```bash
pip install -r requirements.txt
python verify_claims.py             # every claim, against results_*/
python verify_claims.py --verbose   # print each one
python verify_claims.py --todo      # claims whose input table is missing
```

**322 of 322 claims reproduce.** The script states each claim as it appears in
the manuscript, recomputes it from the released tables, and exits non-zero on
any mismatch. A claim whose table does not exist is reported as `PENDING` and
also fails, so a number cannot reach the manuscript before it can be
recomputed. Three of the errors above were found by this process.

Where a per-unit summary could hide the same class of mistake, the checks are
written against item-level facts. The focal-length checks assert that the wide
lens appears at more than one farm, not that the per-farm medians agree,
because the medians agreed while the claim was false.

Two checks assert the argument rather than a number:
`k=16 mean unchanged from k=1` (evaluation units buy precision, not score) and
`spread falls as 1/sqrt(k)` with a finite-population correction. If the
aggregation convention is ever changed, these fail.

```bash
python scripts/analyse/make_figures.py --out figures/
```

---

## Two analyses that need no GPU

```bash
python eval_unit_curve.py --root .
```

Resamples evaluation units from the per-unit tables: for each unit count *k*,
draw *k* units and recompute the aggregate. Every unit's score was produced by
weights that never saw it, so no retraining is needed. Reports both equal-
weight and instance-weighted aggregation, and inverts the curve to give the
number of evaluation units needed for a 90% interval of a given width (32
durian bursts, 13 GWHD sessions, 24 BreaKHis patients, 3 HAR subjects for a
width of 0.10, equal-weight). Writes `results_eval_unit_curve.csv`.

The two aggregations differ in an instructive way. Under equal weighting the
mean is flat in *k*, as it must be. Under instance weighting it drifts from
the equal-weight mean toward the pooled figure, because at *k*=1 a draw *is*
the equal-weight mean and at *k*=N it *is* the pooled figure. The drift is
visible where unit size correlates with score (GWHD +0.056, durian -0.037) and
negligible where it does not (BreaKHis +0.004, HAR +0.001).

```bash
python scripts/analyse/checkpoint_bias_bkh.py --dry-run   # clear the weights first
python scripts/analyse/checkpoint_bias_bkh.py             # 200 re-validations
```

Re-validates BreaKHis final-epoch weights through `bkh_run.val_once`, the same
code path that produced every other figure. It reads each run's `args.yaml`
and refuses to run if the recorded `imgsz` differs from the one passed, rather
than quietly producing a plausible-looking wrong number. (`bkh_run.py`'s
module-level `IMGSZ = 640` is a stale constant; the runs were trained at 224.)

---

## Layout

```
results_durian/     in_region, cross_island, per-burst (four thresholds),
                    per-tree, per-orchard, site_count, abstention,
                    checkpoint_bias, capture conditions, focal composition
results_gwhd/       in_region, gwhd_by_domain, site_count, checkpoint_bias
results_breakhis/   in_region, breakhis_by_patient, checkpoint_bias
results_har/        in_region, har_by_subject, site_count
results_eval_unit_curve.csv    evaluation-unit resampling, both aggregations

scripts/prepare/    corpus -> analysable dataset, one per dataset
  match_by_phash    dHash every annotated image against the originals
  verify_phash      validate on name-matched pairs; unit consensus
  audit_pool_join   find items joined to the wrong original
  rejoin_by_content final attribution: nearest original by content
  make_splits_v2    leave-one-unit-out + item-level, with cross-fold asserts
  prepare_gwhd      parquet -> YOLO, duplicate audit, domain-disjoint folds
  prepare_breakhis  directory -> manifest, patient-disjoint folds

scripts/analyse/
  autodl_run / colab_run_v2     durian training, eval, cross-region
  gwhd_run / bkh_run / har_run  the other three datasets end to end
  per_site_v2                   per-burst and per-tree evaluation
  site_count                    dose-response with item count held fixed
  variance_decomposition        crossed random-effects components
  checkpoint_bias_bkh           BreaKHis final-epoch re-validation
  make_figures                  the five main figures

scripts/diagnostics/
  focal_lesion_check     lesion scale stratified by lens
  capture_conditions_v2  per-farm covariates at image level
  abstention_v2          superseded; see abstention_v3.py in the root
  audit_unmatched        what the unattributed images are
  fix_shadowing          finds project files shadowing the standard library
                         (an inspect.py in the root broke numpy for a week)

scripts/oneoff/         one-time patch and merge scripts, kept for the record

eval_unit_curve.py      evaluation-unit resampling, no GPU
abstention_v3.py        corrected abstention analysis; see the note below
unitcheck.py            the released tool
verify_claims.py        the interface between the tables and the prose
manuscript.md/.docx     the paper
docs/                   manuscript_nmi_analysis.md (restructured), cover
                        letter, revision plan, references, survey
figures/                five main figures, png and pdf
```

### A note on abstention_v2 and v3

`abstention_v2.py` matched detections within class but then discarded the
class before computing AP, pooling all six classes into one ranked curve. That
is a class-agnostic, instance-weighted AP; `mAP50` is the mean of per-class
APs. The two disagree by up to 0.31 on the same farm, in either direction
depending on class composition. `abstention_v3.py` computes AP per class and
averages over the classes present, runs all five seeds, and has a `--check`
flag that compares its output against `results_durian/in_region.csv` farm by
farm.

The confidence and silence statistics are class-independent and were not
affected, so the correlations were recomputed directly against the correct
per-farm mAP50: confidence against score is r = +0.05, not the -0.26 in
earlier drafts, and silence against score is r = +0.32, not +0.61. The
conclusion is weaker in form and stronger in substance — confidence carries no
information about which farm will fail, rather than pointing the wrong way —
and the vivid case survives: the worst farm (0.130) never falls silent, while
the farm that is silent most often scores above the median. The
risk-coverage curve still comes from the pooled computation and will change
when v3 is run.

---

## Rebuilding from scratch

Durian, locally (needs `metadata.csv`, the originals, and `merged_*`):

```bash
python scripts/diagnostics/fix_shadowing.py --root . --fix
python scripts/prepare/match_by_phash.py
python scripts/prepare/verify_phash.py
python scripts/prepare/audit_pool_join.py
python scripts/prepare/rejoin_by_content.py
python scripts/prepare/make_splits_v2.py
python scripts/diagnostics/capture_conditions_v2.py
python scripts/diagnostics/focal_lesion_check.py --all-classes \
    --emit results_durian/farm_focal_annotation_scale.csv
```

On a GPU box:

```bash
python scripts/analyse/autodl_run.py --step train --models yolo11s rtdetr-l --yes
python scripts/analyse/autodl_run.py --step eval  --models yolo11s rtdetr-l --append
python scripts/analyse/autodl_run.py --step cross --models yolo11s rtdetr-l --append
python scripts/analyse/per_site_v2.py --step both --gap 60 --min-images 5
python scripts/analyse/site_count.py --step build --manifest split_assignment.csv \
    --site-col farm --ks 1 2 3 4 6 --draws 6 --holdout 2 --n-train 100
python scripts/analyse/variance_decomposition.py --dir results_durian
python abstention_v3.py --root . --model rtdetr-l --check
```

The three public datasets each have a `prepare_*` and a `*_run` script that
take them from download to result table; see the module docstrings.

Training jobs are ordered seed-first, so an interrupted run still yields one
complete round across every configuration.

**Keep `last.pt`.** The durian YOLO11s runs predate final-epoch weight
retention and neither checkpoint survives, so that one combination is missing
from the checkpoint-bias comparison and cannot be recovered without retraining
the whole row. Every other run kept both.

---

## Known gaps

`HANDOVER.md`. The ones that would change a number are marked in the
manuscript's Limitations; `verify_claims.py --todo` lists any table the checks
still wait on.
