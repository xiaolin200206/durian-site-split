# Durian disease detection: what a single accuracy figure measures

Code and result tables for **"A single accuracy figure for crop disease
detection describes the sample, not the model"**.

The paper asks what an aggregate mAP figure actually reports when the
images come from the field. Three answers, in order of how much they cost
to establish:

- Withholding a whole farm instead of splitting images at random lowers
  mAP50 from 0.478 to **0.246** — the random-split figure is **94%** higher,
  or equivalently the corrected figure is **48.5%** lower — an effect **6.4×**
  larger than training-seed noise, because photographs of one leaf land on
  both sides of the boundary.
- The corrected figure is not a property of the model. It ranges **2.8×**
  across withheld farms, and at individual capture sites from zero to 0.864.
- Crossing 1,800 km to Borneo costs **nothing** beyond withholding a
  neighbouring farm (0.246 against 0.246). Distance is not the operative
  variable; how many independent management units were pooled is.

Everything here recomputes from `results/`. No image or model weight is
needed to check a number in the paper.

---

## Reproduce the paper's numbers

```bash
pip install -r requirements.txt
python verify_claims.py            # 113 claims, checked against results/
python verify_claims.py --verbose  # print each one
```

Then the figures, which need the resampling step first:

```bash
python scripts/analyse/aggregation_curve.py
python scripts/analyse/make_figures.py
```

The script states each claim as it appears in the manuscript, recomputes it
from the tables, and exits non-zero on any mismatch. If you edit a number in
the text without re-deriving it, this fails.

---

## Repository layout

```
results/                    the tables the paper is written from
  in_region.csv             every model on its own validation set
  cross_island.csv          the same models on the Sabah held-out set
  sabah_by_tree.csv         Sabah scored one tree at a time
  sabah_by_orchard.csv      Sabah scored one orchard at a time
  peninsula_by_burst_15s.csv   peninsula scored one capture burst at a time
  split_assignment.csv      per image: farm, attribution method, split membership
  farm_annotation_scale.csv    median box area per class per farm
  farm_capture_conditions.csv  hour, solar elevation, device, ISO, focal length
  runs_summary.csv          per run: epochs, best epoch, losses, weight hash
  dataset_stats.json        image and box counts per class per region

manuscript.md / .docx       the paper
scripts/prepare/            corpus -> analysable dataset
  heic_to_jpg.py            iPhone HEIC to JPEG, EXIF preserved
  exif_audit.py             metadata inventory; GPS clustering into farms,
                            with a capture-time fallback for the second phone
  rename_with_tree_id.py    embed the tree id in the filename before upload
  merge_regions.py          merge per-class exports into one label space,
                            and verify both regions share it exactly
  make_splits.py            random and by-farm GroupKFold splits

scripts/analyse/            the experiments
  colab_run.py              one-file driver: unpack, split, train, evaluate
  multiseed.py              every configuration across seeds
  sabah_by_tree.py          per-tree and per-orchard evaluation
  peninsula_by_burst.py     per-burst evaluation on held-out weights
  aggregation_curve.py      how the figure stabilises as sites are pooled
  make_figures.py           the three manuscript figures

docs/                       everything the manuscript needs and does not have
  related_work.md           section to insert, and the edits it forces
  references.bib            with VERIFY flags on unchecked entries
  figure_captions.md        captions plus a new Results subsection
  dataset_provenance_survey.md   Table S1, half-filled, with a protocol

figures/                    fig1 per-site, fig2 aggregation, fig3 coverage

scripts/diagnostics/        checks that changed what the paper says
  farm_covariates.py        capture conditions and annotation scale per farm
  check_lesion_scale.py     render annotations to test lesion size vs distance
  shortcut_check.py         run the detector on healthy trunks as negatives
  package_results.py        collect everything into one archive
```

---

## The pipeline, in the order it was run

**1. Prepare.** Convert HEIC, audit EXIF, cluster GPS into farms.

```bash
python scripts/prepare/heic_to_jpg.py
python scripts/prepare/exif_audit.py /path/to/photos -o metadata.csv
```

`exif_audit.py` attributes a farm to every image it can. Images with GPS are
clustered at 1.5 km. Images without — one handset had location services off
— inherit the farm of the nearest GPS-bearing image of the same day, if
within 30 minutes, on the reasoning that a multi-farm day is a sequence of
visits. The attribution method is recorded per image so the inferred ones
can be excluded; `make_splits.py` builds a `gps_only` set of folds for
exactly that sensitivity check.

**2. Merge.** Each class was annotated in its own project, so one photograph
appears in several exports under different platform-assigned filenames.

```bash
python scripts/prepare/rename_with_tree_id.py --apply
python scripts/prepare/merge_regions.py
```

`merge_regions.py` recovers the original filename, merges the label files,
remaps each project's local indices onto one canonical list, and drops boxes
duplicated across projects at IoU ≥ 0.90. It refuses to proceed unless both
regions end up with the same class list **in the same order**: the integer
in a label file means nothing on its own, and an order mismatch silently
relabels every box. Names differing only in case are folded together, a
failure mode we had already been bitten by once.

**3. Split.**

```bash
python scripts/prepare/make_splits.py
```

Two regimes over one pool of 560 images, so the split rule is the only
difference between them. Sabah is never split; all 281 images are held out
for every peninsular model.

**4. Train and evaluate.**

```bash
python scripts/analyse/colab_run.py --step all --batch 32 --seeds 42 1 2 3 4
```

30 runs, 188 minutes on one A100. Every reported figure comes from
re-validating `best.pt` through one code path, so no number depends on the
training loop's checkpoint-selection logic.

**5. Per-site evaluation.** This is where the paper's argument is.

```bash
python scripts/analyse/sabah_by_tree.py
python scripts/analyse/sabah_by_tree.py --by orchard
python scripts/analyse/peninsula_by_burst.py
```

Peninsular filenames carry no tree id, so images are grouped into capture
bursts — consecutive frames under 15 s apart. Each burst is scored with the
fold whose validation set contains that burst's farm. Scoring bursts with
random-split weights measures fit rather than generalisation and is not
comparable to the Sabah figures; the script enforces the pairing rather than
trusting the caller.

---

## Diagnostics that changed the paper

Three checks are in `scripts/diagnostics/` because each one altered a claim.

**`farm_covariates.py`** tests whether capture conditions explain the worst
fold. They do not: farm 2 is 99% midday, but farms 3 and 5 are 100% midday,
farm 5 at a higher solar elevation, and both score better. The same script
found what does hold — median lesion area for one class differs **93-fold**
between farms, because early and spread presentations share a label.

**`check_lesion_scale.py`** tests the obvious alternative, that the small
lesions are simply photographed from further away. They are not: the frames
show the same framing at both farms. This script also prints a distance
surrogate — the largest annotated box per image — that appears to support
the distance account and does not survive looking at the pictures, because
it is measuring class composition. It is kept in the repository for that
reason.

**`shortcut_check.py`** runs the detector over 100 photographs of healthy
trunks, where every detection is wrong by construction, to test whether a
class was keyed on capture-source characteristics. Pink disease, the class
we suspected because part of its data came from elsewhere, produces no
detection above confidence 0.1. Two other classes did fire, and inspection
showed they were right and the images were not as healthy as labelled — a
result about the negatives, not the model.

---

## What the tables do not contain

- **Images and weights.** 30 checkpoints are ~600 MB and the numbers are
  what you need. `runs_summary.csv` records a hash per checkpoint.
- **The 473 excluded peninsular images.** Of 1,033 annotated, 473 have no
  recoverable location — an early batch arrived by instant messaging, which
  strips EXIF, and a second handset had location off. They enter neither
  pool. They are not a random sample and the direction of any resulting bias
  is unknown.
- **Disease stage.** Nothing here records it, which is the point of one of
  the paper's findings. It appears only as a distribution of box sizes.

## Numbers that should not be interpreted

`leaf_hopper_damage` has 134 boxes in the peninsula and 12 in Sabah, from
two trees. It appears in the tables and in one line of the paper, as a
contrast between what two evaluation protocols would lead a reader to
believe — 0.980 under a random split, 0.057 in Borneo. It is not a
measurement of anything on its own.

Subsets below five images are reported and flagged in the per-site tables.
Several exist. Treat them as noise.

## Data

Sabah images and annotations, the minimum needed to verify the central
comparison, are at [DOI] under CC BY-NC 4.0, with GPS removed from EXIF.
Peninsular images are at the same DOI; peninsular annotations are under
embargo until [date] and available to editors and reviewers on request.

## Start here

`HANDOVER.md` records what is done, what is outstanding, and two conclusions
that were held confidently and later withdrawn. Read those two before
editing the manuscript.

## Citation

```bibtex
@article{lim2026pooled,
  title  = {A single accuracy figure for crop disease detection
            describes the sample, not the model},
  author = {Lim, Ding Shan},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Licence

Code MIT. Data CC BY-NC 4.0.
