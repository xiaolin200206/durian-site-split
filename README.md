# What a new orchard costs

Code, result tables and manuscript for **"What a new orchard costs: farm-level
generalisation and data budgets for on-device durian disease and pest
detection"** (Lin Ding Shan; submitted to *Computers and Electronics in Agriculture*).

A handheld durian disease-and-pest detector is used on farms it was never
trained on. This repository measures what that costs, and how a limited
field-collection budget should be spent to reduce it, on 827 lesion-annotated
photographs from eight commercial farms in Peninsular Malaysia and 281 from two
orchards in Sabah, every image tagged with its farm.

## Reproduce every number

```bash
pip install -r requirements.txt        # plus pandoc for the .docx
python verify_applied.py --full
```

`verify_applied.py` does three things and exits non-zero if any check fails:

1. re-derives `results_applied/paper_numbers.json` from the raw per-run tables
   with `scripts/applied/paper_analysis.py`, and recomputes the key numbers a
   second time with independent code;
2. re-renders `paper/manuscript.md` from `paper/manuscript_template.md` and
   checks it is identical — every number in the text is a placeholder filled
   from the JSON, and every table is built from a CSV, so nothing numeric is
   typed by hand;
3. checks the journal's rules and internal consistency: abstract length,
   highlights, keywords, every citation against the reference list, figure and
   table order.

It runs on CPU in about a minute and reads no images or weights. CI runs it on
every push.

## What it found

| | |
|---|---|
| Random split → unseen farm, YOLO11n (deployed) | 0.428 → 0.223 mAP50 (−47.8%) |
| Same, six detectors from three families | −39.3% to −55.8% |
| Spread across held-out farms (mean of six detectors) | 0.094 – 0.356 |
| Largest mean difference between two detectors | 0.022 mAP50 |
| Share of random-split AP kept on a new farm | psyllid classes 25–30%, other classes 48–59% |
| Doubling training farms vs doubling photos per farm (fixed iterations) | +0.038 vs +0.028 mAP50 (difference within uncertainty) |
| …restricted to training sets covering every class | +0.035 vs +0.030 |
| 7 farms × 50 photos vs 4 farms × all photos (~350–390 images) | 0.182 vs 0.146 |
| Fine-tuning on a new farm's images alone (5 / ~50 images) | −0.111 / −0.029 mAP50 |
| Retraining with those images added (5 / ~50 images) | +0.004 / +0.053 mAP50 |

The numbers above are the seed-1/42 data-budget run; `paper/manuscript.md` is
always the current rendering.

## Layout

```
paper/
  manuscript_template.md    text with {{placeholders}} and [[TABLE:…]] markers
  render.py                 template + JSON + CSVs -> manuscript.md/.docx, highlights, cover letter
  render_supp.py            supplementary.md/.docx
  manuscript.md / .docx     rendered manuscript
  supplementary.md / .docx  Tables S1–S7, Notes S1–S2
  highlights.*, cover_letter.*, declaration_of_interest.*
results_applied/
  farm_budget.csv           data-budget experiment, one row per run × evaluation set
  paper_numbers.json        every number the text quotes
  table*.csv, fig*.png/.pdf tables and figures of the paper
results_clean/              leave-one-farm-out and random-split runs (clean protocol)
results_durian/split_assignment.csv   image -> farm, classes, split manifests
scripts/applied/
  farm_budget.py            data-budget experiment: build / train / eval (GPU)
  new_farm_calibration.py   new-farm calibration experiment: build / train / eval (GPU)
  paper_analysis.py         all tables, figures and paper_numbers.json
  per_class_transfer.py     per-class transfer table (no training)
scripts/clean/              training scripts for the leave-one-farm-out runs
verify_applied.py           the check above
archive/withdrawn_methods_paper/   the earlier, broader manuscript (see below)
```

## Re-running training

Training needs a GPU; nothing in the reproduction path depends on it.

```bash
# data-budget experiment (336 YOLO11n runs for two seeds; ~5–10 GPU-hours per seed on an RTX 4090)
python scripts/applied/farm_budget.py --root /path/to/durian --step build
python scripts/applied/farm_budget.py --root /path/to/durian --step train --yes --seeds 42
python scripts/applied/farm_budget.py --root /path/to/durian --step eval
```

`--root` must contain `split_assignment.csv` and `dataset_store/merged_peninsula/`
(and optionally `merged_sabah/`) in YOLO format, as in the Zenodo deposit.
Every run keeps the held-out farm out of training, validation, early stopping
and checkpoint selection.

## Data

Images and annotations: https://doi.org/10.5281/zenodo.22030622 (CC BY-NC 4.0),
640 × 640, location coordinates removed, with farm attribution and split manifests.

## The earlier manuscript

The leave-one-farm-out runs were first analysed, together with GWHD 2021,
BreaKHis and UCI HAR, in a methodological manuscript ("Held-out unit
variability overshadows architectural differences in clustered machine-learning
benchmarks") that was withdrawn before review. Its manuscript, supplementary
information, figures and README are in `archive/withdrawn_methods_paper/`, and
`python verify_claims.py` still reproduces its 185 claims from the tables in
`results_*/`. The applied paper replaces it for the durian results.

## Licence

Code: see `LICENSE`. Data: CC BY-NC 4.0 (Zenodo deposit).
