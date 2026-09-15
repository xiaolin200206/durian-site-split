# Self-check

Subject: `manuscript.md` / `.docx` (22 pp.), `supplementary.md` / `.docx`,
`figures/`, and the repository as a whole.

Method: machine checks first, then a line-by-line read, then an honest list of
what I could not verify myself.

---

## 0. Summary

The manuscript is internally consistent and `verify_claims.py` reports
**153 of 153** claims reproducing against the released tables. Three things
still stand between this and a submission, listed in §5, and one of them —
the reference list — only you can close.

The rest of this document records what was checked, what was wrong and fixed,
and where each number came from.

---

## 1. Machine checks

### 1.1 Verification script

```
153 checks   153 reproduce   0 pending   0 mismatches
```

Every quantitative claim in the manuscript is recomputed from a released
table. A claim whose input table is absent is reported as `PENDING` and fails
the build, so a number cannot reach the prose before it can be recomputed.

Coverage: durian 78 checks (both architectures, variance decomposition,
per-burst, per-tree, cross-region, four burst thresholds, dose-response,
abstention, focal stratification), GWHD 14, BreaKHis 13, HAR 15,
cross-dataset 3, split-integrity assertions 30.

### 1.2 Number extraction

Every numeric token in the body was extracted and classified against two
sets: recomputed by me from an uploaded table, or read from your terminal
output. **Unclassified: none**, excluding years, citation indices, DOI
fragments, the email address and the ORCID.

### 1.3 References

```
entries 25   numbering contiguous   cited in text 25   never cited 0
```

The first draft listed nine references that were never cited. One (the
edge-AI roadmap) belonged to the earlier agricultural framing and was
removed; the other eight were given citation points in Related work and
Methods. The list was renumbered and every in-text bracket remapped, with a
check for dangling indices returning zero.

### 1.4 Format

```
em-dash parentheticals   0  (39 in the first draft, converted to commas)
Chinese characters       0
placeholder markers      0
top-level sections      13
figure captions          4, contiguous
table cells damaged by the em-dash pass   6, repaired
```

---

## 2. Errors found by reading, and fixed

These are mistakes I made in drafting, found on a line-by-line pass.

**1. The abstract was 327 words.** Nature-family and IJCV both want 200–250.
Cut to 287 by dropping one per-unit range that the table already gives.

**2. The dose-response wording was not honest.** I had written that the
expected score "rises modestly or not at all, 61.5% on durian". A 61.5% rise
is not modest. Rewritten to state the two effects separately: the mean
responds differently in each dataset (+61.5% / +6.6% / no trend), while the
dispersion collapse is consistent across all three (3.8× / 4.8× / 6.4×).
Figure 3's panel title was changed to match.

**3. "Training set size held fixed" was false.** In the HAR dose-response,
`n_train` runs 1,000 / 992 / 984 — items are allocated evenly across the
drawn units, and where the budget does not divide exactly the total falls
short. It is a 1.6% spread and its direction is opposite to the measured
effect, so it cannot manufacture the result, but it cannot be described as
held constant. Corrected in the manuscript, the supplement and the
verification assertion.

**4. "The best subject makes one error in 600" was wrong.** 1 − 0.9983 =
0.0017, which is one in 575. Changed to 570.

**5. "Thirteen Sabah trees" conflicted with the table's twelve.** Thirteen
were photographed; twelve carry at least five images and are scored. Now
stated explicitly.

**6. Two adjacent paragraphs gave the same explanation.** "A durian farm is
one manager, one cultivar mix, one visit" appeared twice. Merged.

**7. The GWHD dose-response table lacked a seed column** that the other two
had, without explanation. A note was added.

**8. The Discussion cited a per-class result** that had been moved to the
supplement during the reframing. Now points to Supplementary Table 2.

**9. The em-dash pass damaged six table cells.** `| — |`, an empty cell,
became `|, |`. Repaired.

**10. One Chinese character had leaked into the supplement.** Fixed.

**11. Renumbering the references broke the in-text brackets.** After removing
reference 1, every subsequent citation had to shift. Remapped
programmatically and re-checked.

### Found after the first build

**12. GWHD's k = 2 has a single surviving draw.** Its across-draw standard
deviation is undefined; my helper returned 0.0 for a single sample, which on
a log axis is negative infinity and drew a spurious vertical line in Figure 3.
More seriously, the manuscript quoted the GWHD collapse as 0.032 → 0.008,
taking that k = 2 value as the starting point — but 0.0319 is the spread
across *seeds*, not across *draws*. They are not the same quantity. The curve
now starts at k = 4 (0.0366), the factor is 4.8 rather than 4.0, and the
single-draw point is excluded, with the exclusion itself asserted in the
checks.

**13. `make_figures.py` still used the old directory name.** Results were
renamed from `results_v2` to `results_durian`; the verification script was
updated and the plotting script was not. Two panels of Figure 1 and one curve
of Figure 3 were silently omitted while the command still returned success.
This is the failure mode the paper describes: "skip what is missing" avoids
inventing data but makes the absence inconspicuous. The script now reports
when a figure has fewer panels or curves than expected.

**14. Figure 1 mislabelled two lines on the GWHD panel.** Label placement was
decided by whether a line fell below the panel's median score. GWHD is the
only one of the five panels where the equal-weight mean sits below the pooled
figure, so both labels landed in the same place and the figure printed 0.554
for both lines. Labels are now laid out together, anchored left, ordered by
value, and pushed apart when adjacent.

**15. The pooled line and the markers' centre of mass did not agree, with no
explanation.** A pooled mAP is instance-weighted; the markers are one per unit
and equally weighted. On the detection sets they differ materially (0.275
against 0.330 on the durian bursts; 0.554 against 0.504 on GWHD) and on the
classification sets they do not, within 0.004. The figure now draws both, and
the caption and Results text say why.

---

## 3. Synthetic tables removed

While developing the scripts I generated several test tables with the same
filenames as the real ones: `sabah_by_tree.csv`, `sabah_by_orchard.csv`,
`farm_capture_conditions.csv`, `farm_focal_composition.csv`,
`farm_focal_annotation_scale.csv`.

**These were deleted from the repository.** I flag it specifically because had
they stayed in `results_durian/`, `verify_claims.py` would have passed on
fabricated inputs — the class of error this paper is about.

---

## 4. Provenance of every number

Each figure in the manuscript falls into one of three classes.

### Class A — recomputed by me from a table you uploaded

`results_durian/{in_region, cross_island, site_count, abstention_per_image,
abstention_summary, peninsula_by_burst_60s_min5, split_assignment}.csv`,
`results_breakhis/{in_region, breakhis_by_patient}.csv`,
`results_har/{in_region, har_by_subject, site_count}.csv`.

Covers: both architectures' main results; the full variance decomposition with
bootstrap intervals; Spearman ρ = 1.000; all 58-burst statistics; the durian
and HAR dose-response curves; every abstention correlation; the per-unit
distributions for BreaKHis and HAR.

### Class B — read from your terminal output

| Figure | Source |
|---|---|
| All GWHD numbers: 17.6%, 0.6722 / 0.5537, 47 domains, CV 0.311, 8.6× span, six dose-response points | `gwhd_run.py --step eval / per_domain`, `site_count.py` |
| Sabah, 12 trees: mean 0.333, s.d. 0.056, CV 0.167 | `per_site_v2.py --step both` |
| Sabah orchards: o1 0.277, o2 0.329 | same |
| Burst thresholds: 45 / 54 / 103 bursts, CV 0.75 / 0.70 / 0.74 | four runs you pasted |
| Focal stratification: 75.0 → 15.0, 7.5 → 7.0, 3.0 → 3.2; within-farm 1.8–4.0× | `focal_lesion_check.py` |
| Per-farm capture conditions | `capture_conditions_v2.py` |
| Eight covariate correlations | computed by me from the printed capture-conditions values |
| Per-class transfer, six rows | `q.py` |
| GWHD audit: 547 degenerate boxes, one conflicting duplicate, zero cross-split duplicates | `prepare_gwhd.py --step build` |
| The 8.9% misattribution chain | `audit_pool_join.py`, `rejoin_by_content.py` |

All of these now have tables in the repository and the verification passes on
them, so the exposure has largely closed. They are listed because I read them
from printed output rather than computing them, and a transcription error on
my part would not have been caught by me alone.

### Class C — from your documents or statements, not independently verified

- The earlier classification paper's 560 images / 73 sessions / 79.6% / nine
  architectures / sign test *p* = 0.004, read from the manuscript you uploaded.
- Collection sites, dates, orchard areas, tree ages, cultivars, device models.
- The Malaysian pesticide registration groupings.
- **All 25 references. I verified none of them.** The list carries over your
  earlier statement that each was checked against the publisher record. The
  two I added — Anguita et al. 2013 for UCI HAR and Spanhol et al. 2016 for
  BreaKHis — I wrote from memory and **the volume and page numbers must be
  checked**.
- The provenance judgements for eight of the fourteen datasets in
  Supplementary Table 1, carried over from your earlier survey. GWHD, the two
  durian sets and PlantVillage I looked at myself.

---

## 5. Three decisions for you

**5.1 The framing is now an evaluation-protocol paper throughout.** Title,
abstract, introduction and discussion all assume it. Reverting to an
agricultural framing means rewriting those four and moving GWHD, BreaKHis and
HAR to the supplement; Methods, tables and figures are unaffected. A day's
work, but not a decision to leave until the hour before submission.

**5.2 The durian dataset is released under CC BY-NC 4.0.** If the target
journal requires data reusable without restriction, change it on Zenodo before
submitting.

**5.3 The earlier paper is cited as "under review (2026)"** with its Zenodo
DOI, and only for the statement that the effect was positive in all nine
architectures — not for the 12.2-point figure, which could change in review.
If it is accepted before you submit, convert to a full citation.

---

## 6. Figures

Four main figures, PNG and PDF.

```
figure1_per_unit          five panels, per-unit scatter
figure2_variance          variance components with bootstrap intervals,
                          and both architectures' fold rankings
figure3_dose_response     two panels, three curves
figure4_unpredictable     eight covariates, and confidence against score
```

`suppfig1_stage.png` is Supplementary Figure 1, trunk disease by tree over
four days, and stays.

**Three v1 figures must not be submitted**: `fig1_per_site`,
`fig2_aggregation`, `fig3_site_coverage`. They use the superseded numbers
(0.478 / 0.246 / 34 sites / 5 farms). Move them to `figures/archive_v1/`.

Rendering problems fixed during this pass: labels sitting on markers; value
labels covering bootstrap intervals; an annotation covering a bar; the
spurious vertical line from a log of zero; and the mislabelled GWHD lines in
§2.14.

---

## 7. Repository

```
manuscript.md / .docx        the paper, 22 pp.
supplementary.md / .docx     seven tables, one figure, three notes
verify_claims.py             153 checks across four datasets
unitcheck.py                 the audit as a standalone tool
README.md                    rewritten, including a section on what this
                             repository documents about its own errors
HANDOVER.md                  rewritten: what remains, what was decided,
                             what went wrong
figures/                     four main figures, PNG and PDF
results_{durian,gwhd,breakhis,har}/
scripts/{prepare,analyse,diagnostics}/
```

`unitcheck.py` is the piece a reader can take away. It runs the paper's audit
on any dataset with a unit identifier, depends only on the standard library,
reads neither images nor weights, and answers the first question — does an
item-level split leak — before any model is trained. It is described in the
Discussion, its usage is in Supplementary Note 3, and the README leads with it.

Still to tidy: `docs/` holds a cover letter written for a different journal
and a different framing.

---

## 8. The five things a reviewer is most likely to raise

**8.1 Three of the four datasets are images.** HAR breaks the pattern but is
also the only one near its ceiling and the only one using hand-engineered
features. The Discussion says plainly that the two mechanisms cannot be
separated here. A fifth dataset outside vision and outside wearables is the
only real answer, and it is the most likely revision request.

**8.2 The variance decomposition rests on one dataset.** Only durian has a
fully crossed design over two architecture families. A second architecture on
GWHD would cost roughly 30 GPU-hours, on BreaKHis roughly 10.

**8.3 Eight farms.** The bootstrap intervals already expose this: the
architecture share runs 0.3–25.2%. Reporting them is right, but a reviewer
will note that the 86.4% point estimate carries an interval of 56–94%.

**8.4 The 15-fold lesion-scale figure rests on 10 and 14 images.** Stated in
Limitations. A bootstrap interval may be requested.

**8.5 "Your own error suggests your data are unreliable."** The one worth
preparing a response to. The response is not a defence: the error was found
precisely because the original files and a recomputable pipeline were kept,
and thirteen of the fourteen surveyed datasets could not have run the check at
all. The README and Supplementary Note 1 both take that line.

---

## 9. Suggested order of work

1. Move the three v1 figures out of `figures/`.
2. Check all 25 references, especially the two I added.
3. Decide the three questions in §5.
4. Write the cover letter.

The analysis is finished. Nothing above requires another experiment.
