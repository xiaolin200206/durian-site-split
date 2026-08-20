# Handover

Written so the next working session can start without re-deriving anything.
Read the two "conclusions that were withdrawn" sections before touching the
manuscript; both were confidently held and both were wrong.

---

## Where things stand

**Experiments: done.** 30 training runs, 5 seeds × 6 configurations,
188 minutes on an A100. All evaluations complete. Nothing needs retraining
for the paper as written.

**Manuscript: revised and verified.** `manuscript.md` and `manuscript.docx`
contain the full text. Related Work, the three figures and their captions,
the Figure 2 Results subsection, Supplementary Table 1, references and the
front-matter sections are all inserted. All 113 quantitative claims are
checked against the tables by `verify_claims.py`, which passes.

Outstanding on the text: main text is ~6,100 words against a 5,000 limit,
and the references marked VERIFY have not been checked against publisher
records.

**Figures: generated.** Three main figures in `figures/`, as PDF and PNG,
plus `suppfig1_stage.png` — a four-panel trunk-disease stage sequence from
the Sabah site (active, resolved x2, dead). Trunk disease is not among the
six analysed classes and those images are not in the dataset; the panel is
there because it shows what stage looks like when a grower can follow one
individual over time, which the foliar labels cannot. Captions in
`docs/figure_captions.md`.

**Practitioner material is now used, and the Methods wording changed with
it.** The consultation section previously said the input "was not collected
or analysed as qualitative research data". The paper now cites grower
estimates of cost and loss and reads active ingredients off photographed
product labels, so that sentence was replaced with an accurate one:
statements are reported as the growers' own accounts, corroborated by
labels where the claim concerns chemistry, and monetary figures are
estimates rather than measurements. The strongest single addition is not a
quote at all — scale and mealybug receive products in different IRAC
mode-of-action groups (4A/7C against 1A), which establishes the limit of
the treatment-equivalence merging rule from the registration record instead
of from anyone's opinion.

**Repository: complete.** `verify_claims.py` runs in CI on every push.

---

## Outstanding, in priority order

### Blocking submission

1. **~~Cut ~1,100 words~~ — done.** Main text is 5,105 (Nature counts main
   text excluding abstract, Methods, references and captions). Nothing was
   deleted. Roughly half came from compressing prose and half from moving
   the full Limitations list into Methods, where Nature format allows the
   detail; a condensed "What the results do not license" carrying the three
   load-bearing caveats stays at the end of the Discussion. One block of
   genuine duplication was removed: the three alternative explanations were
   stated at length in both Results and Methods.

2. **Verify the references.** Entries marked `VERIFY` in the .bib were
   located by search and their bibliographic fields reconstructed. Check
   every one against the publisher record. The two Vietnamese durian
   datasets are now verified and their entries corrected — the .bib
   previously collapsed two different datasets into one key.

3. **Ethics determination.** A request has been sent to the Institute's
   Head of Research and the Director asking whether the grower consultation
   required IEC review, and requesting a written determination either way.
   The Methods section carries a bracketed placeholder for the result.
   Written consent from both growers is held (WhatsApp, timestamped;
   redact the phone numbers before it goes anywhere).

4. **Decide the Introduction's deployment framing.** It currently refers to
   "several Southeast Asian governments" without citation. The specific
   figures found by search — a ten-week pilot with 42 farmers in Perak
   scaling toward ~110,000 growers — are unverified and naming a live
   national programme carries risk. Either verify from a primary source or
   keep the framing general.

5. **Fill the placeholders**: `[DOI]`, `[repository]`, `[email]`, `[ORCID]`,
   in the manuscript, the cover letter and the README.

### Strengthens the paper materially

6. **~~Complete the dataset survey~~ — the two durian rows are done.** Both
   Vietnamese datasets were downloaded and inspected. Neither releases a
   site identifier, so the analysis cannot be repeated on independent data
   and the paper's external validity gap stands. What was gained instead is
   the strongest row in Supplementary Table 1: the Vinh Long release is
   described as raw iPhone photographs but ships 5,274 RGBA PNGs whose XMP
   records `exif:UserComment = Screenshot`, with no camera model, timestamp
   or GPS in any sampled file, plus 177 JPEGs including files named at a
   224x224 resolution. Describe the files; do not speculate about how they
   came to be that way. Eight rows of the survey remain UNVERIFIED.

7. **The training-site-count experiment.** The claim that transfer tracks
   site coverage currently rests on a correlation across six classes. Train
   on 1, 2, 3, 4 farms — several combinations each — and evaluate all on
   Sabah, holding image count roughly constant. If Sabah performance rises
   with farm count but not with image count, the claim becomes causal.
   Roughly 12–16 runs, two hours on the same hardware.

### Optional

8. **The 473 excluded images.** Mode B (keeping them in training) was never
   run on the current taxonomy. It would quantify how much the exclusion
   costs. Low priority; the limitation is stated honestly as it stands.

9. **Leaf_rot stage ablation.** Splitting the class by lesion area was run on
   the *previous* taxonomy and those results are void. Re-running is cheap
   but the finding is already carried by the 93-fold size table without it.

---

## Two conclusions that were withdrawn

Both are recorded because each was held with confidence, acted on, and
turned out to be wrong. If either resurfaces in a later session, this is
why it should not.

### "Random splitting overstates accuracy by 68%"

Held after a single-seed run in which fold 0 scored 0.121. Five seeds later,
fold 0 scored 0.518 ± 0.010 and fold 4 scored 0.773, *above* the random
split. The 68% figure was an artifact of one run, and the framing built on
it — that random splitting produces a uniform inflation — does not survive.

What replaced it: the inflation is real (48.5% on the current taxonomy) but
secondary. The finding is that the corrected figure is itself unstable
across folds, which no amount of seed averaging fixes because the variation
is between sites, not between runs.

**The lesson that generalises:** every headline number in this project
changed when it was run five times. Nothing should be built on a single run
again.

### "The Leaf_rot size table is 96/21/109/45/124 px"

It was not. `farm_covariates.py` gives 83/10/55/45/96. Four of the five
cells in the manuscript were wrong and every other row of the same table
was right, so the row was carried forward from a superseded run and nothing
caught it — because `verify_claims.py` had no check on annotation scale at
all. The 92.8-fold claim sat in the Results, the Discussion and the
Limitations for the life of the project without a single test behind it.

The wrong numbers were also internally inconsistent in a way that was
visible without any data: 124/21 is a 5.9-fold difference in side, which is
35-fold in area, not 93. The correct numbers are consistent — 96/10 is 9.6
in side, 92.2 in area — so the arithmetic check that would have caught it
costs one line and is now in the script.

The correction strengthens the sentence it appears in. A detector trained
on 45-96 px targets and evaluated on **10** px targets is a starker claim
than the same sentence with 21 px.

**The lesson that generalises:** a number that no script checks is a number
that is not verified, whatever the CI badge says. 73 of 73 passing meant 73
of the claims that had been written down, not 73 of the claims in the paper.

### "Farm 2 fails because it was photographed at midday"

Held because farm 2 is 99% midday and is the worst fold. Farms 3 and 5 are
100% midday, farm 5 at a higher solar elevation, and both score better; farm
6 is entirely pre-09:00 and also scores poorly. No monotone relationship.

What replaced it: farm 2's Leaf_rot lesions are an order of magnitude
smaller than everywhere else, because they are early-stage. That survived
three attempts to break it, including one summary statistic that appeared to
support a camera-distance explanation and did not survive looking at the
photographs. The statistic was measuring class composition; it is kept in
`scripts/diagnostics/check_lesion_scale.py` and described in the manuscript
for that reason.

---

## Decisions and why

**Six classes, not twelve.** The paper is a standalone study and does not
reference the author's other work, so it does not need the earlier label
space. Sabah carries only these six, and a shared label space across regions
is a precondition for the central comparison.

**Psyllid merged from four classes to two.** A grower confirmed that a
single egg and a cluster trigger the same application. Treatment equivalence
is the paper's stated organising principle, so they merge.

**Phomopsis and Leaf_rot not merged**, although a grower said both get the
same fungicide and both are visually inseparable once spread. Treatment
equivalence could not be established with the same confidence, and merging
would have destroyed the 93-fold size finding by mixing two size regimes
into one class.

**Mode A, not mode B.** 473 peninsular images have no recoverable location.
Keeping them would make the random and by-farm regimes differ in training
set size as well as in split rule, confounding the one comparison the paper
exists to make.

**mAP50 as the primary metric.** A missed lesion costs a grower more than an
imprecise box, and mAP50-95 penalises sub-pixel offsets on targets a few
pixels across. mAP50-95 is reported alongside throughout.

**Single-egg psyllid images withdrawn from annotation.** During quality
control they required magnification far beyond normal review scale and were
not separable from leaf reflections at that magnification. The photographs
are retained; only the annotations were withdrawn. This is stated in
Limitations because the withdrawal was a judgement.

---

## Venue

Submitted to Nature Sustainability, with a stated reservation. The work is a
measurement study about evaluation protocol; NS publishes sustainability
science. The bridge is that the decision it bears on — public investment in
agricultural AI, justified by expected reductions in loss and pesticide use —
is a sustainability-policy decision. That bridge is real and thin, and the
cover letter says so and asks for an early assessment of fit.

If declined, the content needs no change. Only the cover letter does.
Reasonable alternatives, in order: *Computers and Electronics in
Agriculture*, *Plant Phenomics* (which published GWHD and would recognise
the argument immediately), *Precision Agriculture*, or a
Datasets-and-Benchmarks track.

---

## Files

```
manuscript.md / .docx        the paper; docx has line numbers for review
cover_letter.md / .docx
verify_claims.py             73 assertions, run before every edit lands
results/                     8 tables; the paper's entire evidence base
figures/                     3 figures, PDF and PNG
docs/related_work.md         section to insert, plus the two edits it forces
docs/references.bib          with VERIFY flags
docs/figure_captions.md      captions and the new Results subsection
docs/dataset_provenance_survey.md   Table S1, half-filled, with a protocol
scripts/prepare/             corpus to dataset
scripts/analyse/             the experiments and the per-site evaluations
scripts/diagnostics/         the three checks that changed what the paper says
```

Run `python verify_claims.py` after any edit that touches a number. It
exists because a figure was wrong once — the per-burst minimum was stated as
0.004 when one burst scores exactly zero — and the script caught it.
