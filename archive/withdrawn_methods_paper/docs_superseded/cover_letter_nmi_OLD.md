# Cover letter — Nature Machine Intelligence (Analysis)

> **Status: draft.** The bracketed items must be filled before submission. The
> previous version of this file was written for *Nature Sustainability* and
> carried the earlier five-farm numbers (0.478 → 0.246); it is kept as
> `cover_letter_nature_sustainability_OLD.md` but none of those figures
> survive the re-analysis.

[Date]

The Editors
*Nature Machine Intelligence*

Dear Editors,

I am submitting **"Evaluation-unit sampling can dominate reported
machine-learning performance"** for consideration as an Analysis.

**What the paper reports.** When observations are collected in bouts — a farm
visit, a patient, a wearer — a held-out score is a property of which bouts
were withheld as much as of the model. That leakage arises from random
splitting is established; what has not been measured is what remains once the
leakage is removed. On four datasets spanning two modalities, three
disciplines and eight architectures, partitioning by acquisition unit rather
than by item lowers the reported score by 3.7% to 44.3%, with error rates
rising up to fifteen-fold. Once corrected, the held-out sample accounts for 56
to 89% of the variance in the reported figure on three of the four datasets,
against 0.4 to 14.6% for the choice of model; on the fourth, training-seed
variance dominates, and I report that counterexample rather than setting it
aside. Holding training-set size fixed, adding independent units narrows the
spread across draws three- to six-fold whether or not it raises the mean.
Resampling the evaluation units separately shows a second and different
regularity: more evaluation units buy precision but not score — the expected
figure is unchanged at every unit count, while the dispersion falls as the
inverse square root. Training-unit diversity and evaluation-unit diversity are
therefore not substitutes, and neither is visible in a single reported number.

**Why an Analysis.** The conclusions come from a systematic re-analysis across
four corpora rather than from a new method. The statistics of clustered data
are not new, and the manuscript says so: what it contributes is the magnitude
of the effect in machine-learning practice, measured against the quantities
practitioners actually report, and a reporting convention that follows from
it.

**Three faults in my own work, reported rather than repaired quietly.** Farm
identity in the durian corpus was originally joined by filename; a collision
between two visits put 8.9% of the pool under the wrong farm, and the fold
that withheld farm 0 trained on images taken at farm 6 and scored highest of
the five. A claim about lesion size across farms was true of the per-farm
medians and false of the images, because an ultra-wide lens present at every
farm changes annotated box area up to four-fold. And under leave-one-unit-out
the checkpoint is chosen on the withheld unit itself. I measured that third
fault rather than only declaring it: re-validating final-epoch weights on the
seven dataset–model combinations where they survived raises the overstatement
by 2.4 to 17.3 points, in the same order as the partitioning effect itself.
Every figure I report is therefore the conservative one. Each fault is an
instance of the mechanism the paper is about, which is why they appear in the
main text rather than only in a limitations section.

**Reproducibility.** The durian images and annotations, with farm and tree
identifiers and coordinates stripped, are deposited at
https://doi.org/10.5281/zenodo.22030622 under CC BY-NC 4.0. Code, result
tables and the manuscript are at
https://github.com/xiaolin200206/durian-site-split, archived at
https://doi.org/10.5281/zenodo.22031684. Every quantitative claim in the paper
— 322 of them — is recomputed from the released tables by a single script that
runs on every commit; a claim whose input table is absent fails the build
rather than passing silently. Three of the errors above were found by that
process rather than by inspection. The paper argues that datasets which do not
release a unit identifier cannot be checked for unit leakage; this one
releases the identifier, the originals needed to audit it, and a tool
(unitcheck) that performs the check before any model is trained.

**Ethics.** The study is principally computational analysis of plant
photographs. One component involved consulting two adult growers about their
professional practice; informed consent was obtained verbally and recorded at
the time, and both have since confirmed in writing. No names, farm names,
locations or other identifying information appear in the manuscript. [A
written determination on whether this required institutional ethics review has
been requested and is pending / was issued on DATE by BODY.]

**Declarations.** This work has not been published elsewhere and is not under
consideration by another journal. An earlier study of capture-session leakage
in five-class classification, drawing on the same field collection but a
different analysis set, unit, task and attribution procedure, is under review
and is cited in the Methods. I am the sole author. I declare one competing
interest, stated in the manuscript: I am developing an expert system for crop
disease detection and therefore have a potential financial interest in the
domain the durian study evaluates. Access to the Sabah orchards was
facilitated by a commercial partner who had no role in the study and provided
no funding, products or data.

I gratefully suggest the following as reviewers with relevant expertise, none
of whom I have collaborated with: [names and affiliations].

Thank you for your consideration.

Yours sincerely,

Lin Ding Shan
Faculty of Computer Science (Data Science)
UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my
ORCID 0009-0009-6031-8479
