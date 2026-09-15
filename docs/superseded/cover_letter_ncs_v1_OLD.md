# Cover letter — Nature Computational Science (Article)

> **Status: draft.** Bracketed items must be filled before submission. Earlier versions written for *Nature Sustainability* and *Nature Machine Intelligence* are in `docs/superseded/`; their figures do not survive the re-analysis.

[Date]

The Editors
*Nature Computational Science*

Dear Editors,

I am submitting **"Which units were withheld determines much of a reported machine-learning score"** for consideration as an Article, under the subscription publishing route.

**What the paper reports.** Machine learning is now a measurement instrument across the computational sciences, and its output is a single number read as a property of the model. When the observations are clustered — photographs from one farm, sections from one patient, windows from one wearer — that number is also a property of which clusters happened to be withheld. On four datasets spanning agriculture, histopathology and wearable sensing, partitioning by acquisition unit rather than by item lowers the reported figure by 3.7 to 55.8%. Which unit a score is computed on accounts for 68 to 88% of the variance in that figure, against 0.0 to 2.7% for the model. The consequence that matters for practice is the last one. Of eighteen model pairs spanning ten configurations and five architecture families, only four separate reliably at the unit counts these corpora provide; at the counts a five-fold unit-disjoint design leaves, two detectors separated by one point of mAP50 return the wrong ranking on roughly a quarter of resampled evaluation sets.

**Why this journal.** Your readers use machine learning to measure things in their own fields, and this paper is about whether those measurements can be read as reported. It is close in shape to work you have published on the gap between benchmark performance and behaviour under real complexity, and to benchmark analyses concluding that the best method is dataset-dependent. The contribution is a measurement of current practice and a reporting convention that follows from it, not a new algorithm.

**A protocol problem we found in our own work and then fixed.** Under the leave-one-unit-out designs in common use, the withheld unit is also the training loop's validation set. It is scored every epoch, it triggers early stopping, and it selects which weights are kept. The unit never enters a gradient, but it enters model selection. Having measured how large that effect was, we retrained every image model — 412 runs — under a protocol in which the withheld units enter no training decision at all, and the headline figures all come from that protocol. The contaminated results are retained as a comparison, because the difference between the two is itself a result. It raises the measured overstatement by up to 14.7 points, and on one dataset it reverses which of two detectors wins.

**Three faults in our own pipeline, reported rather than repaired quietly.** A filename join put 8.9% of our durian corpus under the wrong farm, and the fold that withheld farm 0 had trained on images taken at farm 6. A claim about lesion size across farms was true of per-farm medians and false of the images, because an ultra-wide lens present at every farm changes annotated box area up to four-fold. And the checkpoint-selection problem above. Each is an instance of the mechanism the paper is about, which is why they appear in the main text rather than only in a limitations section.

**Reproducibility.** The durian images and annotations, with coordinates stripped, are deposited at https://doi.org/10.5281/zenodo.22030622 under CC BY-NC 4.0. Code and all result tables under both protocols are at https://github.com/xiaolin200206/durian-site-split, archived at https://doi.org/10.5281/zenodo.22031684. Every quantitative claim in the paper — 138 of them — is recomputed from the released tables by a script that runs on every commit; a claim whose input table is absent fails the build rather than passing silently. A second script checks the manuscript against the supplementary information for cross-references and for quantities stated in more than one place. Two of the three errors above were found by these processes rather than by inspection.

**Ethics.** The study is principally computational analysis of plant photographs. One component involved consulting two adult growers about their professional practice; informed consent was obtained verbally and recorded at the time, and both have since confirmed in writing. No names, farm names, locations or other identifying information appear in the manuscript. [A written determination on whether this constitutes human-subjects research requiring institutional review was requested on DATE from BODY and is awaited / was issued on DATE.]

**Declarations.** This work has not been published elsewhere and is not under consideration by another journal. An earlier study of capture-session leakage in five-class classification, drawing on the same field collection but a different analysis set, unit, task, annotation level and evaluation protocol, is under review elsewhere and is cited in the Methods; I can supply it on request. I am the sole author. I declare one competing interest, stated in the manuscript: I am developing an expert system for crop disease detection and therefore have a potential financial interest in the domain the durian study evaluates. Access to the Sabah orchards was facilitated by a commercial partner who had no role in the study and provided no funding, products or data.

I gratefully suggest the following as reviewers with relevant expertise, none of whom I have collaborated with: [names and affiliations].

Thank you for your consideration.

Yours sincerely,

Lin Ding Shan
Faculty of Computer Science (Data Science)
UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my
ORCID 0009-0009-6031-8479
