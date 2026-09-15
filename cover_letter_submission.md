# Cover letter — Nature Computational Science (Article)

[Date]

The Editors
*Nature Computational Science*

Dear Editors,

I am submitting **"Held-out unit variability overshadows architectural differences in clustered machine-learning benchmarks"** for consideration as an Article, under the subscription publishing route.

**The problem.** Across the computational sciences a model's held-out score is used as a measurement, and read as a property of the model. Two things are known separately: that clustered observations make random splitting optimistic, and that in clustered data the precision of an estimate is governed by the number of independent clusters. What has not been measured is what happens once the first is fixed. This paper measures it on four corpora from three disciplines, and the answer bears on how any benchmark table in a clustered domain should be read.

**What is new in the results.** Three things, none of which is a new algorithm.

First, a decomposition that separates which units were withheld from *training* from which of them a score is *computed on*. On every corpus the second dominates. The share is not an artefact of comparing similar models: repeating the decomposition over all 57 subsets of six durian configurations spanning three architecture families, the unit share stays between 70% and 93% while the model share never exceeds 1.1%.

Second, a test of whether that matters for the decision a benchmark table is used to make. Bootstrapping units and taking the interval of the mean paired difference, four of eighteen model pairs are resolved; among the image datasets none of the four crosses an architecture family.

Third, a protocol effect large enough to change conclusions. Under leave-one-unit-out as commonly implemented, the withheld unit is also the training loop's validation set, so it selects the checkpoint. We retrained every image model — 412 runs — with the inner validation set drawn only from training data. The measured overstatement rises by up to 14.7 points; on one corpus the ranking of two detectors reverses between protocols; and two comparisons that appeared robust under the old protocol (*d* = 0.87 and 0.95) fall to 0.11 and 0.25.

**Practical value.** The paper ends with four numbers that can accompany any aggregate score wherever unit identifiers are retained, and a small tool that computes them. The wider consequence is that in a clustered domain, a benchmark margin of one point on nine or ten acquisition units is not evidence of an architectural difference.

**What the paper does not claim.** Several conclusions are stated conditionally because the design does not support more. The variance components carry wide bootstrap intervals; the intervals resample units rather than folds, and overlapping training sets mean they may still be narrow. The separation of measurement noise from between-unit heterogeneity holds comfortably on patients but, on wearers, only under an intraclass correlation that adjacent windows are unlikely to satisfy — we report it as a function of that correlation rather than as a number. An earlier version converted the standardised paired difference into a required number of units; propagating its uncertainty gives intervals spanning three orders of magnitude, and that conversion has been removed. The training-unit dose–response predates the protocol change and is reported as preliminary. Each limitation is stated where the result appears and again in Supplementary Note 4.

**Data and code availability, including one limitation.** The durian corpus is deposited at https://doi.org/10.5281/zenodo.22030622 under CC BY-NC 4.0 — images at 640 × 640 with coordinates stripped, together with content-based farm attribution, tree identifiers and all split manifests. The non-commercial clause is the only restriction and I can supply reviewers with unrestricted access on request. GWHD, BreaKHis and UCI HAR are public and the processing scripts are released. Code and all result tables under both protocols are at https://github.com/xiaolin200206/durian-site-split, archived at https://doi.org/10.5281/zenodo.22031684; everything needed to recompute every number in the paper runs on CPU from the released tables. Trained weights are not deposited, as 512 checkpoints exceed practical archive limits; they can be regenerated from the released splits and scripts, or supplied on request.

Given that the journal performs code peer review, two things may be useful. Every quantitative claim in the manuscript — 185 of them — is recomputed from the released tables by a script that runs on each commit, and a claim whose input table is missing fails the build rather than passing silently. A second script checks the manuscript against the supplementary information for unresolved cross-references and for quantities stated in more than one place. Three of the errors reported in the paper were found by these processes rather than by inspection, including a filename join that had placed 8.9% of the corpus under the wrong farm.

**Related manuscripts.** An earlier study of capture-session leakage in five-class classification is under review elsewhere. It draws on the same field collection but differs in analysis set, unit, task, annotation level and evaluation protocol; it is cited in the Methods and I can supply it on request.

**Prior contact.** I have not discussed this work with a *Nature Computational Science* editor.

**Competing interests.** I am developing an expert system for crop disease detection and therefore have a potential financial interest in the domain the durian study evaluates; this is stated in the manuscript. Access to the Sabah orchards was facilitated by a commercial partner who had no role in study design, data collection, annotation, analysis or the decision to publish, and provided no funding, products or data.

**Ethics.** The study is principally computational analysis of plant photographs. One component involved consulting two adult growers about their professional practice; informed consent was obtained verbally and recorded at the time, and both have since confirmed in writing. No names, farm names, locations or other identifying information appear. [A written determination on whether this constitutes human-subjects research requiring institutional review was requested on DATE from BODY and is awaited / was issued on DATE.]

**Suggested reviewers.** [Name, affiliation, email — expertise in clustered evaluation or leakage in applied machine learning.] [Two to four more.] I request no exclusions.

I am the sole author and am solely responsible for correspondence.

Thank you for your consideration.

Yours sincerely,

Lin Ding Shan
Faculty of Computer Science (Data Science)
UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my
ORCID 0009-0009-6031-8479
