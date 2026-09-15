# Cover letter

[Date]

The Editors
*Nature Sustainability*

Dear Editors,

I am submitting **"A single accuracy figure for crop disease detection describes the sample, not the model"** for consideration as an Article.

**What the paper reports.** AI advisory systems for crop disease are moving from district pilots toward national coverage in tropical smallholder agriculture, and the technical evidence behind those decisions is a single aggregate accuracy figure computed on a randomly split validation set. Using 841 durian field photographs collected and annotated by one person across five Malaysian farms and thirteen Bornean trees 1,800 km away, I show that this figure cannot support the inference drawn from it. Withholding an entire farm rather than splitting images at random lowers mAP50 from 0.478 to 0.246 — an effect 6.4 times larger than training-seed noise. More consequentially, the corrected figure is not a property of the model either: it varies 2.8-fold with which farm is withheld, and individual capture sites span 0.000 to 0.864. Crossing to Borneo costs nothing beyond withholding a neighbouring farm, so distance is not the operative variable; the number of independent management units pooled into the figure is.

**Why this belongs in a sustainability journal rather than a methods venue.** The divergence is not distributed at random, and that is the finding I would most like reviewers to weigh. The worst-performing farm in the set is the one whose lesions are earliest — small, confined to the leaf tip and margin — and early detection is the mechanism through which these tools are argued to reduce yield loss and prophylactic pesticide application. A detector that performs worst precisely where symptoms are earliest is weakest at the task it is deployed to perform, and serves least well the growers with most to gain. Recall falls further than precision at an unseen farm, so the failure mode is silence rather than error: for a grower deciding whether to spray, a missed lesion and a clean tree are indistinguishable on screen, and the rational fallback is the calendar-based application the tool was meant to displace.

The paper also offers something a programme could act on. Resampling the site-level scores turns the observed dispersion into a sample-size requirement: at the between-farm variability measured here, roughly eleven independent farms are needed before a reported mAP50 is reliable to ±0.05. The programmes now scaling are, on published evidence, working from one. I do not offer eleven as a threshold, but the gap between one and that order of magnitude is not estimation error. Two figures would be more informative than the one currently reported — how many independent management units contributed to training, and what the dispersion was across withheld units — and both are recoverable at no additional collection cost from any dataset that records where its images came from.

This connects directly to work the journal has published: El Jarroudi et al. (*Nat. Sustain.* **7**, 846–854, 2024) set out a roadmap for edge AI in food production and identify infrastructure, investment and training as the binding constraints on adoption. What that roadmap does not specify is the standard of evidence a system should meet before public money moves it from pilot to province. This paper is an attempt to supply one.

**A reservation I would rather state than have inferred.** This is a measurement study about evaluation protocol, and *Nature Sustainability* publishes sustainability science. The bridge is that the decision the measurement bears on — public investment in agricultural AI, justified by expected reductions in loss and pesticide use — is a sustainability-policy decision. I believe that bridge is real, but it is narrow, and I would be grateful for an early assessment of fit rather than a long review that founders on scope.

**Reproducibility.** All 1,314 images and their annotations, with farm and tree identifiers, are deposited at https://doi.org/10.5281/zenodo.22030622 under CC BY-NC 4.0, with GPS coordinates stripped. Code, result tables and the manuscript are at https://github.com/xiaolin200206/durian-site-split, archived at https://doi.org/10.5281/zenodo.22031684. Every quantitative claim in the paper — 113 of them — is recomputed from the released tables by a single script. The paper argues that datasets which do not release a site identifier cannot be checked for site leakage; this one releases the identifier and the means to check it.

**Ethics.** The study is principally computational analysis of plant photographs. One component involved consulting two adult growers about their professional practice; informed consent was obtained verbally and recorded at the time, and both have since confirmed in writing. No names, farm names, locations or other identifying information appear in the manuscript. A written determination on whether this required institutional ethics review has been requested from my institution and is pending; I will supply it as soon as it is issued, and would rather disclose the position now than have it surface later.

**Declarations.** This work has not been published elsewhere and is not under consideration by another journal. I am the sole author. I declare one competing interest, stated in the manuscript: I am developing an expert system for crop disease detection, and therefore have a potential financial interest in the domain this manuscript evaluates. Access to the Sabah orchards was facilitated by a commercial partner who had no role in the study and provided no funding, products or data. I gratefully suggest the following as reviewers with relevant expertise, none of whom I have collaborated with: [names and affiliations].

Thank you for your consideration.

Yours sincerely,

Lin Ding Shan
Institute of Computer Science and Digital Innovation
UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my
[ORCID]
