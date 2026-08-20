# A single accuracy figure for crop disease detection describes the sample, not the model

Lin Ding Shan

Institute of Computer Science and Digital Innovation, UCSI University, Kuala Lumpur, Malaysia

---

## Abstract

Image-based crop disease detection is entering national deployment in tropical smallholder systems on the strength of pilots run at single sites. Using 841 durian field photographs captured and annotated by one person across five Malaysian farms and thirteen Bornean trees 1,800 km away, we show that the evaluation protocol behind such pilots cannot support the inference drawn from it. Withholding an entire farm rather than splitting images at random lowers mAP50 from 0.478 to 0.246, an effect 6.4 times larger than training-seed variation. The corrected figure is not a property of the model: it ranges from 0.139 to 0.394 depending on which farm is withheld, and from 0.000 to 0.864 across individual capture sites. Crossing to Borneo costs nothing beyond withholding a neighbouring farm, so distance is not the operative variable; the heterogeneity of the units pooled is. Worst served are farms with the earliest lesions — the growers early detection is meant to help most. A pooled accuracy figure states how much was pooled, not what a grower will experience.

---

## Introduction

Precision agriculture is advanced as a sustainability intervention: earlier detection of crop disease is argued to reduce yield loss, reduce prophylactic pesticide application, and extend the productive life of perennial trees. A recent roadmap for edge artificial intelligence in agriculture sets out how on-device inference could deliver these benefits, identifying infrastructure, investment and training as the binding constraints on adoption [1]. These arguments now underwrite public investment: several Southeast Asian governments are scaling AI advisory systems from district pilots toward national coverage [25], and comparable programmes exist across tropical smallholder agriculture more widely.

What that roadmap does not specify, and what the programmes acting on it have not established, is the standard of evidence a system should meet before it is scaled. That evidence is at present an aggregate accuracy figure: a detector achieves some mean average precision on a validation set, and the number travels — into funding cases, procurement, and the expectation that a system validated in one district will work in the next. What it means depends on how the validation set was built, and it is almost always built by randomly splitting the available images.

Random splitting is unremarkable where images are independent. Field-collected agricultural imagery is not. A person walks an orchard, stops at a symptomatic leaf, takes several frames from different angles, and moves on; those frames share a leaf, a tree, a canopy and a light condition. A random split distributes them across the train–test boundary, so a model can score well by recognising the leaf rather than the lesion. The concern has been stated in the plant disease literature [2]; its magnitude on real field data, and what remains once it is removed, have not been measured.

We report both, and a third result we did not anticipate. Removing the leakage costs 48.5% of the reported score, but the corrected score is not a single number either — it varies 2.8-fold with which farm is withheld. Evaluating one site at a time, on two islands, resolves the picture: site-level variation is large everywhere, geographic separation adds nothing to it, and the apparent stability of any pooled figure is produced by the number of distinct management units it averages over. One component of that variation is measurable and has been invisible: the same disease label carries lesions an order of magnitude apart in size, because no dataset records disease stage.

The consequence for deployment policy is direct. A single-site pilot produces precisely the quantity our analysis shows to be least generalisable, and the property that governs transfer is how many independent management units contributed to training — not how many images.

---

## Related work

**The concern has been stated; it has not been measured.** A 2024 review of plant disease recognition datasets identifies random splitting as the field's dominant practice and names its principal defect: several images of one observation, differing only slightly, can land on both sides of the boundary, so test performance is overestimated [2]. That is the mechanism this paper examines. What the review does not do — and what we are not aware of anyone doing on field-collected disease imagery — is measure how large the overestimate is, or ask what remains once it is removed. A related bias in the same datasets is already quantified: a classifier trained on eight background pixels from PlantVillage reaches 49.0% accuracy against a 2.6% chance baseline [3], which concerns background shortcut rather than site leakage but establishes in weaker form that aggregate accuracy here can be produced by information having nothing to do with the disease. Our contribution is not the observation that random splitting leaks; it is the measurement, and what the measurement exposes about the corrected figure.

**Grouped splitting is already standard in one agricultural vision task.** The Global Wheat Head Detection dataset organises its images into subdatasets — a consistent set acquired over the same experimental unit, during the same session, with the same vector and sensor, 47 such domains in the 2021 release [4,5]. Its authors state the reasoning directly: although random splitting is common practice, the competition aims to test performance on unseen genotypes, environments and observational conditions, so images are grouped by continent instead [4]. The 2021 edition made validation and test sets entirely disjoint by session where 2020 had drawn them from shared subdatasets [6], and WILDS subsequently adopted the dataset as a domain-generalisation benchmark in which a domain *is* an acquisition session [7]. The precedent is therefore established in agricultural imaging — for wheat head counting. It has not reached disease detection, where the datasets the field is built on do not record the metadata that would make it possible (Supplementary Table 1). GWHD reports per-session error distributions but does not treat the dispersion between sessions as the quantity of interest; we do.

**The failure has a name and a remedy in adjacent fields.** In medical imaging and physiological signal analysis it is called identity confounding: a model learns to recognise the subject alongside the diagnostic feature, so record-wise cross-validation inflates accuracy relative to subject-wise [8,9]. It is documented across MRI, optical coherence tomography, accelerometry, voice and EEG, in deep networks and random forests alike [10,11], and patient-level partitioning is now routine, its absence treated as a defect rather than a design choice [12,13]. Our contribution relative to that literature is not the diagnosis but that the agricultural analogue of a patient is not obvious: a field dataset could be grouped by leaf, plant, plot, farm, session or region, and the choice changes the answer — we measure a 48.5% reduction at farm level and a considerably larger dispersion at capture-site level. For a handheld tool used at one orchard the relevant unit is the one the field does not record.

---

## Results

### Study system

We assembled 1,033 annotated photographs from Peninsular Malaysia and 281 from two orchards near Lahad Datu, Sabah — 1,314 in total, of which 841 enter the analyses below. All were captured hand-held under natural light by one person and annotated and quality-controlled by the same person, with no background removal at any stage. Six classes were retained: three foliar symptom classes (Algal, Leaf_rot, Phomopsis) and three pest classes (Psyllid, Psyllid_damage, leaf_hopper_damage).

Farm attribution was recoverable from EXIF for 560 peninsular images (Methods). Those 560, spanning five farms of usable size, form the pool for every split comparison, so the random and by-farm regimes draw from an identical set and the split rule is the only difference between them; with the 281 Sabah images they constitute the 841 analysed. Peninsular farms lie within 134 km of one another; the Sabah orchards are 1,765–1,872 km distant.

A YOLO11s detector [23] was trained under two regimes with five random seeds each: a stratified random 80/20 split over images, and GroupKFold with farm as the group. All other settings were held fixed (Methods).

### A random split overstates accuracy by half

| Split | mAP50 | s.d. | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Random 80/20 | **0.478** | 0.003 | 0.303 | 0.597 | 0.490 |
| By farm, mean of five folds | **0.246** | 0.099 | 0.128 | 0.387 | 0.271 |

The gap is 0.232 mAP50: the by-farm figure is 48.5% lower, or equivalently the random split reports a value 94.3% higher on the same images. Training noise does not account for it — seed-to-seed s.d. within a fold averages 0.015 against 0.099 between fold means, a ratio of 6.4. Recall falls further than precision (0.490 to 0.271 against 0.597 to 0.387): at an unseen farm the detector does not misclassify so much as fail to fire, and a detector that misses rather than misidentifies returns a clean-looking screen over a diseased tree.

### The corrected figure is not a stable quantity

| Fold | Withheld farm | n images | mAP50 | s.d. over 5 seeds |
|---|---|---|---|---|
| 0 | farm 0 | 165 | 0.394 | 0.014 |
| 3 | farm 3 | 59 | 0.292 | 0.035 |
| 2 | farm 5 | 141 | 0.209 | 0.013 |
| 4 | farms 6, 7, 8 | 41 | 0.196 | 0.004 |
| 1 | farm 2 | 154 | 0.139 | 0.011 |

Best to worst is a factor of 2.84, while each fold is individually reproducible to within a few thousandths. Fold 4 withholds farm 6 together with the two single-image groups, and is in effect an evaluation on farm 6. Reporting the mean of these five numbers as *the* accuracy of the detector describes no farm in the set.

### Crossing to Borneo costs nothing beyond withholding a neighbour

The Sabah set was never seen in training under any configuration — another island, another management history, another season.

| Evaluation | mAP50 | s.d. |
|---|---|---|
| Peninsula, random split | 0.478 | 0.003 |
| Peninsula, withheld farm (five folds) | 0.246 | 0.099 |
| Sabah, all 281 images (6 configurations x 5 seeds) | **0.246** | 0.009 |

Every training configuration, averaged over its five seeds, scores between 0.236 and 0.263 on Sabah, whatever subset of peninsular farms it saw. The cross-island figure is indistinguishable from the within-island withheld-farm figure and far more reproducible: the standard deviation across configuration means is 0.009 against 0.099 across peninsular folds. Individual models range more widely, from 0.148 to 0.296 (s.d. 0.027), but that spread is seed noise of the same order seen within any single fold, not a function of which farms the model was trained on. Distance is not what makes transfer hard.

### Per-class transfer tracks site coverage, not annotation volume

| Class | Farms present | Random split AP50 | Sabah AP50 |
|---|---|---|---|
| Algal | 5 | 0.563 | **0.582** |
| Leaf_rot | 5 | 0.357 | **0.426** |
| Phomopsis | 5 | 0.469 | 0.298 |
| Psyllid | 3 | 0.329 | 0.075 |
| Psyllid_damage | 2 | 0.169 | 0.036 |
| leaf_hopper_damage | 3 | **0.980** | **0.057** |

*[Figure 3 here]*

The direction is mixed: two classes score higher on another island than on a random split of the training island, three fall, and one collapses. What separates them is not annotation volume — Phomopsis carries 6,627 boxes and loses 36%, Leaf_rot carries 570 and gains — but the number of farms at which the class was photographed. The two present at all five are the two that do not degrade.

leaf_hopper_damage is the instructive case. Its 134 boxes come from a few capture sessions on adjacent trees; under a random split it is the best class in the dataset at 0.980, a figure that would be reported without comment, and on Sabah it reaches 0.057. A random split could not have revealed that, because it never withholds a site. Twelve boxes in Sabah support no interpretation alone; the point is the contrast between what the two protocols would have led a reader to believe.

### The folds do not ask the same question

The five folds do not test the same classes: Phomopsis is present in every image at farm 6 and none at farm 3, and Psyllid at no image from farms 0 or 6, so fold 3 returns no Phomopsis figure and fold 0 none for either pest class. A fold's aggregate mAP50 averages over whichever classes its farm happened to carry, so the five numbers are not five measurements of one quantity.

More consequentially, one label spans an order of magnitude in target size across farms. Median Leaf_rot box area differs 92.8-fold between extreme farms (0.02242 of frame area at farm 6 against 0.00024 at farm 2); as an approximate box side at the 640-pixel training input:

| Class | farm 0 | farm 2 | farm 3 | farm 5 | farm 6 | Area ratio |
|---|---|---|---|---|---|---|
| Leaf_rot | 83 px | **10 px** | 55 px | 45 px | 96 px | **92.8** |
| Algal | 19 | 7 | 13 | 14 | — | 7.4 |
| Phomopsis | 8 | 4 | — | 5 | 6 | 3.8 |
| Psyllid | — | 3 | — | 5 | — | 3.7 |
| Psyllid_damage | — | 8 | — | 8 | — | 1.1 |
| leaf_hopper_damage | — | 87 | — | 96 | — | 1.2 |

Cells are box sides; the ratio column is of areas, so a 9.6-fold difference in side is a 92.8-fold difference in area. Dashes mark cells with fewer than five contributing images.

The farm withheld in the worst fold, farm 2, is the one whose Leaf_rot lesions are an order of magnitude smaller than everywhere else. A detector trained on 45–96 px targets and evaluated on 10 px targets is not failing to recognise a lesion; it is being asked for a different object under the same name.

Annotation granularity, capture conditions and camera distance were each tested and rejected as explanations (Methods). One of them is worth recording because the wrong summary statistic is persuasive: the largest annotated box per image, an apparently reasonable distance surrogate, is 35-fold smaller at farm 2 than at farm 6 — but only because farm 2 is 72% Phomopsis, a class annotated as many 4-pixel lesions. The surrogate measured class composition.

What remains is disease stage. Lesions at farm 2 begin at the leaf tip or margin, where the fungal genera associated with these symptoms enter; lesions elsewhere have progressed into the lamina. Growers describe early and spread presentations as different problems — separable early, not separable once spread — and act on the distinction when spraying, tracking individual trees through active, resolved and lost states over years (Supplementary Fig. 1). No dataset we are aware of, including ours, records stage. It appears here only as a distribution of box sizes that no aggregate metric exposes.

Whether that distribution predicts fold score we can pose but not answer. Taking median Leaf_rot box side as a stage proxy, the association with fold mAP50 is positive across five farms (Pearson *r* = 0.54; 0.62 on log area), while the obvious confound does not explain it (images per fold, *r* = 0.15). Farm 6 is the sole exception and a decisive one: largest lesions of any farm, and still 0.196. Among the remaining four the relationship is near-perfect (*r* = 0.97), but we do not report that as a result — removing the one point that breaks a pattern is how a pattern is manufactured, and *n* = 4 could not support it anyway. Farm 6 also differs in optics, photographed at a 2.22 mm focal length where the others used 6.76 mm, so the exception may be a lens artefact rather than evidence against stage. The test is directional: stage is visible in the data and plausibly implicated, but nothing here would let an evaluator forecast a site's score before evaluating it.

The two effects are not in tension. Leaf_rot carries the widest size range of any class and still transfers without loss, because it was photographed at all five farms and training saw the full span of presentations rather than one end of it. Heterogeneity within a label degrades performance when a fold withholds the only site carrying one end of the range; distributed across training sites, the same heterogeneity is what makes the class robust. This is the site-coverage result at the level of a single label.

### One site at a time, on both islands

If crossing the sea costs nothing, what does the between-fold variation represent? We evaluated each region at the finest available unit, always with weights that had never seen that site. Sabah filenames carry a tree identifier, giving thirteen units, twelve with at least five images. Peninsular filenames do not, so images were grouped into capture bursts — consecutive frames less than 15 s apart, which given the capture pattern almost always means one target — each scored with the fold whose validation set is that burst's farm.

*[Figure 1 here]*

| Unit | n | Mean mAP50 | s.d. | CV | IQR ratio |
|---|---|---|---|---|---|
| Peninsula, per burst | 34 | 0.316 | 0.227 | **0.72** | 3.5 |
| Sabah, per tree | 12 | 0.308 | 0.086 | **0.28** | 1.3 |

The means agree to within 0.008. The dispersions differ 2.6-fold. Peninsular bursts span 0.000 to 0.864; Sabah trees span 0.201 to 0.540.

Grouping peninsular bursts by farm locates the dispersion:

| Farm | Bursts | Mean | s.d. | Range |
|---|---|---|---|---|
| 3 | 3 | 0.653 | 0.183 | 0.534–0.864 |
| 6 | 2 | 0.463 | 0.025 | 0.446–0.481 |
| 0 | 8 | 0.403 | 0.247 | 0.000–0.699 |
| 5 | 12 | 0.280 | 0.186 | 0.004–0.668 |
| 2 | 9 | 0.143 | 0.108 | 0.016–0.354 |

Farm means span 4.6-fold. The Sabah orchards differ by 1.3-fold (o1, eleven trees: 0.250; o2, two trees: 0.335).

The thirteen Sabah trees sit under one manager, planted and treated alike, photographed across four consecutive days in one season. The five peninsular farms are separate operations visited on four dates across eight months, differing in cultivar, spray regime, canopy management and disease stage at the time of capture. Sabah is not uniform because it is in Borneo. It is uniform because it is one management unit.

### How much is pooled determines how much the figure can move

The two panels of Fig. 1 have the same pooled value and different spread, suggesting the pooled figure's stability is a function of how many sites it averages. We tested this by resampling: for each subset size *k*, 4,000 random subsets of *k* sites were drawn and the distribution of their mean score recorded.

*[Figure 2 here]*

The expected value is flat in *k*: pooling does not make the model better, and any subset size returns the same 0.31 in both regions. What changes is how far a particular evaluation can fall from it. In the peninsula the 90% interval on a reported figure is 0.695 wide at one site, 0.304 at five and 0.106 at twenty; in Sabah, whose sites are more alike, 0.339 at one tree and 0.091 at five.

A single-site evaluation is therefore not a noisy estimate of a true underlying accuracy. It measures a different quantity — performance at that site — and the two coincide only where every site is alike. Ours are not: even in Sabah, thirteen trees under one manager in one season, one returns 0.201 and another 0.540.

Read in the other direction, the curve is a sample-size requirement. Treating site scores as draws from a population with the dispersion we observe, a reported figure lands within ±0.10 of its expectation with 90% probability once 14 capture sites are pooled, within ±0.05 once 56 are; at the farm level, where our dispersion is s.d. 0.099, the requirements are 3 and 11. These assume sites are exchangeable and our dispersion representative, neither verifiable from five farms — they are an order-of-magnitude guide, not a standard. The order of magnitude is the point: a single-site pilot is not a small version of an adequate evaluation.

**A caveat on what was resampled.** The curve is the sampling distribution of the *mean of k site-level scores*, not a pooled mAP recomputed over the union of *k* sites' images. The two differ in weighting: a pooled mAP weights a site by the instances it contributes, an equal-weight mean does not. We report the equal-weight version because the question is how much a headline figure depends on which sites were chosen, and because instance counts vary by an order of magnitude between our sites for reasons unrelated to performance.

---

## Discussion

### What the results license

Four claims follow. The first confirms a stated concern rather than discovering one: random splitting of field-collected crop imagery overstates detection accuracy substantially — here the corrected figure is 48.5% lower — with an effect 6.4 times larger than training noise. The remaining three are novel. The corrected figure is a property of the evaluation sample rather than of the detector, withholding different farms yielding scores that differ 2.8-fold and individual capture sites by far more. Class-level transferability tracks the number of distinct sites a class was photographed at, not the number of boxes annotated. And part of the between-farm spread has a nameable cause: one symptom label spans a 92.8-fold range in lesion area, so folds differ not only in difficulty but in what object they ask the detector to find.

One expectation is refuted, and we state it plainly because it was ours. We designed this study to quantify a cross-island generalisation penalty under controlled acquisition, following an earlier report in which a durian classifier lost 64.1 percentage points on a Vietnamese dataset [19] — a figure confounded by that dataset's background removal and resizing. There is no such penalty to quantify: performance in Borneo equals performance on a withheld peninsular farm 1,800 km closer. The geographic framing was wrong.

### Consequences for deployment at scale

A pooled accuracy figure answers "how well does this model do on average across the sites I happened to collect?" Deployment asks "how well will it do at the farm in front of me?" These diverge further as the evaluation sample pools more sites: a model reported at 0.478 delivers 0.139 at the worst of five farms, and capture sites span 0.000 to 0.864. A grower does not farm the average orchard, and whether a tool will help them is, on this evidence, not predictable from any published aggregate.

The divergence is not distributed at random, and this is where the measurement bears on the sustainability case rather than only on methodology. The worst-performing farm, at 0.139, is the one whose lesions are early — small, confined to the leaf tip and margin, not yet spread into the lamina. Early detection is the mechanism through which these tools are argued to reduce yield loss and prophylactic spraying, so a detector that performs worst precisely where symptoms are earliest is weakest at the task it is deployed to perform, and serves least well the growers with most to gain. A pooled figure conceals this: it reports an average over management units and says nothing about which fall below it.

The failure mode compounds the problem. Recall falls further than precision at an unseen farm, so the detector returns nothing rather than misclassifying, and for a grower deciding whether to spray a false negative and a clean tree are indistinguishable on screen. If the tool is not trusted to find disease, the rational fallback is the calendar-based application it was meant to displace, and the reduction in pesticide use that justifies the investment does not materialise. We did not measure spray decisions and cannot quantify this; the chain from detection accuracy to environmental outcome runs through a decision that a missed lesion silently removes. The magnitudes involved are not small: at the Sabah site a single whole-orchard application costs roughly MYR 2,700 in product, a season's unsaleable fruit was put at about MYR 4,000, and a mature tree lost to trunk disease at MYR 10,000–20,000. A tool that changes spray frequency in either direction moves sums of this order.

This bears on how AI-for-agriculture programmes are validated before scale-up. Existing guidance for deploying edge AI in food systems addresses infrastructure, energy and training [1], but not the evidential standard a system should meet before public money moves it from pilot to province. A pilot at one site produces exactly the quantity we show to be least generalisable. Two figures would be more informative than the one currently reported: how many independent management units contributed to training, and what the dispersion was across withheld units. Neither appears in the literature we are aware of. Both are recoverable at no additional collection cost from any dataset that records where its images came from.

Our resampling puts an approximate floor under the first. At the between-farm dispersion we observe, roughly eleven independent farms are needed before a reported mAP50 is reliable to ±0.05, and three to ±0.10; the programmes now scaling toward national coverage are, on published evidence, working from one. We do not offer eleven as a threshold — it follows from a dispersion estimated on five farms — but the gap between one and the order of magnitude our data imply is not estimation error.

The finding also reframes what makes a dataset valuable: Phomopsis has 6,627 boxes and loses 36% across the sea while Leaf_rot has 570 and loses nothing, so collecting more images at existing sites buys less than collecting at new ones — a difference no metric computed on a random split can show.

### Why this has not been measured before

A grouped split requires a site identifier. Of fourteen public crop disease datasets surveyed (Supplementary Table 1), one releases one at a granularity that permits it. The one that does — Global Wheat Head Detection — was built for domain generalisation and rejects random splitting explicitly [4,5]; the disease detection datasets the field is built on were not, and cannot be retrofitted, because the metadata was not recorded at capture or was removed before release.

The two durian datasets are illustrative because they are recent, field-collected and describe multiple farms. The Binh Phuoc and Tien Giang release [19] names four orchards; its files carry a class name and a global index, no side-car metadata, and no EXIF in twenty sampled images, having been background-removed and resized to 400 × 400. The Vinh Long release [18] names five and is described as raw iPhone photographs in natural light, but as released it is 5,274 RGBA PNGs whose XMP records `exif:UserComment = Screenshot`, with no camera model, timestamp or GPS in ten sampled files, alongside 177 JPEGs including files named at a 224 × 224 resolution. Whatever the origin of these images, no released metadata field permits grouping by orchard, tree or session, and both ship a preset random partition drawn across all farms.

This shows the measurement is difficult to repeat elsewhere. It does not show the effect exists elsewhere, which would require the experiment on another dataset — and no public durian dataset currently permits it.

### A consequence for system design

If performance at an unseen site cannot be predicted, a system that always returns a confident class is misspecified regardless of its average accuracy. This is an argument from measurement, not principle: we observe an order-of-magnitude spread in per-site performance with no covariate that predicts it in advance.

The requirement is sharper for the edge deployments current roadmaps favour [1]. Durian is grown where connectivity is unreliable, so a handheld device has no fallback when uncertain and no telemetry to reveal that it is failing at a particular farm: the property that makes on-device inference attractive is the same property that prevents anyone from discovering it has stopped working. Abstention — routing low-confidence cases to human inspection — is then the mechanism by which unpredictable site-level failure becomes visible rather than silently wrong.

### On the labels

Our class list mixes two naming systems, and we retain it because the mixture is the field's, not ours. *Phomopsis* names a fungal genus, treated by current taxonomy as the asexual form of *Diaporthe* and with species boundaries requiring sequence data [20,21]; *Leaf_rot* and *Algal* name symptoms attributable to more than one organism. These are not mutually exclusive — a *Phomopsis* infection can present as leaf rot — so an annotator choosing between them from a photograph is not identifying a pathogen. The names are not even stable across our two regions: what the peninsular class list calls leaf_hopper_damage is known by a different local name in Sabah, which is one reason we treat that class as a contrast between protocols rather than a measurement. We keep the pathogen-derived names because they are the names growers and extension services use and under which fungicides are registered; renaming for taxonomic accuracy would make the list less usable by the people it is built for and no more accurate about what is in the image.

Growers make the point from the other direction: they name diseases with the official labels but decide treatment from the product registration, so two symptom classes sharing a registered product are, for the decision that matters, one thing. Asked what a merged class should be called, one grower advised keeping the official names, since the two receive the same spray. Two naming systems coexist because the grower never has to choose. A detector must. We acted on this once — the four psyllid categories were reduced to two after a grower confirmed that a single egg and a cluster trigger the same application — but did not merge the foliar symptom classes, because treatment equivalence could not be established with the same confidence.

The principle has a limit, and the products establish it independently of anyone's opinion. Scale insects and mealybugs are visually similar sap-feeders that a detector would be tempted to merge, but the grower applies different registered products to them, and those products belong to different insecticide mode-of-action groups: acetamiprid with pyriproxyfen (IRAC 4A and 7C) against scale, carbosulfan (IRAC 1A) against mealybug. Treatment equivalence is therefore not a matter of grower preference that a tidier class list could override; where it fails, it fails at the level of registered chemistry. A class list built on what a grower does must respect both the merges and the separations that practice already encodes.

A second gap is dimensional rather than nominal. Growers distinguish early from spread lesions and act on it; our labels do not, and the consequence is the 92.8-fold size range reported above. Stage is not reliably readable from a single photograph without the leaf's history, so we note the omission rather than repair it — it is shared by every crop disease dataset we know of and invisible in all of them.

---

### What the results do not license

Three caveats bear directly on the claims above; the full set is in Methods. The five folds are not five measurements of one quantity — class composition differs so sharply between farms that some folds return no figure for some classes, so the standard deviation of the fold means describes the sample rather than estimating an uncertainty. The by-farm analysis rests on five effective farms and excludes 46% of the peninsular labelled set for want of recoverable location, so the dispersion we report is a lower bound on what more sites would show. And the scale comparison has a confound we cannot resolve: farm 6, the one farm that breaks the stage association, is also the only farm photographed at a different focal length. All results are offline evaluations of one architecture on one crop; no grower has used this detector in the field.

---

## Methods

### Sites and acquisition

Photographs were taken at seven durian farms in Peninsular Malaysia (Selangor, Negeri Sembilan, and the Muar district of Johor) between December 2025 and July 2026, and at two orchards near Lahad Datu, Sabah, between 13 and 16 August 2026. The Sabah sites are a 20-acre mature planting under a single manager of eleven years' tenure (trees approximately nine years old; 70% Musang King, 20% Black Thorn) and a separate 9-acre planting of 3.5-year-old trees. Peninsular farms lie within 134 km of one another; the Sabah orchards are 1,765–1,872 km from them.

All images were captured hand-held under natural light by one person. The primary device was an iPhone 16 Pro Max; an iPhone 13 Pro was used at three peninsular farms when the primary device throttled thermally and did not record GPS. No background removal, cropping or colour correction was applied at any stage. Two framings occur — a leaf attached to the tree and held steady, and a detached leaf laid on the palm — both within single farms; the distinction was not recorded and was recovered only by inspection.

### Field consultation with growers

During field data collection, growers were consulted about their professional practice, covering symptom recognition, treatment decisions, product choice and application frequency, and the approximate costs and losses those decisions carry. Two growers contributed: the manager of the Sabah orchards and a collaborating peninsular grower, the latter by voice message and photograph exchange. Their input informed class definition and the interpretation of the results reported here. It was not analysed as qualitative research data; statements are reported as the growers' own accounts of their practice, and are corroborated by product labels where the claim concerns chemistry. Agrochemical product labels were photographed where available and active ingredients read from the label in preference to recalled product names. Monetary figures are the growers' estimates, not measurements.

Informed consent was obtained verbally and recorded at the time of consultation, and both growers subsequently confirmed in writing that they were content for this information to be used on that basis. No names, farm names, locations or other identifying information are reported. [*Ethics determination pending — insert institutional statement and reference number, or a statement that the reviewing body determined the activity did not constitute human subjects research requiring review.*]

### Annotation and class definition

Annotation is lesion-level: each visible lesion or insect receives its own box. Annotation was outsourced per class and quality-controlled by the author, who reviewed and corrected every image. Six classes were retained after consolidation. Images whose only annotation was a single psyllid egg were withdrawn for the reason given in Limitations; the photographs are retained.

Each class was annotated in a separate project, so a photograph carrying several diseases appears in several exports under different platform-assigned filenames. Exports were merged by stripping the platform hash to recover the original filename, remapping each project's local class indices onto one canonical list, and discarding boxes duplicated across projects at IoU ≥ 0.90. Class names differing only in case were folded together — a failure mode found in an earlier dataset, where Early_Blight and early_blight had been trained as separate classes. The canonical list and its order are identical for both regions, verified programmatically before training, since a mismatch in order silently relabels every box.

### Farm attribution

Farms were recovered from EXIF GPS by single-link clustering at 1.5 km. Images without GPS were attributed by capture-time proximity: on a multi-farm day, visits are sequential, so an image from the second handset sits in time between GPS-bearing images of the farm then being visited. Attribution was accepted only where the nearest GPS-bearing image of the same day was within 30 minutes. Attribution source is recorded per image; 560 of 1,033 peninsular images were attributed (427 by GPS, 133 by timing). Clustering returned seven groups; two contain a single image and are not analysed separately.

### Splits

Two regimes over the same 560-image pool: stratified random 80/20 by dominant class (448 train, 112 validation), and GroupKFold with five folds and farm as group. Sabah was never split; all 281 images serve as a held-out test set for every peninsular model.

### Training

Ultralytics YOLO11s [23], 640 px input, batch 32, maximum 150 epochs with patience 50, five seeds (42, 1, 2, 3, 4) per configuration, on a single A100. Runs terminated between 56 and 150 epochs (median 124.5). Thirty training runs; 188 minutes total.

### Evaluation

All reported figures come from re-validating best.pt through one code path, so no number depends on the checkpoint-selection logic of the training loop. Per-site evaluation constructs one dataset manifest per site and re-runs validation without retraining. Peninsular bursts are scored with the fold whose validation set contains that burst's farm; scoring them with random-split weights measures fit rather than generalisation and is not reported. Sabah trees carrying fewer than five images were excluded from the per-tree analysis, leaving twelve of thirteen.

mAP50 [24] is the primary metric. For a handheld advisory tool a missed lesion costs more than an imprecise box, and mAP50-95 penalises small offsets on targets a few pixels across; mAP50-95 is reported alongside throughout.

### Alternative explanations for the fold spread

Three were tested before attributing the spread to disease stage. *Annotation granularity*: median boxes per image within Leaf_rot is 1 to 3 at every farm (3, 2, 1, 1, 1 at farms 0, 2, 3, 5, 6), so lesions were not subdivided more finely at one site. *Capture conditions*: median capture hour, solar elevation, midday share, device, ISO and focal length were tabulated per farm. Farm 2 was photographed 99% between noon and 15:00, but farms 3 and 5 were photographed 100% in that window, farm 5 at a higher median solar elevation (72.1° against 62.0°), and both score above farm 2; farm 6, at a median capture hour of 08:52, scores 0.196. Hour of capture, solar elevation and midday share show no monotone relationship with fold score. *Camera distance*: annotated frames were inspected directly. At farm 2 the subject is a single leaf, held by hand and filling a large fraction of the frame — the same framing used elsewhere — with small lesions at the leaf tip and margin on otherwise intact laminae; at farms 0, 3 and 6 the boxes enclose necrotic regions extending into the leaf body.

The stage association reported in Results is a Pearson correlation between median Leaf_rot box side and fold mAP50 over five farms, with a log-area variant and images per fold as a control. Per-farm annotation scale and capture covariates are released as `farm_annotation_scale.csv` and `farm_capture_conditions.csv`. No inference is drawn from *n* = 5.

### Aggregation analysis

For each subset size *k* from 1 to the number of available sites, 4,000 subsets of *k* site-level scores were drawn without replacement and the mean of each recorded. Figure 2 plots the mean of that distribution and its 5th-to-95th percentile interval. Peninsular sites are the 34 capture bursts, Sabah sites the 12 trees.

### Limitations

**The folds are not comparable to each other.** Class composition differs so sharply between farms that some folds return no figure for some classes. We report the fold mean because it is the conventional summary, but the five scores are not five measurements of one quantity, and their standard deviation describes the sample rather than estimating an uncertainty.

**Five effective farms, and 46% of the peninsular labelled set excluded.** Seven clusters were recovered but two contain a single image, so the by-farm analysis rests on five units and the dispersion we observe is a lower bound on what more sites would show. Of 1,033 annotated images, 473 carry no recoverable location — an early batch transferred by instant messaging, which strips EXIF, and a second handset with location services off — and enter neither pool. They are not a random sample of the corpus and the direction of any resulting bias is unknown.

**Sabah is one manager, one season, four days, and bursts are only a proxy for trees.** Sabah's internal homogeneity — the finding that carries the interpretation — is a property of that sample; a second Bornean orchard under different management would very likely widen the spread, and the season was unusually dry by the manager's account, which suppresses *Phytophthora* expression [22] and may suppress other disease pressure. On the peninsular side the 15-second burst threshold is a choice: a tree photographed in two visits a minute apart becomes two units, two adjacent trees photographed in quick succession become one. The peninsular per-burst scores also come from models trained on four farms while Sabah scores come from models trained on all five; that asymmetry disadvantages Sabah, so the finding that Sabah is the more uniform domain is conservative.

**One collector, one annotator, and a protocol less uniform than intended.** A single collector, device family and protocol is what distinguishes this study from cross-dataset comparisons confounded by preprocessing, but it is an overstatement in one respect: some leaves were photographed attached to the tree and some detached and laid on the palm, both within single farms, and we did not record which. Protocol consistency is likewise a strength for the comparison and a weakness for the labels — we cannot report the rate at which a second annotator would disagree, and for two classes we expect it would be high. Single psyllid eggs required magnification far beyond normal review scale to locate and were not separable at that magnification from reflections, trichomes and water droplets; images whose only annotation was a single egg were withdrawn, which was itself a judgement.

**leaf_hopper_damage is uninterpretable.** 134 boxes in the peninsula, 12 in Sabah from two trees; we use it as a contrast between protocols, never as a measurement.

**Single crop, architecture and input resolution.** All results are YOLO11s at 640 px on durian, and small-target classes are known to be resolution-sensitive. We cannot show the effect generalises to other crops, and the datasets that would permit the test do not record the necessary metadata (Supplementary Table 1). What we can say is that the mechanism — burst-structured capture and heterogeneous management units — is a property of how field agricultural imagery is collected, not of durian.

**No inference on the headline comparison.** The random-split and by-farm figures are point estimates over five seeds and five folds; we report dispersion at both levels but construct no confidence interval on their difference, because the fold scores are not five draws from one distribution. The resampling analysis (Fig. 2) characterises uncertainty where the units are exchangeable.

**The scale comparison has two weaknesses.** Farm 6 was photographed at a 2.22 mm focal length where every other farm used 6.76 mm; a change of lens alters apparent object size, depth of field and background, so farm 6 is not exchangeable for any argument that turns on target scale — and it is also the farm that breaks the stage association, which one farm cannot disentangle. Separately, the 92.8-fold range is between medians computed on 35, 25, 7, 5 and 7 images. The two extremes carry the most, which is the favourable case, but three cells are thin, and that thinness also qualifies the claim that Leaf_rot is present at all five farms: present, but at five to seven images at three of them.

**No deployment.** No grower has used this detector in the field. All results are offline evaluations and none predicts the detection rate a user would experience.

### Data availability

Images, annotations and per-run result tables for the Sabah test set are deposited at [DOI], CC BY-NC 4.0, with GPS coordinates removed from EXIF. Peninsular images and annotations are deposited at the same DOI. Split manifests, per-run summaries and all result tables are released without restriction.

### Code availability

Merging, farm attribution, split construction, training, per-site evaluation and the aggregation analysis are at [repository]. Every quantitative claim in this paper is recomputed from the released tables by a single script.

---

## Figure captions

**Figure 1 | The same weights, evaluated one site at a time.** Each marker is one capture site scored with weights that never saw it; error bars are the standard deviation across five training seeds. Left, 34 peninsular capture bursts, each scored by the fold whose validation set contains that burst's farm. Right, 12 Sabah trees with at least five images, scored by models trained only on peninsular data. Solid lines are the pooled figures those same weights produce — 0.246 in both cases. The dashed line is what a random image split reports for the peninsular data, 0.478. Site scores span 0.000 to 0.864 in the peninsula and 0.201 to 0.540 in Sabah; the pooled number reports none of that range, and no site score can be predicted from it.

**Figure 2 | A reported figure is a statement about how many sites it averages over.** For each subset size *k*, 4,000 random subsets of *k* sites were drawn and the mean of their scores recorded; the line is the mean of that distribution and the shaded band its 5th-to-95th percentile interval. The expected value does not move with *k* — pooling does not make a model better — but the interval within which a particular evaluation can land narrows from 0.695 at one peninsular site to 0.106 at twenty. An evaluation at a single site is not a noisy estimate of the pooled figure; it is the quantity a grower experiences, and the pooled figure discards it.

**Figure 3 | Transfer tracks how many farms a class was photographed at, not how many boxes it has.** Grey, AP50 under a random split of the peninsular data; blue, AP50 on the held-out Sabah set; a line joins the two evaluations of one class. The two classes present at all five farms lose nothing or gain. `leaf_hopper_damage`, whose 134 boxes come from a few capture sessions at three farms, is the best class in the dataset under a random split at 0.980 and reaches 0.057 in Borneo. `Phomopsis` carries 6,627 boxes, the most of any class, and loses 36%. Points are offset horizontally within each farm count for legibility.

---

## Supplementary Figure 1

**Supplementary Fig. 1 | Disease stage is a real and consequential axis that no dataset records.** Trunk disease at the Sabah site, photographed by tree over four days. **a**, active infection, exudate on the trunk of tree 11. **b**, the same tree at a lesion that has resolved and callused. **c**, an older resolved lesion on tree 10. **d**, tree 12, lost. The manager tracks individual trees through these states and acts on the transitions: a lesion caught while the exudate is fresh is treatable, one that has reached the root system is not, and recovery takes two to three years and can relapse. Trunk disease is not among the six foliar and pest classes analysed in this paper, and these images are not part of the dataset; they are included because they show what stage looks like when a grower can follow one individual over time. Our foliar labels collapse the same axis, and the consequence is the 92.8-fold range in Leaf_rot lesion area reported in Results.

---

## Acknowledgements

The author thanks the orchard manager in Lahad Datu and the collaborating grower in Peninsular Malaysia for access to their orchards and for their guidance on symptom recognition and treatment practice.

## Author contributions

L.D.S. conceived the study, collected and annotated the data, performed all analyses, and wrote the manuscript.

## Competing interests

The author declares no competing interests.

---

## References

1. El Jarroudi, M. et al. Leveraging edge artificial intelligence for sustainable agriculture. *Nat. Sustain.* **7**, 846–854 (2024).
2. Lu, M. et al. Plant disease recognition datasets in the age of deep learning: challenges and opportunities. *Front. Plant Sci.* (2024). [VERIFY authors and volume]
3. Noyan, M. A. Uncovering bias in the PlantVillage dataset. Preprint at arXiv:2206.04374 (2022).
4. David, E. et al. Global Wheat Head Detection (GWHD) dataset: a large and diverse dataset of high-resolution RGB labelled images to develop and benchmark wheat head detection methods. *Plant Phenomics* **2020**, 3521852 (2020).
5. David, E. et al. Global Wheat Head Detection 2021: an improved dataset for benchmarking wheat head detection methods. *Plant Phenomics* **2021**, 9846158 (2021).
6. David, E. et al. Global Wheat Head Detection challenges: winning models and application for head counting. *Plant Phenomics* (2023). [VERIFY authors]
7. Koh, P. W. et al. WILDS: a benchmark of in-the-wild distribution shifts. *Proc. 38th Int. Conf. Machine Learning* (2021).
8. Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C. & Kording, K. P. The need to approximate the use-case in clinical machine learning. *GigaScience* **6**, gix019 (2017).
9. Chaibub Neto, E. et al. Detecting the impact of subject characteristics on machine learning-based diagnostic applications. *npj Digit. Med.* (2019). [VERIFY authors and volume]
10. Brookshire, G. et al. Data leakage in deep learning studies of translational EEG. *Front. Neurosci.* **18**, 1373515 (2024). [VERIFY authors]
11. Tampu, I. E., Eklund, A. & Haj-Hosseini, N. Inflation of test accuracy due to data leakage in deep learning analysis of optical coherence tomography. *Sci. Data* **9**, 580 (2022).
12. Varoquaux, G. & Cheplygina, V. Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digit. Med.* **5**, 48 (2022).
13. Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* **4**, 100804 (2023).
14. Hughes, D. P. & Salathé, M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. Preprint at arXiv:1511.08060 (2015).
15. Mohanty, S. P., Hughes, D. P. & Salathé, M. Using deep learning for image-based plant disease detection. *Front. Plant Sci.* **7**, 1419 (2016).
16. Singh, D. et al. PlantDoc: a dataset for visual plant disease detection. *Proc. 7th ACM IKDD CoDS and 25th COMAD* 249–253 (2020).
17. Moupojou, E. et al. FieldPlant: a dataset of field plant images for plant disease detection and classification with deep learning. *IEEE Access* **11**, 35398–35410 (2023).
18. Nguyen, T. Image dataset of ten durian diseases captured in real-field conditions from a family orchard in Vinh Long, Vietnam. *Data Brief* **63**, 112244 (2025). Dataset: Mendeley Data, https://doi.org/10.17632/mhjwyb5p48.1
19. A durian leaf image dataset of common diseases in Vietnam for agricultural diagnosis. *Data Brief* **61**, 111845 (2025). Dataset: Mendeley Data, https://doi.org/10.17632/pxzvksbwnj.4 [VERIFY authors]
20. Gomes, R. R. et al. *Diaporthe*: a genus of endophytic, saprobic and plant pathogenic fungi. *Persoonia* **31**, 1–41 (2013).
21. Udayanga, D. et al. The genus *Phomopsis*: biology, applications, species concepts and names of common phytopathogens. *Fungal Divers.* **50**, 189–225 (2011).
22. O'Gara, E., Guest, D. I. & Hassan, N. M. Managing *Phytophthora* diseases of durian. *ACIAR Monograph* (2004). [VERIFY]
23. Ultralytics. YOLO11. https://docs.ultralytics.com/models/yolo11/ (2024).
24. Lin, T.-Y. et al. Microsoft COCO: common objects in context. *Eur. Conf. Computer Vision* 740–755 (2014).
25. [VERIFY BEFORE CITING — Malaysian national AI advisory rollout. Confirm against a primary source or remove the specific programme from the Introduction.]

---

## Supplementary Table 1 | Provenance metadata in public crop disease datasets

| Dataset | Year | Size | Setting | Site metadata | What is recorded | EXIF | Split in source paper |
|---|---|---|---|---|---|---|---|
| PlantVillage [14] | 2015 | 54,305 img, 38 cls | lab, detached leaves, uniform background | **no** | leaf identity implicit (4–7 orientations per leaf); no field or plot | stripped in common redistributions | random / stratified random |
| PlantDoc [16] | 2020 | 2,598 img, 27 cls | internet-scraped | **no** | nothing; provenance unknown by construction | n/a | random |
| FieldPlant [17] | 2023 | 5,170 img | field, in plantations | UNVERIFIED | plantation named in paper; per-image mapping unconfirmed | UNVERIFIED | UNVERIFIED |
| PlantWild | 2024 | ~50,000 img | internet-scraped | **no** | — | n/a | UNVERIFIED |
| PlantSeg | 2024 | 11,458 img | internet-scraped | **no** | — | n/a | UNVERIFIED |
| Plant Pathology (FGVC apple) | 2020–21 | ~23,000 img | field, one orchard | UNVERIFIED | single orchard; tree ID unconfirmed | UNVERIFIED | random (Kaggle) |
| Cassava Leaf Disease | 2019/20 | ~21,000 img | field, crowdsourced | UNVERIFIED | collected by farmers; contributor ID unconfirmed | UNVERIFIED | random (Kaggle) |
| BRACOL (coffee) | 2019 | 4,407 img | field/lab | UNVERIFIED | | UNVERIFIED | UNVERIFIED |
| RoCoLe (coffee) | 2019 | 1,560 img | field | UNVERIFIED | | UNVERIFIED | UNVERIFIED |
| DiaMOS Plant (pear) | 2021 | 3,505 img | field, one orchard | UNVERIFIED | one orchard, multiple sessions claimed | UNVERIFIED | UNVERIFIED |
| Citrus (Rauf et al.) | 2019 | 750 img | lab | UNVERIFIED | | UNVERIFIED | UNVERIFIED |
| Durian, Binh Phuoc & Tien Giang [19] | 2025 | 2,595 img, 6 cls | field, 4 orchards | **no** | class name + global index only; orchards named in paper, none identifiable in files | stripped (0/20 sampled); background removed, resized 400×400 | preset random train/test/val across all orchards |
| Durian, Vinh Long [18] | 2025 | 5,451 img, 10 cls | field, 5 orchards | **no** | bare numeric filenames; orchards named in paper, none identifiable in files | absent (0/10 sampled); files are RGBA PNG with XMP `UserComment = Screenshot` | preset Train/Test/Validation across all orchards |
| Global Wheat Head Detection 2021 [5] | 2021 | 6,515 img | field, 12 countries | **yes** | 47 subdatasets, each one site, one date, one sensor | n/a | **grouped by continent, then by session** |
| DeepWeeds | 2019 | 17,509 img | field, 8 locations | **partial** | eight collection sites named | UNVERIFIED | UNVERIFIED |
| *This work* | 2026 | 1,314 img | field | **yes** | GPS per image; farm, tree and capture session recoverable | retained, released with GPS removed | grouped by farm; per-site reported |

Of the fourteen public datasets listed, one (GWHD) demonstrably supports a grouped split as released, one (DeepWeeds) does so partially, six demonstrably do not, and six remain unverified. Rows marked UNVERIFIED report what the source paper or repository states; they require the dataset to be downloaded and its files inspected before any proportion is quoted.
