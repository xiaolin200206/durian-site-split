# Reported accuracy is a property of the evaluation sample: a measurement on four datasets, two modalities and eight architectures

Lin Ding Shan

Faculty of Computer Science (Data Science), UCSI University, Jalan Menara Gading, UCSI Heights, Cheras 56000, Kuala Lumpur, Malaysia

1002475487@ucsiuniversity.edu.my · ORCID 0009-0009-6031-8479

## Abstract

A held-out score is read as a property of a model. When data are collected in bouts, a session, a farm, a patient, a person, it is also a property of which bouts were sampled, and the number alone does not separate the two. We measure that dependence on four datasets spanning two modalities, with eight architectures trained as ten dataset-model combinations. Partitioning by acquisition unit rather than by item lowers the reported score by 37.4 to 44.3% on durian disease detection, 19.6 to 22.4% on wheat head detection, 8.0 to 8.7% on breast histopathology and 3.7 to 4.5% on wearable activity recognition, where the error rate rises 13-fold and 4-fold. Crossed decompositions assign the evaluation unit 88.9% of the variance in the reported figure against 1.1% for the model on durian, 78.7% against 0.4% on histopathology, and 56.1% against 14.6% on activity recognition. Scored one unit at a time, every dataset runs from near-failure to near-perfect. Holding training set size fixed and varying only the number of contributing units, the spread across which units were drawn falls three- to six-fold. No covariate we recorded predicts which unit will fail, and neither does the model's own confidence, whose correlation with per-unit score is −0.26. We also measure a selection bias inside our own protocol: because the checkpoint is chosen on the withheld fold, the figures above are conservative by 8 to 17 points. Two numbers should accompany any aggregate score: how many independent units contributed, and the dispersion across withheld units.

## Introduction

A model is evaluated by withholding part of the data and reporting a score on what was withheld. The number is then read as a property of the model. Two conditions have to hold for that reading to be safe: the evaluation sample has to resemble the population the model will meet, and the items in it have to be close enough to independent that removing some and keeping others does not change what is being measured.

Neither holds when data are collected in bouts. A person walks an orchard and photographs one symptomatic leaf from several angles. A pathology laboratory sections one patient's tumour into dozens of fields of view. A volunteer wears an accelerometer for an afternoon and generates several hundred windows. The items within a bout share far more than the label: a leaf, a stain batch, a gait, a light condition, a device. Splitting them at random puts near-copies on both sides of the boundary, so a model can score well by recognising the bout rather than the phenomenon. Splitting by bout removes that, and leaves a second problem that is less discussed: the score now depends on which bouts were withheld, and different bouts give different answers.

The first problem has a name and a remedy. In medical imaging and physiological signal analysis it is called identity confounding, and subject-wise partitioning is routine [7–11]. Kapoor and Narayanan place this class within a taxonomy of leakage across machine-learning-based science [12]. In agricultural vision the Global Wheat Head Detection dataset organises its images by acquisition session and rejects random splitting explicitly [3,4]. The remedy is established. What is not established is what it leaves behind: how much of a reported figure is attributable to the choice of evaluation units, how that compares with the choices practitioners do report, and how many units are needed before the figure stops moving.

We measure those three quantities on four datasets spanning two modalities (images and inertial time series), three disciplines (agriculture, pathology, human activity) and eight architectures (three capacities of a single-stage CNN detector, a detection transformer, two capacities of a CNN classifier, a multilayer perceptron and a random forest), trained as ten dataset-model combinations. On each we compare item-level with unit-level partitioning, decompose the variance of the reported figure into unit, model, interaction and seed components, score each dataset one unit at a time, and hold the training set size fixed while varying only the number of contributing units.

Five results follow. The cost of correcting the partition ranges over an order of magnitude across datasets, from 3.7% to 44.3%, and tracks how heterogeneous the units are rather than anything about the model. The corrected figure is not itself stable: in every dataset the per-unit scores run from near-failure to near-perfect. The decompositions put the evaluation unit above the model on three of the four datasets, by two orders of magnitude on two of them and by a factor of four on the third, and the ranking of units by difficulty largely survives a change of model family. The dose-response curves narrow how far a particular evaluation can land from the expected score in every dataset tested, while their effect on that expected score ranges from none to +61.5%. And nothing available before deployment predicts which unit will fail, including the model's own confidence, whose correlation with per-unit score has the wrong sign.

We also report three faults inside our own work, because each is an instance of the mechanism this paper is about. Farm identity in the durian corpus was joined by filename; a collision between two visits put 8.9% of the pool under the wrong farm, and the fold that withheld farm 0 trained on farm 6 and tested on images taken at farm 6, scoring highest of the five. A claim about lesion size across farms was true of the per-farm medians and false of the images, because a lens that appears at every farm changes annotated box area up to four-fold. And under leave-one-unit-out the fold's validation set is the withheld unit itself, so the checkpoint that produced every unit-level score in this paper was chosen on the unit it was then scored on. We measure the size of that third fault rather than only declaring it: removing the selection raises the overstatement by 8 to 17 points, which means every figure we report is the conservative one.

The practical consequence is a reporting convention rather than a new method. Two numbers should accompany any aggregate score on bout-structured data: how many independent units contributed to training, and what the dispersion was across withheld units. Both are recoverable at no extra collection cost from any dataset that records where its items came from, and we release a tool that computes them.

## Related work

**The mechanism is documented; its magnitude is not.** A 2024 perspective on plant disease recognition datasets names random splitting as the field's dominant practice and its principal defect: several images of one observation, differing only slightly, land on both sides of the boundary [1]. A related bias in the same datasets is already quantified, in that a classifier trained on eight background pixels from PlantVillage reaches 49.0% accuracy against a 2.6% chance baseline [2], which establishes in weaker form that aggregate accuracy there can be produced by information unrelated to the phenomenon. Neither line measures how large the overestimate is, or what remains once it is removed.

**Grouped splitting is standard in one agricultural vision task, and in medicine.** The Global Wheat Head Detection dataset organises 6,512 images into 47 acquisition domains, each a consistent set acquired over the same experimental unit, session, vector and sensor [3,4]. Its authors state the reasoning directly: although random splitting is common practice, the competition aims to test performance on unseen genotypes, environments and observational conditions, so images are grouped instead. The 2021 edition made validation and test disjoint by session where 2020 had drawn them from shared subdatasets [5], and WILDS adopted the dataset as a domain-generalisation benchmark in which a domain is an acquisition session [6]. In medical imaging and physiological signal analysis the same practice is called subject-wise cross-validation, is documented across MRI, optical coherence tomography, accelerometry, voice and EEG [7–10], and its absence is treated as a defect rather than a design choice [11]. Both literatures establish that grouping is necessary. Neither reports the dispersion between groups as a quantity of interest, and neither asks how many groups are enough.

**The unit is not obvious, and the choice changes the answer.** The agricultural analogue of a patient could be a leaf, a plant, a plot, a farm, a session or a region. We measure at farm level and at capture-burst level on the same durian data and obtain dispersions differing by a factor of four from the same weights. Of fourteen public crop disease datasets surveyed (Supplementary Table 1), spanning laboratory collections [13], web-scraped sets [14] and field collections [15], one releases a site identifier at a granularity that permits grouping at all. For a handheld tool used at one orchard, the relevant unit is the one the field does not record.

## Results

### Four datasets, four evaluation units, eight architectures

| | Modality | Task | Unit | Units | Items | Models | Runs |
|---|---|---|---|---|---|---|---|
| Durian | Images | Detection, 6 classes | Farm | 8 | 827 | YOLO11n/s/m, RT-DETR-L | 180 |
| GWHD 2021 | Images | Detection, 1 class | Acquisition session | 47 | 6,510 | YOLO11s, YOLO11n | 59 |
| BreaKHis | Images | Classification, 2 classes | Patient | 81 | 7,909 | YOLO11s-cls, YOLO11n-cls | 100 |
| UCI HAR | Inertial series | Classification, 6 classes | Subject | 30 | 10,299 | MLP, random forest | 100 |

The durian corpus is our own: 1,314 photographs from eight farms in Peninsular Malaysia and two orchards near Lahad Datu, Sabah, 1,800 km away, captured hand-held under natural light by one person and annotated to a standard set in the field by a crop protection practitioner (Methods). The other three are public and were chosen because the unit is unambiguous and recorded, which is what makes the measurement possible at all.

Two of the four span model families. On durian the levels are three capacities of a single-stage CNN detector and a detection transformer; on HAR a neural network and a tree ensemble, which share only the input features. On GWHD and BreaKHis they are two capacities within one family, because a detection transformer on 5,208 training images costs an estimated 125 GPU-hours against 24 on durian's 662, and because RT-DETR does not apply to a classification task. The model component on those two is therefore a lower bound on what a family change would contribute. The design was not held constant across datasets because the datasets answer different questions: durian carries the model axis with four levels including a family change, GWHD and BreaKHis carry the unit axis with 47 and 81 units against durian's eight, and HAR carries the modality axis with the most dissimilar pair of the four.

In every dataset, item-level random splitting places essentially every unit on both sides of the boundary: 100% of durian farms, GWHD sessions, BreaKHis patients and HAR subjects appear in both training and validation under a random split.

### Partitioning by unit lowers the reported score by 3.7% to 44.3%

| Dataset | Model | Item-level | Unit-level | Overstatement |
|---|---|---|---|---|
| Durian | YOLO11s | 0.4943 ± 0.0055 | **0.2752** ± 0.1077 | **44.3%** |
| | YOLO11m | 0.4966 ± 0.0106 | **0.2847** ± 0.0992 | **42.7%** |
| | RT-DETR-L | 0.5260 ± 0.0108 | **0.3098** ± 0.1002 | **41.1%** |
| | YOLO11n | 0.4558 ± 0.0104 | **0.2855** ± 0.1096 | **37.4%** |
| GWHD | YOLO11s | 0.6733 ± 0.0083 | **0.5228** ± 0.0753 | **22.4%** |
| | YOLO11n | 0.6415 ± 0.0069 | **0.5157** ± 0.0534 | **19.6%** |
| BreaKHis | YOLO11s-cls | 0.9931 ± 0.0015 | **0.9063** ± 0.0334 | **8.7%** |
| | YOLO11n-cls | 0.9915 ± 0.0012 | **0.9126** ± 0.0378 | **8.0%** |
| HAR | Random forest | 0.9790 ± 0.0056 | **0.9347** ± 0.0281 | **4.5%** |
| | MLP | 0.9885 ± 0.0022 | **0.9519** ± 0.0210 | **3.7%** |

Durian and GWHD are mAP50, BreaKHis top-1 accuracy, HAR macro F1; the ± is the standard deviation across folds. Within a dataset the overstatement is nearly invariant to the model: 37.4 to 44.3% across three CNN capacities and a transformer, 19.6 to 22.4% across two capacities, 8.0 to 8.7%, 3.7 to 4.5%. Across datasets it spans an order of magnitude.

Where accuracy is near its ceiling the percentage understates the effect. On BreaKHis the reported figure falls by 8.7 points but the error rate rises from 0.69% to 9.37%, a factor of 13.6; on HAR by 3.7 points and a factor of 4.2. A reader who sees "0.99 against 0.91" and a reader who sees "one error in 145 against one in 11" are being told the same thing with very different force.

The dispersion moves in the opposite direction to the mean, and by more. Across folds, the standard deviation under unit-level partitioning is 20 times the item-level value on durian, 9 times on GWHD, 22 times on BreaKHis and 10 times on HAR. An item-level protocol does not only report a higher number; it reports it with a false precision.

### The corrected figure is not a stable quantity either

| Withheld farm | n | YOLO11s | YOLO11n | YOLO11m | RT-DETR-L |
|---|---|---|---|---|---|
| farm 1 | 102 | **0.458** | **0.466** | **0.457** | **0.440** |
| farm 3 | 62 | 0.373 | 0.420 | 0.348 | 0.373 |
| farm 5 | 135 | 0.281 | 0.273 | 0.311 | 0.371 |
| farm 4 | 101 | 0.277 | 0.260 | 0.317 | 0.353 |
| farm 7 | 57 | 0.275 | 0.291 | 0.280 | 0.315 |
| farm 0 | 141 | 0.260 | 0.253 | 0.248 | 0.292 |
| farm 2 | 147 | 0.164 | 0.147 | 0.166 | 0.205 |
| farm 6 | 82 | **0.115** | 0.175 | 0.151 | **0.130** |

Best to worst is a factor of 4.00 for YOLO11s and 3.38 for RT-DETR-L, while each fold is individually reproducible to within a few hundredths across seeds. The mean of these eight numbers, reported as the detector's accuracy, describes no farm in the set.

The ranking is largely preserved across models. The mean pairwise Spearman correlation over the four architectures is 0.94, ranging from 0.88 to 1.00. All four place farm 1 first. Three of the four place farm 6 last; the smallest model, YOLO11n, reverses farm 6 and farm 2, which are the two lowest-scoring farms under every model and differ by 0.028 under YOLO11n. Which site a model fails on is therefore substantially, though not entirely, a property of the site: the hardest cases are recognisably the same, while the middle of the ranking moves with the model.

### The evaluation unit against the model

Each dataset with two or more models was trained on every fold with every seed, a fully crossed design, so the variance of the reported figure can be decomposed by expected mean squares.

| Dataset | Models | Unit | Model | Interaction | Seed | Unit/model |
|---|---|---|---|---|---|---|
| Durian | 4 | **88.9%** | 1.1% | 5.3% | 4.7% | 79 |
| BreaKHis | 2 | **78.7%** | 0.4% | 0.5% | 20.4% | 215 |
| HAR | 2 | **56.1%** | 14.6% | 28.7% | 0.6% | 3.8 |
| GWHD | 2 | 38.8% | 0.0% | 0.0% | **61.2%** | – |

The evaluation unit is the largest component on three of the four datasets, and its share depends on how different the models being compared are. It is 88.9% across four detectors spanning three capacities and one family change, 78.7% across two capacities of one classifier, and 56.1% across a neural network and a tree ensemble that share only their input features. The model component moves in the same order, 1.1%, 0.4%, 14.6%. Even where the two models are as different as we could make them, the unit accounts for nearly four times what the model does.

**The share is not an artefact of how many models were compared.** Holding the farms, the folds and the seeds fixed and using only YOLO11s and RT-DETR-L, the unit share is 86.4% and the model share 4.4%. Adding YOLO11n and YOLO11m moves them to 88.9% and 1.1%. The seed share is 4.6% and 4.7% under the two designs. Increasing the number of models does not inflate the unit share; it dilutes the model share, because the spread among four models is small relative to the spread among eight farms either way.

**The share is not an artefact of class composition either.** Durian folds differ in which classes their farm happens to carry, so a fold's aggregate mAP averages over a different class mixture. Repeating the decomposition on a single class removes that. On Leaf_rot, present at seven of eight farms, the unit share is 84.9% and the model share 0.0%; on Algal, also at seven, 97.9% and 0.2%. Fixing the class does not reduce the unit share and reduces the model share to nothing.

**GWHD reverses, and part of the reason is measurable.** There the seed accounts for 61.2% and the unit for 38.8%, the only dataset of the four where the unit is not the largest component. The seed standard deviation within a fold is 0.066 for YOLO11s against a between-fold standard deviation of 0.075: the same withheld sessions under three seeds return scores as far apart as 0.40 and 0.60. Part of that is checkpoint selection, quantified below; the remainder is genuine training instability on a task where each fold trains on 37 or 38 sessions and is scored on nine or ten it has never seen. Where a task is that unstable, the seed and the unit are comparable sources of variance, and reporting a single number conceals both.

### A selection bias inside our own protocol

Under leave-one-unit-out the fold's validation set is the withheld unit itself. The training loop computes a fitness on it every epoch, keeps the weights from the best epoch, and stops early on the same signal. The withheld unit did not only receive a score: it selected which epoch produced that score. Every unit-level figure in this paper carries that bias, and so does every item-level figure, but not equally.

Re-validating the final-epoch weights, which no validation signal selected, gives a selection-free estimate. On durian, for the three models whose final checkpoint was retained:

| Model | Item-level, selected → final | Unit-level, selected → final | Overstatement |
|---|---|---|---|
| RT-DETR-L | 0.5260 → 0.5174 | 0.3109 → 0.2162 | 40.9% → **58.2%** |
| YOLO11m | 0.4966 → 0.4924 | 0.2847 → 0.2216 | 42.7% → **55.0%** |
| YOLO11n | 0.4558 → 0.4453 | 0.2855 → 0.2156 | 37.4% → **51.6%** |

The item-level figure barely moves, by 0.004 to 0.011, because that regime's validation set is near-duplicated in training and every epoch scores about the same on it. The unit-level figure falls by 0.063 to 0.095. The overstatement rises by 14.6 points on average.

The same measurement on GWHD gives 22.4% → 30.1% for YOLO11s and 19.6% → 28.5% for YOLO11n, and moves the unit share of the variance from 38.8% to 48.1% while the seed share falls from 61.2% to 51.9%. Selection accounts for about a fifth of that dataset's anomalous seed component; the rest is real.

Two things follow. The overstatements reported throughout this paper are conservative, by 8 to 15 points on the two datasets where we could measure it. And the mechanism is the same one the paper is about, operating one level up: a held-out set that influences any training decision, including which epoch to keep, stops being held out.

### Scored one unit at a time, every dataset spans near-failure to near-perfect

| Dataset | Unit | n | Mean | s.d. | CV | Range |
|---|---|---|---|---|---|---|
| Durian | Capture burst | 58 | 0.330 | 0.247 | **0.75** | 0.000 – 0.995 |
| Durian, Sabah | Tree | 12 | 0.333 | 0.056 | **0.17** | 0.252 – 0.435 |
| GWHD | Session | 47 | 0.474 | 0.126 | **0.27** | 0.173 – 0.767 |
| BreaKHis | Patient | 81 | 0.902 | 0.175 | **0.19** | 0.062 – 1.000 |
| HAR | Subject | 30 | 0.950 | 0.052 | **0.055** | 0.756 – 0.998 |

Each unit was scored with weights that had never seen it. The coefficient of variation is not comparable across datasets whose means sit at different distances from the ceiling: on HAR it is 0.055, but the best subject errs on one window in 570 and the worst on one in four, a ratio of 140. On BreaKHis five of 81 patients fall below 0.60 and 18 are perfect, against a pooled 0.906. On GWHD the best session is 4.4 times the worst, inside a pooled 0.519 reported by the most carefully grouped public dataset in this comparison.

The per-unit distributions are also stable across models where two were run. On BreaKHis the two capacities agree on the hardest patient and their patient rankings correlate at ρ = 0.90; on HAR the neural network and the tree ensemble agree on the hardest subject and on three of their five hardest, at ρ = 0.76.

The durian data allow the same weights to be scored at two granularities in two regions. The peninsular bursts and the Sabah trees have means that agree to 0.004 and dispersions differing 4.5-fold. The Sabah trees sit under one manager, planted and treated alike, photographed across four consecutive days; the peninsular farms are separate operations visited on different dates across eight months. Sabah is uniform because it is one management unit, not because it is in Borneo. Within-farm variation is not negligible either: at farm 7 the five bursts run from 0.011 to 0.995, a standard deviation of 0.424, four times the spread across the eight farm means.

### Crossing 1,800 km costs nothing beyond withholding a neighbouring farm

The Sabah set was never seen in training under any configuration.

| Evaluation | YOLO11s | YOLO11n | YOLO11m | RT-DETR-L |
|---|---|---|---|---|
| Peninsula, item-level split | 0.494 | 0.456 | 0.497 | 0.526 |
| Peninsula, withheld farm | 0.275 ± 0.108 | 0.286 ± 0.110 | 0.285 ± 0.099 | 0.310 ± 0.100 |
| Sabah, all 281 images | **0.270** ± 0.009 | **0.255** ± 0.014 | **0.279** ± 0.014 | **0.303** ± 0.013 |

Every training configuration, averaged over its seeds, scores within a narrow band on Sabah whatever subset of peninsular farms it saw. The standard deviation across the nine configurations is 0.009 to 0.015, against 0.099 to 0.110 across withheld peninsular farms, a factor of seven to twelve. Which farm is withheld moves the result an order of magnitude more than which island is scored. We had designed this study to quantify a cross-region generalisation penalty under controlled acquisition. There is none to quantify.

### How many units are enough

Holding the number of training items fixed and varying only the number of units they came from separates "more data" from "more sites". For each unit count *k*, several disjoint draws of *k* units were made, each subsampled to the same total item count, and each trained with two seeds.

| Dataset | k range | Mean at k_min → k_max | Spread across draws | Fall |
|---|---|---|---|---|
| Durian, farms | 1 → 6 | 0.091 → 0.148 (+61.5%) | 0.033 → 0.009 | 3.8× |
| GWHD, sessions | 4 → 38 | 0.266 → 0.286 (no trend) | 0.040 → 0.007 | 5.9× |
| HAR, subjects, MLP | 4 → 24 | 0.872 → 0.930 (+6.6%) | 0.035 → 0.006 | 6.4× |
| HAR, subjects, forest | 4 → 24 | 0.875 → 0.916 (+4.8%) | 0.016 → 0.005 | 2.9× |

Two things happen at once and they are not equally consistent. The expected score responds differently in each dataset: it rises 61.5% on durian, 6.6% and 4.8% on HAR, and shows no monotone trend on GWHD. The spread across which units were drawn falls in all of them. Adding units buys a report that does not depend on which units you happened to have; whether it also buys a better model depends on the dataset.

The difference follows from how internally varied a unit is. A durian farm is one manager, one cultivar mix and one visit, so going from one farm to six genuinely widens what the model has seen and the mean climbs with it. A GWHD session already mixes location, genotype, growth stage and sensor, so a few sessions span much of the variation and only sampling noise is left to remove. The same ordering governs where draw-to-draw variation falls below seed-to-seed variation: six farms for durian, around sixteen sessions for GWHD, twenty-four subjects for HAR. The answer to "how many units" is not a constant, and a dataset cannot borrow another's.

The durian and GWHD curves were run at 100 and 600 training items, far below the 662–770 and 5,208 used in the main experiments, so their absolute level is not comparable with the tables above. Only the trend with *k* is interpretable.

### Nothing available before deployment predicts which unit will fail

Eight per-farm covariates were available on the durian data.

| Covariate | Pearson *r* | Spearman |
|---|---|---|
| Median capture hour | **+0.73** | +0.69 |
| Median Leaf_rot box side | +0.29 | – |
| Midday share | −0.35 | +0.05 |
| Median solar elevation | −0.32 | −0.02 |
| Number of focal lengths | −0.32 | −0.38 |
| Number of devices | −0.30 | −0.26 |
| Images in fold | −0.23 | −0.19 |
| Share of images at 2.22 mm | +0.06 | +0.02 |

Only capture hour reaches a coefficient worth reporting, and we do not think it is usable. It rests on one farm: farm 1 is the only site photographed after 17:00 and the only one scoring above 0.40, and removing it collapses the association. Solar elevation should carry the same information if light were the mechanism, and it gives −0.32 with a rank correlation of essentially zero.

The same holds elsewhere. On BreaKHis the number of images per patient correlates with that patient's score at *r* = +0.06 over 81 patients; on HAR the number of windows per subject at *r* = +0.18 over 30. Neither the amount of data from a unit nor anything else we recorded says which units will be hard.

In an earlier version of this analysis, on five farms, lesion size correlated with fold score at *r* = +0.54, and we read it as evidence that farms with the earliest lesions are served worst. On eight farms *r* = +0.29, on the six farms with adequate samples within a single focal length *r* = +0.35, and for a second class the sign reverses at −0.21. The worst-scoring farm has the second-largest lesions and the second-worst has the smallest. The claim is withdrawn.

### The model's own confidence does not flag the sites where it fails

If no recorded covariate predicts site-level failure, the remaining candidate is the model's confidence: a system that abstained when uncertain would at least make unpredictable failure visible. We tested this on the durian data with RT-DETR-L, predicting at conf = 0.001 on each withheld farm with the fold that never saw it.

| Withheld farm | Mean max confidence | Silent at 0.25 | AP50 |
|---|---|---|---|
| farm 6 | 0.773 | 0.0% | 0.300 |
| farm 7 | 0.750 | 0.0% | **0.120** |
| farm 1 | 0.737 | 6.9% | 0.130 |
| farm 4 | 0.719 | 6.9% | 0.246 |
| farm 5 | 0.694 | 2.2% | 0.252 |
| farm 0 | 0.683 | 9.9% | 0.351 |
| farm 3 | 0.629 | **14.5%** | **0.580** |
| farm 2 | 0.565 | 8.8% | 0.156 |

Mean confidence correlates with per-farm AP50 at *r* = −0.26, the wrong sign. The farm with the highest score is the one that most often returns nothing, and the farm with the lowest score is never silent. The rank correlation of silence rate with AP50 is +0.61, in the same reversed direction. This is consistent with the known tendency of modern networks to be overconfident away from their training distribution, though we did not measure calibration directly and cannot attribute it.

Abstention is correspondingly weak as a mechanism. Discarding the least confident images and rescoring the remainder raises AP50 from 0.193 to 0.242 at 50% abstention, while discarding 46.8% of the true lesions. For a disease-detection tool that trade is not available: the missed lesions are the failure the tool exists to prevent.

### One label, an order of magnitude in target size, and an optical confound

Class composition differs so sharply between the durian farms that five of the eight folds return no Phomopsis figure: the class occurs at three farms and carries 6,627 boxes, the most of any class, while Leaf_rot occurs at seven and carries 570.

More consequentially, one symptom label spans an order of magnitude in target size across farms, and part of that span turned out to be the camera. Median Leaf_rot box area differs 75-fold between the extreme farms. But the metadata record six focal lengths, and the 2.22 mm ultra-wide lens appears at every farm, from 2% to 65% of a farm's images. Within a single farm, holding disease, cultivar and management fixed, Leaf_rot boxes photographed at 2.22 mm are 1.8- to 4.0-fold larger in area than those at 6.76 mm.

Restricting the comparison to one lens removes the confound: the Leaf_rot ratio falls from 75.0 to 15.0, while Algal moves from 7.5 to 7.0 and Phomopsis from 3.0 to 3.2. For those two classes the between-farm difference is not optical. Leaf_rot drops five-fold and still spans an order of magnitude: median box sides at 6.76 mm are 80 px at farm 1 and 21 px at farm 2, at the 640-pixel training input. A detector trained on 80 px targets and evaluated on 21 px targets is being asked for a different object under the same name.

We had previously reported the unstratified figure and attributed the one farm that broke the pattern to a lens change. Both statements were wrong. No farm was photographed with one lens; the focal-length column had been summarised by its median, which hid the mixing, and the check that should have caught it was testing the median rather than the images.

### An instance of the same mechanism in the corpus itself

Farm identity was originally joined to the annotated durian images by filename. Two visits produced overlapping camera counters; on import the later batch received "(1)" suffixes, and the resize step that produced the 640-pixel annotation set dropped them. Ninety-six stems in the metadata have a sibling differing only by that suffix and attributed to a different farm.

We found this by comparing each annotated image against the originals with a perceptual hash rather than by name. On the 570 images whose filenames still matched an original, the hash recovered that original with a median Hamming distance of 0. Applied to the earlier 560-image pool it showed that 50 images, 8.9%, carried another photograph's metadata; 49 of them were labelled farm 0 and had been taken at farm 6. Because the fold withholding farm 0 trained on farm 6, 29.7% of that fold's held-out set came from a farm the model had seen. That fold scored 0.394, the highest of the five. Under content-based attribution, and the enlarged pool and eight-fold design it made possible, farm 0 scores 0.260.

The error was not visible in the pooled figure, the fold table or the per-farm covariates. It was found only by hashing image content against the originals, which requires that the originals were kept. The same audit on GWHD found one group of duplicate images with inconsistent annotation and 547 degenerate boxes of sub-pixel width among 274,824, and no duplicates spanning the official train, validation and test partition. On BreaKHis it found no duplicates. We report these because the audit is cheap and because its yield on the most carefully constructed public dataset in this comparison was not zero.

## Discussion

Six claims follow from these measurements, and one is withdrawn.

Item-level splitting of bout-structured data overstates the reported score, and the size of that overstatement is a property of the corpus rather than of the practice. It ranges from 3.7% to 44.3% across four datasets and is nearly invariant to the model within each: three CNN capacities and a transformer give 37.4 to 44.3% on the same farms. Where accuracy sits near its ceiling the percentage understates what happened, and the error rate is the honest statement: it rises 13-fold on histopathology and 4-fold on activity recognition.

The corrected figure is a property of the evaluation sample rather than of the model. In all four datasets, scoring one unit at a time produces a range from near-failure to near-perfect while the aggregate reports a single number in the middle of it. A pooled 0.906 on histopathology contains one patient at 0.062 and eighteen at 1.000; a pooled 0.519 on wheat contains sessions from 0.173 to 0.767.

The decomposition puts the evaluation unit above the model on three of the four datasets, by a factor of 80 and 200 where the models compared differ only in capacity and by a factor of four where they are a neural network and a tree ensemble. Its share depends on how different the models compared are rather than on how many. Holding everything else fixed and going from two models to four moves the unit share from 86.4% to 88.9% while the model share falls from 4.4% to 1.1%; repeating the decomposition on a single class, which fixes the class composition that differs between folds, gives unit shares of 84.9% and 97.9% for the two classes present at seven of eight farms, with model shares of 0.0% and 0.2%. The one dataset where the unit is not the largest component is the one where training is least stable, and about a fifth of that instability is checkpoint selection.

Adding units narrows the draw-to-draw spread three- to six-fold in every dataset tested, while its effect on the expected score ranges from none to +61.5% depending on how internally varied a unit is. The point at which draw-to-draw spread falls below training noise differs by dataset, at six farms, sixteen sessions and twenty-four subjects, so "how many units" has no constant answer and a dataset cannot borrow another's.

Nothing available before deployment predicts which unit will fail. Eight capture covariates, the amount of data per unit in three datasets, and the model's own confidence all fail, the last with the wrong sign. It would be more comfortable if the failing sites shared a feature, because a deployer could then screen for it. We looked and did not find one. Some units will fall well below the reported figure, neither the deployer nor the evaluator can say in advance which, and a pooled figure hides both facts.

Finally, we measured a selection bias in our own protocol rather than only declaring it. Because the checkpoint is chosen on the withheld fold, every unit-level figure here is optimistic; removing the selection raises the overstatement by 14.6 points on durian and 8.3 on wheat. The figures we report are the conservative ones. The mechanism is the paper's own subject operating one level up, and it applies to any leave-one-unit-out protocol that selects a checkpoint or stops early on the withheld fold, which is to say most of them.

The withdrawn claim is the stage association. On five farms, lesion size correlated with fold score at *r* = +0.54 and we read it as evidence that early-stage disease is where detection fails. On eight farms it does not hold. Lesion size does vary an order of magnitude between farms even after the optical confound is removed, but it does not tell an evaluator which farm will fail.

A pooled score answers how well a model does on average across the units that happened to be collected. Deployment asks how well it will do at the unit in front of the user. These diverge further the more units the evaluation pools, and the divergence is invisible in the number itself. Two figures would make it visible, and both are recoverable at no extra collection cost from any dataset that records provenance: how many independent units contributed to training, and what the dispersion was across withheld units. Neither appears in the literature we surveyed. We release `unitcheck`, which computes both from a table of item-to-unit assignments and, where available, a table of per-unit scores, reads neither images nor weights, and answers the first question, whether an item-level split leaks, before any model is trained. On all four datasets here that first question returns 100%.

Why this has not been measured before is partly a matter of what the datasets permit. A grouped split requires a unit identifier, and of fourteen public crop disease datasets surveyed, one releases one at a granularity that permits grouping. The two durian datasets [16,24] are illustrative because they are recent, field-collected and describe multiple farms: one names four orchards and ships files carrying a class name and a global index with no EXIF in twenty sampled images, having been background-removed and resized; the other names five and is described as raw smartphone photographs, but as released is 5,274 RGBA PNGs whose XMP records a screenshot provenance. Recording provenance is necessary but not sufficient, and our own pipeline shows why: we recorded location and timestamps for every original, then joined by filename, and 8.9% of the pool ended up under the wrong farm. The audit that found it needed the original files, not just the metadata. Datasets that release derived images without provenance foreclose the check; datasets that release provenance without originals foreclose the audit of the provenance.

Three of our four datasets are images and the fourth is inertial time series, and the overstatement is smallest there. Two mechanisms plausibly contribute and we can separate neither with the data at hand: HAR sits near its ceiling, which compresses any percentage difference, and its 561 features are hand-engineered summary statistics designed to be subject-invariant, so the channel through which unit identity could leak is narrower than in an end-to-end network reading pixels. The dispersion result is unaffected by both, and between-subject variation there is 9.4 times the seed variation, the largest such ratio of the four.

What the results do not license. The eight durian folds are not eight measurements of one quantity: class composition differs so sharply between farms that five return no Phomopsis figure, so the standard deviation of the fold means describes the sample rather than estimating an uncertainty. The by-farm analysis rests on eight farms and excludes 20% of the annotated peninsular set for want of a content match to a located original, and those exclusions are not a random sample, being 56% Phomopsis against 41% in the attributed set. Capture optics were not controlled within farms, so every between-farm scale comparison is reported within one lens, which shrinks the headline Leaf_rot samples to 10 and 14 images. Only two of the four datasets span model families; on the other two the model component is a lower bound. GWHD uses three seeds rather than five because its runs cost two to seven times what the others do. The dose-response curves were run at reduced training set sizes and only their trend with *k* is interpretable. The abstention analysis is on one architecture and one dataset. And all results are offline evaluations: no grower, pathologist or wearer has used any of these systems in the field.

## Methods

### Datasets and units

**Durian.** Photographs were taken at farms in Peninsular Malaysia (Selangor, Negeri Sembilan and the Muar district of Johor) between December 2025 and July 2026, and at two orchards near Lahad Datu, Sabah, between 13 and 16 August 2026. Location clustering of the originals returned nine peninsular sites among the annotated images; eight hold at least 20 and are the farms analysed. The Sabah sites are a 20-acre mature planting under a single manager of eleven years' tenure (trees approximately nine years old; 70% Musang King, 20% Black Thorn) and a separate 9-acre planting of 3.5-year-old trees. Peninsular farms lie within 134 km of one another; the Sabah orchards are 1,765–1,872 km away.

All images were captured hand-held under natural light by one person. The primary device was an iPhone 16 Pro Max; an iPhone 13 Pro was used at three peninsular farms when the primary device throttled thermally, and a single image was taken on an iPhone 13. The second device did not record location, so those farms are the three with the lowest location coverage (44%, 64%, 74%) and their images were attributed by capture-time proximity. Six focal lengths occur in the pool (2.22, 5.1, 5.7, 6.76, 9.0 and 15.66 mm) and every farm mixes at least two. No background removal or colour correction was applied. Images were resized to 640 × 640 for training, which is the form in which they are released, since it is the form every reported number was computed on.

Annotation is lesion-level: each visible lesion or insect receives its own box. Class definitions and symptom recognition were calibrated in the field with a crop protection practitioner in Johor with several decades of regional experience, who identified lesions on the tree during collection. Annotation was then outsourced per class against that standard and reviewed image by image by the author. Six classes were retained after consolidation. Of 1,033 annotated peninsular images, 908 carry at least one box; the remaining 125 are background frames retained as negatives.

Registered products set a limit on how far classes can be merged. Scale insects and mealybugs are visually similar sap-feeders that a detector would be tempted to merge, but the products registered against them in Malaysia belong to different insecticide mode-of-action groups. A class list built for a spray decision must respect the separations that registered chemistry already encodes, as well as the merges; we applied this once in the other direction, reducing four psyllid categories to two on the ground that a single egg and a cluster receive the same registered application. The standard was set by a practitioner rather than a plant pathologist because the label a handheld tool must return is the one that maps onto a registered product, not the one that maps onto a species boundary requiring sequence data; the taxonomic limits of names such as *Phomopsis* [17,18] are not resolvable from a photograph by anyone.

**GWHD 2021** [4] was obtained as the released parquet distribution, decoded to 6,512 images with their acquisition domain, country, location and development stage, and converted to single-class detection labels. Sub-pixel boxes were removed (547 of 274,824). One group of two byte-identical images carrying inconsistent annotations was dropped, leaving 6,510 across 47 domains.

**BreaKHis** [22] was obtained from the maintaining laboratory and parsed from its directory structure, which encodes patient identifier, magnification and class in each filename: 7,909 images from 81 patients at four magnifications, binary benign and malignant, no duplicates by content hash and no patient carrying both labels. The original description reports 82 patients; we recovered 81 distinct identifiers from the released filenames and analyse those.

**UCI HAR** [21] was obtained from the standard distribution: 10,299 windows from 30 subjects, 561 engineered features, six activities. Its official release is already partitioned by subject (21 train, 9 test), which we did not use; we constructed our own five-fold partitions for comparability with the other datasets.

### Farm attribution by image content

Farm identity for the durian originals was recovered from EXIF location by single-link clustering at 1.5 km. Originals without location were attributed by capture-time proximity, accepted only where the nearest location-bearing image of the same day was within 30 minutes. Of 2,038 originals, 1,662 were attributed by location and 278 by timing.

Farm identity was originally joined to the annotated images by filename, and that join failed for the reason given in Results. Attribution is now decided by content. Each annotated image is compared against the originals with a 64-bit difference hash and takes the farm of its nearest original by Hamming distance, accepted only when no original from a different farm lies within three bits of the best match; images that fail this test are dropped. Validation on the 570 images whose filenames still matched an original recovered that original with a median Hamming distance of 0.

The procedure attributed 829 of 1,033 annotated images and recovered 303 whose filenames had been replaced by platform sequence numbers. Of the 204 not attributed, 160 matched no original within eight bits and 44 matched an original that itself had no location. Of the 160, 25 had passed through instant-messaging applications, which re-encode images, and 14 are named as extracted frames. The excluded images over-represent Phomopsis (56% of their boxes against 41% in the attributed set) and under-represent Psyllid_damage (4% against 17%).

### Partitions

For each dataset, two regimes over an identical pool, so the split rule is the only difference between them.

*Item-level*: a stratified random 80/20 split by dominant class on durian (662 train, 165 validation), and stratified random five-fold on the other three.

*Unit-level*: leave-one-unit-out where the unit count permits (durian, eight folds), otherwise unit-disjoint five-fold assembled by greedy bin-packing on item count (GWHD, BreaKHis, HAR). Every fold's held-out units are absent from its training set; both this and the absence of any item spanning folds are asserted by the split scripts and re-checked by the verification script.

The durian pool is 827 images across eight farms; a ninth cluster of two images is excluded from both regimes. Sabah was never split: all 281 images serve as a held-out test set for every peninsular model.

### Training

Durian detection: Ultralytics YOLO11n, YOLO11s, YOLO11m and RT-DETR-L [19], 640 px, maximum 150 epochs with patience 50, five seeds (42, 1, 2, 3, 4) per configuration, 180 runs. Batch size was set per model to the largest that fits in 24 GB at 640 px: 32 for YOLO11n and YOLO11s, 16 for YOLO11m, 24 for RT-DETR-L.

GWHD: YOLO11s and YOLO11n, 640 px, 150 epochs, patience 50, batch 32, three seeds (42, 1, 2), 59 runs. Three rather than five because a single GWHD run costs two to seven times what a run on the other datasets costs: the training set is 5,208 images at 640 px and few runs stop early.

BreaKHis: YOLO11s-cls and YOLO11n-cls, 224 px (the standard input for classification; 640 px exceeded the container memory limit), 100 epochs, patience 50, batch 256, five seeds, 100 runs.

HAR: a single-hidden-layer perceptron (256 units, 300 iterations) on standardised features and a random forest (500 trees, no scaling), five seeds each, 100 runs. The two share only the input features, which is why they are the widest model contrast in the study.

### Evaluation

All reported figures come from re-validating a saved checkpoint through one code path. Per-unit evaluation constructs one dataset manifest per unit and re-runs validation without retraining; each unit is scored with the fold whose held-out set contains it, so no unit is ever scored by a model that saw it during training.

**The checkpoint is chosen on the withheld fold.** Under leave-one-unit-out the fold's validation set is the withheld unit, and the training loop keeps the weights from the epoch with the best fitness on it and stops early on the same signal. The withheld unit therefore influences which weights are reported, and every unit-level figure in this paper carries that bias. We quantify it rather than only declaring it: `checkpoint_bias.py` re-validates the final-epoch weights, which no validation signal selected, on every run that retained them, and the comparison is reported in Results. The item-level regime carries the same bias but is almost unaffected by it, because its validation set is near-duplicated in training.

mAP50 [20] is the primary metric for detection; mAP50-95 is reported alongside in the released tables. BreaKHis uses top-1 accuracy and HAR macro F1.

Units with fewer than five items are excluded: twelve of thirteen Sabah trees remain, and at the 60 s burst gap 58 of 148 peninsular bursts remain, covering 652 of the pool's 827 images.

**Capture bursts.** Sabah filenames carry a tree identifier. Peninsular filenames do not, so images were grouped into capture bursts, consecutive frames from the same farm less than 60 s apart. The gap is a choice, and the analysis was repeated at four settings (Supplementary Table 3): coverage runs from 42% to 79% of the pool while the coefficient of variation stays between 0.70 and 0.75. The threshold determines how many images enter the analysis, not how dispersed the sites are. We report the 60 s setting because it covers the most images.

### Variance decomposition

Models × folds × seeds, fully crossed. Components were estimated by expected mean squares for a two-factor random-effects model. Both factors are treated as random because the question is how much variability each contributes rather than which model is better; this is the framing used in measurement system analysis, where the operator factor is likewise a fixed roster treated as a variance source. Reading the model term as a variance component does assume the levels sample a population of models, which two to four architectures do only loosely, and the intervals below reflect that. Confidence intervals are 90% percentile bootstrap over units (2,000 resamples), which is the level at which the design is exchangeable.

Two controls are reported. The decomposition was repeated with two models and with four on the same durian folds, to test whether the unit share depends on how many models are compared; and on single classes, to test whether it depends on the class composition that differs between farms.

### Dose-response

For each unit count *k*, disjoint draws of *k* units were made at random from the pool remaining after the held-out units were removed. Each draw was subsampled to a fixed total item count, allocated evenly across its units, so that item count is constant across *k* and only unit count varies. Where the budget does not divide exactly the realised total falls a few items short: on HAR it runs from 1,000 windows at *k* = 4 to 984 at *k* = 24, a 1.6% spread in the direction opposite to the measured effect. Each draw was trained with two seeds. Draws that could not reach the item budget were skipped, which is why the durian curve starts at *k* = 1 with four draws and the GWHD curve has a single draw at *k* = 2, whose across-draw spread is undefined and excluded.

### Abstention

The selected checkpoint for each durian fold was run over that fold's held-out farm at conf = 0.001, so the low-confidence tail, the object of the analysis, is retained. Detections were matched to ground truth greedily by descending confidence at IoU ≥ 0.5 within class. Per-image maximum confidence defines the abstention order; AP50 on the retained subset was recomputed from the matched detections with all-point interpolation.

### Relation to prior work by the same author

The peninsular photographs derive from the same field collection as an earlier study of capture-session leakage in five-class classification (under review; dataset and code at https://doi.org/10.5281/zenodo.22177133). The two analyses share source photography but not an analysis set. That work partitions 560 images into 73 capture sessions, reports classification macro F1, and finds that re-partitioning by session lowers it in all nine architectures tested (sign test *p* = 0.004). This work re-annotates at the box level, re-derives farm attribution by perceptual hash against the original files rather than by filename, recovers 303 images whose filenames the annotation platform had replaced, and analyses 827 images across eight farms with detection models. The evaluation unit, the task, the annotation level, the attribution procedure and the analysis set all differ. The two results are consistent in direction at two different granularities, which is why the earlier one is cited here as an independent measurement rather than as a component of this one.

### Data availability

The durian images and annotations are deposited at https://doi.org/10.5281/zenodo.22030622 under CC BY-NC 4.0, at the 640 × 640 resolution on which every reported number was computed, with location coordinates stripped from EXIF. The content-based farm attribution for every image, the tree identifiers for Sabah, and all split manifests are released alongside, so every partition reported here and others we did not use can be reconstructed. GWHD, BreaKHis and UCI HAR are public; the scripts that reproduce our processing of each are released. All result tables are released without restriction.

### Code availability

Corpus preparation, farm attribution by perceptual hash, split construction, training, per-unit evaluation, variance decomposition, dose-response, abstention and the checkpoint-bias measurement are at https://github.com/xiaolin200206/durian-site-split, archived at https://doi.org/10.5281/zenodo.22031684. The audit is packaged separately as `unitcheck.py`, which runs on any dataset with a unit identifier and depends only on the standard library. Every quantitative claim in this paper is recomputed from the released tables by a single script that runs on every commit; a claim whose input table does not exist fails the build rather than passing silently. Three of the errors reported in this paper were found by that process rather than by inspection.

### Limitations

**The folds are not comparable to each other.** Class composition differs so sharply between durian farms that five folds return no Phomopsis figure. The fold mean is reported because it is the conventional summary, but the eight scores are not eight measurements of one quantity, and their standard deviation describes the sample rather than estimating an uncertainty. The single-class decomposition in Results is the check on this.

**Eight farms, and 20% of the peninsular labelled set excluded.** The excluded images are not a random sample: their boxes are 56% Phomopsis against 41% in the attributed set. The direction of any resulting bias in the headline comparison is not known.

**Only two of four datasets span model families.** On GWHD and BreaKHis the two levels are capacities within one family, so the model component there is a lower bound on what a family change would contribute. The two datasets where a family change was affordable are also the two furthest apart in modality.

**GWHD uses three seeds.** Its runs cost two to seven times what the others do. The residual degrees of freedom fall from 40 to 20, so its variance components are less precisely estimated than the others'.

**Capture optics were not controlled.** Six focal lengths occur and every farm mixes at least two. The lens changes annotated box area by up to four-fold for one class within a single farm, so every between-farm scale comparison is reported within one lens, which shrinks the headline Leaf_rot samples to 10 and 14 images.

**Sabah is one manager, one season, four days, and bursts are only a proxy for trees.** A second Bornean orchard under different management would very likely widen the spread, and the season was unusually dry by the manager's account, which suppresses *Phytophthora* expression [23]. On the peninsular side a burst is defined by a time gap; the coefficient of variation is insensitive to that gap between 15 and 60 s, but coverage is not, and at the reported setting a fifth of the pool's images sit in bursts too small to score. The peninsular per-burst scores come from models trained on seven farms while Sabah scores come from models trained on all eight; that asymmetry disadvantages Sabah, so the finding that Sabah is the more uniform domain is conservative.

**One collector, one annotator, and a protocol less uniform than intended.** Two devices, six focal lengths and two leaf framings all occur within single farms, and none was recorded at capture. Inter-annotator agreement was not measured, and the labelling standard was set in the field rather than in a written protocol, so a reader cannot reproduce it from the released material; what is released is the standard's output. Single psyllid eggs required magnification far beyond normal review scale to locate and were not separable at that magnification from reflections, trichomes and water droplets; images whose only annotation was a single egg were withdrawn, which was itself a judgement.

**The checkpoint bias was measured on three durian models and two GWHD models.** The YOLO11s durian runs were trained before final-epoch weights were retained, so they are absent from that comparison. All five models measured move in the same direction.

**The dose-response curves are at reduced scale**, 100 training images on durian and 600 on GWHD against 662–770 and 5,208 in the main experiments. Only the trend with *k* is interpretable.

**Modality is confounded with ceiling and with feature engineering.** The smallest overstatement is on the only non-image dataset, which is also the only one near its accuracy ceiling and the only one using hand-engineered features.

**No inference on the headline comparisons.** The item-level and unit-level figures are point estimates over the available seeds and folds; we report dispersion at both levels but construct no confidence interval on their difference, because the fold scores are not draws from one distribution. The variance decompositions carry bootstrap intervals over units, which is the level at which the design is exchangeable.

**No deployment.** All results are offline evaluations. None predicts the detection rate a user would experience.

## Figure captions

**Figure 1 | The same weights, scored one unit at a time.** Each marker is one evaluation unit scored with weights that never saw it; error bars are the standard deviation across training seeds. Panels: 58 peninsular durian capture bursts, 12 Sabah trees, 47 GWHD acquisition sessions, 81 BreaKHis patients, 30 HAR subjects. Solid blue lines are the pooled unit-level figures those same weights produce; red dashed lines are what an item-level split reports on the same data. Pooled figures are instance-weighted and the markers are one per unit, so where a unit's item count varies the two differ; the dotted line marks the equal-weight mean where it separates from the pooled figure. In every dataset the units span from near-failure to near-perfect and the pooled number reports none of that range. The two durian panels share weights, agree to 0.004 in mean, and differ 4.5-fold in dispersion.

**Figure 2 | The reported figure is a property of the sample, not the model.** Variance components of the durian mAP50 over a fully crossed design of four models × eight withheld farms × five seeds, estimated by expected mean squares, with 90% percentile bootstrap intervals over farms. The evaluation farm accounts for 88.9% of the variance; the model for 1.1%. Inset: fold means under the four models, showing a preserved ranking (mean pairwise Spearman ρ = 0.94); all four place farm 1 first and three of four place farm 6 last.

**Figure 3 | Adding units buys stability; whether it also buys accuracy depends on the dataset.** For each unit count *k*, disjoint draws of *k* units were trained at a fixed total item count and evaluated on held-out units. Lines are the mean across draws and seeds; bands are the spread across draws. The draw-to-draw spread falls three- to six-fold in all three datasets; the expected score rises 61.5% on durian, 6.6% on HAR and not at all on GWHD. The dashed line marks seed-to-seed variation; the crossing point differs by dataset, at six farms, sixteen sessions and twenty-four subjects, in the order predicted by how internally varied a unit is.

**Figure 4 | Nothing available at deployment predicts which unit will fail.** Left, eight per-farm capture covariates against fold score on the durian data; only capture hour reaches |*r*| > 0.5 and it rests on a single farm, while solar elevation, which should carry the same signal, gives −0.32. Right, the model's own mean confidence against per-farm AP50: the correlation is −0.26, and the farm that most often returns nothing is the one that scores highest.

**Figure 5 | A selection bias inside the protocol.** Under leave-one-unit-out the fold's validation set is the withheld unit, so the checkpoint was chosen on the unit it was then scored on. Re-validating the final-epoch weights, which no validation signal selected, leaves the item-level figure almost unchanged and lowers the unit-level figure substantially, raising the overstatement by 14.6 points on durian and 8.3 on GWHD. Every figure reported in this paper is therefore the conservative one.

## Supplementary information

Supplementary Tables 1–8 and Supplementary Figure 1 are provided as a separate file.

## Acknowledgements

The author thanks the collaborating grower in Johor, who set the symptom recognition standard in the field and identified lesions on the tree, and the orchard manager in Lahad Datu for access to the Sabah site.

## Author contributions

L.D.S. conceived the study, collected and annotated the durian data, performed all analyses, and wrote the manuscript.

## Use of AI tools

All fieldwork, image capture, annotation review, class definition, model training and the decision to withdraw the claims noted above were carried out by the author. A large language model (Claude, Anthropic) was used as a coding and editing assistant: it drafted analysis and audit scripts operating on the released tables and data, including the perceptual-hash attribution, the focal-length stratification, the variance decomposition, the dose-response experiment, the checkpoint-bias measurement and the verification script, and assisted with editing the manuscript and locating candidate references. The author reviewed every script and its output and re-ran each on the released data.

The verification described under Code availability is the check on this arrangement: every quantitative claim is recomputed from the released tables by a script that runs on each commit, and a claim whose input table is absent fails the build rather than passing silently. Three errors reported in this paper were found by that process and not by inspection. Every reference was checked against the publisher record. The author takes full responsibility for the content of this paper.

## Competing interests

The author is developing an expert system for crop disease detection and thus has a potential financial interest in the domain the durian study evaluates. Access to the Sabah orchards was facilitated by a commercial partner who had no role in study design, data collection, annotation, analysis or the decision to publish, and provided no funding, products or data.

## References

1. Xu, M., Park, J.-E., Lee, J., Yang, J. & Yoon, S. Plant disease recognition datasets in the age of deep learning: challenges and opportunities. *Front. Plant Sci.* **15**, 1452551 (2024).
2. Noyan, M. A. Uncovering bias in the PlantVillage dataset. Preprint at arXiv:2206.04374 (2022).
3. David, E. et al. Global Wheat Head Detection (GWHD) dataset: a large and diverse dataset of high-resolution RGB labelled images to develop and benchmark wheat head detection methods. *Plant Phenomics* **2020**, 3521852 (2020).
4. David, E. et al. Global Wheat Head Detection 2021: an improved dataset for benchmarking wheat head detection methods. *Plant Phenomics* **2021**, 9846158 (2021).
5. David, E. et al. Global Wheat Head Detection challenges: winning models and application for head counting. *Plant Phenomics* **5**, 0059 (2023).
6. Koh, P. W. et al. WILDS: a benchmark of in-the-wild distribution shifts. *Proc. 38th Int. Conf. Machine Learning* (2021).
7. Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C. & Kording, K. P. The need to approximate the use-case in clinical machine learning. *GigaScience* **6**, gix019 (2017).
8. Chaibub Neto, E. et al. Detecting the impact of subject characteristics on machine learning-based diagnostic applications. *npj Digit. Med.* **2**, 99 (2019).
9. Brookshire, G. et al. Data leakage in deep learning studies of translational EEG. *Front. Neurosci.* **18**, 1373515 (2024).
10. Tampu, I. E., Eklund, A. & Haj-Hosseini, N. Inflation of test accuracy due to data leakage in deep learning analysis of optical coherence tomography. *Sci. Data* **9**, 580 (2022).
11. Varoquaux, G. & Cheplygina, V. Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digit. Med.* **5**, 48 (2022).
12. Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* **4**, 100804 (2023).
13. Hughes, D. P. & Salathé, M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. Preprint at arXiv:1511.08060 (2015).
14. Singh, D. et al. PlantDoc: a dataset for visual plant disease detection. *Proc. 7th ACM IKDD CoDS and 25th COMAD* 249–253 (2020).
15. Moupojou, E. et al. FieldPlant: a dataset of field plant images for plant disease detection and classification with deep learning. *IEEE Access* **11**, 35398–35410 (2023).
16. Thanh, T. N., Nguyen, L. X., Cap, T. & Le, T. A durian leaf image dataset of common diseases in Vietnam for agricultural diagnosis. *Data Brief* **61**, 111845 (2025).
17. Gomes, R. R. et al. *Diaporthe*: a genus of endophytic, saprobic and plant pathogenic fungi. *Persoonia* **31**, 1–41 (2013).
18. Udayanga, D. et al. The genus *Phomopsis*: biology, applications, species concepts and names of common phytopathogens. *Fungal Divers.* **50**, 189–225 (2011).
19. Ultralytics. YOLO11 and RT-DETR. https://docs.ultralytics.com/models/ (2024).
20. Lin, T.-Y. et al. Microsoft COCO: common objects in context. *Eur. Conf. Computer Vision* 740–755 (2014).
21. Anguita, D., Ghio, A., Oneto, L., Parra, X. & Reyes-Ortiz, J. L. A public domain dataset for human activity recognition using smartphones. *Proc. 21st European Symposium on Artificial Neural Networks* 437–442 (2013).
22. Spanhol, F. A., Oliveira, L. S., Petitjean, C. & Heutte, L. A dataset for breast cancer histopathological image classification. *IEEE Trans. Biomed. Eng.* **63**, 1455–1462 (2016).
23. O'Gara, E., Guest, D. I. & Hassan, N. M. Botany and production of durian (*Durio zibethinus*) in Southeast Asia. In *Diversity and Management of Phytophthora in Southeast Asia* (eds Drenth, A. & Guest, D. I.) 180–186 (ACIAR Monograph 114, 2004).
24. Nguyen, T. Image dataset of ten durian diseases captured in real-field conditions from a family orchard in Vinh Long, Vietnam. *Data Brief* **63**, 112244 (2025).
