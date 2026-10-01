# What a new orchard costs: farm-level generalisation and data budgets for durian disease and pest detection

**Lin Ding Shan**

Faculty of Computer Science (Data Science), UCSI University, Kuala Lumpur, Malaysia

Corresponding author: Lin Ding Shan, 1002475487@ucsiuniversity.edu.my, ORCID 0009-0009-6031-8479

## Abstract

Handheld detectors of durian (*Durio zibethinus*) diseases and pests are meant for growers whose farms were not in the training data, yet are often scored on a random split of images from their training farms. We measured what this hides with 827 annotated images (733 with boxes) from eight commercial farms in Peninsular Malaysia and 281 from two orchards in Sabah. On a random split, six detectors reached 1.6–2.3 times the mean average precision (mAP50) they reached on farms held out from every training decision. Per-farm scores, averaged over detectors, spanned 0.094–0.356, whereas the detectors' mean scores differed by at most 0.022. Psyllid and psyllid damage kept 25–30% of their random-split average precision on a new farm, and the other classes 48–59%. In 336 training runs of equal length that varied the number of training farms (1–7) and images per farm (15 to all), a doubling of farms raised unseen-farm mAP50 by 0.038 and a doubling of images per farm by 0.028, a difference within the uncertainty. With the recipe tested, fine-tuning on a new farm's images alone lowered that farm's mAP50 with 5–20 images (by 0.111 with 5) and did not improve it with about 50. Adding about 50 of those images to the training set and retraining raised it by 0.053. Orchard detectors should be evaluated on held-out farms, collections should first cover the region's diseases and pests across farms, and, with the fine-tuning recipe tested, a new farm's images belong in the training data.

**Keywords:** Durian; Plant disease detection; Insect pest detection; Object detection; Domain shift; Leave-one-farm-out validation; Training data collection

## 1. Introduction

Durian is a high-value fruit crop across Southeast Asia (O'Gara et al., 2004), and its diseases and insect pests are commonly identified in the field from their visible symptoms. A handheld device that photographs a leaf or trunk and suggests which disease or pest is present could shorten the time between a symptom appearing and a grower acting on it. Such a device is used by growers whose farms were not in its training data, so the number that matters to them is how well the detector works on a farm it has never seen.

That number is not always the one reported. Plant-disease detectors are often evaluated on a random partition of one pool of images, so that images from the same farm, and often from the same tree, appear on both sides of the split. The consequences of mismatched evaluation are documented in general terms. Models trained on laboratory images of single leaves lose most of their accuracy on images from other sources (Mohanty et al., 2016), limited variety in training data is a principal barrier to field use (Barbedo, 2018a, 2018b), and training on in-the-wild images improves field performance (Singh et al., 2020). In camera-trap ecology, where the unit is a camera location, accuracy at locations absent from training is consistently lower than at trained locations (Beery et al., 2018; Tabak et al., 2019; Schneider et al., 2020; Norman et al., 2023). Benchmarks of in-the-wild distribution shift show the same pattern for wheat-head detection across acquisition domains (Koh et al., 2021). Grouped or blocked validation is the standard remedy for spatially structured data (Roberts et al., 2017), and evaluating on data that share structure with the training set is a recognised form of leakage (Kapoor and Narayanan, 2023).

A team building a detector for a new crop therefore faces three questions. How large is the gap between a random split and an unseen farm for multi-class, lesion-level detection of orchard diseases and pests, and does it fall evenly across classes? Should a limited collection budget go to more farms or to more images from each farm? And do a few images taken on arrival at a new farm recover what the change of farm costs? Related work bears on the second and third. For camera traps, a supplementary experiment of Beery et al. (2018) found that accuracy at new locations was stable once more than two training locations were used. For weed detection in arable fields, Ruigrok et al. (2023) found that, at a constant number of training images, drawing them from more sub-datasets improved generalisation, that more images from the same sub-datasets helped only when they added new variation, and that fine-tuning on as few as 25 images from the new field already reduced the generalisation error. To our knowledge, none of these has been measured for disease and pest detection in orchards, where the targets are small lesions and insects and where each farm carries only some of the classes. They cannot be measured on most public plant-disease datasets, which do not record where each image was taken (Xu et al., 2024), and the durian datasets released to date either come from one family orchard and a few neighbouring farms (Nguyen, 2025) or do not describe a per-image orchard identifier (Nguyen Thanh et al., 2025). A recent durian pest-and-disease detector was developed and evaluated on a random split of images from a single site (Tang et al., 2025).

This paper answers the three questions on a durian corpus in which every image carries its farm. Its contributions are:

1. A measurement of the unseen-farm gap for six detectors from three architecture families, with each held-out farm excluded from every training decision, broken down by farm and by class (five detectors for the per-class and Sabah analyses), and checked on two orchards on another island.
2. A controlled experiment of 336 training runs of equal length (about 2,000 iterations each) that varies the number of training farms and the number of images per farm independently.
3. An experiment on adapting a trained detector to a new farm with images from that farm, from 5 images up to half of them, by fine-tuning and by retraining.
4. A released dataset with farm identifiers and split manifests, and code that recomputes every number reported here.

## 2. Materials and methods

### 2.1. Study sites and image collection

Images were taken at commercial durian farms in Selangor, Negeri Sembilan and the Muar district of Johor between December 2025 and July 2026, and at two orchards near Lahad Datu, Sabah, on 13–16 August 2026. The peninsular collection comprises nine sites, of which the eight holding at least 20 annotated images form the analysis pool (Table 1). The peninsular farms lie within 134 km of one another and are separate operations under different managers. The Sabah orchards, 1,765–1,872 km away, are an 8.1 ha mature planting under one manager and a separate 3.6 ha planting; 261 of the 281 annotated Sabah images come from the first and 20 from the second. In what follows, *farm* denotes a peninsular farm and *orchard* one of the two Sabah plantings, except in the title and in general statements about orchard crops. All images were captured hand-held under natural light by one person, primarily on an iPhone 16 Pro Max; an iPhone 13 Pro was used at three farms and an iPhone 13 for two images at a fourth. Six focal lengths occur (2.22–15.66 mm) and every farm mixes at least two. Six of the eight peninsular farms were photographed on a single day and two over two days, so here a farm largely coincides with one visit. Images were resized to 640 × 640 pixels for training and are released at that resolution.

### 2.2. Classes and annotation

Six classes were annotated at lesion level, one box per visible lesion or insect: algal spot, leaf rot, *Phomopsis*, psyllid, psyllid damage and leafhopper damage. The class list mixes symptom names with a fungal genus because these are the names growers and extension services use. The annotation standard was agreed in the field with a grower and crop-protection practitioner in Johor with several decades of regional experience, annotation was outsourced per class against that standard, and every image was reviewed by the author. Inter-annotator agreement was not measured. Classes are unevenly distributed across farms (Fig. 1, Table 1): *Phomopsis* occurs on only three of the eight farms, and each farm carries between four and six of the six classes. The 827 images in the pool contain 13,134 boxes; 94 images show no annotated symptom.

![](../results_applied/fig1_data.png){width=16cm}

**Fig. 1.** Number of images containing each class on each of the eight peninsular farms (total images per farm in brackets); a dash marks a class absent from the farm. Farms differ both in size and in which diseases and pests they carry.

**Table 1.** The eight peninsular farms: images, annotated boxes, and the number of images containing each class (a dash marks a class absent from the farm).

| Farm | Images | Boxes | Algal spot | Leaf rot | *Phomopsis* | Psyllid | Psyllid damage | Leafhopper damage |
|---|---|---|---|---|---|---|---|---|
| 0 | 141 | 938 | 54 | 26 | – | 4 | 4 | 18 |
| 1 | 102 | 1,364 | 29 | 17 | – | 21 | 21 | 2 |
| 2 | 147 | 4,094 | 50 | 22 | 108 | 48 | 26 | 7 |
| 3 | 62 | 217 | 30 | 5 | – | 1 | – | 8 |
| 4 | 101 | 2,036 | 8 | 33 | – | 50 | 26 | 14 |
| 5 | 135 | 2,440 | 26 | 3 | 57 | 29 | 35 | 31 |
| 6 | 82 | 1,576 | 7 | 16 | 80 | – | – | 2 |
| 7 | 57 | 469 | – | 18 | – | 14 | 25 | 19 |
| All | 827 | 13,134 | 204 | 140 | 245 | 167 | 137 | 101 |

### 2.3. Farm attribution

Farm identity was recovered for each original photograph from the location stored in its EXIF (exchangeable image file format) metadata by single-link clustering at 1.5 km; originals without location were attributed to the farm of the nearest located image taken within 30 minutes on the same day. Each annotated image was then matched to its original by a 64-bit difference hash, accepted only when no original from a different farm lay within three bits, and assigned that original's farm. This attributed 829 of 1,033 annotated peninsular images; the eight farms with at least 20 images hold the 827 used here. An earlier attribution that joined images to their metadata by filename had placed 50 images (8.9% of the pool at the time) under the wrong farm, because two field visits produced overlapping camera counters. The error did not show in pooled scores and was found by matching each image to its kept original photograph (Note S2).

### 2.4. Detectors

The handheld device runs YOLO11n (Jocher and Qiu, 2024) exported to ONNX (Open Neural Network Exchange) format on a Raspberry Pi 5, offline, as a decision aid used on demand; the device itself is not evaluated here. To test whether the conclusions depend on that choice, five further detectors were trained and evaluated under the same protocol: YOLO11s, YOLO11m and YOLO11l; RT-DETR-L, a real-time detection transformer (Zhao et al., 2024); and Faster R-CNN, a two-stage anchor-based detector sharing no code with the others, in torchvision's ResNet-50 feature-pyramid-network (FPN) v2 configuration (Ren et al., 2017; Li et al., 2021b). The Ultralytics detectors were pretrained on COCO (Common Objects in Context; Lin et al., 2014) and trained at 640 px for up to 150 epochs with patience 50, default augmentation and five seeds. Faster R-CNN was trained from COCO weights with its own training loop and three seeds (stochastic gradient descent; settings in Note S1). Its augmentation (horizontal flips only) and input size (torchvision's default resizing) therefore differ from those of the Ultralytics detectors. mAP50, the mean over classes of average precision (AP) at an intersection-over-union threshold of 0.5 (Lin et al., 2014), is the primary metric; Faster R-CNN was scored with torchmetrics' implementation of the same COCO definition, so its comparison with the other five carries additional implementation uncertainty.

### 2.5. Evaluation protocol

**Unseen farm.** Each of the eight farms was held out in turn (leave-one-farm-out) and the detector was trained on the other seven. The held-out farm entered no training decision: early stopping and checkpoint selection used an inner validation set of 10% of the training images, stratified by each image's dominant class and drawn with a fixed seed, and the held-out farm was scored once, after training ended. Widely used detection frameworks score the validation split every epoch and keep the best-scoring weights, so a held-out farm passed to the trainer as its validation set would choose the checkpoint (Varma and Simon, 2006; Cawley and Talbot, 2010). In an earlier run that did this, the YOLO11n gap appeared as 37.4% instead of 47.8%, although that run also trained on all of the training images, so the difference is not due to checkpoint selection alone.

**Random split.** For comparison, the same pool was split once at random into 80% training and 20% test images (165 images), stratified by dominant class, with the same inner-validation rule. This is the figure a conventional evaluation reports: one mAP50 computed over the pooled test images.

**Aggregation.** The unseen-farm score of a detector is the mAP50 of each held-out farm's images, averaged over seeds, and then averaged over farms with equal weight, because the unit of use is a grower's farm, not an image. Each farm's mAP50 averages only the classes present on that farm. Because the conventional random-split figure is pooled over farms and classes, the random-split models were also scored on the test images of each farm separately and averaged in the same way (farm-weighted random split).

**Another island.** Every leave-one-farm-out model of the five Ultralytics detectors was also scored on each of the two Sabah orchards, which never entered any training set.

### 2.6. Data-budget experiment

To separate the effect of how many farms supply the training data from the effect of how many images each farm supplies, YOLO11n was retrained under a factorial design. For each held-out farm, *k* ∈ {1, 2, 4, 7} training farms were drawn from the other seven, and *m* ∈ {15, 50, all} images were drawn at random from each drawn farm. For *k* < 7, two different farm combinations were drawn; for *k* = 7 there is only one. Within a combination the three image counts are nested (15 ⊂ 50 ⊂ all), so comparisons between them are paired. The design gives 168 training sets over eight held-out farms, each trained with two seeds (1 and 42).

Every run was trained for the same number of iterations, so that budgets differ only in their data; Section 4.2 and Table S7 show how this choice affects the results. The batch size was 32, and Ultralytics by default accumulates gradients over two batches, a nominal batch of 64. The list of training images was repeated until each epoch held at least 20 batches, and the number of epochs was set to give about 2,000 iterations (1,988–2,015 in practice, or about 1,000 optimiser steps). Mosaic augmentation was switched off for the last 10% of epochs. Because 10% of 15 images is too few for a meaningful stopping signal, no validation set, early stopping or checkpoint selection was used: the final weights were evaluated once on the held-out farm and on each Sabah orchard. The held-out farm therefore entered no part of training. An earlier run of this experiment capped training at 300 epochs, which gave the smallest budgets far fewer iterations than the largest; its results are reported for comparison in Table S7.

### 2.7. New-farm calibration experiment

Calibration here means adapting the detector to a new farm with labelled images from that farm. To test how much a few such images recover, each held-out farm's images were ordered by capture time and split into halves, with three images discarded at the boundary to make it unlikely that a burst of images of one tree falls on both sides; one half served as the calibration half and the other as the test half, and the roles were then swapped. From the calibration half, *n* ∈ {5, 10, 20, all} images were drawn at random (nested). The starting model was the data-budget model trained on all images of the other seven farms. Two ways of using the calibration images were compared, each with a fixed optimisation budget and no checkpoint selection. Fine-tuning trained the starting model on the calibration images alone, with the weights and batch-normalisation statistics of the first 10 modules frozen (AdamW, learning rate 5 × 10⁻⁴, no warm-up, batch 16 without gradient accumulation, about 300 optimiser steps, default augmentation; Note S1). Retraining started again from COCO weights on the other seven farms plus the calibration images, under the protocol of Section 2.6. Both were scored on the same test half as the uncalibrated starting model. Fine-tuning was run with two seeds (1 and 42) and retraining with one seed (42). Only this one fine-tuning recipe, fixed before the runs, was tested.

### 2.8. Statistical analysis

All summaries average within a held-out farm first (over draws, seeds and, for calibration, the two halves) and then over farms with equal weight. Intervals are 90% percentile bootstrap intervals over held-out farms (4,000 resamples). Eight farms is a small sample, and the training sets for different held-out farms share farms, so these intervals are approximate and probably too narrow. We use them to describe the spread across farms, not as tests. Paired comparisons are computed per held-out farm and then bootstrapped, and the number of farms on which a comparison favours one side is reported alongside; under a two-sided sign test, 6 of 8 corresponds to p ≈ 0.29, 7 of 8 to p ≈ 0.07 and 8 of 8 to p ≈ 0.008. Differences are computed from unrounded values.

To summarise the data-budget design, unseen-farm mAP50 was regressed on log₂ *k* and log₂ (images per farm) with a fixed effect for each held-out farm, fitted to all 336 runs; the two slopes are the expected change in mAP50 per doubling of the number of farms at fixed images per farm, and per doubling of images per farm at fixed farms. Because a log-linear form can hide uneven steps, the individual steps (one to two, two to four and four to seven farms; 15 to 50 and 50 to all images per farm) are also reported as paired contrasts (Table S6). For each training set we computed its class coverage: the share of the held-out farm's annotated images whose classes all occur in the images actually drawn for training. A detector cannot find a class it was never shown, so coverage indicates how much of the value of a farm lies in the classes it carries; it was added to the regression as a covariate and used to restrict it.

## 3. Results

### 3.1. The unseen-farm gap

On a random split the six detectors reached 0.366–0.499 mAP50; on farms they had never seen they reached 0.214–0.236 (Table 2); the random-split figure was 1.65–2.26 times the unseen-farm figure. The loss was 47.8–55.8% for the four YOLO11 sizes and RT-DETR-L and 39.3% for Faster R-CNN, whose random-split figure was the lowest of the six. YOLO11n, the deployed model, fell from 0.428 to 0.223. The gap does not come from averaging: scored farm by farm and averaged the same way as the unseen-farm figure, the random-split models still reached 0.492–0.527. Nor is the gap confined to some classes: every class lost AP on unseen farms (Section 3.3).

**Table 2.** mAP50 on a random 80/20 split and on farms never seen in training (leave-one-farm-out, farms weighted equally), and on the two Sabah orchards (261 and 20 images). The pooled random-split figure is computed over all test images; the farm-weighted figure scores each farm's test images separately and averages farms with equal weight, as for the unseen-farm figure. Loss compares the pooled random split with the unseen farm and is computed from unrounded values. Faster R-CNN was not scored on Sabah or per farm on the random split.

| Detector | Random split, pooled | Random split, farm-weighted | Unseen farm | Loss (%) | Sabah orchard 1 | Sabah orchard 2 |
|---|---|---|---|---|---|---|
| YOLO11n (deployed) | 0.428 | 0.492 | 0.223 | 47.8 | 0.264 | 0.315 |
| YOLO11s | 0.462 | 0.502 | 0.214 | 53.8 | 0.262 | 0.314 |
| YOLO11m | 0.465 | 0.499 | 0.233 | 49.9 | 0.267 | 0.350 |
| YOLO11l | 0.472 | 0.503 | 0.236 | 50.1 | 0.287 | 0.342 |
| RT-DETR-L | 0.499 | 0.527 | 0.221 | 55.8 | 0.271 | 0.340 |
| Faster R-CNN R50-FPN | 0.366 | – | 0.222 | 39.3 | – | – |

### 3.2. Farm difficulty outweighs detector choice

Scored one farm at a time and averaged over the six detectors, unseen-farm mAP50 ranged from 0.094 to 0.356, a 3.8-fold spread (Fig. 2a; Table S1). The detectors largely agreed on which farms were hard: the mean pairwise Spearman correlation of their farm rankings was 0.85 (lowest pair 0.62), and all six placed farms 2 and 6 last. Averaged over farms, no two detectors differed by more than 0.022 mAP50, and YOLO11n was 0.012 below the best (computed before rounding); on individual farms they differed by up to 0.149, but the best detector changed from farm to farm and none was best on more than three of the eight. We also split the variance of the per-farm score across five detectors, eight farms and five seeds. The held-out farm accounted for 82.4%, the farm × detector interaction for 7.8% and the residual, which includes seed-to-seed variation, for 9.8%. The estimate for the detector itself was negative and is reported as 0.0%.

![](../results_applied/fig2_new_farm_cost.png){width=16cm}

**Fig. 2.** The unseen-farm gap. (a) Unseen-farm mAP50 of each peninsular farm, ordered by difficulty; grey points are the six detectors, blue diamonds their mean, and the dashed line is the mean pooled random-split figure. (b) Per-class AP50 on a random split (open circles) and on unseen farms (filled), averaged over the five detectors that report per-class AP; the percentage is the share retained. For each class, unseen-farm AP is averaged over the farms that carry it.

### 3.3. Psyllid classes transfer worst

Compared class by class, every class lost AP on unseen farms, but not equally (Fig. 2b, Table 3). Psyllid and psyllid damage retained 30% and 25% of their random-split AP, against 48–59% for leaf rot, algal spot, *Phomopsis* and leafhopper damage. The per-class figure on a single farm is noisy when that farm holds few examples of the class, so the calculation was repeated using only farms with at least 10 images of the class: psyllid and psyllid damage then retained 37% and 29%, and the other classes 55–75%. The two psyllid classes remained the lowest, although the order of the other four changed (Table S2 gives every class on every farm).

**Table 3.** Per-class AP50 on a random split and on unseen farms, averaged over the five detectors that report per-class AP. Unseen-farm AP is averaged over the farms that carry the class; the range is over those farms. Retained is the mean over detectors of each detector's unseen-farm/random-split ratio, so it can differ slightly from the ratio of the column means; the last column repeats the calculation using only farms with at least 10 images of the class.

| Class | Farms carrying it | Random split | Unseen farm (range over farms) | Retained (%) | Retained, farms with ≥ 10 images (%) |
|---|---|---|---|---|---|
| Algal spot | 7 | 0.504 | 0.272 (0.000–0.646) | 54 | 75 (5 farms) |
| Leaf rot | 8 | 0.488 | 0.236 (0.000–0.461) | 48 | 55 (6 farms) |
| *Phomopsis* | 3 | 0.460 | 0.266 (0.205–0.366) | 58 | 58 (3 farms) |
| Psyllid | 7 | 0.410 | 0.125 (0.032–0.266) | 30 | 37 (5 farms) |
| Psyllid damage | 6 | 0.337 | 0.087 (0.018–0.286) | 25 | 29 (5 farms) |
| Leafhopper damage | 8 | 0.592 | 0.349 (0.137–0.689) | 59 | 61 (4 farms) |

### 3.4. More farms or more images per farm

With the number of iterations held constant, unseen-farm mAP50 rose with both the number of training farms and the number of images per farm (Fig. 3a, Table 4; each held-out farm in Table S4). One farm with all of its images gave 0.078; seven farms with all of theirs gave 0.228.

Fitted over all 336 runs, a doubling of farms changed mAP50 by 0.038 (90% interval 0.023 to 0.053) and a doubling of images per farm by 0.028 (0.021 to 0.038). The farm slope was larger by 0.010, but the 90% interval of the difference (−0.003 to 0.025) includes zero. That difference came from the seven-farm training sets (Table 5B). Leaving them out reduced it to 0.001 (−0.012 to 0.015); leaving out the one-farm runs instead made it 0.013 (0.002 to 0.024).

**Table 4.** Data-budget experiment (YOLO11n, about 2,000 iterations per run). Each row averages two farm draws (one for seven farms) and the seeds within each held-out farm, and then the eight held-out farms with equal weight. Class coverage is the share of the held-out farm's annotated images whose classes all occur in the images drawn for training. Sabah is the mean of the two orchards' scores.

| Training farms | Images per farm | Training images | Unseen-farm mAP50 [90% CI] | Worst–best farm | Class coverage | Sabah mAP50 |
|---|---|---|---|---|---|---|
| 1 | 15 | 15 | 0.028 [0.017, 0.042] | 0.001–0.074 | 0.54 | 0.040 |
| 1 | 50 | 50 | 0.046 [0.030, 0.062] | 0.008–0.088 | 0.66 | 0.065 |
| 1 | all | 106 | 0.078 [0.056, 0.101] | 0.029–0.145 | 0.69 | 0.078 |
| 2 | 15 | 30 | 0.052 [0.034, 0.071] | 0.014–0.113 | 0.83 | 0.073 |
| 2 | 50 | 100 | 0.083 [0.059, 0.105] | 0.019–0.140 | 0.84 | 0.118 |
| 2 | all | 218 | 0.135 [0.097, 0.176] | 0.037–0.272 | 0.84 | 0.161 |
| 4 | 15 | 60 | 0.070 [0.051, 0.088] | 0.016–0.124 | 0.98 | 0.115 |
| 4 | 50 | 200 | 0.103 [0.075, 0.128] | 0.030–0.168 | 1.00 | 0.169 |
| 4 | all | 392 | 0.146 [0.105, 0.185] | 0.038–0.252 | 1.00 | 0.236 |
| 7 | 15 | 105 | 0.127 [0.080, 0.176] | 0.010–0.285 | 1.00 | 0.166 |
| 7 | 50 | 350 | 0.182 [0.137, 0.222] | 0.022–0.245 | 1.00 | 0.235 |
| 7 | all | 724 | 0.228 [0.171, 0.289] | 0.076–0.435 | 1.00 | 0.302 |

Every step in images per farm raised mAP50, at every number of farms (Table S6). The steps in the number of farms were uneven. With all images per farm, going from one to two farms changed mAP50 by 0.058, from two to four by 0.011 (−0.007 to 0.030) and from four to seven by 0.082. With 50 images per farm the three steps were 0.036, 0.020 and 0.080. The step to seven farms is confounded. Seven farms is a single combination that always includes every other farm, among them those most similar to the held-out one, whereas the two- and four-farm sets are two draws each. The gain at seven farms may therefore reflect which farms were included as much as how many.

At matched image budgets (Fig. 3b, Table 5A), neither way of spending the budget was consistently better. Seven farms × 15 images scored 0.127 against 0.083 for two farms × 50 (difference 0.044, 0.014 to 0.075; more farms better on 6/8 held-out farms). Seven farms × 50 scored 0.182 against 0.146 for four farms × all (0.036, 0.009 to 0.061; 6/8). In the middle of the range the order reversed: four farms × 50 scored 0.103 against 0.135 for two farms × all (−0.033, −0.066 to −0.005; 1/8). Both comparisons won by more farms use the seven-farm training sets.

![](../results_applied/fig3_budget.png){width=16cm}

**Fig. 3.** How the training budget is spent (YOLO11n, fixed number of iterations). (a) Unseen-farm mAP50 against the number of training images, for training sets drawn from 1, 2, 4 or 7 farms with 15, 50 or all images per farm; vertical bars are 90% bootstrap intervals over held-out farms. (b) Paired differences at three matched image budgets; grey points are held-out farms, blue diamonds and bars the mean and its 90% interval.

**Table 5.** (A) Matched-budget comparisons, paired by held-out farm. (B) Change in unseen-farm mAP50 per doubling of training farms and of images per farm, from a regression with held-out-farm fixed effects; intervals resample held-out farms.

*(A) Matched budgets*

| More farms | Fewer farms | Images | mAP50 | Difference [90% CI] | Held-out farms where more farms won |
|---|---|---|---|---|---|
| 7 farms × 15 | 2 farms × 50 | 105 / 100 | 0.127 / 0.083 | 0.044 [0.014, 0.075] | 6 of 8 |
| 4 farms × 50 | 2 farms × all | 200 / 218 | 0.103 / 0.135 | −0.033 [−0.066, −0.005] | 1 of 8 |
| 7 farms × 50 | 4 farms × all | 350 / 392 | 0.182 / 0.146 | 0.036 [0.009, 0.061] | 6 of 8 |

*(B) Regression slopes*

| Runs included | n | Doubling farms [90% CI] | Doubling images per farm [90% CI] | Difference [90% CI] |
|---|---|---|---|---|
| all runs | 336 | 0.038 [0.023, 0.053] | 0.028 [0.021, 0.038] | 0.010 [−0.003, 0.025] |
| full class coverage only | 252 | 0.035 [0.020, 0.052] | 0.030 [0.022, 0.041] | 0.005 [−0.009, 0.021] |
| excluding k = 7 | 288 | 0.028 [0.014, 0.042] | 0.027 [0.019, 0.036] | 0.001 [−0.012, 0.015] |
| excluding k = 1 | 240 | 0.043 [0.030, 0.056] | 0.030 [0.022, 0.040] | 0.013 [0.002, 0.024] |
| adjusted for class coverage | 336 | 0.039 [0.021, 0.057] | 0.028 [0.021, 0.038] | 0.011 [−0.005, 0.028] |

Part of the value of a farm is the classes it carries. Training sets drawn from one farm covered, on average, 63% of the held-out farm's annotated images, and those from two farms 84%; almost every set drawn from four or more farms covered all of them. Across the one-farm and the two-farm training sets, where coverage varies, it was correlated with unseen-farm mAP50 (Spearman ρ = 0.62 and 0.64). Coverage does not account for the farm slope, however. With coverage added to the regression the slope for a doubling of farms was 0.039 (0.021 to 0.057), and in the 252 runs whose training images covered every class on the held-out farm it was 0.035, against 0.030 for images per farm (Table 5B).

### 3.5. Which classes gain from more farms

With all images per farm, every class scored higher on the unseen farm with seven training farms than with one, although not every intermediate step was an improvement (Fig. 4; Table S3). Psyllid damage was the lowest class at every number of farms, and psyllid the second lowest from two farms on. With seven farms they reached 0.058 and 0.108, against 0.217–0.388 for the other four classes.

![](../results_applied/fig4_class_by_k.png){width=12cm}

**Fig. 4.** Unseen-farm AP50 of each class against the number of training farms, with all images per farm (YOLO11n), averaged over the held-out farms that carry the class. The seven-farm point is a single farm combination per held-out farm; the others average two.

### 3.6. Calibrating to a new farm

Without calibration, the detector trained on the other seven farms scored 0.240 on the test halves (0.247 for seed 42, the seed used for retraining; Table 6, Fig. 5). Fine-tuning it on images from the calibration half lowered mAP50 by 0.111 with 5 images, 0.080 with 10 and 0.055 with 20. With the whole half, about 50 images, the loss was 0.029, and its interval (−0.056 to 0.003) includes zero; fine-tuning beat no calibration on 2/8 farms. The fine-tuned detector became more conservative. With 5 images its precision rose by 0.086 and its recall fell by 0.141, and most of the loss fell on leafhopper damage and leaf rot (−0.350 and −0.179 AP50).

Retraining with the same images added to the other seven farms did not lower accuracy on average. It changed mAP50 by +0.004, +0.004, +0.030 (−0.001 to 0.062) and +0.053 (0.014 to 0.100; better on 6/8 farms), and with the whole half it raised both precision and recall. The changes with 5 and 10 images are smaller than the seed-to-seed variation of a single farm's score; only with the whole half did the interval exclude zero.

![](../results_applied/fig5_calibration.png){width=12cm}

**Fig. 5.** Change in mAP50 on the test half of a new farm when a detector trained on the other seven farms is calibrated with images from the calibration half, by fine-tuning (blue) or by retraining with them added (orange); bands are 90% bootstrap intervals over farms.

**Table 6.** New-farm calibration (YOLO11n). Each farm's images are split by capture time into a calibration half and a test half, in both directions; changes are paired with the uncalibrated model on the same test half and averaged over directions and seeds within a farm, then over farms. Fine-tuning changes are paired with the uncalibrated model of the same seed (both seeds); retraining, run with one seed, is paired with that seed's uncalibrated model, shown in the second row.

| Calibration | Images from the new farm | mAP50 on the test half | Change [90% CI] | Farms improved |
|---|---|---|---|---|
| None | 0 | 0.240 | – | – |
| None, retraining seed only | 0 | 0.247 | – | – |
| Fine-tune | 5 | 0.129 | −0.111 [−0.157, −0.062] | 1 of 8 |
| Fine-tune | 10 | 0.160 | −0.080 [−0.121, −0.036] | 1 of 8 |
| Fine-tune | 20 | 0.185 | −0.055 [−0.088, −0.017] | 1 of 8 |
| Fine-tune | all (27–72; mean 50) | 0.211 | −0.029 [−0.056, 0.003] | 2 of 8 |
| Retrain with images added | 5 | 0.251 | 0.004 [−0.017, 0.022] | 6 of 8 |
| Retrain with images added | 10 | 0.251 | 0.004 [−0.025, 0.037] | 3 of 8 |
| Retrain with images added | 20 | 0.277 | 0.030 [−0.001, 0.062] | 5 of 8 |
| Retrain with images added | all (27–72; mean 50) | 0.300 | 0.053 [0.014, 0.100] | 6 of 8 |

### 3.7. Another island

The five Ultralytics detectors trained on seven peninsular farms scored 0.262–0.287 on the larger Sabah orchard (261 images) and 0.314–0.350 on the smaller one (20 images), against 0.214–0.236 on held-out peninsular farms (Table 2). Two orchards cannot measure transfer between regions, but on these two the peninsular detectors scored no lower than on held-out peninsular farms; why they scored higher was not established. Where the two orchards are averaged (Table 4), they are weighted equally.

## 4. Discussion

### 4.1. What an orchard detector should report

For every detector tested, a random split of images from the training farms overstated unseen-farm accuracy by a factor of 1.6 to 2.3. The factor is a property of the corpus: it depends on how much farms differ and on how many near-duplicate images a random split places on both sides. In this corpus most farms were photographed on a single visit, so the gap combines a change of farm with a change of day, light and season; whichever dominates, a grower's farm differs from the training data in both respects. In practice this means holding out whole farms, keeping the held-out farm out of early stopping and checkpoint selection, and reporting the spread across farms alongside the mean. With a 3.8-fold range between the easiest and hardest farm, the mean alone says little about what a detector will do on a particular farm.

The held-out farm mattered more than the detector. Six detectors from three architecture families (four of them sizes of one family) differed on average by at most 0.022 mAP50 across the same farms, far less than the farms differed from one another. Eight farms cannot show that the detectors are equivalent, but the differences are small enough that latency, energy and memory can decide the choice for a device.

### 4.2. How to spend a collection budget

Under the fixed-iteration schedule, more farms and more images per farm helped by amounts the main analysis cannot separate: a doubling of each raised unseen-farm mAP50 by 0.038 and 0.028. The small advantage of farms rests on the seven-farm training sets. Without them the difference fell to 0.001; the step from two to four farms was small with all images per farm (0.011, against 0.020 with 50); and both matched-budget comparisons that more farms won involved seven farms. Seven farms is also the one design point where a single combination always contains the farms most similar to the held-out one.

The comparison also depends on the training schedule. An earlier version of the experiment capped training at 300 epochs, which gave the smallest budgets far fewer iterations. For the same seed its farm slope was the same as with fixed iterations (0.039), but its slope for images per farm was 0.026 rather than 0.030, so the difference, 0.013 (0.001 to 0.027), would have favoured farms, where the fixed schedule gave 0.009 (−0.003 to 0.023; Table S7). The capped schedule also scored higher in 11 of the 12 budget cells, including cells in which both schedules ran about the same number of iterations, so the two differ in more than training length.

A budget counted in images also hides the cost that matters in the field. Most farms here were photographed in one visit, so seven farms × 15 images meant about seven field days and two farms × 50 about two. A new farm can add classes that more images from known farms cannot, and a training set that lacks a class present on the target farm cannot detect it; *Phomopsis*, for example, was photographed on only three of eight farms. Beery et al. (2018) found camera-trap accuracy at new locations stable beyond two training locations; here the step from two to four farms was also small, at least with all images per farm, and the rise at seven cannot be separated from which farms were included. Ruigrok et al. (2023) found for weed detection that more sub-datasets improved generalisation at a constant image count; the durian results agree only in part, since more farms did not win at every budget. A first collection campaign should therefore visit enough farms to cover the region's diseases and pests. Beyond that point, the data do not show whether further images from farms already visited are worth more or less than images from new ones.

### 4.3. Why the psyllid classes transfer worst

The two psyllid classes kept roughly a quarter to a third of their random-split AP on unseen farms and stayed among the lowest classes with every number of training farms. Size alone does not explain this. Psyllid and psyllid-damage boxes are small (an average side of about 6 and 8 pixels at 640 × 640), but *Phomopsis* lesions are smaller still (about 5 pixels) and kept 58% of their AP. Three explanations are possible, and these data cannot separate them. The two psyllid classes occur together and look alike, so the line between them is partly a matter of annotation convention. An infestation also looks different at different stages, and the farms were visited at different times. And insect detection is hard even within one domain (Wu et al., 2019; Li et al., 2021a). On a farm outside its training set, a device's psyllid suggestions deserve less weight than its other suggestions. The psyllid classes are also where a revised class definition, a close-up capture protocol or more labelled data would be worth testing first.

### 4.4. Calibrating to a new farm

A device that meets a new farm could be adapted with a few images taken on arrival. Fine-tuning on those images alone made the detector worse: with 5 images mAP50 fell by 0.111, and with about 50 it was still 0.029 below the starting point of 0.240, although its 90% interval (−0.056 to 0.003) includes zero. The fine-tuned detector found fewer lesions but was more often right about those it found, and the loss was concentrated in some classes. This is expected when a detector is tuned on a few images that contain only some classes, with few examples of each: it learns to suppress the classes they lack. Ruigrok et al. (2023) found fine-tuning on as few as 25 images helpful for weed detection, with a different task and training setup. Only one fine-tuning recipe was tested here (Section 2.7); a lower learning rate, fewer steps or mixing the original training images into fine-tuning might avoid the loss.

Adding the same images to the training set and retraining avoided the loss, but it helped measurably only with the whole calibration half, about 50 images, when it raised mAP50 by 0.053. Retraining also costs a full training run and would be done off the device, after the images are sent back. With the fine-tuning recipe tested here, a new farm's labelled images were better used as additional training data, and a few tens of them were needed, not a handful. The calibration images were labelled by the author, so these results show what calibration can achieve when a grower's images are labelled correctly; in practice that labelling is the cost.

### 4.5. Limitations

The corpus has eight peninsular farms and two Sabah orchards, all photographed by one person, mostly on one visit per farm, so farm and visit cannot be separated, and a grower photographing with another phone may see a larger gap. The class composition differs sharply between farms: five held-out farms carry no *Phomopsis*, so its per-class figure rests on three. Inter-annotator agreement was not measured, and any bias in the annotation standard could interact with farm composition. Per-class figures on individual farms are noisy where a class has few examples; the two psyllid classes stayed lowest when such farms were excluded, but the other classes moved. The experiments on training data used one detector, two farm draws per budget (which differed by a median of 0.032 mAP50 on the same held-out farm, pooled over budgets; Table S5) and two seeds per configuration; the seed-to-seed standard deviation of a leave-one-farm-out YOLO11n score on a single farm averaged 0.026 mAP50, comparable to several of the paired differences, so per-farm win counts should be read with that in mind. The seven-farm training sets are a single combination per held-out farm, and the fixed number of iterations is one of several reasonable schedules (Table S7). The random split is a single partition, so its split-to-split variation is unknown. The calibration experiment tested one fine-tuning recipe. The bootstrap intervals resample eight farms that share training data and are approximate. All results are offline evaluations of images; the device's latency and energy, and how growers act on its suggestions, were not measured here.

## 5. Conclusions

For six detectors, a random image split reported 1.6–2.3 times the mAP50 reached on durian farms the detector had never seen. The held-out farm determined far more of the score than the detector did, and the psyllid classes transferred worst. Under the fixed-iteration schedule, more farms and more images per farm both improved unseen-farm accuracy, by amounts that could not be distinguished, and part of the value of a farm lay in the classes it carried. With the recipe tested, fine-tuning on a new farm's images alone did not help and, with 5–20 images, lowered accuracy; adding about 50 of them to the training set and retraining raised mAP50 by 0.053. Orchard detectors should be evaluated on held-out farms with the held-out farm excluded from every training decision, and collection campaigns should be planned around the farms and classes of the target region as well as the number of images.

## Supplementary material

Supplementary material (Tables S1–S7, Notes S1–S2) is available online.

## CRediT authorship contribution statement

**Lin Ding Shan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization, Project administration.

## Declaration of competing interest

The author is developing a handheld decision-support device for durian disease and pest scouting and has a potential financial interest in the domain this study evaluates. Access to the Sabah orchards was facilitated by a commercial partner who had no role in study design, data collection, annotation, analysis or the decision to publish, and provided no funding, products or data.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Ethics statement

The study is principally computational analysis of plant images. Two adult growers were consulted about their professional practice; informed consent was obtained verbally and recorded at the time, and both have since confirmed in writing. No names, farm names, locations or other identifying information appear.

## Data availability

The durian images and annotations are deposited on Zenodo (https://doi.org/10.5281/zenodo.22030623; CC BY-NC 4.0) at 640 × 640 pixels with location coordinates removed. The content-based farm attribution used in this paper (Note S2) and all split manifests are in the code repository (results_durian/split_assignment.csv) and supersede any farm labels in the deposit; they identify the 827-image analysis pool. The 281 Sabah images are the deposit's Sabah folder, with the orchard given by the filename prefix (o1 or o2).

## Code availability

Training, evaluation and analysis code, the result tables behind every figure and table, and a script that recomputes every number in this paper from those tables are available at https://github.com/xiaolin200206/durian-site-split.

## Relation to prior work by the same author

The peninsular images derive from the same collection as an earlier study of capture-session leakage in five-class image classification (under review; https://doi.org/10.5281/zenodo.22177133), which partitions 560 images into 73 capture sessions and reports macro F1. The present work uses box-level annotation, content-based farm attribution, 827 images across eight farms, detection models, a different unit and evaluation protocol, and experiments on training-data budgets and new-farm calibration that the earlier study does not contain. The leave-one-farm-out runs of Sections 3.1–3.3 were also analysed, together with three non-agricultural datasets, in a methodological manuscript that was withdrawn before review; the present paper replaces it for the durian results.

## Acknowledgements

The author thanks the collaborating grower and crop-protection practitioner in Johor, who agreed the annotation standard in the field, and the orchard manager in Lahad Datu for access to the Sabah orchards.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Claude (Anthropic) to draft and audit analysis scripts operating on the released result tables, to edit and restructure the manuscript, and to locate candidate references. All fieldwork, image capture, annotation review, class definition, protocol design and model training were carried out by the author, who reviewed every script and its output and re-ran each on the released data. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

Barbedo, J.G.A., 2018a. Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. Comput. Electron. Agric. 153, 46–53. https://doi.org/10.1016/j.compag.2018.08.013

Barbedo, J.G.A., 2018b. Factors influencing the use of deep learning for plant disease recognition. Biosyst. Eng. 172, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013

Beery, S., Van Horn, G., Perona, P., 2018. Recognition in terra incognita, in: Computer Vision – ECCV 2018, Lecture Notes in Computer Science 11220. Springer, Cham, pp. 472–489. https://doi.org/10.1007/978-3-030-01270-0_28

Cawley, G.C., Talbot, N.L.C., 2010. On over-fitting in model selection and subsequent selection bias in performance evaluation. J. Mach. Learn. Res. 11, 2079–2107.

Jocher, G., Qiu, J., 2024. Ultralytics YOLO11, version 11.0.0 [software]. https://github.com/ultralytics/ultralytics

Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4, 100804. https://doi.org/10.1016/j.patter.2023.100804

Koh, P.W., Sagawa, S., Marklund, H., et al., 2021. WILDS: a benchmark of in-the-wild distribution shifts, in: Proceedings of the 38th International Conference on Machine Learning, PMLR 139, pp. 5637–5664.

Li, W., Wang, D., Li, M., Gao, Y., Wu, J., Yang, X., 2021a. Field detection of tiny pests from sticky trap images using deep learning in agricultural greenhouse. Comput. Electron. Agric. 183, 106048. https://doi.org/10.1016/j.compag.2021.106048

Li, Y., Xie, S., Chen, X., Dollár, P., He, K., Girshick, R., 2021b. Benchmarking detection transfer learning with vision transformers. arXiv:2111.11429. https://doi.org/10.48550/arXiv.2111.11429

Lin, T.-Y., Maire, M., Belongie, S., et al., 2014. Microsoft COCO: common objects in context, in: Computer Vision – ECCV 2014, Lecture Notes in Computer Science 8693. Springer, Cham, pp. 740–755. https://doi.org/10.1007/978-3-319-10602-1_48

Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning for image-based plant disease detection. Front. Plant Sci. 7, 1419. https://doi.org/10.3389/fpls.2016.01419

Nguyen, T., 2025. Image dataset of ten durian diseases captured in real-field conditions from a family orchard in Vinh Long, Vietnam. Data Brief 63, 112244. https://doi.org/10.1016/j.dib.2025.112244

Nguyen Thanh, T., Nguyen, L.X., Cap, T., Le, T., 2025. A durian leaf image dataset of common diseases in Vietnam for agricultural diagnosis. Data Brief 61, 111845. https://doi.org/10.1016/j.dib.2025.111845

Norman, D.L., Bischoff, P.H., Wearn, O.R., et al., 2023. Can CNN-based species classification generalise across variation in habitat within a camera trap survey? Methods Ecol. Evol. 14, 242–251. https://doi.org/10.1111/2041-210X.14031

O'Gara, E., Guest, D.I., Hassan, N.M., 2004. Botany and production of durian (*Durio zibethinus*) in Southeast Asia, in: Drenth, A., Guest, D.I. (Eds.), Diversity and Management of *Phytophthora* in Southeast Asia, ACIAR Monograph No. 114. Australian Centre for International Agricultural Research, Canberra, pp. 180–186.

Ren, S., He, K., Girshick, R., Sun, J., 2017. Faster R-CNN: towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell. 39, 1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031

Roberts, D.R., Bahn, V., Ciuti, S., et al., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40, 913–929. https://doi.org/10.1111/ecog.02881

Ruigrok, T., van Henten, E.J., Kootstra, G., 2023. Improved generalization of a plant-detection model for precision weed control. Comput. Electron. Agric. 204, 107554. https://doi.org/10.1016/j.compag.2022.107554

Schneider, S., Greenberg, S., Taylor, G.W., Kremer, S.C., 2020. Three critical factors affecting automated image species recognition performance for camera traps. Ecol. Evol. 10, 3503–3517. https://doi.org/10.1002/ece3.6147

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N., 2020. PlantDoc: a dataset for visual plant disease detection, in: Proceedings of the 7th ACM IKDD CoDS and 25th COMAD. ACM, New York, pp. 249–253. https://doi.org/10.1145/3371158.3371196

Tabak, M.A., Norouzzadeh, M.S., Wolfson, D.W., et al., 2019. Machine learning to classify animal species in camera trap images: applications in ecology. Methods Ecol. Evol. 10, 585–590. https://doi.org/10.1111/2041-210X.13120

Tang, R., Jun, T., Chu, Q., Sun, W., Sun, Y., 2025. Small object detection in agriculture: a case study on durian orchards using EN-YOLO and thermal fusion. Plants 14, 2619. https://doi.org/10.3390/plants14172619

Varma, S., Simon, R., 2006. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7, 91. https://doi.org/10.1186/1471-2105-7-91

Wu, X., Zhan, C., Lai, Y.-K., Cheng, M.-M., Yang, J., 2019. IP102: a large-scale benchmark dataset for insect pest recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8787–8796. https://doi.org/10.1109/CVPR.2019.00899

Xu, M., Park, J.-E., Lee, J., Yang, J., Yoon, S., 2024. Plant disease recognition datasets in the age of deep learning: challenges and opportunities. Front. Plant Sci. 15, 1452551. https://doi.org/10.3389/fpls.2024.1452551

Zhao, Y., Lv, W., Xu, S., et al., 2024. DETRs beat YOLOs on real-time object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16965–16974. https://doi.org/10.1109/CVPR52733.2024.01605
