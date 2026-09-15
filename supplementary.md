# Supplementary Information

**Which units were withheld determines much of a reported machine-learning score**

Lin Ding Shan

Supplementary Tables 1–17, Supplementary Figure 1 and Supplementary Notes 1–4.

All figures in this document come from the clean protocol, in which the withheld units enter no training decision, unless a row is explicitly labelled as coming from the contaminated protocol. Every number is recomputed from `results_clean/summary_*.csv` by `verify_claims.py`.

---

## Supplementary Table 1 | Unit identifiers in public crop-disease datasets

Fourteen public datasets were surveyed for whether they release an identifier at a granularity that permits grouping — that is, an identifier that distinguishes one acquisition bout, plant or site from another.

| Dataset | Images | Unit identifier released | Grouping possible |
|---|---|---|---|
| PlantVillage [13] | 54,306 | no | no |
| PlantDoc [14] | 2,598 | no | no |
| FieldPlant [15] | 5,170 | plantation (3 sites) | coarse |
| Durian leaf disease, Vietnam [16] | 1,500 | no | no |
| Ten durian diseases, Vinh Long [25] | 4,145 | single orchard | no |
| GWHD 2021 [4] | 6,515 | acquisition session (47) | **yes** |
| Nine further crop-disease sets | — | no | no |

One of fourteen releases an identifier fine enough to support a unit-disjoint split. The survey is not exhaustive of all plant-pathology imagery; it covers the datasets a practitioner is most likely to reach for.

The point is not that authors were careless. Recording provenance is necessary but not sufficient: we recorded location and time for every original in our own corpus, joined those records to the annotated images by filename, and put 8.9% of the pool under the wrong farm (Supplementary Note 1).

---

## Supplementary Table 2 | Item-level against unit-level partitioning, clean protocol

Each row is one dataset–model combination. Item-level is a stratified random partition ignoring unit identity; unit-level is leave-one-farm-out on durian and unit-disjoint five-fold elsewhere. Both arms use the same inner-validation rule, so the difference between them is not confounded by the protocol.

| Dataset | Model | Metric | Item-level | Unit-level | Overstatement (%) | Seeds |
|---|---|---|---|---|---|---|
| durian | frcnn-r50 | mAP50 | 0.3656 | 0.222 | 39.29 | 3.0 |
| durian | rtdetr-l | mAP50 | 0.4987 | 0.2205 | 55.78 | 5.0 |
| durian | yolo11l | mAP50 | 0.4721 | 0.2356 | 50.08 | 5.0 |
| durian | yolo11m | mAP50 | 0.4652 | 0.2332 | 49.87 | 5.0 |
| durian | yolo11n | mAP50 | 0.4281 | 0.2234 | 47.81 | 5.0 |
| durian | yolo11s | mAP50 | 0.4622 | 0.2137 | 53.77 | 5.0 |
| gwhd | yolo11n | mAP50 | 0.6357 | 0.4926 | 22.51 | 3.0 |
| gwhd | yolo11s | mAP50 | 0.6723 | 0.4765 | 29.12 | 3.0 |
| breakhis | yolo11n-cls | top1 | 0.9527 | 0.8759 | 8.06 | 5.0 |
| breakhis | yolo11s-cls | top1 | 0.9504 | 0.8627 | 9.23 | 5.0 |
| har | mlp256 | macro_f1 | 0.9885 | 0.9519 | 3.7 | 5.0 |
| har | rf | macro_f1 | 0.979 | 0.9347 | 4.52 | 5.0 |

**Three patterns.** Within a dataset the overstatement is similar across capacities of one architecture family: the five one-stage durian detectors span 47.8–55.8%, the two histopathology classifiers 8.1–9.2%. It is not similar across families: Faster R-CNN overstates by 39.3%, 8 to 16 points below every one-stage detector on the same folds. Its unit-level figure is in the middle of the pack (0.222 against 0.214–0.236) while its item-level figure is the lowest of the six (0.366 against 0.428–0.499). The quantity being measured is how much a model exploits the near-duplication inside a cluster, and a two-stage anchor-based detector exploits it least.

Across datasets the overstatement spans an order of magnitude and follows the coefficient of variation across units (0.39, 0.30, 0.25, 0.06) rather than their standard deviation.

Near the ceiling the percentage understates the effect. On histopathology the error rate rises from 4.73% to 12.41% and on activity recognition 4.2-fold.

**Dispersion.** On the two datasets where both regimes are five-fold, the standard deviation across folds is 0.0024 against 0.0958 on wheat (40-fold) and 0.0041 against 0.0308 on histopathology (7.6-fold). On durian the item-level regime is a single 80/20 split: it has no across-fold dispersion at all, which is the sharper version of the same point.

---

## Supplementary Table 3 | Burst thresholds on durian

Capture bursts are consecutive frames from one farm less than a set gap apart. The per-burst analysis in the earlier protocol used a 60 s gap and a five-image minimum; the coefficient of variation across bursts stays between 0.70 and 0.75 across four settings while coverage runs from 42% to 79% of the pool. The clean-protocol analysis reports per-farm rather than per-burst scores, because a fold is a farm and the burst resolution adds no independent units. The burst table is retained because it bounds the sensitivity of the dispersion estimate to the unit definition.

---

## Supplementary Table 4 | Lens and annotated lesion scale

A 2.22 mm ultra-wide lens appears at every one of the eight farms, in 2% to 65% of a farm's images. Within a single farm it enlarges annotated Leaf_rot box area 1.8- to 4.0-fold relative to the 6.76 mm stratum. The unstratified ratio of median lesion size between the extreme farms is 75; restricted to one focal length it is 15. The check that failed to catch this was written against per-farm medians, which agreed while the claim was false; the replacement check asserts an image-level fact — that the wide lens appears at more than one farm.

---

## Supplementary Table 5 | Variance components

On durian a fold is one farm, so a two-factor crossed random-effects model over models × farms × seeds gives a per-unit component directly; the five models with five seeds each are used so the design is balanced, and Faster R-CNN (three seeds) is excluded from this table. On the other three datasets a fold contains several units, so the crossed design would conflate which units were withheld from training with which of them is scored; there the per-unit scores are decomposed hierarchically with units nested within folds.

| dataset | design | models | unit | model | interaction | seed | fold | residual |
|---|---|---|---|---|---|---|---|---|
| durian | crossed, fold = one farm | rtdetr-l,yolo11l,yolo11m,yolo11n,yolo11s | 82.4 | 0.0 | 7.8 | 9.8 |  |  |
| gwhd | nested, unit within fold | yolo11n,yolo11s | 68.0 | 0.3 |  | 0.1 | 9.6 | 22.1 |
| breakhis | nested, unit within fold | yolo11n-cls,yolo11s-cls | 88.1 | 0.0 |  | 0.0 | 1.9 | 9.9 |
| har | nested, unit within fold | mlp256,rf | 73.3 | 2.7 |  | 0.0 | 13.3 | 10.6 |

Values are percentages of the total. The unit component is 68.0–88.1% and the model component 0.0–2.7% on every dataset.

**Controls.** Restricting the durian decomposition to two models rather than five moves the farm share from 82.4% to 86.4% and the model share from 0.0% to 4.4%, so a larger roster of similar models lowers the model share rather than raising the farm share. Repeating it on a single class, which removes the class-composition differences between farms, gives farm shares of 84.9% and 97.9% with model shares of 0.0% and 0.2%. Within each histopathology label the unit component is 69.5% among 24 benign and 82.4% among 57 malignant patients, so the histopathology result is not produced by which classes a fold happens to carry.

**What the model term means.** Both factors are treated as random because the question is how much variability each contributes rather than which model is better. Reading the model term as a variance component assumes its levels sample a population of models, which two to five architectures do only loosely. The decomposition is descriptive of the roster tested. That roster spans one detection transformer, four capacities of one one-stage CNN family, a two-stage anchor-based detector, a perceptron and a random forest; within it the model term never exceeds 2.7%.

---

## Supplementary Table 6 | Three evaluation protocols

`old_best`: the withheld unit is the training loop's validation set and selects the checkpoint. `old_last`: the final-epoch weights of the same runs, which removes best-epoch selection but not early stopping. `clean`: the inner validation set is drawn from training data only and the withheld units enter no training decision.

| dataset | model | old_best | old_last | clean |
|---|---|---|---|---|
| durian | rtdetr-l | 41.1 | 58.2 | 55.8 |
| durian | yolo11m | 42.7 | 55.0 | 49.9 |
| durian | yolo11n | 37.4 | 51.6 | 47.8 |
| durian | yolo11s | 44.3 |  | 53.8 |
| gwhd | yolo11n | 19.6 | 28.5 | 22.5 |
| gwhd | yolo11s | 22.4 | 30.1 | 29.1 |
| breakhis | yolo11n-cls | 8.0 | 10.4 | 8.1 |
| breakhis | yolo11s-cls | 8.7 | 11.6 | 9.2 |

Values are the overstatement of the item-level protocol, in per cent.

**The clean value falls between the other two in all seven combinations where both contaminated values exist** (durian YOLO11s retained no final-epoch weights). That ordering is what the earlier measurement of checkpoint bias predicted, and it is the reason the contaminated results are retained rather than discarded: they bracket the clean value from both sides.

Moving from `old_best` to `clean` raises the measured overstatement by 0.1 to 14.7 points, 6.5 on average. The two histopathology rows move least (0.1 and 0.5) and the durian rows most (7.2 to 14.7), in the same order as the size of the partitioning effect itself.

**Two further signatures.** Under the clean protocol the choice of epoch almost stops mattering: withholding best-epoch selection moves the unit-level figure by −1.1 to +1.6 points, against 2.5 to 9.5 under the contaminated protocol. And the worst units move most between protocols: the lowest-scoring wheat session is 0.190 under the contaminated protocol and 0.071 under the clean one, because selecting the best epoch on a unit rescues precisely the unit the model handles worst.

**Epoch counts.** Of 135 durian runs under the contaminated protocol, 109 stopped early and 26 reached the epoch cap; the latter carry no early-stopping selection at all. In the fourteen of those that are unit-level the selected checkpoint still scored 0.051 above the final epoch, against 0.007 in the twelve item-level ones, so the asymmetry survived where best-epoch selection was the only selection left. Those runs are superseded by the clean protocol but are reported here because they were the evidence that motivated it. The runs that reach the cap are not a random subset — they are the ones that never went 50 epochs without improving — so that figure is not an unbiased estimate of the full-sample effect.

---

## Supplementary Table 7 | Training-unit dose-response

Holding the number of training items fixed and varying only the number of units they came from separates "more data" from "more sites".

| Dataset | k range | Expected score | Spread across draws |
|---|---|---|---|
| Durian, farms | 1 → 6 | +61.5% | 3.8-fold narrower |
| GWHD, sessions | 4 → 38 | no monotone trend | 5.9-fold narrower |
| HAR, subjects | 4 → 24 | +6.6% and +4.8% | 6.4- and 2.9-fold narrower |

The spread falls in every dataset; the expected score rises in two of three. The difference follows how internally varied a unit is: a durian farm is one manager, one cultivar mix and one visit, while a wheat session already mixes location, genotype, growth stage and sensor.

**Three limits on this table.** Draws that could not reach the item budget were skipped, and at small *k* that is a large fraction — 4 of 6 intended draws survive at *k* = 1 on durian, 1 of 8 at *k* = 2 and 3 of 8 at *k* = 4 on wheat. The skipped draws are those whose units are too small to supply the budget, so the low-*k* end is biased towards larger units and the durian increase is measured against a baseline of four farms. Two seeds per draw. And these runs predate the protocol change: they use the contaminated validation arrangement, which affects their absolute level but not the comparison across *k*, since every draw shares it. The crossing points at which draw-to-draw variation falls below seed-to-seed variation — six farms, around sixteen sessions, twenty-four subjects — are descriptive of these draws rather than estimates with an uncertainty attached.

---

## Supplementary Table 8 | Every unit, scored on its own

Each unit was scored with the fold that withheld it from training, under weights that no signal from that unit selected.

| Dataset | Units | Mean | s.d. | CV | Min | Max | Max/min |
|---|---|---|---|---|---|---|---|
| Durian, farms | 8 | 0.2247 | 0.0883 | 0.39 | 0.094 | 0.356 | 3.8 |
| GWHD, sessions | 47 | 0.4646 | 0.1404 | 0.30 | 0.085 | 0.803 | 9.5 |
| BreaKHis, patients | 81 | 0.8693 | 0.2142 | 0.25 | 0.053 | 1.000 | 18.7 |
| HAR, subjects | 30 | 0.9406 | 0.0550 | 0.06 | 0.750 | 0.999 | 1.3 |

Means are equal-weight over units and therefore differ slightly from the pooled figures in Supplementary Table 2 wherever unit size correlates with score; the largest such gap is 0.028 on wheat.

The ratio column is the least stable of these statistics, since it is a ratio of extremes, but the ordering it gives matches the coefficient of variation. On activity recognition the coefficient of variation of 0.06 conceals a difference in error rate between the best and worst subject of 635 against 4.8 windows per error, a factor of 130.

---

## Supplementary Table 9 | How far a figure moves when the evaluation units change

For each unit count *k*, *k* units were drawn without replacement from the pool and the aggregate recomputed, 4,000 times. Equal weighting.

| dataset | n_units | k_eval | mean | sd | width90 |
|---|---|---|---|---|---|
| durian farms | 8 | 1 | 0.2245 | 0.0822 | 0.2627 |
| durian farms | 8 | 2 | 0.2242 | 0.0546 | 0.1644 |
| durian farms | 8 | 4 | 0.2249 | 0.0313 | 0.1071 |
| durian farms | 8 | 8 | 0.2247 | 0.0 | 0.0 |
| GWHD sessions | 47 | 1 | 0.4727 | 0.1478 | 0.4571 |
| GWHD sessions | 47 | 2 | 0.4726 | 0.1063 | 0.3524 |
| GWHD sessions | 47 | 4 | 0.4752 | 0.0733 | 0.2415 |
| GWHD sessions | 47 | 8 | 0.4736 | 0.0496 | 0.164 |
| GWHD sessions | 47 | 16 | 0.4735 | 0.031 | 0.102 |
| GWHD sessions | 47 | 32 | 0.4734 | 0.015 | 0.0493 |
| GWHD sessions | 47 | 47 | 0.4735 | 0.0 | 0.0 |
| BreaKHis patients | 81 | 1 | 0.8638 | 0.2154 | 0.6631 |
| BreaKHis patients | 81 | 2 | 0.8628 | 0.1562 | 0.4723 |
| BreaKHis patients | 81 | 4 | 0.8651 | 0.1056 | 0.3159 |
| BreaKHis patients | 81 | 8 | 0.8647 | 0.072 | 0.2277 |
| BreaKHis patients | 81 | 16 | 0.8647 | 0.0489 | 0.1613 |
| BreaKHis patients | 81 | 32 | 0.8643 | 0.0293 | 0.0948 |
| BreaKHis patients | 81 | 64 | 0.8648 | 0.0126 | 0.0418 |
| BreaKHis patients | 81 | 81 | 0.8646 | 0.0 | 0.0 |
| HAR subjects | 30 | 1 | 0.9522 | 0.0493 | 0.1272 |
| HAR subjects | 30 | 2 | 0.9502 | 0.0345 | 0.1179 |
| HAR subjects | 30 | 4 | 0.9504 | 0.0242 | 0.0827 |
| HAR subjects | 30 | 8 | 0.9504 | 0.0156 | 0.0512 |
| HAR subjects | 30 | 16 | 0.9502 | 0.0088 | 0.0284 |
| HAR subjects | 30 | 30 | 0.9502 | 0.0 | 0.0 |

`width90` is the 5th-to-95th-percentile range of the resampled aggregate.

**What this is and is not.** The mean is flat in *k* by construction, so the entire effect is on precision. But this is variation *within the observed pool*: sampling without replacement forces the range to zero once the pool is exhausted, so the numbers bound how much the reported figure would have moved had a different subset of the available units been reported. They are not a confidence interval for a new farm or patient, and should not be read as a sample-size recommendation for future studies. A cluster-robust interval for that purpose would need either a model of how units vary or more units than any of these corpora contain.

Read within that limit, the figures are still large. A wheat score computed on the nine or ten sessions a five-fold design leaves sits in a range about 0.16 wide, a third of the figure itself.

**Instance weighting.** Under instance weighting the mean is not flat: it runs from the equal-weight mean at *k* = 1 to the pooled figure at *k* = N, and the drift equals the gap between the two aggregations. That gap is +0.028 on wheat, where large sessions score higher, and below 0.002 on the other three. The full instance-weighted curve is in `results_clean/summary_eval_curve.csv`.

**One approximation.** Each unit's score was produced by the fold that withheld it, so resampling units mixes training sets. The curve approximates a *k*-unit evaluation rather than reproducing one. Reproducing it exactly would mean fixing a training set and resampling only the evaluation units, which requires retraining.

---

## Supplementary Table 10 | Per-farm scores, six durian detectors

mAP50 on the withheld farm, averaged over seeds, clean protocol.

| Withheld farm | YOLO11n | YOLO11s | YOLO11m | YOLO11l | RT-DETR-L | Faster R-CNN |
|---|---|---|---|---|---|---|
| farm 1 | 0.404 | 0.390 | 0.389 | 0.406 | 0.257 | 0.295 |
| farm 3 | 0.234 | 0.227 | 0.303 | 0.295 | 0.282 | 0.277 |
| farm 5 | 0.342 | 0.223 | 0.234 | 0.253 | 0.274 | 0.276 |
| farm 4 | 0.185 | 0.226 | 0.270 | 0.294 | 0.272 | 0.283 |
| farm 7 | 0.234 | 0.199 | 0.229 | 0.237 | 0.221 | 0.201 |
| farm 0 | 0.190 | 0.176 | 0.192 | 0.192 | 0.213 | 0.177 |
| farm 2 | 0.113 | 0.100 | 0.115 | 0.103 | 0.103 | 0.138 |
| farm 6 | 0.097 | 0.057 | 0.099 | 0.079 | 0.097 | 0.134 |

Best over worst is 3.8 for the mean across models. The ranking is largely preserved across the six detectors: mean pairwise Spearman ρ = 0.87, and all six place farms 2 and 6 in the bottom two. Which site a model fails on is therefore substantially a property of the site, across a change of architecture family.

Class composition differs sharply between these folds, so the standard deviation of the eight numbers describes the sample rather than estimating an uncertainty; the single-class decompositions in Supplementary Table 5 are the control on that.

---

## Supplementary Table 11 | Per-farm covariates against fold score

Eight covariates recorded at capture, correlated with the withheld farm's score across the eight folds, clean protocol.

| Covariate | Pearson *r* |
|---|---|
| Median capture hour | **+0.73** |
| Median Leaf_rot box side | +0.29 |
| Midday share | −0.35 |
| Median solar elevation | −0.32 |
| Number of focal lengths | −0.32 |
| Number of devices | −0.30 |
| Images in fold | −0.23 |
| Share of images at 2.22 mm | +0.06 |

Only capture hour reaches a coefficient worth reporting and it is not usable: farm 1 is the only site photographed after 17:00 and the only one above 0.40, and removing it collapses the association. Solar elevation, which would carry the same signal if light were the mechanism, gives −0.32.

Elsewhere the available covariate is unit size: images per patient against that patient's score gives *r* = +0.06 over 81 patients, windows per subject +0.18 over 30.

**A withdrawn claim.** An earlier five-farm analysis found lesion size correlated with fold score at *r* = +0.54 and read it as evidence that farms with the earliest lesions are served worst. On eight farms *r* = +0.29, within a single focal length +0.35, and for a second class the sign reverses at −0.21. The worst-scoring farm has the second-largest lesions and the second-worst has the smallest. The claim is withdrawn and `verify_claims.py` fails if it returns.

---

## Supplementary Table 12 | Cross-region evaluation

The Sabah set, 281 images from two orchards 1,765–1,872 km from the peninsular farms, was never seen in training under any configuration. Clean protocol.

| Model | Sabah | s.d. across training configurations | Mean withheld peninsular farm |
|---|---|---|---|
| YOLO11n | 0.2558 | 0.0156 | 0.2234 |
| YOLO11s | 0.2526 | 0.0183 | 0.2137 |
| YOLO11m | 0.2623 | 0.0121 | 0.2332 |
| YOLO11l | 0.2811 | 0.0080 | 0.2356 |
| RT-DETR-L | 0.2664 | 0.0115 | 0.2205 |

Sabah scores slightly *above* the mean withheld peninsular farm under all five detectors, and the spread across training configurations when scoring a fixed Sabah set (0.008–0.018) is a fraction of the spread across withheld peninsular farms (0.088). Crossing to another island costs about what withholding a neighbouring farm costs.

**What this does not show.** The design does not separate distance from management regime, season or aggregation level. The Sabah trees are one manager, one planting season, four consecutive days; the peninsular farms are eight separate operations across eight months. The comparison bounds the cross-region penalty rather than attributing it, and the plausible reading is that the number and heterogeneity of management units pooled into a figure matters more than the distance between them. An asymmetry works against the conclusion rather than for it: the per-farm scores come from models trained on seven farms and the Sabah scores from models trained on all eight.

---

## Supplementary Table 13 | Minimum unit size

Standard deviation across units as the minimum items per unit is raised, equal weight.

| Minimum items | Durian bursts | GWHD sessions | BreaKHis patients | HAR subjects |
|---|---|---|---|---|
| 5 | 0.247 (58) | 0.128 (47) | 0.175 (81) | 0.052 (30) |
| 10 | 0.198 (29) | 0.128 (47) | 0.175 (81) | 0.052 (30) |
| 20 | 0.235 (8) | 0.131 (44) | 0.175 (81) | 0.052 (30) |
| 50 | – | 0.124 (31) | 0.178 (77) | 0.052 (30) |

Surviving units in brackets. At the largest threshold each dataset retains 97%, 102% and 100% of its five-item dispersion. The durian column thins out quickly and bounds the claim rather than establishing it; for the two classification datasets the analytic decomposition in Supplementary Table 15 is the stronger control. These figures come from the contaminated protocol, since the sweep was run before the retraining; the quantity it bounds — whether small units manufacture the dispersion — does not depend on which protocol produced the scores.

---

## Supplementary Table 14 | Units nested within folds, and the robustness of the decomposition

Where a fold contains several units, a crossed decomposition of fold-aggregate scores cannot separate which units were withheld from *training* from which of the withheld units a score is *computed on*. The hierarchical decomposition in Supplementary Table 5 separates them. The fold component is 1.9% on histopathology, 9.6% on wheat and 13.3% on activity recognition, against unit components of 88.1%, 68.0% and 73.3%. At these unit counts — 64 to 66 training patients, 37 or 38 training sessions, 24 training subjects — removing one group of units from training rather than another does not measurably change the model, while which unit the resulting model is scored on changes the number a great deal.

The point estimates above are what the main text quotes. Three checks bound how far they can be trusted. `scripts/analyse/decomposition_robustness.py`.

**(a) Uncertainty.** Ninety per cent percentile bootstrap intervals, resampling units (farms on durian, folds elsewhere) 1,500 times. Seeds and models are not resampled, because they are not draws from a population.

| Dataset | Component | Point (%) | 90% interval |
|---|---|---|---|
| durian | unit | 82.4 | [57.9, 87.2] |
| durian | model | 0.0 | [0.0, 5.0] |
| durian | interaction | 7.8 | [1.2, 18.3] |
| durian | seed | 9.8 | [7.9, 21.1] |
| gwhd | unit | 68.0 | [46.0, 86.5] |
| gwhd | fold | 9.6 | [0.0, 14.6] |
| gwhd | model | 0.3 | [0.0, 2.5] |
| gwhd | seed | 0.1 | [0.0, 1.2] |
| gwhd | residual | 22.1 | [8.8, 43.3] |
| breakhis | unit | 88.1 | [80.1, 92.8] |
| breakhis | fold | 1.9 | [0.1, 2.5] |
| breakhis | model | 0.0 | [0.0, 0.1] |
| breakhis | seed | 0.0 | [0.0, 0.6] |
| breakhis | residual | 9.9 | [6.5, 17.4] |
| har | unit | 73.3 | [64.8, 81.9] |
| har | fold | 13.3 | [0.9, 19.9] |
| har | model | 2.7 | [0.2, 8.3] |
| har | seed | 0.0 | [0.0, 0.1] |
| har | residual | 10.6 | [5.2, 19.9] |

The intervals are wide, as eight farms and five folds warrant, and the point estimates should not be read as precise. What they support is narrower and holds on every dataset: the lower bound of the unit component (46–80%) exceeds the upper bound of the model component (0.1–8.3%).

**(a2) The resampling unit matters.** The intervals above resample folds. A five-fold design offers only five folds to resample, and their training sets share 60% of the units, so those intervals rest on five numbers and may not cover as stated. Resampling units instead — 30 to 81 of them, and the level at which the design is exchangeable — gives:

| Dataset | Component | Point (%) | Fold bootstrap | Unit bootstrap |
|---|---|---|---|---|
| gwhd | unit | 68.0 | [46.0, 86.6] | [44.4, 74.9] |
| gwhd | fold | 9.6 | [0.0, 14.6] | [5.0, 30.8] |
| gwhd | model | 0.3 | [0.0, 2.5] | [0.0, 1.7] |
| breakhis | unit | 88.1 | [79.6, 92.8] | [75.0, 88.8] |
| breakhis | fold | 1.9 | [0.1, 2.7] | [1.4, 13.2] |
| breakhis | model | 0.0 | [0.0, 0.1] | [0.0, 0.2] |
| har | unit | 73.3 | [65.0, 81.8] | [45.0, 77.4] |
| har | fold | 13.3 | [0.9, 20.2] | [7.1, 41.9] |
| har | model | 2.7 | [0.2, 8.5] | [0.6, 8.2] |

The unit-level intervals are the ones the main text quotes. They are wider on two of three datasets, and the separation survives: the unit component's lower bound (44–75%) exceeds the model component's upper bound (0.2–8.2%) on all three. Resampling units does not remove the dependence induced by overlapping training sets, so these intervals may still be narrow. Repeated group partitions with retraining would settle it; that is a GPU-bound experiment and is not attempted here.

**(b) Model as a fixed effect.** The random-effects reading of the model term assumes the architectures tested sample a population of architectures, which two to six configurations do only loosely. Treating the model as a fixed effect drops that assumption and reports the share of the observed sum of squares between the specific models tested. On durian:

| Component | Random effect (%) | Fixed effect (%) |
|---|---|---|
| unit | 82.4 | 82.8 |
| model | 0.0 | 0.7 |
| interaction | 7.8 | 7.7 |
| seed | 9.8 | 8.8 |

The two agree to within one point. The three nested datasets already report sum-of-squares shares, which do not depend on the assumption.

**(c) Sensitivity to the model roster.** The most direct version of the concern is that a unit share of 68–88% could be an artefact of comparing models that are too alike. Durian has six configurations across three architecture families, so the decomposition can be repeated for every subset of them.

By number of families the subset spans:

| Families | Subsets | Mean unit share (%) | Range | Mean model share (%) | Max model share (%) |
|---|---|---|---|---|---|
| 1 | 11 | 90.2 | 87.3–92.7 | 0.6 | 1.1 |
| 2 | 31 | 80.7 | 71.5–85.3 | 0.5 | 1.0 |
| 3 | 15 | 76.5 | 70.1–80.2 | 0.5 | 0.8 |

By number of configurations:

| Configurations | Subsets | Mean unit share (%) | Range | Max model share (%) |
|---|---|---|---|---|
| 2 | 15 | 82.6 | 71.5–92.7 | 1.1 |
| 3 | 20 | 81.3 | 70.1–90.8 | 1.0 |
| 4 | 15 | 80.8 | 74.0–89.4 | 0.9 |
| 5 | 6 | 80.4 | 77.1–85.0 | 0.8 |
| 6 | 1 | 80.2 | 80.2–80.2 | 0.7 |

Across all 57 subsets the unit share runs from 70 to 93% and the model share never exceeds 1.1%. The direction is the one the concern predicts — subsets confined to one family give a mean unit share of 90%, subsets spanning three families 77% — and the magnitude is what the conclusion rests on: even the most heterogeneous roster available leaves three quarters of the variance with the unit.

**What (c) does not establish.** It shows the conclusion is not sensitive to which of *these* six configurations are compared. It cannot speak to architectures that were not trained — a vision transformer backbone, a diffusion-feature classifier — and the share under those is not known. On wheat and histopathology, where only two capacities of one family were trained, the model component is a lower bound on what a family change would contribute; the durian result across three families is the best available indication of how much that bound could move.

---

## Supplementary Table 15 | Measurement noise and within-unit dependence

**The metric matters.** On BreaKHis the per-unit score is top-1 accuracy, a correct-classification proportion. On HAR the main text reports macro F1, which is **not** a proportion and whose variance is not p(1−p)/n; the analysis below uses per-subject accuracy, which the released table also carries. The two correlate at 0.99 across subjects but are not interchangeable here. An earlier version of this manuscript applied the binomial decomposition to macro F1, which was incorrect.

**Two partial estimates, neither an upper bound.** Each unit is scored five times by independently seeded training runs; the standard error of a unit's mean across repeats measures instability without assuming anything about the items inside the unit, but captures training randomness rather than within-unit sampling. Treating the score as a binomial mean over the unit's items captures sampling instead, but assumes the items are independent. Neither bounds the other, and taking the larger — as an earlier version did — still omits the within-unit dependence.

**Sensitivity to the intraclass correlation.** Inflating the binomial variance by a design effect 1 + (m̄ − 1)·ICC gives the noise share as a function of the within-unit correlation. ICC = 0 recovers the independence assumption.

| Dataset | Model | ICC 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|---|
| BreaKHis patients | yolo11n-cls | 3% | 12% | 20% | 36% | 53% | 86% |
| BreaKHis patients | yolo11s-cls | 5% | 14% | 22% | 39% | 57% | 91% |
| HAR subjects | mlp256 | 6% | 107% | 207% | 409% | 610% | 1012% |
| HAR subjects | rf | 6% | 99% | 192% | 379% | 565% | 939% |

Values are the share of the between-unit variance attributable to measurement noise.

**The two corpora behave very differently, because their units differ in size.** BreaKHis patients average 98 images, so the design effect grows slowly: noise reaches half the observed variance at an intraclass correlation of 0.26. HAR subjects average 343 windows, and the same threshold is crossed at 0.02. Adjacent accelerometer windows from one wearer are unlikely to have a correlation that low.

**What this means for the claim.** The statement that between-unit heterogeneity dominates the per-unit spread is well supported on patients, where it survives any plausible within-slide correlation, and is conditional on an implausibly low correlation on subjects. The main text states it as a conditional rather than a number. Settling it requires per-item predictions and a block bootstrap over slides or contiguous window segments, which the released per-unit tables do not support.

Average precision is not a proportion, so none of this is available on the two detection datasets, where the minimum-unit-size sweep in Supplementary Table 13 is the weaker control.

---

## Supplementary Table 16 | When the evaluation sample changes which model wins

A large unit component does not by itself unsettle a model comparison. If one model beats another by the same margin on every unit, the unit effect can be arbitrarily large while the comparison stays stable. What matters is the difference between two models *on the same unit*.

**The primary quantity is the paired-difference interval.** Units are resampled with replacement 6,000 times and the 5th-to-95th percentile of the mean paired difference is taken. This asks whether the comparison survives a different draw of units, assumes no distributional form, and does not extrapolate beyond the corpus.

| Dataset | Pair | Units | Mean diff. | 90% interval | Excludes 0 | *d* | *d* 90% CI | k from (1.645/d)², 90% CI |
|---|---|---|---|---|---|---|---|---|
| durian farms | yolo11m vs yolo11s | 8 | +0.0195 | [+0.0069, +0.0318] | **yes** | 0.84 | [0.30, 1.81] | [1, 31] |
| durian farms | yolo11l vs yolo11s | 8 | +0.0220 | [+0.0069, +0.0379] | **yes** | 0.78 | [0.31, 1.53] | [2, 28] |
| HAR subjects | mlp256 vs rf | 30 | +0.0192 | [+0.0086, +0.0303] | **yes** | 0.52 | [0.27, 0.80] | [5, 37] |
| durian farms | yolo11n vs yolo11s | 8 | +0.0097 | [-0.0071, +0.0247] | no | 0.33 | [0.03, 1.48] | [2, 2223] |
| durian farms | yolo11l vs yolo11n | 8 | +0.0122 | [-0.0090, +0.0351] | no | 0.30 | [0.04, 0.87] | [4, 1796] |
| durian farms | yolo11m vs yolo11n | 8 | +0.0098 | [-0.0083, +0.0310] | no | 0.27 | [0.04, 0.82] | [5, 1519] |
| durian farms | frcnn-r50 vs yolo11m | 8 | -0.0112 | [-0.0346, +0.0105] | no | 0.27 | [0.03, 0.87] | [4, 2467] |
| durian farms | frcnn-r50 vs yolo11l | 8 | -0.0137 | [-0.0426, +0.0126] | no | 0.27 | [0.04, 0.98] | [3, 2080] |
| durian farms | rtdetr-l vs yolo11l | 8 | -0.0151 | [-0.0512, +0.0117] | no | 0.26 | [0.06, 0.83] | [4, 696] |
| durian farms | rtdetr-l vs yolo11m | 8 | -0.0126 | [-0.0454, +0.0103] | no | 0.25 | [0.07, 0.88] | [4, 601] |
| BreaKHis patients | yolo11n-cls vs yolo11s-cls | 81 | +0.0093 | [+0.0013, +0.0175] | **yes** | 0.21 | [0.05, 0.41] | [17, 1223] |
| GWHD sessions | yolo11n vs yolo11s | 47 | -0.0177 | [-0.0371, +0.0031] | no | 0.21 | [0.03, 0.57] | [9, 3948] |
| durian farms | yolo11l vs yolo11m | 8 | +0.0024 | [-0.0058, +0.0115] | no | 0.15 | [0.03, 0.84] | [4, 2987] |
| durian farms | frcnn-r50 vs yolo11s | 8 | +0.0083 | [-0.0236, +0.0396] | no | 0.14 | [0.02, 1.11] | [3, 4660] |
| durian farms | rtdetr-l vs yolo11s | 8 | +0.0069 | [-0.0311, +0.0363] | no | 0.11 | [0.02, 1.98] | [1, 5537] |
| durian farms | frcnn-r50 vs rtdetr-l | 8 | +0.0014 | [-0.0125, +0.0156] | no | 0.06 | [0.03, 0.86] | [4, 3903] |
| durian farms | rtdetr-l vs yolo11n | 8 | -0.0029 | [-0.0424, +0.0338] | no | 0.04 | [0.03, 0.87] | [4, 3841] |
| durian farms | frcnn-r50 vs yolo11n | 8 | -0.0014 | [-0.0367, +0.0328] | no | 0.02 | [0.02, 0.79] | [5, 5604] |

**Four of eighteen pairs have an interval that excludes zero.** Three are capacity comparisons within one family; the fourth, the perceptron against the random forest on activity recognition, is the one comparison in the study between models sharing nothing but their input features. Among the image datasets no resolved pair crosses an architecture family. The two wheat detectors differ by 0.018 mAP50 with an interval from −0.037 to +0.003; the fourteen unresolved pairs are not shown to be equal, only not separated by the units available.

**Why the standardised effect is not converted into a required unit count.** An earlier version of this manuscript reported (1.645/*d*)² as the number of units a comparison needs. The last column shows why that was withdrawn: the conversion squares the reciprocal of *d*, so the bootstrap uncertainty in *d* is amplified into intervals spanning three orders of magnitude. For the two wheat detectors *d* = 0.21 with a 90% interval of [0.03, 0.58], and the implied unit count runs from 9 to 3,948. On the eight-farm durian pairs the median interval width is over two thousand units. The expression remains a correct asymptotic scaling — the required count grows as *d*⁻² — but it is not an estimate at these unit counts, and the manuscript no longer presents it as one.

**A third quantity, reported in Fig. 4b.** Drawing *k* units without replacement and recording how often the ranking flips relative to the full pool describes how quickly a comparison becomes stable *within its own corpus*. Because the sampling is without replacement, every curve reaches zero at *k* = *N* by construction; the panel says nothing about a larger population of units. An earlier version of this manuscript conflated this within-pool rate with the population extrapolation above.

**What none of the three establishes.** All are conditional on the roster trained here, and on these corpora. Both models in each pair share folds, seeds and preprocessing, so the only thing resampled is which units the comparison is computed on.

---

## Supplementary Table 17 | Does the model's confidence anticipate a failing unit?

Mean maximum confidence or probability on a withheld unit, against that unit's score.

| Dataset | Units | Statistic | Pearson *r* | Spearman | Between-unit CV |
|---|---|---|---|---|---|
| Durian, farms | 8 | mean max confidence | +0.09 | +0.02 | 0.39 |
| GWHD, sessions | 47 | mean max confidence | +0.26 | +0.39 | 0.30 |
| BreaKHis, patients | 81 | mean max probability | **+0.57** | **+0.79** | 0.25 |
| BreaKHis, patients | 81 | mean entropy | −0.57 | −0.79 | 0.25 |
| BreaKHis, patients | 81 | mean margin | +0.57 | +0.79 | 0.25 |

Entropy and margin carry the same information as the maximum probability to three decimal places, so the choice of uncertainty statistic is not what determines the answer.

**The association strengthens as the units become more alike**, which is the same ordering as the partitioning effect and the number of units needed for a given precision. As a screen on histopathology, flagging the least confident fifth of patients retrieves eight of the twelve scoring below 0.80 against a base rate of 15%; on wheat the same rule retrieves three of eleven against a base rate of 23%, which barely beats chance.

**Three caveats.** The histopathology correlation is partly what calibration means — for a well-calibrated classifier the mean maximum probability estimates accuracy — so a positive association is closer to expected than to a discovery. What is not automatic is that it survives on patients the model never trained on, and it does: removing the five worst patients raises the correlation to +0.76. Second, the tail escapes it: the five patients scoring below 0.60 carry a mean confidence of 0.916 against 0.968 for the corpus, five points of confidence for fifty-nine points of accuracy. Third, these are single-seed statistics on wheat and histopathology against five-seed scores, and they come from the contaminated protocol; the gap between +0.09 and +0.57 is far larger than either source of error, but the individual values would move.

Silence — returning no detection at all — carries no usable signal on either detection dataset: on durian the silence rate correlates with the score at +0.30, the wrong sign for an abstention rule, and the worst-scoring farm returns nothing on 0.2% of its images.

---

## Supplementary Figure 1 | Disease stage is an axis no dataset records

Two photographs of the same lesion type at different stages, from the same farm on the same visit, with the annotated boxes overlaid. Nothing in the released metadata of any surveyed dataset — including ours — distinguishes them. The per-unit variation this paper measures is partly a stand-in for axes like this one that no corpus records.

---

## Supplementary Note 1 | The filename join, in full

Farm identity was originally joined to the annotated images by filename. Two field visits produced overlapping camera counters; the second import received `(1)` suffixes, and the resize step that produced the 640-pixel annotation set dropped them. Fifty of 560 images (8.9%) therefore carried another photograph's farm, 49 of them labelled farm 0 when they had been taken at farm 6.

The consequence was exactly the failure this paper is about, one level up. The fold that withheld farm 0 had trained on farm 6, so 29.7% of its held-out set came from a farm the model had seen. That fold scored 0.394, the highest of five.

It was found by hashing image content against the original files rather than by inspection. Attribution is now content-based: each annotated image takes the farm of its nearest original by 64-bit difference hash, accepted only when no original from a different farm lies within three bits. On the 570 images whose filenames still matched an original the hash recovered it with a median Hamming distance of 0. The procedure attributed 829 of 1,033 images and recovered 303 whose filenames the annotation platform had replaced, raising the pool from 560 to 827 images and the usable farms from five to eight. Under the corrected attribution that farm scores 0.190.

The 204 images that could not be attributed are not a random sample: their boxes over-represent one class (56% against 41%) and under-represent another (4% against 17%).

**Class definition.** Six classes were retained after consolidation. Merges and splits were constrained by registered-product mode-of-action groups — two symptoms that call for the same intervention are not usefully separated by a detector intended to support that intervention — and by the taxonomy of the causal genera [17,18]. Images whose only annotation was a single psyllid egg were withdrawn, which was itself a judgement.

---

## Supplementary Note 2 | Audits of the three public datasets

**BreaKHis: 81 patient identifiers against the 82 reported.** The original description states 82 patients. Parsing the released directory structure returns 7,909 images carrying 81 distinct patient identifiers, and every image in the distribution is accounted for by one of them. The discrepancy is therefore in the identifiers, not in missing images: either two slides from one patient were issued under identifiers we cannot distinguish, or the published count includes a patient whose images are not in the public release. Since the patient is this paper's evaluation unit on that corpus the number matters, and we report what the files support. Nothing in the analysis depends on which explanation is correct.

**GWHD.** Decoding the released parquet distribution gave 6,512 images. One byte-identical pair with inconsistent annotation was removed, as were 547 sub-pixel boxes among 274,824, leaving 6,510 images across 47 domains. No further duplicates by content hash.

**UCI HAR.** No content duplicates; the official subject-wise release was not used, and five-fold partitions were constructed for comparability with the other datasets.

---

## Supplementary Note 3 | A reporting checklist

Four numbers should accompany any aggregate score computed on clustered data.

**How many independent units contributed to training.** Not the item count. The dose-response curves show the expected score still moving at six farms and sixteen sessions.

**How many independent units the evaluation rests on.** This is a different number from the first and is what bounds the precision the figure is entitled to claim. On wheat, nine or ten sessions leaves a subset range about 0.16 wide.

**The dispersion across withheld units.** A single pooled figure conceals a 3.8- to 18.7-fold ratio between the best and worst unit in these four corpora.

**Whether units or items are weighted.** The expected score for a random item and for a random unit are different estimands, and only one of them is usually reported. Here they differ by up to 0.028.

Three further practices follow from the results.

**Do not let the withheld unit select the checkpoint.** Under leave-one-unit-out the fold's validation set is the thing being reported, and keeping the best epoch on it changed the measured overstatement by up to 14.7 points and reversed a model ranking on one dataset. Draw the inner validation set from training units, or fix the epoch budget in advance.

**Report the number of units a comparison needs, not only its margin.** A difference of one point of mAP50 between two architectures on nine sessions is the outcome of a draw.

**Release a unit identifier.** One of fourteen surveyed crop-disease datasets does. Without it none of the above can be computed by anyone but the original authors.

`unitcheck.py` computes the first, third and fourth from a table of item-to-unit assignments and, where available, per-unit scores. It reads neither images nor weights and depends only on the standard library.

---

## Supplementary Note 4 | What the results do not license

**The durian folds are not eight measurements of one quantity.** Class composition differs so sharply between farms that five folds return no figure for one class, so the standard deviation of the fold means describes the sample rather than estimating an uncertainty. The single-class decompositions are the control.

**Eight units is few.** The durian resampling analyses draw from a pool of eight; the finite-population correction makes their flip rates lower than the same *d* would give in a large pool, and their *d* values carry wide uncertainty. The three public corpora, with 47, 81 and 30 units, carry the inference.

**Twenty per cent of the annotated peninsular set is excluded** for want of a content match, and the exclusions are not a random sample.

**The inner-validation rule differs between datasets.** It is drawn at item level on durian and wheat and at whole-patient level on histopathology, because sections from one patient are near-duplicates and an item-level inner split would give early stopping a near-perfect and meaningless signal. The histopathology protocol is therefore stricter, and its item-level figures fall further relative to the contaminated protocol (0.99 to 0.95) than the other datasets'. Both arms within each dataset share the rule, so the item-versus-unit comparison is not confounded; the comparison *between* datasets of how much the protocol change cost is.

**Model rosters are uneven.** Durian has six configurations across three families; wheat and histopathology have two capacities within one family each, so their model components are lower bounds on what a family change would contribute. Faster R-CNN and the dose-response curves use three and two seeds rather than five.

**The dose-response curves predate the protocol change** and use the contaminated validation arrangement, and skip a large fraction of draws at small *k*.

**The evaluation-unit resampling mixes training sets**, since each unit's score comes from the fold that withheld it, and it cannot speak to uncertainty about units outside the pool.

**Confidence and covariate analyses come from the contaminated protocol** and, on wheat and histopathology, from a single seed.

**Modality is confounded with ceiling and with feature engineering.** The smallest overstatement is on the only non-image dataset, which is also the only one near its accuracy ceiling and the only one using hand-engineered, subject-invariant features. These three explanations cannot be separated here.

**Annotation.** One collector, one reviewer, no measured inter-annotator agreement, and a standard calibrated in the field rather than written down. The same reviewer passed over every image, so the standard applied was constant across farms; whether its bias is constant is a different question, since it could interact with the class composition that differs between farms, and it was not measured.

**No deployment.** All results are offline evaluations. None predicts the detection rate a grower, pathologist or wearer would experience.
