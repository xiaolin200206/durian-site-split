<!-- DRAFT: every block below must be re-read against the v2 results before submission; delete this line when done -->

Result-dependent paragraphs of the manuscript. Each block is inserted where the template has {{block_name}}; numbers inside are placeholders resolved from results_applied/paper_numbers.json like the rest of the text.

<!-- block: abstract_rq3_sentence -->
Fine-tuning on {{rq3/ft/20/images:.0f}} images from the new farm changed its mAP50 by {{rq3/ft/20/gain:+.3f}}, and retraining with them added by {{rq3/rt/20/gain:+.3f}}.

<!-- block: random_farm_weighted_sentence -->
The gap does not come from the way the two figures are averaged: scored farm by farm and averaged in the same way as the unseen-farm figure, the random-split models reached {{random_split_farm_weighted_range/0:.3f}}–{{random_split_farm_weighted_range/1:.3f}}.

<!-- block: rq2_results_opening -->
With the number of iterations held constant, unseen-farm mAP50 rose with both the number of training farms and the number of images per farm (Fig. 3a, Table 4; each held-out farm in Table S4). Training on one farm gave {{grid/k1_mall/mAP50:.3f}} with all of its images; all images of seven farms gave {{grid/k7_mall/mAP50:.3f}}. Fitted over all {{rq2_design/runs}} runs, doubling the number of farms changed mAP50 by {{regression/all runs/farms:.3f}} (90% interval {{regression/all runs/farms_ci/0:.3f}} to {{regression/all runs/farms_ci/1:.3f}}) and doubling images per farm by {{regression/all runs/photos:.3f}} ({{regression/all runs/photos_ci/0:.3f}} to {{regression/all runs/photos_ci/1:.3f}}); the difference between the two slopes was {{regression/all runs/diff:.3f}} ({{regression/all runs/diff_ci/0:.3f}} to {{regression/all runs/diff_ci/1:.3f}}).

<!-- block: rq2_steps_paragraph -->
The steps were uneven (Table S6). With all images per farm, going from one to two farms changed mAP50 by {{steps/farms 1->2 @ all per farm/diff:.3f}}, from two to four by {{steps/farms 2->4 @ all per farm/diff:.3f}} ({{steps/farms 2->4 @ all per farm/ci/0:.3f}} to {{steps/farms 2->4 @ all per farm/ci/1:.3f}}) and from four to seven by {{steps/farms 4->7 @ all per farm/diff:.3f}}; with 50 images per farm the three steps were {{steps/farms 1->2 @ 50 per farm/diff:.3f}}, {{steps/farms 2->4 @ 50 per farm/diff:.3f}} and {{steps/farms 4->7 @ 50 per farm/diff:.3f}}. The step from four to seven farms is confounded: seven farms is a single combination that always includes every other farm, including those most similar to the held-out one, whereas the two- and four-farm sets are two draws each. The design therefore cannot say whether the gain at seven farms reflects the number of farms or the inclusion of particular farms.

<!-- block: rq2_matched_paragraph -->
At matched image budgets (Fig. 3b, Table 5A), seven farms × 15 images scored {{equal_budget/k7m15_vs_k2m50/mAP/0:.3f}} against {{equal_budget/k7m15_vs_k2m50/mAP/1:.3f}} for two farms × 50 (difference {{equal_budget/k7m15_vs_k2m50/diff:.3f}}; {{equal_budget/k7m15_vs_k2m50/ci/0:.3f}} to {{equal_budget/k7m15_vs_k2m50/ci/1:.3f}}; more farms better on {{equal_budget/k7m15_vs_k2m50/wins}} held-out farms); four farms × 50 scored {{equal_budget/k4m50_vs_k2mall/mAP/0:.3f}} against {{equal_budget/k4m50_vs_k2mall/mAP/1:.3f}} for two farms × all ({{equal_budget/k4m50_vs_k2mall/diff:.3f}}; {{equal_budget/k4m50_vs_k2mall/ci/0:.3f}} to {{equal_budget/k4m50_vs_k2mall/ci/1:.3f}}; {{equal_budget/k4m50_vs_k2mall/wins}}); and seven farms × 50 scored {{equal_budget/k7m50_vs_k4mall/mAP/0:.3f}} against {{equal_budget/k7m50_vs_k4mall/mAP/1:.3f}} for four farms × all ({{equal_budget/k7m50_vs_k4mall/diff:.3f}}; {{equal_budget/k7m50_vs_k4mall/ci/0:.3f}} to {{equal_budget/k7m50_vs_k4mall/ci/1:.3f}}; {{equal_budget/k7m50_vs_k4mall/wins}}). Differences are computed from unrounded values.

<!-- block: rq2_coverage_paragraph -->
Part of the value of a farm is the classes it carries. Training sets drawn from one farm covered, on average, {{coverage_k1_pct:.0f}}% of the held-out farm's annotated images, and those from two farms {{coverage_k2_pct:.0f}}%; almost every set drawn from four or more farms covered all of them. Within the one-farm and two-farm sets, where coverage varies, it was correlated with unseen-farm mAP50 (Spearman ρ = {{coverage/spearman_within_k1:.2f}} and {{coverage/spearman_within_k2:.2f}}). Coverage does not account for the farm slope, however: with coverage added to the regression the slope for doubling farms was {{regression/adjusted for class coverage/farms:.3f}} ({{regression/adjusted for class coverage/farms_ci/0:.3f}} to {{regression/adjusted for class coverage/farms_ci/1:.3f}}), and restricted to the {{regression/full class coverage only/n_runs}} runs whose training images covered every class present on the held-out farm it was {{regression/full class coverage only/farms:.3f}} against {{regression/full class coverage only/photos:.3f}} for doubling images per farm (Table 5B).

<!-- block: rq2_classes_paragraph -->
With all images per farm, AP50 on the unseen farm was higher with seven training farms than with one for every class, although not every intermediate step was an improvement, and the psyllid classes remained lowest throughout (Fig. 4; Table S3): with seven farms, psyllid reached {{class_by_k/Psyllid/7:.3f}} and psyllid damage {{class_by_k/Psyllid_damage/7:.3f}}, against {{class_by_k_other_min7:.3f}}–{{class_by_k_other_max7:.3f}} for the other four classes.

<!-- block: rq3_results_paragraph -->
Without calibration, the detector trained on the other seven farms scored {{rq3/base_mAP50:.3f}} on the test halves (Table 6, Fig. 5). Fine-tuning on 5, 10 and 20 images from the calibration half changed mAP50 by {{rq3/ft/5/gain:+.3f}}, {{rq3/ft/10/gain:+.3f}} and {{rq3/ft/20/gain:+.3f}}, and on the whole calibration half (about {{rq3/ft/all/images:.0f}} images) by {{rq3/ft/all/gain:+.3f}}. Retraining with the same images added to the other seven farms changed it by {{rq3/rt/5/gain:+.3f}}, {{rq3/rt/10/gain:+.3f}}, {{rq3/rt/20/gain:+.3f}} and {{rq3/rt/all/gain:+.3f}}.

<!-- block: sabah_paragraph -->
The five Ultralytics detectors trained on seven peninsular farms scored {{sabah_o1_range/0:.3f}}–{{sabah_o1_range/1:.3f}} on the larger Sabah orchard ({{sabah_orchard_images/sabah_o1}} images) and {{sabah_o2_range/0:.3f}}–{{sabah_o2_range/1:.3f}} on the smaller one ({{sabah_orchard_images/sabah_o2}} images), against {{headline_range/peninsula_five_min:.3f}}–{{headline_range/peninsula_five_max:.3f}} on held-out peninsular farms (Table 2). With two orchards, one of which contributes only {{sabah_orchard_images/sabah_o2}} images, this is a check that the peninsular figures are not specific to one region rather than a measurement of cross-regional transfer.

<!-- block: discussion_budget_paragraph -->
Both levers helped: more farms and more images per farm each raised unseen-farm accuracy when training length was held constant. The farm slope was the larger over the whole design, but the steps were uneven, with little change from two to four farms and a large gain at seven that the design cannot attribute to the number of farms alone. Part of the value of an extra farm lies in the classes it adds, and a training set that lacks a class present on the target farm cannot detect it; *Phomopsis* was photographed on three of eight farms. These results sit between two precedents. Beery et al. (2018) found that accuracy at new camera-trap locations was stable beyond two training locations; here accuracy did not keep rising steadily with farms, but it had not stopped rising at seven. Ruigrok et al. (2023) found for weed detection that, at a constant image count, more sub-datasets improved generalisation; the durian results agree in direction and extend the observation to diseases and pests in orchards, with the qualification that the farm advantage was not uniform across budgets. In practical terms, a first collection campaign should visit enough farms to cover the region's diseases and pests before photographing any one farm exhaustively.

<!-- block: discussion_rq3_section -->
### 4.4. Calibrating on arrival

A device that meets a new farm could be adapted with a few images taken on arrival. In this experiment, fine-tuning on {{rq3/ft/20/images:.0f}} images changed the new farm's mAP50 by {{rq3/ft/20/gain:+.3f}} and retraining with them added by {{rq3/rt/20/gain:+.3f}}; the uncalibrated starting point was {{rq3/base_mAP50:.3f}}. The calibration images were labelled by the author, so the result bounds what calibration can achieve when a grower's images are labelled correctly; in practice that labelling is the cost.

<!-- block: conclusion_budget_sentence -->
With training length held constant, both more farms and more images per farm improved unseen-farm accuracy, a doubling of farms by somewhat more, and part of the value of a farm lay in the classes it carried.

<!-- block: conclusion_rq3_sentence -->
A handful of labelled images from the new farm changed its accuracy by {{rq3/ft/20/gain:+.3f}} (fine-tuning) to {{rq3/rt/20/gain:+.3f}} (retraining) mAP50.

<!-- block: highlight_rq3 -->
Twenty labelled images from a new farm changed its mAP50 by {{rq3/ft/20/gain:+.3f}} to {{rq3/rt/20/gain:+.3f}}
