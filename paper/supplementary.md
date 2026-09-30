# Supplementary material

**What a new orchard costs: farm-level generalisation and data budgets for durian disease and pest detection**

Lin Ding Shan

## Note S1. Training details

**Leave-one-farm-out and random-split runs (Sections 2.4–2.5).** Ultralytics 8 (version 8.4.138 for the data-budget and calibration runs). YOLO11n, YOLO11s, YOLO11m, YOLO11l and RT-DETR-L: COCO-pretrained, 640 px, up to 150 epochs, patience 50, batch 32, 32, 16, 8 and 24 respectively, five seeds (42, 1, 2, 3, 4), default augmentation. Faster R-CNN ResNet-50 FPN (torchvision, improved recipe): COCO-pretrained, SGD with cosine annealing, 40 epochs, patience 12, batch 4, three seeds. The inner validation set is 10% of each fold's training images, stratified by each image's dominant class (at least one image per class with two or more images), drawn with seed 20260911 independently of the training seed; the held-out farm is evaluated once with the weights selected on the inner set.

**Data-budget runs (Section 2.6).** YOLO11n, COCO-pretrained, 640 px, batch 32 with the Ultralytics default nominal batch of 64 (gradients accumulated over two batches), no validation during training, no early stopping, final weights evaluated. The training list of n images is repeated r = ceil(640 / n) times so that an epoch has at least 20 batches, and epochs = round(2000 / ceil(n r / 32)), giving 1,988–2,015 iterations in every run. Mosaic augmentation is switched off for the last 10% of epochs; warm-up is the Ultralytics default (at least 100 iterations); the optimiser was selected automatically (AdamW, learning rate 0.001). Farm combinations and image subsets were drawn with a fixed design seed (20260928); within a combination the 15-, 50- and all-image subsets are nested prefixes of one shuffled order per farm. The exact images of every training set are listed in results_applied/farm_budget_train_lists.csv.

**New-farm calibration (Section 2.7).** Fine-tuning: from the seven-farm data-budget model, backbone frozen (first 10 modules), AdamW, learning rate 5 × 10⁻⁴, no warm-up, batch 16 with nominal batch 16 (no gradient accumulation), calibration images repeated so that an epoch has at least 20 batches, 300 optimiser steps, final weights. Retraining: the data-budget protocol applied to the other seven farms plus the calibration images. Capture order uses EXIF DateTimeOriginal where available and the camera counter otherwise.

## Table S1. Unseen-farm mAP50 of each farm under each detector

Clean leave-one-farm-out protocol; mean over seeds.

| Held-out farm | YOLO11n | YOLO11s | YOLO11m | YOLO11l | RT-DETR-L | Faster R-CNN R50-FPN | Mean |
|---|---|---|---|---|---|---|---|
| Farm 0 | 0.188 | 0.209 | 0.195 | 0.192 | 0.212 | 0.176 | 0.195 |
| Farm 1 | 0.398 | 0.393 | 0.389 | 0.406 | 0.257 | 0.295 | 0.356 |
| Farm 2 | 0.113 | 0.104 | 0.115 | 0.103 | 0.133 | 0.139 | 0.118 |
| Farm 3 | 0.340 | 0.299 | 0.301 | 0.303 | 0.282 | 0.275 | 0.300 |
| Farm 4 | 0.185 | 0.227 | 0.273 | 0.270 | 0.277 | 0.283 | 0.253 |
| Farm 5 | 0.235 | 0.224 | 0.260 | 0.295 | 0.282 | 0.272 | 0.261 |
| Farm 6 | 0.093 | 0.058 | 0.097 | 0.079 | 0.100 | 0.134 | 0.094 |
| Farm 7 | 0.235 | 0.196 | 0.234 | 0.237 | 0.220 | 0.200 | 0.220 |

## Table S2. Per-class AP50 on each unseen farm

Mean of the five detectors that report per-class AP (clean protocol). A dash marks a class absent from that farm.

| Class | Farm 0 | Farm 1 | Farm 2 | Farm 3 | Farm 4 | Farm 5 | Farm 6 | Farm 7 |
|---|---|---|---|---|---|---|---|---|
| Algal spot | 0.356 | 0.233 | 0.147 | 0.646 | 0.015 | 0.505 | 0.000 | – |
| Leaf rot | 0.360 | 0.459 | 0.001 | 0.203 | 0.312 | 0.090 | 0.000 | 0.461 |
| *Phomopsis* | – | – | 0.228 | – | – | 0.366 | 0.205 | – |
| Psyllid | 0.071 | 0.176 | 0.097 | 0.032 | 0.266 | 0.175 | – | 0.055 |
| Psyllid damage | 0.018 | 0.286 | 0.021 | – | 0.083 | 0.054 | – | 0.058 |
| Leafhopper damage | 0.191 | 0.689 | 0.188 | 0.340 | 0.555 | 0.365 | 0.137 | 0.324 |

## Table S3. Per-class unseen-farm AP50 by number of training farms

Data-budget experiment, all images per farm, YOLO11n; held-out farms weighted equally.

| Class | 1 farm | 2 farms | 4 farms | 7 farms |
|---|---|---|---|---|
| Algal spot | 0.056 | 0.150 | 0.211 | 0.256 |
| Leaf rot | 0.095 | 0.156 | 0.161 | 0.279 |
| *Phomopsis* | 0.048 | 0.136 | 0.194 | 0.217 |
| Psyllid | 0.048 | 0.061 | 0.065 | 0.108 |
| Psyllid damage | 0.006 | 0.040 | 0.039 | 0.058 |
| Leafhopper damage | 0.158 | 0.232 | 0.209 | 0.388 |

## Table S4. Data-budget experiment, per held-out farm

YOLO11n unseen-farm mAP50; mean over the two farm draws (one for seven farms) and the seeds.

| Farms | Images per farm | Farm 0 | Farm 1 | Farm 2 | Farm 3 | Farm 4 | Farm 5 | Farm 6 | Farm 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 0.025 | 0.012 | 0.001 | 0.016 | 0.074 | 0.020 | 0.027 | 0.052 |
| 1 | 50 | 0.063 | 0.058 | 0.008 | 0.016 | 0.078 | 0.027 | 0.033 | 0.088 |
| 1 | all | 0.071 | 0.115 | 0.029 | 0.030 | 0.100 | 0.054 | 0.145 | 0.076 |
| 2 | 15 | 0.047 | 0.072 | 0.015 | 0.081 | 0.034 | 0.039 | 0.014 | 0.113 |
| 2 | 50 | 0.079 | 0.105 | 0.019 | 0.140 | 0.093 | 0.095 | 0.023 | 0.107 |
| 2 | all | 0.117 | 0.272 | 0.052 | 0.173 | 0.135 | 0.135 | 0.037 | 0.162 |
| 4 | 15 | 0.068 | 0.060 | 0.016 | 0.064 | 0.087 | 0.124 | 0.040 | 0.099 |
| 4 | 50 | 0.116 | 0.114 | 0.030 | 0.140 | 0.107 | 0.168 | 0.033 | 0.114 |
| 4 | all | 0.151 | 0.252 | 0.038 | 0.194 | 0.137 | 0.209 | 0.055 | 0.132 |
| 7 | 15 | 0.090 | 0.187 | 0.046 | 0.285 | 0.140 | 0.174 | 0.010 | 0.081 |
| 7 | 50 | 0.210 | 0.216 | 0.106 | 0.245 | 0.216 | 0.228 | 0.022 | 0.216 |
| 7 | all | 0.207 | 0.435 | 0.136 | 0.315 | 0.190 | 0.240 | 0.076 | 0.226 |

## Table S5. Difference between the two farm draws

Absolute difference in unseen-farm mAP50 between the two farm combinations drawn for the same held-out farm and budget (k < 7).

| Farms | Images per farm | Median over held-out farms | Maximum |
|---|---|---|---|
| 1 | 15 | 0.025 | 0.104 |
| 1 | 50 | 0.029 | 0.172 |
| 1 | all | 0.057 | 0.213 |
| 2 | 15 | 0.023 | 0.066 |
| 2 | 50 | 0.036 | 0.082 |
| 2 | all | 0.054 | 0.100 |
| 4 | 15 | 0.021 | 0.068 |
| 4 | 50 | 0.043 | 0.080 |
| 4 | all | 0.018 | 0.048 |

## Table S6. Step-wise paired contrasts in the data-budget experiment

Change in unseen-farm mAP50 for one step in the number of training farms (at fixed images per farm) or in images per farm (at fixed farms), paired by held-out farm; 90% bootstrap interval over held-out farms, and the number of farms on which the step helped.

| Step | At | Change [90% CI] | Farms improved |
|---|---|---|---|
| farms 1 → 2 | 15 images per farm | 0.023 [0.003, 0.044] | 6 of 8 |
| farms 2 → 4 | 15 images per farm | 0.018 [−0.001, 0.038] | 5 of 8 |
| farms 4 → 7 | 15 images per farm | 0.057 [0.016, 0.104] | 6 of 8 |
| farms 1 → 2 | 50 images per farm | 0.036 [0.015, 0.061] | 7 of 8 |
| farms 2 → 4 | 50 images per farm | 0.020 [0.008, 0.034] | 7 of 8 |
| farms 4 → 7 | 50 images per farm | 0.080 [0.056, 0.099] | 7 of 8 |
| farms 1 → 2 | all images per farm | 0.058 [0.010, 0.100] | 7 of 8 |
| farms 2 → 4 | all images per farm | 0.011 [−0.007, 0.030] | 5 of 8 |
| farms 4 → 7 | all images per farm | 0.082 [0.055, 0.113] | 8 of 8 |
| images per farm 15 → 50 | 1 farms | 0.018 [0.009, 0.028] | 7 of 8 |
| images per farm 50 → all | 1 farms | 0.031 [0.012, 0.053] | 7 of 8 |
| images per farm 15 → 50 | 2 farms | 0.031 [0.017, 0.045] | 7 of 8 |
| images per farm 50 → all | 2 farms | 0.053 [0.032, 0.083] | 8 of 8 |
| images per farm 15 → 50 | 4 farms | 0.033 [0.018, 0.048] | 7 of 8 |
| images per farm 50 → all | 4 farms | 0.043 [0.025, 0.069] | 8 of 8 |
| images per farm 15 → 50 | 7 farms | 0.056 [0.026, 0.086] | 7 of 8 |
| images per farm 50 → all | 7 farms | 0.046 [0.011, 0.092] | 6 of 8 |

## Table S7. Earlier, epoch-capped schedule versus fixed iterations

The same farm draws and image subsets trained under two schedules (seed 42 in both). Capped: epochs = clamp(round(2000 × b / n), 100, 300) with batch b = min(32, n), the Ultralytics nominal batch of 64 (so gradients were accumulated over more batches, and fewer optimiser steps taken, for small n), the training list not repeated, and mosaic switched off for the last 10 epochs. Fixed iterations (the paper): batch 32, the list repeated so that an epoch has at least 20 batches, about 2,000 iterations, and mosaic switched off for the last 10% of epochs. Mean unseen-farm mAP50 over held-out farms.

| Farms | Images per farm | Iterations (capped) | mAP50 (capped) | mAP50 (fixed iterations) |
|---|---|---|---|---|
| 1 | 15 | 300–300 | 0.035 | 0.029 |
| 1 | 50 | 600–600 | 0.055 | 0.045 |
| 1 | all | 600–1500 | 0.082 | 0.074 |
| 2 | 15 | 300–300 | 0.070 | 0.050 |
| 2 | 50 | 1200–1200 | 0.104 | 0.087 |
| 2 | all | 1200–2168 | 0.148 | 0.141 |
| 4 | 15 | 600–600 | 0.086 | 0.068 |
| 4 | 50 | 2100–2100 | 0.116 | 0.108 |
| 4 | all | 2002–2145 | 0.150 | 0.149 |
| 7 | 15 | 1200–1200 | 0.132 | 0.119 |
| 7 | 50 | 2013–2013 | 0.206 | 0.181 |
| 7 | all | 2200–2500 | 0.230 | 0.235 |

## Note S2. Farm attribution and the filename error

Farm identity was first joined to annotated images by filename. Two field visits produced overlapping camera counters, and a resize step dropped the suffixes that had distinguished them, so 50 images of the 560-image pool then in use (8.9%) carried another photograph's metadata; 49 of them were labelled farm 0 and had been taken at farm 6. Every annotated image was therefore re-matched to its original photograph by a 64-bit difference hash and assigned that original's farm, accepted only when no original from a different farm lay within three bits. On the 570 images whose filenames still matched an original, the hash recovered that same original (median Hamming distance 0); it also recovered originals for 303 images whose filenames the annotation platform had replaced. Because some originals themselves carry neither a location nor a located image within 30 minutes, 829 of the 1,033 annotated peninsular images could be assigned a farm. The 204 that could not be over-represent one class (56% of boxes against 41%) and under-represent another (4% against 17%), and are excluded from every analysis.
