# Supplementary material

**What a new orchard costs: farm-level generalisation and data budgets for on-device durian disease and pest detection**

Lin Ding Shan

## Note S1. Training details

**Leave-one-farm-out and random-split runs (Sections 2.4–2.5).** Ultralytics 8.x. YOLO11n, YOLO11s, YOLO11m, YOLO11l and RT-DETR-L: COCO-pretrained, 640 px, up to 150 epochs, patience 50, batch 32, 32, 16, 8 and 24 respectively, five seeds (42, 1, 2, 3, 4), default augmentation. Faster R-CNN ResNet-50 FPN (torchvision, improved recipe): COCO-pretrained, SGD with cosine annealing, 40 epochs, patience 12, batch 4, three seeds. The inner validation set is 10% of each fold's training images, stratified by each image's dominant class (at least one image per class with two or more images), drawn with seed 20260911 independently of the training seed; the held-out farm is evaluated once with the weights selected on the inner set.

**Data-budget runs (Section 2.6).** YOLO11n, COCO-pretrained, 640 px, batch min(32, n), no validation during training, no early stopping, final weights evaluated. Epochs = round(2000 × batch / n) bounded to [100, 300], where n is the number of training images; the optimiser was selected automatically by Ultralytics (AdamW, learning rate 0.001, for every run in this experiment). Farm combinations and photograph subsets were drawn with a fixed design seed (20260928); within a combination the 15-, 50- and all-photograph subsets are nested prefixes of one shuffled order per farm.

## Table S1. Unseen-farm mAP50 of each farm under each detector

Clean leave-one-farm-out protocol; mean over seeds.

| Held-out farm | YOLO11n | YOLO11s | YOLO11m | YOLO11l | RT-DETR-L | Faster R-CNN R50-FPN | Mean |
|---|---|---|---|---|---|---|---|
| Farm 0 | 0.188 | 0.209 | 0.195 | 0.192 | 0.212 | 0.176 | 0.195 |
| Farm 1 | 0.398 | 0.393 | 0.389 | 0.406 | 0.257 | 0.295 | 0.356 |
| Farm 2 | 0.113 | 0.104 | 0.115 | 0.102 | 0.133 | 0.139 | 0.118 |
| Farm 3 | 0.340 | 0.299 | 0.301 | 0.303 | 0.282 | 0.275 | 0.300 |
| Farm 4 | 0.185 | 0.227 | 0.273 | 0.270 | 0.277 | 0.283 | 0.253 |
| Farm 5 | 0.235 | 0.224 | 0.260 | 0.295 | 0.282 | 0.272 | 0.261 |
| Farm 6 | 0.093 | 0.058 | 0.097 | 0.079 | 0.100 | 0.135 | 0.094 |
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

Data-budget experiment, all photographs per farm, YOLO11n; held-out farms weighted equally.

| Class | 1 farm | 2 farms | 4 farms | 7 farms |
|---|---|---|---|---|
| Algal spot | 0.079 | 0.157 | 0.216 | 0.245 |
| Leaf rot | 0.086 | 0.181 | 0.172 | 0.277 |
| *Phomopsis* | 0.052 | 0.151 | 0.198 | 0.223 |
| Psyllid | 0.045 | 0.068 | 0.061 | 0.111 |
| Psyllid damage | 0.010 | 0.034 | 0.034 | 0.064 |
| Leafhopper damage | 0.167 | 0.258 | 0.220 | 0.386 |

## Table S4. Data-budget experiment, per held-out farm

YOLO11n unseen-farm mAP50; mean over the two farm draws (one for seven farms) and seeds.

| Farms | Photos per farm | Farm 0 | Farm 1 | Farm 2 | Farm 3 | Farm 4 | Farm 5 | Farm 6 | Farm 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 0.032 | 0.016 | 0.004 | 0.030 | 0.080 | 0.022 | 0.035 | 0.065 |
| 1 | 50 | 0.055 | 0.066 | 0.013 | 0.032 | 0.117 | 0.028 | 0.030 | 0.099 |
| 1 | all | 0.071 | 0.097 | 0.033 | 0.045 | 0.122 | 0.068 | 0.133 | 0.086 |
| 2 | 15 | 0.069 | 0.081 | 0.016 | 0.130 | 0.066 | 0.054 | 0.014 | 0.127 |
| 2 | 50 | 0.097 | 0.150 | 0.029 | 0.190 | 0.113 | 0.107 | 0.025 | 0.121 |
| 2 | all | 0.105 | 0.305 | 0.073 | 0.189 | 0.118 | 0.167 | 0.038 | 0.189 |
| 4 | 15 | 0.083 | 0.101 | 0.028 | 0.078 | 0.098 | 0.148 | 0.044 | 0.110 |
| 4 | 50 | 0.123 | 0.135 | 0.038 | 0.199 | 0.124 | 0.168 | 0.050 | 0.088 |
| 4 | all | 0.136 | 0.273 | 0.047 | 0.187 | 0.136 | 0.222 | 0.056 | 0.143 |
| 7 | 15 | 0.100 | 0.168 | 0.063 | 0.251 | 0.110 | 0.192 | 0.017 | 0.157 |
| 7 | 50 | 0.211 | 0.338 | 0.112 | 0.305 | 0.214 | 0.201 | 0.025 | 0.244 |
| 7 | all | 0.224 | 0.433 | 0.101 | 0.328 | 0.202 | 0.210 | 0.093 | 0.251 |

## Table S5. Difference between the two farm draws

Absolute difference in unseen-farm mAP50 between the two farm combinations drawn for the same held-out farm and budget (k < 7).

| Farms | Photos per farm | Median over held-out farms | Maximum |
|---|---|---|---|
| 1 | 15 | 0.030 | 0.130 |
| 1 | 50 | 0.056 | 0.197 |
| 1 | all | 0.073 | 0.184 |
| 2 | 15 | 0.018 | 0.087 |
| 2 | 50 | 0.028 | 0.120 |
| 2 | all | 0.062 | 0.154 |
| 4 | 15 | 0.024 | 0.097 |
| 4 | 50 | 0.032 | 0.084 |
| 4 | all | 0.031 | 0.058 |

## Note S2. Farm attribution and the filename error

Farm identity was first joined to annotated images by filename. Two field visits produced overlapping camera counters, and a resize step dropped the suffixes that had distinguished them, so 50 images of the 560-image pool then in use (8.9%) carried another photograph's metadata; 49 of them were labelled farm 0 and had been taken at farm 6. Every annotated image was therefore re-matched to its original photograph by a 64-bit difference hash (median Hamming distance 0 on the 570 images whose filenames still matched an original) and assigned that original's farm, accepted only when no original from a different farm lay within three bits. This recovered 303 images whose filenames the annotation platform had replaced and attributed 829 of 1,033 annotated peninsular images. The 204 images that could not be attributed over-represent one class and under-represent another and are excluded from every analysis.
