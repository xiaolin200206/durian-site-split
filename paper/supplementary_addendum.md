# Supplementary material: addendum for the prospective test

*To be appended to the existing supplementary file (Tables S1--S7, Notes S1--S3).*

## Table S8. Prospective test on the Seremban farm: mAP50 of each detector by held-out-farm model set

Each cell is the mean over the five seeds of the leave-one-farm-out models that were trained without the farm named in the first column (Section 2.5), evaluated on the 82 Seremban images. The last rows give the mean and standard deviation over the eight model sets, the mean within-set standard deviation over seeds, and the mean mAP50-95 aggregated the same way. Per-model scores (200 rows, with mAP50-95 and per-class AP50) are in the code repository.

| Held-out farm of the model set | YOLO11n | YOLO11s | YOLO11m | YOLO11l | RT-DETR-L |
|---|---|---|---|---|---|
| 0 | 0.250 | 0.255 | 0.262 | 0.292 | 0.323 |
| 1 | 0.341 | 0.339 | 0.355 | 0.381 | 0.357 |
| 2 | 0.295 | 0.297 | 0.329 | 0.352 | 0.321 |
| 3 | 0.310 | 0.301 | 0.322 | 0.337 | 0.316 |
| 4 | 0.276 | 0.293 | 0.308 | 0.332 | 0.350 |
| 5 | 0.277 | 0.275 | 0.281 | 0.307 | 0.301 |
| 6 | 0.313 | 0.299 | 0.300 | 0.360 | 0.320 |
| 7 | 0.320 | 0.342 | 0.338 | 0.348 | 0.343 |
| Mean over the eight sets | 0.298 | 0.300 | 0.312 | 0.339 | 0.329 |
| SD over the eight sets | 0.027 | 0.027 | 0.029 | 0.027 | 0.018 |
| Mean within-set SD over five seeds | 0.020 | 0.015 | 0.016 | 0.018 | 0.020 |
| Mean mAP50-95 over the eight sets | 0.168 | 0.171 | 0.180 | 0.193 | 0.193 |

## Table S9. Prospective test: per-class AP50 of YOLO11n on the Seremban farm

Averaged over seeds within each model set and then over the eight model sets. The unseen-farm column is the YOLO11n value from Table S2 (mean over the held-out farms that carry the class).

| Class | Images on Seremban | Boxes | Seremban AP50 | Unseen-farm AP50 (eight farms) |
|---|---|---|---|---|
| Algal spot | 36 | 410 | 0.568 | 0.254 |
| Leaf rot | 20 | 44 | 0.340 | 0.274 |
| *Phomopsis* | 24 | 130 | 0.182 | 0.217 |
| Psyllid | 5 | 35 | 0.245 | 0.099 |
| Psyllid damage | 18 | 178 | 0.066 | 0.055 |
| Leafhopper damage | 13 | 13 | 0.387 | 0.371 |

## Note S4. Lock file and procedure of the prospective test

The prospective test was run in three steps by one script (`prospective_seremban.py` in the code repository): *prepare*, which merged the per-class annotation exports of the Seremban farm into one six-class test set, mapping class names to the training classes and dropping boxes of any class outside the six (one such class, scale insect, occurred; its boxes were removed and the images kept); *lock*; and *score*.

The lock step wrote `prospective_lock.json` on 2026-10-15 at 07:56 local time. It records: the SHA-256 hash, path and modification time of each of the 200 weight files (`best.pt` of the clean-protocol leave-one-farm-out runs of YOLO11n, YOLO11s, YOLO11m, YOLO11l and RT-DETR-L, eight held-out farms × five seeds; the newest weight file predates the Seremban visit); a SHA-256 fingerprint over the names and contents of the 82 images and 82 label files; the class list; the metric definition; the paper values the predictions refer to (Table 2 and the per-farm range of Section 3.2); the four predictions and the P1 threshold (0.3255, the midpoint of 0.223 and 0.428); and a note stating that photographs whose symptoms were too ambiguous to label confidently had been set aside by the author before annotation, 82 being retained, by visual judgement only, that this favours a higher mAP50 and therefore works against P1 (the number taken, 105, is from the camera roll and is not in the lock file). The newest of the 200 weight files is dated 14 September 2026, four weeks before the Seremban visit. The photographs were taken during a field test of the handheld device, by the same person and phone as the main collection. The lock file's SHA-256 (`41060961225e046373774dbece48f1e74e97c4592394718df45bf670fad66869`) was recorded outside the analysis machine before scoring.

The score step, run at 08:02 the same morning, first recomputed the fingerprint of the test set and the hash of every weight file and would have stopped on any difference; it then evaluated each model once with Ultralytics 8.4.138 (`model.val`, image size 640, batch 16, default confidence and IoU thresholds; PyTorch 2.8.0, one NVIDIA RTX 3080 Ti), wrote the per-model table, aggregated seeds within model set and then model sets, and printed the verdict on each prediction against the locked threshold. No parameter of the scoring was changed after the lock, and the test set was scored once.

Two caveats are inherent in the design. The predictions were set at a coarseness that eight farms can support, so a pass is weak evidence for the exact values of Table 2 and strong evidence only against the random-split figure as a forecast. And setting aside 23 of the 105 photographs before annotation, although it involved no model, selected the clearer images of the farm, so the Seremban score should be read as an upper estimate for that farm; the lock note records this.
