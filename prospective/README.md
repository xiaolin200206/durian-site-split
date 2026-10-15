# Prospective test: the Seremban farm

A ninth peninsular durian farm, near Seremban (Negeri Sembilan), photographed on
11 October 2026 — after every analysis in the manuscript was complete and written
up — and scored with the 200 frozen leave-one-farm-out models against four
predictions locked before scoring. All four held (manuscript Sections 2.9 and
3.8, Table 7; Supplementary Tables S8–S9 and Note S4).

## Data

`seremban_640/` — 82 images (`images/test/`), YOLO-format labels
(`labels/test/`) and `data.yaml`. Six classes in the training order
(0 Algal, 1 Leaf_rot, 2 Phomopsis, 3 Psyllid, 4 Psyllid_damage,
5 leaf_hopper_damage); 810 boxes; 3 images without boxes. Filenames encode the
tree (`o1t1`–`o1t7`) and the original photo number.

105 photographs were taken, by the same person, phone (iPhone 16 Pro Max) and
conditions as the main collection, during a field test of the handheld device.
23 were set aside by the author before annotation because their symptoms were
too ambiguous to label with confidence (visual judgement only; no model output
was viewed). This can only raise the farm's score and is recorded in the lock
file. Images are Roboflow exports (auto-orient, stretch to 640 × 640) and carry
no EXIF or location data. Boxes of one class outside the six (scale insect)
were removed; the images were kept.

**Licence for the images and annotations: CC BY-NC 4.0**, the same as the main
dataset on Zenodo (https://doi.org/10.5281/zenodo.22030623). The repository
LICENSE applies to the code only.

## Procedure and files

| File | What it is |
|---|---|
| `prospective_seremban.py` | `prepare` → `lock` → `score`, as described in Note S4 |
| `2_lock.sh`, `3_score.sh` | the exact invocations used |
| `prospective_lock.json` | hashes of the 200 weight files, test-set fingerprint, the four predictions and thresholds, the QC note |
| `prospective_per_model.csv` | 200 rows: family, fold, seed, mAP50, mAP50-95, per-class AP50 |
| `prospective_result.txt` | the printed verdict |
| `score.log` | full console log of the scoring run |

Lock file SHA-256, recorded before scoring:
`41060961225e046373774dbece48f1e74e97c4592394718df45bf670fad66869`
(locked 2026-10-15 07:56 local time; scored 08:02; newest weight file 2026-09-14; newest weight file 2026-09-14; Ultralytics 8.4.138,
PyTorch 2.8.0, one RTX 3080 Ti).

## Results

| Detector | Random split, pooled | Unseen farm (8 held-out farms) | Seremban | Range over the 8 model sets |
|---|---|---|---|---|
| YOLO11n (deployed) | 0.428 | 0.223 | 0.298 | 0.250–0.341 |
| YOLO11s | 0.462 | 0.214 | 0.300 | 0.255–0.342 |
| YOLO11m | 0.465 | 0.233 | 0.312 | 0.262–0.355 |
| YOLO11l | 0.472 | 0.236 | 0.339 | 0.292–0.381 |
| RT-DETR-L | 0.499 | 0.221 | 0.329 | 0.301–0.357 |

| Prediction (locked before scoring) | Threshold | Observed | Held |
|---|---|---|---|
| P1 YOLO11n nearer its unseen-farm figure than its random-split figure | < 0.3255 | 0.298 | yes |
| P2 YOLO11n within the range of the eight held-out farms | 0.094–0.356 | 0.298 | yes |
| P3 spread of the five detectors' means | ≤ 0.05 | 0.041 | yes |
| P4 a psyllid class among the two lowest classes for YOLO11n | — | psyllid damage lowest (0.066) | yes |

Reproduce the summary and Figure 2 from the repository root:

```bash
python scripts/applied/prospective_summary.py     # -> results_applied/prospective_seremban_summary.json
python scripts/applied/fig2_prospective.py        # -> results_applied/fig2_new_farm_cost.{png,pdf}
```
