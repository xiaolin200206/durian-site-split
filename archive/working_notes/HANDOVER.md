# HANDOVER

---

# v6 — 应用版稿件（2026-09-28）。**这一节取代下面的一切。**

- 稿件：`paper/manuscript_template.md` → `python paper/render.py` → `paper/manuscript.md/.docx`。
  文中每个数字都是 `{{json/path:fmt}}` 占位符，表格从 CSV 生成，**不要直接改 manuscript.md**。
- 数字：`python scripts/applied/paper_analysis.py --root .` → `results_applied/paper_numbers.json`、表、图。
- 检查：`python verify_applied.py --full`（CI 每次 push 跑）。
- 数据预算实验目前只有种子 42；种子 1 和新果园校准实验（`new_farm_calibration.py`）结果回来后：
  把新的 `farm_budget.csv` / `new_farm_calibration.csv` 放进 `results_applied/`，
  重跑 paper_analysis → render → render_supp → verify；校准实验要在 template 里加一节。
- 旧的方法论稿件（已撤）在 `archive/withdrawn_methods_paper/`，`verify_claims.py` 仍复现它的 185 条。

# v5 — 应用版重塑（2026-09-28）。**这一节取代下面的一切。**

PR-D-26-13571（Pattern Recognition）已撤。新版只做榴莲，问的是"新果园要付多少代价"。
计划、已有数字、新文章结构：`docs/applied_reshape_plan.md`。

新增 `scripts/applied/`：
- `per_class_transfer.py` —— 不训练，已跑，结果在 `results_applied/per_class_*.csv`
- `farm_budget.py` —— RQ2，更多果园还是更多照片（336 run）
- `new_farm_calibration.py` —— RQ3，新果园拍几张校准（192 run，依赖 RQ2 的 k=7 m=all）
- `analyse_applied.py` —— 两个实验的表和图；用合成数据测过

三个训练脚本在 CPU 上用假图端到端试跑过（build/train/eval 全通，Ultralytics 8.4）。
训练协议：held-out 果园不进任何训练决策，不留内层验证、不早停、固定步数、评 last.pt。

# v4 — 干净协议（2026-09-15）。**这一节取代下面的一切。**

## 一句话

留出单元原本参与了 checkpoint 选择与早停。全部图像模型已在「留出单元不进入
任何训练决策」的协议下重训（412 个 run），主表、方差分解、逐单元分布、模型
比较全部换成干净协议的结果。旧结果保留，因为两者之差本身是一个发现。

## 先跑这个

```bash
python verify_claims.py      # 185/185，CPU，几分钟
```

`REPRODUCE.md` 是给编辑和审稿人的入口，`README.md` 是全貌，
`docs/checking.md` 讲三层检查各抓什么、各漏什么。

## 结论怎么变的

| | 旧协议 | 干净协议 |
|---|---|---|
| 高估幅度 | 3.7–44.3% | **3.7–55.8%** |
| durian 农场分量 | 88.9% | **82.4%**（种子分量从 4.7% 升到 9.8%） |
| GWHD 单元分量 | 62.8% | **68.0%** |
| BreaKHis 单元分量 | 84.8% | **88.1%** |
| 模型比较 | 「跨族稳、同族脆」 | **五个模型在现有单元数上分不出胜负** |

旧协议下 best checkpoint 是在留出单元上挑的，相当于替每个格子去噪，于是
单元份额被抬高、种子份额被压低。干净协议下种子的真实贡献露出来。

**两个只有干净协议才看得见的发现**：高估幅度随架构族变化（两阶段
Faster R-CNN 39.3%，五个一阶段 47.8–55.8%）；GWHD 上两个检测器的排名在
两个协议之间直接翻转。

## 四轮外部审读改了什么

`docs/review_log_2026-09-14.md` 有完整记录。三条实质错误值得记住：

1. **macro F1 当二项均值做噪声分解** —— macro F1 不是比例，方差不是
   p(1−p)/n。改用 accuracy，并加了不假设独立的第二种估计。
2. **池内稳定性与总体推断混用** —— 无放回抽满全池反转率必为零，所以
   「需要 63 个 session 而只有 47 个」在池内定义下不成立。现在分开报。
3. **所需单元数未传播 d 的不确定性** —— (1.645/d)² 的区间跨三个数量级，
   整个换算已撤回，改用配对差 bootstrap 区间是否含零。

这三条 CI 原本都抓不到，因为它们不是「数字算错」而是「主张超出证据范围」。
`cross_audit.py` 的概念表为此扩了五条。

## 当前状态

投 Nature Computational Science，订阅制路径（免费）。正文 3,483 词、摘要
150、6 个 display item，全部在限内。初投不需要特别格式。

`docs/cover_letter.md` 按 NCS 的 cover letter 社论（*Nat Comput Sci* 2, 617）
写，包含代码可得性一段——他们做代码同行评审。Code Ocean 胶囊选 Yes。

**还缺**：伦理判定、reporting summary、推荐审稿人、Zenodo 重新存档。

## 已识别但未做的补强（按代价排序）

1. **逐 item 预测 + 块 bootstrap**（BreaKHis 约 1 GPU-小时，HAR 几分钟 CPU）
   —— 能把噪声结论从条件式变成确定，性价比最高
2. **BreaKHis 重复分组划分**（约 6 GPU-小时）—— 检验区间覆盖率
3. **dose-response 干净协议重跑**（约 30 GPU-小时）—— 已降级为 preliminary
4. **加异构架构到 GWHD/BreaKHis** —— 扩大模型族覆盖

真实审稿意见回来之前不做，免得猜错方向。


Written for whoever picks this up next, including me in three months.
Sections are in reverse chronological order; **v3 below supersedes everything
after it where they conflict.** The older text is kept because the mistakes it
records are part of the paper's argument, and because several of them were
held with confidence before being disproved.

---

# HANDOVER — v3 (11 September 2026)

## Where the project stands

Four datasets, eight architectures, ten dataset-model combinations, 439
training runs. All experiments complete. **322 checks in `verify_claims.py`,
all green.** No further training is planned.

```
Durian     8 farms, 827 images, YOLO11n/s/m + RT-DETR-L, 180 runs
           + per-burst (4 thresholds), per-tree, cross-region,
             dose-response, abstention, focal stratification,
             checkpoint bias (3 of 4 models)
GWHD       47 sessions, 6,510 images, YOLO11s + YOLO11n, 59 runs
           + per-domain, dose-response, checkpoint bias
BreaKHis   81 patients, 7,909 images, YOLO11s-cls + YOLO11n-cls, 100 runs
           + per-patient, checkpoint bias
HAR        30 subjects, 10,299 windows, MLP + random forest, 100 runs
           + per-subject, dose-response
```

Headline: overstatement 3.7-44.3% across ten combinations; held-out sample
88.9% of the variance against 1.1% for architecture on durian, 78.7/0.4 on
BreaKHis, 56.1/14.6 on HAR, reversed on GWHD (seed 61.2%); per-unit ranges
wide in all four; adding evaluation units narrows the interval without moving
the mean.

## What changed in v3

**BreaKHis checkpoint bias measured.** 100 `last.pt` re-validated, no
retraining. item-level moves +0.0010 / +0.0012; unit-level +0.0247 / +0.0292.
Overstatement 8.0% -> 10.4% and 8.7% -> 11.6%. Under the selection-free
protocol the error rate goes 0.82% -> 12.29%, a factor of 15.0.

With durian (3 models) and GWHD (2 models) this makes seven dataset-model
combinations, all moving the same way, with the size of the selection effect
tracking the size of the partitioning effect: durian +12.3 to +17.3 points,
GWHD +7.7 to +8.9, BreaKHis +2.4 to +2.8. **That is why the nested protocol
was not run.** A reviewer suggested rebuilding every headline experiment with
an outer unit-disjoint test and an inner validation drawn only from training
units. Seven measured combinations already bound how loose the upper bound is;
retraining 339 runs would not add information. This is a decision, not an
omission — see "Decisions taken" below.

**Evaluation-unit dose-response added.** `eval_unit_curve.py`, no GPU. The
existing dose-response varies how many units the *training* data came from;
this varies how many units the *evaluation* is computed on, by resampling the
per-unit tables. The two behave differently: training units may or may not
raise the mean, evaluation units never do. Both narrow dispersion. This gives
the paper a third number to recommend reporting — how many evaluation units a
figure rests on — and it costs nothing to compute for anyone with a unit
identifier.

**Abstention analysis found to be on a different metric.** `abstention_v2.py`
pools all six classes into one AP curve; `mAP50` averages per-class APs. Same
farm, up to 0.31 apart. Confidence statistics are class-independent so the
correlations were recomputed against the correct scores: **r = +0.05, not
-0.26**. The claim changes from "confidence points the wrong way" to
"confidence carries no information", which is weaker in form and harder to
attack, and the worst farm still never abstains. `abstention_v3.py` is written
and **has not been run**; the risk-coverage numbers (0.193 -> 0.242 at 50%
abstention) are still from the pooled computation.

**Manuscript restructured to Nature Machine Intelligence Analysis format**
(`docs/manuscript_nmi_analysis.md`): abstract <=150 words, main text ~2,900,
six display items, Related Work folded into the untitled opening, Limitations
compressed into the Discussion with the full list in Supplementary Note 4.
Nine tables moved to Supplementary. The old `manuscript.md` is unchanged and
still verifies.

**Repository cleaned.** Three `.bak` files and an empty `0.030/` removed;
eight one-time scripts moved to `scripts/oneoff/`; the Nature Sustainability
cover letter renamed `cover_letter_nature_sustainability_OLD.md` because its
numbers (five farms, 0.478 -> 0.246) contradict the current manuscript in
every cell.

## Still to do, in the order I would do it

1. **Run `abstention_v3.py --root . --model rtdetr-l --check`.** Look at the
   last line: the per-farm difference against `in_region.csv` must be under
   0.02 before anything from it goes in the manuscript. Then update the
   risk-coverage sentence.
2. **Extend the confidence analysis to GWHD (47), BreaKHis (81), HAR (30).**
   Inference only, no retraining. This is the single most valuable thing left.
   At n=8 the durian result is an observation; if mean confidence, entropy or
   margin fails to identify the bottom decile across three more disciplines,
   it becomes a finding and Fig. 4 should be redrawn as four panels.
3. **Hierarchical variance decomposition**, units nested within folds. Pure
   computation, the per-unit tables and fold assignments are enough. Until it
   is done, the BreaKHis 78.7% and HAR 56.1% are *held-out fold composition*,
   not unit variance, and the manuscript says so. Durian is unaffected because
   there a fold is one farm.
4. **BreaKHis fold class balance.** Patient folds were assembled by greedy
   bin-packing on item count with no class balancing, so patient identity and
   class composition are confounded in the decomposition. Durian has the
   single-class control; BreaKHis needs the analogue.
5. **Verify references 19, 25, 26.** Ref 19 now cites the Ultralytics
   repository and the RT-DETR CVPR paper rather than a documentation page. Ref
   26 was named by a reviewer from memory and must be located or dropped.
6. **Fill the ethics determination** in the Methods and the cover letter.
7. **Regenerate figures** — Fig. 4 and Fig. 5 both change with items 1 and 2.
8. **Decide Article versus Analysis.** With the nested protocol declined, the
   current manuscript is the Analysis version. See `docs/revision_plan.md`.

## Decisions taken in v3, so they are not silently reversed

- **The nested checkpoint protocol will not be run.** Seven combinations
  measured, all in the same direction, effect size tracking the partitioning
  effect. Every unit-level figure is an upper bound and the amount is
  quantified. Retraining would cost 339 runs and change no conclusion.
- **durian YOLO11s checkpoint bias will not be recovered.** Both `best.pt` and
  `last.pt` are gone from the machine. Recovering it means retraining all 45
  runs *and* recomputing that row of the main table, because new weights are
  not comparable with the old table. Three durian models is enough.
- **HAR needs no checkpoint correction at all** and this should be stated
  rather than left implicit: the perceptron runs a fixed 300 iterations and
  the forest has no early stopping, so no held-out subject ever influenced a
  training decision.
- **The main text reports equal-weight evaluation-unit curves**, instance-
  weighted in Supplementary. Equal weighting is the unbiased quantity and the
  one "how many units" should be based on; instance weighting is what a pooled
  report actually does, so both belong somewhere.
- **`abstention_v3.py` imports `bkh_run.val_once` style reuse wherever
  possible.** The v2 failure came from writing a second scoring path. Any new
  evaluation script should call the existing one rather than reimplement it.
- **The title is now "Evaluation-unit sampling can dominate reported
  machine-learning performance."** "Accuracy" was inaccurate (mAP50 and macro
  F1 are also reported) and "is a property of the evaluation sample" was too
  absolute given GWHD. The word "can" is load-bearing.

## What v3 got wrong along the way

- **The self-check note in `eval_unit_curve.py` was wrong on first release.**
  It said a drifting mean indicates a bad weight column; in fact the drift is
  the arithmetic difference between equal and instance weighting and is
  expected wherever unit size correlates with score. Corrected, and the script
  now prints both curves so the comparison is visible rather than inferred.
- **The BreaKHis k=1 spread was first written down from a 2,000-draw Monte
  Carlo estimate (0.164) rather than the population s.d. (0.175).** On a
  long-tailed distribution — five patients below 0.60 — the estimate is
  biased low. The check now computes the population s.d. exactly and uses it
  as the expected value, which also validates the resampling implementation.

Both were caught by a number failing to match an expectation. Same as
everything else in this file.

---

Written for whoever picks this up next, including me in three months.
The v1 sections below are kept because the mistakes they record are part of
the paper's argument.

> **Superseded by v3 above.** The run counts, check count and pending-table
> list below are from an earlier state and are no longer accurate. Kept for the
> record of decisions and mistakes.

## Where the project stands

Four datasets, all experiments complete. 123 checks in `verify_claims.py`.

```
Durian     8 farms, 827 images, YOLO11s + RT-DETR-L, 90 training runs
           + per-burst (4 thresholds), per-tree, cross-region,
             dose-response, abstention, focal stratification
GWHD       47 sessions, 6,510 images, YOLO11s, 50 runs
           + per-domain, dose-response
BreaKHis   81 patients, 7,909 images, YOLO11s-cls, 50 runs
           + per-patient
HAR        30 subjects, 10,299 windows, MLP, 50 runs
           + per-subject, dose-response
```

Headline: overstatement 44.3 / 41.1 / 17.6 / 8.7 / 3.7 per cent; variance
share of the evaluation unit 86.4 per cent against 4.4 for architecture;
per-unit ranges from near-failure to near-perfect in all four.

## What has to happen before submission

**1. Copy eight tables into this repository.** The checks that report PENDING
are not failures of the analysis; the tables exist but live on the machines
that produced them. Run `python verify_claims.py --todo` for the list. In
short: `results_gwhd/` needs all three of its tables, `results_durian/` needs
`sabah_by_tree.csv`, `sabah_by_orchard.csv`, the three burst-threshold tables
other than 60s/min5, `farm_capture_conditions.csv`,
`farm_focal_composition.csv` and `farm_focal_annotation_scale.csv`.
The verification must be green before the docx is final.

**2. Regenerate the figures once those tables are in.** `make_figures.py`
silently omits a panel or a curve whose table is missing, by design. Figure 1
currently has three panels and should have five; Figure 3 has two curves and
should have three.

**3. Cover letter.** `docs/` still holds a version written for a different
journal and a different framing. The first paragraph should be the facts that
cannot be fabricated: the collection dates and sites, the number of training
runs, the continuous-integration check, and the two errors this work found in
its own pipeline.

**4. Decide the venue.** The manuscript is written to the evaluation-protocol
framing throughout. Reverting it to an agricultural framing means rewriting
the title, abstract, introduction and discussion, and moving GWHD, BreaKHis
and HAR to supplementary; the Methods, the tables and the figures are
unaffected. That is a day's work, not a re-analysis.

## Decisions taken, so they are not silently reversed

- **Evaluation unit is the frame, not the crop.** Durian is the most detailed
  of four datasets, not the subject.
- **Primary burst setting is 60 s with at least five images**, chosen for
  coverage (79 per cent). The coefficient of variation is 0.70 to 0.75 across
  all four settings tested, so the choice does not move the result.
- **The withdrawn stage claim stays visible.** It is stated in Results and in
  Discussion, and `verify_claims.py` fails if the correlation returns above
  0.5 or if the worst farm turns out to have the smallest lesions.
- **The lesion-scale figure is the stratified one (15-fold), not the raw one
  (75-fold).** Both are checked, so the prose cannot quietly revert.
- **The focal-length checks are written at image level.** The v1 checks
  compared per-farm medians, passed, and verified a weaker statement than the
  prose asserted. That is how the confound survived a whole version.
- **Grower consultation is background, not data.** The monetary figures and
  the pending ethics determination were removed. If an ethics determination is
  obtained the section can be restored; do not reinstate the figures without it.
- **Author name is Lin Ding Shan**, matching the earlier classification paper.

## Known gaps, in the order I would close them

1. **Abstention on a second architecture and a second dataset.** Currently
   RT-DETR-L on durian only. The result is negative and load-bearing, so it
   deserves a replication. No retraining needed for the second architecture.
2. **A fifth dataset outside vision and outside wearables.** Text or tabular
   would test whether the shape holds where "a bout" is something else
   entirely. This is the main thing a reviewer can still ask for.
3. **Site-count on BreaKHis.** 81 patients would give the widest k range of
   any of the four, but `site_count.py` is written for detection and needs a
   classification path.
4. **Inter-annotator agreement on durian.** Never measured. For two classes
   we expect it would be poor, and we say so.
5. **The 204 unattributed durian images.** 160 match no original within eight
   bits, of which 25 passed through instant messaging and 14 are video frames.
   Recovering any of them would need the platform's export manifest.

## Things that went wrong, recorded so they are not repeated

- **A filename join misattributed 8.9 per cent of the durian pool** and
  inflated the best fold by a third. Found by content hashing, not inspection.
  Full account in Supplementary Note 1.
- **A focal-length claim was true of per-farm medians and false of images**,
  which inflated a lesion-scale figure five-fold. The check that should have
  caught it was testing the median.
- **`inspect.py` in the project root shadowed the standard library** and broke
  every `import numpy` for a week. The symptom pointed at numpy. `scripts/
  diagnostics/fix_shadowing.py` now exists.
- **Two analysis scripts written during this revision looked up metadata by
  annotated-image stem** and silently dropped the 303 content-matched images:
  the same class of error being fixed. Both now print the pool size against
  the expected 827, which is how it was caught.
- **`per_site_v2.py` output filenames encoded the gap but not the minimum
  burst size**, so a sensitivity run overwrote the primary table. Filenames now
  carry both.
- **An `--append` flag that did not take effect** let one architecture's
  cross-region results overwrite the other's. Recovered from a copy;
  `merge_cross.py` exists to rebuild the merge.

The common thread is that every one of these was found by a number failing to
match an expectation, not by reading code. That is the argument for stating
expected values in the scripts themselves.

---


# HANDOVER — v2 (September 2026)

This section supersedes the numbered items below where they conflict.
The v1 text is kept because the mistakes it records are part of the paper's
argument.

## What v2 is

The v1 analysis pool (560 images, 5 farms, GroupKFold) was built on a
filename join that was wrong for 50 images (8.9%). Farm attribution is now
by perceptual hash against the originals; the pool is 827 images on 8
farms; the design is leave-one-farm-out. All 45 training runs, the
cross-island evaluation and the per-site evaluation have been redone.
`results_v2/` holds the new tables; `results_v1_superseded/` the old ones.

Headline numbers, v1 -> v2:

| | v1 | v2 |
|---|---|---|
| random split mAP50 | 0.478 | 0.494 |
| by-farm mean | 0.246 | 0.275 |
| overstatement | 48.5% | 44.3% |
| fold spread (best/worst) | 2.84x | 4.00x |
| between/within seed ratio | 6.4 | 4.8 |
| Sabah mean | 0.246 | 0.270 |
| Sabah s.d. across configs | 0.009 | 0.009 |
| Leaf_rot lesion-area ratio | 92.8x (unstratified) | 15.0x (within 6.76 mm) |
| stage association r | +0.54 (n=5) | +0.29 (n=7), withdrawn |
| per-burst CV (peninsula) | 0.72 | 0.72 |
| per-tree CV (Sabah) | 0.28 | 0.17 |
| exclusion from labelled set | 46% | 20% |

## Still to do before submission

Marked `[[RERUN]]` in `manuscript.md`. In order:

1. **Re-run `per_site_v2.py --step both --gap 60 --min-images 5`.** The
   version that produced the current burst tables looked up timestamps by
   annotated-image stem only, so the 303 content-matched images (all of
   farms 1, 4, 7) had no timestamp and formed no bursts. The script is
   fixed (timestamps now resolved through `matched_file`). Coverage, CV
   and the per-farm burst table in Results will change; the Fig. 2
   peninsula curve and the 13/52 sample-size figures with them.
   Output filenames now carry `_min{n}` so settings cannot overwrite each
   other; also re-run the 15 s / 30 s / 15 s-min3 settings for the
   sensitivity table.
2. **Fill the per-class Sabah AP50 column** from `results_v2/cross_island.csv`
   (`AP50::<class>` means over the 45 evaluations) and the pest-class farm
   counts from `farm_focal_annotation_scale.csv` (rows with `focal == ALL`
   and `n_images >= 5`). Then re-read the per-class paragraph and the
   Fig. 3 caption: the v1 pattern (classes at all farms do not degrade;
   Phomopsis loses 36%) may not hold and must not be asserted until checked.
3. **Sabah resampling intervals** for Fig. 2 (`aggregation_curve.py` on
   `sabah_by_tree.csv`).
4. **Capture covariates on eight farms.** `capture_conditions_v2.py` has
   written hour / solar / midday columns for all eight; the manuscript
   still cites the five-farm tabulation. Read the CSV, update the sentence
   in "Nothing we recorded predicts which farm will fail".
5. **Regenerate figures** from `results_v2/` (`make_figures.py` still reads
   `results/`).
6. `verify_claims.py` must be green with zero `PENDING` before the docx is
   final.

## Decisions taken in v2 that the author should confirm

- **Grower consultation demoted to informal background.** The v1 Methods
  had a "Field consultation with growers" section with monetary figures
  and a pending ethics determination. v2 removes the monetary figures and
  the interview framing, and describes the consultation as informal
  guidance. If an ethics determination is obtained, the section can be
  restored. If not, do not reinstate the MYR figures or the quoted advice.
- **Primary burst setting is 60 s / >=5.** Chosen for coverage (50%); the
  CV is insensitive to the choice (0.66-0.72 across four settings). The
  15 s setting from v1 covered 25% of the pool.
- **Ninth cluster (2 images) excluded** from both regimes rather than
  folded into training.
- **Withdrawn claim kept visible.** The manuscript says, in Results and
  Discussion, that the stage association was claimed in an earlier version
  and does not hold. `verify_claims.py` has a check that fails if |r| >=
  0.5 or if the worst farm has the smallest lesions.

## Deferred to revision (not blocking)

- Architecture comparison: `colab_run_v2.py --step train --models rtdetr-l yolo11n`
  (2 seeds each, about 4 h).
- `abstention_analysis.py` (no retraining; turns the "system design" section
  from argument into result).
- Site-count experiment (train on 1..7 farms, image count held constant,
  evaluate on Sabah). This is the only thing that would make "transfer
  tracks site coverage" causal rather than observational; the 9.8% Sabah
  gain from a 47% larger pool is currently confounded with the three new
  sites.
- `gps_only_fold*` (40 runs) as a robustness check on time-attributed images.

## Things that went wrong in this pass, for the record

- `inspect.py` in the project root shadowed the standard library and broke
  every `import numpy` for a week; symptom was a `UnicodeEncodeError` on a
  Chinese print, then `AttributeError: cleandoc`. `fix_shadowing.py` now
  exists.
- Two analysis scripts written during v2 (`capture_conditions_v2.py`,
  `per_site_v2.py`) initially looked up metadata by annotated-image stem and
  silently dropped the 303 content-matched images. Same class of error as
  the one being fixed. Both print the pool size against the expected 827
  now, which is how it was caught.
- `per_site_v2.py` output filenames encoded the gap but not `min_images`,
  so a `--min-images 3` run overwrote the `--min-images 5` table.

---

# Handover

Written so the next working session can start without re-deriving anything.
Read the two "conclusions that were withdrawn" sections before touching the
manuscript; both were confidently held and both were wrong.

---

## Where things stand

**Experiments: done.** 30 training runs, 5 seeds × 6 configurations,
188 minutes on an A100. All evaluations complete. Nothing needs retraining
for the paper as written.

**Manuscript: revised and verified.** `manuscript.md` and `manuscript.docx`
contain the full text. Related Work, the three figures and their captions,
the Figure 2 Results subsection, Supplementary Table 1, references and the
front-matter sections are all inserted. All 113 quantitative claims are
checked against the tables by `verify_claims.py`, which passes.

Outstanding on the text: main text is ~6,100 words against a 5,000 limit,
and the references marked VERIFY have not been checked against publisher
records.

**Figures: generated.** Three main figures in `figures/`, as PDF and PNG,
plus `suppfig1_stage.png` — a four-panel trunk-disease stage sequence from
the Sabah site (active, resolved x2, dead). Trunk disease is not among the
six analysed classes and those images are not in the dataset; the panel is
there because it shows what stage looks like when a grower can follow one
individual over time, which the foliar labels cannot. Captions in
`docs/figure_captions.md`.

**Practitioner material is now used, and the Methods wording changed with
it.** The consultation section previously said the input "was not collected
or analysed as qualitative research data". The paper now cites grower
estimates of cost and loss and reads active ingredients off photographed
product labels, so that sentence was replaced with an accurate one:
statements are reported as the growers' own accounts, corroborated by
labels where the claim concerns chemistry, and monetary figures are
estimates rather than measurements. The strongest single addition is not a
quote at all — scale and mealybug receive products in different IRAC
mode-of-action groups (4A/7C against 1A), which establishes the limit of
the treatment-equivalence merging rule from the registration record instead
of from anyone's opinion.

**Repository: complete.** `verify_claims.py` runs in CI on every push.

---

## Outstanding, in priority order

### Blocking submission

1. **~~Cut ~1,100 words~~ — done.** Main text is 5,105 (Nature counts main
   text excluding abstract, Methods, references and captions). Nothing was
   deleted. Roughly half came from compressing prose and half from moving
   the full Limitations list into Methods, where Nature format allows the
   detail; a condensed "What the results do not license" carrying the three
   load-bearing caveats stays at the end of the Discussion. One block of
   genuine duplication was removed: the three alternative explanations were
   stated at length in both Results and Methods.

2. **Verify the references.** Entries marked `VERIFY` in the .bib were
   located by search and their bibliographic fields reconstructed. Check
   every one against the publisher record. The two Vietnamese durian
   datasets are now verified and their entries corrected — the .bib
   previously collapsed two different datasets into one key.

3. **Ethics determination.** A request has been sent to the Institute's
   Head of Research and the Director asking whether the grower consultation
   required IEC review, and requesting a written determination either way.
   The Methods section carries a bracketed placeholder for the result.
   Written consent from both growers is held (WhatsApp, timestamped;
   redact the phone numbers before it goes anywhere).

4. **Decide the Introduction's deployment framing.** It currently refers to
   "several Southeast Asian governments" without citation. The specific
   figures found by search — a ten-week pilot with 42 farmers in Perak
   scaling toward ~110,000 growers — are unverified and naming a live
   national programme carries risk. Either verify from a primary source or
   keep the framing general.

5. **Fill the placeholders**: `[DOI]`, `[repository]`, `[email]`, `[ORCID]`,
   in the manuscript, the cover letter and the README.

### Strengthens the paper materially

6. **~~Complete the dataset survey~~ — the two durian rows are done.** Both
   Vietnamese datasets were downloaded and inspected. Neither releases a
   site identifier, so the analysis cannot be repeated on independent data
   and the paper's external validity gap stands. What was gained instead is
   the strongest row in Supplementary Table 1: the Vinh Long release is
   described as raw iPhone photographs but ships 5,274 RGBA PNGs whose XMP
   records `exif:UserComment = Screenshot`, with no camera model, timestamp
   or GPS in any sampled file, plus 177 JPEGs including files named at a
   224x224 resolution. Describe the files; do not speculate about how they
   came to be that way. Eight rows of the survey remain UNVERIFIED.

7. **The training-site-count experiment.** The claim that transfer tracks
   site coverage currently rests on a correlation across six classes. Train
   on 1, 2, 3, 4 farms — several combinations each — and evaluate all on
   Sabah, holding image count roughly constant. If Sabah performance rises
   with farm count but not with image count, the claim becomes causal.
   Roughly 12–16 runs, two hours on the same hardware.

### Optional

8. **The 473 excluded images.** Mode B (keeping them in training) was never
   run on the current taxonomy. It would quantify how much the exclusion
   costs. Low priority; the limitation is stated honestly as it stands.

9. **Leaf_rot stage ablation.** Splitting the class by lesion area was run on
   the *previous* taxonomy and those results are void. Re-running is cheap
   but the finding is already carried by the 93-fold size table without it.

---

## Two conclusions that were withdrawn

Both are recorded because each was held with confidence, acted on, and
turned out to be wrong. If either resurfaces in a later session, this is
why it should not.

### "Random splitting overstates accuracy by 68%"

Held after a single-seed run in which fold 0 scored 0.121. Five seeds later,
fold 0 scored 0.518 ± 0.010 and fold 4 scored 0.773, *above* the random
split. The 68% figure was an artifact of one run, and the framing built on
it — that random splitting produces a uniform inflation — does not survive.

What replaced it: the inflation is real (48.5% on the current taxonomy) but
secondary. The finding is that the corrected figure is itself unstable
across folds, which no amount of seed averaging fixes because the variation
is between sites, not between runs.

**The lesson that generalises:** every headline number in this project
changed when it was run five times. Nothing should be built on a single run
again.

### "The Leaf_rot size table is 96/21/109/45/124 px"

It was not. `farm_covariates.py` gives 83/10/55/45/96. Four of the five
cells in the manuscript were wrong and every other row of the same table
was right, so the row was carried forward from a superseded run and nothing
caught it — because `verify_claims.py` had no check on annotation scale at
all. The 92.8-fold claim sat in the Results, the Discussion and the
Limitations for the life of the project without a single test behind it.

The wrong numbers were also internally inconsistent in a way that was
visible without any data: 124/21 is a 5.9-fold difference in side, which is
35-fold in area, not 93. The correct numbers are consistent — 96/10 is 9.6
in side, 92.2 in area — so the arithmetic check that would have caught it
costs one line and is now in the script.

The correction strengthens the sentence it appears in. A detector trained
on 45-96 px targets and evaluated on **10** px targets is a starker claim
than the same sentence with 21 px.

**The lesson that generalises:** a number that no script checks is a number
that is not verified, whatever the CI badge says. 73 of 73 passing meant 73
of the claims that had been written down, not 73 of the claims in the paper.

### "Farm 2 fails because it was photographed at midday"

Held because farm 2 is 99% midday and is the worst fold. Farms 3 and 5 are
100% midday, farm 5 at a higher solar elevation, and both score better; farm
6 is entirely pre-09:00 and also scores poorly. No monotone relationship.

What replaced it: farm 2's Leaf_rot lesions are an order of magnitude
smaller than everywhere else, because they are early-stage. That survived
three attempts to break it, including one summary statistic that appeared to
support a camera-distance explanation and did not survive looking at the
photographs. The statistic was measuring class composition; it is kept in
`scripts/diagnostics/check_lesion_scale.py` and described in the manuscript
for that reason.

---

## Decisions and why

**Six classes, not twelve.** The paper is a standalone study and does not
reference the author's other work, so it does not need the earlier label
space. Sabah carries only these six, and a shared label space across regions
is a precondition for the central comparison.

**Psyllid merged from four classes to two.** A grower confirmed that a
single egg and a cluster trigger the same application. Treatment equivalence
is the paper's stated organising principle, so they merge.

**Phomopsis and Leaf_rot not merged**, although a grower said both get the
same fungicide and both are visually inseparable once spread. Treatment
equivalence could not be established with the same confidence, and merging
would have destroyed the 93-fold size finding by mixing two size regimes
into one class.

**Mode A, not mode B.** 473 peninsular images have no recoverable location.
Keeping them would make the random and by-farm regimes differ in training
set size as well as in split rule, confounding the one comparison the paper
exists to make.

**mAP50 as the primary metric.** A missed lesion costs a grower more than an
imprecise box, and mAP50-95 penalises sub-pixel offsets on targets a few
pixels across. mAP50-95 is reported alongside throughout.

**Single-egg psyllid images withdrawn from annotation.** During quality
control they required magnification far beyond normal review scale and were
not separable from leaf reflections at that magnification. The photographs
are retained; only the annotations were withdrawn. This is stated in
Limitations because the withdrawal was a judgement.

---

## Venue

Submitted to Nature Sustainability, with a stated reservation. The work is a
measurement study about evaluation protocol; NS publishes sustainability
science. The bridge is that the decision it bears on — public investment in
agricultural AI, justified by expected reductions in loss and pesticide use —
is a sustainability-policy decision. That bridge is real and thin, and the
cover letter says so and asks for an early assessment of fit.

If declined, the content needs no change. Only the cover letter does.
Reasonable alternatives, in order: *Computers and Electronics in
Agriculture*, *Plant Phenomics* (which published GWHD and would recognise
the argument immediately), *Precision Agriculture*, or a
Datasets-and-Benchmarks track.

---

## Files

```
manuscript.md / .docx        the paper; docx has line numbers for review
cover_letter.md / .docx
verify_claims.py             73 assertions, run before every edit lands
results/                     8 tables; the paper's entire evidence base
figures/                     3 figures, PDF and PNG
docs/related_work.md         section to insert, plus the two edits it forces
docs/references.bib          with VERIFY flags
docs/figure_captions.md      captions and the new Results subsection
docs/dataset_provenance_survey.md   Table S1, half-filled, with a protocol
scripts/prepare/             corpus to dataset
scripts/analyse/             the experiments and the per-site evaluations
scripts/diagnostics/         the three checks that changed what the paper says
```

Run `python verify_claims.py` after any edit that touches a number. It
exists because a figure was wrong once — the per-burst minimum was stated as
0.004 when one burst scores exactly zero — and the script caught it.
