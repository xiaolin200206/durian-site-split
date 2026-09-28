<!-- DRAFT: re-check the third bullet against the v2 results; delete this line when done -->
Dear Editor,

I am pleased to submit the manuscript "What a new orchard costs: farm-level generalisation and data budgets for durian disease and pest detection" for consideration as an Original Research Paper in *Computers and Electronics in Agriculture*.

Detectors for crop diseases and pests are commonly evaluated on a random split of images from the farms they were trained on, while the growers who use them farm elsewhere. Using {{farms/images}} annotated images from {{farms/n_farms:w}} commercial durian farms in Peninsular Malaysia, each tagged with its farm, and {{sabah_images}} from two orchards in Sabah, the manuscript measures what that difference costs, how a data-collection budget should be spent to reduce it, and how much a few images from a new farm recover. Its main findings are:

- With each held-out farm excluded from training, early stopping and checkpoint selection, six detectors from three architecture families scored {{headline_range/ratio_min:.1f}}–{{headline_range/ratio_max:.1f}} times lower mAP50 on unseen farms than on a random split. The held-out farm, not the detector, determined most of the score: averaged over the same farms, no two detectors differed by more than {{model_pairs/max_abs_mean_diff:.3f}} mAP50.
- Psyllid and psyllid damage transferred worst, retaining {{per_class_summary/psyllid_retained_range/0:.0f}}–{{per_class_summary/psyllid_retained_range/1:.0f}}% of their random-split AP on unseen farms, and size alone does not explain it.
- In {{rq2_design/runs}} training runs with training length held constant, more farms and more images per farm both raised unseen-farm accuracy ({{regression/all runs/farms:.3f}} and {{regression/all runs/photos:.3f}} mAP50 per doubling); part of the value of a farm lay in the classes it carried. {{abstract_rq3_sentence}}

The work extends to orchard disease and pest detection a question this journal has published for weed detection in arable fields (Ruigrok et al., 2023, *Comput. Electron. Agric.* 204, 107554), and it gives practitioners direct recommendations for evaluation and for planning collection campaigns.

The images, annotations, farm identifiers and split manifests are deposited on Zenodo, and the code, result tables and a script that recomputes every number in the manuscript are public on GitHub.

The peninsular images derive from the same collection as a separate manuscript on capture-session leakage in image classification, which is under review elsewhere and differs in task, unit, annotation and analysis; this is described in the manuscript. The leave-one-farm-out runs reported in Sections 3.1–3.3 were also analysed, together with three non-agricultural datasets, in a broader methodological manuscript that I withdrew from another journal before review; the present manuscript replaces it for the durian results and is not under consideration elsewhere. I am developing a handheld decision-support device for durian growers, and this interest is declared in the manuscript.

Thank you for considering this submission.

Yours sincerely,

Lin Ding Shan
Faculty of Computer Science (Data Science), UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my · ORCID 0009-0009-6031-8479
