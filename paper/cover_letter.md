Dear Editor,

I am pleased to submit the manuscript "What a new orchard costs: farm-level generalisation and data budgets for durian disease and pest detection" for consideration as an Original Research Paper in *Computers and Electronics in Agriculture*.

Detectors for crop diseases and pests are commonly evaluated on a random split of images from the farms they were trained on, while the growers who use them farm elsewhere. Using 827 annotated images from eight commercial durian farms in Peninsular Malaysia, each tagged with its farm, and 281 from two orchards in Sabah, the manuscript measures what that difference costs, how a data-collection budget should be spent to reduce it, and how much a few images from a new farm recover. Its main findings are:

- With each held-out farm excluded from training, early stopping and checkpoint selection, six detectors from three architecture families scored 1.6–2.3 times lower mAP50 on unseen farms than on a random split. The held-out farm, not the detector, determined most of the score: averaged over the same farms, no two detectors differed by more than 0.022 mAP50.
- Psyllid and psyllid damage transferred worst, retaining 25–30% of their random-split AP on unseen farms, and size alone does not explain it.
- In 336 training runs with training length held constant, more farms and more images per farm both raised unseen-farm accuracy, by amounts that could not be distinguished (0.038 and 0.028 mAP50 per doubling); part of the value of a farm lay in the classes it carried.
- With the fine-tuning recipe tested, fine-tuning a trained detector on a few images from a new farm made it worse on that farm, whereas adding about 50 such images to the training set and retraining improved it; a new farm's images are better used as additional training data.

The work extends to orchard disease and pest detection a question this journal has published for weed detection in arable fields (Ruigrok et al., 2023, *Comput. Electron. Agric.* 204, 107554), and it gives practitioners direct recommendations for evaluation and for planning collection campaigns.

The images, annotations, farm identifiers and split manifests are deposited on Zenodo, and the code, result tables and a script that recomputes every number in the manuscript are public on GitHub.

The peninsular images derive from the same collection as a separate manuscript on capture-session leakage in image classification, which is under review elsewhere and differs in task, unit, annotation and analysis; this is described in the manuscript. The leave-one-farm-out runs reported in Sections 3.1–3.3 were also analysed, together with three non-agricultural datasets, in a broader methodological manuscript that I withdrew from another journal before review; the present manuscript replaces it for the durian results and is not under consideration elsewhere. I am developing a handheld decision-support device for durian growers, and this interest is declared in the manuscript.

Thank you for considering this submission.

Yours sincerely,

Lin Ding Shan
Faculty of Computer Science (Data Science), UCSI University, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my · ORCID 0009-0009-6031-8479
