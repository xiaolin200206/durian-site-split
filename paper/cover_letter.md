Dear Editor,

I am pleased to submit the manuscript "What a new orchard costs: farm-level generalisation and data budgets for durian disease and pest detection" for consideration as an original research article in *[JOURNAL]*.

Detectors for crop diseases and pests are usually evaluated on a random split of images from the farms they were trained on, while the growers who use them farm elsewhere. Using 827 lesion-annotated images from eight commercial durian farms in Peninsular Malaysia, each tagged with its farm, and 281 from two orchards in Sabah, the manuscript measures what that difference costs, how a data-collection budget should be spent to reduce it, and how a few labelled images from a new farm should be used. It then tests the farm-level figures prospectively on a ninth farm photographed after the analyses were complete. Its main findings are:

- With each held-out farm excluded from training, early stopping and checkpoint selection, six detectors from three architecture families scored 1.6–2.3 times higher mAP50 on a random split than on unseen farms. The held-out farm determined far more of the score than the detector: averaged over the same farms, no two detectors differed by more than 0.022 mAP50, whereas farms spanned 0.094–0.356.
- Psyllid and psyllid damage transferred worst, retaining 25–30% of their random-split AP on unseen farms.
- In 336 training runs with training length held constant, more farms and more images per farm both raised unseen-farm accuracy, by amounts that could not be distinguished (0.038 and 0.028 mAP50 per doubling); part of the value of a farm lay in the classes it carried.
- With the fine-tuning recipe tested, fine-tuning a trained detector on a few images from a new farm made it worse on that farm, whereas adding about 50 such images to the training set and retraining improved it.
- On the ninth farm, scored with the frozen models against four predictions whose thresholds, together with the hashes of every model and test image, were locked before scoring, all four predictions held: the deployed detector scored 0.298 mAP50, inside the held-out-farm range and far below its random-split figure of 0.428.

[FIT PARAGRAPH — keep one:]

[CEA] The work extends to orchard disease and pest detection a question this journal has published for weed detection in arable fields (Ruigrok et al., 2023, *Comput. Electron. Agric.* 204, 107554) and for dataset size and variety in plant disease classification (Barbedo, 2018, *Comput. Electron. Agric.* 153, 46–53), and it gives practitioners direct recommendations for evaluation and for planning collection campaigns.

[Plant Phenomics] The work asks for lesion-level disease and pest detection in orchards the question that Andvaag et al. (2024, *Plant Phenomics* 6, 0268) asked for aerial plant counting, whether training-set diversity or size governs generalisation to new fields, and it adds a preregistered prospective test of the kind that, to my knowledge, has not been reported for a crop-phenotyping detector.

[AIIA] The work is a data-centric study of a deployed machine-vision system: it measures how evaluation design, training-data composition and new-farm adaptation each change what a grower will experience, and closes with a preregistered prospective test. It fits the journal's scope in machine vision, precision agriculture and decision support.

The images and annotations are deposited on Zenodo (https://doi.org/10.5281/zenodo.22030623); the farm identifiers, split manifests, code, result tables, the lock file of the prospective test and a script that recomputes every number in the manuscript are public on GitHub (https://github.com/xiaolin200206/durian-site-split).

The peninsular images derive from the same collection as a separate manuscript on capture-session leakage in image classification, which is under review elsewhere. That manuscript classifies whole images and treats the capture session as the unit; the present one detects individual lesions and insects, treats the farm as the unit, and adds the data-budget, new-farm and prospective experiments. The leave-one-farm-out runs reported in Sections 3.1–3.3 were also analysed, together with three non-agricultural datasets, in a broader methodological manuscript that I withdrew from another journal before review; the present manuscript replaces it for the durian results and is not under consideration elsewhere. I am an independent researcher without institutional or grant funding; I am developing a handheld decision-support device for durian growers, and this interest is declared in the manuscript.

Thank you for considering this submission.

Yours sincerely,

Lin Ding Shan
Independent researcher, Kuala Lumpur, Malaysia
lim20020725@gmail.com · ORCID 0009-0009-6031-8479
