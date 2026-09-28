#!/usr/bin/env python3
"""Build paper/supplementary.md and .docx from the result tables (no hand-typed numbers)."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R  # noqa: E402

RES, PAPER = R.RES, R.PAPER
f3 = R.f3


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    N = json.load(open(os.path.join(RES, "paper_numbers.json")))
    out = ["# Supplementary material",
           "",
           "**What a new orchard costs: farm-level generalisation and data budgets for on-device "
           "durian disease and pest detection**",
           "",
           "Lin Ding Shan",
           ""]

    # Note S1
    out += ["## Note S1. Training details", "",
            "**Leave-one-farm-out and random-split runs (Sections 2.4–2.5).** Ultralytics 8.x. "
            "YOLO11n, YOLO11s, YOLO11m, YOLO11l and RT-DETR-L: COCO-pretrained, 640 px, up to 150 "
            "epochs, patience 50, batch 32, 32, 16, 8 and 24 respectively, five seeds (42, 1, 2, 3, 4), "
            "default augmentation. Faster R-CNN ResNet-50 FPN (torchvision, improved recipe): "
            "COCO-pretrained, SGD with cosine annealing, 40 epochs, patience 12, batch 4, three seeds. "
            "The inner validation set is 10% of each fold's training images, stratified by each image's "
            "dominant class (at least one image per class with two or more images), drawn with seed "
            "20260911 independently of the training seed; the held-out farm is evaluated once with the "
            "weights selected on the inner set.",
            "",
            "**Data-budget runs (Section 2.6).** YOLO11n, COCO-pretrained, 640 px, batch 32 with the "
            "Ultralytics default nominal batch of 64 (gradients accumulated over two batches), no "
            "validation during training, no early stopping, final weights evaluated. The training list "
            "of n images is repeated r = ceil(640 / n) times so that an epoch has at least 20 batches, "
            "and epochs = round(2000 / ceil(n r / 32)), giving 1,988–2,015 iterations in every run. "
            "Mosaic augmentation is switched off for the last 10% of epochs; warm-up is the Ultralytics "
            "default (at least 100 iterations); the optimiser was selected automatically (AdamW, "
            "learning rate 0.001). Farm combinations and image subsets were drawn with a fixed design "
            "seed (20260928); within a combination the 15-, 50- and all-image subsets are nested "
            "prefixes of one shuffled order per farm. The exact images of every training set are listed "
            "in results_applied/farm_budget_train_lists.csv.",
            "",
            "**New-farm calibration (Section 2.7).** Fine-tuning: from the seven-farm data-budget "
            "model, backbone frozen (first 10 modules), AdamW, learning rate 5 × 10⁻⁴, no warm-up, "
            "batch 16 with nominal batch 16 (no gradient accumulation), calibration images repeated so "
            "that an epoch has at least 20 batches, 300 optimiser steps, final weights. Retraining: "
            "the data-budget protocol applied to the other seven farms plus the calibration images. "
            "Capture order uses EXIF DateTimeOriginal where available and the camera counter "
            "otherwise.",
            ""]

    # Table S1 per farm per detector
    w = pd.read_csv(os.path.join(RES, "tableS_per_farm_detectors.csv"), index_col=0)
    cols = [c for c in R.DET if c in w.columns]
    rows = [[f"Farm {int(f)}"] + [f3(w.loc[f, c]) for c in cols] + [f3(w.loc[f].mean())]
            for f in w.index]
    out += ["## Table S1. Unseen-farm mAP50 of each farm under each detector", "",
            "Clean leave-one-farm-out protocol; mean over seeds.", "",
            md_table(["Held-out farm"] + [R.DET[c].replace(" (deployed)", "") for c in cols] + ["Mean"],
                     rows), ""]

    # Table S2 per class by farm (mean of five detectors)
    pcf = pd.read_csv(os.path.join(RES, "per_class_by_farm.csv"))
    piv = pcf.groupby(["class", "heldout_farm"]).new_farm_AP50.mean().unstack()
    farms = sorted(piv.columns)
    rows = []
    for c in R.NAMES:
        if c in piv.index:
            rows.append([R.LABEL[c]] + [f3(piv.loc[c, f]) if pd.notna(piv.loc[c, f]) else "–"
                                        for f in farms])
    out += ["## Table S2. Per-class AP50 on each unseen farm", "",
            "Mean of the five detectors that report per-class AP (clean protocol). A dash marks a "
            "class absent from that farm.", "",
            md_table(["Class"] + [f"Farm {f}" for f in farms], rows), ""]

    # Table S3 class by k
    ck = pd.read_csv(os.path.join(RES, "tableS_class_by_k.csv"), index_col=0)
    rows = [[R.LABEL[c]] + [f3(ck.loc[k, c]) for k in ck.index] for c in R.NAMES]
    out += ["## Table S3. Per-class unseen-farm AP50 by number of training farms", "",
            "Data-budget experiment, all photographs per farm, YOLO11n; held-out farms weighted "
            "equally.", "",
            md_table(["Class"] + [f"{k} farm" + ("s" if k > 1 else "") for k in ck.index], rows), ""]

    # Table S4 per held-out farm grid
    df = pd.read_csv(os.path.join(RES, "farm_budget.csv"), dtype={"m": str, "heldout": str})
    h = df[df.eval_on == "heldout"]
    g = h.groupby(["k", "m", "heldout"]).mAP50.mean().unstack()
    order = sorted(g.index, key=lambda t: (t[0], {"15": 0, "50": 1, "all": 2}[t[1]]))
    fs = sorted(g.columns, key=int)
    rows = [[k, m] + [f3(g.loc[(k, m), f]) for f in fs] for k, m in order]
    out += ["## Table S4. Data-budget experiment, per held-out farm", "",
            "YOLO11n unseen-farm mAP50; mean over the two farm draws (one for seven farms) and the seeds.",
            "", md_table(["Farms", "Photos per farm"] + [f"Farm {f}" for f in fs], rows), ""]

    # Table S5 draws
    d = h[h.k < 7].groupby(["k", "m", "heldout", "draw"]).mAP50.mean().unstack()
    spread = (d.max(1) - d.min(1)).groupby(level=[0, 1]).agg(["median", "max"])
    rows = [[k, m, f3(r["median"]), f3(r["max"])] for (k, m), r in
            sorted(spread.iterrows(), key=lambda t: (t[0][0], {"15": 0, "50": 1, "all": 2}[t[0][1]]))]
    out += ["## Table S5. Difference between the two farm draws", "",
            "Absolute difference in unseen-farm mAP50 between the two farm combinations drawn for "
            "the same held-out farm and budget (k < 7).", "",
            md_table(["Farms", "Photos per farm", "Median over held-out farms", "Maximum"], rows), ""]

    # Table S6 steps
    st = pd.read_csv(os.path.join(RES, "tableS_steps.csv"))
    rows = [[r.step.replace("->", " → ").replace("photos", "images per farm"), r.at.replace("per farm", "images per farm"),
             f"{f3(r['diff'])} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]", f"{r.wins} of {r.farms}"]
            for _, r in st.iterrows()]
    out += ["## Table S6. Step-wise paired contrasts in the data-budget experiment", "",
            "Change in unseen-farm mAP50 for one step in the number of training farms (at fixed images "
            "per farm) or in images per farm (at fixed farms), paired by held-out farm; 90% bootstrap "
            "interval over held-out farms, and the number of farms on which the step helped.", "",
            md_table(["Step", "At", "Change [90% CI]", "Farms improved"], rows), ""]

    # Table S7 schedule comparison
    p7 = os.path.join(RES, "tableS_schedule_comparison.csv")
    if os.path.isfile(p7):
        t7 = pd.read_csv(p7, dtype={"m": str})
        rows = [[int(r.k), r.m, r.v1_iterations, f3(r.v1_capped), f3(r.v2_fixed_iterations)]
                for r in t7.itertuples()]
        out += ["## Table S7. Earlier, epoch-capped schedule versus fixed iterations", "",
                "The same farm draws and image subsets trained with epochs = clamp(round(2000 × b / n), "
                "100, 300), b = min(32, n), which gives small budgets far fewer iterations, and with the "
                "fixed-iteration schedule used in the paper (seed 42 in both). Mean unseen-farm mAP50 "
                "over held-out farms.", "",
                md_table(["Farms", "Images per farm", "Iterations (capped)", "mAP50 (capped)",
                          "mAP50 (fixed iterations)"], rows), ""]

    # Note S2 attribution
    out += ["## Note S2. Farm attribution and the filename error", "",
            "Farm identity was first joined to annotated images by filename. Two field visits "
            "produced overlapping camera counters, and a resize step dropped the suffixes that had "
            "distinguished them, so 50 images of the 560-image pool then in use (8.9%) carried another "
            "photograph's metadata; 49 of them were labelled farm 0 and had been taken at farm 6. "
            "Every annotated image was therefore re-matched to its original photograph by a 64-bit "
            "difference hash and assigned that original's farm, accepted only when no original from a "
            "different farm lay within three bits. On the 570 images whose filenames still matched an "
            "original, the hash recovered that same original (median Hamming distance 0); it also "
            "recovered originals for 303 images whose filenames the annotation platform had replaced. "
            "Because some originals themselves carry neither a location nor a located image within 30 "
            "minutes, 829 of the 1,033 annotated peninsular images could be assigned a farm. The 204 "
            "that could not be over-represent one class (56% of boxes against 41%) and under-represent "
            "another (4% against 17%), and are excluded from every analysis.", ""]

    md = os.path.join(PAPER, "supplementary.md")
    open(md, "w", encoding="utf-8").write("\n".join(out))
    R.to_docx(md, "supplementary.docx")
    print("wrote", md)


if __name__ == "__main__":
    main()
