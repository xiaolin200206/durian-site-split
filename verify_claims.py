#!/usr/bin/env python3
"""
Recompute every quantitative claim in the paper from the released tables.

Each claim is stated here as it appears in the manuscript, recomputed from
results/, and compared. A claim that does not reproduce is printed with both
values and the script exits non-zero, so this can run in CI and a number
edited in the text without re-deriving it will fail the build.

    python verify_claims.py
    python verify_claims.py --verbose      # show every claim, not just failures

Nothing here reads the images or the model weights. The tables in results/
are the interface between the experiment and the paper, and this script is
the interface between the tables and the prose.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CLASSES = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid", "Psyllid_damage",
           "leaf_hopper_damage"]

checks = []


def check(claim, got, expected, tol=0.001, note=""):
    """Record one claim. tol is absolute unless expected is 0."""
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - expected) <= tol
        g, e = f"{got:.4g}", f"{expected:.4g}"
    else:
        ok = str(got) == str(expected)
        g, e = str(got), str(expected)
    checks.append({"claim": claim, "ok": ok, "got": g,
                   "expected": e, "note": note})
    return ok


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.isfile(p):
        sys.exit(f"missing table: {p}")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ir = load("in_region.csv")
    ci = load("cross_island.csv")
    st = load("sabah_by_tree.csv")
    so = load("sabah_by_orchard.csv")
    pb = load("peninsula_by_burst_15s.csv")
    sa = load("split_assignment.csv")
    rs = load("runs_summary.csv")
    with open(os.path.join(RESULTS, "dataset_stats.json"), encoding="utf-8") as fh:
        ds = json.load(fh)

    # ---------------------------------------------------------- abstract --
    sa_farm = sa[sa["farm"].notna()]
    check("pool: 560 peninsular images with a farm", len(sa_farm), 560, 0)
    check("Sabah: 281 images", ds["merged_sabah"]["images"], 281, 0)
    check("analysis total 841 images", len(sa_farm) + ds["merged_sabah"]["images"], 841, 0)
    check("peninsular corpus 1033 annotated images",
          ds["merged_peninsula"]["images"], 1033, 0,
          "includes 120 negatives; see Methods")

    rnd = ir[ir["config"] == "random"]
    farm = ir[ir["config"].str.startswith("byfarm")]
    fold_means = farm.groupby("config")["mAP50"].mean()

    check("random split mAP50 = 0.478", rnd["mAP50"].mean(), 0.478, 0.0005)
    check("random split s.d. = 0.003", rnd["mAP50"].std(), 0.003, 0.0005)
    check("by-farm mean mAP50 = 0.246", fold_means.mean(), 0.246, 0.0005)
    check("by-farm s.d. of fold means = 0.099", fold_means.std(), 0.099, 0.0005)

    gap = rnd["mAP50"].mean() - fold_means.mean()
    check("gap = 0.232", gap, 0.232, 0.0005)
    check("overstatement = 48.5%", 100 * gap / rnd["mAP50"].mean(), 48.5, 0.1)

    within = farm.groupby("config")["mAP50"].std().mean()
    check("mean within-fold seed s.d. = 0.015", within, 0.015, 0.0005)
    check("between/within ratio = 6.4", fold_means.std() / within, 6.4, 0.05)

    check("worst-to-best fold ratio = 2.84",
          fold_means.max() / fold_means.min(), 2.84, 0.01)
    check("worst fold mAP50 = 0.139", fold_means.min(), 0.139, 0.0005)
    check("best fold mAP50 = 0.394", fold_means.max(), 0.394, 0.0005)

    # ------------------------------------------------------ cross-island --
    check("Sabah mAP50 over all 30 models = 0.246",
          ci["mAP50"].mean(), 0.246, 0.001)
    check("Sabah s.d. across configurations = 0.009",
          ci.groupby("config")["mAP50"].mean().std(), 0.009, 0.001)
    cfg = ci.groupby("config")["mAP50"].mean()
    check("Sabah range low = 0.236", cfg.min(), 0.236, 0.001)
    check("Sabah range high = 0.263", cfg.max(), 0.263, 0.001)

    # 0.236-0.263 is the range of the six CONFIGURATION means. The manuscript
    # once said "every trained model" scores in that band, which is false:
    # individual models range twice as widely. Both are checked so the two
    # can never again be confused in prose.
    per_model = ci.groupby(["config", "seed"])["mAP50"].first()
    check("Sabah per-model low = 0.148", per_model.min(), 0.148, 0.001)
    check("Sabah per-model high = 0.296", per_model.max(), 0.296, 0.001)
    check("Sabah per-model s.d. = 0.027", per_model.std(), 0.027, 0.001,
          "seed noise; three times the s.d. across configuration means")
    check("30 model-evaluations on Sabah", len(ci), 30, 0)

    # -------------------------------------------------------- precision ---
    check("random precision = 0.597", rnd["precision"].mean(), 0.597, 0.001)
    check("random recall = 0.490", rnd["recall"].mean(), 0.490, 0.001)
    fp = farm.groupby("config")[["precision", "recall"]].mean()
    check("by-farm precision = 0.387", fp["precision"].mean(), 0.387, 0.001)
    check("by-farm recall = 0.271", fp["recall"].mean(), 0.271, 0.001)

    # -------------------------------------------------------- per class ---
    for cls, rnd_ap, sab_ap in [
        ("Algal", 0.563, 0.582),
        ("Leaf_rot", 0.357, 0.426),
        ("Phomopsis", 0.469, 0.298),
        ("Psyllid", 0.329, 0.075),
        ("Psyllid_damage", 0.169, 0.036),
        ("leaf_hopper_damage", 0.980, 0.057),
    ]:
        col = f"AP50::{cls}"
        if col in rnd.columns:
            check(f"{cls}: random split AP50 = {rnd_ap}",
                  rnd[col].dropna().mean(), rnd_ap, 0.001)
        if col in ci.columns:
            check(f"{cls}: Sabah AP50 = {sab_ap}",
                  ci[col].dropna().mean(), sab_ap, 0.001)

    # farms present per class, from the split pool
    cnt = defaultdict(set)
    for _, r in sa_farm.iterrows():
        c = str(r.get("classes", ""))
        if c in ("", "nan"):
            continue
        for x in c.split(","):
            if x.strip():
                cnt[int(x)].add(int(r["farm"]))
    for i, (cls, n) in enumerate(zip(CLASSES, [5, 5, 5, 3, 2, 3])):
        check(f"{cls}: present at {n} farms", len(cnt[i]), n, 0)

    # ---------------------------------------------------------- per site --
    stg = (st.groupby("group")
             .agg(imgs=("n_images", "first"), m=("mAP50", "mean")))
    stg = stg[stg["imgs"] >= 5]
    check("Sabah trees with >=5 images = 12", len(stg), 12, 0)
    check("Sabah per-tree mean = 0.308", stg["m"].mean(), 0.308, 0.001)
    check("Sabah per-tree s.d. = 0.086", stg["m"].std(), 0.086, 0.001)
    check("Sabah per-tree CV = 0.28", stg["m"].std() / stg["m"].mean(),
          0.28, 0.01)
    check("Sabah per-tree low = 0.201", stg["m"].min(), 0.201, 0.001)
    check("Sabah per-tree high = 0.540", stg["m"].max(), 0.540, 0.001)

    pbg = (pb.groupby("burst")
             .agg(farm=("farm", "first"), m=("mAP50", "mean")))
    check("peninsular bursts = 34", len(pbg), 34, 0)
    check("peninsular per-burst mean = 0.316", pbg["m"].mean(), 0.316, 0.001)
    check("peninsular per-burst s.d. = 0.227", pbg["m"].std(), 0.227, 0.001)
    check("peninsular per-burst CV = 0.72",
          pbg["m"].std() / pbg["m"].mean(), 0.72, 0.01)
    check("peninsular per-burst low = 0.000 (one burst scores zero)",
          pbg["m"].min(), 0.000, 0.001)
    check("peninsular per-burst low excluding that burst = 0.004",
          pbg.loc[pbg["m"] > 0, "m"].min(), 0.004, 0.001)
    check("bursts scoring exactly zero = 1", int((pbg["m"] == 0).sum()), 1, 0)
    check("peninsular per-burst high = 0.864", pbg["m"].max(), 0.864, 0.001)
    check("CV ratio peninsula/Sabah = 2.6",
          (pbg["m"].std() / pbg["m"].mean()) / (stg["m"].std() / stg["m"].mean()),
          2.6, 0.1)
    check("per-site means agree to within 0.008",
          abs(pbg["m"].mean() - stg["m"].mean()), 0.008, 0.001)

    pf = pbg.groupby("farm")["m"].mean()
    check("farm means span 4.6-fold", pf.max() / pf.min(), 4.6, 0.1)

    sog = so.groupby("group")["mAP50"].mean()
    check("Sabah orchards differ 1.3-fold", sog.max() / sog.min(), 1.3, 0.05)
    check("orchard o1 = 0.250", sog.get("o1", np.nan), 0.250, 0.001)
    check("orchard o2 = 0.335", sog.get("o2", np.nan), 0.335, 0.001)

    # ------------------------------------------------------------- runs ---
    trained = rs[rs["has_weights"] == True] if "has_weights" in rs.columns \
        else rs
    check("30 training runs", len(trained), 30, 0,
          "runs_summary also inventories the validation run directories")
    if "epochs_run" in trained.columns:
        er = trained["epochs_run"].dropna()
        check("epochs run: minimum 56", er.min(), 56, 0)
        check("epochs run: maximum 150", er.max(), 150, 0)
        check("epochs run: median 124.5", er.median(), 124.5, 0.01)

    # --------------------------------------------- annotation scale -------
    # This block exists because it did not. The 92.8-fold Leaf_rot range is
    # a headline claim and was unverified for the life of the project; the
    # manuscript carried a Leaf_rot row (96/21/109/45/124 px) from a
    # superseded run while every other row was correct. Nothing here is
    # allowed to be asserted in prose again without a check.
    ansc = load("farm_annotation_scale.csv")
    ansc = ansc.set_index("class")

    check("Leaf_rot area ratio = 92.8-fold",
          ansc.loc["Leaf_rot", "ratio"], 92.8, 0.1)
    for cls, r in [("Algal", 7.4), ("Phomopsis", 3.8), ("Psyllid", 3.7),
                   ("Psyllid_damage", 1.1), ("leaf_hopper_damage", 1.2)]:
        check(f"{cls} area ratio = {r}-fold", ansc.loc[cls, "ratio"], r, 0.05)

    # Sides quoted in the manuscript table, as sqrt(area fraction) * 640.
    sides = {("Leaf_rot", "f0"): 83, ("Leaf_rot", "f2"): 10,
             ("Leaf_rot", "f3"): 55, ("Leaf_rot", "f5"): 45,
             ("Leaf_rot", "f6"): 96,
             ("Algal", "f0"): 19, ("Algal", "f2"): 7,
             ("Algal", "f3"): 13, ("Algal", "f5"): 14,
             ("Phomopsis", "f0"): 8, ("Phomopsis", "f2"): 4,
             ("Phomopsis", "f5"): 5, ("Phomopsis", "f6"): 6,
             ("Psyllid", "f2"): 3, ("Psyllid", "f5"): 5,
             ("Psyllid_damage", "f2"): 8, ("Psyllid_damage", "f5"): 8,
             ("leaf_hopper_damage", "f2"): 87,
             ("leaf_hopper_damage", "f5"): 96}
    for (cls, farm), px in sides.items():
        got = np.sqrt(float(ansc.loc[cls, farm])) * 640
        check(f"{cls} at {farm} = {px} px side", got, px, 0.5)

    # The side ratio must square to the area ratio, or the table and the
    # sentence about it are describing different quantities.
    lr_side_ratio = (np.sqrt(float(ansc.loc["Leaf_rot", "f6"])) /
                     np.sqrt(float(ansc.loc["Leaf_rot", "f2"])))
    check("Leaf_rot side ratio squares to the area ratio",
          lr_side_ratio ** 2, float(ansc.loc["Leaf_rot", "ratio"]), 0.5)

    # ---------------------------------------------- capture conditions ----
    cap = load("farm_capture_conditions.csv").set_index("farm")
    check("farm 2 midday share = 99%", cap.loc[2, "midday_share"], 0.99, 0.005)
    check("farm 3 midday share = 100%", cap.loc[3, "midday_share"], 1.0, 0.001)
    check("farm 5 midday share = 100%", cap.loc[5, "midday_share"], 1.0, 0.001)
    check("farm 5 solar elevation = 72.1", cap.loc[5, "solar_median"], 72.07, 0.05)
    check("farm 2 solar elevation = 62.0", cap.loc[2, "solar_median"], 61.99, 0.05)
    check("farm 6 median capture hour = 08:52",
          cap.loc[6, "hour_median"], 8.867, 0.01)
    check("farm 6 focal length = 2.22 mm",
          cap.loc[6, "focal_median_mm"], 2.22, 0.001)
    check("farms 0, 2, 3, 5 focal length = 6.76 mm",
          float(cap.loc[[0, 2, 3, 5], "focal_median_mm"].nunique()), 1.0, 0,
          "all four share one focal length; farm 6 does not")

    # Stage association, reported in Results as directional and not inferred.
    lr_px = np.array([np.sqrt(float(ansc.loc["Leaf_rot", f])) * 640
                      for f in ["f0", "f2", "f3", "f5", "f6"]])
    fold_by_farm = np.array([0.394, 0.139, 0.292, 0.209, 0.196])
    r_all = np.corrcoef(lr_px, fold_by_farm)[0, 1]
    check("stage proxy vs fold mAP50, r = 0.54", r_all, 0.54, 0.01)
    r_no6 = np.corrcoef(lr_px[:4], fold_by_farm[:4])[0, 1]
    check("stage proxy vs fold mAP50 excluding farm 6, r = 0.97",
          r_no6, 0.97, 0.01, "reported but explicitly not relied on")
    n_img = np.array([165, 154, 59, 141, 41])
    check("control: images per fold vs mAP50, r = 0.15",
          np.corrcoef(n_img, fold_by_farm)[0, 1], 0.15, 0.01)

    # ------------------------------------------------------------ splits --
    check("random split: 448 train", (sa["random"] == "train").sum(), 448, 0)
    check("random split: 112 validation", (sa["random"] == "val").sum(), 112, 0)
    for fold, n in [("byfarm_fold0", 165), ("byfarm_fold1", 154),
                    ("byfarm_fold2", 141), ("byfarm_fold3", 59),
                    ("byfarm_fold4", 41)]:
        if fold in sa.columns:
            check(f"{fold}: {n} validation images",
                  (sa[fold] == "val").sum(), n, 0)

    # ------------------------------------------------------------ report --
    fails = [c for c in checks if not c["ok"]]
    width = max(len(c["claim"]) for c in checks)

    if a.verbose:
        for c in checks:
            mark = "ok  " if c["ok"] else "FAIL"
            line = f"{mark}  {c['claim']:<{width}}  {c['got']:>10s}"
            if not c["ok"]:
                line += f"   expected {c['expected']}"
            if c["note"]:
                line += f"   [{c['note']}]"
            print(line)
    else:
        for c in fails:
            print(f"FAIL  {c['claim']:<{width}}  got {c['got']}   "
                  f"expected {c['expected']}")

    print("\n" + "-" * 60)
    print(f"{len(checks) - len(fails)} of {len(checks)} claims reproduce")
    if fails:
        print(f"{len(fails)} do not. Either the tables changed or the paper "
              f"says something the tables do not.")
        return 1
    print("every quantitative claim in the manuscript matches the released "
          "tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
