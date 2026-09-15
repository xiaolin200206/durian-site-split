"""
Build two competing splits of the merged dataset.

  A. random   -- stratified 80/20 over images, i.e. the split you have been
                 using. Photos of the same tree, same day, same farm land on
                 both sides of the boundary.
  B. by-farm  -- GroupKFold with the farm as the group, so a farm in the
                 validation fold was never seen in training.

The farm labels come from metadata.csv, which exif_audit.py now writes.
This script does not re-derive them: two different clusterings of the same
coordinates would make the two halves of the pipeline disagree.

Images whose farm could not be established always go to train in the
by-farm splits, and are counted separately. If that pool is large the
comparison is contaminated by a difference in training-set size as well as
by the split rule, so the report prints it prominently.

A third set of folds, gps_only_*, repeats the by-farm split using only the
images whose farm came from GPS rather than from timing. It is the
sensitivity check for the inferred assignments.

Outputs, under OUT_DIR:
    random/            data.yaml, train.txt, val.txt
    byfarm_fold0..k/   data.yaml, train.txt, val.txt
    gps_only_fold0..k/ data.yaml, train.txt, val.txt
    split_assignment.csv

Setup:
    pip install pandas numpy scikit-learn pyyaml

Usage:
    python make_splits.py
"""

import os
import sys
import glob
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, train_test_split

# ---------------------------------------------------------------- settings --
BASE = (r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper"
        r"\durian_for_nature_food")

MERGED = os.path.join(BASE, "dataset_store", "merged_peninsula")
META = os.path.join(BASE, "metadata.csv")   # falls back to the backup
OUT_DIR = os.path.join(BASE, "dataset_store", "splits")   # suffixed by mode below

N_FOLDS = 5
VAL_FRACTION = 0.20
SEED = 42

# Mode A: drop every image whose farm is unknown, so the random split and
#         the by-farm split are drawn from exactly the same pool and the only
#         difference between them is the split rule.
# Mode B: keep them; they sit in train for every by-farm fold, which means the
#         two splits also differ in training-set size.
ONLY_WITH_FARM = True

STRICT_SOURCES = {"gps"}      # farm_source values treated as directly evidenced
# ---------------------------------------------------------------------------


def stem_of(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_labels(merged, stem):
    p = os.path.join(merged, "labels", stem + ".txt")
    if not os.path.isfile(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                out.append(int(float(parts[0])))
    return out


def write_split(out_dir, name, df, col, nc, names):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    for part in ("train", "val"):
        with open(os.path.join(d, f"{part}.txt"), "w", encoding="utf-8") as fh:
            for p in df.loc[df[col] == part, "image"]:
                fh.write(str(p).replace("\\", "/") + "\n")
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({
            "train": os.path.join(d, "train.txt").replace("\\", "/"),
            "val": os.path.join(d, "val.txt").replace("\\", "/"),
            "nc": nc,
            "names": names,
        }, fh, sort_keys=False, allow_unicode=True)


def class_counts(df, col):
    out = {}
    for part in ("train", "val"):
        c = Counter()
        for s in df.loc[df[col] == part, "classes"]:
            if s:
                for x in str(s).split(","):
                    c[int(x)] += 1
        out[part] = c
    return out


def main():
    meta_path = META
    if not os.path.isfile(meta_path):
        alt = os.path.join(BASE, "metadata_backup.csv")
        if os.path.isfile(alt):
            print(f"metadata.csv missing, using {alt}")
            meta_path = alt
    for p, what in [(MERGED, "merged peninsula set"),
                    (meta_path, "metadata csv")]:
        if not os.path.exists(p):
            sys.exit(f"{what} not found: {p}")

    with open(os.path.join(MERGED, "data.yaml"), "r", encoding="utf-8") as fh:
        dy = yaml.safe_load(fh)
    names = dy["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    nc = len(names)

    imgs = sorted(p for p in glob.glob(os.path.join(MERGED, "images", "*.*"))
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not imgs:
        sys.exit("No images in the merged set.")

    # ------------------------------------------------------ metadata join --
    meta = pd.read_csv(meta_path)
    if "farm" not in meta.columns:
        sys.exit("metadata.csv has no 'farm' column. Re-run exif_audit.py.")
    if "stem" not in meta.columns:
        meta["stem"] = meta["file"].map(stem_of)

    # a stem can appear twice (DNG + HEIC of one shot); keep the first
    meta = meta.drop_duplicates(subset="stem", keep="first").set_index("stem")
    has_src = "farm_source" in meta.columns

    rows = []
    for p in imgs:
        s = stem_of(p)
        cls = load_labels(MERGED, s)
        m = meta.loc[s] if s in meta.index else None
        rows.append({
            "stem": s,
            "image": p,
            "farm": (m["farm"] if m is not None and pd.notna(m["farm"])
                     else np.nan),
            "farm_source": (m["farm_source"] if (m is not None and has_src)
                            else None),
            "date": (m["date"] if m is not None and "date" in meta.columns
                     else None),
            "model": (m["Model"] if m is not None and "Model" in meta.columns
                      else None),
            "in_meta": m is not None,
            "n_boxes": len(cls),
            "classes": ",".join(str(c) for c in sorted(set(cls))),
            "primary": Counter(cls).most_common(1)[0][0] if cls else -1,
        })

    df = pd.DataFrame(rows)

    n_meta = int(df["in_meta"].sum())
    n_farm = int(df["farm"].notna().sum())
    print(f"Images in merged set   : {len(df)}")
    print(f"  found in metadata    : {n_meta}")
    print(f"  with a farm          : {n_farm}  ({n_farm/len(df):.0%})")
    print(f"  without a farm       : {len(df)-n_farm}")

    if n_meta < len(df):
        print(f"\n  {len(df)-n_meta} merged images are absent from "
              f"metadata.csv:")
        for s in df.loc[~df["in_meta"], "stem"].head(10):
            print(f"    {s}")
        print("  These sit in train for every by-farm fold.")

    if has_src:
        print("\nfarm_source among merged images:")
        print(df["farm_source"].value_counts(dropna=False).to_string())

    if ONLY_WITH_FARM:
        dropped = int(df["farm"].isna().sum())
        df = df[df["farm"].notna()].reset_index(drop=True)
        print(f"\nMODE A: dropped {dropped} images with no farm.")
        print(f"        pool is now {len(df)} images, identical for both splits.")
    else:
        print("\nMODE B: images with no farm are kept and always go to train.")

    known = df.dropna(subset=["farm"]).copy()
    known["farm"] = known["farm"].astype(int)

    print("\nPer farm:")
    fs = (known.groupby("farm")
               .agg(images=("stem", "size"),
                    boxes=("n_boxes", "sum"),
                    dates=("date", lambda s: ",".join(sorted(
                        {str(x) for x in s.dropna()})[:2])),
                    devices=("model", lambda s: ",".join(sorted(
                        {str(x) for x in s.dropna()})))))
    print(fs.sort_values("images", ascending=False).to_string())

    # -------------------------------------------------------- random split --
    vc = df["primary"].value_counts()
    strat = df["primary"].where(df["primary"].map(vc) >= 2, -99)
    tr, va = train_test_split(df.index, test_size=VAL_FRACTION,
                              random_state=SEED, stratify=strat)
    df["split_random"] = "train"
    df.loc[va, "split_random"] = "val"

    # ------------------------------------------------------- by-farm split --
    def build_group_folds(pool, prefix):
        if len(pool) == 0:
            return []
        n_groups = pool["farm"].nunique()
        if n_groups < 2:
            print(f"\n{prefix}: only {n_groups} farm(s), skipped")
            return []
        folds = min(N_FOLDS, n_groups)
        gkf = GroupKFold(n_splits=folds)
        cols = []
        for k, (_, vai) in enumerate(gkf.split(pool, groups=pool["farm"])):
            col = f"{prefix}_fold{k}"
            df[col] = "train"
            df.loc[pool.index[vai], col] = "val"
            cols.append(col)
        return cols

    farm_cols = build_group_folds(known, "split_farm")

    strict = (known[known["farm_source"].isin(STRICT_SOURCES)]
              if has_src else known.iloc[0:0])
    strict_cols = build_group_folds(strict, "split_gps")

    # ------------------------------------------------------------- write ----
    out_dir = OUT_DIR + ("_A" if ONLY_WITH_FARM else "_B")
    os.makedirs(out_dir, exist_ok=True)
    write_split(out_dir, "random", df, "split_random", nc, names)
    for k, col in enumerate(farm_cols):
        write_split(out_dir, f"byfarm_fold{k}", df, col, nc, names)
    for k, col in enumerate(strict_cols):
        write_split(out_dir, f"gps_only_fold{k}", df, col, nc, names)

    df.to_csv(os.path.join(out_dir, "split_assignment.csv"),
              index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ report ----
    print("\n" + "=" * 74)
    print("IMAGES CARRYING EACH CLASS, IN EACH VALIDATION SET")
    print("=" * 74)
    tabs = {"random": class_counts(df, "split_random")}
    for k, col in enumerate(farm_cols):
        tabs[f"f{k}"] = class_counts(df, col)

    hdr = f"{'class':24s} {'rand tr':>8s} {'rand val':>9s}"
    for k in range(len(farm_cols)):
        hdr += f" {'f'+str(k):>6s}"
    print(hdr)
    for i, n in enumerate(names):
        line = (f"{n:24s} {tabs['random']['train'].get(i,0):8d} "
                f"{tabs['random']['val'].get(i,0):9d}")
        for k in range(len(farm_cols)):
            line += f" {tabs[f'f{k}']['val'].get(i,0):6d}"
        print(line)
    print("\nA zero above means that class cannot be evaluated on that fold.")

    dead = []
    for i, n in enumerate(names):
        z = [k for k in range(len(farm_cols)) if tabs[f"f{k}"]["val"].get(i, 0) == 0]
        if z:
            dead.append((n, z))
    if dead:
        print("\nClasses with no validation instances on some folds:")
        for n, z in dead:
            print(f"  {n:24s} missing on folds {z}")
        print("  This is class-farm confounding, not a bug: those classes were")
        print("  only ever photographed at certain farms. Report the mean over")
        print("  the folds where the class is present, and say how many that is.")

    print("\nValidation set sizes:")
    print(f"  random            : {(df['split_random']=='val').sum():5d}")
    for k, col in enumerate(farm_cols):
        v = df[df[col] == "val"]
        print(f"  byfarm_fold{k}      : {len(v):5d}   "
              f"farms {sorted(int(x) for x in v['farm'].dropna().unique())}")
    for k, col in enumerate(strict_cols):
        v = df[df[col] == "val"]
        print(f"  gps_only_fold{k}    : {len(v):5d}   "
              f"farms {sorted(int(x) for x in v['farm'].dropna().unique())}")

    print(f"\nWritten to {out_dir}")
    print("\nTrain, holding everything else fixed:")
    print(f'  yolo detect train '
          f'data="{os.path.join(out_dir, "random", "data.yaml")}" '
          f'model=yolo11s.pt imgsz=640 epochs=100 seed={SEED}')
    print("  ... then each byfarm fold, and compare.")
    print("\nThe gap between the random result and the mean over the by-farm")
    print("folds is the quantity of interest.")


if __name__ == "__main__":
    main()
