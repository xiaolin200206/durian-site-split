"""
Colab driver for the durian split-comparison experiment.

Paste this into Colab as a file (or upload it) and run:

    !pip -q install ultralytics
    from google.colab import drive; drive.mount('/content/drive')
    !python colab_run.py --step all

Everything lives under Drive so a disconnect costs nothing: the archive is
unpacked to local disk for speed, but runs and results are written back to
Drive, and a run that already has best.pt is skipped on the next attempt.

Steps, runnable separately with --step:
    unpack   extract the archive, report what is inside
    splits   rebuild train/val lists with Linux paths, grouped by farm
    train    multi-seed training over every split
    cross    evaluate each trained model on the Sabah set
    all      the four in order
"""

import os
import re
import sys
import glob
import json
import time
import shutil
import zipfile
import argparse
import subprocess
from collections import Counter, defaultdict

# ---------------------------------------------------------------- settings --
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
ARCHIVE = os.path.join(DRIVE, "durian_colab.zip")

WORK = "/content/durian"                      # local disk, fast
RUNS = os.path.join(DRIVE, "runs_colab")      # on Drive, survives a restart
RESULTS = os.path.join(DRIVE, "results")

MODEL = "yolo11s.pt"
IMGSZ = 640
EPOCHS = 150
PATIENCE = 50
BATCH = 32
WORKERS = 8
SEEDS = [42, 1, 2, 3, 4]

N_FOLDS = 5
VAL_FRACTION = 0.20
SPLIT_SEED = 42
ONLY_WITH_FARM = True          # mode A: identical pool for both split rules
# ---------------------------------------------------------------------------


def sh(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=False)


def find_one(root, *names):
    """Locate a file or directory by name anywhere under root."""
    for dp, dns, fns in os.walk(root):
        for n in names:
            if n in dns:
                return os.path.join(dp, n)
            if n in fns:
                return os.path.join(dp, n)
    return None


# ------------------------------------------------------------------ unpack --
def step_unpack(a):
    if not os.path.isfile(a.archive):
        sys.exit(f"archive not found: {a.archive}\n"
                 f"check the path in Drive")

    os.makedirs(WORK, exist_ok=True)
    print(f"extracting {a.archive}  ->  {WORK}")
    t0 = time.time()
    with zipfile.ZipFile(a.archive) as zf:
        zf.extractall(WORK)
    print(f"done in {time.time()-t0:.0f}s")

    print("\ntop level:")
    for n in sorted(os.listdir(WORK))[:40]:
        p = os.path.join(WORK, n)
        kind = "dir " if os.path.isdir(p) else "file"
        print(f"  {kind}  {n}")

    for key in ("merged_peninsula", "merged_sabah"):
        p = find_one(WORK, key)
        if p:
            imgs = len(glob.glob(os.path.join(p, "images", "*.*")))
            lbls = len(glob.glob(os.path.join(p, "labels", "*.txt")))
            print(f"\n  {key}: {p}")
            print(f"    images {imgs}   labels {lbls}")
        else:
            print(f"\n  {key}: NOT FOUND")

    meta = find_one(WORK, "metadata.csv", "metadata_backup.csv")
    print(f"\n  metadata: {meta or 'NOT FOUND'}")


# ------------------------------------------------------------------ splits --
def step_splits(a):
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.model_selection import GroupKFold, train_test_split

    merged = find_one(WORK, "merged_peninsula")
    meta_p = find_one(WORK, "metadata.csv", "metadata_backup.csv")
    if not merged or not meta_p:
        sys.exit("run --step unpack first, or the archive is missing pieces")

    with open(os.path.join(merged, "data.yaml"), encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    names = y["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    nc = len(names)
    print(f"classes ({nc}): {names}")

    imgs = sorted(p for p in glob.glob(os.path.join(merged, "images", "*.*"))
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))

    meta = pd.read_csv(meta_p)
    if "stem" not in meta.columns:
        meta["stem"] = meta["file"].astype(str).str.replace(
            r"\.[^.]+$", "", regex=True)
    meta = meta.drop_duplicates("stem").set_index("stem")

    rows = []
    for p in imgs:
        s = os.path.splitext(os.path.basename(p))[0]
        lp = os.path.join(merged, "labels", s + ".txt")
        cls = []
        if os.path.isfile(lp):
            with open(lp, encoding="utf-8") as fh:
                for line in fh:
                    v = line.split()
                    if v:
                        cls.append(int(float(v[0])))
        m = meta.loc[s] if s in meta.index else None
        rows.append({
            "stem": s, "image": p,
            "farm": (m["farm"] if m is not None and pd.notna(m["farm"])
                     else np.nan),
            "classes": ",".join(str(c) for c in sorted(set(cls))),
            "primary": Counter(cls).most_common(1)[0][0] if cls else -1,
        })
    df = pd.DataFrame(rows)

    print(f"images {len(df)}   with a farm {df['farm'].notna().sum()}")
    if ONLY_WITH_FARM:
        n0 = len(df)
        df = df[df["farm"].notna()].reset_index(drop=True)
        print(f"mode A: dropped {n0-len(df)}, pool is {len(df)}")
    df["farm"] = df["farm"].fillna(-1).astype(int)

    known = df[df["farm"] >= 0]
    print("\nper farm:")
    print(known.groupby("farm").size().to_string())

    vc = df["primary"].value_counts()
    strat = df["primary"].where(df["primary"].map(vc) >= 2, -99)
    _, va = train_test_split(df.index, test_size=VAL_FRACTION,
                             random_state=SPLIT_SEED, stratify=strat)
    df["random"] = "train"
    df.loc[va, "random"] = "val"

    folds = min(N_FOLDS, known["farm"].nunique())
    gkf = GroupKFold(n_splits=folds)
    cols = ["random"]
    for k, (_, vai) in enumerate(gkf.split(known, groups=known["farm"])):
        col = f"byfarm_fold{k}"
        df[col] = "train"
        df.loc[known.index[vai], col] = "val"
        cols.append(col)

    out = os.path.join(WORK, "splits")
    if os.path.exists(out):
        shutil.rmtree(out)
    for col in cols:
        d = os.path.join(out, col)
        os.makedirs(d)
        for part in ("train", "val"):
            with open(os.path.join(d, f"{part}.txt"), "w",
                      encoding="utf-8") as fh:
                for p in df.loc[df[col] == part, "image"]:
                    fh.write(p + "\n")
        with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump({"train": os.path.join(d, "train.txt"),
                            "val": os.path.join(d, "val.txt"),
                            "nc": nc, "names": names},
                           fh, sort_keys=False, allow_unicode=True)

    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "split_assignment.csv"),
              index=False, encoding="utf-8-sig")

    print("\nimages carrying each class per validation set:")
    hdr = f"  {'class':24s}" + "".join(f"{c.replace('byfarm_','')[:8]:>9s}"
                                       for c in cols)
    print(hdr)
    for i, n in enumerate(names):
        line = f"  {n:24s}"
        for col in cols:
            k = sum(1 for s in df.loc[df[col] == "val", "classes"]
                    if s and str(i) in s.split(","))
            line += f"{k:9d}"
        print(line)

    print("\nval sizes:")
    for col in cols:
        v = df[df[col] == "val"]
        fs = sorted(set(v.loc[v["farm"] >= 0, "farm"]))
        print(f"  {col:16s} {len(v):5d}   farms {fs}")
    print(f"\nwritten to {out}")


# ------------------------------------------------------------------- train --
def step_train(a):
    from ultralytics import YOLO

    splits = os.path.join(WORK, "splits")
    cfgs = sorted(d for d in os.listdir(splits)
                  if os.path.isfile(os.path.join(splits, d, "data.yaml")))
    cfgs.sort(key=lambda c: (c != "random", c))

    jobs = [(c, s) for c in cfgs for s in a.seeds]
    print(f"{len(jobs)} runs: {cfgs} x {a.seeds}")
    os.makedirs(RUNS, exist_ok=True)

    t0 = time.time()
    for i, (cfg, seed) in enumerate(jobs, 1):
        name = f"{cfg}_s{seed}"
        if glob.glob(os.path.join(RUNS, name, "weights", "best.pt")):
            print(f"[{i}/{len(jobs)}] {name}: done, skipping")
            continue
        print(f"\n[{i}/{len(jobs)}] {name}   "
              f"elapsed {(time.time()-t0)/60:.0f} min")
        YOLO(a.model).train(
            data=os.path.join(splits, cfg, "data.yaml"),
            imgsz=a.imgsz, epochs=a.epochs, seed=seed,
            batch=a.batch, workers=a.workers, patience=a.patience,
            project=RUNS, name=name, exist_ok=True,
            deterministic=True, plots=False, verbose=True)
    print(f"\ntraining done in {(time.time()-t0)/60:.1f} min")


# ------------------------------------------------------------------- eval ---
def _val(model_path, data_yaml, imgsz, tag, names):
    from ultralytics import YOLO
    r = YOLO(model_path).val(data=data_yaml, imgsz=imgsz, split="val",
                             batch=8, workers=2, project=RUNS,
                             name=tag, exist_ok=True, verbose=False,
                             plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            cn = names[int(c)] if int(c) < len(names) else str(c)
            row[f"AP50::{cn}"] = float(r.box.ap50[i])
    except Exception:
        pass
    return row


def step_eval(a, cross=False):
    import pandas as pd
    import yaml

    splits = os.path.join(WORK, "splits")
    cfgs = sorted(d for d in os.listdir(splits)
                  if os.path.isfile(os.path.join(splits, d, "data.yaml")))
    with open(os.path.join(splits, cfgs[0], "data.yaml"),
              encoding="utf-8") as fh:
        names = yaml.safe_load(fh)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    sabah_yaml = None
    if cross:
        sabah = find_one(WORK, "merged_sabah")
        if not sabah:
            sys.exit("merged_sabah not found")
        sabah_yaml = os.path.join(WORK, "sabah_eval.yaml")
        with open(sabah_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"path": sabah, "train": "images", "val": "images",
                            "nc": len(names), "names": names},
                           fh, sort_keys=False, allow_unicode=True)
        print(f"Sabah evaluated as one held-out set: {sabah}")

    rows = []
    for cfg in cfgs:
        for seed in a.seeds:
            name = f"{cfg}_s{seed}"
            best = os.path.join(RUNS, name, "weights", "best.pt")
            if not os.path.isfile(best):
                continue
            if cross:
                r = _val(best, sabah_yaml, a.imgsz, f"{name}_sabah", names)
                r.update({"config": cfg, "seed": seed, "eval_on": "sabah"})
            else:
                r = _val(best, os.path.join(splits, cfg, "data.yaml"),
                         a.imgsz, f"{name}_val", names)
                r.update({"config": cfg, "seed": seed, "eval_on": "peninsula"})
            rows.append(r)
            print(f"  {name:24s} {r['eval_on']:10s} "
                  f"mAP50 {r['mAP50']:.4f}")

    if not rows:
        sys.exit("no trained models found")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS,
                       "cross_island.csv" if cross else "in_region.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("  BY CONFIG")
    print("=" * 70)
    g = df.groupby("config").agg(runs=("seed", "size"),
                                 mAP50=("mAP50", "mean"),
                                 sd=("mAP50", "std"),
                                 mAP=("mAP50_95", "mean"))
    print(g.round(4).to_string())

    print("\nper class, mean over runs:")
    for n in names:
        col = f"AP50::{n}"
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                print(f"  {n:24s} {s.mean():.4f} +/- {s.std():.4f}  "
                      f"n={len(s)}")
    print(f"\nwrote {out}")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["unpack", "splits", "train", "eval",
                             "cross", "all"])
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()

    if a.step in ("unpack", "all"):
        step_unpack(a)
    if a.step in ("splits", "all"):
        step_splits(a)
    if a.step in ("train", "all"):
        step_train(a)
    if a.step in ("eval", "all"):
        step_eval(a, cross=False)
    if a.step in ("cross", "all"):
        step_eval(a, cross=True)


if __name__ == "__main__":
    main()
