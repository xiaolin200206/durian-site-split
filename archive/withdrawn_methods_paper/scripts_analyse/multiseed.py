"""
Multi-seed replication of the split comparison.

The single-seed run gave a by-farm mean of 0.287 with a standard deviation
of 0.295, i.e. the spread across folds is as large as the effect. Fold 4
scored 0.778, above the random split, because its validation set is 41
images that are almost entirely one class. So the aggregate gap is not a
stable quantity and should not be reported as one.

What did look stable is the per-class pattern: a class that was
photographed at many farms barely drops, and a class confined to one or two
farms collapses. This script repeats every configuration across several
seeds so that pattern can be reported with a spread rather than as a single
number.

For the random split the seed also changes which images land in
validation, so its variance covers both training noise and split noise.
For the by-farm folds the membership is fixed by the farm, so the seed
covers training noise only. That asymmetry is deliberate and is stated in
the report.

Setup:
    pip install ultralytics pandas pyyaml

Usage:
    python multiseed.py                          # 3 seeds, by-farm + random
    python multiseed.py --seeds 42 1 2 3 4       # explicit seeds
    python multiseed.py --collect-only           # re-read finished runs
"""

import os
import sys
import glob
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- settings --
BASE = (r"C:\path\to\workdir\Durian project and paper"
        r"\durian_for_nature_food")
SPLITS = os.path.join(BASE, "dataset_store", "splits_A")

MODEL = "yolo11s.pt"
IMGSZ = 640
EPOCHS = 150
BATCH = 4
WORKERS = 2
PATIENCE = 50
VAL_BATCH = 4

SEEDS = [42, 1, 2]
PROJECT = os.path.join(BASE, "runs", "multiseed")
OUT_CSV = os.path.join(BASE, "multiseed_results.csv")
# ---------------------------------------------------------------------------


def n_lines(p):
    if not os.path.isfile(p):
        return 0
    with open(p, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def find_configs(splits_dir, skip_gps):
    out = []
    for d in sorted(os.listdir(splits_dir)):
        p = os.path.join(splits_dir, d, "data.yaml")
        if not os.path.isfile(p):
            continue
        if skip_gps and d.startswith("gps_only"):
            continue
        out.append((d, p))
    order = {"random": 0}
    out.sort(key=lambda t: (order.get(t[0], 1), t[0]))
    return out


def run_name(cfg, seed):
    return f"{cfg}_s{seed}"


def train_one(cfg, data_yaml, seed, args):
    from ultralytics import YOLO
    name = run_name(cfg, seed)
    if glob.glob(os.path.join(PROJECT, name, "weights", "best.pt")):
        print(f"  {name}: done, skipping")
        return

    print(f"\n{'='*70}\n  {name}\n{'='*70}")

    def _go(batch):
        YOLO(args.model).train(
            data=data_yaml, imgsz=args.imgsz, epochs=args.epochs,
            seed=seed, batch=batch, workers=args.workers,
            device=args.device, patience=args.patience,
            project=PROJECT, name=name, exist_ok=True,
            deterministic=True, verbose=True, plots=False)

    try:
        _go(args.batch)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        import torch
        torch.cuda.empty_cache()
        half = max(1, args.batch // 2)
        print(f"  OOM at batch {args.batch}, retrying at {half}")
        args.batch = half
        _go(half)


def evaluate(cfg, data_yaml, seed, names):
    from ultralytics import YOLO
    name = run_name(cfg, seed)
    best = os.path.join(PROJECT, name, "weights", "best.pt")
    if not os.path.isfile(best):
        return None

    r = YOLO(best).val(data=data_yaml, imgsz=IMGSZ, split="val",
                       batch=VAL_BATCH, workers=0, project=PROJECT,
                       name=f"{name}_val", exist_ok=True,
                       verbose=False, plots=False)

    row = {"config": cfg, "seed": seed,
           "kind": "random" if cfg == "random" else
                   ("gps_only" if cfg.startswith("gps_only") else "byfarm"),
           "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            cn = names[int(c)] if int(c) < len(names) else str(c)
            row[f"AP50::{cn}"] = float(r.box.ap50[i])
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--skip-gps-only", action="store_true", default=True)
    ap.add_argument("--include-gps-only", dest="skip_gps_only",
                    action="store_false")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()

    cfgs = find_configs(a.splits, a.skip_gps_only)
    if not cfgs:
        sys.exit("no configs found")
    names = class_names(cfgs[0][1])

    jobs = [(c, p, s) for c, p in cfgs for s in a.seeds]
    print(f"splits  : {a.splits}")
    print(f"seeds   : {a.seeds}")
    print(f"configs : {[c for c, _ in cfgs]}")
    print(f"total runs: {len(jobs)}")
    for c, p in cfgs:
        d = os.path.dirname(p)
        print(f"  {c:18s} train {n_lines(os.path.join(d,'train.txt')):5d}"
              f"  val {n_lines(os.path.join(d,'val.txt')):5d}")

    if not a.collect_only:
        t0 = time.time()
        for i, (c, p, s) in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}]  elapsed {(time.time()-t0)/60:.0f} min")
            train_one(c, p, s, a)
        print(f"\nall done in {(time.time()-t0)/60:.1f} min")

    rows = [r for c, p, s in jobs
            if (r := evaluate(c, p, s, names)) is not None]
    if not rows:
        sys.exit("nothing evaluated")

    df = pd.DataFrame(rows)
    for c, p in cfgs:
        d = os.path.dirname(p)
        df.loc[df["config"] == c, "n_train"] = n_lines(
            os.path.join(d, "train.txt"))
        df.loc[df["config"] == c, "n_val"] = n_lines(
            os.path.join(d, "val.txt"))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------- report ----
    print("\n" + "=" * 78)
    print("  EVERY RUN")
    print("=" * 78)
    print(df[["config", "seed", "n_train", "n_val", "mAP50", "mAP50_95",
              "precision", "recall"]].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("  PER CONFIG, ACROSS SEEDS")
    print("=" * 78)
    g = df.groupby("config").agg(
        runs=("seed", "size"),
        mAP50_mean=("mAP50", "mean"), mAP50_sd=("mAP50", "std"),
        mAP50_min=("mAP50", "min"), mAP50_max=("mAP50", "max"),
        mAP_mean=("mAP50_95", "mean"), mAP_sd=("mAP50_95", "std"))
    print(g.round(4).to_string())

    rnd = df[df["kind"] == "random"]
    farm = df[df["kind"] == "byfarm"]

    print("\n" + "=" * 78)
    print("  RANDOM VERSUS BY-FARM")
    print("=" * 78)
    if len(rnd):
        print(f"  random   mAP50 {rnd['mAP50'].mean():.4f} "
              f"+/- {rnd['mAP50'].std():.4f}   over {len(rnd)} seeds")
    if len(farm):
        per_fold = farm.groupby("config")["mAP50"].mean()
        print(f"  by-farm  mAP50 {per_fold.mean():.4f} "
              f"+/- {per_fold.std():.4f}   over {len(per_fold)} folds "
              f"(fold means)")
        print("\n  fold means:")
        for c, v in per_fold.sort_values().items():
            sd = farm.loc[farm["config"] == c, "mAP50"].std()
            print(f"    {c:18s} {v:.4f}  +/- {sd:.4f}")
        print("\n  The spread between folds is the thing to report. A single")
        print("  averaged gap hides that one fold can beat the random split")
        print("  when its validation set happens to hold one easy class.")

    # -------------------------------------------------- per class ----------
    print("\n" + "=" * 78)
    print("  PER CLASS: DOES SITE COVERAGE PREDICT TRANSFER?")
    print("=" * 78)
    sp = pd.read_csv(os.path.join(a.splits, "split_assignment.csv"))
    sp = sp[sp["farm"].notna()]
    cover = {}
    for n in names:
        i = names.index(n)
        farms = set()
        for _, r in sp.iterrows():
            cs = str(r.get("classes", ""))
            if cs and str(i) in cs.split(","):
                farms.add(int(r["farm"]))
        cover[n] = len(farms)

    print(f"  {'class':24s} {'farms':>6s} {'random':>16s} "
          f"{'by-farm':>16s} {'folds':>6s} {'gap':>8s}")
    for n in names:
        col = f"AP50::{n}"
        if col not in df.columns:
            continue
        r = rnd[col].dropna()
        f = farm[col].dropna()
        if not len(r) and not len(f):
            continue
        rs = f"{r.mean():.3f}+/-{r.std():.3f}" if len(r) else "-"
        fs = f"{f.mean():.3f}+/-{f.std():.3f}" if len(f) else "-"
        nf = farm.loc[farm[col].notna(), "config"].nunique()
        gap = (r.mean() - f.mean()) if len(r) and len(f) else float("nan")
        print(f"  {n:24s} {cover.get(n,0):6d} {rs:>16s} {fs:>16s} "
              f"{nf:6d} {gap:+8.3f}")

    print("""
  Read the first and last columns together. If the classes recorded at
  many farms are the ones with a small gap, then site coverage, not box
  count, is what makes a class transferable, and a random split cannot
  show that because it never withholds a site.""")

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
