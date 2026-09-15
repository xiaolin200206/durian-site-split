"""
Package the experiment.

Collects the result tables, the split assignment, one row per trained run
with its training curve endpoint, and a manifest recording what produced
what. Weights are inventoried rather than copied: thirty YOLO11s
checkpoints are about 600 MB and the numbers are what you need to carry
around.

Writes a single zip to Drive, plus a README naming every file and the
question it answers, so the archive is still readable in three months.

Usage, in Colab:
    !cd /content && python package_results.py
    !cd /content && python package_results.py --with-weights   # include .pt
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import zipfile
import argparse
from datetime import datetime, timezone

import pandas as pd
import yaml

# ---------------------------------------------------------------- settings --
WORK = "/content/durian"
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
RUNS = os.path.join(DRIVE, "runs_colab")
RESULTS = os.path.join(DRIVE, "results")

STAGE = "/content/package"
OUT_ZIP = os.path.join(DRIVE, "durian_results.zip")
# ---------------------------------------------------------------------------

FILE_NOTES = {
    "in_region.csv":
        "peninsula only. each trained model scored on its own validation "
        "set. random split versus by-farm folds, five seeds each.",
    "cross_island.csv":
        "the same models scored on the whole Sabah set, 281 images held out "
        "entirely from training.",
    "sabah_by_tree.csv":
        "Sabah scored one tree at a time, to see whether the stability of "
        "the pooled figure is aggregation.",
    "sabah_by_orchard.csv":
        "the same, one orchard at a time.",
    "peninsula_by_burst_15s.csv":
        "peninsula scored one capture burst at a time, each burst scored "
        "with the fold whose validation set is that burst's farm, so it is "
        "held out the way Sabah is. this is the like-for-like comparison.",
    "split_assignment.csv":
        "one row per peninsular image: farm, how the farm was established, "
        "and which side of every split it falls on.",
    "runs_summary.csv":
        "one row per trained run: config, seed, epochs actually run, best "
        "epoch, final losses, and the checkpoint hash.",
}


def sha1_of(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def collect_runs():
    """One row per run directory, read from its own results.csv and args."""
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        best = os.path.join(d, "weights", "best.pt")
        res = os.path.join(d, "results.csv")
        args = os.path.join(d, "args.yaml")

        row = {"run": name,
               "has_weights": os.path.isfile(best)}

        if os.path.isfile(args):
            try:
                with open(args, encoding="utf-8") as fh:
                    y = yaml.safe_load(fh) or {}
                for k in ("data", "imgsz", "epochs", "batch", "seed",
                          "patience", "model", "optimizer", "lr0"):
                    if k in y:
                        row[k] = y[k]
            except Exception:
                pass

        if os.path.isfile(res):
            try:
                r = pd.read_csv(res)
                r.columns = [c.strip() for c in r.columns]
                row["epochs_run"] = len(r)
                m = [c for c in r.columns if "mAP50(B)" in c
                     and "95" not in c]
                if m:
                    row["best_mAP50"] = float(r[m[0]].max())
                    row["best_epoch"] = int(r[m[0]].idxmax()) + 1
                m95 = [c for c in r.columns if "mAP50-95(B)" in c]
                if m95:
                    row["best_mAP50_95"] = float(r[m95[0]].max())
                for c in r.columns:
                    if c.startswith("train/") and "loss" in c:
                        row[c.replace("train/", "final_")] = float(
                            r[c].iloc[-1])
            except Exception as e:
                row["results_error"] = repr(e)

        if row["has_weights"]:
            row["weights_mb"] = round(os.path.getsize(best) / 1e6, 1)
            row["weights_sha1"] = sha1_of(best)[:16]

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-weights", action="store_true",
                    help="include best.pt for every run (large)")
    ap.add_argument("--out", default=OUT_ZIP)
    a = ap.parse_args()

    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    # ------------------------------------------------------------ tables --
    tables = os.path.join(STAGE, "tables")
    os.makedirs(tables)
    copied = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        shutil.copy2(p, tables)
        copied.append(os.path.basename(p))
    print(f"result tables: {len(copied)}")
    for c in copied:
        print(f"  {c}")

    # -------------------------------------------------------------- runs --
    runs = collect_runs()
    runs_csv = os.path.join(tables, "runs_summary.csv")
    runs.to_csv(runs_csv, index=False, encoding="utf-8-sig")
    copied.append("runs_summary.csv")
    print(f"\nruns inventoried: {len(runs)} "
          f"({int(runs['has_weights'].sum())} with weights)")
    if "epochs_run" in runs.columns:
        er = runs["epochs_run"].dropna()
        if len(er):
            print(f"  epochs actually run: median {er.median():.0f}, "
                  f"range {er.min():.0f}-{er.max():.0f}")
            if er.max() < 150:
                print("  every run stopped early; the epoch budget was not "
                      "the binding constraint")

    # ------------------------------------------------------------ splits --
    splits_src = os.path.join(WORK, "splits")
    if os.path.isdir(splits_src):
        dst = os.path.join(STAGE, "splits")
        os.makedirs(dst)
        for d in sorted(os.listdir(splits_src)):
            sd = os.path.join(splits_src, d)
            if not os.path.isdir(sd):
                continue
            os.makedirs(os.path.join(dst, d), exist_ok=True)
            for f in ("data.yaml", "train.txt", "val.txt"):
                p = os.path.join(sd, f)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(dst, d, f))
        print(f"\nsplit lists copied for {len(os.listdir(dst))} configs")

    # ----------------------------------------------------- dataset stats --
    stats = {}
    for tag in ("merged_peninsula", "merged_sabah"):
        d = os.path.join(WORK, "dataset_store", tag)
        if not os.path.isdir(d):
            continue
        imgs = glob.glob(os.path.join(d, "images", "*.*"))
        boxes = 0
        per_class = {}
        for lp in glob.glob(os.path.join(d, "labels", "*.txt")):
            with open(lp, encoding="utf-8") as fh:
                for line in fh:
                    v = line.split()
                    if v:
                        boxes += 1
                        per_class[int(float(v[0]))] = per_class.get(
                            int(float(v[0])), 0) + 1
        with open(os.path.join(d, "data.yaml"), encoding="utf-8") as fh:
            names = yaml.safe_load(fh)["names"]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        stats[tag] = {"images": len(imgs), "boxes": boxes,
                      "classes": names,
                      "boxes_per_class": {names[i]: per_class.get(i, 0)
                                          for i in range(len(names))}}
    if stats:
        with open(os.path.join(STAGE, "dataset_stats.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        print("\ndataset:")
        for k, v in stats.items():
            print(f"  {k}: {v['images']} images, {v['boxes']} boxes")

    # ---------------------------------------------------------- weights ---
    if a.with_weights:
        wd = os.path.join(STAGE, "weights")
        os.makedirs(wd)
        n = 0
        for _, r in runs[runs["has_weights"]].iterrows():
            src = os.path.join(RUNS, r["run"], "weights", "best.pt")
            shutil.copy2(src, os.path.join(wd, f"{r['run']}.pt"))
            n += 1
        print(f"\nweights copied: {n}")

    # ----------------------------------------------------------- readme ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Durian detection: split-granularity experiment",
        "",
        f"Packaged {now}.",
        "",
        "## What this measures",
        "",
        "A detector was trained on peninsular Malaysian durian imagery and",
        "scored three ways: on a random split of the same images, on folds",
        "that withhold a whole farm, and on a Sabah set that was never seen",
        "at all. Every configuration was repeated across seeds, so the",
        "spread attributable to training noise can be separated from the",
        "spread attributable to which site is being scored.",
        "",
        "## Files",
        "",
    ]
    for f in sorted(set(copied)):
        note = FILE_NOTES.get(f, "")
        lines.append(f"- `tables/{f}`" + (f"  -  {note}" if note else ""))
    lines += [
        "",
        "- `splits/` - the exact image lists behind every configuration.",
        "- `dataset_stats.json` - image and box counts per class per region.",
    ]
    if a.with_weights:
        lines.append("- `weights/` - best.pt for every run.")
    lines += [
        "",
        "## Reading the numbers",
        "",
        "`in_region.csv` and `cross_island.csv` report pooled scores. Pooled",
        "scores are stable and hide the thing worth reporting, which is in",
        "`sabah_by_tree.csv` and `peninsula_by_burst_15s.csv`: the same",
        "weights score very differently depending on which single site they",
        "are asked about. Compare the coefficient of variation between those",
        "two files rather than the means, since the means are close and the",
        "spreads are not.",
        "",
        "In `peninsula_by_burst_15s.csv` each burst is scored with the fold",
        "whose validation set is that burst's farm. Scoring bursts with the",
        "random-split weights instead measures fit, not generalisation, and",
        "is not comparable to the Sabah figures.",
        "",
        "## Caveats carried by these files",
        "",
        "- Bursts are a proxy for a tree, derived from capture timestamps at",
        "  a 15 s gap, because peninsular filenames carry no tree id.",
        "- Sabah's thirteen trees sit in two orchards under one manager,",
        "  photographed across four days. Its internal homogeneity is a",
        "  property of that sample, not of the island.",
        "- Subsets below five images are reported but should be read as",
        "  noise; several appear in both per-site tables.",
        "- leaf_hopper_damage has 12 boxes in Sabah and 134 in the",
        "  peninsula; no figure for that class supports interpretation.",
    ]
    with open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # -------------------------------------------------------------- zip ---
    if os.path.exists(a.out):
        os.remove(a.out)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fns in os.walk(STAGE):
            for fn in fns:
                p = os.path.join(dp, fn)
                zf.write(p, os.path.relpath(p, STAGE))

    size = os.path.getsize(a.out) / 1e6
    print(f"\nwrote {a.out}  ({size:.1f} MB)")
    print(f"sha1 {sha1_of(a.out)}")
    print("\ncontents:")
    for dp, _, fns in os.walk(STAGE):
        for fn in sorted(fns):
            rel = os.path.relpath(os.path.join(dp, fn), STAGE)
            print(f"  {rel}")


if __name__ == "__main__":
    main()
