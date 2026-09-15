"""
Is the cross-island stability real, or is it aggregation?

Each peninsular fold validates on one farm and the results range from
mAP50 0.139 to 0.394. The Sabah set validates on 281 images from thirteen
trees across two orchards, and every configuration lands between 0.236 and
0.263. The obvious candidate explanation is that Sabah is not a harder or
easier domain, it is simply a larger and more varied sample, so the
site-level variation averages out.

This splits the Sabah set by tree and by orchard and re-evaluates the same
trained weights on each piece. No training happens. If per-tree scores
scatter as widely as the peninsular folds do, the stability was
aggregation. If they stay tight, the domain really is uniform and that is
the finding.

Usage, in Colab after --step cross has run:
    !cd /content && python sabah_by_tree.py
    !cd /content && python sabah_by_tree.py --by orchard
    !cd /content && python sabah_by_tree.py --models random_s42 random_s1
"""

import os
import re
import sys
import glob
import json
import shutil
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- settings --
WORK = "/content/durian"
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
RUNS = os.path.join(DRIVE, "runs_colab")
RESULTS = os.path.join(DRIVE, "results")

SABAH = os.path.join(WORK, "dataset_store", "merged_sabah")
SCRATCH = os.path.join(WORK, "sabah_subsets")

IMGSZ = 640
MIN_IMAGES = 5          # a subset smaller than this is reported but flagged
# ---------------------------------------------------------------------------

TREE_RE = re.compile(r"^(o\d+t\d+|trunk\d+)_", re.I)
ORCH_RE = re.compile(r"^(o\d+)t\d+_", re.I)


def group_of(stem, mode):
    m = (ORCH_RE if mode == "orchard" else TREE_RE).match(stem)
    return m.group(1).lower() if m else None


def read_names():
    with open(os.path.join(SABAH, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def boxes_of(stem):
    p = os.path.join(SABAH, "labels", stem + ".txt")
    out = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                v = line.split()
                if len(v) >= 5:
                    out.append(int(float(v[0])))
    return out


def build_subsets(mode, names):
    """One yaml per group, pointing at a txt list. Images are not copied."""
    imgs = sorted(p for p in glob.glob(os.path.join(SABAH, "images", "*.*"))
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))
    groups = defaultdict(list)
    ungrouped = []
    for p in imgs:
        s = os.path.splitext(os.path.basename(p))[0]
        g = group_of(s, mode)
        (groups[g] if g else ungrouped).append(p)

    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(SCRATCH)

    made = {}
    for g, ps in sorted(groups.items()):
        d = os.path.join(SCRATCH, g)
        os.makedirs(d)
        txt = os.path.join(d, "val.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            for p in ps:
                fh.write(p + "\n")
        y = os.path.join(d, "data.yaml")
        with open(y, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"train": txt, "val": txt,
                            "nc": len(names), "names": names},
                           fh, sort_keys=False, allow_unicode=True)
        made[g] = (y, len(ps))

    if ungrouped:
        print(f"  {len(ungrouped)} images have no {mode} prefix, excluded")
    return made


def evaluate(weights, data_yaml, tag, names):
    from ultralytics import YOLO
    r = YOLO(weights).val(data=data_yaml, imgsz=IMGSZ, split="val",
                          batch=8, workers=2, project=os.path.join(
                              WORK, "sabah_eval_runs"),
                          name=tag, exist_ok=True, verbose=False, plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
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
    ap.add_argument("--by", default="tree", choices=["tree", "orchard"])
    ap.add_argument("--models", nargs="+", default=None,
                    help="run names under runs_colab; default is every "
                         "random_s* model")
    a = ap.parse_args()

    if not os.path.isdir(SABAH):
        sys.exit(f"not found: {SABAH}")

    names = read_names()

    if a.models:
        models = a.models
    else:
        models = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                        for p in glob.glob(os.path.join(
                            RUNS, "random_s*", "weights", "best.pt")))
    if not models:
        sys.exit("no trained weights found under " + RUNS)
    print(f"models: {models}")

    subsets = build_subsets(a.by, names)
    print(f"\n{a.by} subsets: {len(subsets)}")

    # composition of each subset, so a low score can be read against it
    comp = {}
    for g in subsets:
        cnt = Counter()
        n_img = 0
        for p in open(os.path.join(SCRATCH, g, "val.txt"), encoding="utf-8"):
            s = os.path.splitext(os.path.basename(p.strip()))[0]
            n_img += 1
            for c in set(boxes_of(s)):
                cnt[c] += 1
        comp[g] = (n_img, cnt)

    print(f"\n  {'group':10s} {'imgs':>5s}  " +
          " ".join(f"{n[:9]:>9s}" for n in names))
    for g in sorted(subsets):
        n_img, cnt = comp[g]
        print(f"  {g:10s} {n_img:5d}  " +
              " ".join(f"{cnt.get(i,0):9d}" for i in range(len(names))))

    rows = []
    for m in models:
        w = os.path.join(RUNS, m, "weights", "best.pt")
        for g, (y, n) in sorted(subsets.items()):
            r = evaluate(w, y, f"{m}__{g}", names)
            r.update({"model": m, "group": g, "n_images": n})
            rows.append(r)
            print(f"  {m:16s} {g:10s} n={n:4d}  mAP50 {r['mAP50']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"sabah_by_{a.by}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ report ---
    print("\n" + "=" * 74)
    print(f"  SABAH, EVALUATED PER {a.by.upper()}")
    print("=" * 74)
    g = (df.groupby("group")
           .agg(imgs=("n_images", "first"),
                runs=("model", "size"),
                mAP50=("mAP50", "mean"),
                sd=("mAP50", "std"))
           .sort_values("mAP50"))
    print(g.round(4).to_string())

    small = g[g["imgs"] < MIN_IMAGES]
    if len(small):
        print(f"\n  fewer than {MIN_IMAGES} images, treat as noise: "
              f"{list(small.index)}")

    big = g[g["imgs"] >= MIN_IMAGES]
    print("\n" + "-" * 74)
    print(f"  across {a.by}s (n>={MIN_IMAGES}):")
    print(f"    mean   {big['mAP50'].mean():.4f}")
    print(f"    sd     {big['mAP50'].std():.4f}")
    print(f"    range  {big['mAP50'].min():.4f} - {big['mAP50'].max():.4f}"
          f"   ({big['mAP50'].max()/max(big['mAP50'].min(),1e-6):.1f}x)")
    print(f"    within-group sd across seeds: "
          f"{df.groupby('group')['mAP50'].std().mean():.4f}")

    whole = os.path.join(RESULTS, "cross_island.csv")
    if os.path.isfile(whole):
        w = pd.read_csv(whole)
        w = w[w["config"] == "random"]
        if len(w):
            print(f"\n  the same weights on the whole Sabah set: "
                  f"{w['mAP50'].mean():.4f} +/- {w['mAP50'].std():.4f}")

    print("""
  How to read this. The peninsular folds ranged 0.139 to 0.394, a spread of
  2.8x, each fold being one farm. If the per-tree scores here scatter by a
  comparable factor while the whole-set score stays near 0.24, then the
  cross-island stability was an artefact of pooling thirteen trees, and the
  claim to make is about sample composition rather than about geography. If
  the per-tree scores stay tight, the Sabah domain is genuinely uniform and
  that is a different, stronger finding.""")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
