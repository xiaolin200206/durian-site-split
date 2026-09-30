"""
Evaluate the peninsula at the same granularity as Sabah.

Sabah splits into thirteen trees and the per-tree scores span 2.7x. The
peninsula so far only splits into farms, spanning 2.8x. Those two numbers
look alike but they are not measured on the same unit, so the comparison is
not yet a comparison.

Peninsular filenames carry no tree id. What they do carry is a timestamp,
and the capture pattern is a walk: stand at one target, take a handful of
frames, move on. Consecutive frames less than BURST_GAP seconds apart are
therefore almost always the same target, and a burst is the closest
available proxy for a tree.

This groups the peninsular images into bursts and evaluates each burst
with weights that never saw the farm it belongs to. That last part matters:
the random-split models trained on most of these images, so scoring a burst
with them measures fit, not generalisation, and cannot be set beside the
Sabah figure, which is genuinely held out. Each burst is therefore scored
with the by-farm fold whose validation set is that burst's farm.

If the resulting spread lands near Sabah's 2.7x, the claim becomes:
individual-level variation is about the same size on both islands, and
pooling is what hides it.

No training happens.

Usage, in Colab:
    !cd /content && python peninsula_by_burst.py
    !cd /content && python peninsula_by_burst.py --gap 30 --min-images 5
"""

import os
import re
import sys
import glob
import shutil
import argparse
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- settings --
WORK = "/content/durian"
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
RUNS = os.path.join(DRIVE, "runs_colab")
RESULTS = os.path.join(DRIVE, "results")

MERGED = os.path.join(WORK, "dataset_store", "merged_peninsula")
META = os.path.join(WORK, "metadata_backup.csv")
SCRATCH = os.path.join(WORK, "peninsula_subsets")

IMGSZ = 640
BURST_GAP = 15          # seconds; a longer gap starts a new burst
MIN_IMAGES = 5
# ---------------------------------------------------------------------------


def read_names():
    with open(os.path.join(MERGED, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def boxes_of(stem):
    p = os.path.join(MERGED, "labels", stem + ".txt")
    out = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                v = line.split()
                if len(v) >= 5:
                    out.append(int(float(v[0])))
    return out


def parse_ts(s):
    if not isinstance(s, str):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def build_bursts(meta, gap):
    """stem -> burst id, grouped within farm and ordered by capture time."""
    m = meta.dropna(subset=["farm"]).copy()
    m["ts"] = m.get("DateTimeOriginal", m.get("DateTime")).map(parse_ts)
    m = m.dropna(subset=["ts"]).sort_values(["farm", "ts"])

    stem2burst = {}
    counts = Counter()
    for farm, g in m.groupby("farm"):
        prev = None
        idx = 0
        for _, r in g.iterrows():
            if prev is not None and (r["ts"] - prev).total_seconds() > gap:
                idx += 1
            bid = f"f{int(farm)}b{idx:03d}"
            stem2burst[r["stem"]] = bid
            counts[bid] += 1
            prev = r["ts"]
    return stem2burst, counts


def holdout_map(runs_dir, splits_dir):
    """
    farm -> fold name whose validation set is that farm.
    Read from the split lists rather than assumed, so a change upstream
    cannot silently break the pairing.
    """
    import pandas as pd
    mapping = {}
    assign = os.path.join(RESULTS, "split_assignment.csv")
    if not os.path.isfile(assign):
        return mapping
    df = pd.read_csv(assign)
    for col in [c for c in df.columns if c.startswith("byfarm_fold")]:
        farms = sorted(set(df.loc[df[col] == "val", "farm"].dropna()
                           .astype(int)))
        for f in farms:
            mapping[f] = col
    return mapping


def evaluate(weights, data_yaml, tag, names):
    from ultralytics import YOLO
    r = YOLO(weights).val(data=data_yaml, imgsz=IMGSZ, split="val",
                          batch=8, workers=2,
                          project=os.path.join(WORK, "peninsula_eval_runs"),
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
    ap.add_argument("--gap", type=int, default=BURST_GAP)
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES)
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()

    for p in (MERGED, META):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    names = read_names()

    meta = pd.read_csv(META)
    if "stem" not in meta.columns:
        meta["stem"] = meta["file"].astype(str).str.replace(
            r"\.[^.]+$", "", regex=True)
    meta = meta.drop_duplicates("stem")

    stem2burst, counts = build_bursts(meta, a.gap)
    print(f"bursts at a {a.gap}s gap: {len(counts)}")
    sizes = np.array(list(counts.values()))
    print(f"  images per burst: median {np.median(sizes):.1f}, "
          f"mean {sizes.mean():.1f}, max {sizes.max()}")

    # only images that are actually in the merged set
    present = {os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(MERGED, "images", "*.*"))}
    groups = defaultdict(list)
    for stem, bid in stem2burst.items():
        if stem in present:
            hits = glob.glob(os.path.join(MERGED, "images", stem + ".*"))
            if hits:
                groups[bid].append(hits[0])

    groups = {g: ps for g, ps in groups.items() if len(ps) >= a.min_images}
    print(f"  bursts with at least {a.min_images} images in the merged set: "
          f"{len(groups)}")
    if not groups:
        sys.exit("nothing to evaluate; try a larger --gap or smaller "
                 "--min-images")

    # write one yaml per burst
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
            yaml.safe_dump({"train": txt, "val": txt, "nc": len(names),
                            "names": names}, fh, sort_keys=False,
                           allow_unicode=True)
        made[g] = (y, len(ps))

    burst_farm = {g: int(re.match(r"f(\d+)b", g).group(1)) for g in made}

    fold_for = holdout_map(RUNS, os.path.join(WORK, "splits"))
    if not fold_for:
        sys.exit("could not read split_assignment.csv; run --step splits")

    seeds = sorted({int(re.search(r"_s(\d+)$", os.path.basename(
        os.path.dirname(os.path.dirname(p)))).group(1))
        for p in glob.glob(os.path.join(RUNS, "*_s*", "weights", "best.pt"))})
    if a.models:
        seeds = [int(re.search(r"_s(\d+)$", m).group(1)) for m in a.models]
    print(f"seeds: {seeds}")

    print("\n  farm -> held-out fold used to score its bursts:")
    for f in sorted(set(burst_farm.values())):
        print(f"    farm {f:2d} -> {fold_for.get(f, 'NONE')}")

    missing = [f for f in set(burst_farm.values()) if f not in fold_for]
    if missing:
        print(f"  farms with no held-out fold, skipped: {missing}")

    rows = []
    for seed in seeds:
        done = 0
        for g, (y, n) in sorted(made.items()):
            farm = burst_farm[g]
            fold = fold_for.get(farm)
            if fold is None:
                continue
            m = f"{fold}_s{seed}"
            w = os.path.join(RUNS, m, "weights", "best.pt")
            if not os.path.isfile(w):
                continue
            r = evaluate(w, y, f"{m}__{g}", names)
            r.update({"model": m, "fold": fold, "seed": seed, "burst": g,
                      "farm": farm, "n_images": n})
            rows.append(r)
            done += 1
        print(f"  seed {seed}: {done} bursts scored on held-out weights")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"peninsula_by_burst_{a.gap}s.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ report ---
    g = (df.groupby("burst")
           .agg(farm=("farm", "first"), imgs=("n_images", "first"),
                runs=("model", "size"), mAP50=("mAP50", "mean"),
                sd=("mAP50", "std"))
           .sort_values("mAP50"))

    print("\n" + "=" * 74)
    print(f"  PENINSULA, EVALUATED PER BURST ({a.gap}s gap)")
    print("=" * 74)
    print(g.round(4).to_string())

    lo, hi = g["mAP50"].min(), g["mAP50"].max()
    nz = g[g["mAP50"] > 0]["mAP50"]
    print("\n" + "-" * 74)
    print(f"  bursts        {len(g)}")
    print(f"  mean          {g['mAP50'].mean():.4f}")
    print(f"  sd            {g['mAP50'].std():.4f}")
    print(f"  range         {lo:.4f} - {hi:.4f}")
    n_zero = int((g["mAP50"] == 0).sum())
    if n_zero:
        print(f"  zero-scoring  {n_zero} burst(s); a ratio against zero is "
              f"undefined, so the spread below excludes them")
    if len(nz) >= 2:
        print(f"  spread        {nz.min():.4f} - {nz.max():.4f}   "
              f"({nz.max()/nz.min():.1f}x over non-zero bursts)")
    print(f"  IQR ratio     "
          f"{g['mAP50'].quantile(.75)/max(g['mAP50'].quantile(.25),1e-6):.1f}x"
          f"   (robust to both tails)")
    print(f"  seed sd       {df.groupby('burst')['mAP50'].std().mean():.4f}")

    print("\n  per farm, over its own bursts:")
    pf = (g.groupby("farm")
            .agg(bursts=("mAP50", "size"), mean=("mAP50", "mean"),
                 sd=("mAP50", "std"), lo=("mAP50", "min"),
                 hi=("mAP50", "max")))
    pf["ratio"] = np.where(pf["lo"] > 0, (pf["hi"] / pf["lo"]).round(1),
                           np.nan)
    print(pf.round(4).to_string())

    sab = os.path.join(RESULTS, "sabah_by_tree.csv")
    if os.path.isfile(sab):
        s = pd.read_csv(sab)
        sg = (s.groupby("group").agg(imgs=("n_images", "first"),
                                     mAP50=("mAP50", "mean")))
        sg = sg[sg["imgs"] >= a.min_images]
        def spread(x):
            nzx = x[x > 0]
            r = (nzx.max() / nzx.min()) if len(nzx) >= 2 else float("nan")
            iqr = (x.quantile(.75) / max(x.quantile(.25), 1e-6))
            return r, iqr

        pr, pi = spread(g["mAP50"])
        sr, si = spread(sg["mAP50"])
        print("\n" + "=" * 74)
        print("  SIDE BY SIDE, both on weights that never saw the site")
        print("=" * 74)
        print(f"  {'unit':26s} {'n':>4s} {'mean':>8s} {'sd':>8s} "
              f"{'cv':>7s} {'ratio':>8s} {'IQR':>7s}")
        for label, x, n in (("peninsula, per burst", g["mAP50"], len(g)),
                            ("Sabah, per tree", sg["mAP50"], len(sg))):
            r, i = spread(x)
            print(f"  {label:26s} {n:4d} {x.mean():8.4f} {x.std():8.4f} "
                  f"{x.std()/max(x.mean(),1e-6):7.2f} {r:8.1f} {i:7.1f}")
        print("\n  cv is the coefficient of variation, sd divided by mean; it")
        print("  compares spread between two sets whose means differ.")
        print("""
  If the two ratios are close, individual-level variation is the same size
  on both islands, and crossing the sea adds nothing to it. The instability
  then belongs to the unit of evaluation, not to geography, and a single
  pooled mAP is a statement about how much was pooled.""")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
