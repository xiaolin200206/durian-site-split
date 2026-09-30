#!/usr/bin/env python3
"""
How many sites does a reported figure average over?

The paper's title claims a single accuracy figure describes the sample. The
direct evidence for that is the relationship between how many sites an
evaluation pools and how much the resulting number moves: pool one tree and
the answer ranges 0.201 to 0.540; pool thirteen and it is 0.246 with a
standard deviation of 0.009 across thirty models.

This computes the curve between those two points by resampling. For each
k from 1 to the number of available sites, it draws many random subsets of k
sites and records the spread of their mean score. The output is the sampling
distribution of a k-site evaluation.

What this is and is not. It resamples site-level mAP values, so it gives the
spread of the *mean of k site scores*, not a recomputed pooled mAP over the
union of their images. Those differ: pooled mAP weights a site by how many
instances it contributes, the mean of site scores weights every site equally.
The equal-weight version is the one the argument needs — it answers "if I
evaluate at k sites, how much does my headline number depend on which k" —
and it is stated as such rather than presented as a pooled figure.

Both regions are computed where data allow, since the claim is that this is a
property of site-level variation and not of either island.

Usage:
    python aggregation_curve.py
    python aggregation_curve.py --draws 5000 --min-images 5
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")

DRAWS = 4000
MIN_IMAGES = 5
SEED = 0


def load_sites(path, group_col, size_col, score_col="mAP50", min_images=5):
    p = os.path.join(RESULTS, path)
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    g = (d.groupby(group_col)
           .agg(imgs=(size_col, "first"), score=(score_col, "mean"),
                seed_sd=(score_col, "std")))
    return g[g["imgs"] >= min_images]


def curve(scores, draws, rng):
    """Sampling distribution of the mean of k sites, for every k."""
    n = len(scores)
    rows = []
    for k in range(1, n + 1):
        if k == n:
            means = np.array([scores.mean()])
        else:
            idx = np.array([rng.choice(n, size=k, replace=False)
                            for _ in range(draws)])
            means = scores[idx].mean(axis=1)
        rows.append({
            "k_sites": k,
            "mean": float(means.mean()),
            "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "p05": float(np.percentile(means, 5)),
            "p95": float(np.percentile(means, 95)),
            "lo": float(means.min()),
            "hi": float(means.max()),
        })
    df = pd.DataFrame(rows)
    df["spread_p05_p95"] = df["p95"] - df["p05"]
    df["cv"] = df["sd"] / df["mean"].clip(lower=1e-9)
    return df


def report(name, g, draws, rng, out_csv):
    scores = g["score"].to_numpy()
    n = len(scores)
    print("\n" + "=" * 74)
    print(f"  {name}: {n} sites")
    print("=" * 74)
    print(f"  site scores: {np.sort(scores).round(3).tolist()}")
    print(f"  one site   : {scores.min():.3f} to {scores.max():.3f}   "
          f"({scores.max()/max(scores.min(),1e-9):.1f}x)")
    print(f"  all sites  : {scores.mean():.3f}")
    print(f"  mean seed sd within a site: {g['seed_sd'].mean():.4f}")

    df = curve(scores, draws, rng)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n  {'k':>3s} {'mean':>7s} {'sd':>7s} {'5th':>7s} {'95th':>7s} "
          f"{'90% range':>10s} {'cv':>6s}")
    for _, r in df.iterrows():
        print(f"  {int(r.k_sites):3d} {r['mean']:7.3f} {r['sd']:7.3f} "
              f"{r.p05:7.3f} {r.p95:7.3f} {r.spread_p05_p95:10.3f} "
              f"{r.cv:6.2f}")

    one = df[df.k_sites == 1].iloc[0]
    allk = df[df.k_sites == n].iloc[0]
    half = df[df.k_sites == max(1, n // 2)].iloc[0]
    print(f"\n  the 90% interval on a reported figure narrows from "
          f"{one.spread_p05_p95:.3f} at one site")
    print(f"  to {half.spread_p05_p95:.3f} at {int(half.k_sites)} sites, "
          f"and vanishes at {n} only because there is nothing left to draw.")
    print(f"  the mean is {allk['mean']:.3f} throughout: pooling does not "
          f"change the expected value,")
    print("  only how far a particular evaluation can land from it.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    outputs = {}

    sab = load_sites("sabah_by_tree.csv", "group", "n_images",
                     min_images=a.min_images)
    if sab is not None and len(sab) >= 3:
        outputs["sabah"] = report(
            "Sabah, per tree", sab, a.draws, rng,
            os.path.join(RESULTS, "aggregation_curve_sabah.csv"))
    else:
        print("sabah_by_tree.csv not usable")

    pen = load_sites("peninsula_by_burst_15s.csv", "burst", "n_images",
                     min_images=a.min_images)
    if pen is not None and len(pen) >= 3:
        outputs["peninsula"] = report(
            "Peninsula, per capture burst", pen, a.draws, rng,
            os.path.join(RESULTS, "aggregation_curve_peninsula.csv"))

    # ------------------------------------------------------ the headline ---
    if "sabah" in outputs:
        d = outputs["sabah"]
        one = d[d.k_sites == 1].iloc[0]
        print("\n" + "=" * 74)
        print("  THE SENTENCE THIS SUPPORTS")
        print("=" * 74)
        print(f"""
  Evaluated at a single tree, the same weights return anything between
  {sab['score'].min():.3f} and {sab['score'].max():.3f}. Pooled over all
  {len(sab)}, they return {sab['score'].mean():.3f}, and thirty independently
  trained models agree on that pooled value to within 0.009. The pooled
  figure is stable because it is an average, and it is silent about which
  end of the range any particular orchard sits at. A 90% interval of
  {one.spread_p05_p95:.3f} at one site is the uncertainty a grower faces;
  the published number reports none of it.""")

    print(f"\nwrote aggregation_curve_*.csv to {RESULTS}")
    print("plot with: python scripts/analyse/make_figures.py")


if __name__ == "__main__":
    main()
