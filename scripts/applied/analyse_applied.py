#!/usr/bin/env python3
"""
analyse_applied.py -- turn the two applied experiments into tables and figures.

Inputs (results_applied/):
  farm_budget.csv             from farm_budget.py --step eval
  new_farm_calibration.csv    from new_farm_calibration.py --step eval   (optional)
  results_durian/split_assignment.csv  for class coverage of each training set

Every aggregate weights held-out farms equally: a run's score is averaged over
seeds and draws within its farm first, then over farms. Intervals are 90%
percentile bootstrap intervals over held-out farms (the unit a new customer is).

Outputs (results_applied/):
  table_farm_budget.csv, table_equal_budget.csv, table_class_coverage.csv,
  table_calibration.csv, fig_farm_budget.{png,pdf}, fig_calibration.{png,pdf}
Usage:
  python scripts/applied/analyse_applied.py --root .
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

NAMES = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid", "Psyllid_damage",
         "leaf_hopper_damage"]
B = 4000
RNG = np.random.default_rng(20260928)

# reference palette (dataviz skill): sequential blue for ordinal k,
# categorical slots 1-2 for the two calibration arms, text/grid tokens
K_COLOURS = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 7: "#0d366b"}
ARM = {"ft": ("#2a78d6", "Fine-tune on new-farm photos"),
       "rt": ("#eb6834", "Retrain with new-farm photos added")}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def boot_ci(per_farm):
    v = np.asarray(per_farm, float)
    if len(v) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, 5), np.percentile(m, 95)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ farm budget --
def class_coverage(root, df):
    """Share of the held-out farm's images whose every class occurs somewhere
    in the training farms. A model cannot find a disease it never saw."""
    man = pd.read_csv(os.path.join(root, "results_durian", "split_assignment.csv"),
                      encoding="utf-8-sig", dtype=str)
    man["cls"] = man["classes"].fillna("").map(
        lambda s: {int(c) for c in s.replace(";", ",").split(",") if c.strip()})
    farm_cls = man.groupby("farm")["cls"].agg(lambda x: set().union(*x))
    out = []
    for (cfg, h, tf), _ in df.groupby(["config", "heldout", "train_farms"]):
        seen = set().union(*[farm_cls[f] for f in str(tf).split(";")])
        imgs = man[man["farm"] == str(h)]
        imgs = imgs[imgs["cls"].map(len) > 0]
        cov = imgs["cls"].map(lambda c: c <= seen).mean()
        out.append({"config": cfg, "class_coverage": cov})
    return pd.DataFrame(out)


def farm_budget(root, out):
    f = os.path.join(out, "farm_budget.csv")
    if not os.path.isfile(f):
        print("no farm_budget.csv yet"); return None
    df = pd.read_csv(f, dtype={"m": str, "heldout": str, "train_farms": str})
    df["m"] = df["m"].astype(str)
    held = df[df["eval_on"] == "heldout"].copy()
    held = held.merge(class_coverage(root, held), on="config", how="left")

    # per (k, m, farm): mean over draws and seeds
    cell = held.groupby(["k", "m", "heldout"]).agg(
        mAP50=("mAP50", "mean"), n_train=("n_train", "mean"),
        coverage=("class_coverage", "mean")).reset_index()
    rows = []
    for (k, m), g in cell.groupby(["k", "m"]):
        lo, hi = boot_ci(g["mAP50"])
        sab = df[(df["eval_on"] == "sabah") & (df["k"] == k) & (df["m"] == m)]
        rows.append({"k": k, "m": m, "farms": len(g),
                     "train_images": round(g["n_train"].mean()),
                     "new_farm_mAP50": round(g["mAP50"].mean(), 4),
                     "ci90_lo": round(lo, 4), "ci90_hi": round(hi, 4),
                     "worst_farm": round(g["mAP50"].min(), 4),
                     "best_farm": round(g["mAP50"].max(), 4),
                     "class_coverage": round(g["coverage"].mean(), 3),
                     "sabah_mAP50": round(sab["mAP50"].mean(), 4) if len(sab) else np.nan})
    tab = pd.DataFrame(rows)
    mo = {"15": 0, "50": 1, "all": 2}
    tab = tab.sort_values(["k", "m"], key=lambda s: s.map(mo) if s.name == "m" else s)
    tab.to_csv(os.path.join(out, "table_farm_budget.csv"), index=False)
    print("\nNew-farm mAP50 by training farms (k) and photos per farm (m)")
    print(tab.to_string(index=False))

    # equal-budget contrasts: same image count, spent on more or fewer farms
    pairs = [((7, "15"), (2, "50"), "7 farms x 15 vs 2 farms x 50  (~100 images)"),
             ((4, "50"), (2, "all"), "4 farms x 50 vs 2 farms x all (~200 images)"),
             ((7, "50"), (4, "all"), "7 farms x 50 vs 4 farms x all (~350-400 images)")]
    eq = []
    for (a_, b_, label) in pairs:
        A = cell[(cell.k == a_[0]) & (cell.m == a_[1])].set_index("heldout")
        Bb = cell[(cell.k == b_[0]) & (cell.m == b_[1])].set_index("heldout")
        common = A.index.intersection(Bb.index)
        if len(common) < 2:
            continue
        d = (A.loc[common, "mAP50"] - Bb.loc[common, "mAP50"]).values
        lo, hi = boot_ci(d)
        eq.append({"contrast": label,
                   "images_a": round(A.loc[common, "n_train"].mean()),
                   "images_b": round(Bb.loc[common, "n_train"].mean()),
                   "mean_diff": round(d.mean(), 4), "ci90_lo": round(lo, 4),
                   "ci90_hi": round(hi, 4), "farms_a_wins": int((d > 0).sum()),
                   "farms": len(d)})
    if eq:
        eq = pd.DataFrame(eq)
        eq.to_csv(os.path.join(out, "table_equal_budget.csv"), index=False)
        print("\nSame number of images, more farms (a) vs fewer farms (b)")
        print(eq.to_string(index=False))

    # does class coverage explain the k effect?
    r = held.groupby("config").agg(mAP50=("mAP50", "mean"),
                                   cov=("class_coverage", "first"),
                                   k=("k", "first")).reset_index()
    rho = r[["mAP50", "cov"]].corr(method="spearman").iloc[0, 1]
    full = r[r["cov"] >= 0.999]
    pd.DataFrame([{"spearman_coverage_vs_mAP50": round(rho, 3),
                   "configs": len(r), "configs_full_coverage": len(full),
                   "mean_mAP50_full_coverage": round(full["mAP50"].mean(), 4),
                   "mean_mAP50_partial": round(r[r["cov"] < 0.999]["mAP50"].mean(), 4)}]
                 ).to_csv(os.path.join(out, "table_class_coverage.csv"), index=False)
    print(f"\nSpearman(class coverage of training set, new-farm mAP50) = {rho:+.2f}")

    fig_farm_budget(tab, held, out)
    return tab


def fig_farm_budget(tab, held, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(SURF)
    style(a); style(b)
    marker = {"15": "o", "50": "s", "all": "D"}
    for k, g in tab.groupby("k"):
        g = g.sort_values("train_images")
        c = K_COLOURS.get(k, INK2)
        a.plot(g["train_images"], g["new_farm_mAP50"], color=c, lw=2, zorder=2)
        for _, r in g.iterrows():
            a.plot([r["train_images"]] * 2, [r["ci90_lo"], r["ci90_hi"]], color=c,
                   lw=1, alpha=.5, zorder=1)
            a.plot(r["train_images"], r["new_farm_mAP50"], marker[r["m"]], ms=7,
                   color=c, mec=SURF, mew=2, zorder=3)
        last = g.iloc[-1]
        a.annotate(f"{k} farm" + ("s" if k > 1 else ""),
                   (last["train_images"], last["new_farm_mAP50"]),
                   xytext=(6, 0), textcoords="offset points", va="center",
                   fontsize=8, color=INK)
    a.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator
    a.xaxis.set_major_locator(FixedLocator([15, 30, 60, 100, 200, 400, 700]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["15", "30", "60", "100", "200", "400", "700"])
    a.set_xlabel("Training images", color=INK2, fontsize=9)
    a.set_ylabel("mAP50 on a farm never seen in training", color=INK2, fontsize=9)
    a.set_title("a  More farms or more photos per farm?", loc="left", fontsize=10,
                color=INK)
    for m, lab in (("15", "15 per farm"), ("50", "50 per farm"), ("all", "all photos")):
        a.plot([], [], marker[m], color=INK2, ms=6, label=lab, ls="none")
    a.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2)

    # b: per-class, m = all, by k
    cols = [f"AP50::{c}" for c in NAMES if f"AP50::{c}" in held]
    sub = held[held["m"] == "all"]
    ks = sorted(sub["k"].unique())
    x = np.arange(len(cols))
    w = 0.8 / max(1, len(ks))
    for i, k in enumerate(ks):
        s = sub[sub["k"] == k]
        per_farm = s.groupby("heldout")[cols].mean()
        vals = per_farm.mean()
        b.bar(x + (i - (len(ks) - 1) / 2) * w, vals.values, w * 0.9,
              color=K_COLOURS.get(k, INK2), label=f"{k} farm" + ("s" if k > 1 else ""),
              edgecolor=SURF, linewidth=1)
    b.set_xticks(x, [c[6:].replace("_", " ") for c in cols], rotation=35,
                 ha="right", fontsize=7.5)
    b.set_ylabel("AP50 on unseen farm", color=INK2, fontsize=9)
    b.set_title("b  By class, all photos", loc="left", fontsize=10, color=INK)
    b.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=len(ks),
             loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
    b.set_title("b  By class, all photos", loc="left", fontsize=10, color=INK,
                pad=22)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_farm_budget.{ext}"), facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------ calibration --
def calibration(out):
    f = os.path.join(out, "new_farm_calibration.csv")
    if not os.path.isfile(f):
        print("\nno new_farm_calibration.csv yet"); return None
    df = pd.read_csv(f, dtype={"m": str, "farm": str})
    base = df[df.arm == "base"].set_index(["farm", "dir", "seed"])["mAP50"]
    df = df[df.arm != "base"].copy()
    df["base"] = [base.get((r.farm, r.dir, r.seed), np.nan) for r in df.itertuples()]
    df["gain"] = df["mAP50"] - df["base"]
    cell = df.groupby(["arm", "m", "farm"]).agg(
        mAP50=("mAP50", "mean"), gain=("gain", "mean"),
        n_calib=("n_calib", "mean")).reset_index()
    bcell = base.groupby("farm").mean()
    rows = [{"arm": "base", "m": "0", "photos": 0, "mAP50": round(bcell.mean(), 4),
             "gain": 0.0, "ci90_lo": np.nan, "ci90_hi": np.nan,
             "farms_improved": np.nan, "farms": len(bcell)}]
    mo = {"5": 0, "10": 1, "20": 2, "all": 3}
    for (arm, m), g in sorted(cell.groupby(["arm", "m"]), key=lambda t: (t[0][0], mo.get(t[0][1], 9))):
        lo, hi = boot_ci(g["gain"])
        rows.append({"arm": arm, "m": m, "photos": round(g["n_calib"].mean()),
                     "mAP50": round(g["mAP50"].mean(), 4),
                     "gain": round(g["gain"].mean(), 4), "ci90_lo": round(lo, 4),
                     "ci90_hi": round(hi, 4),
                     "farms_improved": int((g["gain"] > 0).sum()), "farms": len(g)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "table_calibration.csv"), index=False)
    print("\nNew-farm calibration: gain in mAP50 over the uncalibrated model")
    print(tab.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    fig.patch.set_facecolor(SURF); style(ax)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.annotate("no calibration", (0.02, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    for arm, (c, lab) in ARM.items():
        g = tab[tab.arm == arm]
        if g.empty:
            continue
        ax.fill_between(g["photos"], g["ci90_lo"], g["ci90_hi"], color=c, alpha=.15, lw=0)
        ax.plot(g["photos"], g["gain"], "-o", color=c, lw=2, ms=7, mec=SURF, mew=2,
                label=lab)
    ax.set_xlabel("Photos taken on the new farm", color=INK2, fontsize=9)
    ax.set_ylabel("Change in mAP50 on that farm", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_calibration.{ext}"), facecolor=SURF)
    plt.close(fig)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--results", default=None,
                    help="folder holding the CSVs (default <root>/results_applied)")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.abspath(a.results or os.path.join(root, "results_applied"))
    farm_budget(root, out)
    calibration(out)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    sys.exit(main())
