#!/usr/bin/env python3
"""
Generate the manuscript figures from the released tables.

    pip install matplotlib pandas numpy
    python scripts/analyse/make_figures.py

Figure 1  per-site scores against the pooled figure, both regions.
          This is the paper's argument in one panel: the same weights,
          evaluated one site at a time, land anywhere across a wide band,
          and the published number is the average of that band.

Figure 2  how the reported figure stabilises as more sites are pooled.
          Requires aggregation_curve.py to have been run first.

Figure 3  per-class transfer against the number of farms the class was
          photographed at. Six points; the outlier is the whole argument.

Nothing is trained and no image is read. Figures are written to figures/ as
both PDF (for submission) and PNG (for looking at).
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

# A restrained palette: one colour per region, grey for reference lines.
PEN = "#B4432E"
SAB = "#2E6B8E"
GREY = "#8A8A8A"
LIGHT = "#D8D8D8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def table(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def save(fig, stem):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf and .png")


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------- figure 1 --
def figure1():
    sab = table("sabah_by_tree.csv")
    pen = table("peninsula_by_burst_15s.csv")
    ci = table("cross_island.csv")
    ir = table("in_region.csv")
    if sab is None or pen is None:
        print("  figure 1 skipped: per-site tables missing")
        return

    sg = (sab.groupby("group")
             .agg(imgs=("n_images", "first"), m=("mAP50", "mean"),
                  sd=("mAP50", "std")))
    sg = sg[sg["imgs"] >= 5].sort_values("m")

    pg = (pen.groupby("burst")
             .agg(farm=("farm", "first"), imgs=("n_images", "first"),
                  m=("mAP50", "mean"), sd=("mAP50", "std")))
    pg = pg.sort_values("m")

    pooled_sab = ci["mAP50"].mean() if ci is not None else np.nan
    rnd = ir[ir["config"] == "random"]["mAP50"].mean() if ir is not None else np.nan
    byfarm = (ir[ir["config"].str.startswith("byfarm")]
              .groupby("config")["mAP50"].mean().mean()) if ir is not None else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [34, 12],
                                          "wspace": 0.18})

    for ax, g, colour, label, pooled, pooled_label in (
        (axes[0], pg, PEN, "Peninsula, one capture site at a time", byfarm,
         "pooled, withheld farm"),
        (axes[1], sg, SAB, "Sabah, one tree at a time", pooled_sab,
         "pooled, all 281 images"),
    ):
        x = np.arange(len(g))
        ax.vlines(x, 0, g["m"], color=LIGHT, linewidth=0.8, zorder=1)
        ax.errorbar(x, g["m"], yerr=g["sd"].fillna(0), fmt="o",
                    ms=3.4, color=colour, ecolor=colour, elinewidth=0.8,
                    capsize=0, alpha=0.9, zorder=3,
                    label="site score, ± s.d. over seeds")
        if np.isfinite(pooled):
            ax.axhline(pooled, color="#333333", linewidth=1.1, zorder=2)
            # place the label in whichever corner the data leaves empty
            ax.text(0.02, pooled - 0.030, f"{pooled_label}: {pooled:.3f}",
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="top", fontsize=7.2, color="#333333")

        ax.set_title(label, fontsize=8.4, pad=7, loc="left")
        ax.set_ylim(-0.02, 0.92)
        ax.set_xlim(-1, len(g))
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        strip(ax)

    axes[0].set_ylabel("mAP50")
    axes[1].set_yticklabels([])

    if np.isfinite(rnd):
        axes[0].axhline(rnd, color=GREY, linewidth=1.0, linestyle=(0, (4, 2)),
                        zorder=2)
        axes[0].text(0, rnd + 0.022, f"random split: {rnd:.3f}",
                     ha="left", va="bottom", fontsize=7.2, color=GREY)

    axes[0].text(0, -0.155, "each marker is one site, ordered by score; "
                 "weights never saw that site",
                 transform=axes[0].transAxes, fontsize=7, color="#666666")

    save(fig, "fig1_per_site")


# ---------------------------------------------------------------- figure 2 --
def figure2():
    a = table("aggregation_curve_sabah.csv")
    b = table("aggregation_curve_peninsula.csv")
    if a is None and b is None:
        print("  figure 2 skipped: run aggregation_curve.py first")
        return

    fig, ax = plt.subplots(figsize=(4.0, 3.1))

    for d, colour, label in ((b, PEN, "Peninsula, capture sites"),
                             (a, SAB, "Sabah, trees")):
        if d is None:
            continue
        ax.fill_between(d["k_sites"], d["p05"], d["p95"],
                        color=colour, alpha=0.16, linewidth=0)
        ax.plot(d["k_sites"], d["mean"], color=colour, linewidth=1.4,
                label=label)

    ax.set_xlabel("sites pooled into the reported figure")
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0.6, None)
    ax.legend(loc="upper right", fontsize=7.4)
    strip(ax)
    ax.text(0, -0.185, "shaded: 90% interval over random subsets of that size",
            transform=ax.transAxes, fontsize=7, color="#666666")

    save(fig, "fig2_aggregation")


# ---------------------------------------------------------------- figure 3 --
def figure3():
    ir = table("in_region.csv")
    ci = table("cross_island.csv")
    sa = table("split_assignment.csv")
    if ir is None or ci is None or sa is None:
        print("  figure 3 skipped: tables missing")
        return

    classes = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid",
               "Psyllid_damage", "leaf_hopper_damage"]

    sa = sa[sa["farm"].notna()]
    farms = {}
    for i, c in enumerate(classes):
        s = set()
        for _, r in sa.iterrows():
            v = str(r.get("classes", ""))
            if v not in ("", "nan") and str(i) in v.split(","):
                s.add(int(r["farm"]))
        farms[c] = len(s)

    rnd = ir[ir["config"] == "random"]
    rows = []
    for c in classes:
        col = f"AP50::{c}"
        if col in rnd.columns and col in ci.columns:
            rows.append({"cls": c, "farms": farms[c],
                         "random": rnd[col].dropna().mean(),
                         "sabah": ci[col].dropna().mean()})
    d = pd.DataFrame(rows)
    if d.empty:
        print("  figure 3 skipped: no per-class columns")
        return
    d["gap"] = d["random"] - d["sabah"]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    # several classes share a farm count, so offset them horizontally
    d = d.sort_values(["farms", "random"]).reset_index(drop=True)
    d["x"] = d["farms"].astype(float)
    for f, grp in d.groupby("farms"):
        n = len(grp)
        if n > 1:
            offs = np.linspace(-0.16, 0.16, n)
            d.loc[grp.index, "x"] = f + offs

    for _, r in d.iterrows():
        ax.plot([r.x, r.x], [r.sabah, r["random"]],
                color=LIGHT, linewidth=1.2, zorder=1)
    ax.scatter(d["x"], d["random"], s=30, color=GREY, zorder=3,
               label="random split")
    ax.scatter(d["x"], d["sabah"], s=30, color=SAB, zorder=3,
               label="Sabah, held out")

    for _, r in d.iterrows():
        top = max(r["random"], r["sabah"])
        ax.annotate(r.cls.replace("_", " "), (r.x, top + 0.035),
                    fontsize=6.6, ha="center", va="bottom", color="#444444",
                    rotation=0)

    ax.set_xlabel("farms at which the class was photographed")
    ax.set_ylabel("AP50")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.6, 5.6)
    ax.set_ylim(-0.06, 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, fontsize=7.4)
    strip(ax)
    ax.text(0, -0.34, "a line joins the two evaluations of one class",
            transform=ax.transAxes, fontsize=7, color="#666666")

    save(fig, "fig3_site_coverage")


def main():
    print("generating figures from", RESULTS)
    figure1()
    figure2()
    figure3()
    print("\nCaptions are in docs/figure_captions.md.")


if __name__ == "__main__":
    main()
