#!/usr/bin/env python3
"""
paper_analysis.py -- every number, table and figure in the applied manuscript.

One script, one output folder. The manuscript quotes results_applied/paper_numbers.json;
verify_applied.py checks the manuscript text against that file.

Inputs
  results_clean/            clean-protocol leave-one-farm-out runs (RQ1)
  results_durian/split_assignment.csv
  results_applied/farm_budget.csv            (RQ2, from farm_budget.py)
  results_applied/new_farm_calibration.csv   (RQ3, optional)

Outputs (results_applied/)
  paper_numbers.json
  table1_farms.csv  table2_per_class.csv  table3_budget_grid.csv
  table4_equal_budget.csv  table5_regression.csv  tableS_per_farm.csv ...
  fig1_data.{png,pdf}  fig2_new_farm_cost.{png,pdf}
  fig3_budget.{png,pdf}  fig4_class_by_k.{png,pdf}  [fig5_calibration]

Aggregation rule used everywhere: average within a held-out farm first (over
seeds and draws), then weight farms equally. Intervals are 90% percentile
bootstrap intervals that resample held-out farms (4,000 resamples, fixed seed).
Usage:  python scripts/applied/paper_analysis.py --root .
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd

NAMES = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid", "Psyllid_damage",
         "leaf_hopper_damage"]
LABEL = {"Algal": "Algal spot", "Leaf_rot": "Leaf rot", "Phomopsis": r"$\it{Phomopsis}$",
         "Psyllid": "Psyllid", "Psyllid_damage": "Psyllid damage",
         "leaf_hopper_damage": "Leafhopper damage"}
PEST = {"Psyllid", "Psyllid_damage", "leaf_hopper_damage"}
DETECTORS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "rtdetr-l", "frcnn-r50"]
B = 4000
SEED = 20260928

# reference palette (dataviz skill, light mode)
BLUE = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 7: "#0d366b"}
ACCENT, ACCENT2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a9993", "#e4e3df", "#ffffff"


def rng():
    return np.random.default_rng(SEED)


def boot_mean(v, r=None):
    v = np.asarray(v, float)
    r = r or rng()
    idx = r.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def rd(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def style(ax, ygrid=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7.5, width=0.8)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=9, color=INK, pad=6)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), facecolor=SURF, dpi=300)


# ------------------------------------------------------------------ inputs --
def manifest(root):
    m = pd.read_csv(os.path.join(root, "results_durian", "split_assignment.csv"),
                    encoding="utf-8-sig", dtype=str)
    m["cls"] = m["classes"].fillna("").map(
        lambda s: {int(c) for c in s.replace(";", ",").split(",") if c.strip()})
    return m


def clean_runs(root):
    fs = ["durian_in_region_clean.csv", "durian_in_region_clean_yolo11l.csv"]
    d = pd.concat([pd.read_csv(os.path.join(root, "results_clean", f), encoding="utf-8-sig")
                   for f in fs], ignore_index=True)
    fr = pd.read_csv(os.path.join(root, "results_clean", "durian_in_region_clean_frcnn.csv"),
                     encoding="utf-8-sig")
    return pd.concat([d, fr], ignore_index=True)


# --------------------------------------------------------------------- RQ1 --
def rq1(root, out, N):
    man = manifest(root)
    # Table 1: farms
    t1 = []
    for f, g in man.groupby("farm"):
        row = {"farm": int(f), "images": len(g),
               "images_with_boxes": int((g["cls"].map(len) > 0).sum()),
               "boxes": int(g["n_boxes"].astype(int).sum())}
        for i, c in enumerate(NAMES):
            row[c] = int(g["cls"].map(lambda s: i in s).sum())
        t1.append(row)
    t1 = pd.DataFrame(t1).sort_values("farm")
    t1.to_csv(os.path.join(out, "table1_farms.csv"), index=False)
    N["farms"] = {"n_farms": len(t1), "images": int(t1.images.sum()),
                  "min_images": int(t1.images.min()), "max_images": int(t1.images.max()),
                  "boxes": int(t1.boxes.sum()),
                  "images_with_boxes": int(t1.images_with_boxes.sum()),
                  "images_without_boxes": int(t1.images.sum() - t1.images_with_boxes.sum()),
                  "classes_present_per_farm": {int(r["farm"]): int(sum(r[c] > 0 for c in NAMES))
                                               for _, r in t1.iterrows()},
                  "farms_per_class": {c: int((t1[c] > 0).sum()) for c in NAMES}}

    # headline: random split vs unseen farm, per detector
    sm = pd.read_csv(os.path.join(root, "results_clean", "summary_main.csv"), encoding="utf-8-sig")
    fm = sm[(sm.dataset == "durian") & (sm.unit == "farm")]
    N["headline"] = {r.model: {"random_split": rd(r.item), "unseen_farm": rd(r.unit_level),
                               "drop_pct": rd(r.overstatement_pct, 1)}
                     for r in fm.itertuples()}
    # raw per-run tables: every per-farm and Sabah number is computed from these
    raw = clean_runs(root)
    raw = raw[raw.checkpoint == "best"]
    lofo = raw[raw.config.astype(str).str.startswith("byfarm")].copy()
    lofo["farm"] = lofo.config.str.replace("byfarm_fold", "").astype(int)

    # Sabah: leave-one-farm-out models only (seven peninsular farms each)
    sb = lofo[lofo.eval_on == "sabah"]
    sab = sb.groupby(["model", "farm"]).mAP50.mean().groupby("model").mean()
    N["sabah_clean"] = {m: rd(v) for m, v in sab.items()}
    five = fm[fm.model.isin(sab.index)]
    N["headline_range"] = {
        "random_min": rd(fm.item.min(), 3), "random_max": rd(fm.item.max(), 3),
        "unseen_min": rd(fm.unit_level.min(), 3), "unseen_max": rd(fm.unit_level.max(), 3),
        "drop_min": rd(fm.overstatement_pct.min(), 1), "drop_max": rd(fm.overstatement_pct.max(), 1),
        "drop_one_stage_min": rd(fm[fm.model != "frcnn-r50"].overstatement_pct.min(), 1),
        "random_mean": rd(fm.item.mean(), 3),
        "ratio_min": rd((fm.item / fm.unit_level).min(), 2),
        "ratio_max": rd((fm.item / fm.unit_level).max(), 2),
        "sabah_min": rd(sab.min(), 3), "sabah_max": rd(sab.max(), 3),
        "peninsula_five_min": rd(five.unit_level.min(), 3),
        "peninsula_five_max": rd(five.unit_level.max(), 3)}

    # per farm, averaged over the six detectors (unrounded, from the raw runs)
    out_ = lofo[lofo.eval_on == "outer"]
    w = out_.groupby(["farm", "model"]).mAP50.mean().unstack()
    w = w[[m for m in DETECTORS if m in w.columns]]
    per_farm = w.mean(axis=1)
    rho = {(a, b): w[a].rank().corr(w[b].rank()) for a, b in combinations(w.columns, 2)}
    worst2 = {m: tuple(sorted(w[m].nsmallest(2).index)) for m in w.columns}
    lo_pair = min(rho, key=rho.get)
    N["per_farm"] = {"mean_over_detectors": {int(k): rd(v, 3) for k, v in per_farm.items()},
                     "min": rd(per_farm.min(), 3), "max": rd(per_farm.max(), 3),
                     "ratio_best_worst": rd(per_farm.max() / per_farm.min(), 1),
                     "mean_pairwise_spearman": rd(np.mean(list(rho.values())), 2),
                     "min_pairwise_spearman": rd(rho[lo_pair], 2),
                     "min_pair": list(lo_pair),
                     "same_two_worst_all_detectors": len(set(worst2.values())) == 1,
                     "two_worst": list(next(iter(worst2.values())))}
    w.round(6).to_csv(os.path.join(out, "tableS_per_farm_detectors.csv"))
    wins = w.idxmax(axis=1).value_counts()
    N["per_farm"]["max_range_within_farm"] = rd((w.max(axis=1) - w.min(axis=1)).max(), 3)
    N["per_farm"]["max_farms_won_by_one_detector"] = int(wins.max())
    N["per_farm"]["detectors_winning_a_farm"] = int(len(wins))
    y11 = out_[out_.model == "yolo11n"].groupby("farm").mAP50.std()
    N["seed_sd_yolo11n_per_farm"] = rd(y11.mean(), 3)
    big = w.mean().sort_values()
    N["detector_gap"] = {"best": big.index[-1], "best_unseen": rd(big.iloc[-1], 3),
                         "yolo11n_unseen": rd(big["yolo11n"], 3),
                         "yolo11n_below_best": rd(big.iloc[-1] - big["yolo11n"], 3)}

    # capture conditions and annotation scale (recorded at collection; protocol-independent)
    cc = pd.read_csv(os.path.join(root, "results_durian", "farm_capture_conditions.csv"))
    cc = cc[cc.farm.astype(str).isin([str(x) for x in t1.farm])]
    N["capture"] = {"farms_single_date": int((cc.dates == 1).sum()),
                    "farms_two_dates": int((cc.dates == 2).sum()),
                    "farms_two_devices": int((cc.n_devices == 2).sum())}
    sc = pd.read_csv(os.path.join(root, "results_durian", "farm_focal_annotation_scale.csv"))
    sc = sc[(sc.focal == "ALL") & sc.farm.astype(str).isin([str(x) for x in t1.farm])]
    side = sc.groupby("class").apply(lambda g: (g.side_px * g.n_boxes).sum() / g.n_boxes.sum(),
                                     include_groups=False)
    N["box_side_px"] = {c: rd(side[c], 0) for c in NAMES}

    # optional supplementary evaluations with the same weights (extra_eval_clean.py)
    xf = os.path.join(out, "extra_eval_clean.csv")
    if os.path.isfile(xf):
        x = pd.read_csv(xf)
        so = x[x.eval_on.str.startswith("sabah_")].copy()
        if len(so):
            so["farm"] = so.config.str.replace("byfarm_fold", "").astype(int)
            orch = so.groupby(["model", "eval_on", "farm"]).mAP50.mean().groupby(
                ["model", "eval_on"]).mean().unstack()
            N["sabah_by_orchard"] = {m: {o: rd(v, 3) for o, v in r.items()} for m, r in orch.iterrows()}
            N["sabah_orchard_images"] = {o: int(v) for o, v in
                                         so.groupby("eval_on").n_images.first().items()}
            N["sabah_orchard_range"] = {o: [rd(orch[o].min(), 3), rd(orch[o].max(), 3)]
                                        for o in orch.columns}
        rf = x[x.eval_on.str.startswith("random_farm")].copy()
        if len(rf):
            rf["farm"] = rf.eval_on.str.replace("random_farm", "").astype(int)
            rpf = rf.groupby(["model", "farm"]).mAP50.mean().groupby("model").mean()
            N["random_split_farm_weighted"] = {m: rd(v, 3) for m, v in rpf.items()}
            N["random_split_farm_weighted_range"] = [rd(rpf.min(), 3), rd(rpf.max(), 3)]
            N["random_split_test_images_per_farm"] = {int(f): int(n) for f, n in
                                                      rf.groupby("farm").n_images.first().items()}

    # variance shares (crossed design on durian) and model pairs
    dec = pd.read_csv(os.path.join(root, "results_clean", "summary_decomposition.csv"), encoding="utf-8-sig")
    d0 = dec[dec.dataset == "durian"].iloc[0].to_dict()
    N["variance_shares"] = {k: (rd(v, 1) if isinstance(v, (int, float)) else v)
                            for k, v in d0.items() if k not in ("dataset", "design")}
    mp = pd.read_csv(os.path.join(root, "results_clean", "summary_model_pairs.csv"), encoding="utf-8-sig")
    mpd = mp[mp.iloc[:, 0].astype(str).str.startswith("durian")]
    diffcol = mpd.columns[3]
    N["model_pairs"] = {"n_pairs": len(mpd),
                        "max_abs_mean_diff": rd(mpd[diffcol].abs().max(), 3)}

    # checkpoint-selection leakage, yolo11n
    pc = pd.read_csv(os.path.join(root, "results_clean", "summary_protocol_comparison.csv"), encoding="utf-8-sig")
    r = pc[(pc.dataset == "durian") & (pc.model == "yolo11n")].iloc[0]
    N["checkpoint_leakage_yolo11n"] = {"drop_pct_if_heldout_selects": rd(r.old_best, 1),
                                       "drop_pct_clean": rd(r.clean, 1),
                                       "understated_by_points": rd(r.clean - r.old_best, 1)}

    # per class: random split vs unseen farm (five detectors with per-class AP)
    cr = clean_runs(root)
    cr = cr[(cr.checkpoint == "best") & (cr.eval_on == "outer")]
    carriers = {c: [int(f) for f in t1.farm[t1[c] > 0]] for c in NAMES}
    rows = []
    for m in [x for x in DETECTORS if x != "frcnn-r50"]:
        s = cr[cr.model == m]
        for c in NAMES:
            col = f"AP50::{c}"
            rnd = s[s.config == "random"][col].mean()
            nf = [s[s.config == f"byfarm_fold{f}"][col].mean() for f in carriers[c]]
            nf = [v for v in nf if not np.isnan(v)]
            rows.append({"model": m, "class": c, "farms_with_class": len(carriers[c]),
                         "random_split_AP50": rnd, "unseen_farm_AP50": np.mean(nf),
                         "unseen_min": np.min(nf), "unseen_max": np.max(nf)})
    pc_ = pd.DataFrame(rows)
    pc_["retained_pct"] = 100 * pc_.unseen_farm_AP50 / pc_.random_split_AP50
    t2 = pc_.groupby("class").agg(farms=("farms_with_class", "first"),
                                  random_split=("random_split_AP50", "mean"),
                                  unseen_farm=("unseen_farm_AP50", "mean"),
                                  retained_pct=("retained_pct", "mean")).reindex(NAMES)
    # range over farms of the five-detector mean AP on each carrying farm
    for c in NAMES:
        fmeans = []
        for f in carriers[c]:
            v = [cr[(cr.model == m) & (cr.config == f"byfarm_fold{f}")][f"AP50::{c}"].mean()
                 for m in DETECTORS if m != "frcnn-r50"]
            fmeans.append(np.nanmean(v))
        t2.loc[c, "unseen_min"], t2.loc[c, "unseen_max"] = np.nanmin(fmeans), np.nanmax(fmeans)
    y = pc_[pc_.model == "yolo11n"].set_index("class")
    t2["yolo11n_random"] = y.random_split_AP50
    t2["yolo11n_unseen"] = y.unseen_farm_AP50
    t2 = t2.round(6)
    t2.to_csv(os.path.join(out, "table2_per_class.csv"))
    N["per_class"] = {c: {k: rd(v, 3) if k != "farms" else int(v) for k, v in t2.loc[c].items()}
                      for c in NAMES}
    pests = ["Psyllid", "Psyllid_damage"]
    others = [c for c in NAMES if c not in pests]
    N["per_class_summary"] = {
        "psyllid_retained_range": [rd(t2.loc[pests, "retained_pct"].min(), 0),
                                   rd(t2.loc[pests, "retained_pct"].max(), 0)],
        "other_retained_range": [rd(t2.loc[others, "retained_pct"].min(), 0),
                                 rd(t2.loc[others, "retained_pct"].max(), 0)]}
    # sensitivity: only farms holding >= 10 images of the class
    sens = {}
    for c in NAMES:
        far = [int(f) for f in t1.farm[t1[c] >= 10]]
        vals = []
        for m in [x for x in DETECTORS if x != "frcnn-r50"]:
            sm_ = cr[cr.model == m]
            rnd = sm_[sm_.config == "random"][f"AP50::{c}"].mean()
            nf = np.nanmean([sm_[sm_.config == f"byfarm_fold{f}"][f"AP50::{c}"].mean() for f in far])
            vals.append(100 * nf / rnd)
        sens[c] = {"farms": len(far), "retained_pct": rd(np.mean(vals), 0)}
    N["per_class_sensitivity_min10"] = sens
    fig1(t1, out)
    rfw = N.get("random_split_farm_weighted")
    fig2(w, per_farm, fm, t2, out, np.mean(list(rfw.values())) if rfw else None)
    return t1


def fig1(t1, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    M = t1.set_index("farm")[NAMES].T.values
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    fig.patch.set_facecolor(SURF)
    im = ax.imshow(np.where(M > 0, M, np.nan), cmap=cmap, aspect="auto", vmin=0, vmax=M.max())
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, str(v) if v else "–", ha="center", va="center", fontsize=7,
                    color=("#ffffff" if v > 0.55 * M.max() else INK) if v else MUTED)
    ax.set_xticks(range(M.shape[1]), [f"Farm {f}\n({n})" for f, n in zip(t1.farm, t1.images)],
                  fontsize=7, color=INK2)
    ax.set_yticks(range(len(NAMES)), [LABEL[c] for c in NAMES], fontsize=7.5, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color=SURF, lw=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    save(fig, out, "fig1_data")
    plt.close(fig)


def fig2(w, per_farm, fm, t2, out, rfw=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURF)
    style(a); style(b, ygrid=False)
    order = per_farm.sort_values().index
    x = np.arange(len(order))
    rs = fm.item.mean()
    a.axhline(rs, color=INK2, lw=1, ls="--")
    a.annotate(f"random split, pooled (mean of six detectors), {rs:.2f}", (-0.3, rs), xytext=(0, 4),
               textcoords="offset points", ha="left", fontsize=6.8, color=INK2)
    if rfw is not None:
        a.axhline(rfw, color=INK2, lw=1, ls=":")
        a.annotate(f"random split, farm-weighted (mean of five detectors), {rfw:.2f}", (-0.3, rfw),
                   xytext=(0, -9), textcoords="offset points", ha="left", fontsize=6.8, color=INK2)
    for m in w.columns:
        a.plot(x, w.loc[order, m], "o", ms=3.5, color=MUTED, alpha=.8, mec="none", zorder=2)
    a.plot(x, per_farm[order], "D", ms=6.5, color=ACCENT, mec=SURF, mew=1.5, zorder=3,
           label="mean of six detectors")
    a.set_xticks(x, [f"F{f}" for f in order], fontsize=7.5)
    a.set_ylim(0, 0.6)
    a.set_ylabel("mAP50", fontsize=8, color=INK2)
    a.set_xlabel("Held-out farm (ordered by difficulty)", fontsize=8, color=INK2)
    a.plot([], [], "o", ms=3.5, color=MUTED, label="each detector")
    a.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK2,
             bbox_to_anchor=(0, 0.8))
    panel(a, "a", "Each farm, scored by models that never saw it")

    t = t2.sort_values("retained_pct")
    y = np.arange(len(t))
    for i, (c, r) in enumerate(t.iterrows()):
        b.plot([r.unseen_farm, r.random_split], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    b.plot(t.random_split, y, "o", ms=6, mfc=SURF, mec=INK2, mew=1.3, zorder=2, label="random split")
    b.plot(t.unseen_farm, y, "o", ms=6.5, color=ACCENT, mec=SURF, mew=1.2, zorder=3, label="unseen farm")
    for i, (c, r) in enumerate(t.iterrows()):
        b.annotate(f"{r.retained_pct:.0f}%", (r.random_split, i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=7, color=INK2)
    b.set_yticks(y, [LABEL[c] for c in t.index], fontsize=7.5)
    b.set_xlim(0, 0.72)
    b.set_xlabel("AP50 (mean of five detectors)", fontsize=8, color=INK2)
    b.grid(axis="x", color=GRID, lw=0.7)
    b.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK2)
    panel(b, "b", "AP kept on a new farm, by class")
    fig.tight_layout()
    save(fig, out, "fig2_new_farm_cost")
    plt.close(fig)


# --------------------------------------------------------------------- RQ2 --
def coverage(man, df, lists_path):
    """Share of the held-out farm's annotated images whose classes all occur in the
    images actually drawn for training (from farm_budget_train_lists.csv); falls back
    to the training farms' class sets if the lists are absent."""
    stem_cls = dict(zip(man.stem, man["cls"]))
    farm_cls = man.groupby("farm")["cls"].agg(lambda x: set().union(*x))
    lists = None
    if os.path.isfile(lists_path):
        L = pd.read_csv(lists_path)
        lists = L.groupby("config").stem.apply(list).to_dict()
    cov = {}
    for cfg, h, tf in df[["config", "heldout", "train_farms"]].drop_duplicates().itertuples(index=False):
        if lists is not None:
            seen = set().union(*[stem_cls[x] for x in lists[cfg]])
        else:
            seen = set().union(*[farm_cls[f] for f in str(tf).split(";")])
        imgs = man[(man.farm == str(h)) & (man["cls"].map(len) > 0)]
        cov[cfg] = imgs["cls"].map(lambda c: c <= seen).mean()
    return cov, lists is not None


def fit(h, farms_col="heldout", with_coverage=False):
    X = pd.get_dummies(h[farms_col], drop_first=True).astype(float)
    if with_coverage:
        X["coverage"] = h.coverage.values
    X["log2_farms"] = np.log2(h.k.values)
    X["log2_photos_per_farm"] = np.log2(h.per_farm.values)
    X["const"] = 1.0
    b = np.linalg.lstsq(X.values, h.mAP50.values, rcond=None)[0]
    pred = X.values @ b
    y = h.mAP50.values
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b[-3], b[-2], r2


def fit_boot(h, with_coverage=False):
    r = rng()
    farms = sorted(h.heldout.unique())
    groups = {f: h[h.heldout == f] for f in farms}
    out = []
    for _ in range(B):
        s = r.choice(farms, len(farms))
        hh = pd.concat([groups[f].assign(bf=f"{f}_{i}") for i, f in enumerate(s)])
        if hh.k.nunique() < 2 or hh.per_farm.nunique() < 2:
            continue
        out.append(fit(hh, "bf", with_coverage)[:2])
    return np.array(out)


def rq2(root, out, N):
    f = os.path.join(out, "farm_budget.csv")
    if not os.path.isfile(f):
        return None
    man = manifest(root)
    df = pd.read_csv(f, dtype={"m": str, "heldout": str, "train_farms": str})
    df["per_farm"] = df.n_train / df.k
    cov, from_lists = coverage(man, df, os.path.join(out, "farm_budget_train_lists.csv"))
    df["coverage"] = df.config.map(cov)
    h = df[df.eval_on == "heldout"].copy()
    v2 = bool(df.eval_on.str.startswith("sabah_o").any())
    if v2:   # Sabah = mean of the two orchards' scores, as for the leave-one-farm-out models
        s = (df[df.eval_on.str.startswith("sabah_o")].groupby(["run", "k", "m", "heldout"])
             .mAP50.mean().reset_index())
    else:
        s = df[df.eval_on == "sabah"].copy()
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import farm_budget as fbm
    if v2:
        its = [fbm.schedule(int(n))[1] * fbm.schedule(int(n))[2] for n in h.n_train.unique()]
    else:   # v1: epochs = clamp(round(2000 * b / n), 100, 300), b = min(32, n)
        its = []
        for n in h.n_train.unique():
            n = int(n); b = min(32, n)
            its.append(max(100, min(300, round(2000 * b / n))) * -(-n // b))
    N["rq2_design"] = {"runs": int(h.run.nunique()), "seeds": sorted(int(x) for x in h.seed.unique()),
                       "configs": int(h.config.nunique()), "heldout_farms": int(h.heldout.nunique()),
                       "protocol": "v2-fixed-iterations" if v2 else "v1-capped-epochs",
                       "coverage_from_drawn_images": from_lists,
                       "iterations_min": int(min(its)), "iterations_max": int(max(its))}

    mo = {"15": 0, "50": 1, "all": 2}
    cell = h.groupby(["k", "m", "heldout"]).agg(mAP50=("mAP50", "mean"), n=("n_train", "mean"),
                                                cov=("coverage", "mean")).reset_index()
    scell = s.groupby(["k", "m", "heldout"]).mAP50.mean().reset_index()
    grid = []
    for (k, m), g in cell.groupby(["k", "m"]):
        lo, hi = boot_mean(g.mAP50)
        sg = scell[(scell.k == k) & (scell.m == m)]
        grid.append({"k": k, "m": m, "train_images": int(round(g.n.mean())),
                     "unseen_farm_mAP50": g.mAP50.mean(), "ci_lo": lo, "ci_hi": hi,
                     "worst_farm": g.mAP50.min(), "best_farm": g.mAP50.max(),
                     "class_coverage": g["cov"].mean(), "sabah_mAP50": sg.mAP50.mean()})
    grid = pd.DataFrame(grid)
    grid = grid.iloc[sorted(range(len(grid)), key=lambda i: (grid.k[i], mo[grid.m[i]]))]
    grid.to_csv(os.path.join(out, "table3_budget_grid.csv"), index=False, float_format="%.6f")
    N["grid"] = {f"k{r.k}_m{r.m}": {"images": int(r.train_images), "mAP50": rd(r.unseen_farm_mAP50, 3),
                                    "ci": [rd(r.ci_lo, 3), rd(r.ci_hi, 3)],
                                    "coverage": rd(r.class_coverage, 2), "sabah": rd(r.sabah_mAP50, 3)}
                 for r in grid.itertuples()}

    # equal budgets
    pairs = [((7, "15"), (2, "50")), ((4, "50"), (2, "all")), ((7, "50"), (4, "all"))]
    eq, eqrows = [], {}
    for A_, B_ in pairs:
        A = cell[(cell.k == A_[0]) & (cell.m == A_[1])].set_index("heldout")
        Bb = cell[(cell.k == B_[0]) & (cell.m == B_[1])].set_index("heldout")
        ix = A.index.intersection(Bb.index)
        dlt = (A.loc[ix, "mAP50"] - Bb.loc[ix, "mAP50"])
        lo, hi = boot_mean(dlt.values)
        key = f"k{A_[0]}m{A_[1]}_vs_k{B_[0]}m{B_[1]}"
        eqrows[key] = dlt
        eq.append({"contrast": key, "more_farms": f"{A_[0]} farms x {A_[1]}",
                   "fewer_farms": f"{B_[0]} farms x {B_[1]}",
                   "images_more": int(round(A.loc[ix, 'n'].mean())),
                   "images_fewer": int(round(Bb.loc[ix, 'n'].mean())),
                   "mAP_more": A.loc[ix, "mAP50"].mean(), "mAP_fewer": Bb.loc[ix, "mAP50"].mean(),
                   "diff": dlt.mean(), "ci_lo": lo, "ci_hi": hi,
                   "farms_more_wins": int((dlt > 0).sum()), "farms": len(dlt)})
    eq = pd.DataFrame(eq)
    eq.to_csv(os.path.join(out, "table4_equal_budget.csv"), index=False, float_format="%.6f")
    N["equal_budget"] = {r.contrast: {"images": [r.images_more, r.images_fewer],
                                      "mAP": [rd(r.mAP_more, 3), rd(r.mAP_fewer, 3)],
                                      "diff": rd(r.diff, 3), "ci": [rd(r.ci_lo, 3), rd(r.ci_hi, 3)],
                                      "wins": f"{r.farms_more_wins}/{r.farms}"} for r in eq.itertuples()}

    # regression: log2 farms + log2 photos per farm + farm fixed effects
    regs = []
    def reg(name, sub, wc=False):
        bk, bp, r2 = fit(sub, "heldout", wc)
        bb = fit_boot(sub, wc)
        regs.append({"model": name, "n_runs": len(sub), "doubling_farms": bk,
                     "farms_lo": np.percentile(bb[:, 0], 5), "farms_hi": np.percentile(bb[:, 0], 95),
                     "doubling_photos": bp, "photos_lo": np.percentile(bb[:, 1], 5),
                     "photos_hi": np.percentile(bb[:, 1], 95), "diff": bk - bp,
                     "diff_lo": np.percentile(bb[:, 0] - bb[:, 1], 5),
                     "diff_hi": np.percentile(bb[:, 0] - bb[:, 1], 95),
                     "p_farms_gt_photos": float((bb[:, 0] > bb[:, 1]).mean()), "r2": r2})
    reg("all runs", h)
    reg("full class coverage only", h[h.coverage >= 0.999])
    reg("excluding k = 7", h[h.k < 7])
    reg("excluding k = 1", h[h.k > 1])
    reg("adjusted for class coverage", h, True)
    rg = pd.DataFrame(regs)
    rg.to_csv(os.path.join(out, "table5_regression.csv"), index=False, float_format="%.6f")
    N["regression"] = {r.model: {"n_runs": int(r.n_runs), "farms": rd(r.doubling_farms, 3),
                                 "farms_ci": [rd(r.farms_lo, 3), rd(r.farms_hi, 3)],
                                 "photos": rd(r.doubling_photos, 3),
                                 "photos_ci": [rd(r.photos_lo, 3), rd(r.photos_hi, 3)],
                                 "diff": rd(r.diff, 3), "diff_ci": [rd(r.diff_lo, 3), rd(r.diff_hi, 3)],
                                 "p": rd(r.p_farms_gt_photos, 2), "r2": rd(r.r2, 2)}
                       for r in rg.itertuples()}

    # class coverage (of the images actually drawn)
    rc = h.groupby("config").agg(mAP50=("mAP50", "mean"), cov=("coverage", "first"),
                                 k=("k", "first"), m=("m", "first"))
    within = {}
    for k in (1, 2):
        g = rc[rc.k == k]
        within[k] = rd(g[["mAP50", "cov"]].corr(method="spearman").iloc[0, 1], 2)
    covtab = rc.groupby(["k", "m"])["cov"].mean()
    N["coverage"] = {"spearman": rd(rc[["mAP50", "cov"]].corr(method="spearman").iloc[0, 1], 2),
                     "spearman_within_k1": within[1], "spearman_within_k2": within[2],
                     "by_k_m": {f"k{k}_m{m}": rd(v, 2) for (k, m), v in covtab.items()},
                     "k1_mean": rd(rc[rc.k == 1]["cov"].mean(), 2),
                     "k2_mean": rd(rc[rc.k == 2]["cov"].mean(), 2),
                     "k4_min": rd(rc[rc.k == 4]["cov"].min(), 2),
                     "full_coverage_configs": int((rc["cov"] >= 0.999).sum()),
                     "full_coverage_runs_k_ge_4_incomplete": int(((rc.k >= 4) & (rc["cov"] < 0.999)).sum())}

    # step-wise paired contrasts: one more doubling of farms or of photos, per held-out farm
    steps = []
    kk = [1, 2, 4, 7]
    mm = ["15", "50", "all"]
    for m in mm:
        for a_, b_ in zip(kk[:-1], kk[1:]):
            A = cell[(cell.k == b_) & (cell.m == m)].set_index("heldout").mAP50
            Bb = cell[(cell.k == a_) & (cell.m == m)].set_index("heldout").mAP50
            dlt = (A - Bb).dropna()
            lo, hi = boot_mean(dlt.values)
            steps.append({"step": f"farms {a_}->{b_}", "at": f"{m} per farm", "diff": dlt.mean(),
                          "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()), "farms": len(dlt)})
    for k in kk:
        for a_, b_ in zip(mm[:-1], mm[1:]):
            A = cell[(cell.k == k) & (cell.m == b_)].set_index("heldout").mAP50
            Bb = cell[(cell.k == k) & (cell.m == a_)].set_index("heldout").mAP50
            dlt = (A - Bb).dropna()
            lo, hi = boot_mean(dlt.values)
            steps.append({"step": f"photos {a_}->{b_}", "at": f"{k} farms", "diff": dlt.mean(),
                          "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()), "farms": len(dlt)})
    st_ = pd.DataFrame(steps)
    st_.to_csv(os.path.join(out, "tableS_steps.csv"), index=False, float_format="%.6f")
    N["steps"] = {f"{r.step} @ {r.at}": {"diff": rd(r.diff, 3), "ci": [rd(r.ci_lo, 3), rd(r.ci_hi, 3)],
                                         "wins": f"{r.wins}/{r.farms}"} for r in st_.itertuples()}

    # the earlier, epoch-capped schedule (v1) for comparison (Table S7)
    v1f = os.path.join(out, "farm_budget_v1_capped.csv")
    if v2 and os.path.isfile(v1f):
        v1 = pd.read_csv(v1f, dtype={"m": str, "heldout": str})
        v1 = v1[v1.eval_on == "heldout"]
        c1 = v1.groupby(["k", "m", "heldout"]).mAP50.mean().groupby(["k", "m"]).mean()
        c2 = cell.groupby(["k", "m"]).mAP50.mean()
        its1 = {}
        for (k, m), g in v1.groupby(["k", "m"]):
            v = []
            for n in g.n_train.unique():
                n = int(n); b = min(32, n)
                v.append(max(100, min(300, round(2000 * b / n))) * -(-n // b))
            its1[(k, m)] = (min(v), max(v))
        t7 = pd.DataFrame({"v1_capped": c1, "v2_fixed_iterations": c2})
        t7["v1_iterations"] = [f"{its1[i][0]}–{its1[i][1]}" for i in t7.index]
        t7 = t7.reset_index()
        t7 = t7.iloc[sorted(range(len(t7)), key=lambda i: (t7.k[i], mo[t7.m[i]]))]
        t7.to_csv(os.path.join(out, "tableS_schedule_comparison.csv"), index=False, float_format="%.6f")
        b1, p1, _ = fit(v1.assign(per_farm=v1.n_train / v1.k))
        N["v1_schedule"] = {"farms": rd(b1, 3), "photos": rd(p1, 3)}

    # Sabah by orchard in the budget experiment (v2 runs only)
    so = df[df.eval_on.str.startswith("sabah_o")]
    if len(so):
        g = so.groupby(["k", "m", "eval_on", "heldout"]).mAP50.mean().groupby(["k", "m", "eval_on"]).mean()
        N["grid_sabah_orchard"] = {f"k{k}_m{m}": {o: rd(v, 3) for (kk_, mm_, o), v in g.items()
                                                 if kk_ == k and mm_ == m}
                                   for k, m in {(a, b) for a, b, _ in g.index}}

    # per class by k (m = all)
    cols = [f"AP50::{c}" for c in NAMES]
    pk = h[h.m == "all"].groupby(["k", "heldout"])[cols].mean().groupby("k").mean()
    pk.columns = NAMES
    pk.round(6).to_csv(os.path.join(out, "tableS_class_by_k.csv"))
    N["class_by_k"] = {c: {int(k): rd(v, 3) for k, v in pk[c].items()} for c in NAMES}

    # draws and consistency with the clean protocol
    dr = h[h.k < 7].groupby(["heldout", "k", "m"]).mAP50.agg(["min", "max"])
    k7 = cell[(cell.k == 7) & (cell.m == "all")].set_index("heldout").mAP50
    N["checks"] = {"median_draw_spread": rd((dr["max"] - dr["min"]).median(), 3),
                   "k7_all_fixed_steps": rd(k7.mean(), 3),
                   "k7_all_per_farm": {str(i): rd(v, 3) for i, v in k7.sort_index().items()},
                   "clean_protocol_yolo11n": N["headline"]["yolo11n"]["unseen_farm"]}
    mid = eqrows["k4m50_vs_k2mall"]
    N["equal_budget_middle_large_losses"] = int((mid < -0.05).sum())
    N["equal_budget_middle_small_abs"] = rd(mid[mid >= -0.05].abs().max(), 3)
    fig3(grid, eqrows, eq, out)
    fig4(pk, out)
    return grid


def fig3(grid, eqrows, eq, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor(SURF)
    style(a); style(b, ygrid=False)
    marker = {"15": "o", "50": "s", "all": "D"}
    for k, g in grid.groupby("k"):
        g = g.sort_values("train_images")
        c = BLUE[k]
        a.plot(g.train_images, g.unseen_farm_mAP50, color=c, lw=2, zorder=2)
        for r in g.itertuples():
            a.plot([r.train_images] * 2, [r.ci_lo, r.ci_hi], color=c, lw=1, alpha=.45, zorder=1)
            a.plot(r.train_images, r.unseen_farm_mAP50, marker[r.m], ms=6, color=c, mec=SURF,
                   mew=1.5, zorder=3)
        a.plot([], [], "-", color=c, lw=2, label=f"{k} farm" + ("s" if k > 1 else ""))
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator([15, 30, 60, 100, 200, 400, 700]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["15", "30", "60", "100", "200", "400", "700"])
    a.set_xlim(12, 1100)
    a.set_ylim(0, 0.32)
    a.set_xlabel("Training images (log scale)", fontsize=8, color=INK2)
    a.set_ylabel("mAP50 on the unseen farm", fontsize=8, color=INK2)
    for m, lab in (("15", "15 images per farm"), ("50", "50 images per farm"), ("all", "all images")):
        a.plot([], [], marker[m], color=INK2, ms=5, ls="none", label=lab)
    a.legend(frameon=False, fontsize=6.8, loc="upper left", labelcolor=INK2, ncol=2,
             columnspacing=1.0, handlelength=1.6)
    panel(a, "a", "More farms or more images per farm?")

    y = np.arange(len(eq))[::-1]
    b.axvline(0, color=INK2, lw=0.9)
    for yi, r in zip(y, eq.itertuples()):
        d = eqrows[r.contrast].values
        jit = np.linspace(-0.12, 0.12, len(d))
        b.plot(d, yi + jit, "o", ms=3.5, color=MUTED, mec="none", zorder=2)
        b.plot([r.ci_lo, r.ci_hi], [yi, yi], color=ACCENT, lw=2, zorder=3)
        b.plot(r.diff, yi, "D", ms=6.5, color=ACCENT, mec=SURF, mew=1.3, zorder=4)
    b.set_yticks(y, [f"{r.more_farms.replace(' x ', ' × ')}\nvs {r.fewer_farms.replace(' x ', ' × ')}"
                     f"\n(~{r.images_more}/{r.images_fewer} img)" for r in eq.itertuples()],
                 fontsize=6.8)
    b.set_xlabel("Δ mAP50, more farms minus fewer farms", fontsize=8, color=INK2)
    b.grid(axis="x", color=GRID, lw=0.7)
    b.set_xlabel("Δ mAP50, more farms minus fewer farms\n(positive: more farms better)",
                 fontsize=7.5, color=INK2)
    panel(b, "b", "Matched image budgets")
    fig.tight_layout()
    save(fig, out, "fig3_budget")
    plt.close(fig)


def fig4(pk, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.patch.set_facecolor(SURF)
    style(ax)
    ks = list(pk.index)
    order = sorted(NAMES, key=lambda c: -pk.loc[7, c])
    mk = dict(zip(NAMES, ["o", "s", "^", "D", "v", "P"]))
    for c in order:
        pest = c in {"Psyllid", "Psyllid_damage"}
        col = ACCENT2 if pest else ACCENT
        ax.plot(ks, pk[c], "-", marker=mk[c], color=col, lw=1.8 if pest else 1.4, ms=5.5,
                mec=SURF, mew=1.0, alpha=1 if pest else .8)
        ax.annotate(LABEL[c], (ks[-1], pk.loc[ks[-1], c]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(0.85, 16)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("Training farms (all images per farm)", fontsize=8, color=INK2)
    ax.set_ylabel("AP50 on the unseen farm", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig4_class_by_k")
    plt.close(fig)


# --------------------------------------------------------------------- RQ3 --
def rq3(root, out, N):
    f = os.path.join(out, "new_farm_calibration.csv")
    if not os.path.isfile(f):
        N["rq3"] = None
        return None
    df = pd.read_csv(f, dtype={"m": str, "farm": str})
    base = df[df.arm == "base"].set_index(["farm", "dir", "seed"]).mAP50
    d = df[df.arm != "base"].copy()
    d["base"] = [base.get((r.farm, r.dir, r.seed), np.nan) for r in d.itertuples()]
    d["gain"] = d.mAP50 - d.base
    cell = d.groupby(["arm", "m", "farm"]).agg(mAP50=("mAP50", "mean"), gain=("gain", "mean"),
                                               n=("n_calib", "mean")).reset_index()
    bf = base.groupby("farm").mean()
    R = {"base_mAP50": rd(bf.mean(), 3), "farms": int(len(bf)),
         "n_test_median": int(df[df.arm == "base"].n_test.median()),
         "seeds_ft": sorted(int(x) for x in d[d.arm == "ft"].seed.unique()),
         "seeds_rt": sorted(int(x) for x in d[d.arm == "rt"].seed.unique())}
    rows = []
    for arm in ("ft", "rt"):
        R[arm] = {}
        for m in ("5", "10", "20", "all"):
            g = cell[(cell.arm == arm) & (cell.m == m)]
            if g.empty:
                continue
            lo, hi = boot_mean(g.gain.values)
            R[arm][m] = {"images": rd(g.n.mean(), 0), "mAP50": rd(g.mAP50.mean(), 3),
                         "gain": rd(g.gain.mean(), 3), "ci": [rd(lo, 3), rd(hi, 3)],
                         "wins": f"{int((g.gain > 0).sum())}/{len(g)}"}
            rows.append({"arm": arm, "m": m, "images": g.n.mean(), "mAP50": g.mAP50.mean(),
                         "gain": g.gain.mean(), "ci_lo": lo, "ci_hi": hi,
                         "farms_improved": int((g.gain > 0).sum()), "farms": len(g)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "table6_calibration.csv"), index=False, float_format="%.6f")
    N["rq3"] = R
    fig5(tab, R, out)
    return tab


def fig5(tab, R, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    fig.patch.set_facecolor(SURF)
    style(ax)
    ax.axhline(0, color=INK2, lw=0.9, ls="--")
    ax.annotate(f"no calibration (mAP50 {R['base_mAP50']:.3f})", (0.02, 0),
                xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                fontsize=7, color=INK2)
    lab = {"ft": ("Fine-tune on new-farm images", ACCENT), "rt": ("Retrain with them added", ACCENT2)}
    for arm, (name, c) in lab.items():
        g = tab[tab.arm == arm].sort_values("images")
        if g.empty:
            continue
        ax.fill_between(g.images, g.ci_lo, g.ci_hi, color=c, alpha=.15, lw=0)
        ax.plot(g.images, g.gain, "-o", color=c, lw=2, ms=6, mec=SURF, mew=1.3, label=name)
    ax.set_xlabel("Images from the new farm used for calibration", fontsize=8, color=INK2)
    ax.set_ylabel("Change in mAP50 on that farm", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig5_calibration")
    plt.close(fig)


def derive(N, root):
    """Numbers the text quotes that are simple functions of the ones above."""
    so = pd.read_csv(os.path.join(root, "results_durian", "sabah_by_orchard.csv"), encoding="utf-8-sig")
    per_orch = so.groupby("group").n_images.first()
    N.setdefault("sabah_orchard_images", {f"sabah_{k}": int(v) for k, v in per_orch.items()})
    N["sabah_images"] = int(sum(N["sabah_orchard_images"].values()))
    meta = json.load(open(os.path.join(root, "results_clean", "durian_split_meta.json")))
    N["random_test_images"] = int(meta["folds"]["random"]["outer_val"])
    if "sabah_orchard_range" in N:
        N["sabah_o1_range"] = N["sabah_orchard_range"]["sabah_o1"]
        N["sabah_o2_range"] = N["sabah_orchard_range"]["sabah_o2"]
    if N.get("rq3"):
        R = N["rq3"]
        N["rq3_max_images"] = max(v["images"] for arm in ("ft", "rt") for v in R.get(arm, {}).values())
        import new_farm_calibration as nfc
        N["rq3_ft_steps"] = nfc.FT_ITERS
        word = {1: "one seed", 2: "two seeds", 3: "three seeds"}
        N["rq3_ft_seed_word"] = f"{word[len(R['seeds_ft'])]} ({' and '.join(map(str, R['seeds_ft']))})"
        N["rq3_rt_seed_word"] = f"{word[len(R['seeds_rt'])]} ({' and '.join(map(str, R['seeds_rt']))})"
    cp = N["farms"]["classes_present_per_farm"].values()
    N["farms"]["classes_present_min"], N["farms"]["classes_present_max"] = min(cp), max(cp)
    others = [c for c in NAMES if c not in ("Psyllid", "Psyllid_damage")]
    sv = [N["per_class_sensitivity_min10"][c]["retained_pct"] for c in others]
    N["sens_other_min"], N["sens_other_max"] = min(sv), max(sv)
    if "regression" in N:
        N["regression_p_pct"] = 100 * N["regression"]["all runs"]["p"]
        N["coverage_k1_pct"] = 100 * N["coverage"]["k1_mean"]
        N["coverage_k2_pct"] = 100 * N["coverage"]["k2_mean"]
        k7 = [N["class_by_k"][c][7] for c in others]
        N["class_by_k_other_min7"], N["class_by_k_other_max7"] = min(k7), max(k7)
        N["class_by_k_higher_at_7_than_1"] = all(
            N["class_by_k"][c][7] > N["class_by_k"][c][1] for c in NAMES)
        N["class_by_k_monotone"] = all(
            N["class_by_k"][c][a] <= N["class_by_k"][c][b]
            for c in NAMES for a, b in ((1, 2), (2, 4), (4, 7)))
        seeds = N["rq2_design"]["seeds"]
        N["seed_list"] = " and ".join(str(x) for x in seeds)
        N["seed_count_word"] = {1: "one seed", 2: "two seeds", 3: "three seeds"}.get(
            len(seeds), f"{len(seeds)} seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.join(root, "results_applied")
    os.makedirs(out, exist_ok=True)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    N = {}
    rq1(root, out, N)
    rq2(root, out, N)
    rq3(root, out, N)
    derive(N, root)
    with open(os.path.join(out, "paper_numbers.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(N, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
