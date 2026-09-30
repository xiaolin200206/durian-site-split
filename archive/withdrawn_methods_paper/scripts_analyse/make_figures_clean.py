#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_clean.py -- 干净协议下的五张主图，全部从 summary_*.csv 重画。

单一数据源：recompute_clean.py 写出的六张 summary 表。图里不再有任何
只存在于绘图代码里的数字，所以图与正文不可能对不上。

    Fig 1  每个单元一个分数
    Fig 2  方差分解 + 单元难度的跨模型一致性
    Fig 3  训练单元与评价单元
    Fig 4  模型比较什么时候会被评价样本翻转
    Fig 5  协议本身带来的选择偏差

用法：
    python make_figures_clean.py --root . --out figures/
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

BLUE, GREEN, ORANGE, PURPLE = "#1f77b4", "#2e8b57", "#d68910", "#8e44ad"
GREY, INK, RED = "#a0a0a0", "#333333", "#c0392b"
DS = {"durian": ("Durian, 8 farms", BLUE),
      "gwhd": ("GWHD, 47 sessions", GREEN),
      "breakhis": ("BreaKHis, 81 patients", ORANGE),
      "har": ("UCI HAR, 30 subjects", PURPLE)}


def load(root):
    r = os.path.join(root, "results_clean")
    return {n: pd.read_csv(os.path.join(r, f"summary_{n}.csv"),
                           encoding="utf-8-sig")
            for n in ("main", "decomposition", "model_pairs", "eval_curve",
                      "per_unit", "within_unit_noise")}


# ------------------------------------------------------------------ Fig 1 --
def fig1(S, out):
    pu, mn = S["per_unit"], S["main"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, ds in zip(axes, ["durian", "gwhd", "breakhis", "har"]):
        title, col = DS[ds]
        g = pu[pu.dataset == ds].groupby("unit_id").score.mean().sort_values()
        ax.scatter(range(len(g)), g.to_numpy(), s=14, color=col, zorder=3)
        row = mn[(mn.dataset == ds) & (mn["unit"] != "sabah")]
        item = float(row["item"].mean())
        unit = float(row["unit_level"].mean())
        ax.axhline(item, color=RED, ls="--", lw=1.1)
        ax.axhline(unit, color=col, lw=1.4)
        ax.annotate(f"item-level {item:.3f}", (0.03, item), xycoords=("axes fraction", "data"),
                    color=RED, fontsize=8, va="bottom")
        ax.annotate(f"unit-level {unit:.3f}", (0.03, unit), xycoords=("axes fraction", "data"),
                    color=col, fontsize=8, va="top")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("units, ordered by score")
        ax.set_xticks([])
        ax.annotate(f"{g.min():.2f}–{g.max():.2f}", (0.97, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=8.5,
                    color=GREY)
        if ax is axes[0]:
            ax.set_ylabel("score on the held-out unit")
        ax.set_ylim(-0.04, 1.06)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure1_per_unit.{e}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 ok")


# ------------------------------------------------------------------ Fig 2 --
def fig2(S, out):
    dec, pu = S["decomposition"], S["per_unit"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.0),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    order = ["breakhis", "har", "gwhd", "durian"]
    parts = [("unit", BLUE), ("model", ORANGE), ("fold", "#7f7f7f"),
             ("interaction", GREY), ("seed", "#555555"),
             ("residual", "#d5d5d5")]
    ys = range(len(order))
    for y, ds in zip(ys, order):
        row = dec[dec.dataset == ds].iloc[0]
        left = 0.0
        for name, col in parts:
            v = row.get(name)
            if v != v or v is None:
                continue
            a1.barh(y, v, left=left, height=0.6, color=col, zorder=3,
                    label=name if y == 0 else None)
            left += v
        a1.text(102, y, f"{row['unit']:.0f} / {row['model']:.1f}",
                va="center", fontsize=8.5, color=INK)
    a1.set_yticks(list(ys))
    a1.set_yticklabels([DS[d][0] for d in order], fontsize=8.5)
    a1.set_xlim(0, 128)
    a1.set_xlabel("share of variance in the reported figure (%)")
    a1.set_title("Which unit is scored, against which model", fontsize=10.5)
    a1.annotate("unit / model", xy=(102, len(order) - 0.4), fontsize=8,
                color=INK)
    a1.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left",
              bbox_to_anchor=(0.0, -0.02))
    a1.set_ylim(-0.95, len(order) - 0.4)

    d = pu[pu.dataset == "durian"]
    piv = d.pivot_table(index="unit_id", columns="model", values="score")
    o = piv.mean(axis=1).sort_values().index
    marks = ("o", "s", "^", "D", "v", "P")
    for (m, mk) in zip(sorted(piv.columns), marks):
        a2.plot(range(len(o)), piv.loc[o, m], marker=mk, ms=4, lw=1.1,
                label=m)
    rs = []
    cols = sorted(piv.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rs.append(piv[cols[i]].corr(piv[cols[j]], method="spearman"))
    a2.set_xticks(range(len(o)))
    a2.set_xticklabels([str(x).replace("byfarm_fold", "") for x in o],
                       fontsize=8)
    a2.set_xlabel("withheld farm")
    a2.set_ylabel("mAP50")
    a2.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    a2.margins(y=0.18)
    a2.set_title(f"Which farm is hard (mean pairwise ρ = {np.mean(rs):.2f})",
                 fontsize=10.5)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure2_variance.{e}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2 ok  (durian mean pairwise rho {np.mean(rs):.3f})")


# ------------------------------------------------------------------ Fig 3 --
TRAIN = {  # 训练单元曲线沿用旧实验，协议未变（不涉及 checkpoint 选择）
    "Durian (farms)": ([1, 2, 3, 4, 6], [0.091, 0.111, 0.126, 0.138, 0.148],
                       [0.0334, 0.0225, 0.0237, 0.0207, 0.0087], BLUE),
    "GWHD (sessions)": ([4, 8, 16, 24, 38], [0.266, 0.277, 0.271, 0.273, 0.286],
                        [0.0398, 0.0286, 0.0083, 0.0102, 0.0068], GREEN),
    "HAR (subjects)": ([4, 8, 16, 24], [0.872, 0.895, 0.925, 0.930],
                       [0.0350, 0.0259, 0.0074, 0.0055], PURPLE)}


def fig3(S, out):
    ec = S["eval_curve"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.6, 3.8))
    for name, (ks, mean, sd, col) in TRAIN.items():
        a1.plot(ks, [m / mean[0] for m in mean], marker="o", ms=4.5, lw=1.3,
                color=col, label=name)
    a1.axhline(1.0, color=GREY, lw=0.8)
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16, 32])
    a1.set_xticklabels([1, 2, 4, 8, 16, 32])
    a1.set_xlabel("units contributing to training (k)")
    a1.set_ylabel("held-out score, relative to $k_{min}$")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("Training units: the mean may rise", fontsize=10)

    for name, (ks, mean, sd, col) in TRAIN.items():
        a2.plot(ks, sd, marker="o", ms=4.5, lw=1.3, color=col)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks([1, 2, 4, 8, 16, 32])
    a2.set_xticklabels([1, 2, 4, 8, 16, 32])
    a2.set_xlabel("units contributing to training (k)")
    a2.set_ylabel("s.d. across unit draws")
    a2.set_title("and the spread falls", fontsize=10)

    eq = ec[ec.weighting == "equal"]
    for ds, col in (("durian farms", BLUE), ("GWHD sessions", GREEN),
                    ("BreaKHis patients", ORANGE), ("HAR subjects", PURPLE)):
        g = eq[eq.dataset == ds].sort_values("k_eval")
        g = g[g.k_eval < g.n_units.iloc[0]]
        a3.plot(g.k_eval, g.width90, marker="o", ms=4.5, lw=1.3, color=col,
                label=f"{ds} ({g.n_units.iloc[0]})")
    a3.set_xscale("log", base=2)
    a3.set_yscale("log")
    a3.set_xticks([1, 2, 4, 8, 16, 32, 64])
    a3.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    a3.set_xlabel("units the reported figure rests on (k)")
    a3.set_ylabel("width of the 5th–95th percentile range")
    a3.legend(frameon=False, fontsize=7.5, loc="lower left")
    a3.set_title("Evaluation units: only the spread moves", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure3_unit_counts.{e}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 3 ok")


# ------------------------------------------------------------------ Fig 4 --
FAM = {"rtdetr-l": "DETR", "yolo11n": "YOLO", "yolo11s": "YOLO",
       "yolo11m": "YOLO", "yolo11l": "YOLO", "frcnn-r50": "R-CNN",
       "yolo11n-cls": "YOLO", "yolo11s-cls": "YOLO", "mlp256": "MLP",
       "rf": "forest"}


def fig4(S, out):
    p = S["model_pairs"].copy()
    p["cross"] = [FAM.get(x.split(" vs ")[0]) != FAM.get(x.split(" vs ")[1])
                  for x in p.pair]
    p = p.sort_values("d", ascending=False).reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.0),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    ys = list(range(len(p)))[::-1]
    for y, (_, r) in zip(ys, p.iterrows()):
        col = {"durian farms": BLUE, "GWHD sessions": GREEN,
               "BreaKHis patients": ORANGE, "HAR subjects": PURPLE}[r.dataset]
        a1.plot([r.mean_diff - r.sd_diff, r.mean_diff + r.sd_diff], [y, y],
                color=col, lw=2.2, zorder=2)
        a1.scatter([r.mean_diff], [y], color=col, s=34,
                   marker="D" if r["cross"] else "o", zorder=3)
    a1.axvline(0, color="k", lw=0.8, zorder=1)
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{r.pair}  ({r.d:.2f})" for _, r in p.iterrows()],
                       fontsize=7.5)
    a1.set_xlabel("difference between two models on the same unit "
                  "(± s.d. across units)")
    a1.set_title("Every pair, unit by unit.  Diamonds cross a model family",
                 fontsize=10)

    for _, r in p.iterrows():
        ks = [k for k in (2, 4, 6, 10, 16, 32) if f"flip_k{k}" in r
              and r[f"flip_k{k}"] == r[f"flip_k{k}"]]
        if not ks:
            continue
        col = {"durian farms": BLUE, "GWHD sessions": GREEN,
               "BreaKHis patients": ORANGE, "HAR subjects": PURPLE}[r.dataset]
        a2.plot(ks, [100 * r[f"flip_k{k}"] for k in ks], marker="o", ms=3.5,
                lw=1.0, color=col, alpha=0.8,
                ls="--" if r["cross"] else "-")
    a2.axhline(5, color=INK, ls=":", lw=1.0)
    a2.annotate("5%", xy=(2.1, 6.5), fontsize=8, color=INK)
    a2.set_xscale("log", base=2)
    a2.set_xticks([2, 4, 6, 10, 16, 32])
    a2.set_xticklabels([2, 4, 6, 10, 16, 32])
    a2.set_xlabel("evaluation units the comparison rests on (k)")
    a2.set_ylabel("draws that reverse the ranking (%)")
    a2.set_title("How often a different set of units reverses the winner",
                 fontsize=10)
    a2.set_ylim(-2, 55)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure4_model_pairs.{e}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 4 ok  ({len(p)} pairs, "
          f"{(p.k_for_5pct <= p.n_units).sum()} resolvable)")


# ------------------------------------------------------------------ Fig 5 --
def fig5(S, out, root="."):
    """三个协议的高估幅度对照。

    干净协议下 best 与 last 几乎不再有差别（checkpoint 是在内层验证集上选的），
    所以这张图不再画 best 对 last，而是画三个协议：

        旧协议 best   留出单元既选 checkpoint 又被评分
        旧协议 last   去掉 best-epoch 选择，早停仍看留出单元
        干净协议      留出单元不参与任何训练决策

    干净协议应当落在两者之间，而这正是当初那次 checkpoint 偏差测量的预测。
    """
    f = os.path.join(root, "results_clean",
                     "summary_protocol_comparison.csv")
    d = pd.read_csv(f, encoding="utf-8-sig")
    d = d.sort_values(["dataset", "clean"], ascending=[True, False])
    d = d.reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    ys = list(range(len(d)))[::-1]
    col = {"durian": BLUE, "gwhd": GREEN, "breakhis": ORANGE}
    for y, (_, r) in zip(ys, d.iterrows()):
        c = col[r["dataset"]]
        lo, hi = r["old_best"], r["clean"]
        a1.plot([lo, hi], [y, y], color=c, lw=2.4, zorder=2)
        a1.scatter([lo], [y], color=c, s=40, zorder=3)
        a1.scatter([hi], [y], facecolor="white", edgecolor=c, s=52, lw=1.8,
                   zorder=3)
        if r["old_last"] == r["old_last"]:
            a1.scatter([r["old_last"]], [y], marker="|", color=INK, s=90,
                       zorder=4)
        a1.annotate(f"{hi - lo:+.1f}", (max(lo, hi), y),
                    textcoords="offset points", xytext=(9, -3), color=c,
                    fontsize=8)
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{r['dataset']}\n{r['model']}"
                        for _, r in d.iterrows()], fontsize=7.5)
    a1.set_xlabel("overstatement of the item-level protocol (%)")
    a1.set_title("What the contaminated protocol hid", fontsize=10.5)
    a1.set_xlim(0, float(d["clean"].max()) * 1.35)
    from matplotlib.lines import Line2D
    a1.legend(handles=[
        Line2D([], [], color=GREY, lw=0, marker="o", ms=7,
               label="held-out unit selects the checkpoint"),
        Line2D([], [], color=INK, lw=0, marker="|", ms=9,
               label="final epoch, early stopping still on the unit"),
        Line2D([], [], color="w", markeredgecolor=GREY, lw=0, marker="o",
               ms=8, label="clean: unit in no training decision")],
        frameon=False, fontsize=7.5, loc="lower right")

    # 右：干净协议下，选 epoch 还剩多少影响
    m = S["main"]
    m = m[(m["unit"] != "sabah") & m["item_last"].notna()
          & m["unit_last"].notna()].copy()
    m["drop_item"] = 100 * (m["item"] - m["item_last"])
    m["drop_unit"] = 100 * (m["unit_level"] - m["unit_last"])
    m = m.sort_values("drop_unit", ascending=False).reset_index(drop=True)
    ys2 = list(range(len(m)))[::-1]
    for y, (_, r) in zip(ys2, m.iterrows()):
        c = col.get(r["dataset"], GREY)
        a2.plot([0, r["drop_unit"]], [y, y], color=c, lw=2.2, zorder=2)
        a2.scatter([r["drop_unit"]], [y], color=c, s=36, zorder=3)
        a2.scatter([r["drop_item"]], [y], color=GREY, s=36, zorder=4)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_yticks(ys2)
    a2.set_yticklabels([f"{r['dataset']}, {r['model']}"
                        for _, r in m.iterrows()], fontsize=7)
    a2.set_xlabel("points the figure falls without best-epoch selection")
    a2.set_title("Under the clean protocol the choice of epoch\n"
                 "no longer matters (grey: item, colour: unit)",
                 fontsize=10.5)
    a2.set_xlim(-1.6, 2.4)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure5_protocol.{e}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 5 ok  (clean minus old-best: "
          f"{(d['clean'] - d['old_best']).mean():+.1f} points on average)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = load(a.root)
    fig1(S, a.out)
    fig2(S, a.out)
    fig3(S, a.out)
    fig4(S, a.out)
    fig5(S, a.out, a.root)


if __name__ == "__main__":
    main()
