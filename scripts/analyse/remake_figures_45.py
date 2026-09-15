#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remake_figures_45.py -- 重画 Fig 4 与 Fig 5。

Fig 4 右图原来用的是 abstention_summary.csv 的 ap50，那是把六个类混成一条
曲线算出来的 class-agnostic AP，与 mAP50 不是同一个量（同一农场差到 0.31）。
置信度与沉默率与类别无关，所以这里直接对 in_region.csv 的五种子 mAP50 作图：
r 从 -0.26 变成 +0.05。结论从「置信度指向反方向」变成「置信度不携带信息」。

Fig 5 原来只有 durian 三个模型和 GWHD 两个。BreaKHis 两个已经测出来了，
补上，共七个 dataset-model 组合。左图混用 mAP50 与 top-1，轴标已注明。

用法：
    python remake_figures_45.py --root . --out figures/
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

BLUE, GREY, RED = "#1f77b4", "#8c8c8c", "#c0392b"
ORANGE = "#d68910"


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


# ------------------------------------------------------------------ Fig 4 --
def figure4(root, out):
    """模型比较在多少个评价单元上才稳定。

    这张图回答的是「单元异质性什么时候改变结论」，不是「单元异质性有多大」。
    左：每对模型的配对差在单元间的分布。右：重抽 k 个单元时排名翻转的频率。
    """
    import itertools
    import numpy as np

    rng = np.random.default_rng(0)
    pairs = []

    ir = pd.read_csv(os.path.join(root, "results_durian", "in_region.csv"))
    p_ = (ir[ir.config.str.startswith("byfarm")]
          .pivot_table(index="config", columns="model", values="mAP50"))
    for a, b in itertools.combinations(sorted(p_.columns), 2):
        pairs.append(("Durian farms", f"{a} − {b}", p_[a].to_numpy(),
                      p_[b].to_numpy(), BLUE))
    for ds, rel, u, m, col in (
            ("GWHD sessions", "results_gwhd/gwhd_by_domain.csv", "group",
             "mAP50", "#2e8b57"),
            ("BreaKHis patients", "results_breakhis/breakhis_by_patient.csv",
             "group", "top1", ORANGE),
            ("HAR subjects", "results_har/har_by_subject.csv", "group",
             "macro_f1", "#8e44ad")):
        q = (pd.read_csv(os.path.join(root, rel)).dropna(subset=[m])
             .pivot_table(index=u, columns=m and "model", values=m).dropna())
        for a, b in itertools.combinations(sorted(q.columns), 2):
            pairs.append((ds, f"{a} − {b}", q[a].to_numpy(),
                          q[b].to_numpy(), col))

    rec = []
    for ds, lab, A, B, col in pairs:
        diff = A - B
        mu, sd = diff.mean(), diff.std(ddof=1)
        d = abs(mu) / sd
        ks, flips = [], []
        for k in (2, 4, 6, 10, 16, 32):
            if k >= len(A):
                continue
            f = 0
            for _ in range(3000):
                s = rng.choice(len(A), k, replace=False)
                if np.sign(A[s].mean() - B[s].mean()) != np.sign(mu):
                    f += 1
            ks.append(k)
            flips.append(f / 3000)
        rec.append((ds, lab, mu, sd, d, ks, flips, col, len(A)))

    rec.sort(key=lambda r: -r[4])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ypos = list(range(len(rec)))[::-1]
    for y, (ds, lab, mu, sd, d, ks, fl, col, n) in zip(ypos, rec):
        a1.plot([mu - sd, mu + sd], [y, y], color=col, lw=2.4, zorder=2)
        a1.scatter([mu], [y], color=col, s=42, zorder=3)
        a1.annotate(f"d = {d:.2f}", (max(mu + sd, 0.002), y),
                    textcoords="offset points", xytext=(8, -3), fontsize=8,
                    color="#555555")
    a1.axvline(0, color="k", lw=0.8, zorder=1)
    a1.set_yticks(ypos)
    a1.set_yticklabels([f"{lab}\n{dset.split()[0]}, {n} units"
                        for dset, lab, _, _, _, _, _, _, n in rec],
                       fontsize=7.5)
    a1.set_xlabel("paired difference on the same unit (± s.d. across units)")
    a1.set_title("How much two models differ, unit by unit", fontsize=10.5)

    for ds, lab, mu, sd, d, ks, fl, col, n in rec:
        a2.plot(ks, [100 * f for f in fl], marker="o", ms=4, lw=1.2,
                color=col, alpha=0.85)
        if ks:
            a2.annotate(f"{d:.2f}", (ks[-1], 100 * fl[-1]),
                        textcoords="offset points", xytext=(7, -2),
                        fontsize=7.5, color=col)
    a2.axhline(5, color=GREY, ls="--", lw=0.9)
    a2.annotate("5%", xy=(2.1, 6), fontsize=8, color="#666666")
    a2.set_xscale("log", base=2)
    a2.set_xticks([2, 4, 6, 10, 16, 32])
    a2.set_xticklabels([2, 4, 6, 10, 16, 32])
    a2.set_xlabel("evaluation units the comparison rests on (k)")
    a2.set_ylabel("draws that reverse the ranking (%)")
    a2.set_title("How often the ranking reverses", fontsize=10.5)
    a2.set_ylim(-2, 55)
    a2.set_xlim(1.8, 52)
    a2.annotate("labels give $d$ = mean paired difference / s.d. across units",
                xy=(0.99, 0.97), xycoords="axes fraction", ha="right",
                fontsize=8, color="#666666")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure4_model_comparison.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    for ds, lab, mu, sd, d, ks, fl, col, n in rec:
        print(f"Fig 4: {ds[:10]:<11}{lab[:24]:<25}d={d:.2f}  "
              f"flip@{ks[0] if ks else '-'}={100*fl[0] if fl else 0:.0f}%")


# ------------------------------------------------------------------ Fig 5 --
def figure5(root, out):
    rows = []

    for ds, sub, mcol, item_pref in (
            ("Durian", "results_durian", "mAP50", "random"),
            ("GWHD", "results_gwhd", "mAP50", "random"),
            ("BreaKHis", "results_breakhis", "top1", "random")):
        p = os.path.join(root, sub, "checkpoint_bias.csv")
        if not os.path.isfile(p):
            print(f"  skip {ds}: no checkpoint_bias.csv")
            continue
        cb = pd.read_csv(p)
        b, l = f"best_{mcol}", f"last_{mcol}"
        for m in sorted(cb.model.unique()):
            g = cb[cb.model == m]
            it = g[g.config.str.startswith(item_pref)]
            un = g[~g.config.str.startswith(item_pref)]
            if it.empty or un.empty:
                continue
            rows.append(dict(ds=ds, model=m, metric=mcol,
                             ib=it[b].mean(), il=it[l].mean(),
                             ub=un[b].mean(), ul=un[l].mean()))

    for r in rows:
        r["sel"] = (r["ib"] - r["ub"]) / r["ib"] * 100
        r["fin"] = (r["il"] - r["ul"]) / r["il"] * 100

    order = {"Durian": 0, "GWHD": 1, "BreaKHis": 2}
    rows.sort(key=lambda r: (order[r["ds"]], -r["fin"]))
    rows = rows[::-1]          # y 轴向上，倒序使 durian 在最上
    labels = [f"{r['ds']}\n{r['model']}" for r in rows]
    ypos = range(len(rows))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # 左图：选择带来的变化量（分）。混用 mAP50 与 top-1 的绝对水平无法共轴，
    # 变化量可以：三个数据集共享一个以零为原点的尺度，不对称一眼可见。
    for i, r in zip(ypos, rows):
        di = (r["ib"] - r["il"]) * 100
        du = (r["ub"] - r["ul"]) * 100
        a1.plot([0, du], [i, i], color=BLUE, lw=3, zorder=2)
        a1.scatter([du], [i], color=BLUE, s=50, zorder=3)
        a1.scatter([di], [i], color=GREY, s=50, zorder=4)
        a1.annotate(f"{du:.1f}", (du, i), textcoords="offset points",
                    xytext=(9, -3), color=BLUE, fontsize=9)
    a1.axvline(0, color="k", lw=0.8, zorder=1)
    a1.set_yticks(list(ypos))
    a1.set_yticklabels(labels, fontsize=9)
    a1.set_xlabel("points the reported figure falls when the selection is "
                  "removed")
    a1.set_title("The selection moves the unit-level figure\n"
                 "and leaves the item-level one alone", fontsize=10.5)
    a1.set_xlim(-0.6, 11.5)
    a1.set_ylim(-0.7, len(rows) - 0.3)
    from matplotlib.lines import Line2D
    a1.legend(handles=[
        Line2D([], [], color=GREY, lw=0, marker="o", markersize=7,
               label="item-level split"),
        Line2D([], [], color=BLUE, lw=3, marker="o", markersize=7,
               label="unit-level split")],
        loc="lower right", frameon=False, fontsize=9, handlelength=1.8)

    for i, r in zip(ypos, rows):
        a2.plot([r["sel"], r["fin"]], [i, i], color="k", lw=1.2, zorder=2)
        a2.scatter([r["sel"]], [i], color="k", s=45, zorder=3)
        a2.scatter([r["fin"]], [i], facecolor="white", edgecolor=RED,
                   s=55, lw=1.8, zorder=3)
        a2.annotate(f"+{r['fin'] - r['sel']:.1f}", (r["fin"], i),
                    textcoords="offset points", xytext=(10, -3),
                    color=RED, fontsize=9)
    a2.set_yticks(list(ypos))
    a2.set_yticklabels([])
    a2.set_ylim(-0.7, len(rows) - 0.3)
    a2.set_xlabel("overstatement (%)")
    a2.set_title("So the overstatement rises,\n"
                 "on every combination measured", fontsize=10.5)
    a2.set_xlim(0, 78)
    a2.annotate("filled: checkpoint chosen on the withheld fold\n"
                "open: final epoch, chosen by nothing",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right",
                fontsize=8.5, color="#666666")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure5_checkpoint_bias.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    for r in rows[::-1]:
        print(f"Fig 5: {r['ds']:<9} {r['model']:<12} "
              f"{r['sel']:5.1f}% -> {r['fin']:5.1f}%  "
              f"(+{r['fin'] - r['sel']:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figure4(a.root, a.out)
    figure5(a.root, a.out)


if __name__ == "__main__":
    main()
