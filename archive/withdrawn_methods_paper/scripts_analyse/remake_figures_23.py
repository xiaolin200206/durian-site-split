#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remake_figures_23.py -- 给 Fig 2 与 Fig 3 加面板。

Fig 2 原来只有 durian 的方差分解。四个数据集的分解都算得出来，而且 GWHD
是反例（seed 占 61.2%），把它画出来比藏在补充材料里诚实。新版：

    左  durian 四个模型 × 八个农场 × 五种子，90% bootstrap 区间
    中  四个数据集并排，分量堆叠
    右  四个模型下的农场排序（mean pairwise Spearman 0.94）

★ 中图的第一个分量在 durian 上是「被留出的农场」，在其余三个数据集上是
  「被留出的那一折的构成」，因为那里一折含多个单元。图例与图注都按此措辞，
  不要写成 per-unit variance。

Fig 3 原来两个面板都讲训练单元。加第三个面板讲评价单元，因为两者规律不同：
训练单元可能抬高均值，评价单元从不抬高，只收窄区间。这是全文最容易被读成
同一件事的两个量。

用法：
    python remake_figures_23.py --root . --out figures/
"""

import argparse
import math
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

BLUE, ORANGE, GREY, INK, RED = ("#1f77b4", "#d68910", "#b0b0b0",
                                "#333333", "#c0392b")

# 各数据集的分解结果，来自 variance_decomposition.py，与 Supp Table 5 一致。
# (数据集, 第一分量, 模型, 交互, 种子, 第一分量的名称)
DECOMP = [
    ("Durian\n4 models × 8 farms", 88.9, 1.1, 5.3, 4.7, "withheld farm"),
    ("BreaKHis\n2 models × 5 folds", 78.7, 0.4, 0.5, 20.4, "held-out fold"),
    ("HAR\n2 models × 5 folds", 56.1, 14.6, 28.7, 0.6, "held-out fold"),
    ("GWHD\n2 models × 5 folds", 38.8, 0.0, 0.0, 61.2, "held-out fold"),
]

EU_SETS = [
    ("Durian bursts", "results_durian/peninsula_by_burst_60s_min5.csv",
     "burst", "mAP50", "yolo11s", BLUE),
    ("GWHD sessions", "results_gwhd/gwhd_by_domain.csv",
     "group", "mAP50", "yolo11s", "#2e8b57"),
    ("BreaKHis patients", "results_breakhis/breakhis_by_patient.csv",
     "group", "top1", "yolo11s-cls", ORANGE),
    ("HAR subjects", "results_har/har_by_subject.csv",
     "group", "macro_f1", "mlp256", "#8e44ad"),
]


def ems(cells):
    """两因素交叉随机效应，期望均方。cells[(model, fold)] = [种子分数]。"""
    models = sorted({m for m, _ in cells})
    folds = sorted({f for _, f in cells})
    a, b = len(models), len(folds)
    n = min(len(v) for v in cells.values())
    grand = sum(sum(v[:n]) for v in cells.values()) / (a * b * n)
    ma = {m: sum(sum(cells[(m, f)][:n]) for f in folds) / (b * n)
          for m in models}
    mf = {f: sum(sum(cells[(m, f)][:n]) for m in models) / (a * n)
          for f in folds}
    ss_a = b * n * sum((ma[m] - grand) ** 2 for m in models)
    ss_f = a * n * sum((mf[f] - grand) ** 2 for f in folds)
    ss_af = ss_e = 0.0
    for m in models:
        for f in folds:
            v = cells[(m, f)][:n]
            cm = sum(v) / n
            ss_af += n * (cm - ma[m] - mf[f] + grand) ** 2
            ss_e += sum((x - cm) ** 2 for x in v)
    ms_a = ss_a / (a - 1)
    ms_f = ss_f / (b - 1)
    ms_af = ss_af / ((a - 1) * (b - 1))
    ms_e = ss_e / (a * b * (n - 1))
    v_e = ms_e
    v_af = max(0.0, (ms_af - ms_e) / n)
    v_a = max(0.0, (ms_a - ms_af) / (n * b))
    v_f = max(0.0, (ms_f - ms_af) / (n * a))
    return v_f, v_a, v_af, v_e


def figure2(root, out):
    ir = pd.read_csv(os.path.join(root, "results_durian", "in_region.csv"))
    d = ir[ir.config.str.startswith("byfarm")]
    cells = defaultdict(list)
    for _, r in d.iterrows():
        cells[(r["model"], r["config"])].append(float(r["mAP50"]))
    models = sorted({m for m, _ in cells})
    folds = sorted({f for _, f in cells})
    v_f, v_a, v_af, v_e = ems(cells)
    tot = v_f + v_a + v_af + v_e

    rng = random.Random(0)
    boots = defaultdict(list)
    for _ in range(2000):
        fb = [rng.choice(folds) for _ in folds]
        cb = {}
        for i, f in enumerate(fb):
            for m in models:
                cb[(m, f"b{i}")] = cells[(m, f)]
        try:
            r_ = ems(cb)
        except (ZeroDivisionError, ValueError):
            continue
        t_ = sum(r_)
        if t_ <= 0:
            continue
        for nm, val in zip(("farm", "arch", "inter", "seed"), r_):
            boots[nm].append(100 * val / t_)

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.25, 1.15, 1]})

    labels = ["Withheld\nfarm", "Architecture\n× farm", "Training\nseed",
              "Architecture"]
    keys = ["farm", "inter", "seed", "arch"]
    shares = [100 * v / tot for v in (v_f, v_af, v_e, v_a)]
    cols = [BLUE, GREY, GREY, ORANGE]
    ypos = list(range(len(labels)))[::-1]
    for y, lab, s, k, c in zip(ypos, labels, shares, keys, cols):
        a1.barh(y, s, height=0.6, color=c, zorder=3)
        right = s
        if boots[k]:
            q = sorted(boots[k])
            lo, hi = q[int(0.05 * len(q))], q[int(0.95 * len(q)) - 1]
            a1.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
            for x in (lo, hi):
                a1.plot([x, x], [y - .12, y + .12], color=INK, lw=1.0,
                        zorder=4)
            right = max(s, hi)
        a1.text(right + 2.5, y, f"{s:.1f}%", va="center", fontsize=8.5)
    a1.set_yticks(ypos)
    a1.set_yticklabels(labels, fontsize=8.5)
    a1.set_xlim(0, 118)
    a1.set_xlabel("share of variance in the reported mAP50 (%)")
    a1.set_title("Durian: 4 architectures × 8 farms × 5 seeds", fontsize=10)

    # 中图：四个数据集并排
    names = [d_[0] for d_ in DECOMP][::-1]
    ypos2 = list(range(len(names)))
    parts = [("held-out sample", BLUE, 1), ("model", ORANGE, 2),
             ("interaction", GREY, 3), ("seed", "#555555", 4)]
    for y, row in zip(ypos2, DECOMP[::-1]):
        left = 0.0
        for name, col, idx in parts:
            w = row[idx]
            a2.barh(y, w, left=left, height=0.62, color=col,
                    label=name if y == 0 else None, zorder=3)
            left += w
        a2.text(101.5, y, f"{row[1]:.0f} / {row[4]:.0f}", va="center",
                fontsize=8, color=INK)
    a2.set_yticks(ypos2)
    a2.set_yticklabels(names, fontsize=8)
    a2.set_xlim(0, 132)
    a2.set_ylim(-0.95, len(names) - 0.4)
    a2.set_xlabel("share of variance (%)")
    a2.set_title("Sample against model, four datasets", fontsize=10)
    a2.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower left",
              bbox_to_anchor=(-0.02, -0.03), columnspacing=1.0,
              handlelength=1.2)
    a2.annotate("sample / seed", xy=(101.5, len(names) - 0.42), fontsize=7.5,
                color=INK)

    # 右图：农场排序
    fm = {m: [pd.Series(cells[(m, f)]).mean() for f in folds] for m in models}
    order = sorted(range(len(folds)), key=lambda i: fm[models[0]][i])
    marks = ("o", "s", "^", "D")
    colours = (BLUE, ORANGE, "#2e8b57", RED)
    for m, c, mk in zip(models, colours, marks):
        a3.plot(range(len(folds)), [fm[m][i] for i in order], marker=mk,
                ms=4, lw=1.1, color=c, label=m)
    a3.set_xticks(range(len(folds)))
    a3.set_xticklabels([folds[i].replace("byfarm_fold", "") for i in order],
                       fontsize=8)
    a3.set_xlabel("withheld farm")
    a3.set_ylabel("mAP50")
    a3.legend(frameon=False, fontsize=7.5, loc="upper left")
    a3.margins(y=0.14)
    a3.set_title("Ranking preserved (mean pairwise ρ = 0.94)", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure2_variance.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 2: farm {shares[0]:.1f}%  arch {shares[3]:.1f}%  "
          f"seed {shares[2]:.1f}%")


def figure3(root, out):
    # 面板一、二：训练单元（数值来自 Supplementary Table 7）
    train = {
        "Durian (farms)": ([1, 2, 3, 4, 6],
                           [0.091, 0.111, 0.126, 0.138, 0.148],
                           [0.0334, 0.0225, 0.0237, 0.0207, 0.0087],
                           [0.0063, 0.0076, 0.0088, 0.0088, 0.0122], BLUE),
        "GWHD (sessions)": ([4, 8, 16, 24, 38],
                            [0.266, 0.277, 0.271, 0.273, 0.286],
                            [0.0398, 0.0286, 0.0083, 0.0102, 0.0068],
                            None, "#2e8b57"),
        "HAR (subjects)": ([4, 8, 16, 24],
                           [0.872, 0.895, 0.925, 0.930],
                           [0.0350, 0.0259, 0.0074, 0.0055],
                           [0.0037, 0.0057, 0.0013, 0.0034], ORANGE),
    }

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 3.8))

    for name, (ks, mean, sd, seed, col) in train.items():
        base = mean[0]
        a1.plot(ks, [m / base for m in mean], marker="o", ms=4.5, lw=1.3,
                color=col, label=name)
    a1.axhline(1.0, color=GREY, lw=0.8, zorder=1)
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16, 32])
    a1.set_xticklabels([1, 2, 4, 8, 16, 32])
    a1.set_xlabel("units contributing to training (k)")
    a1.set_ylabel("held-out score, relative to $k_{min}$")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("Training units: the mean may or may not rise", fontsize=10)

    for name, (ks, mean, sd, seed, col) in train.items():
        a2.plot(ks, sd, marker="o", ms=4.5, lw=1.3, color=col, label=name)
        if seed:
            a2.plot(ks, seed, lw=1.0, ls="--", color=col, alpha=0.65)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xticks([1, 2, 4, 8, 16, 32])
    a2.set_xticklabels([1, 2, 4, 8, 16, 32])
    a2.set_xlabel("units contributing to training (k)")
    a2.set_ylabel("s.d. across unit draws")
    a2.set_title("and the spread falls three- to six-fold", fontsize=10)
    a2.annotate("dashed: seed-to-seed s.d.", xy=(0.97, 0.95),
                xycoords="axes fraction", ha="right", fontsize=8,
                color="#666666")

    # 面板三：评价单元，等权重抽
    rng = random.Random(0)
    B = 2000
    for name, rel, ukey, metric, model, col in EU_SETS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        df = df[df.model == model].dropna(subset=[metric])
        scores = df.groupby(ukey)[metric].mean().tolist()
        N = len(scores)
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= N]
        sds, means = [], []
        for k in ks:
            vals = [sum(rng.sample(scores, k)) / k for _ in range(B)]
            m = sum(vals) / len(vals)
            means.append(m)
            sds.append(math.sqrt(sum((v - m) ** 2 for v in vals)
                                 / (len(vals) - 1)))
        a3.plot(ks, sds, marker="o", ms=4.5, lw=1.3, color=col,
                label=f"{name} ({N})")
        drift = max(means) - min(means)
        print(f"Fig 3 panel 3: {name:<18} mean drift over k = {drift:.4f}")
    a3.set_xscale("log", base=2)
    a3.set_yscale("log")
    a3.set_xticks([1, 2, 4, 8, 16, 32, 64])
    a3.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    a3.set_xlabel("units the evaluation rests on (k)")
    a3.set_ylabel("s.d. across unit draws")
    a3.legend(frameon=False, fontsize=8, loc="lower left")
    a3.set_title("Evaluation units: the mean never moves", fontsize=10)
    a3.annotate("mean flat to within Monte Carlo error at every k;\n"
                "only the spread falls, as $1/\\sqrt{k}$", xy=(0.97, 0.93),
                xycoords="axes fraction", ha="right", va="top", fontsize=8,
                color="#666666")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"figure3_dose_response.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    figure2(a.root, a.out)
    figure3(a.root, a.out)


if __name__ == "__main__":
    main()
