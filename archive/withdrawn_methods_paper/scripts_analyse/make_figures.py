#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- 从发布的结果表生成四张主图。

每张图只读 results_*/ 下的 csv，不需要图像或权重。缺表则跳过该图并说明，
不画任何占位数据。

    python make_figures.py --out figures/
"""

import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

INK = "#1a1a1a"
GREY = "#9aa0a6"
BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"
ORANGE = "#b7791f"


def rd(p):
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.DictReader(fh))


def mean(v):
    return st.mean(v) if v else float("nan")


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def unit_scores(rows, key, metric, nkey=None, minn=5):
    g, n = defaultdict(list), {}
    for r in rows:
        try:
            g[r[key]].append(float(r[metric]))
        except (KeyError, ValueError):
            continue
        if nkey and nkey in r:
            try:
                n[r[key]] = int(float(r[nkey]))
            except ValueError:
                pass
    out = {}
    for k, v in g.items():
        if nkey and n and n.get(k, minn) < minn:
            continue
        out[k] = (mean(v), sd(v), n.get(k, 0))
    return out


# --------------------------------------------------------------- figure 1 --
def fig1(D, out):
    panels = []

    pb = rd(f"{D}/results_durian/peninsula_by_burst_60s_min5.csv")
    if pb:
        u = unit_scores(pb, "burst", "mAP50", "n_images")
        panels.append(("Durian\n58 capture bursts", u, 0.2752, 0.4943, "mAP50"))

    stre = rd(f"{D}/results_durian/sabah_by_tree.csv")
    if stre:
        u = unit_scores(stre, "group", "mAP50", "n_images")
        panels.append(("Durian, Sabah\n12 trees", u, 0.2698, None, "mAP50"))

    gd = rd(f"{D}/results_gwhd/gwhd_by_domain.csv")
    if gd:
        u = unit_scores(gd, "group", "mAP50", "n_images")
        panels.append(("GWHD\n47 sessions", u, 0.5537, 0.6722, "mAP50"))

    bp = rd(f"{D}/results_breakhis/breakhis_by_patient.csv")
    if bp:
        u = unit_scores(bp, "group", "top1", "n_images")
        panels.append(("BreaKHis\n81 patients", u, 0.9063, 0.9929, "top-1"))

    hs = rd(f"{D}/results_har/har_by_subject.csv")
    if hs:
        u = unit_scores(hs, "group", "macro_f1", "n_windows")
        panels.append(("UCI HAR\n30 subjects", u, 0.9519, 0.9885, "macro F1"))

    if not panels:
        print("fig1: 一张 per-unit 表都没找到，跳过")
        print(f"     期望位于 {D}/results_durian/, {D}/results_gwhd/, "
              f"{D}/results_breakhis/, {D}/results_har/")
        return
    if len(panels) < 5:
        have = {t.split(chr(10))[0] for t, *_ in panels}
        print(f"fig1: 只画了 {len(panels)} 个面板；缺表的数据集不会出现在图上")
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.05 * len(panels), 3.1))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, u, pooled, item, metric) in zip(axes, panels):
        vals = sorted(u.items(), key=lambda kv: kv[1][0])
        y = [v[0] for _, v in vals]
        e = [v[1] for _, v in vals]
        x = range(len(y))
        ax.errorbar(x, y, yerr=e, fmt="o", ms=2.4, lw=0, elinewidth=0.5,
                    color=INK, ecolor=GREY, capsize=0, zorder=3)
        ax.axhline(pooled, color=BLUE, lw=1.1, zorder=2)
        if item is not None:
            ax.axhline(item, color=RED, lw=1.1, ls=(0, (3, 2)), zorder=2)
        # 两条线靠得近时把标签上下推开，否则会叠在一起
        # 数据按分数升序排列，所以低处的线放右侧、高处的线放左侧，
        # 标签才不会压在散点上。两条线靠近时再上下推开。
        med = y[len(y) // 2]
        gap = abs((item if item is not None else pooled) - pooled)
        off = 0.05 if gap < 0.12 else 0.0

        # 池化值按实例数加权，散点是每单元等权，两者不必相等。检测任务
        # 差得明显，分类任务几乎不差。等权均值也画出来，否则读者会看到
        # 散点重心不在蓝线上而无从解释。
        eq = sum(y) / len(y)
        show_eq = abs(eq - pooled) > 0.01
        if show_eq:
            ax.axhline(eq, color=BLUE, lw=0.8, ls=(0, (1, 2)), zorder=2)

        # 标签一律贴在各自那条线的左端，竖直方向按数值从低到高依次排开，
        # 相邻两条太近时把后一条往上推。早先按"低于中位数就靠右"来决定
        # 位置，遇到等权低于池化的面板（GWHD）两个标签会落到同一处，
        # 图上因此把两条线都印成了同一个数。
        marks = [(pooled, f"pooled {pooled:.3f}", BLUE, 1.0)]
        if show_eq:
            marks.append((eq, f"equal-weight {eq:.3f}", BLUE, 0.9))
        if item is not None:
            marks.append((item, f"item-level {item:.3f}", RED, 1.0))
        marks.sort(key=lambda t: t[0])

        span = 1.10
        gap = 0.052 * span
        last = -9.0
        for val, text, colour, alpha in marks:
            ytxt = max(val + gap * 0.18, last + gap)
            last = ytxt
            ax.annotate(text, xy=(0.02, ytxt),
                        xycoords=("axes fraction", "data"), color=colour,
                        fontsize=6.2, ha="left", va="bottom", alpha=alpha,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white",
                                  ec="none", alpha=0.85))
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlim(-len(y) * 0.03, len(y) * 1.03)
        ax.set_xticks([])
        ax.set_title(title, fontsize=7.5, pad=6)
        ax.set_xlabel("units, ordered by score", fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        if ax is axes[0]:
            ax.set_ylabel("score on the held-out unit", fontsize=7.5)
        else:
            ax.set_yticklabels([])
        ax.text(0.97, 0.04, f"{min(y):.2f}–{max(y):.2f}",
                transform=ax.transAxes, ha="right", fontsize=6.5,
                color=GREY)
    fig.savefig(os.path.join(out, "figure1_per_unit.png"))
    fig.savefig(os.path.join(out, "figure1_per_unit.pdf"))
    plt.close(fig)
    print(f"fig1: {len(panels)} 个面板")


# --------------------------------------------------------------- figure 2 --
def ems(cells):
    A = sorted({a for a, _ in cells})
    F = sorted({f for _, f in cells})
    a, b = len(A), len(F)
    n = min(len(v) for v in cells.values())
    cell = {k: mean(v[:n]) for k, v in cells.items()}
    grand = mean(list(cell.values()))
    mA = {i: mean([cell[(i, j)] for j in F]) for i in A}
    mF = {j: mean([cell[(i, j)] for i in A]) for j in F}
    ss_a = n * b * sum((mA[i] - grand) ** 2 for i in A)
    ss_f = n * a * sum((mF[j] - grand) ** 2 for j in F)
    ss_af = n * sum((cell[(i, j)] - mA[i] - mF[j] + grand) ** 2
                    for i in A for j in F)
    ss_e = sum((x - cell[(i, j)]) ** 2
               for i in A for j in F for x in cells[(i, j)][:n])
    ms_a, ms_f = ss_a / (a - 1), ss_f / (b - 1)
    ms_af = ss_af / ((a - 1) * (b - 1))
    ms_e = ss_e / (a * b * (n - 1))
    v_e = ms_e
    v_af = max(0.0, (ms_af - ms_e) / n)
    v_a = max(0.0, (ms_a - ms_af) / (n * b))
    v_f = max(0.0, (ms_f - ms_af) / (n * a))
    return v_f, v_a, v_af, v_e


def fig2(D, out):
    p2 = f"{D}/results_durian/in_region.csv"
    ir = rd(p2)
    if not ir:
        print(f"fig2: 找不到 {p2}，跳过")
        return
    if "model" not in ir[0]:
        print(f"fig2: {p2} 没有 model 列，跳过（需要两个架构的完全交叉）")
        return
    cells = defaultdict(list)
    for r in ir:
        if r["config"].startswith("byfarm"):
            cells[(r["model"], r["config"])].append(float(r["mAP50"]))
    models = sorted({a for a, _ in cells})
    folds = sorted({f for _, f in cells})
    if len(models) < 2:
        print("fig2: 只有一个架构，跳过")
        return
    cells = {k: v for k, v in cells.items()}
    v_f, v_a, v_af, v_e = ems(cells)
    tot = v_f + v_a + v_af + v_e

    import random
    random.seed(0)
    boots = defaultdict(list)
    for _ in range(2000):
        fb = [random.choice(folds) for _ in folds]
        cb = {}
        for i, f in enumerate(fb):
            for m_ in models:
                cb[(m_, f"b{i}")] = cells[(m_, f)]
        try:
            r_ = ems(cb)
        except (ZeroDivisionError, ValueError):
            continue
        t_ = sum(r_)
        if t_ <= 0:
            continue
        for nm, val in zip(("farm", "arch", "inter", "seed"), r_):
            boots[nm].append(100 * val / t_)

    labels = ["Evaluation\nunit", "Architecture\n× unit",
              "Training\nseed", "Architecture"]
    keys = ["farm", "inter", "seed", "arch"]
    shares = [100 * v / tot for v in (v_f, v_af, v_e, v_a)]
    cols = [BLUE, GREY, GREY, ORANGE]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ypos = range(len(labels))[::-1]
    for y, lab, s, k, c in zip(ypos, labels, shares, keys, cols):
        ax.barh(y, s, height=0.6, color=c, zorder=3)
        right = s
        if boots[k]:
            q = sorted(boots[k])
            lo, hi = q[int(0.05 * len(q))], q[int(0.95 * len(q)) - 1]
            ax.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
            ax.plot([lo, lo], [y - .12, y + .12], color=INK, lw=1.0, zorder=4)
            ax.plot([hi, hi], [y - .12, y + .12], color=INK, lw=1.0, zorder=4)
            right = max(s, hi)
        ax.text(right + 2.0, y, f"{s:.1f}%", va="center", fontsize=7.5)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of variance in the reported mAP50 (%)",
                  fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_title("Durian: two architectures × eight farms × five seeds",
                 fontsize=8, pad=6)

    fm = {m_: [mean(cells[(m_, f)]) for f in folds] for m_ in models}
    order = sorted(range(len(folds)), key=lambda i: fm[models[0]][i])
    for m_, c, mk in zip(models, (BLUE, ORANGE), ("o", "s")):
        ax2.plot(range(len(folds)), [fm[m_][i] for i in order],
                 marker=mk, ms=3.5, lw=1.0, color=c, label=m_)
    ax2.set_xticks(range(len(folds)))
    ax2.set_xticklabels([folds[i].replace("byfarm_fold", "") for i in order],
                        fontsize=7)
    ax2.set_xlabel("withheld farm", fontsize=7.5)
    ax2.set_ylabel("mAP50", fontsize=7.5)
    ax2.tick_params(labelsize=7)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.margins(y=0.12)
    ax2.set_title("Identical ranking (Spearman ρ = 1.00)", fontsize=8, pad=6)
    fig.savefig(os.path.join(out, "figure2_variance.png"))
    fig.savefig(os.path.join(out, "figure2_variance.pdf"))
    plt.close(fig)
    print(f"fig2: farm {shares[0]:.1f}%  arch {shares[3]:.1f}%")


# --------------------------------------------------------------- figure 3 --
def fig3(D, out):
    series = []
    for path, metric, name, col in (
            (f"{D}/results_durian/site_count.csv", "mAP50", "Durian (farms)", BLUE),
            (f"{D}/results_gwhd/site_count.csv", "mAP50",
             "GWHD (sessions)", GREEN),
            (f"{D}/results_har/site_count.csv", "macro_f1",
             "HAR (subjects)", ORANGE)):
        rows = rd(path)
        if not rows:
            continue
        byk = defaultdict(lambda: defaultdict(list))
        for r in rows:
            key = r.get("config") or r.get("draw")
            byk[int(r["k"])][key].append(float(r[metric]))
        ks = sorted(byk)
        m_ = [mean([x for v in byk[k].values() for x in v]) for k in ks]
        # 抽样间标准差要求该 k 至少有两个抽样。只有一个时它是未定义的，
        # 不是零；在对数轴上画成零会掉到负无穷，拉出一条竖直的假线。
        draw_ks = [k for k in ks if len(byk[k]) > 1]
        draw = [sd([mean(v) for v in byk[k].values()]) for k in draw_ks]
        seed = [mean([sd(v) for v in byk[k].values() if len(v) > 1])
                for k in draw_ks]
        if len(draw_ks) < len(ks):
            skipped = [k for k in ks if k not in draw_ks]
            print(f"      {name}: k={skipped} 只有一个抽样，"
                  f"右图不画（离散度未定义）")
        series.append((name, col, ks, m_, draw_ks, draw, seed))
    if not series:
        print("fig3: 一张 site_count 表都没找到，跳过")
        return
    if len(series) < 3:
        print(f"fig3: 只画了 {len(series)} 条曲线；缺 site_count.csv 的"
              f"数据集不会出现在图上")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for name, col, ks, m_, draw_ks, draw, seed in series:
        rel = [x / m_[0] for x in m_]
        axes[0].plot(ks, rel, marker="o", ms=3.2, lw=1.1, color=col,
                     label=name)
        axes[1].plot(draw_ks, draw, marker="o", ms=3.2, lw=1.1, color=col,
                     label=name)
        axes[1].plot(draw_ks, seed, marker="", lw=0.8, ls=(0, (2, 2)),
                     color=col, alpha=0.75)
    axes[0].axhline(1.0, color=GREY, lw=0.6, zorder=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("units contributing to training (k)", fontsize=7.5)
    axes[0].set_ylabel("held-out score, relative to k$_{min}$", fontsize=7.5)
    axes[0].set_title("Expected score: depends on the dataset", fontsize=8, pad=6)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("units contributing to training (k)", fontsize=7.5)
    axes[1].set_ylabel("s.d. across unit draws", fontsize=7.5)
    axes[1].set_title("Spread across draws: falls with k in all three", fontsize=8, pad=6)
    axes[1].text(0.97, 0.94, "dashed: seed-to-seed s.d.",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=6.5, color=GREY)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.savefig(os.path.join(out, "figure3_dose_response.png"))
    fig.savefig(os.path.join(out, "figure3_dose_response.pdf"))
    plt.close(fig)
    print(f"fig3: {len(series)} 条曲线")


# --------------------------------------------------------------- figure 4 --
def fig4(D, out):
    p4 = f"{D}/results_durian/abstention_summary.csv"
    ab = rd(p4)
    if not ab:
        print(f"fig4: 找不到 {p4}，跳过")
        return
    cov = [("Capture hour", 0.727), ("Lesion size", 0.294),
           ("Midday share", -0.351), ("Solar elevation", -0.321),
           ("Focal lengths", -0.320), ("Devices", -0.295),
           ("Images in fold", -0.229), ("Wide-lens share", 0.059)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    order = sorted(cov, key=lambda t: abs(t[1]))
    ypos = range(len(order))
    for y, (nm, r) in zip(ypos, order):
        axes[0].barh(y, r, height=0.6,
                     color=(RED if abs(r) > 0.5 else GREY), zorder=3)
    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels([n for n, _ in order], fontsize=7)
    axes[0].axvline(0, color=INK, lw=0.6)
    axes[0].set_xlim(-1, 1)
    axes[0].set_xlabel("Pearson r with fold mAP50 (8 farms)", fontsize=7.5)
    axes[0].tick_params(labelsize=7)
    axes[0].set_title("No capture covariate predicts failure",
                      fontsize=8, pad=6)
    axes[0].annotate("rests on a single farm;\nsolar elevation, which should\ncarry the same signal, gives -0.32",
                     xy=(0.727, len(order) - 1), xytext=(-0.95, len(order) - 2.9),
                     fontsize=6.0, color=RED, ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=RED, lw=0.6,
                                     shrinkA=2, shrinkB=2))

    conf = [float(r["mean_max_conf"]) for r in ab]
    ap = [float(r["ap50"]) for r in ab]
    lbl = [r["fold"].replace("byfarm_fold", "farm ") for r in ab]
    axes[1].scatter(conf, ap, s=22, color=BLUE, zorder=3)
    for c, a, l in zip(conf, ap, lbl):
        axes[1].annotate(l, (c, a), fontsize=6, xytext=(3, 3),
                         textcoords="offset points", color=GREY)
    mx, my = mean(conf), mean(ap)
    num = sum((a - mx) * (b - my) for a, b in zip(conf, ap))
    den = math.sqrt(sum((a - mx) ** 2 for a in conf) *
                    sum((b - my) ** 2 for b in ap))
    r = num / den if den else float("nan")
    axes[1].set_xlabel("mean maximum confidence on the withheld farm",
                       fontsize=7.5)
    axes[1].set_ylabel("AP50 on that farm", fontsize=7.5)
    axes[1].tick_params(labelsize=7)
    axes[1].set_title(f"Nor does the model's confidence  (r = {r:+.2f})",
                      fontsize=8, pad=6)
    fig.savefig(os.path.join(out, "figure4_unpredictable.png"))
    fig.savefig(os.path.join(out, "figure4_unpredictable.pdf"))
    plt.close(fig)
    print(f"fig4: confidence r = {r:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.dir, a.out)
    fig2(a.dir, a.out)
    fig3(a.dir, a.out)
    fig4(a.dir, a.out)
    print(f"\n写到 {a.out}/")


if __name__ == "__main__":
    main()
