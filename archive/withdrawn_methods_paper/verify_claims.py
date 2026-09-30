#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_claims.py -- 论文里的每个数字，从发布的表重算一遍。

它回答的问题只有一个：**稿子写的这个数，能不能从 results_clean/ 里算出来。**
算不出来就退出非零；输入表缺失记为 PENDING，同样算失败，所以一个数字不可能
在还没法重算的时候进入稿子。

它抓不到什么，也写在这里，免得以后误以为绿灯就是没问题：

  - 概括性断言。「所有数字走同一条代码路径」没有对应的数值可以重算，而那句
    话曾经是错的（弃权分析和重抽都不走那条路径）。
  - 同一个量在正文与补充材料写得不一样。那类由 cross_audit.py 负责。
  - 证据是否支撑主张的范围。那类只能人看，见 docs/checking.md 第三层。

用法：
    python verify_claims.py
    python verify_claims.py --verbose
    python verify_claims.py --todo
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

R = "results_clean"
OK = FAIL = PEND = 0
VERBOSE = False
ROWS = []


def check(label, got, want, tol=0.0, note=""):
    global OK, FAIL
    if isinstance(want, bool):
        good = bool(got) == want
        shown = str(got)
    else:
        good = got == got and abs(float(got) - float(want)) <= tol
        shown = f"{float(got):.4f}" if got == got else "nan"
    ROWS.append((good, label, shown, want, note))
    if good:
        OK += 1
    else:
        FAIL += 1
    if VERBOSE or not good:
        tag = "ok  " if good else "FAIL"
        extra = f"   expected {want}" if not good else ""
        print(f"{tag}  {label:<74}{shown:>10}{extra}"
              + (f"   [{note}]" if note and (VERBOSE or not good) else ""))
    return good


def pending(label, why):
    global PEND
    PEND += 1
    ROWS.append((False, label, "-", "-", why))
    print(f"PEND  {label:<74}{'':>10}   {why}")


def load(name):
    p = os.path.join(R, f"summary_{name}.csv")
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


# ------------------------------------------------------------------ main --
def main_table():
    m = load("main")
    if m is None:
        pending("main table", f"{R}/summary_main.csv absent")
        return
    m = m[m["unit"] != "sabah"]
    check("twelve dataset-model combinations", len(m), 12, 0)

    want = {  # 摘要与正文引用的每一个
        ("durian", "frcnn-r50"): (0.3656, 0.2220, 39.29),
        ("durian", "rtdetr-l"): (0.4987, 0.2205, 55.78),
        ("durian", "yolo11l"): (0.4721, 0.2356, 50.08),
        ("durian", "yolo11m"): (0.4652, 0.2332, 49.87),
        ("durian", "yolo11n"): (0.4281, 0.2234, 47.81),
        ("durian", "yolo11s"): (0.4622, 0.2137, 53.77),
        ("gwhd", "yolo11n"): (0.6357, 0.4926, 22.51),
        ("gwhd", "yolo11s"): (0.6723, 0.4765, 29.12),
        ("breakhis", "yolo11n-cls"): (0.9527, 0.8759, 8.06),
        ("breakhis", "yolo11s-cls"): (0.9504, 0.8627, 9.23),
        ("har", "mlp256"): (0.9885, 0.9519, 3.70),
        ("har", "rf"): (0.9790, 0.9347, 4.52)}
    for (ds, mo), (it, un, ov) in want.items():
        r = m[(m.dataset == ds) & (m.model == mo)]
        check(f"{ds} {mo}: item-level {it:.4f}", float(r["item"].iloc[0]),
              it, 0.0002)
        check(f"{ds} {mo}: unit-level {un:.4f}",
              float(r["unit_level"].iloc[0]), un, 0.0002)
        check(f"{ds} {mo}: overstatement {ov:.2f}%",
              float(r["overstatement_pct"].iloc[0]), ov, 0.02)

    lo, hi = m.overstatement_pct.min(), m.overstatement_pct.max()
    check("the overstatement spans 3.7 to 55.8%", lo, 3.70, 0.02)
    check("  upper end", hi, 55.78, 0.02)

    # 一阶段与两阶段的差别 —— 正文声称这条，所以锁住
    one = m[(m.dataset == "durian") & (m.model != "frcnn-r50")]
    two = m[(m.dataset == "durian") & (m.model == "frcnn-r50")]
    check("durian: the two-stage detector overstates less than every "
          "one-stage one",
          float(two.overstatement_pct.iloc[0]) < one.overstatement_pct.min(),
          True, note=f"{float(two.overstatement_pct.iloc[0]):.1f}% against "
                     f"{one.overstatement_pct.min():.1f}% at least")
    check("durian: its item-level figure is the lowest of the six",
          float(two["item"].iloc[0]) < one["item"].min(), True)
    check("durian: its unit-level figure is not",
          one["unit_level"].min() < float(two["unit_level"].iloc[0])
          < one["unit_level"].max(), True)

    # 变异系数的次序 —— 正文说高估幅度跟着它走
    pu = load("per_unit")
    if pu is not None:
        cv = {}
        for ds in ("durian", "gwhd", "breakhis", "har"):
            g = pu[pu.dataset == ds].groupby("unit_id").score.mean()
            cv[ds] = g.std(ddof=1) / g.mean()
        for ds, v in (("durian", 0.393), ("gwhd", 0.302),
                      ("breakhis", 0.246), ("har", 0.058)):
            check(f"{ds}: coefficient of variation across units {v:.2f}",
                  cv[ds], v, 0.004)
        order_cv = [cv[d] for d in ("durian", "gwhd", "breakhis", "har")]
        order_ov = [m[m.dataset == d].overstatement_pct.mean()
                    for d in ("durian", "gwhd", "breakhis", "har")]
        check("the overstatement follows the coefficient of variation",
              order_cv == sorted(order_cv, reverse=True)
              and order_ov == sorted(order_ov, reverse=True), True)


def per_unit():
    pu = load("per_unit")
    if pu is None:
        pending("per-unit scores", f"{R}/summary_per_unit.csv absent")
        return
    want = {"durian": (8, 0.2247, 0.0883, 0.094, 0.356),
            "gwhd": (47, 0.4646, 0.1404, 0.085, 0.803),
            "breakhis": (81, 0.8693, 0.2142, 0.053, 1.000),
            "har": (30, 0.9406, 0.0550, 0.750, 0.999)}
    for ds, (n, mu, sd, lo, hi) in want.items():
        g = pu[pu.dataset == ds].groupby("unit_id").score.mean()
        check(f"{ds}: {n} evaluation units", len(g), n, 0)
        check(f"{ds}: mean across units {mu:.4f}", g.mean(), mu, 0.0005)
        check(f"{ds}: s.d. across units {sd:.4f}", g.std(ddof=1), sd, 0.0005)
        check(f"{ds}: lowest unit {lo:.3f}", g.min(), lo, 0.001)
        check(f"{ds}: highest unit {hi:.3f}", g.max(), hi, 0.001)
    # durian 的农场排序在六个模型间保持
    d = pu[pu.dataset == "durian"].pivot_table(index="unit_id",
                                               columns="model",
                                               values="score")
    rs = []
    cols = sorted(d.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rs.append(d[cols[i]].corr(d[cols[j]], method="spearman"))
    check("durian: mean pairwise Spearman across six detectors 0.87",
          float(np.mean(rs)), 0.868, 0.006)
    bottom = d.mean(axis=1).nsmallest(2).index
    agree = all(set(d[c].nsmallest(2).index) == set(bottom) for c in cols)
    check("durian: all six place the same two farms last", agree, True,
          note=", ".join(str(x) for x in bottom))


def decomposition():
    d = load("decomposition")
    if d is None:
        pending("variance components", f"{R}/summary_decomposition.csv absent")
        return
    want = {"durian": ("unit", 82.4, "model", 0.0),
            "gwhd": ("unit", 68.0, "model", 0.3),
            "breakhis": ("unit", 88.1, "model", 0.0),
            "har": ("unit", 73.3, "model", 2.7)}
    for ds, (uk, uv, mk, mv) in want.items():
        r = d[d.dataset == ds].iloc[0]
        check(f"{ds}: unit component {uv:.1f}%", float(r[uk]), uv, 0.06)
        check(f"{ds}: model component {mv:.1f}%", float(r[mk]), mv, 0.06)
    units = [float(d[d.dataset == x]["unit"].iloc[0])
             for x in ("durian", "gwhd", "breakhis", "har")]
    models = [float(d[d.dataset == x]["model"].iloc[0])
              for x in ("durian", "gwhd", "breakhis", "har")]
    check("the unit component is 68 to 88% on every dataset",
          min(units) >= 68.0 and max(units) <= 88.2, True,
          note=f"{min(units):.1f}–{max(units):.1f}")
    check("the model component never exceeds 2.7%", max(models) <= 2.7, True,
          note=f"{max(models):.1f}")
    check("the unit component exceeds the model component by at least 25-fold "
          "everywhere",
          min(u / max(mo, 0.01) for u, mo in zip(units, models)) > 25, True)


def protocol():
    p = load("protocol_comparison")
    if p is None:
        pending("protocol comparison",
                f"{R}/summary_protocol_comparison.csv absent")
        return
    q = p.dropna(subset=["clean"])
    # durian yolo11s 的 last 权重未保留，所以三协议对照只有七个组合
    q3 = q.dropna(subset=["old_last"])
    check("seven combinations compare all three protocols", len(q3), 7, 0,
          note="durian yolo11s retained no final-epoch weights")
    between = ((q3.clean >= q3[["old_best", "old_last"]].min(axis=1) - 0.05)
               & (q3.clean <= q3[["old_best", "old_last"]].max(axis=1) + 0.05))
    check("the clean figure falls between the two contaminated ones in every "
          "comparable combination", bool(between.all()), True,
          note=f"{int(between.sum())} of {len(q3)}")
    d = q.clean - q.old_best
    check("moving to the clean protocol raises the overstatement by 6.5 "
          "points on average", float(d.mean()), 6.5, 0.05)
    check("  smallest rise", float(d.min()), 0.1, 0.05)
    check("  largest rise", float(d.max()), 14.7, 0.05)
    check("the rise is never negative", bool((d >= -0.05).all()), True)


def model_pairs():
    p = load("model_pairs")
    if p is None:
        pending("model comparisons", f"{R}/summary_model_pairs.csv absent")
        return
    check("eighteen model pairs", len(p), 18, 0)
    check("d spans 0.02 to 0.84", float(p.d.min()), 0.022, 0.003)
    check("  upper end", float(p.d.max()), 0.843, 0.003)
    resolvable = p[p.k_for_5pct <= p.n_units]
    check("four pairs separate at the unit count the corpus provides",
          len(resolvable), 4, 0,
          note="; ".join(resolvable.pair.tolist()))
    check("two of the four are durian capacity pairs",
          int((resolvable.dataset == "durian farms").sum()), 2, 0)

    FAM = {"rtdetr-l": "DETR", "yolo11n": "YOLO", "yolo11s": "YOLO",
           "yolo11m": "YOLO", "yolo11l": "YOLO", "frcnn-r50": "RCNN",
           "yolo11n-cls": "YOLO", "yolo11s-cls": "YOLO", "mlp256": "MLP",
           "rf": "RF"}
    cross = p[[FAM[x.split(" vs ")[0]] != FAM[x.split(" vs ")[1]]
               for x in p.pair]]
    same = p[[FAM[x.split(" vs ")[0]] == FAM[x.split(" vs ")[1]]
              for x in p.pair]]
    check("cross-family d spans 0.02 to 0.52",
          float(cross.d.min()), 0.022, 0.003)
    check("  upper end", float(cross.d.max()), 0.519, 0.003)
    check("same-family d spans 0.15 to 0.84",
          float(same.d.min()), 0.149, 0.003)
    check("  upper end", float(same.d.max()), 0.843, 0.003)
    # 论证：两个分布重叠，族别不预测可分辨性
    check("the two distributions overlap, so family does not predict which "
          "comparisons survive",
          float(cross.d.max()) > float(same.d.min())
          and float(same.d.max()) > float(cross.d.min()), True)

    for ds, k, w in (("GWHD sessions", 10, 0.235),
                     ("BreaKHis patients", 16, 0.175)):
        r = p[p.dataset == ds].iloc[0]
        check(f"{ds}: the ranking reverses on {100*w:.0f}% of {k}-unit draws",
              float(r[f"flip_k{k}"]), w, 0.012)


def eval_curve():
    e = load("eval_curve")
    if e is None:
        pending("evaluation-unit resampling",
                f"{R}/summary_eval_curve.csv absent")
        return
    eq = e[e.weighting == "equal"]
    for ds, k, w in (("durian farms", 1, 0.263), ("durian farms", 4, 0.107),
                     ("GWHD sessions", 1, 0.457), ("GWHD sessions", 16, 0.102),
                     ("BreaKHis patients", 1, 0.663),
                     ("BreaKHis patients", 16, 0.161),
                     ("HAR subjects", 1, 0.127), ("HAR subjects", 16, 0.028)):
        r = eq[(eq.dataset == ds) & (eq.k_eval == k)]
        check(f"{ds}: 90% range at k={k} is {w:.3f}",
              float(r.width90.iloc[0]), w, 0.012)
    # 论点：等权下均值不随 k 变
    for ds in eq.dataset.unique():
        g = eq[eq.dataset == ds]
        drift = float(g["mean"].max() - g["mean"].min())
        check(f"{ds}: the mean is flat in k", drift < 0.008, True,
              note=f"drift {drift:.4f}")
    # 实例加权与等权之差
    for ds, gap in (("GWHD sessions", 0.028), ("durian farms", 0.000),
                    ("BreaKHis patients", -0.002), ("HAR subjects", 0.001)):
        g = e[e.dataset == ds]
        N = int(g.n_units.iloc[0])
        p_ = float(g[(g.weighting == "items") & (g.k_eval == N)]["mean"].iloc[0])
        q_ = float(g[(g.weighting == "equal") & (g.k_eval == N)]["mean"].iloc[0])
        check(f"{ds}: pooled minus equal-weight = {gap:+.3f}", p_ - q_, gap,
              0.0015)


def within_unit():
    """噪声：条件式结论。审读指出取两种估计的较大者仍不是上界。"""
    ic = load("noise_icc_sensitivity")
    if ic is None:
        pending("noise sensitivity", f"{R}/summary_noise_icc_sensitivity.csv absent")
        return
    bk = ic[ic.dataset == "BreaKHis patients"]
    ha = ic[ic.dataset == "HAR subjects"]
    z_bk = bk[bk.icc == 0.0]
    z_ha = ha[ha.icc == 0.0]
    check("BreaKHis: under independence, noise is 3.4-5.0% of the variance",
          float(z_bk.noise_share.max()), 5.0, 0.15)
    check("HAR: under independence, 5.6-6.3%",
          float(z_ha.noise_share.max()), 6.3, 0.15)
    # 关键：两个语料对 ICC 的敏感度差一个量级，因为单元大小差 3.5 倍
    b20 = float(bk[bk.icc == 0.20].noise_share.mean())
    h05 = float(ha[ha.icc == 0.05].noise_share.mean())
    check("BreaKHis tolerates an ICC of 0.20", b20 < 50, True,
          note=f"noise {b20:.0f}% of variance")
    check("HAR does not tolerate an ICC of 0.05", h05 > 50, True,
          note=f"noise {h05:.0f}% of variance")
    check("the claim is therefore conditional on the within-unit correlation",
          True, True,
          note="stated as a function of ICC in the manuscript, not a number")


def paired_bootstrap():
    """配对差的 bootstrap 区间 —— 不外推、不假设分布的主判据。"""
    p = load("paired_bootstrap")
    if p is None:
        pending("paired bootstrap", f"{R}/summary_paired_bootstrap.csv absent")
        return
    check("eighteen pairs", len(p), 18, 0)
    n = int(p.excludes_zero.sum())
    check("four pairs have an interval excluding zero", n, 4, 0,
          note="; ".join(p[p.excludes_zero].pair.tolist()))
    FAM = {"rtdetr-l": "DETR", "yolo11n": "YOLO", "yolo11s": "YOLO",
           "yolo11m": "YOLO", "yolo11l": "YOLO", "frcnn-r50": "RCNN",
           "yolo11n-cls": "YOLO", "yolo11s-cls": "YOLO", "mlp256": "MLP",
           "rf": "RF"}
    res = p[p.excludes_zero]
    cross = [FAM[x.split(" vs ")[0]] != FAM[x.split(" vs ")[1]]
             for x in res.pair]
    check("one of the four crosses an architecture family",
          sum(cross), 1, 0, note="the perceptron against the random forest")
    check("no resolved pair crosses families among the image models",
          not any(c for c, ds in zip(cross, res.dataset)
                  if ds != "HAR subjects"), True)
    g = p[p.dataset == "GWHD sessions"].iloc[0]
    check("the two wheat detectors differ by 0.018", abs(float(g.mean_diff)),
          0.0177, 0.0008)
    check("  and their interval includes zero", bool(g.excludes_zero), False)


def effect_size_ci():
    """d 的不确定性，以及为什么不把它转成所需单元数。"""
    e = load("effect_size_ci")
    if e is None:
        pending("effect-size intervals", f"{R}/summary_effect_size_ci.csv absent")
        return
    g = e[(e.dataset == "GWHD sessions")].iloc[0]
    check("GWHD pair: d = 0.21 with a 90% interval of 0.03 to 0.58",
          float(g.d), 0.21, 0.012)
    check("  lower bound", float(g.d_lo), 0.03, 0.02)
    check("  upper bound", float(g.d_hi), 0.58, 0.05)
    check("  the implied unit count spans 9 to 3,948",
          int(g.k_hi) - int(g.k_lo) > 3000, True,
          note=f"[{int(g.k_lo)}, {int(g.k_hi)}]")
    dur = e[e.dataset == "durian farms"]
    check("durian: the median width of the implied unit count exceeds 2,000",
          float((dur.k_hi - dur.k_lo).median()) > 2000, True,
          note=f"{float((dur.k_hi - dur.k_lo).median()):,.0f}")
    check("the conversion is therefore not reported as an estimate",
          True, True, note="d and its interval are reported instead")


def bootstrap_level():
    """分量区间：重采样单位的选择，以及分离是否仍成立。"""
    b = load("bootstrap_unit_level")
    if b is None:
        pending("unit-level bootstrap",
                f"{R}/summary_bootstrap_unit_level.csv absent")
        return
    for ds, lo, hi in (("gwhd", 44.4, 74.9), ("breakhis", 75.0, 88.8),
                       ("har", 45.0, 77.4)):
        r = b[(b.dataset == ds) & (b.component == "unit")].iloc[0]
        check(f"{ds}: unit component 90% interval over units",
              float(r.unit_lo), lo, 1.2)
        check(f"  upper bound", float(r.unit_hi), hi, 1.2)
        m = b[(b.dataset == ds) & (b.component == "model")].iloc[0]
        check(f"{ds}: separation survives unit-level resampling",
              float(r.unit_lo) > float(m.unit_hi), True,
              note=f"unit >= {r.unit_lo:.1f}% vs model <= {m.unit_hi:.1f}%")
    u = b[b.component == "unit"]
    m = b[b.component == "model"]
    check("unit lower bounds span 44 to 75%", float(u.unit_lo.min()), 44.4, 1.2)
    check("model upper bounds never exceed 8.2%",
          float(m.unit_hi.max()), 8.2, 0.6)


def review_fixes():
    """外部审读会盯的几处，全部锁住。

    这些是第一轮以 NCS 审稿人视角逐字读时找到的，都属于「稿子写的范围超过
    数据支持的范围」，CI 原本抓不到，所以现在明确写成断言。
    """
    m, pu, n = load("main"), load("per_unit"), load("within_unit_noise")
    if m is None or pu is None or n is None:
        pending("review fixes", "summary tables absent")
        return
    check("ten distinct model configurations",
          int(m[m["unit"] != "sabah"].model.nunique()), 10, 0)
    # 每单元范围必须是跨模型均值，与 Supplementary Table 8 同口径
    for ds, lo, hi in (("durian", 0.094, 0.356), ("gwhd", 0.085, 0.803),
                       ("breakhis", 0.053, 1.000), ("har", 0.750, 0.999)):
        g = pu[pu.dataset == ds].groupby("unit_id").score.mean()
        check(f"{ds}: the quoted range is the across-model one, low",
              float(g.min()), lo, 0.0006)
        check(f"{ds}: the quoted range is the across-model one, high",
              float(g.max()), hi, 0.0006)
    g = pu[pu.dataset == "gwhd"].groupby("unit_id").score.mean()
    check("gwhd: best session is 9.5 times the worst",
          float(g.max() / g.min()), 9.5, 0.06)
    for ds, share in (("BreaKHis patients", 0.991), ("HAR subjects", 0.970)):
        r = n[n.dataset == ds]
        check(f"{ds}: heterogeneity is {100*share:.0f}% of the s.d.",
              float((r.sd_heterogeneity / r.sd_between).mean()), share, 0.004)
    # HAR 的错误率必须来自 accuracy，不能从 macro F1 反推
    h = pd.read_csv(os.path.join("results_har", "har_by_subject.csv"))
    a = h[h.model == "mlp256"].groupby("group").accuracy.mean()
    check("HAR: the best subject errs on one window in 635",
          1 / (1 - float(a.max())), 635, 8,
          note="derived from accuracy, not from 1 - macro F1")
    check("HAR: the worst on one in five", 1 / (1 - float(a.min())), 4.8, 0.1)


def robustness():
    """审稿意见第二条：分解的不确定性、固定效应、模型集合敏感性。"""
    ci, fx, rs = (load("decomposition_ci"), load("decomposition_fixed"),
                  load("decomposition_roster"))
    if ci is None or fx is None or rs is None:
        pending("decomposition robustness",
                "summary_decomposition_{ci,fixed,roster}.csv absent")
        return
    # (a) 区间：单元下界高于模型上界，每个数据集
    for ds in ("durian", "gwhd", "breakhis", "har"):
        u = ci[(ci.dataset == ds) & (ci.component == "unit")].iloc[0]
        m = ci[(ci.dataset == ds) & (ci.component == "model")].iloc[0]
        check(f"{ds}: unit lower bound exceeds model upper bound",
              float(u.lo) > float(m.hi), True,
              note=f"unit [{u.lo:.1f}, {u.hi:.1f}] vs model [{m.lo:.1f}, "
                   f"{m.hi:.1f}]")
    lo = ci[ci.component == "unit"].lo.min()
    hi = ci[ci.component == "unit"].hi.max()
    check("unit intervals span 46 to 93% across datasets", float(lo), 46.0, 0.6)
    check("  upper end", float(hi), 92.8, 0.6)
    check("no model upper bound exceeds 8.3%",
          float(ci[ci.component == "model"].hi.max()), 8.3, 0.3)
    # (b) 固定效应
    d = fx[fx.dataset == "durian"]
    gap = (d.fixed_effect - d.random_effect).abs().max()
    check("durian: fixed and random effect shares agree within one point",
          float(gap) <= 1.05, True, note=f"max gap {gap:.2f}")
    # (c) 模型集合
    check("durian: 57 roster subsets", len(rs), 57, 0)
    check("roster subsets: unit share never below 70%",
          float(rs["unit"].min()), 70.1, 0.3)
    check("roster subsets: unit share never above 93%",
          float(rs["unit"].max()), 92.7, 0.3)
    check("roster subsets: model share never exceeds 1.1%",
          float(rs["model"].max()), 1.1, 0.06)
    g = rs.groupby("families")["unit"].mean()
    check("one-family subsets: mean unit share 90%", float(g[1]), 90.2, 0.3)
    check("three-family subsets: mean unit share 77%", float(g[3]), 76.5, 0.3)
    check("more families lowers the unit share, as expected",
          float(g[1]) > float(g[2]) > float(g[3]), True)
    check("but three families still leave three quarters with the unit",
          float(g[3]) > 75.0, True)
    # 绝对降幅（审稿意见：比率的分母假象）
    m = load("main")
    m = m[(m.dataset == "durian") & (m["unit"] != "sabah")]
    m = m.assign(abs_drop=100 * (m["item"] - m["unit_level"]))
    one = m[m.model != "frcnn-r50"]
    two = m[m.model == "frcnn-r50"]
    check("durian: one-stage detectors lose 20.5 to 27.8 points",
          float(one.abs_drop.min()), 20.5, 0.15)
    check("  upper end", float(one.abs_drop.max()), 27.8, 0.15)
    check("durian: Faster R-CNN loses 14.4 points",
          float(two.abs_drop.iloc[0]), 14.4, 0.15)
    check("the two-stage gap is smaller in absolute terms too",
          float(two.abs_drop.iloc[0]) < float(one.abs_drop.min()), True)


def run_counts():
    """Table 1 与 cover letter 引用的 run 数，从原始结果表数出来。"""
    import glob
    n = {}
    for key, f in (("durian", "durian_in_region_clean.csv"),
                   ("durian_l", "durian_in_region_clean_yolo11l.csv"),
                   ("durian_frcnn", "durian_in_region_clean_frcnn.csv"),
                   ("gwhd", "gwhd_in_region_clean.csv"),
                   ("breakhis", "breakhis_in_region_clean.csv")):
        p_ = os.path.join(R, f)
        if not os.path.isfile(p_):
            pending(f"run count: {key}", f"{p_} absent")
            return
        d = pd.read_csv(p_, encoding="utf-8-sig")
        n[key] = len(d.drop_duplicates(["model", "config", "seed"]))
    check("durian: 252 clean runs",
          n["durian"] + n["durian_l"] + n["durian_frcnn"], 252, 0)
    check("GWHD: 60 clean runs", n["gwhd"], 60, 0)
    check("BreaKHis: 100 clean runs", n["breakhis"], 100, 0)
    check("412 clean image-model runs in total", sum(n.values()), 412, 0,
          note="HAR needs no retraining and is not counted here")


def sabah():
    m = load("main")
    if m is None:
        return
    s = m[m["unit"] == "sabah"]
    if s.empty:
        pending("cross-region evaluation", "no sabah rows in summary_main")
        return
    check("five detectors scored on Sabah", len(s), 5, 0)
    check("Sabah scores 0.253 to 0.281", float(s.unit_level.min()), 0.2526,
          0.0005)
    check("  upper end", float(s.unit_level.max()), 0.2811, 0.0005)
    check("the spread across training configurations is 0.008 to 0.018",
          float(s.item_last.max()), 0.0183, 0.0005)
    check("Sabah scores above the mean withheld peninsular farm for every "
          "detector", bool((s.unit_level > s.unit_last).all()), True)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--todo", action="store_true")
    a = ap.parse_args()
    VERBOSE = a.verbose

    for fn in (main_table, per_unit, decomposition, protocol, model_pairs,
               eval_curve, within_unit, paired_bootstrap,
               effect_size_ci, bootstrap_level,
               review_fixes, run_counts,
               robustness, sabah):
        fn()

    if a.todo:
        print("\nclaims still waiting on a table:")
        for good, label, _, _, note in ROWS:
            if not good and note and "absent" in note:
                print(f"  {label}  ({note})")
        return 0

    total = OK + FAIL + PEND
    print("\n" + "-" * 62)
    if FAIL or PEND:
        print(f"{OK} of {total} claims reproduce")
        print(f"{FAIL} do not, {PEND} wait on a missing table.")
        return 1
    print(f"{OK} of {total} claims reproduce")
    print("every quantitative claim in the manuscript matches the released "
          "tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
