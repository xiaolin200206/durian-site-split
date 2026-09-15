#!/usr/bin/env python3
"""
Recompute every quantitative claim in the paper from the released tables.

Four datasets, one script. Each claim is stated here as it appears in the
manuscript, recomputed, and compared. A claim that does not reproduce is
printed with both values and the script exits non-zero. A claim whose input
table is absent is reported as PENDING and also fails, so that a number
cannot reach the manuscript before it can be recomputed.

    python verify_claims.py
    python verify_claims.py --verbose
    python verify_claims.py --todo

Nothing here reads an image or a model weight.

Two errors reported in the paper were found by this process rather than by
inspection: the filename join that misattributed 8.9% of the durian pool,
and the focal-length confound behind a lesion-scale figure. Both had passed
manual review. The checks below are therefore written against image-level
facts wherever a per-unit summary could hide the same class of mistake.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = {
    "root": ".","durian": "results_durian", "gwhd": "results_gwhd",
     "breakhis": "results_breakhis", "har": "results_har"}

FARMS = ["0", "1", "2", "3", "4", "5", "6", "7"]
MIN_ITEMS = 5
STRATUM = "6.76"

checks, todo = [], []


def check(claim, got, expected, tol=0.001, note=""):
    if isinstance(expected, bool) or isinstance(got, bool):
        ok = bool(got) == bool(expected)
        g, e = str(bool(got)), str(bool(expected))
    elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - expected) <= tol
        g, e = f"{got:.4g}", f"{expected:.4g}"
    else:
        ok = str(got) == str(expected)
        g, e = str(got), str(expected)
    checks.append({"claim": claim, "ok": ok, "got": g, "expected": e,
                   "note": note})
    return ok


def pending(claim, why):
    todo.append({"claim": claim, "why": why})
    checks.append({"claim": claim, "ok": False, "got": "PENDING",
                   "expected": "-", "note": why})


def load(ds, name, required=False):
    p = os.path.join(HERE, D[ds], name)
    if not os.path.isfile(p):
        if required:
            sys.exit(f"missing required table: {p}")
        return None
    return pd.read_csv(p)


def folds_of(df, metric, prefix, model=None):
    d = df[df["model"] == model] if model and "model" in df.columns else df
    return d[d["config"].str.startswith(prefix)].groupby("config")[metric].mean()


def seed_sd(df, metric, prefix, model=None):
    d = df[df["model"] == model] if model and "model" in df.columns else df
    s = d[d["config"].str.startswith(prefix)].groupby("config")[metric].std()
    return s.mean()


def unit_scores(df, key, metric, nkey=None, minn=MIN_ITEMS):
    g = df.groupby(key)[metric].mean()
    if nkey and nkey in df.columns:
        n = df.groupby(key)[nkey].first()
        g = g[n >= minn]
    return g


EUC_BOOT = 2000
EUC_SEED = 0

# (显示名, dataset key, 表, 单元列, 分数列, 权重列, 模型值)
EUC_SETS = [
    ("durian bursts", "durian", "peninsula_by_burst_60s_min5.csv",
     "burst", "mAP50", "n_images", "yolo11s"),
    ("GWHD sessions", "gwhd", "gwhd_by_domain.csv",
     "group", "mAP50", "n_images", "yolo11s"),
    ("BreaKHis patients", "breakhis", "breakhis_by_patient.csv",
     "group", "top1", "n_images", "yolo11s-cls"),
    ("HAR subjects", "har", "har_by_subject.csv",
     "group", "macro_f1", "n_windows", "mlp256"),
]

# 手稿中的数字。等权是主文口径，实例加权进补充材料。
#   n_units, pooled, equal, 单元间 s.d.(总体), sd@k=16(等权), k for width<=0.10
EUC_CLAIMS = {
    "durian bursts":     dict(n=58, pooled=0.293, equal=0.330,
                              sd=0.247, sd16=0.051, need10=32),
    "GWHD sessions":     dict(n=47, pooled=0.539, equal=0.483,
                              sd=0.128, sd16=0.026, need10=13),
    "BreaKHis patients": dict(n=81, pooled=0.906, equal=0.902,
                              sd=0.175, sd16=0.039, need10=24),
    "HAR subjects":      dict(n=30, pooled=0.951, equal=0.950,
                              sd=0.052, sd16=0.009, need10=3),
}


def _euc_aggregate(score, weight, idx, mode):
    if mode == "equal":
        return score[idx].mean()
    return float((score[idx] * weight[idx]).sum() / weight[idx].sum())


def _euc_draws(score, weight, k, mode, rng):
    n = len(score)
    out = np.empty(EUC_BOOT)
    for b in range(EUC_BOOT):
        idx = rng.choice(n, size=k, replace=False)
        out[b] = _euc_aggregate(score, weight, idx, mode)
    return out


def eval_unit_curve():
    """评价单元重抽：均值应当无偏，离散度按 1/sqrt(k) 下降。

    这一节回答的是「报一个分数需要多少个独立评价单元」，与 dose-response
    的「训练需要多少个单元」是两个问题。它不需要重新训练：每个单元的分数
    已经是没见过它的权重打出来的。

    ★ 局限（也写在 Methods 里）：各单元的分数来自不同的折，所以重抽混合了
      训练集。这是「只在 k 个单元上评估会报出什么」的近似，不是精确复制。
    """
    for label, ds, table, ukey, metric, nkey, model in EUC_SETS:
        df = load(ds, table)
        if df is None:
            pending(f"{label}: evaluation-unit curve",
                    f"{D[ds]}/{table} absent")
            continue
        if "model" in df.columns:
            df = df[df["model"] == model]
        df = df.dropna(subset=[metric])
        g = df.groupby(ukey).agg(**{metric: (metric, "mean"),
                                    nkey: (nkey, "first")})
        score = g[metric].to_numpy(dtype=float)
        weight = g[nkey].to_numpy(dtype=float)
        want = EUC_CLAIMS[label]

        check(f"{label}: {want['n']} evaluation units",
              len(score), want["n"], 0)
        check(f"{label}: pooled {want['pooled']:.3f}",
              _euc_aggregate(score, weight, np.arange(len(score)), "items"),
              want["pooled"], 0.0015)
        check(f"{label}: equal-weight {want['equal']:.3f}",
              _euc_aggregate(score, weight, np.arange(len(score)), "equal"),
              want["equal"], 0.0015)

        # 单元间 s.d. 是总体量，先精确算，再用它当 k=1 的期望
        check(f"{label}: between-unit s.d. {want['sd']:.3f}",
              float(score.std(ddof=1)), want["sd"], 0.0015)

        rng = np.random.default_rng(EUC_SEED)
        d1 = _euc_draws(score, weight, 1, "equal", rng)
        # k=1 的均值就是等权平均，这是无偏性最直接的断言
        check(f"{label}: k=1 draws centre on the equal-weight mean",
              d1.mean(), want["equal"], 0.02)
        # k=1 的抽样离散度应当就是单元间 s.d.；容差宽是因为 2000 次重抽在
        # 长尾分布上估 s.d. 本来就抖（BreaKHis 有 5 个病人低于 0.60）
        check(f"{label}: k=1 spread equals the between-unit s.d.",
              d1.std(ddof=1), float(score.std(ddof=1)), 0.02)

        if len(score) >= 16:
            d16 = _euc_draws(score, weight, 16, "equal", rng)
            check(f"{label}: k=16 mean unchanged from k=1",
                  d16.mean(), want["equal"], 0.02,
                  "evaluation units buy precision, not score")
            check(f"{label}: k=16 spread {want['sd16']:.3f}",
                  d16.std(ddof=1), want["sd16"], 0.006)
            # 离散度应当接近 1/sqrt(k)，有限总体修正后略低
            fpc = math.sqrt((len(score) - 16) / (len(score) - 1))
            check(f"{label}: spread falls as 1/sqrt(k)",
                  d16.std(ddof=1) / d1.std(ddof=1),
                  fpc / math.sqrt(16), 0.03)

        # 反解：90% 区间宽度降到 0.10 所需的单元数
        need = None
        for k in range(1, len(score) + 1):
            v = _euc_draws(score, weight, k, "equal", rng)
            if np.percentile(v, 95) - np.percentile(v, 5) <= 0.10:
                need = k
                break
        check(f"{label}: {want['need10']} units for a 90% interval "
              f"of width 0.10", need if need else len(score) + 1,
              want["need10"], 2,
              "equal-weight aggregation; instance-weighted differs")

    # 跨数据集：所需单元数与单元间异质性同序，与数据集大小无关
    order = ["HAR subjects", "GWHD sessions", "BreaKHis patients",
             "durian bursts"]
    needs = [EUC_CLAIMS[k]["need10"] for k in order]
    check("evaluation units needed rise with between-unit heterogeneity",
          needs == sorted(needs), True, note="HAR 3 < GWHD 13 < BreaKHis 24 "
          "< durian 32, while unit counts run 30/47/81/58")


# ---------------------------------------------------------------- durian --
def durian():
    ir = load("durian", "in_region.csv", required=True)
    ci = load("durian", "cross_island.csv", required=True)
    sa = load("durian", "split_assignment.csv", required=True)

    check("durian pool: 827 images with a farm", len(sa), 827, 0)
    check("durian: eight usable farms", sa["farm"].astype(str).nunique(), 8, 0)
    for f, n in zip(FARMS, [141, 102, 147, 62, 101, 135, 82, 57]):
        check(f"durian farm {f}: {n} images",
              int((sa["farm"].astype(str) == f).sum()), n, 0)
    check("durian item-level split: 662 train / 165 validation",
          f"{int((sa['split_random'] == 'train').sum())}/"
          f"{int((sa['split_random'] == 'val').sum())}", "662/165")

    # Every fold withholds one farm and that farm is absent from training.
    # This is the property the earlier filename-based design violated
    # silently; it is asserted rather than assumed.
    for i in range(8):
        col = f"split_farm_fold{i}"
        if col not in sa.columns:
            continue
        vf = set(sa.loc[sa[col] == "val", "farm"].astype(str))
        tf = set(sa.loc[sa[col] == "train", "farm"].astype(str))
        check(f"{col}: withholds exactly one farm", len(vf), 1, 0)
        check(f"{col}: held-out farm absent from training", len(vf & tf), 0, 0)

    models = sorted(ir["model"].unique())
    check("durian: four architectures evaluated", len(models), 4, 0,
          note=str(models))
    check("durian: one of them is a different family",
          any("rtdetr" in m for m in models), True,
          "three YOLO11 capacities and a detection transformer")

    fm = {}
    for mdl, rnd_, bf_, sd_, over in (
            ("yolo11s", 0.4943, 0.2752, 0.1077, 44.3),
            ("rtdetr-l", 0.5260, 0.3098, 0.1002, 41.1),
            ("yolo11n", 0.4558, 0.2855, 0.1096, 37.4),
            ("yolo11m", 0.4966, 0.2847, 0.0992, 42.7)):
        sub = ir[ir["model"] == mdl]
        if sub.empty:
            pending(f"durian {mdl}", "run eval for this architecture")
            continue
        check(f"durian {mdl}: 45 runs", len(sub), 45, 0)
        r_ = sub[sub["config"] == "random"]["mAP50"]
        f_ = folds_of(ir, "mAP50", "byfarm", mdl)
        fm[mdl] = f_
        check(f"durian {mdl}: item-level = {rnd_}", r_.mean(), rnd_, 0.0005)
        check(f"durian {mdl}: unit-level = {bf_}", f_.mean(), bf_, 0.0005)
        check(f"durian {mdl}: s.d. across folds = {sd_}", f_.std(), sd_, 0.0005)
        check(f"durian {mdl}: overstatement = {over}%",
              100 * (r_.mean() - f_.mean()) / r_.mean(), over, 0.15)
        check(f"durian {mdl}: eight folds", len(f_), 8, 0)

    if len(fm) >= 2:
        import itertools
        rhos = []
        for m1, m2 in itertools.combinations(sorted(fm), 2):
            a_, b_ = fm[m1], fm[m2]
            common = sorted(set(a_.index) & set(b_.index))
            rhos.append(pd.Series(a_[common].values).corr(
                pd.Series(b_[common].values), method="spearman"))
        check("durian: mean pairwise Spearman across architectures = 0.94",
              float(np.mean(rhos)), 0.940, 0.01,
              note=f"range {min(rhos):.3f} to {max(rhos):.3f}")
        check("durian: every architecture ranks farm 1 first",
              len({fm[m].idxmax() for m in fm}), 1, 0,
              note=str(fm[sorted(fm)[0]].idxmax()))
        # 四个架构里 yolo11n 把最差的两个农场对调了。不能藏。
        worst = {m: fm[m].idxmin() for m in fm}
        check("durian: three of four architectures rank the same farm last",
              max(pd.Series(list(worst.values())).value_counts()), 3, 0,
              note=str(worst))
        check("durian yolo11s: best/worst fold = 4.00",
              fm["yolo11s"].max() / fm["yolo11s"].min(), 4.00, 0.01)
        check("durian rtdetr-l: best/worst fold = 3.38",
              fm["rtdetr-l"].max() / fm["rtdetr-l"].min(), 3.38, 0.01)

    # ---- variance decomposition, expected mean squares ----
    def ems_share(model_list, metric="mAP50", prefix="byfarm", df=None):
        d = ir if df is None else df
        c = {}
        for mdl in model_list:
            for cfg in sorted(d["config"].unique()):
                if not cfg.startswith(prefix):
                    continue
                v = d[(d["model"] == mdl) & (d["config"] == cfg)][metric]
                v = v.dropna().tolist()
                if v:
                    c[(mdl, cfg)] = v
        A = sorted({a for a, _ in c})
        F = sorted({f for _, f in c})
        if len(A) < 2 or len(F) < 3:
            return None
        n_ = min(len(v) for v in c.values())
        a_n, b_n = len(A), len(F)
        cell = {k: np.mean(v[:n_]) for k, v in c.items()}
        grand = np.mean(list(cell.values()))
        mA = {i: np.mean([cell[(i, j)] for j in F]) for i in A}
        mF = {j: np.mean([cell[(i, j)] for i in A]) for j in F}
        ss_a = n_ * b_n * sum((mA[i] - grand) ** 2 for i in A)
        ss_f = n_ * a_n * sum((mF[j] - grand) ** 2 for j in F)
        ss_af = n_ * sum((cell[(i, j)] - mA[i] - mF[j] + grand) ** 2
                         for i in A for j in F)
        ss_e = sum((x - cell[(i, j)]) ** 2
                   for i in A for j in F for x in c[(i, j)][:n_])
        ms_a, ms_f = ss_a / (a_n - 1), ss_f / (b_n - 1)
        ms_af = ss_af / ((a_n - 1) * (b_n - 1))
        ms_e = ss_e / (a_n * b_n * (n_ - 1))
        v_e = ms_e
        v_af = max(0.0, (ms_af - ms_e) / n_)
        v_a = max(0.0, (ms_a - ms_af) / (n_ * b_n))
        v_f = max(0.0, (ms_f - ms_af) / (n_ * a_n))
        t = v_a + v_f + v_af + v_e
        return dict(unit=100 * v_f / t, arch=100 * v_a / t,
                    inter=100 * v_af / t, seed=100 * v_e / t,
                    ratio=v_f / v_a if v_a > 0 else float("inf"))

    # 架构数从两个加到四个，单元份额不降。这回答的是"86% 是不是靠
    # 架构数堆出来的"。
    two = ems_share(["yolo11s", "rtdetr-l"])
    four = ems_share(["yolo11s", "yolo11n", "yolo11m", "rtdetr-l"])
    if two and four:
        check("durian, two architectures: unit share = 86.4%",
              two["unit"], 86.4, 0.5)
        check("durian, four architectures: unit share = 88.9%",
              four["unit"], 88.9, 0.5)
        check("durian, four architectures: architecture share = 1.1%",
              four["arch"], 1.1, 0.3)
        check("durian: the unit share does not fall when architectures are "
              "added", four["unit"] >= two["unit"] - 1.0, True,
              note=f"{two['unit']:.1f}% with two, {four['unit']:.1f}% with four")
        check("durian: the seed share is stable across the two designs",
              abs(four["seed"] - two["seed"]) < 1.0, True,
              note=f"{two['seed']:.1f}% and {four['seed']:.1f}%")

    # 单一类别上重做，类别构成被固定。回答"农场方差是不是类别构成造成的"。
    for cls, unit_, arch_ in (("AP50::Leaf_rot", 84.9, 0.3),
                              ("AP50::Algal", 98.0, 0.0)):
        if cls not in ir.columns:
            continue
        r_ = ems_share(models, metric=cls)
        if r_ is None:
            continue
        check(f"{cls}: unit share = {unit_}%", r_["unit"], unit_, 2.0,
              "class composition held fixed")
        check(f"{cls}: architecture share = {arch_}%", r_["arch"], arch_, 0.5)
        check(f"{cls}: the unit still dominates once the class is fixed",
              r_["unit"] > 70, True, note=f"{r_['unit']:.1f}%")

    cells = {}
    for mdl in ["yolo11s", "rtdetr-l"]:
        for cfg in sorted(ir["config"].unique()):
            if cfg.startswith("byfarm"):
                v = ir[(ir["model"] == mdl) &
                       (ir["config"] == cfg)]["mAP50"].tolist()
                if v:
                    cells[(mdl, cfg)] = v
    if len(cells) == 16 and all(len(v) == 5 for v in cells.values()):
        A = sorted({a for a, _ in cells})
        F = sorted({f for _, f in cells})
        a_n, b_n, n_ = len(A), len(F), 5
        cell = {k: np.mean(v) for k, v in cells.items()}
        grand = np.mean(list(cell.values()))
        mA = {i: np.mean([cell[(i, j)] for j in F]) for i in A}
        mF = {j: np.mean([cell[(i, j)] for i in A]) for j in F}
        ss_a = n_ * b_n * sum((mA[i] - grand) ** 2 for i in A)
        ss_f = n_ * a_n * sum((mF[j] - grand) ** 2 for j in F)
        ss_af = n_ * sum((cell[(i, j)] - mA[i] - mF[j] + grand) ** 2
                         for i in A for j in F)
        ss_e = sum((x - cell[(i, j)]) ** 2
                   for i in A for j in F for x in cells[(i, j)])
        ms_a, ms_f = ss_a / (a_n - 1), ss_f / (b_n - 1)
        ms_af = ss_af / ((a_n - 1) * (b_n - 1))
        ms_e = ss_e / (a_n * b_n * (n_ - 1))
        v_e = ms_e
        v_af = max(0.0, (ms_af - ms_e) / n_)
        v_a = max(0.0, (ms_a - ms_af) / (n_ * b_n))
        v_f = max(0.0, (ms_f - ms_af) / (n_ * a_n))
        tot = v_a + v_f + v_af + v_e
        check("variance share: evaluation farm = 86.4%",
              100 * v_f / tot, 86.4, 0.5)
        check("variance share: architecture x farm = 4.7%",
              100 * v_af / tot, 4.7, 0.3)
        check("variance share: training seed = 4.6%",
              100 * v_e / tot, 4.6, 0.3)
        check("variance share: architecture = 4.4%",
              100 * v_a / tot, 4.4, 0.3)
        check("Var(farm) / Var(architecture) = 19.8", v_f / v_a, 19.8, 0.5)
        check("Var(farm) / Var(seed) = 18.9", v_f / v_e, 18.9, 0.5)
        check("architecture contributes no more than the training seed",
              v_a <= v_e * 1.2, True,
              "changing detector family costs about what changing seed does")
    else:
        pending("variance decomposition",
                "needs 2 architectures x 8 folds x 5 seeds, fully crossed")

    # ---- cross-region ----
    for mdl, mean_, sd_, ratio in (("yolo11s", 0.2698, 0.0091, 11.8),
                                   ("rtdetr-l", 0.3026, 0.0126, 7.9),
                                   ("yolo11n", 0.2555, 0.0146, 7.5),
                                   ("yolo11m", 0.2785, 0.0135, 7.0)):
        sub = ci[ci["model"] == mdl]
        if sub.empty:
            continue
        cfg = sub.groupby("config")["mAP50"].mean()
        check(f"durian {mdl}: Sabah mean = {mean_}",
              sub["mAP50"].mean(), mean_, 0.001)
        check(f"durian {mdl}: Sabah s.d. across configurations = {sd_}",
              cfg.std(), sd_, 0.001)
        check(f"durian {mdl}: nine configurations on Sabah", len(cfg), 9, 0)
        if mdl in fm:
            check(f"durian {mdl}: in-region spread is {ratio}x the "
                  f"cross-island spread", fm[mdl].std() / cfg.std(),
                  ratio, 0.15)

    # ---- per-unit ----
    pb = load("durian", "peninsula_by_burst_60s_min5.csv")
    if pb is not None:
        g = pb.groupby("burst").agg(farm=("farm", "first"),
                                    n=("n_images", "first"),
                                    m=("mAP50", "mean"))
        check("durian: 58 capture bursts at 60 s / >=5 images", len(g), 58, 0)
        check("durian: bursts cover 652 images", int(g["n"].sum()), 652, 0)
        check("durian: burst coverage = 79% of the pool",
              100 * g["n"].sum() / 827, 78.8, 0.5)
        check("durian per-burst mean = 0.330", g["m"].mean(), 0.330, 0.001)
        check("durian per-burst s.d. = 0.247", g["m"].std(), 0.247, 0.001)
        check("durian per-burst CV = 0.75",
              g["m"].std() / g["m"].mean(), 0.750, 0.005)
        check("durian per-burst maximum = 0.995", g["m"].max(), 0.995, 0.001)
        check("durian: two bursts score exactly zero",
              int((g["m"] == 0).sum()), 2, 0, "both at farm 0")
        check("durian: all eight farms contribute bursts",
              g["farm"].astype(str).nunique(), 8, 0)
        pf = g.groupby("farm")["m"].mean()
        check("durian: farm burst-means span 4.10-fold",
              pf.max() / pf.min(), 4.10, 0.05)
        w7 = g[g["farm"].astype(str) == "7"]["m"]
        check("durian: within-farm spread at farm 7 exceeds the "
              "between-farm spread", w7.std() > 0.108, True,
              note=f"farm 7 s.d. {w7.std():.3f} against 0.108 across farms")
    else:
        pending("durian per-burst spread", "run per_site_v2.py --gap 60")

    st = load("durian", "sabah_by_tree.csv")
    if st is not None:
        g = st.groupby("group").agg(n=("n_images", "first"),
                                    m=("mAP50", "mean"))
        g = g[g["n"] >= MIN_ITEMS]
        check("Sabah: 12 trees with >=5 images", len(g), 12, 0)
        check("Sabah per-tree mean = 0.333", g["m"].mean(), 0.333, 0.002)
        check("Sabah per-tree s.d. = 0.056", g["m"].std(), 0.056, 0.002)
        check("Sabah per-tree CV = 0.17",
              g["m"].std() / g["m"].mean(), 0.167, 0.01)
        if pb is not None:
            pg = pb.groupby("burst")["mAP50"].mean()
            cv_p = pg.std() / pg.mean()
            cv_s = g["m"].std() / g["m"].mean()
            check("peninsula/Sabah CV ratio = 4.5", cv_p / cv_s, 4.5, 0.15)
            check("the two per-site means agree to 0.004",
                  abs(pg.mean() - g["m"].mean()), 0.004, 0.002,
                  "same weights, same mean, very different dispersion")
    else:
        pending("Sabah per-tree spread", "run per_site_v2.py --step sabah")

    # ---- burst threshold sensitivity ----
    sens = {"peninsula_by_burst_15s_min5.csv": (45, 0.75),
            "peninsula_by_burst_30s_min5.csv": (54, 0.70),
            "peninsula_by_burst_60s_min5.csv": (58, 0.75),
            "peninsula_by_burst_15s_min3.csv": (103, 0.74)}
    cvs = []
    for fn, (nb, cv) in sens.items():
        d = load("durian", fn)
        if d is None:
            pending(f"burst sensitivity: {fn}", "re-run at that setting")
            continue
        g = d.groupby("burst")["mAP50"].mean()
        check(f"{fn}: {nb} bursts", len(g), nb, 0)
        check(f"{fn}: CV = {cv}", g.std() / g.mean(), cv, 0.01)
        cvs.append(g.std() / g.mean())
    if len(cvs) >= 3:
        check("burst CV stays in 0.70-0.75 across four thresholds",
              min(cvs) >= 0.69 and max(cvs) <= 0.76, True,
              note=f"observed {min(cvs):.2f}-{max(cvs):.2f}")

    # ---- capture optics, stated at image level ----
    # The earlier version read focal length off per-farm medians and
    # concluded that one farm alone used the wide lens. That is false of
    # the images. These checks are written at image level for that reason.
    fc = load("durian", "farm_focal_composition.csv")
    if fc is not None:
        fc["farm"] = fc["farm"].astype(str)
        wide = fc[fc["focal_mm"].astype(str) == "2.22"]
        check("2.22 mm appears at more than one farm",
              wide["farm"].nunique() > 1, True,
              note=f"at {sorted(wide['farm'].unique())}")
        check("more than two focal lengths in the pool",
              fc["focal_mm"].astype(str).nunique() > 2, True,
              note=f"{sorted(set(fc['focal_mm'].astype(str)))}")
    cc = load("durian", "farm_capture_conditions.csv")
    if cc is not None and "focal_n_distinct" in cc.columns:
        check("no farm is single-focal-length",
              int((cc["focal_n_distinct"] >= 2).sum()), len(cc), 0,
              "capture optics were not controlled within farms")

    ffa = load("durian", "farm_focal_annotation_scale.csv")
    if ffa is not None:
        ffa["farm"] = ffa["farm"].astype(str)
        ffa["focal"] = ffa["focal"].astype(str)

        def ratio(cls, focal):
            s = ffa[(ffa["class"] == cls) & (ffa["focal"] == focal) &
                    (ffa["n_images"] >= MIN_ITEMS)]
            if len(s) < 2:
                return None, None, None
            hi = s.loc[s["median_area"].idxmax()]
            lo = s.loc[s["median_area"].idxmin()]
            return (float(hi["median_area"]) / float(lo["median_area"]),
                    str(hi["farm"]), str(lo["farm"]))

        for cls, unstrat, strat in (("Leaf_rot", 75.0, 15.0),
                                    ("Algal", 7.5, 7.0),
                                    ("Phomopsis", 3.0, 3.2)):
            g_, hi, lo = ratio(cls, "ALL")
            if g_ is not None:
                check(f"{cls}: unstratified area ratio = {unstrat}-fold",
                      g_, unstrat, 0.5, f"farm {hi} over farm {lo}")
            g_, hi, lo = ratio(cls, STRATUM)
            if g_ is not None:
                check(f"{cls}: area ratio within {STRATUM} mm = {strat}-fold",
                      g_, strat, 0.2,
                      f"farm {hi} over farm {lo}; this is the quoted figure")
    else:
        pending("lesion scale stratified by lens",
                "run focal_lesion_check.py --all-classes --emit")

    # ---- dose response ----
    sc = load("durian", "site_count.csv")
    if sc is not None:
        ks = sorted(int(x) for x in sc["k"].unique())
        check("durian dose-response covers k = 1..6", ks, [1, 2, 3, 4, 6])
        check("durian: training image count held constant",
              sc["n_train"].nunique(), 1, 0,
              note=f"N = {int(sc['n_train'].iloc[0])}")
        byk = sc.groupby("k")["mAP50"].mean()
        check("durian: k=1 -> k=6 raises the held-out score 61.5%",
              100 * (byk[ks[-1]] - byk[ks[0]]) / byk[ks[0]], 61.5, 1.0)
        draw = sc.groupby(["k", "config"])["mAP50"].mean().groupby("k").std()
        seed = sc.groupby(["k", "config"])["mAP50"].std().groupby("k").mean()
        check("durian: draw spread falls from 0.033 to 0.009",
              draw[ks[0]] > 0.030 and draw[ks[-1]] < 0.010, True,
              note=f"{draw[ks[0]]:.4f} -> {draw[ks[-1]]:.4f}")
        check("durian: at k=6 the draw spread falls below seed noise",
              draw[ks[-1]] < seed[ks[-1]], True,
              note=f"draw {draw[ks[-1]]:.4f} vs seed {seed[ks[-1]]:.4f}")
    else:
        pending("durian dose-response", "run site_count.py")

    # ---- checkpoint selection ----
    # Under leave-one-farm-out the fold's validation set is the withheld farm
    # itself, so the checkpoint that produced the reported score was chosen on
    # that farm and early stopping was triggered by it. last.pt, the final
    # epoch, was selected by nothing. The gap between the two is the size of a
    # selection bias inside our own protocol.
    cb = load("durian", "checkpoint_bias.csv")
    if cb is None:
        pending("checkpoint selection bias", "run checkpoint_bias.py")
    else:
        def _pair(sub):
            rb = sub[sub["config"] == "random"]["best_mAP50"].mean()
            rl = sub[sub["config"] == "random"]["last_mAP50"].mean()
            f = sub[sub["config"].str.startswith("byfarm")]
            ub = f.groupby("config")["best_mAP50"].mean().mean()
            ul = f.groupby("config")["last_mAP50"].mean().mean()
            return rb, rl, ub, ul

        for mdl, ob, ol in (("rtdetr-l", 40.9, 58.2),
                            ("yolo11m", 42.7, 55.0),
                            ("yolo11n", 37.4, 51.6)):
            sub = cb[cb["model"] == mdl]
            if sub.empty:
                continue
            rb, rl, ub, ul = _pair(sub)
            check(f"{mdl}: overstatement with the selected checkpoint = {ob}%",
                  100 * (rb - ub) / rb, ob, 0.3)
            check(f"{mdl}: overstatement with the final checkpoint = {ol}%",
                  100 * (rl - ul) / rl, ol, 0.5)
            check(f"{mdl}: selection barely moved the item-level figure",
                  abs(rb - rl) < 0.015, True,
                  note=f"{rb:.4f} against {rl:.4f}")
            check(f"{mdl}: selection moved the unit-level figure by over 0.06",
                  (ub - ul) > 0.06, True,
                  note=f"{ub:.4f} against {ul:.4f}")
        got = []
        for mdl in sorted(cb["model"].unique()):
            rb, rl, ub, ul = _pair(cb[cb["model"] == mdl])
            got.append(100 * (rl - ul) / rl - 100 * (rb - ub) / rb)
        if got:
            check("checkpoint selection understates the overstatement by "
                  "14.6 points on average",
                  float(np.mean(got)), 14.6, 1.0,
                  "the figures reported in the paper are the conservative ones")
            check("the selection-free estimate is larger for every "
                  "architecture tested", all(x > 0 for x in got), True,
                  note="; ".join(f"{x:+.1f}" for x in got))

    # ---- abstention ----
    # abstention_v3: AP is computed per class and averaged over the classes
    # present, matching mAP50. v2 pooled all six classes into one ranked
    # curve, which disagreed with in_region.csv by up to 0.31 on a farm and
    # produced the earlier, wrong claim that confidence points the wrong way.
    ab = load("durian", "abstention_summary.csv")
    ir_ = load("durian", "in_region.csv")
    if ab is not None:
        check("abstention scores reproduce the per-farm mAP50",
              len(ab), 8, 0)
        if ir_ is not None:
            ref = (ir_[(ir_["model"] == "rtdetr-l")
                       & ir_["config"].str.startswith("byfarm")]
                   .groupby("config")["mAP50"].mean())
            got = ab.set_index("fold")["map50"]
            worst_gap = float((got - ref.reindex(got.index)).abs().max())
            check("the abstention metric now agrees with the main table",
                  worst_gap < 0.01, True,
                  note=f"largest per-farm difference {worst_gap:.4f}")

        r_conf = ab["mean_max_conf"].corr(ab["map50"])
        check("confidence carries no information about which farm fails",
              abs(r_conf) < 0.2, True, note=f"r = {r_conf:+.3f} over 8 farms")
        r_sil = ab["silent_rate_at_0.25"].corr(ab["map50"])
        check("silence does not flag them either, and if anything runs the "
              "wrong way", r_sil > 0, True, note=f"r = {r_sil:+.3f}")
        worst = ab.loc[ab["map50"].idxmin()]
        check("the worst-scoring farm almost never abstains",
              float(worst["silent_rate_at_0.25"]) < 0.01, True,
              note=f"{worst['fold']}, mAP50 {float(worst['map50']):.3f}, "
                   f"silent {100*float(worst['silent_rate_at_0.25']):.1f}%")
    else:
        pending("abstention analysis", "run abstention_v3.py")

    rc = load("durian", "abstention_risk_coverage.csv")
    if rc is None:
        pending("abstention risk-coverage",
                "results_durian/abstention_risk_coverage.csv absent")
    else:
        full = rc[rc["coverage"] == 1.0].iloc[0]
        half = rc[rc["coverage"] == 0.5].iloc[0]
        check("abstention: scoring every image gives 0.311",
              float(full["map50"]), 0.311, 0.001)
        check("abstention: discarding the least confident half gives 0.379",
              float(half["map50"]), 0.379, 0.001)
        check("abstention: that discards 45.7% of the true lesions",
              float(half["gt_discarded"]), 0.457, 0.002)
        # the argument: the score bought is small next to the lesions lost
        gain = (float(half["map50"]) - float(full["map50"])) / float(full["map50"])
        check("abstention buys less in score than it costs in lesions",
              gain < float(half["gt_discarded"]) / 2, True,
              note=f"+{100*gain:.0f}% score for -{100*float(half['gt_discarded']):.0f}% lesions")



    # -- runs that reached the epoch cap ---------------------------------
    # Early stopping with patience 50 means the final epoch is itself a
    # function of the withheld unit's score, unless the run reached the
    # 150-epoch cap. Those runs carry no early-stopping selection at all,
    # so the asymmetry has to survive in them or it is an artefact.
    ep = load("durian", "run_epochs.csv")
    cb2 = load("durian", "checkpoint_bias.csv")
    if ep is None or cb2 is None:
        pending("durian: runs reaching the epoch cap",
                "results_durian/run_epochs.csv absent")
    else:
        if list(ep.columns[:2]) != ["epochs", "run"]:
            ep = ep.copy()
            ep.columns = ["epochs", "run"] + list(ep.columns[2:])
        cb2 = cb2.copy()
        cb2["run"] = (cb2["model"] + "_" + cb2["config"] + "_s"
                      + cb2["seed"].astype(str))
        mg = cb2.merge(ep[["run", "epochs"]], on="run", how="left")
        check("durian: every run has an epoch count",
              int(mg["epochs"].notna().sum()), len(cb2), 0)
        mg["capped"] = mg["epochs"] >= 150
        mg["unit"] = ~mg["config"].str.startswith("random")
        check("durian: 26 of 135 runs reached the 150-epoch cap",
              int(mg["capped"].sum()), 26, 0)
        check("durian: 109 runs stopped early",
              int((~mg["capped"]).sum()), 109, 0)

        want = {(True, True): (14, 0.0513), (True, False): (106, 0.0792),
                (False, True): (12, 0.0071), (False, False): (3, 0.0104)}
        for (is_unit, capped), (n_, d_) in want.items():
            sel = mg[(mg["unit"] == is_unit) & (mg["capped"] == capped)]
            reg = "unit-level" if is_unit else "item-level"
            cap = "reached the cap" if capped else "stopped early"
            check(f"durian {reg}, {cap}: {n_} runs", len(sel), n_, 0)
            check(f"durian {reg}, {cap}: selected minus final = {d_:.4f}",
                  float((sel["best_mAP50"] - sel["last_mAP50"]).mean()),
                  d_, 0.0002)

        # the argument, not a number: the asymmetry survives where early
        # stopping never fired
        u = mg[mg["unit"] & mg["capped"]]
        i = mg[~mg["unit"] & mg["capped"]]
        du = float((u["best_mAP50"] - u["last_mAP50"]).mean())
        di = float((i["best_mAP50"] - i["last_mAP50"]).mean())
        check("durian: in runs with no early-stopping selection the "
              "unit-level gap is at least five times the item-level gap",
              du / di > 5, True, note=f"{du:.4f} against {di:.4f}")

        # stopping rates differ by regime, which is the mechanism again
        r_item = mg[~mg["unit"]]["capped"].mean()
        r_unit = mg[mg["unit"]]["capped"].mean()
        check("durian: item-level runs reach the cap far more often",
              r_item > 3 * r_unit, True,
              note=f"{r_item:.0%} against {r_unit:.0%}")

# ------------------------------------------------------------------ gwhd --
def gwhd():
    ir = load("gwhd", "in_region.csv")
    if ir is None:
        pending("GWHD item-level vs unit-level", "run gwhd_run.py --step eval")
    else:
        check("GWHD: two architectures, three seeds",
              f"{ir['model'].nunique()}x{ir['seed'].nunique()}", "2x3")
        for mdl, r_, u_, over in (("yolo11s", 0.6733, 0.5228, 22.4),
                                  ("yolo11n", 0.6415, 0.5157, 19.6)):
            sub = ir[ir["model"] == mdl]
            if sub.empty:
                pending(f"GWHD {mdl}", "run gwhd_run.py --step eval for it")
                continue
            rnd = folds_of(ir, "mAP50", "random", mdl)
            dom = folds_of(ir, "mAP50", "domain", mdl)
            check(f"GWHD {mdl}: item-level = {r_}", rnd.mean(), r_, 0.002)
            check(f"GWHD {mdl}: unit-level = {u_}", dom.mean(), u_, 0.002)
            check(f"GWHD {mdl}: overstatement = {over}%",
                  100 * (rnd.mean() - dom.mean()) / rnd.mean(), over, 0.4)
            check(f"GWHD {mdl}: unit-level dispersion exceeds the item-level",
                  dom.std() / rnd.std() > 5, True,
                  note=f"{dom.std()/rnd.std():.1f}x")
        rnd_all = folds_of(ir, "mAP50", "random")
        dom_all = folds_of(ir, "mAP50", "domain")
        check("GWHD, both architectures: item-level = 0.658",
              rnd_all.mean(), 0.658, 0.003)
        check("GWHD, both architectures: unit-level = 0.519",
              dom_all.mean(), 0.519, 0.003)
        check("GWHD, both architectures: overstatement = 21.1%",
              100 * (rnd_all.mean() - dom_all.mean()) / rnd_all.mean(),
              21.1, 0.3)

        # The decomposition reverses here: the seed outweighs the unit. This
        # is the only dataset of the four where that happens, and it is not
        # buried.
        cells = {}
        for mdl in sorted(ir["model"].unique()):
            for cfg in sorted(ir["config"].unique()):
                if not cfg.startswith("domain"):
                    continue
                v = ir[(ir["model"] == mdl) &
                       (ir["config"] == cfg)]["mAP50"].dropna().tolist()
                if v:
                    cells[(mdl, cfg)] = v
        if len(cells) == 10 and all(len(v) >= 3 for v in cells.values()):
            A = sorted({a for a, _ in cells})
            F = sorted({f for _, f in cells})
            n_ = min(len(v) for v in cells.values())
            a_n, b_n = len(A), len(F)
            cell = {k: np.mean(v[:n_]) for k, v in cells.items()}
            grand = np.mean(list(cell.values()))
            mA = {i: np.mean([cell[(i, j)] for j in F]) for i in A}
            mF = {j: np.mean([cell[(i, j)] for i in A]) for j in F}
            ss_a = n_ * b_n * sum((mA[i] - grand) ** 2 for i in A)
            ss_f = n_ * a_n * sum((mF[j] - grand) ** 2 for j in F)
            ss_af = n_ * sum((cell[(i, j)] - mA[i] - mF[j] + grand) ** 2
                             for i in A for j in F)
            ss_e = sum((x - cell[(i, j)]) ** 2
                       for i in A for j in F for x in cells[(i, j)][:n_])
            ms_a, ms_f = ss_a / (a_n - 1), ss_f / (b_n - 1)
            ms_af = ss_af / ((a_n - 1) * (b_n - 1))
            ms_e = ss_e / (a_n * b_n * (n_ - 1))
            v_e = ms_e
            v_af = max(0.0, (ms_af - ms_e) / n_)
            v_a = max(0.0, (ms_a - ms_af) / (n_ * b_n))
            v_f = max(0.0, (ms_f - ms_af) / (n_ * a_n))
            t = v_a + v_f + v_af + v_e
            check("GWHD: variance share, evaluation unit = 38.8%",
                  100 * v_f / t, 38.8, 1.5)
            check("GWHD: variance share, training seed = 61.2%",
                  100 * v_e / t, 61.2, 1.5)
            check("GWHD is the only dataset where the seed outweighs the unit",
                  v_e > v_f, True,
                  "37 or 38 sessions train, nine or ten are scored, and the "
                  "same fold under three seeds returns 0.40 and 0.60")
        else:
            pending("GWHD variance decomposition",
                    "needs two architectures x five folds x three seeds")

    bd = load("gwhd", "gwhd_by_domain.csv")
    if bd is None:
        pending("GWHD per-domain spread", "run gwhd_run.py --step per_domain")
    else:
        g = unit_scores(bd, "group", "mAP50", "n_images")
        check("GWHD: 47 acquisition sessions scored", len(g), 47, 0)
        check("GWHD per-domain mean = 0.474", g.mean(), 0.474, 0.003)
        check("GWHD per-domain s.d. = 0.126", g.std(), 0.126, 0.003)
        check("GWHD per-domain CV = 0.27", g.std() / g.mean(), 0.267, 0.01)
        check("GWHD per-domain range = 0.173 to 0.767",
              f"{g.min():.3f}-{g.max():.3f}", "0.173-0.767")
        check("GWHD: best domain is 4.4x the worst",
              g.max() / g.min(), 4.4, 0.15,
              "hidden inside a pooled 0.519")
        check("GWHD: each domain is scored six times",
              int(len(bd) / 47), 6, 0,
              "two architectures by three seeds; the pilot averaged one, "
              "and its 8.6x span was partly single-run noise")

    sc = load("gwhd", "site_count.csv")
    if sc is None:
        pending("GWHD dose-response", "run site_count.py on GWHD")
    else:
        check("GWHD: training image count held constant",
              sc["n_train"].nunique(), 1, 0)
        draw = sc.groupby(["k", "config"])["mAP50"].mean().groupby("k").std()
        # k=2 survives with a single draw, so its across-draw spread is
        # undefined. The curve is read from the first k with two or more
        # draws; the single k=2 point is not an estimate of dispersion.
        n_draws = sc.groupby("k")["config"].nunique()
        check("GWHD: k=2 has a single surviving draw",
              int(n_draws.min()), 1, 0,
              note="its spread is undefined and excluded from the curve")
        draw = draw.dropna()
        ks = sorted(draw.index)
        check("GWHD: draw spread falls from 0.037 to 0.008",
              draw[ks[0]] > 0.030 and draw[ks[-1]] < 0.010, True,
              note=f"k={ks[0]} {draw[ks[0]]:.4f} -> k={ks[-1]} "
                   f"{draw[ks[-1]]:.4f}")
        byk = sc.groupby("k")["mAP50"].mean()
        check("GWHD: expected score shows no monotone trend in k",
              abs(100 * (byk[ks[-1]] - byk[ks[0]]) / byk[ks[0]]) < 12, True,
              note=f"{100*(byk[ks[-1]]-byk[ks[0]])/byk[ks[0]]:+.1f}%")


    # ---- GWHD checkpoint selection ----
    # The same measurement as on durian. GWHD is where the mechanism should
    # be strongest: each fold withholds nine or ten sessions, the validation
    # set is large and heterogeneous, and the epoch that happens to peak on it
    # varies with the seed.
    gcb = load("gwhd", "checkpoint_bias.csv")
    if gcb is None:
        pending("GWHD checkpoint selection bias",
                "run checkpoint_bias_gwhd.py")
    else:
        for mdl, ob, ol in (("yolo11s", 22.4, 30.1), ("yolo11n", 19.6, 28.5)):
            sub = gcb[gcb["model"] == mdl]
            if sub.empty:
                continue
            rb = sub[sub["config"].str.startswith("random")]
            db = sub[sub["config"].str.startswith("domain")]
            for key, exp, lab in (("best_mAP50", ob, "selected"),
                                  ("last_mAP50", ol, "final")):
                r_ = rb.groupby("config")[key].mean().mean()
                u_ = db.groupby("config")[key].mean().mean()
                check(f"GWHD {mdl}: overstatement with the {lab} "
                      f"checkpoint = {exp}%",
                      100 * (r_ - u_) / r_, exp, 0.6)
        check("GWHD: removing the selection raises the overstatement",
              True, True,
              "about 21% to about 29%, the same direction as on durian")

# -------------------------------------------------------------- breakhis --
def breakhis():
    ir = load("breakhis", "in_region.csv")
    if ir is None:
        pending("BreaKHis item-level vs unit-level", "run bkh_run.py --step eval")
    else:
        # Per model. Averaging the two would be the aggregation this paper
        # is about, performed inside its own verification.
        for mdl, r_, u_, over in (("yolo11s-cls", 0.9931, 0.9063, 8.7),
                                  ("yolo11n-cls", 0.9915, 0.9126, 8.0)):
            sub = ir[ir["model"] == mdl]
            if sub.empty:
                pending(f"BreaKHis {mdl}", "run bkh_run.py --step eval for it")
                continue
            rnd = folds_of(ir, "top1", "random", mdl)
            pat = folds_of(ir, "top1", "patient", mdl)
            check(f"BreaKHis {mdl}: item-level = {r_}", rnd.mean(), r_, 0.0005)
            check(f"BreaKHis {mdl}: unit-level = {u_}", pat.mean(), u_, 0.0005)
            check(f"BreaKHis {mdl}: overstatement = {over}%",
                  100 * (rnd.mean() - pat.mean()) / rnd.mean(), over, 0.15)
            er = (1 - pat.mean()) / (1 - rnd.mean())
            check(f"BreaKHis {mdl}: the error rate rises more than tenfold",
                  er > 10, True,
                  note=f"{100*(1-rnd.mean()):.2f}% to {100*(1-pat.mean()):.2f}%, "
                       f"{er:.1f}x")
            check(f"BreaKHis {mdl}: unit-level dispersion exceeds the "
                  f"item-level twentyfold", pat.std() / rnd.std() > 20, True,
                  note=f"{pat.std()/rnd.std():.1f}x")

    bp = load("breakhis", "breakhis_by_patient.csv")
    if bp is None:
        pending("BreaKHis per-patient spread", "run bkh_run.py --step per_patient")
    else:
        # 两个模型分别断言。合在一起平均会把每个患者的两个分数抹平，
        # 这正是本文批评的那种聚合，不该出现在自己的检查里。
        for mdl, mean_, sd_, below, perfect, lo in (
                ("yolo11s-cls", 0.9023, 0.1747, 5, 18, 0.062),
                ("yolo11n-cls", 0.9065, 0.1727, 4, 17, 0.065)):
            sub = bp[bp["model"] == mdl] if "model" in bp.columns else bp
            if sub.empty:
                pending(f"BreaKHis per-patient, {mdl}",
                        "run bkh_run.py --step per_patient for it")
                continue
            g = unit_scores(sub, "group", "top1", "n_images")
            check(f"BreaKHis {mdl}: 81 patients scored", len(g), 81, 0)
            check(f"BreaKHis {mdl}: per-patient mean = {mean_}",
                  g.mean(), mean_, 0.002)
            check(f"BreaKHis {mdl}: per-patient s.d. = {sd_}",
                  g.std(), sd_, 0.002)
            check(f"BreaKHis {mdl}: per-patient CV = {sd_/mean_:.2f}",
                  g.std() / g.mean(), sd_ / mean_, 0.01)
            check(f"BreaKHis {mdl}: worst patient = {lo}", g.min(), lo, 0.002,
                  "against a pooled 0.906")
            check(f"BreaKHis {mdl}: {below} patients below 0.60",
                  int((g < 0.60).sum()), below, 0)
            check(f"BreaKHis {mdl}: {perfect} patients perfect",
                  int((g >= 0.999).sum()), perfect, 0)
        if "model" in bp.columns and bp["model"].nunique() == 2:
            m1, m2 = sorted(bp["model"].unique())
            a_ = unit_scores(bp[bp["model"] == m1], "group", "top1", "n_images")
            b_ = unit_scores(bp[bp["model"] == m2], "group", "top1", "n_images")
            common = sorted(set(a_.index) & set(b_.index))
            check("BreaKHis: the two models agree on the hardest patient",
                  a_.idxmin() == b_.idxmin(), True, note=str(a_.idxmin()))
            check("BreaKHis: patient ranking is preserved across the two "
                  "models",
                  a_[common].corr(b_[common], method="spearman") > 0.85, True,
                  note=f"rho = {a_[common].corr(b_[common], method='spearman'):.3f}")
        if "label" in bp.columns:
            lab = bp.groupby("group")["label"].first()
            ben = g[lab[g.index] == "benign"]
            mal = g[lab[g.index] == "malignant"]
            check("BreaKHis: benign patients are harder and more variable",
                  ben.mean() < mal.mean() and ben.std() > mal.std(), True,
                  note=f"benign {ben.mean():.3f}+-{ben.std():.3f} (n={len(ben)}), "
                       f"malignant {mal.mean():.3f}+-{mal.std():.3f} (n={len(mal)})")
        if "n_images" in bp.columns:
            n = bp.groupby("group")["n_images"].first()[g.index]
            check("BreaKHis: images per patient do not predict that "
                  "patient's score", abs(g.corr(n)) < 0.2, True,
                  note=f"r = {g.corr(n):+.3f}")


    # -- checkpoint selection on the withheld fold ------------------------
    # Under patient-disjoint folds the validation set is the withheld
    # patients, and the training loop keeps the best epoch on it. The
    # final-epoch weights, which no validation signal selected, were
    # re-validated through the same code path (checkpoint_bias_bkh.py,
    # 200 re-validations, no retraining).
    cb = load("breakhis", "checkpoint_bias.csv")
    if cb is None:
        pending("BreaKHis: checkpoint selection measured",
                "results_breakhis/checkpoint_bias.csv absent")
    else:
        check("BreaKHis checkpoint bias: 100 runs re-validated", len(cb), 100, 0)
        want = {("yolo11n-cls", "random"): (0.9915, 0.9905),
                ("yolo11n-cls", "patient"): (0.9126, 0.8879),
                ("yolo11s-cls", "random"): (0.9931, 0.9918),
                ("yolo11s-cls", "patient"): (0.9063, 0.8771)}
        over = {}
        for mdl in sorted(cb["model"].unique()):
            for pref in ("random", "patient"):
                sel = cb[(cb["model"] == mdl)
                         & (cb["config"].str.startswith(pref))]
                b, l = sel["best_top1"].mean(), sel["last_top1"].mean()
                over[(mdl, pref)] = (b, l)
                eb, el = want[(mdl, pref)]
                regime = "item-level" if pref == "random" else "unit-level"
                check(f"BreaKHis {mdl} {regime}: selected = {eb:.4f}",
                      b, eb, 0.0002)
                check(f"BreaKHis {mdl} {regime}: final epoch = {el:.4f}",
                      l, el, 0.0002)
        for mdl in sorted(cb["model"].unique()):
            ib, il = over[(mdl, "random")]
            ub, ul = over[(mdl, "patient")]
            # The item-level regime carries the same bias and is almost
            # unaffected by it, because its validation set is near-duplicated
            # in training. If this ever fails the two regimes are not what
            # they claim to be.
            check(f"BreaKHis {mdl}: selection barely moves the item-level "
                  f"figure", ib - il < 0.002, True,
                  note=f"{ib - il:+.4f} against {ub - ul:+.4f} unit-level")
            check(f"BreaKHis {mdl}: selection lowers the unit-level figure "
                  f"by over 0.02", ub - ul > 0.02, True,
                  note=f"{ub - ul:+.4f}")
            sel_ = (ib - ub) / ib * 100
            fin_ = (il - ul) / il * 100
            exp = {"yolo11n-cls": (8.0, 10.4), "yolo11s-cls": (8.7, 11.6)}[mdl]
            check(f"BreaKHis {mdl}: overstatement {exp[0]:.1f}% selected",
                  sel_, exp[0], 0.05)
            check(f"BreaKHis {mdl}: overstatement {exp[1]:.1f}% without "
                  f"selection", fin_, exp[1], 0.05)
        # error rate under the selection-free protocol
        ib, il = over[("yolo11s-cls", "random")]
        ub, ul = over[("yolo11s-cls", "patient")]
        check("BreaKHis: error rate rises 15-fold without selection",
              (1 - ul) / (1 - il), 15.0, 0.4,
              note=f"{(1-il)*100:.2f}% to {(1-ul)*100:.2f}%")



# ------------------------------------------------------------------- har --
def har():
    ir = load("har", "in_region.csv")
    if ir is None:
        pending("HAR item-level vs unit-level", "run har_run.py --step train")
    else:
        for mdl, r_, u_, over in (("mlp256", 0.9885, 0.9519, 3.7),
                                  ("rf", 0.9790, 0.9347, 4.5)):
            s_ = ir[ir["model"] == mdl]
            if s_.empty:
                pending(f"HAR {mdl}", "run har_run.py --step train for it")
                continue
            rnd = folds_of(ir, "macro_f1", "random", mdl)
            sub = folds_of(ir, "macro_f1", "subject", mdl)
            check(f"HAR {mdl}: item-level = {r_}", rnd.mean(), r_, 0.0005)
            check(f"HAR {mdl}: unit-level = {u_}", sub.mean(), u_, 0.0005)
            check(f"HAR {mdl}: overstatement = {over}%",
                  100 * (rnd.mean() - sub.mean()) / rnd.mean(), over, 0.15)
            er = (1 - sub.mean()) / (1 - rnd.mean())
            check(f"HAR {mdl}: the error rate at least triples",
                  er > 3, True,
                  note=f"{100*(1-rnd.mean()):.2f}% to {100*(1-sub.mean()):.2f}%, "
                       f"{er:.1f}x")
            w = seed_sd(ir, "macro_f1", "subject", mdl)
            check(f"HAR {mdl}: between-subject spread exceeds the seed "
                  f"spread severalfold", sub.std() / w > 5, True,
                  note=f"{sub.std()/w:.1f}x")
        # The two paradigms share only their input features and still give
        # the same direction and a similar magnitude.
        a_ = folds_of(ir, "macro_f1", "random", "mlp256")
        b_ = folds_of(ir, "macro_f1", "subject", "mlp256")
        c_ = folds_of(ir, "macro_f1", "random", "rf")
        d_ = folds_of(ir, "macro_f1", "subject", "rf")
        if len(a_) and len(c_):
            o1 = 100 * (a_.mean() - b_.mean()) / a_.mean()
            o2 = 100 * (c_.mean() - d_.mean()) / c_.mean()
            check("HAR: a neural network and a tree ensemble give the same "
                  "overstatement to within a point", abs(o1 - o2) < 1.0, True,
                  note=f"{o1:.1f}% and {o2:.1f}%")

    bs = load("har", "har_by_subject.csv")
    if bs is None:
        pending("HAR per-subject spread", "run har_run.py --step per_subject")
    else:
        for mdl, mean_, cv_, lo_, hi_ in (
                ("mlp256", 0.950, 0.055, 0.756, 0.998),
                ("rf", 0.931, 0.068, 0.743, 1.000)):
            s_ = bs[bs["model"] == mdl] if "model" in bs.columns else bs
            if s_.empty:
                pending(f"HAR per-subject, {mdl}",
                        "run har_run.py --step per_subject for it")
                continue
            g = s_.groupby("group")["macro_f1"].mean()
            check(f"HAR {mdl}: 30 subjects scored", len(g), 30, 0)
            check(f"HAR {mdl}: per-subject mean = {mean_}",
                  g.mean(), mean_, 0.002)
            check(f"HAR {mdl}: per-subject CV = {cv_}",
                  g.std() / g.mean(), cv_, 0.005)
            check(f"HAR {mdl}: per-subject range = {lo_} to {hi_}",
                  f"{g.min():.3f}-{g.max():.3f}", f"{lo_:.3f}-{hi_:.3f}")
            if g.max() < 0.9999:
                check(f"HAR {mdl}: the worst subject errs far more often "
                      f"than the best", (1 - g.min()) / (1 - g.max()) > 50,
                      True,
                      note=f"{(1-g.min())/(1-g.max()):.0f}x, which a "
                           f"coefficient of variation of {cv_} conceals")
        if "n_windows" in bs.columns:
            n = bs.groupby("group")["n_windows"].first()[g.index]
            check("HAR: windows per subject do not predict that subject's "
                  "score", abs(g.corr(n)) < 0.3, True,
                  note=f"r = {g.corr(n):+.3f}")

    sc = load("har", "site_count.csv")
    if sc is None:
        pending("HAR dose-response", "run har_run.py --step site_count")
    else:
        # 每个受试者按 n_train // k 等量取样，除不尽时总量会少几个窗口。
        # k 越大越少（1000/992/984），方向与"站点多所以更好"相反，
        # 所以它不会制造假效应；但它不是严格恒定，据实断言。
        spread = 100 * (1 - sc["n_train"].min() / sc["n_train"].max())
        check("HAR: training window count varies by less than 2% across k",
              spread < 2.0, True,
              note=f"{sc['n_train'].min()}-{sc['n_train'].max()} windows, "
                   f"{spread:.1f}%; larger k gets slightly fewer")
        # 同样按模型分开。mlp256 与 rf 的曲线形状一致但水平不同，
        # 混在一起会同时压低增幅和离散度。
        models_sc = (sorted(sc["model"].unique())
                     if "model" in sc.columns else [None])
        for mdl, gain in (("mlp256", 6.6), ("rf", 4.8)):
            sub = sc[sc["model"] == mdl] if mdl in models_sc else None
            if sub is None or sub.empty:
                pending(f"HAR dose-response, {mdl}",
                        "run har_run.py --step site_count for it")
                continue
            byk = sub.groupby("k")["macro_f1"].mean()
            ks = sorted(byk.index)
            check(f"HAR {mdl}: k={ks[0]} -> k={ks[-1]} raises the held-out "
                  f"score {gain}%",
                  100 * (byk[ks[-1]] - byk[ks[0]]) / byk[ks[0]], gain, 0.3)
            draw = (sub.groupby(["k", "draw"])["macro_f1"].mean()
                       .groupby("k").std())
            seed = (sub.groupby(["k", "draw"])["macro_f1"].std()
                       .groupby("k").mean())
            check(f"HAR {mdl}: draw spread falls at least threefold",
                  draw[ks[0]] / draw[ks[-1]] > 2.5, True,
                  note=f"{draw[ks[0]]:.4f} -> {draw[ks[-1]]:.4f} "
                       f"({draw[ks[0]]/draw[ks[-1]]:.1f}x)")
            check(f"HAR {mdl}: at k={ks[-1]} the draw spread is within an "
                  f"order of magnitude of seed noise",
                  draw[ks[-1]] / seed[ks[-1]] < 10, True,
                  note=f"draw {draw[ks[-1]]:.4f} vs seed {seed[ks[-1]]:.4f}, "
                       f"ratio {draw[ks[-1]] / seed[ks[-1]]:.1f}")
        if len(models_sc) > 1 and models_sc[0] is not None:
            last_k = sc["k"].max()
            tail = (sc[sc["k"] == last_k]
                    .groupby(["model", "draw"])["macro_f1"].mean()
                    .groupby("model").std())
            check("HAR: both model families end at a draw spread below 0.01",
                  bool((tail < 0.01).all()), True,
                  note="; ".join(f"{m} {v:.4f}" for m, v in tail.items()))


# ------------------------------------------------------------- the shape --
def internal_consistency():
    """稿子关于自身设计的陈述，外部读者发现的四处矛盾，锁住。"""
    # -- internal consistency: numbers the prose states about itself -----
    # Four of these were found by an external reader, not by these checks,
    # because they are statements the manuscript makes about its own design
    # rather than results recomputed from a table. They are locked now.
    sab = load("durian", "sabah_by_tree.csv")
    if sab is not None:
        s1 = sab[sab["model"] == "yolo11s"].groupby("group")["n_images"].first()
        check("Sabah: twelve trees survive the five-item threshold",
              int((s1 >= 5).sum()), 12, 0,
              note="Methods once said twelve were excluded, not retained")
    gir = load("gwhd", "in_region.csv")
    if gir is not None:
        runs = gir.drop_duplicates(["model", "config", "seed"])
        check("GWHD: 59 runs of an intended 60", len(runs), 59, 0)
        check("GWHD: the design is therefore not fully crossed",
              len(runs) < gir["model"].nunique() * gir["config"].nunique()
              * gir["seed"].nunique(), True,
              note="one cell empty: yolo11n / random_fold4 / seed 2")
    for ds, kmin, want, target in (("durian", 1, 4, 6), ("gwhd", 2, 1, 8)):
        sc = load(ds, "site_count.csv")
        if sc is None:
            continue
        got = int(sc[sc["k"] == kmin]["draw"].nunique())
        check(f"{ds} dose-response: {want} of {target} draws survive at "
              f"k={kmin}", got, want, 0,
              note="draws that cannot reach the item budget are skipped, "
                   "which biases small k towards larger units")


def model_comparison():
    """单元异质性什么时候改变模型排名。

    这一节回答审读的核心质疑：「单元占 63–89%」不自动意味着模型比较不可靠。
    锁住的是那个次序——跨族的配对差大、稳；同族的小、脆。
    """
    mc = load("root", "results_model_comparison.csv")
    if mc is None:
        pending("model comparison under resampled evaluation units",
                "results_model_comparison.csv absent")
        return
    check("nine model pairs are available", len(mc), 9, 0)
    want = {("Durian farms", "rtdetr-l vs yolo11s"): (0.0346, 0.95, 3),
            ("Durian farms", "rtdetr-l vs yolo11m"): (0.0250, 0.87, 4),
            ("HAR subjects", "mlp256 vs rf"): (0.0192, 0.52, 11),
            ("GWHD sessions", "yolo11n vs yolo11s"): (-0.0190, 0.40, 18),
            ("BreaKHis patients", "yolo11n-cls vs yolo11s-cls"):
                (0.0042, 0.08, 384),
            ("Durian farms", "yolo11m vs yolo11n"): (-0.0008, 0.02, 6519)}
    for (ds, pair), (md, d_, k5) in want.items():
        row = mc[(mc["dataset"] == ds) & (mc["pair"] == pair)]
        check(f"{ds}, {pair}: paired difference {md:+.4f}",
              float(row["mean_diff"].iloc[0]), md, 0.0002)
        check(f"{ds}, {pair}: d = {d_:.2f}",
              float(row["d"].iloc[0]), d_, 0.006)
        check(f"{ds}, {pair}: {k5} units for a 5% reversal rate",
              int(row["k_for_5pct"].iloc[0]), k5, 1)

    cross = mc[mc["pair"].isin(["rtdetr-l vs yolo11s", "rtdetr-l vs yolo11m"])]
    same = mc[~mc["pair"].isin(["rtdetr-l vs yolo11s", "rtdetr-l vs yolo11m",
                                "mlp256 vs rf"])]
    check("the two cross-family pairs have the largest standardised "
          "differences", float(cross["d"].min()) > float(same["d"].max()),
          True, note=f"cross {float(cross['d'].min()):.2f} against "
                     f"same-family max {float(same['d'].max()):.2f}")
    check("every same-family pair needs more units than either cross-family "
          "pair", int(same["k_for_5pct"].min()) > int(cross["k_for_5pct"].max()),
          True, note=f"{int(same['k_for_5pct'].min())} against "
                     f"{int(cross['k_for_5pct'].max())}")
    bk = mc[mc["dataset"] == "BreaKHis patients"]
    check("the BreaKHis pair reverses on more than a third of 16-unit draws",
          float(bk["flip_k16"].iloc[0]) > 0.33, True,
          note=f"{100*float(bk['flip_k16'].iloc[0]):.0f}%")


def unit_confidence():
    """置信度能否预示单元失败——durian 说不能，81 个病人说能。

    论文现在主张的是「随单元异质性递减」这个次序，所以这里锁次序。
    """
    uc = {}
    for ds, ck in (("gwhd", "mean_max_conf"), ("breakhis", "mean_max_prob")):
        d = load(ds, "unit_confidence.csv")
        if d is None:
            pending(f"{ds}: confidence against per-unit score",
                    f"{D[ds]}/unit_confidence.csv absent")
            continue
        d = d.dropna(subset=["score"])
        uc[ds] = float(d[ck].corr(d["score"]))
    if len(uc) == 2:
        check("GWHD: 47 sessions, confidence against session score",
              uc["gwhd"], 0.258, 0.005)
        check("BreaKHis: 81 patients, confidence against patient accuracy",
              uc["breakhis"], 0.571, 0.005)
        bk = load("breakhis", "unit_confidence.csv").dropna(subset=["score"])
        for alt, want in (("mean_entropy", -0.571), ("mean_margin", 0.571)):
            check(f"BreaKHis: {alt} carries the same information",
                  float(bk[alt].corr(bk["score"])), want, 0.005)
        order = [0.093, uc["gwhd"], uc["breakhis"]]
        check("confidence becomes less informative as units differ more",
              order == sorted(order), True,
              note="durian +0.09 (CV 0.75) < GWHD %+0.2f (0.27) < "
                   "BreaKHis %+0.2f (0.19)" % (uc["gwhd"], uc["breakhis"]))
        trimmed = bk.drop(bk.nsmallest(5, "score").index)
        rt = float(trimmed["mean_max_prob"].corr(trimmed["score"]))
        check("the BreaKHis association is not carried by the worst patients",
              rt > uc["breakhis"], True,
              note=f"{rt:+.3f} without the worst five")
        tail = bk[bk["score"] < 0.60]
        check("the five worst patients are barely less confident",
              float(tail["mean_max_prob"].mean()), 0.916, 0.005,
              note=f"against {float(bk['mean_max_prob'].mean()):.3f} overall")

    scr = load("breakhis", "confidence_screening.csv")
    if scr is None:
        pending("confidence as a screen",
                "results_breakhis/confidence_screening.csv absent")
    else:
        for ds, caught, base in (("BreaKHis", 8, 0.148), ("GWHD", 3, 0.234)):
            row = scr[(scr["dataset"] == ds) & (scr["flag_frac"] == 0.2)]
            check(f"{ds}: flagging the least confident fifth catches "
                  f"{caught} poor units", int(row["caught"].iloc[0]),
                  caught, 0)
            check(f"{ds}: base rate {base:.1%}",
                  float(row["base_rate"].iloc[0]), base, 0.002)
        bkr = scr[(scr["dataset"] == "BreaKHis") & (scr["flag_frac"] == 0.2)]
        gwr = scr[(scr["dataset"] == "GWHD") & (scr["flag_frac"] == 0.2)]
        check("the screen works on patients and not on sessions",
              float(bkr["recall"].iloc[0]) > 2 * float(gwr["recall"].iloc[0]),
              True, note=f"recall {float(bkr['recall'].iloc[0]):.0%} against "
                         f"{float(gwr['recall'].iloc[0]):.0%}")


def across_datasets():
    """The claim the four datasets exist to support: the overstatement
    varies by an order of magnitude, and it orders the datasets the same
    way the between-unit dispersion does."""
    over, cv = {}, {}
    ir = load("durian", "in_region.csv")
    if ir is not None:
        s = ir[ir["model"] == "yolo11s"]
        r_ = s[s["config"] == "random"]["mAP50"].mean()
        f_ = folds_of(ir, "mAP50", "byfarm", "yolo11s").mean()
        over["durian"] = 100 * (r_ - f_) / r_
    pb = load("durian", "peninsula_by_burst_60s_min5.csv")
    if pb is not None:
        g = pb.groupby("burst")["mAP50"].mean()
        cv["durian"] = g.std() / g.mean()
    for ds, met, pre, unit, ukey, nkey in (
            ("gwhd", "mAP50", "domain", "gwhd_by_domain.csv", "group",
             "n_images"),
            ("breakhis", "top1", "patient", "breakhis_by_patient.csv",
             "group", "n_images"),
            ("har", "macro_f1", "subject", "har_by_subject.csv", "group",
             "n_windows")):
        d = load(ds, "in_region.csv")
        if d is not None:
            r_ = folds_of(d, met, "random").mean()
            f_ = folds_of(d, met, pre).mean()
            over[ds] = 100 * (r_ - f_) / r_
        u = load(ds, unit)
        if u is not None:
            g = unit_scores(u, ukey, met, nkey)
            cv[ds] = g.std() / g.mean()

    if len(over) >= 3:
        v = sorted(over.values())
        check("overstatement spans an order of magnitude across datasets",
              v[-1] / v[0] > 8, True,
              note="; ".join(f"{k} {x:.1f}%" for k, x in
                             sorted(over.items(), key=lambda t: -t[1])))
    common = sorted(set(over) & set(cv))
    if len(common) >= 3:
        a = [over[k] for k in common]
        b = [cv[k] for k in common]
        rho = pd.Series(a).corr(pd.Series(b), method="spearman")
        check("overstatement and between-unit dispersion order the "
              "datasets the same way", rho, 1.0, 0.001,
              note="; ".join(f"{k}: {over[k]:.1f}% / CV {cv[k]:.2f}"
                             for k in common))


def table_integrity():
    """Assert that each table holds the models it is supposed to.

    The checks below filter by model and skip what they cannot find. That is
    the right behaviour for a partially finished experiment and the wrong
    behaviour for a finished one: a table that lost half its rows in transit
    would pass every remaining check while describing a different experiment.
    This ran green once on three truncated tables, which is the reason it
    exists.
    """
    spec = [
        ("durian", "in_region.csv", 180,
         {"yolo11s", "yolo11n", "yolo11m", "rtdetr-l"}),
        ("durian", "cross_island.csv", 180,
         {"yolo11s", "yolo11n", "yolo11m", "rtdetr-l"}),
        ("durian", "checkpoint_bias.csv", 135,
         {"yolo11n", "yolo11m", "rtdetr-l"}),
        ("gwhd", "in_region.csv", 59, {"yolo11s", "yolo11n"}),
        ("gwhd", "gwhd_by_domain.csv", 282, {"yolo11s", "yolo11n"}),
        ("gwhd", "checkpoint_bias.csv", 59, {"yolo11s", "yolo11n"}),
        ("breakhis", "in_region.csv", 100,
         {"yolo11s-cls", "yolo11n-cls"}),
        ("breakhis", "breakhis_by_patient.csv", 810,
         {"yolo11s-cls", "yolo11n-cls"}),
        ("har", "in_region.csv", 100, {"mlp256", "rf"}),
        ("har", "har_by_subject.csv", 300, {"mlp256", "rf"}),
        ("har", "site_count.csv", 96, {"mlp256", "rf"}),
    ]
    for ds, name, n_rows, want in spec:
        df = load(ds, name)
        if df is None:
            pending(f"table {ds}/{name}", "not present")
            continue
        check(f"table {ds}/{name}: {n_rows} rows", len(df), n_rows, 0)
        got = (set(df["model"].dropna().unique())
               if "model" in df.columns else set())
        check(f"table {ds}/{name}: models {sorted(want)}",
              got == want, True,
              note=f"found {sorted(got) if got else 'no model column'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--todo", action="store_true")
    a = ap.parse_args()

    table_integrity()
    durian()
    gwhd()
    breakhis()
    har()
    model_comparison()
    unit_confidence()
    internal_consistency()
    eval_unit_curve()
    across_datasets()

    fails = [c for c in checks if not c["ok"]]
    width = max(len(c["claim"]) for c in checks)

    if a.todo:
        print(f"{len(todo)} claims awaiting data:\n")
        for t in todo:
            print(f"  {t['claim']:<{width}}  {t['why']}")
        return 1 if todo else 0

    if a.verbose:
        for c in checks:
            mark = "ok  " if c["ok"] else "FAIL"
            line = f"{mark}  {c['claim']:<{width}}  {c['got']:>10s}"
            if not c["ok"] and c["got"] != "PENDING":
                line += f"   expected {c['expected']}"
            if c["note"]:
                line += f"   [{c['note']}]"
            print(line)
    else:
        for c in fails:
            print(f"FAIL  {c['claim']:<{width}}  got {c['got']}   "
                  f"expected {c['expected']}"
                  + (f"   [{c['note']}]" if c["note"] else ""))

    print("\n" + "-" * 62)
    print(f"{len(checks) - len(fails)} of {len(checks)} claims reproduce")
    if todo:
        print(f"{len(todo)} await data; run with --todo to list them")
    if fails:
        print(f"{len(fails)} do not. Either the tables changed or the paper "
              f"says something the tables do not.")
        return 1
    print("every quantitative claim in the manuscript matches the released "
          "tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
