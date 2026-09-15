#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_clean.py -- 干净协议下把论文需要的每个量算一遍。

一个脚本，一处口径。跑完写出 results_clean/summary_*.csv，稿子里的每个数字
都来自这些表，verify_claims.py 也从这些表重算。

输入（results_clean/）：
    durian_in_region_clean.csv          四个模型，外层与 Sabah
    durian_in_region_clean_yolo11l.csv  第五个
    durian_in_region_clean_frcnn.csv    两阶段跨族
    gwhd_in_region_clean.csv            折层面
    gwhd_by_domain_clean.csv            47 个 session
    breakhis_in_region_clean.csv        折层面
    breakhis_by_patient_clean.csv       81 个病人

durian 的逐农场分数就是折层面分数（一折即一农场），不需要单独的表。
HAR 不在其中：它没有 checkpoint 选择，旧结果本身就是干净的。
"""

import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

R = "results_clean"
OLD = {"durian": "results_durian", "gwhd": "results_gwhd",
       "breakhis": "results_breakhis", "har": "results_har"}
RNG = np.random.default_rng(20260914)
BOOT = 4000


def load_durian():
    """五个 YOLO/RT-DETR + Faster R-CNN，统一成一张表。"""
    a = pd.read_csv(f"{R}/durian_in_region_clean.csv", encoding="utf-8-sig")
    b = pd.read_csv(f"{R}/durian_in_region_clean_yolo11l.csv",
                    encoding="utf-8-sig")
    c = pd.read_csv(f"{R}/durian_in_region_clean_frcnn.csv",
                    encoding="utf-8-sig")
    c = c.assign(eval_on="outer")
    keep = ["model", "config", "seed", "checkpoint", "eval_on", "mAP50"]
    d = pd.concat([a[keep], b[keep], c[keep]], ignore_index=True)
    d["regime"] = np.where(d.config.str.startswith("random"), "item", "unit")
    return d


def ems_two_factor(cells):
    """两因素交叉随机效应，期望均方。cells[(model, fold)] = [每个种子的分数]"""
    M = sorted({m for m, _ in cells})
    F = sorted({f for _, f in cells})
    a, b = len(M), len(F)
    n = min(len(v) for v in cells.values())
    g = sum(sum(v[:n]) for v in cells.values()) / (a * b * n)
    ma = {m: sum(sum(cells[(m, f)][:n]) for f in F) / (b * n) for m in M}
    mf = {f: sum(sum(cells[(m, f)][:n]) for m in M) / (a * n) for f in F}
    ssa = b * n * sum((ma[m] - g) ** 2 for m in M)
    ssf = a * n * sum((mf[f] - g) ** 2 for f in F)
    ssaf = sse = 0.0
    for m in M:
        for f in F:
            v = cells[(m, f)][:n]
            cm = sum(v) / n
            ssaf += n * (cm - ma[m] - mf[f] + g) ** 2
            sse += sum((x - cm) ** 2 for x in v)
    msa, msf = ssa / (a - 1), ssf / (b - 1)
    msaf = ssaf / ((a - 1) * (b - 1))
    mse = sse / (a * b * (n - 1))
    ve = mse
    vaf = max(0.0, (msaf - mse) / n)
    va = max(0.0, (msa - msaf) / (n * b))
    vf = max(0.0, (msf - msaf) / (n * a))
    tot = vf + va + vaf + ve
    return {"unit": 100 * vf / tot, "model": 100 * va / tot,
            "interaction": 100 * vaf / tot, "seed": 100 * ve / tot}


def nested_ss(df, metric, unit="group", fold="fold"):
    """嵌套：unit within fold。按平方和分摊，报告份额。"""
    grand = df[metric].mean()
    fm = df.groupby(fold)[metric].transform("mean")
    um = df.groupby([fold, unit])[metric].transform("mean")
    mm = df.groupby("model")[metric].transform("mean")
    sm = df.groupby("seed")[metric].transform("mean")
    sst = ((df[metric] - grand) ** 2).sum()
    out = {"fold": ((fm - grand) ** 2).sum(), "unit": ((um - fm) ** 2).sum(),
           "model": ((mm - grand) ** 2).sum(), "seed": ((sm - grand) ** 2).sum()}
    out["residual"] = sst - sum(out.values())
    return {k: 100 * v / sst for k, v in out.items()}


def paired(piv, label):
    """每对模型的配对差、d、翻转率、5% 所需单元数。"""
    rows = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        A, B = piv[a].to_numpy(), piv[b].to_numpy()
        diff = A - B
        m, s = diff.mean(), diff.std(ddof=1)
        d = abs(m) / s if s > 0 else 0.0
        r = {"dataset": label, "pair": f"{a} vs {b}", "n_units": len(A),
             "mean_diff": round(m, 5), "sd_diff": round(s, 5),
             "d": round(d, 3),
             "wins_frac": round(float((diff > 0).mean()), 3)}
        for k in (2, 4, 6, 10, 16, 32):
            if k >= len(A):
                continue
            fl = 0
            for _ in range(3000):
                i = RNG.choice(len(A), k, replace=False)
                if np.sign(A[i].mean() - B[i].mean()) != np.sign(m):
                    fl += 1
            r[f"flip_k{k}"] = round(fl / 3000, 4)
        r["k_for_5pct"] = (math.ceil((1.645 / d) ** 2) if d > 0 else 99999)
        rows.append(r)
    return rows


def resample_curve(scores, weights, label, model):
    """评价单元重抽。等权与实例加权各一条。"""
    out = []
    N = len(scores)
    ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64) if k <= N} | {N})
    for mode in ("equal", "items"):
        for k in ks:
            vals = np.empty(BOOT)
            for b in range(BOOT):
                i = RNG.choice(N, k, replace=False)
                vals[b] = (scores[i].mean() if mode == "equal"
                           else (scores[i] * weights[i]).sum() / weights[i].sum())
            lo, hi = np.percentile(vals, [5, 95])
            out.append({"dataset": label, "model": model, "weighting": mode,
                        "n_units": N, "k_eval": k,
                        "mean": round(float(vals.mean()), 4),
                        "sd": round(float(vals.std(ddof=1)), 4) if k < N else 0.0,
                        "p5": round(float(lo), 4), "p95": round(float(hi), 4),
                        "width90": round(float(hi - lo), 4)})
    return out


def within_unit_noise(df, metric, nkey, model, label):
    """分类任务：把每单元分数当伯努利均值，分离度量噪声与真实异质性。"""
    g = df[df.model == model].groupby("group").agg(
        acc=(metric, "mean"), n=(nkey, "first"))
    se = []
    for _, r in g.iterrows():
        draws = RNG.binomial(int(r.n), min(max(r.acc, 0), 1), 2000) / r.n
        se.append(draws.std(ddof=1))
    g["se_within"] = se
    tot = g.acc.std(ddof=1)
    noise_var = float((g.se_within ** 2).mean())
    het = math.sqrt(max(0.0, tot ** 2 - noise_var))
    return {"dataset": label, "model": model, "n_units": len(g),
            "sd_between": round(float(tot), 4),
            "mean_se_within": round(float(math.sqrt(noise_var)), 4),
            "sd_heterogeneity": round(het, 4),
            "noise_share_of_variance": round(100 * noise_var / tot ** 2, 2)}


def main():
    os.makedirs(R, exist_ok=True)
    main_rows, dec_rows, pair_rows, curve_rows, unit_rows, noise_rows = (
        [], [], [], [], [], [])

    # ---------------------------------------------------------- durian --
    d = load_durian()
    best = d[(d.checkpoint == "best") & (d.eval_on == "outer")]
    last = d[(d.checkpoint == "last") & (d.eval_on == "outer")]
    for m in sorted(best.model.unique()):
        i = best[(best.model == m) & (best.regime == "item")].mAP50.mean()
        u = best[(best.model == m) & (best.regime == "unit")].mAP50.mean()
        li = last[(last.model == m) & (last.regime == "item")].mAP50.mean()
        lu = last[(last.model == m) & (last.regime == "unit")].mAP50.mean()
        main_rows.append({"dataset": "durian", "unit": "farm", "model": m,
                          "metric": "mAP50", "item": round(i, 4),
                          "unit_level": round(u, 4),
                          "overstatement_pct": round(100 * (i - u) / i, 2),
                          "item_last": round(li, 4) if li == li else None,
                          "unit_last": round(lu, 4) if lu == lu else None,
                          "n_seeds": int(best[(best.model == m)].seed.nunique())})
    # 逐农场（= 折）
    pu = best[best.regime == "unit"]
    piv = pu.pivot_table(index="config", columns="model", values="mAP50")
    for cfg, row in piv.iterrows():
        for m, v in row.items():
            if v == v:
                unit_rows.append({"dataset": "durian", "unit_id": cfg,
                                  "model": m, "score": round(float(v), 4)})
    pair_rows += paired(piv, "durian farms")
    # 方差分解：只用五种子齐全的四个模型，保持平衡
    bal = [m for m in piv.columns
           if pu[pu.model == m].seed.nunique() == 5 and m != "frcnn-r50"]
    cells = defaultdict(list)
    for _, r in pu[pu.model.isin(bal)].iterrows():
        cells[(r["model"], r["config"])].append(float(r["mAP50"]))
    dec_rows.append({"dataset": "durian", "design": "crossed, fold = one farm",
                     "models": ",".join(sorted(bal)), **{
                         k: round(v, 1) for k, v in
                         ems_two_factor(cells).items()}})
    sc = piv.mean(axis=1).to_numpy()
    curve_rows += resample_curve(sc, np.ones_like(sc), "durian farms", "mean")
    # Sabah
    sab = d[(d.checkpoint == "best") & (d.eval_on == "sabah")]
    for m in sorted(sab.model.unique()):
        s = sab[sab.model == m].groupby("config").mAP50.mean()
        u = best[(best.model == m) & (best.regime == "unit")].mAP50.mean()
        main_rows.append({"dataset": "durian", "unit": "sabah", "model": m,
                          "metric": "mAP50", "item": None,
                          "unit_level": round(float(s.mean()), 4),
                          "overstatement_pct": None,
                          "item_last": round(float(s.std()), 4),
                          "unit_last": round(u, 4), "n_seeds": None})

    # ------------------------------------------------------------ GWHD --
    g = pd.read_csv(f"{R}/gwhd_in_region_clean.csv", encoding="utf-8-sig")
    g["regime"] = np.where(g.config.str.startswith("random"), "item", "unit")
    gb = g[g.checkpoint == "best"]
    gl = g[g.checkpoint == "last"]
    for m in sorted(gb.model.unique()):
        i = gb[(gb.model == m) & (gb.regime == "item")].mAP50.mean()
        u = gb[(gb.model == m) & (gb.regime == "unit")].mAP50.mean()
        li = gl[(gl.model == m) & (gl.regime == "item")].mAP50.mean()
        lu = gl[(gl.model == m) & (gl.regime == "unit")].mAP50.mean()
        main_rows.append({"dataset": "gwhd", "unit": "session", "model": m,
                          "metric": "mAP50", "item": round(i, 4),
                          "unit_level": round(u, 4),
                          "overstatement_pct": round(100 * (i - u) / i, 2),
                          "item_last": round(li, 4), "unit_last": round(lu, 4),
                          "n_seeds": int(gb.seed.nunique())})
    gd = pd.read_csv(f"{R}/gwhd_by_domain_clean.csv", encoding="utf-8-sig")
    dec_rows.append({"dataset": "gwhd", "design": "nested, unit within fold",
                     "models": ",".join(sorted(gd.model.unique())),
                     **{k: round(v, 1) for k, v in
                        nested_ss(gd, "mAP50").items()}})
    pg = gd.pivot_table(index="group", columns="model", values="mAP50").dropna()
    pair_rows += paired(pg, "GWHD sessions")
    for gid, row in pg.iterrows():
        for m, v in row.items():
            unit_rows.append({"dataset": "gwhd", "unit_id": gid, "model": m,
                              "score": round(float(v), 4)})
    one = gd[gd.model == "yolo11s"].groupby("group").agg(
        s=("mAP50", "mean"), n=("n_images", "first"))
    curve_rows += resample_curve(one.s.to_numpy(), one.n.to_numpy(),
                                 "GWHD sessions", "yolo11s")

    # -------------------------------------------------------- BreaKHis --
    bk = pd.read_csv(f"{R}/breakhis_in_region_clean.csv", encoding="utf-8-sig")
    bk["regime"] = np.where(bk.config.str.startswith("random"), "item", "unit")
    bb, bl = bk[bk.checkpoint == "best"], bk[bk.checkpoint == "last"]
    for m in sorted(bb.model.unique()):
        i = bb[(bb.model == m) & (bb.regime == "item")].top1.mean()
        u = bb[(bb.model == m) & (bb.regime == "unit")].top1.mean()
        li = bl[(bl.model == m) & (bl.regime == "item")].top1.mean()
        lu = bl[(bl.model == m) & (bl.regime == "unit")].top1.mean()
        main_rows.append({"dataset": "breakhis", "unit": "patient", "model": m,
                          "metric": "top1", "item": round(i, 4),
                          "unit_level": round(u, 4),
                          "overstatement_pct": round(100 * (i - u) / i, 2),
                          "item_last": round(li, 4), "unit_last": round(lu, 4),
                          "n_seeds": int(bb.seed.nunique())})
    bp = pd.read_csv(f"{R}/breakhis_by_patient_clean.csv", encoding="utf-8-sig")
    dec_rows.append({"dataset": "breakhis", "design": "nested, unit within fold",
                     "models": ",".join(sorted(bp.model.unique())),
                     **{k: round(v, 1) for k, v in
                        nested_ss(bp, "top1").items()}})
    pb = bp.pivot_table(index="group", columns="model", values="top1").dropna()
    pair_rows += paired(pb, "BreaKHis patients")
    for gid, row in pb.iterrows():
        for m, v in row.items():
            unit_rows.append({"dataset": "breakhis", "unit_id": gid,
                              "model": m, "score": round(float(v), 4)})
    one = bp[bp.model == "yolo11s-cls"].groupby("group").agg(
        s=("top1", "mean"), n=("n_images", "first"))
    curve_rows += resample_curve(one.s.to_numpy(), one.n.to_numpy(),
                                 "BreaKHis patients", "yolo11s-cls")
    for m in sorted(bp.model.unique()):
        noise_rows.append(within_unit_noise(bp, "top1", "n_images", m,
                                            "BreaKHis patients"))

    # ------------------------------------------------------------- HAR --
    # 没有 checkpoint 选择，旧结果即干净结果，原样带入以便一张表读完
    hi = pd.read_csv(f"{OLD['har']}/in_region.csv")
    hi["regime"] = np.where(hi.config.str.startswith("random"), "item", "unit")
    for m in sorted(hi.model.unique()):
        i = hi[(hi.model == m) & (hi.regime == "item")].macro_f1.mean()
        u = hi[(hi.model == m) & (hi.regime == "unit")].macro_f1.mean()
        main_rows.append({"dataset": "har", "unit": "subject", "model": m,
                          "metric": "macro_f1", "item": round(i, 4),
                          "unit_level": round(u, 4),
                          "overstatement_pct": round(100 * (i - u) / i, 2),
                          "item_last": None, "unit_last": None,
                          "n_seeds": int(hi.seed.nunique())})
    hs = pd.read_csv(f"{OLD['har']}/har_by_subject.csv")
    hs = hs.rename(columns={"group": "group"})
    dec_rows.append({"dataset": "har", "design": "nested, unit within fold",
                     "models": ",".join(sorted(hs.model.unique())),
                     **{k: round(v, 1) for k, v in
                        nested_ss(hs, "macro_f1").items()}})
    ph = hs.pivot_table(index="group", columns="model",
                        values="macro_f1").dropna()
    pair_rows += paired(ph, "HAR subjects")
    for gid, row in ph.iterrows():
        for m, v in row.items():
            unit_rows.append({"dataset": "har", "unit_id": str(gid),
                              "model": m, "score": round(float(v), 4)})
    one = hs[hs.model == "mlp256"].groupby("group").agg(
        s=("macro_f1", "mean"), n=("n_windows", "first"))
    curve_rows += resample_curve(one.s.to_numpy(), one.n.to_numpy(),
                                 "HAR subjects", "mlp256")
    ha = hs.rename(columns={"n_windows": "n_images"})
    for m in sorted(ha.model.unique()):
        noise_rows.append(within_unit_noise(ha, "accuracy", "n_images", m,
                                            "HAR subjects"))

    for name, rows in (("main", main_rows), ("decomposition", dec_rows),
                       ("model_pairs", pair_rows), ("eval_curve", curve_rows),
                       ("per_unit", unit_rows), ("within_unit_noise",
                                                 noise_rows)):
        pd.DataFrame(rows).to_csv(f"{R}/summary_{name}.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"  {name:<20}{len(rows):>5} 行")
    print(f"\n写到 {R}/summary_*.csv")


if __name__ == "__main__":
    sys.exit(main())
