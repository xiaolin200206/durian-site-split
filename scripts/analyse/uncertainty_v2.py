#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_v2.py -- 三条审稿意见的统计修正，零 GPU。

前一版对三处质疑的回应还不够。这一版按意见逐条重做。

[1] 方差分量的区间：重采样单位错了
    三个公开数据集是五折设计，前一版对**折**做 bootstrap，实际只有五个
    可重抽的对象，而且各折的训练集互相重叠。五个对象的百分位区间几乎没有
    覆盖率保证。
    改为对**单元**重抽（47 个 session、81 个病人、30 个受试者），这是
    per-unit 表里真正可交换的层级，重抽对象多一个量级。同时报告两种重抽
    的结果，让读者看到差多少。
    ★ 仍无法解决的：单元重抽不改变训练集，所以它捕捉的是「换一批评价单元
      会怎样」，不是「换一批训练划分会怎样」。后者需要重新划分并重训，
      本脚本明确不声称覆盖它。

[2] 噪声上界：取两者较大值并不构成上界
    审稿意见对。二项估计忽略组内相关（低估），种子重复估计在固定测试样本
    上主要反映训练随机性（不含组内抽样），两者取大仍不覆盖「组内抽样
    误差在有相关时的真实大小」。
    改为给出一个**依赖 ICC 的上界族**：在设计效应 1+(m-1)*rho 下，二项
    方差要乘以设计效应。对一组 rho 报出噪声份额，让读者看到结论在多大的
    组内相关下才失效，而不是给一个单点的「至多」。

[3] 所需单元数：未传播效应量的不确定性
    (1.645/d)^2 直接代入样本 d，在八个农场上把 d 的抽样误差完全忽略了。
    改为对单元做 bootstrap 得到 d 的分布，再把每个 bootstrap 复本代入
    公式，报告所需单元数的区间。区间会很宽，尤其 durian——那正是要显示的。

用法：
    python uncertainty_v2.py --root . --boot 4000
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

R = "results_clean"


def nested_ss(df, metric, unit="group", fold="fold"):
    grand = df[metric].mean()
    fm = df.groupby(fold)[metric].transform("mean")
    um = df.groupby([fold, unit])[metric].transform("mean")
    mm = df.groupby("model")[metric].transform("mean")
    sm = df.groupby("seed")[metric].transform("mean")
    sst = ((df[metric] - grand) ** 2).sum()
    if sst <= 0:
        return None
    out = {"fold": ((fm - grand) ** 2).sum(), "unit": ((um - fm) ** 2).sum(),
           "model": ((mm - grand) ** 2).sum(),
           "seed": ((sm - grand) ** 2).sum()}
    out["residual"] = sst - sum(out.values())
    return {k: 100 * v / sst for k, v in out.items()}


def frames(root):
    return {
        "gwhd": (pd.read_csv(f"{root}/{R}/gwhd_by_domain_clean.csv",
                             encoding="utf-8-sig"), "mAP50", "n_images"),
        "breakhis": (pd.read_csv(f"{root}/{R}/breakhis_by_patient_clean.csv",
                                 encoding="utf-8-sig"), "top1", "n_images"),
        "har": (pd.read_csv(f"{root}/results_har/har_by_subject.csv"),
                "macro_f1", "n_windows")}


# ------------------------------------------------------------------ [1] --
def intervals_by_unit(root, boot, rng):
    """对单元重抽，与对折重抽并排。"""
    print("=" * 78)
    print("[1] 方差分量区间：对单元重抽 vs 对折重抽")
    print("=" * 78)
    rows = []
    for ds, (df, met, _) in frames(root).items():
        units = sorted(df.group.unique())
        folds = sorted(df.fold.unique())
        point = nested_ss(df, met)
        acc_u, acc_f = {k: [] for k in point}, {k: [] for k in point}
        by_unit = {u: df[df.group == u] for u in units}
        by_fold = {f: df[df.fold == f] for f in folds}
        for _ in range(boot):
            pick = rng.choice(units, len(units), replace=True)
            parts = []
            for i, u in enumerate(pick):
                s = by_unit[u].copy()
                s["group"] = f"u{i}"
                parts.append(s)
            r = nested_ss(pd.concat(parts), met)
            if r:
                for k, v in r.items():
                    acc_u[k].append(v)
        for _ in range(boot):
            pick = rng.choice(folds, len(folds), replace=True)
            parts = []
            for i, f in enumerate(pick):
                s = by_fold[f].copy()
                s["fold"] = f"b{i}"
                parts.append(s)
            r = nested_ss(pd.concat(parts), met)
            if r:
                for k, v in r.items():
                    acc_f[k].append(v)
        print(f"\n  {ds}  ({len(units)} units, {len(folds)} folds)")
        print(f"    {'component':<12}{'point':>7}{'unit bootstrap':>22}"
              f"{'fold bootstrap':>22}")
        for k in ("unit", "fold", "model", "seed", "residual"):
            lu, hu = np.percentile(acc_u[k], [5, 95])
            lf, hf = np.percentile(acc_f[k], [5, 95])
            print(f"    {k:<12}{point[k]:6.1f}%   [{lu:5.1f}, {hu:5.1f}]"
                  f"        [{lf:5.1f}, {hf:5.1f}]")
            rows.append(dict(dataset=ds, component=k,
                             point=round(point[k], 2),
                             unit_lo=round(float(lu), 2),
                             unit_hi=round(float(hu), 2),
                             fold_lo=round(float(lf), 2),
                             fold_hi=round(float(hf), 2),
                             n_units=len(units), n_folds=len(folds)))
    print("\n  单元重抽的对象多一个量级，但它不改变训练划分；折重抽改变")
    print("  训练划分却只有五个对象，且各折训练集重叠。两者都不是完整的")
    print("  不确定性，重新划分并重训才是，本脚本不声称覆盖那一项。")
    return rows


# ------------------------------------------------------------------ [2] --
def noise_bounds(root):
    """噪声份额随假定的组内相关变化。"""
    print("\n" + "=" * 78)
    print("[2] 噪声份额对组内相关（ICC）的敏感性")
    print("=" * 78)
    rows = []
    SETS = [("BreaKHis patients",
             f"{root}/{R}/breakhis_by_patient_clean.csv", "top1", "n_images"),
            ("HAR subjects", f"{root}/results_har/har_by_subject.csv",
             "accuracy", "n_windows")]
    RHOS = (0.0, 0.05, 0.1, 0.2, 0.5)
    for label, path, met, nkey in SETS:
        d = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=[met])
        for model in sorted(d.model.unique()):
            s = d[d.model == model]
            g = s.groupby("group").agg(mean=(met, "mean"),
                                       sd_rep=(met, "std"),
                                       k=(met, "size"), n=(nkey, "first"))
            between = float(g["mean"].std(ddof=1))
            var_rep = float(((g.sd_rep / np.sqrt(g.k)) ** 2).mean())
            print(f"\n  {label}, {model}   between-unit s.d. {between:.4f}")
            print(f"    {'rho':<8}{'design effect':>15}{'noise share':>14}"
                  f"{'heterogeneity':>15}")
            for rho in RHOS:
                deff = 1 + (g.n - 1) * rho
                var_bin = float(((g["mean"] * (1 - g["mean"]) / g.n)
                                 * deff).mean())
                var_tot = max(var_rep, var_bin)
                share = 100 * var_tot / between ** 2
                het = 100 * math.sqrt(max(0.0, 1 - var_tot / between ** 2))
                print(f"    {rho:<8.2f}{float(deff.mean()):>15.1f}"
                      f"{share:>13.1f}%{het:>14.1f}%")
                rows.append(dict(dataset=label, model=model, rho=rho,
                                 mean_design_effect=round(float(deff.mean()), 2),
                                 noise_share=round(share, 2),
                                 het_share=round(het, 2)))
            # 结论失效的临界 rho
            crit = None
            for rho in np.arange(0.0, 1.001, 0.005):
                deff = 1 + (g.n - 1) * rho
                vb = float(((g["mean"] * (1 - g["mean"]) / g.n) * deff).mean())
                if max(var_rep, vb) / between ** 2 > 0.5:
                    crit = rho
                    break
            print(f"    噪声占到方差一半所需的 rho: "
                  f"{crit if crit is not None else '>1'}")
            rows.append(dict(dataset=label, model=model, rho=-1,
                             mean_design_effect=None, noise_share=None,
                             het_share=None))
            rows[-1]["rho_for_half_variance"] = (round(float(crit), 3)
                                                 if crit is not None else None)
    print("\n  rho = 0 是独立假设，等于前一版的二项估计。真实的组内相关")
    print("  未知；这张表给出的是结论在多大相关下才失效，而不是一个上界。")
    return rows


# ------------------------------------------------------------------ [3] --
def d_uncertainty(root, boot, rng):
    """把 d 的抽样不确定性传播到所需单元数。"""
    print("\n" + "=" * 78)
    print("[3] 效应量与所需单元数的不确定性")
    print("=" * 78)
    rows = []
    SRC = [("durian farms", None, None, None)]
    # durian：从 in_region 拼
    a = pd.read_csv(f"{root}/{R}/durian_in_region_clean.csv",
                    encoding="utf-8-sig")
    b = pd.read_csv(f"{root}/{R}/durian_in_region_clean_yolo11l.csv",
                    encoding="utf-8-sig")
    c = pd.read_csv(f"{root}/{R}/durian_in_region_clean_frcnn.csv",
                    encoding="utf-8-sig").assign(eval_on="outer")
    keep = ["model", "config", "seed", "checkpoint", "eval_on", "mAP50"]
    dur = pd.concat([a[keep], b[keep], c[keep]], ignore_index=True)
    dur = dur[(dur.checkpoint == "best") & (dur.eval_on == "outer")
              & dur.config.str.startswith("byfarm")]
    pivots = {"durian farms": dur.pivot_table(index="config", columns="model",
                                              values="mAP50")}
    for ds, (df, met, _) in frames(root).items():
        name = {"gwhd": "GWHD sessions", "breakhis": "BreaKHis patients",
                "har": "HAR subjects"}[ds]
        pivots[name] = df.pivot_table(index="group", columns="model",
                                      values=met).dropna()

    print(f"\n{'pair':<30}{'N':>4}{'d':>6}{'d 90% CI':>16}"
          f"{'k pop':>7}{'k pop 90% CI':>18}")
    for label, piv in pivots.items():
        for m1, m2 in itertools.combinations(sorted(piv.columns), 2):
            A, B = piv[m1].to_numpy(), piv[m2].to_numpy()
            diff = A - B
            N = len(diff)
            d0 = abs(diff.mean()) / diff.std(ddof=1)
            ds_boot, ks = [], []
            for _ in range(boot):
                i = rng.integers(0, N, N)
                x = diff[i]
                sd = x.std(ddof=1)
                if sd <= 0:
                    continue
                dd = abs(x.mean()) / sd
                ds_boot.append(dd)
                ks.append(min((1.645 / dd) ** 2, 1e6) if dd > 0 else 1e6)
            dl, dh = np.percentile(ds_boot, [5, 95])
            kl, kh = np.percentile(ks, [5, 95])
            k0 = math.ceil((1.645 / d0) ** 2) if d0 > 0 else None
            print(f"{(m1+' vs '+m2)[:28]:<30}{N:>4}{d0:>6.2f}"
                  f"   [{dl:.2f}, {dh:.2f}]{k0:>9}"
                  f"   [{kl:>7.0f}, {kh:>8.0f}]")
            rows.append(dict(dataset=label, pair=f"{m1} vs {m2}", n_units=N,
                             d=round(float(d0), 3),
                             d_lo=round(float(dl), 3),
                             d_hi=round(float(dh), 3),
                             k_population=k0,
                             k_lo=int(round(float(kl))),
                             k_hi=int(round(float(kh))),
                             k_ci_excludes_n=bool(kl > N)))
    out = pd.DataFrame(rows)
    firm = int(out.k_ci_excludes_n.sum())
    print(f"\n  所需单元数的 90% 下界仍超过现有单元数的配对: {firm} / {len(out)}")
    print("  也就是说，只有这些配对在传播了 d 的不确定性之后，仍能断言")
    print("  「现有单元数不够」。其余配对的区间跨过了现有单元数，说明")
    print("  样本效应量本身估得太不准，不足以支撑任何方向的断言。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name, rows in (("intervals_by_unit", intervals_by_unit(a.root, a.boot, rng)),
                       ("noise_icc", noise_bounds(a.root)),
                       ("d_uncertainty", d_uncertainty(a.root, a.boot, rng))):
        p = os.path.join(a.root, R, f"summary_{name}.csv")
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n写出 {p}  ({len(rows)} 行)")


if __name__ == "__main__":
    sys.exit(main())
