#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_decomposition.py -- 单元嵌套于折的方差分解，零 GPU。

主文的交叉分解在 durian 上没问题（一折就是一个农场），但在 GWHD、BreaKHis
和 HAR 上一折含多个单元，所以那里分解出来的是「留出样本的构成」，不是
per-unit 方差。这个脚本把两者分开。

模型：

    y[m, u, s] = mu + M[m] + F[f(u)] + U[u:f] + S[s] + eps

    M  模型              交叉
    F  折                交叉（决定训练集）
    U  单元，嵌套于折     ★ 主文想报的那个分量
    S  种子              交叉

F 与 U 的区别值得说清楚：F 是「训练集里少了这一组单元」的效应，U 是
「在这一组里被打分的是哪一个单元」的效应。前者掺着训练集变化，后者不掺。
主文现在把两者合并报成 78.7% / 56.1%，这个脚本给出拆开后的数。

单元数在各折间不等，所以用调和平均做不平衡嵌套的矩估计；这是近似，
输出里会打印不平衡程度，差得太多就该改用 REML。

用法：
    python nested_decomposition.py --root .
    python nested_decomposition.py --root . --dataset breakhis
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# (名称, 表, 单元列, 折列, 分数列, 模型列)
SETS = [
    ("GWHD", "results_gwhd/gwhd_by_domain.csv", "group", "fold",
     "mAP50", "model"),
    ("BreaKHis", "results_breakhis/breakhis_by_patient.csv", "group", "fold",
     "top1", "model"),
    ("HAR", "results_har/har_by_subject.csv", "group", "fold",
     "macro_f1", "model"),
]


def nested_components(df, ukey, fkey, skey, metric, mkey):
    """不平衡嵌套设计的矩估计。返回各分量的方差。"""
    df = df.dropna(subset=[metric]).copy()
    grand = df[metric].mean()
    n = len(df)

    # 各因子的平方和（I 型，按 模型 -> 折 -> 单元(折内) -> 种子 的顺序）
    def ss_between(keys):
        g = df.groupby(keys)[metric]
        return float((g.count() * (g.mean() - grand) ** 2).sum())

    ss_total = float(((df[metric] - grand) ** 2).sum())
    ss_m = ss_between([mkey])
    ss_f = ss_between([fkey])

    # 单元（折内）：先在折内中心化
    fold_mean = df.groupby(fkey)[metric].transform("mean")
    unit_mean = df.groupby([fkey, ukey])[metric].transform("mean")
    ss_u = float(((unit_mean - fold_mean) ** 2).sum())

    seed_mean = df.groupby(skey)[metric].transform("mean")
    ss_s = float((df.groupby(skey)[metric].count()
                  * (df.groupby(skey)[metric].mean() - grand) ** 2).sum())

    ss_res = max(0.0, ss_total - ss_m - ss_f - ss_u - ss_s)

    n_m = df[mkey].nunique()
    n_f = df[fkey].nunique()
    n_s = df[skey].nunique()
    n_u = df.groupby([fkey, ukey]).ngroups

    df_m, df_f = n_m - 1, n_f - 1
    df_u = n_u - n_f
    df_s = n_s - 1
    df_res = max(1, n - 1 - df_m - df_f - df_u - df_s)

    ms = {"model": ss_m / max(1, df_m), "fold": ss_f / max(1, df_f),
          "unit": ss_u / max(1, df_u), "seed": ss_s / max(1, df_s),
          "residual": ss_res / df_res}

    # 每个单元的观测数（模型 × 种子），以及每折的平均单元数
    reps = df.groupby([fkey, ukey]).size()
    k_rep = len(reps) / (1.0 / reps).sum()          # 调和平均
    units_per_fold = df.groupby(fkey)[ukey].nunique()
    k_unit = len(units_per_fold) / (1.0 / units_per_fold).sum()

    v_res = ms["residual"]
    v_u = max(0.0, (ms["unit"] - v_res) / k_rep)
    v_f = max(0.0, (ms["fold"] - ms["unit"]) / (k_rep * k_unit))
    v_m = max(0.0, (ms["model"] - v_res) / (n / n_m))
    v_s = max(0.0, (ms["seed"] - v_res) / (n / n_s))

    tot = v_f + v_u + v_m + v_s + v_res
    return {"fold (training set)": v_f, "unit within fold": v_u,
            "model": v_m, "seed": v_s, "residual": v_res}, tot, {
        "n_models": n_m, "n_folds": n_f, "n_units": n_u, "n_seeds": n_s,
        "reps_per_unit": k_rep,
        "units_per_fold_min": int(units_per_fold.min()),
        "units_per_fold_max": int(units_per_fold.max()),
        "imbalance": float(units_per_fold.max() / units_per_fold.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--boot", type=int, default=1000,
                    help="对折做 bootstrap 求区间；0 关闭")
    ap.add_argument("--out", default="results_nested_decomposition.csv")
    a = ap.parse_args()

    rows_out = []
    for name, rel, ukey, fkey, metric, mkey in SETS:
        if a.dataset and a.dataset.lower() not in name.lower():
            continue
        path = os.path.join(a.root, rel)
        if not os.path.isfile(path):
            print(f"缺表，跳过: {rel}")
            continue
        df = pd.read_csv(path)
        for c in (ukey, fkey, "seed", metric, mkey):
            if c not in df.columns:
                print(f"{name}: 缺列 {c}，跳过")
                break
        else:
            comp, tot, info = nested_components(df, ukey, fkey, "seed",
                                                metric, mkey)
            print(f"\n{name}: {info['n_units']} 个单元分布在 "
                  f"{info['n_folds']} 折, {info['n_models']} 模型, "
                  f"{info['n_seeds']} 种子")
            print(f"  每折单元数 {info['units_per_fold_min']}–"
                  f"{info['units_per_fold_max']} "
                  f"(不平衡比 {info['imbalance']:.2f})")
            if info["imbalance"] > 2.0:
                print("  ★ 不平衡比超过 2，矩估计是粗略近似，正式报告前"
                      "应改用 REML")
            for k, v in comp.items():
                print(f"  {k:<22}{100 * v / tot:>7.1f}%")
                rows_out.append([name, k, round(v, 8),
                                 round(100 * v / tot, 2)])

            # 折 bootstrap 区间
            if a.boot:
                folds = sorted(df[fkey].unique())
                rng = np.random.default_rng(0)
                acc = defaultdict(list)
                for _ in range(a.boot):
                    pick = rng.choice(folds, size=len(folds), replace=True)
                    parts = []
                    for i, f in enumerate(pick):
                        sub = df[df[fkey] == f].copy()
                        sub[fkey] = f"b{i}"
                        parts.append(sub)
                    try:
                        c2, t2, _ = nested_components(
                            pd.concat(parts), ukey, fkey, "seed", metric,
                            mkey)
                    except (ZeroDivisionError, ValueError):
                        continue
                    if t2 <= 0:
                        continue
                    for k, v in c2.items():
                        acc[k].append(100 * v / t2)
                print("  90% bootstrap over folds:")
                for k in comp:
                    if acc[k]:
                        lo, hi = np.percentile(acc[k], [5, 95])
                        print(f"    {k:<22}{lo:>6.1f} – {hi:.1f}%")

    if rows_out:
        out = os.path.join(a.root, a.out)
        pd.DataFrame(rows_out, columns=["dataset", "component", "variance",
                                        "share_pct"]).to_csv(
            out, index=False, encoding="utf-8-sig")
        print(f"\n写出 {out}")

    print("\n读法：'fold (training set)' 是「训练集里少了这组单元」的效应，")
    print("     'unit within fold' 是「组里被打分的是哪个单元」的效应。")
    print("     主文交叉分解把两者合并报成一个数；这里拆开。若 unit 分量")
    print("     本身就很大，主文那句「留出样本主导」就能收紧成「单元主导」。")


if __name__ == "__main__":
    main()
