#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decomposition_robustness.py -- 方差分解的稳健性，零 GPU。

审稿意见（NCS 视角，第二条）说得准：正文把 unit 68–88% 对 model 0.0–2.7%
当成接近普遍的结论，而 Methods 自己承认把 model 当随机效应意味着这些架构是
某个「模型总体」的样本，实际每个数据集只有 2 到 5 个配置，这个假设「only
loosely」成立。它要三样东西：

    [1] 方差分量的不确定性区间
    [2] model 作为固定效应重算一遍，看份额变不变
    [3] 改变模型集合（子集重抽），看 68–88% 是不是集合太窄造成的

三样都能从已有的 per-unit 表算出来。durian 有六个配置、跨三个架构族，
所以 [3] 在那里最有说服力：如果抽两个模型和抽五个模型给出的单元份额都在
同一量级，那「份额高是因为模型太像」这条质疑就被数据挡住，而不是靠措辞。

★ 一个必须自己说的限制：子集重抽只能说明**在这个 roster 内部**换一批模型
  结果不变。它不能外推到未被测试的架构。真正的外推需要更多族的模型，那要
  重新训练。这个脚本回答的是「结论对 roster 的选择有多敏感」，不是「结论
  对所有模型成立」。

用法：
    python decomposition_robustness.py --root .
    python decomposition_robustness.py --root . --boot 4000
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

R = "results_clean"


# ------------------------------------------------------- 分解的两种口径 --
def ems_crossed(cells):
    """两因素交叉随机效应，期望均方。cells[(model, unit)] = [每个种子的分数]

    返回四个分量的方差（unit, model, interaction, residual）。
    与 recompute_clean.py 里的实现相同，重复一份是为了本脚本可独立运行。
    """
    M = sorted({m for m, _ in cells})
    F = sorted({f for _, f in cells})
    a, b = len(M), len(F)
    n = min(len(v) for v in cells.values())
    if a < 2 or b < 2 or n < 2:
        return None
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
    if tot <= 0:
        return None
    return dict(unit=100 * vf / tot, model=100 * va / tot,
                interaction=100 * vaf / tot, seed=100 * ve / tot)


def ss_fixed(cells):
    """model 作为固定效应。

    随机效应把 model 的方差解释成「从模型总体里抽一个」的变异；固定效应只问
    「这几个具体模型之间的差异，占观测总变异的多少」。后者不需要「模型是
    总体的样本」这个假设，所以它正是审稿人要的那个对照。

    报告的是 I 型平方和的份额，与嵌套分解同一口径。
    """
    rows = []
    for (m, f), vals in cells.items():
        for i, v in enumerate(vals):
            rows.append({"model": m, "unit": f, "seed": i, "y": v})
    d = pd.DataFrame(rows)
    grand = d.y.mean()
    sst = ((d.y - grand) ** 2).sum()
    mu = d.groupby("unit").y.transform("mean")
    mm = d.groupby("model").y.transform("mean")
    cell = d.groupby(["model", "unit"]).y.transform("mean")
    ss_unit = ((mu - grand) ** 2).sum()
    ss_model = ((mm - grand) ** 2).sum()
    ss_inter = ((cell - mu - mm + grand) ** 2).sum()
    ss_res = sst - ss_unit - ss_model - ss_inter
    return dict(unit=100 * ss_unit / sst, model=100 * ss_model / sst,
                interaction=100 * ss_inter / sst, seed=100 * ss_res / sst)


def nested_ss(df, metric, unit="group", fold="fold"):
    """unit nested in fold，平方和份额。用于一折含多个单元的三个数据集。"""
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


# ---------------------------------------------------------------- 数据 --
def durian_cells(root, models=None):
    """durian：一折即一农场，可用交叉分解。"""
    a = pd.read_csv(f"{root}/{R}/durian_in_region_clean.csv",
                    encoding="utf-8-sig")
    b = pd.read_csv(f"{root}/{R}/durian_in_region_clean_yolo11l.csv",
                    encoding="utf-8-sig")
    c = pd.read_csv(f"{root}/{R}/durian_in_region_clean_frcnn.csv",
                    encoding="utf-8-sig").assign(eval_on="outer")
    keep = ["model", "config", "seed", "checkpoint", "eval_on", "mAP50"]
    d = pd.concat([a[keep], b[keep], c[keep]], ignore_index=True)
    d = d[(d.checkpoint == "best") & (d.eval_on == "outer")
          & d.config.str.startswith("byfarm")]
    if models:
        d = d[d.model.isin(models)]
    cells = defaultdict(list)
    for _, r in d.iterrows():
        cells[(r["model"], r["config"])].append(float(r["mAP50"]))
    return cells


def nested_frames(root):
    return {
        "gwhd": (pd.read_csv(f"{root}/{R}/gwhd_by_domain_clean.csv",
                             encoding="utf-8-sig"), "mAP50"),
        "breakhis": (pd.read_csv(f"{root}/{R}/breakhis_by_patient_clean.csv",
                                 encoding="utf-8-sig"), "top1"),
        "har": (pd.read_csv(f"{root}/results_har/har_by_subject.csv"),
                "macro_f1")}


# ------------------------------------------------------------ [1] 区间 --
def bootstrap_intervals(root, boot, rng):
    """对单元重抽，给每个分量一个 90% 百分位区间。

    重抽的层级是单元（durian 是农场，其余是 fold 内的单元），因为那是设计上
    可交换的层级。种子和模型不重抽：它们不是从总体里抽来的。
    """
    print("=" * 78)
    print("[1] 方差分量的 90% bootstrap 区间（对单元重抽）")
    print("=" * 78)
    out = []

    cells = durian_cells(root, models=[m for m in
                                       {k[0] for k in durian_cells(root)}
                                       if m != "frcnn-r50"])
    farms = sorted({f for _, f in cells})
    point = ems_crossed(cells)
    acc = defaultdict(list)
    for _ in range(boot):
        pick = rng.choice(farms, len(farms), replace=True)
        cb = {}
        for i, f in enumerate(pick):
            for m in {k[0] for k in cells}:
                cb[(m, f"b{i}")] = cells[(m, f)]
        r = ems_crossed(cb)
        if r:
            for k, v in r.items():
                acc[k].append(v)
    print(f"\n  durian (crossed, {len(farms)} farms, "
          f"{len({k[0] for k in cells})} models)")
    for k in ("unit", "model", "interaction", "seed"):
        lo, hi = np.percentile(acc[k], [5, 95])
        print(f"    {k:<14}{point[k]:6.1f}%   [{lo:5.1f}, {hi:5.1f}]")
        out.append(dict(dataset="durian", component=k,
                        point=round(point[k], 2), lo=round(float(lo), 2),
                        hi=round(float(hi), 2), design="crossed"))

    for ds, (df, met) in nested_frames(root).items():
        folds = sorted(df.fold.unique())
        point = nested_ss(df, met)
        acc = defaultdict(list)
        for _ in range(boot):
            pick = rng.choice(folds, len(folds), replace=True)
            parts = []
            for i, f in enumerate(pick):
                s = df[df.fold == f].copy()
                s["fold"] = f"b{i}"
                parts.append(s)
            r = nested_ss(pd.concat(parts), met)
            if r:
                for k, v in r.items():
                    acc[k].append(v)
        print(f"\n  {ds} (nested, {len(folds)} folds, "
              f"{df.group.nunique()} units, {df.model.nunique()} models)")
        for k in ("unit", "fold", "model", "seed", "residual"):
            lo, hi = np.percentile(acc[k], [5, 95])
            print(f"    {k:<14}{point[k]:6.1f}%   [{lo:5.1f}, {hi:5.1f}]")
            out.append(dict(dataset=ds, component=k,
                            point=round(point[k], 2), lo=round(float(lo), 2),
                            hi=round(float(hi), 2), design="nested"))
    return out


# ------------------------------------------------- [2] model 固定效应 --
def fixed_effect(root):
    """把 model 当固定效应重算，与随机效应并排。

    随机效应的 model 分量依赖「这些模型是总体的样本」这个假设；固定效应不
    依赖它，只问这几个具体模型之间差异占多少。两者接近，说明那个假设不是
    结论的来源。
    """
    print("\n" + "=" * 78)
    print("[2] model 作为固定效应，与随机效应并排")
    print("=" * 78)
    out = []
    cells = durian_cells(root, models=[m for m in
                                       {k[0] for k in durian_cells(root)}
                                       if m != "frcnn-r50"])
    rand = ems_crossed(cells)
    fix = ss_fixed(cells)
    print(f"\n  durian, 5 models x 8 farms x 5 seeds")
    print(f"    {'component':<14}{'random':>9}{'fixed':>9}{'diff':>8}")
    for k in ("unit", "model", "interaction", "seed"):
        print(f"    {k:<14}{rand[k]:8.1f}%{fix[k]:8.1f}%"
              f"{fix[k]-rand[k]:+8.1f}")
        out.append(dict(dataset="durian", component=k,
                        random_effect=round(rand[k], 2),
                        fixed_effect=round(fix[k], 2)))
    # 加入 frcnn 的六模型版本（种子不平衡，只能用固定效应的平方和）
    cells6 = durian_cells(root)
    fix6 = ss_fixed(cells6)
    print(f"\n  durian, 6 models incl. Faster R-CNN (fixed effect only, "
          f"unbalanced seeds)")
    for k in ("unit", "model", "interaction", "seed"):
        print(f"    {k:<14}{fix6[k]:8.1f}%")
        out.append(dict(dataset="durian6", component=k,
                        random_effect=None, fixed_effect=round(fix6[k], 2)))
    print("\n  嵌套设计的三个数据集本来就用平方和份额报告，"
          "不依赖随机效应假设。")
    return out


# --------------------------------------------- [3] 模型集合的敏感性 --
def roster_subsampling(root, rng):
    """durian：从六个配置里抽 2 到 5 个，重算分解。

    这是审稿意见里最具体的一条——68–88% 会不会是因为模型集合太窄。durian
    有六个配置横跨 YOLO、DETR、R-CNN 三个族，所以可以直接问：换一批模型，
    单元份额还在不在同一量级。

    ★ 它回答的是「结论对 roster 的选择有多敏感」，不是「结论对所有架构成立」。
      后者需要训练更多族的模型。
    """
    print("\n" + "=" * 78)
    print("[3] 模型集合子集重抽（durian，六个配置）")
    print("=" * 78)
    all_cells = durian_cells(root)
    models = sorted({k[0] for k in all_cells})
    FAM = {"yolo11n": "YOLO", "yolo11s": "YOLO", "yolo11m": "YOLO",
           "yolo11l": "YOLO", "rtdetr-l": "DETR", "frcnn-r50": "R-CNN"}
    print(f"  配置: " + ", ".join(f"{m} ({FAM[m]})" for m in models))
    out = []
    print(f"\n  {'k models':<10}{'subsets':>9}{'unit %':>22}{'model %':>20}")
    for k in range(2, len(models) + 1):
        us, ms, nfam = [], [], []
        for combo in itertools.combinations(models, k):
            sub = {kk: v for kk, v in all_cells.items() if kk[0] in combo}
            r = ss_fixed(sub)          # 固定效应：种子数不等也能用
            us.append(r["unit"])
            ms.append(r["model"])
            nfam.append(len({FAM[m] for m in combo}))
            out.append(dict(k_models=k, models="+".join(combo),
                            families=len({FAM[m] for m in combo}),
                            unit=round(r["unit"], 2),
                            model=round(r["model"], 2)))
        print(f"  {k:<10}{len(us):>9}   {np.mean(us):6.1f}  "
              f"[{min(us):5.1f}, {max(us):5.1f}]   {np.mean(ms):6.1f}  "
              f"[{min(ms):5.1f}, {max(ms):5.1f}]")
    df = pd.DataFrame(out)
    print("\n  按包含的架构族数分组：")
    print(f"    {'families':<10}{'subsets':>9}{'mean unit %':>14}"
          f"{'mean model %':>15}{'max model %':>14}")
    for f in sorted(df.families.unique()):
        g = df[df.families == f]
        print(f"    {f:<10}{len(g):>9}{g.unit.mean():>14.1f}"
              f"{g.model.mean():>15.1f}{g.model.max():>14.1f}")
    print(f"\n  在全部 {len(df)} 个子集里，单元份额的范围是 "
          f"{df.unit.min():.1f}–{df.unit.max():.1f}%，"
          f"模型份额是 {df.model.min():.1f}–{df.model.max():.1f}%。")
    worst = df.loc[df.model.idxmax()]
    print(f"  模型份额最大的子集：{worst.models} "
          f"({int(worst.families)} 族)  unit {worst.unit:.1f}%  "
          f"model {worst.model:.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.root, R), exist_ok=True)

    ci = bootstrap_intervals(a.root, a.boot, rng)
    fx = fixed_effect(a.root)
    rs = roster_subsampling(a.root, rng)

    for name, rows in (("decomposition_ci", ci),
                       ("decomposition_fixed", fx),
                       ("decomposition_roster", rs)):
        p = os.path.join(a.root, R, f"summary_{name}.csv")
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n写出 {p}  ({len(rows)} 行)")

    print("\n" + "=" * 78)
    print("读法")
    print("=" * 78)
    print("  [1] 区间宽，说明分量估得不精确；但只要下界仍远高于 model 分量，")
    print("      「单元主导」这句话就站得住，只是不能说成一个精确的百分比。")
    print("  [2] 固定效应与随机效应接近，说明结论不依赖「模型是总体的样本」")
    print("      这个假设——那正是审稿意见质疑的地方。")
    print("  [3] 子集之间单元份额若都在同一量级，说明 68–88% 不是模型集合")
    print("      太窄造成的。但这只覆盖已训练的六个配置，不能外推到未测架构。")


if __name__ == "__main__":
    sys.exit(main())
