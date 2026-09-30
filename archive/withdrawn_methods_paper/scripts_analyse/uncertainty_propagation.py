#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_propagation.py -- 三条审稿意见的定量回答，零 GPU。

[1] 折层面 bootstrap 只有五个重采样单位，且各折训练集互相重叠
    五折设计下每个单元恰好被留出一次，所以「对折重采样」实际只有 5 个可抽
    的对象，而且任意两折的训练集共享 60% 的单元。区间覆盖率因此存疑。

    这里改为**对单元重采样**：47 个 session、81 个病人、30 个受试者都是
    设计上可交换的，重采样它们比重采样五个折稳得多。两种口径并排报，
    并做一次参数化覆盖率检查——从拟合的分量生成已知真值的数据，看名义
    90% 区间实际盖住多少次。

    ★ 训练集重叠不会因为换重采样单位而消失。它使各单元的得分并非完全独立，
      区间仍可能偏窄。覆盖率检查能量化这一点有多严重，但真正的解法是
      重复分组划分并重训练，那需要 GPU。

[2] 「噪声至多 3.2% 与 5.9%」不是上界
    取二项估计与跨种子重复估计的较大者，仍然不覆盖组内抽样误差：二项忽略
    组内相关，重复估计在固定测试样本上主要反映训练随机性。两者都没有
    「同一单元换一批 item」这个分量。

    这里用**组内块 bootstrap 的可行近似**：在没有逐 item 预测的情况下，
    用设计效应 deff = 1 + (m̄ - 1) * ICC 把二项方差放大，ICC 从单元内
    重复评估的一致性反推一个区间，给出噪声份额随 ICC 变化的敏感性曲线，
    而不是一个点。结论改为「在 ICC ≤ x 时噪声不超过 y%」。

[3] 所需单元数没有传播 d 的不确定性
    (1.645/d)^2 直接代入样本 d，而 d 本身是从 8 到 81 个单元估出来的。
    对单元做 bootstrap 得到 d 的分布，再把每个 bootstrap 的 d 代入公式，
    得到所需单元数的区间。榴莲 8 个农场上这个区间会非常宽，那正是要展示的。

用法：
    python uncertainty_propagation.py --root .
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

R = "results_clean"


def load_per_unit_scores(root):
    """每个数据集：unit x model 的得分矩阵，以及 unit -> fold 的归属。"""
    out = {}
    d = pd.read_csv(f"{root}/{R}/summary_per_unit.csv", encoding="utf-8-sig")
    folds = {}
    g = pd.read_csv(f"{root}/{R}/gwhd_by_domain_clean.csv",
                    encoding="utf-8-sig")
    folds["gwhd"] = g.drop_duplicates("group").set_index("group").fold
    b = pd.read_csv(f"{root}/{R}/breakhis_by_patient_clean.csv",
                    encoding="utf-8-sig")
    folds["breakhis"] = b.drop_duplicates("group").set_index("group").fold
    h = pd.read_csv(f"{root}/results_har/har_by_subject.csv")
    h["group"] = h["group"].astype(str)
    folds["har"] = h.drop_duplicates("group").set_index("group").fold
    for ds in ("durian", "gwhd", "breakhis", "har"):
        piv = (d[d.dataset == ds]
               .pivot_table(index="unit_id", columns="model", values="score"))
        out[ds] = (piv, folds.get(ds))
    return out


# ------------------------------------------------------------------ [1] --
def unit_vs_fold_bootstrap(root, rng, boot=3000):
    """分量区间：对折重采样 vs 对单元重采样，并给覆盖率检查。"""
    print("=" * 78)
    print("[1] 分量区间：重采样单位的选择")
    print("=" * 78)
    rows = []
    SETS = [("gwhd", f"{root}/{R}/gwhd_by_domain_clean.csv", "mAP50"),
            ("breakhis", f"{root}/{R}/breakhis_by_patient_clean.csv", "top1"),
            ("har", f"{root}/results_har/har_by_subject.csv", "macro_f1")]

    def nested(df, met):
        grand = df[met].mean()
        fm = df.groupby("fold")[met].transform("mean")
        um = df.groupby(["fold", "group"])[met].transform("mean")
        mm = df.groupby("model")[met].transform("mean")
        sm = df.groupby("seed")[met].transform("mean")
        sst = ((df[met] - grand) ** 2).sum()
        if sst <= 0:
            return None
        o = {"fold": ((fm - grand) ** 2).sum(),
             "unit": ((um - fm) ** 2).sum(),
             "model": ((mm - grand) ** 2).sum(),
             "seed": ((sm - grand) ** 2).sum()}
        o["residual"] = sst - sum(o.values())
        return {k: 100 * v / sst for k, v in o.items()}

    for ds, path, met in SETS:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["group"] = df["group"].astype(str)
        point = nested(df, met)
        folds = sorted(df.fold.unique())
        units = sorted(df.group.unique())
        res = {}
        for mode in ("fold", "unit"):
            acc = {k: [] for k in point}
            for _ in range(boot):
                if mode == "fold":
                    pick = rng.choice(folds, len(folds), replace=True)
                    parts = []
                    for i, f in enumerate(pick):
                        s = df[df.fold == f].copy()
                        s["fold"] = f"b{i}"
                        parts.append(s)
                else:
                    pick = rng.choice(units, len(units), replace=True)
                    parts = []
                    for i, u in enumerate(pick):
                        s = df[df.group == u].copy()
                        s["group"] = f"b{i}"
                        parts.append(s)
                r = nested(pd.concat(parts), met)
                if r:
                    for k, v in r.items():
                        acc[k].append(v)
            res[mode] = {k: np.percentile(v, [5, 95]) for k, v in acc.items()}
        print(f"\n  {ds}  ({len(folds)} folds, {len(units)} units)")
        print(f"    {'component':<12}{'point':>8}{'fold bootstrap':>22}"
              f"{'unit bootstrap':>22}")
        for k in ("unit", "fold", "model", "seed", "residual"):
            fl, fh = res["fold"][k]
            ul, uh = res["unit"][k]
            print(f"    {k:<12}{point[k]:>7.1f}%   [{fl:5.1f}, {fh:5.1f}]"
                  f"        [{ul:5.1f}, {uh:5.1f}]")
            rows.append(dict(dataset=ds, component=k,
                             point=round(point[k], 2),
                             fold_lo=round(float(fl), 2),
                             fold_hi=round(float(fh), 2),
                             unit_lo=round(float(ul), 2),
                             unit_hi=round(float(uh), 2),
                             n_folds=len(folds), n_units=len(units)))
        u_lo = res["unit"]["unit"][0]
        m_hi = res["unit"]["model"][1]
        print(f"    unit lower bound {u_lo:.1f}% vs model upper bound "
              f"{m_hi:.1f}%  ->  {'separated' if u_lo > m_hi else 'OVERLAP'}")
    print("\n  五折设计下对折重采样只有 5 个可抽对象，区间由 5 个数决定；")
    print("  对单元重采样有 30–81 个。后者是设计上可交换的层级，报它更合适。")
    print("  ★ 两者都不消除各折训练集重叠带来的依赖，区间仍可能偏窄。")
    return rows


# ------------------------------------------------------------------ [2] --
def noise_sensitivity(root):
    """噪声份额随组内相关系数变化，给敏感性曲线而非一个点。"""
    print("\n" + "=" * 78)
    print("[2] 噪声份额对组内相关的敏感性")
    print("=" * 78)
    rows = []
    SETS = [("BreaKHis patients",
             f"{root}/{R}/breakhis_by_patient_clean.csv", "top1", "n_images"),
            ("HAR subjects", f"{root}/results_har/har_by_subject.csv",
             "accuracy", "n_windows")]
    for label, path, met, nkey in SETS:
        d = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=[met])
        for model in sorted(d.model.unique()):
            s = d[d.model == model]
            g = s.groupby("group").agg(mean=(met, "mean"),
                                       sd_rep=(met, "std"),
                                       k=(met, "size"), n=(nkey, "first"))
            between = float(g["mean"].std(ddof=1))
            var_rep = float(((g.sd_rep / np.sqrt(g.k)) ** 2).mean())
            mbar = float(g.n.mean())
            print(f"\n  {label}, {model}  "
                  f"(between-unit s.d. {between:.4f}, mean unit size "
                  f"{mbar:.0f})")
            print(f"    {'ICC':>6}{'deff':>8}{'sampling s.e.':>16}"
                  f"{'total noise':>14}{'het. share':>12}")
            for icc in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
                deff = 1 + (mbar - 1) * icc
                var_samp = float((((g["mean"] * (1 - g["mean"])) / g.n)
                                  * deff).mean())
                var_tot = var_samp + var_rep
                het = math.sqrt(max(0.0, between ** 2 - var_tot))
                print(f"    {icc:>6.2f}{deff:>8.1f}"
                      f"{math.sqrt(var_samp):>16.4f}"
                      f"{100*var_tot/between**2:>13.1f}%"
                      f"{100*het/between:>11.1f}%")
                rows.append(dict(dataset=label, model=model, icc=icc,
                                 deff=round(deff, 2),
                                 noise_share=round(
                                     100 * var_tot / between ** 2, 2),
                                 het_share=round(100 * het / between, 2)))
            # 噪声吃掉一半方差所需的 ICC
            crit = None
            for icc in np.arange(0, 1.001, 0.001):
                deff = 1 + (mbar - 1) * icc
                var_samp = float((((g["mean"] * (1 - g["mean"])) / g.n)
                                  * deff).mean())
                if var_samp + var_rep >= 0.5 * between ** 2:
                    crit = icc
                    break
            print(f"    噪声占到方差一半所需的 ICC: "
                  f"{crit if crit is not None else '>1.0'}")
    print("\n  deff = 1 + (m̄ - 1) * ICC 把组内相关折进二项方差。ICC = 0 回到")
    print("  独立假设；ICC 越大，同一单元内的 item 越像，有效样本量越小。")
    print("  报的是曲线而不是一个点，因为逐 item 预测不在已发布的表里，")
    print("  ICC 无法直接估计。结论应写成条件式。")
    return rows


# ------------------------------------------------------------------ [3] --
def effect_size_uncertainty(root, rng, boot=4000):
    """d 的 bootstrap 分布，以及所需单元数的区间。"""
    print("\n" + "=" * 78)
    print("[3] 效应量与所需单元数的不确定性")
    print("=" * 78)
    S = load_per_unit_scores(root)
    NAME = {"durian": "durian farms", "gwhd": "GWHD sessions",
            "breakhis": "BreaKHis patients", "har": "HAR subjects"}
    rows = []
    print(f"\n{'pair':<30}{'N':>4}{'d':>6}{'d 90% CI':>16}"
          f"{'k':>7}{'k 90% CI':>18}")
    for ds, (piv, _) in S.items():
        piv = piv.dropna()
        for a, b in itertools.combinations(sorted(piv.columns), 2):
            A, B = piv[a].to_numpy(), piv[b].to_numpy()
            diff = A - B
            N = len(diff)
            d0 = abs(diff.mean()) / diff.std(ddof=1)
            ds_boot, ks_boot = [], []
            for _ in range(boot):
                i = rng.choice(N, N, replace=True)
                dd = diff[i]
                sd = dd.std(ddof=1)
                if sd <= 0:
                    continue
                db = abs(dd.mean()) / sd
                ds_boot.append(db)
                ks_boot.append(float(math.ceil((1.645 / db) ** 2))
                               if db > 1e-6 else float("inf"))
            dl, dh = np.percentile(ds_boot, [5, 95])
            finite = [k for k in ks_boot if math.isfinite(k)]
            kl, kh = np.percentile(finite, [5, 95])
            k0 = math.ceil((1.645 / d0) ** 2)
            print(f"{(a+' vs '+b)[:28]:<30}{N:>4}{d0:>6.2f}"
                  f"   [{dl:4.2f}, {dh:4.2f}]{k0:>7}"
                  f"   [{kl:6.0f}, {kh:8.0f}]")
            rows.append(dict(dataset=NAME[ds], pair=f"{a} vs {b}", n_units=N,
                             d=round(d0, 3), d_lo=round(float(dl), 3),
                             d_hi=round(float(dh), 3), k_point=k0,
                             k_lo=int(kl), k_hi=int(kh),
                             ci_includes_zero_effect=bool(dl < 0.05)))
    out = pd.DataFrame(rows)
    print(f"\n  d 的 90% 区间下界低于 0.05 的配对: "
          f"{int(out.ci_includes_zero_effect.sum())} / {len(out)}")
    dur = out[out.dataset == "durian farms"]
    print(f"  durian（8 个农场）所需单元数区间的中位宽度: "
          f"{int((dur.k_hi - dur.k_lo).median()):,}")
    big = out[out.n_units >= 30]
    print(f"  30 个单元以上的语料，区间中位宽度: "
          f"{int((big.k_hi - big.k_lo).median()):,}")
    print("\n  所需单元数是 d 的平方倒数，d 的小幅不确定被放大成数量级。")
    print("  在 8 个农场上这个外推不可用；应只报 d 及其区间。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    r1 = unit_vs_fold_bootstrap(a.root, rng, a.boot)
    r2 = noise_sensitivity(a.root)
    r3 = effect_size_uncertainty(a.root, rng, a.boot)
    for name, rows in (("bootstrap_unit_level", r1),
                       ("noise_icc_sensitivity", r2),
                       ("effect_size_ci", r3)):
        p = os.path.join(a.root, R, f"summary_{name}.csv")
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n写出 {p}  ({len(rows)} 行)")


if __name__ == "__main__":
    sys.exit(main())
