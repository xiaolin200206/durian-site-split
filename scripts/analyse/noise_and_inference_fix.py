#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_and_inference_fix.py -- 处理两条审稿意见，零 GPU。

审稿意见（NCS 视角，第二份）指出两处方法与主张不匹配：

[A] 单元内噪声的估计与实际指标不符
    稿子在 HAR 上报 macro F1，却把两个分类数据集的单元得分都当二项均值做
    噪声分解。macro F1 不是正确分类比例，方差不是 p(1-p)/n；而且即便换成
    准确率，同一单元内的观测也不独立（相邻窗口、同一病人的相邻切片），
    二项方差会系统性低估噪声。原来的「97–99% 是真实异质性」因此没有被
    所述方法支撑。

    这里改为：
      - HAR 明确改用 accuracy 并注明，macro F1 不再用于噪声分解
      - 两个数据集都用 seed-level 的重复观测直接估计单元内噪声，不假设
        任何分布形式。每个单元有 (模型 x 种子) 次独立评估，其跨种子方差
        就是「同一单元重复测量」的方差，这条路径不需要 p(1-p)/n
      - 同时保留二项估计作为对照，并报告两者差多少，让读者看到独立性
        假设值多少

[B] 有限样本池的排序稳定性 vs 对新单位总体的推断
    重抽是无放回的，抽满全池时反转率必为零，所以「某对需要 63 个 session
    而只有 47 个」这种说法混用了两套定义。分开报：
      - pool_flip_k：在现有池内抽 k 个单元时的反转率（描述性，可直接算）
      - k_for_5pct_infinite：若把这些单元视为来自某个总体的独立抽样，
        达到 5% 反转率所需的单元数，(1.645/d)^2，明确标注为对总体的推断
        且不含有限总体修正
      - 另给一个带有限总体修正的版本，说明在现有池内永远达不到的情形

用法：
    python noise_and_inference_fix.py --root .
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

R = "results_clean"


# ------------------------------------------------------------------ [A] --
def within_unit_noise(root, rng, boot=3000):
    """两种估计并排：跨种子重复观测，和二项假设。

    跨种子那条是主估计。同一单元被 (模型 x 种子) 次独立训练的权重各评一次，
    这些评估之间的差异就是重复测量的噪声，不需要对单元内部的样本结构做
    任何假设——相关也好、成块也好，都已经包含在里面。

    ★ 它测的是「同一单元重复评估的不稳定性」，来源是训练随机性而非抽样。
      单元内部由于样本量有限带来的抽样噪声，它捕捉不到。所以两种估计是
      互补的下界与上界，都报出来才诚实。
    """
    print("=" * 78)
    print("[A] 单元内噪声：跨种子重复 vs 二项假设")
    print("=" * 78)
    rows = []
    SETS = [
        ("BreaKHis patients", f"{root}/{R}/breakhis_by_patient_clean.csv",
         "top1", "n_images", "top-1 accuracy"),
        ("HAR subjects", f"{root}/results_har/har_by_subject.csv",
         "accuracy", "n_windows", "accuracy"),
    ]
    for label, path, metric, nkey, metric_name in SETS:
        if not os.path.isfile(path):
            print(f"  缺表，跳过 {label}")
            continue
        d = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=[metric])
        for model in sorted(d.model.unique()):
            s = d[d.model == model]
            g = s.groupby("group").agg(mean=(metric, "mean"),
                                       sd_rep=(metric, "std"),
                                       k=(metric, "size"),
                                       n=(nkey, "first"))
            g["se_rep"] = g.sd_rep / np.sqrt(g.k)      # 单元均值的重复测量误差
            between = float(g["mean"].std(ddof=1))
            # 主估计：重复测量
            var_rep = float((g.se_rep ** 2).mean())
            het_rep = math.sqrt(max(0.0, between ** 2 - var_rep))
            # 对照：二项
            var_bin = float(((g["mean"] * (1 - g["mean"])) / g.n).mean())
            het_bin = math.sqrt(max(0.0, between ** 2 - var_bin))
            rows.append(dict(
                dataset=label, model=model, metric=metric_name,
                n_units=len(g), reps_per_unit=int(g.k.median()),
                sd_between=round(between, 4),
                se_repeat=round(math.sqrt(var_rep), 4),
                se_binomial=round(math.sqrt(var_bin), 4),
                noise_share_repeat=round(100 * var_rep / between ** 2, 2),
                noise_share_binomial=round(100 * var_bin / between ** 2, 2),
                het_share_repeat=round(100 * het_rep / between, 2),
                het_share_binomial=round(100 * het_bin / between, 2)))
            print(f"\n  {label}, {model}  ({metric_name}, {len(g)} units, "
                  f"{int(g.k.median())} repeats each)")
            print(f"    between-unit s.d.            {between:.4f}")
            print(f"    repeat-measurement s.e.      "
                  f"{math.sqrt(var_rep):.4f}   "
                  f"({100*var_rep/between**2:.1f}% of variance)")
            print(f"    binomial s.e. (independence) "
                  f"{math.sqrt(var_bin):.4f}   "
                  f"({100*var_bin/between**2:.1f}% of variance)")
            print(f"    heterogeneity share          "
                  f"{100*het_rep/between:.1f}% (repeat)   "
                  f"{100*het_bin/between:.1f}% (binomial)")
    print("\n  读法：两种估计捕捉不同的噪声来源。重复测量法包含训练随机性")
    print("        但不含单元内抽样；二项法相反，且假设单元内观测独立，")
    print("        在近重复的切片或相邻窗口上会低估。真实噪声不小于两者")
    print("        的较大者，所以异质性份额的保守读法取两者较小的那个。")
    return rows


# ------------------------------------------------------------------ [B] --
def pool_vs_population(root, rng):
    """把「池内稳定性」与「对总体的推断」分开报。"""
    print("\n" + "=" * 78)
    print("[B] 池内排序稳定性 vs 对新单位总体的推断")
    print("=" * 78)
    p = pd.read_csv(f"{root}/{R}/summary_model_pairs.csv",
                    encoding="utf-8-sig")
    rows = []
    for _, r in p.iterrows():
        d, N = float(r.d), int(r.n_units)
        # 池内：现有 k 的反转率已在 summary 里；补一个「池内最大 k」
        # 无放回抽满全池必为 0，所以池内定义下不存在「所需单元数」
        k_inf = math.ceil((1.645 / d) ** 2) if d > 0 else None
        # 有限总体修正：var_subset = (1 - k/N) * s^2 / k
        # 反转率 = Phi(-|mean| / sqrt(var_subset))，解 k
        k_fpc = None
        for k in range(2, N + 1):
            se = math.sqrt(max(1e-12, (1 - k / N)) * 1.0 / k)
            if se > 0 and d / se >= 1.645:
                k_fpc = k
                break
        rows.append(dict(
            dataset=r.dataset, pair=r.pair, n_units=N, d=round(d, 3),
            pool_flip_at_half=r.get(f"flip_k{max([k for k in (2,4,6,10,16,32) if k <= N//2] or [2])}"),
            k_for_5pct_population=k_inf,
            k_for_5pct_within_pool=k_fpc,
            reachable_in_pool=(k_fpc is not None and k_fpc <= N)))
    out = pd.DataFrame(rows).sort_values("d", ascending=False)
    print(f"\n{'pair':<30}{'N':>4}{'d':>6}{'pop k':>8}{'pool k':>8}{'reach':>7}")
    for _, r in out.iterrows():
        pk = r.k_for_5pct_population
        wk = r.k_for_5pct_within_pool
        print(f"{r.pair[:28]:<30}{r.n_units:>4}{r.d:>6.2f}"
              f"{(pk if pk else '-'):>8}{(wk if wk else '-'):>8}"
              f"{'yes' if r.reachable_in_pool else 'no':>7}")
    n_pool = int(out.reachable_in_pool.sum())
    n_pop = int((out.k_for_5pct_population <= out.n_units).sum())
    print(f"\n  在现有池内即可达到 5% 反转率的配对: {n_pool} / {len(out)}")
    print(f"  若视为对总体的推断，现有单元数已足够的配对: {n_pop} / {len(out)}")
    print("\n  两个数字定义不同，不能混用：")
    print("    池内   —— 无放回抽 k 个现有单元，抽满必为 0，问的是")
    print("              「换一批现有单元，结论会不会变」")
    print("    总体   —— 把现有单元当作某总体的独立抽样，问的是")
    print("              「要多少个新单元，结论才稳」，不含有限总体修正")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    noise = within_unit_noise(a.root, rng)
    inf = pool_vs_population(a.root, rng)
    for name, rows in (("within_unit_noise_v2", noise),
                       ("pool_vs_population", inf)):
        p = os.path.join(a.root, R, f"summary_{name}.csv")
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n写出 {p}  ({len(rows)} 行)")


if __name__ == "__main__":
    sys.exit(main())
