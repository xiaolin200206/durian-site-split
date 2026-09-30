#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_unit_curve.py -- 评价单元的 dose-response，不需要重新训练。

现有的 dose-response 改变的是「训练数据来自多少个单元」。论文标题问的却是
「评价一个分数需要多少个独立单元」。这是两个问题，而第二个可以完全从已有
的 per-unit 表算出来：

    每个单元的分数已经是「没见过它的权重」打出来的。
    从这些单元里重复抽 k 个、按实例加权重算聚合分，
    就得到「如果当初只在 k 个单元上评估会报出什么」的分布。

跑出来的规律与训练单元那条曲线不同，这正是它值得单独做一张图的原因：

    训练单元 ↑ : 均值可能升（durian +61.5%）也可能不动（GWHD），离散度降
    评价单元 ↑ : 等权聚合下均值不动，只有离散度按 1/√k 降；实例加权下均值会
                 从等权平均漂向 pooled（单元大小与分数相关时才明显）

★ 一个必须写进 Methods 的限制
   这些单元的分数来自不同的折（每个单元由留出它的那一折打分），所以重抽
   混合了不同的训练集。它是「只用 k 个单元评估会得到什么」的近似，不是
   精确复制。若要精确，需要固定一个训练集再重抽评价单元，那就要重训。

用法：
    python eval_unit_curve.py --root .
    python eval_unit_curve.py --root . --target 0.10   # 反解需要多少单元
"""

import argparse
import collections
import csv
import os
import random
import statistics as st

# (显示名, 相对路径, 单元列, 分数列, 权重列, 模型列, 模型值)
DATASETS = [
    ("Durian bursts", "results_durian/peninsula_by_burst_60s_min5.csv",
     "burst", "mAP50", "n_images", "model", "yolo11s"),
    ("GWHD sessions", "results_gwhd/gwhd_by_domain.csv",
     "group", "mAP50", "n_images", "model", "yolo11s"),
    ("BreaKHis patients", "results_breakhis/breakhis_by_patient.csv",
     "group", "top1", "n_images", "model", "yolo11s-cls"),
    ("HAR subjects", "results_har/har_by_subject.csv",
     "group", "macro_f1", "n_windows", "model", "mlp256"),
]


def load(path, unit, score, weight, mkey, mval):
    """单元 -> (跨种子平均分, 实例数)。"""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig"))
            if r[mkey] == mval]
    by = collections.defaultdict(list)
    w = {}
    for r in rows:
        try:
            s = float(r[score])
        except (ValueError, KeyError):
            continue          # 该单元缺该类/该指标，跳过
        by[r[unit]].append(s)
        w[r[unit]] = float(r[weight])
    return {u: (st.mean(v), w[u]) for u, v in by.items()}


def aggregate(d, units, weight):
    """weight='items' 实例加权（等同于一次 pooled 报告）；'equal' 等权平均。"""
    if weight == "equal":
        return sum(d[u][0] for u in units) / len(units)
    num = sum(d[u][0] * d[u][1] for u in units)
    den = sum(d[u][1] for u in units)
    return num / den


def curve(d, ks, B, rng, weight):
    units = list(d)
    out = []
    for k in ks:
        vals = sorted(aggregate(d, rng.sample(units, k), weight) for _ in range(B))
        sd = st.stdev(vals) if k < len(units) and len(set(vals)) > 1 else 0.0
        p5, p95 = vals[int(.05 * B)], vals[int(.95 * B)]
        out.append((k, st.mean(vals), sd, p5, p95, p95 - p5))
    return out


def units_needed(d, target, B, rng, weight):
    """反解：90% 区间宽度降到 target 需要多少个评价单元。"""
    units = list(d)
    for k in range(1, len(units) + 1):
        vals = sorted(aggregate(d, rng.sample(units, k), weight) for _ in range(B))
        if vals[int(.95 * B)] - vals[int(.05 * B)] <= target:
            return k
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=float, nargs="*", default=[0.10, 0.05],
                   help="90%% 区间宽度目标，反解所需单元数")
    p.add_argument("--weight", choices=["items", "equal", "both"],
                   default="both",
                   help="items=实例加权（等同 pooled 报告）；equal=等权平均")
    p.add_argument("--out", default="results_eval_unit_curve.csv")
    a = p.parse_args()
    rng = random.Random(a.seed)

    rows_out = []
    for name, rel, unit, score, weight, mkey, mval in DATASETS:
        path = os.path.join(a.root, rel)
        if not os.path.isfile(path):
            print(f"缺表，跳过: {rel}")
            continue
        d = load(path, unit, score, weight, mkey, mval)
        N = len(d)
        ks = sorted(set([k for k in (1, 2, 4, 8, 16, 32, 64) if k <= N] + [N]))
        pooled_all = aggregate(d, list(d), "items")
        equal_all = aggregate(d, list(d), "equal")
        print(f"\n{name} ({mval}): {N} 个单元, "
              f"pooled {pooled_all:.3f}, 等权 {equal_all:.3f}, "
              f"差 {pooled_all - equal_all:+.3f}")
        modes = ["items", "equal"] if a.weight == "both" else [a.weight]
        for mode in modes:
            label = "实例加权" if mode == "items" else "等权"
            print(f"  [{label}]")
            print(f"  {'k_eval':>7}{'mean':>8}{'sd':>8}{'p5':>8}"
                  f"{'p95':>8}{'w90':>8}")
            for k, m, sd, p5, p95, w in curve(d, ks, a.boot, rng, mode):
                print(f"  {k:>7}{m:>8.3f}{sd:>8.3f}{p5:>8.3f}"
                      f"{p95:>8.3f}{w:>8.3f}")
                rows_out.append([name, mval, mode, N, k, round(m, 4),
                                 round(sd, 4), round(p5, 4), round(p95, 4),
                                 round(w, 4)])
            for t in a.target:
                k = units_needed(d, t, min(a.boot, 3000), rng, mode)
                print(f"    90% 区间宽度 <= {t:.2f} 需要 "
                      f"{k if k else '>' + str(N)} 个评价单元")

    out = os.path.join(a.root, a.out)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "weighting", "n_units", "k_eval",
                    "mean", "sd_across_draws", "p5", "p95", "width90"])
        w.writerows(rows_out)
    print(f"\n写出 {out}")
    print("\n读法：")
    print("  等权那张表的均值在每个 k 上都无偏，应当基本不动；只有离散度按 1/sqrt(k) 下降。")
    print("  实例加权那张表的均值会从「等权平均」漂向「pooled」，因为 k=1 时抽一个单元")
    print("  就是等权，k=N 时才是 pooled。漂移量恰好等于上面那行的差值，单元大小与分数")
    print("  相关时才明显（GWHD 大 session 分高，durian 大 burst 分低）。这是加权方式的")
    print("  算术后果，不是 bug；等权表的均值若也漂，那才要查权重列和单元-折对应。")


if __name__ == "__main__":
    main()
