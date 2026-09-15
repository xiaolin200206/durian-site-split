#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
variance_decomposition.py -- 报告值的方差有多少归模型，多少归评估单元？

这是把 "site variation is larger than seed variation" 从几个标准差比值
升级成一个可辩护的测量学结论的那一步。

设计: 每个架构 a 在每个 leave-one-farm-out 折 f 上训练 5 个种子 s，
完全交叉，所以交互项可辨识:

    y_afs = mu + A_a + F_f + (AF)_af + eps_s

用 ANOVA 型的期望均方 (EMS) 估计各随机效应的方差成分。全部当随机效应，
因为问题不是 "yolo11s 比 rtdetr 好多少"，而是 "架构这个因子贡献了多少
可变性，相对于农场这个因子"。

同时报告三件 ANOVA 数字之外的事:
  - 各架构下农场排名的一致性 (Spearman)。若排名保持，站点失败是站点的
    属性而不是模型的属性，这比方差比更有说服力。
  - 随机划分与 by-farm 的差距在每个架构上分别是多少。
  - 每个方差成分的 bootstrap 区间 (对农场重抽样)。

用法:
    python variance_decomposition.py --dir results_v2
    python variance_decomposition.py --dir results_v2 --boot 2000
"""

import argparse
import csv
import math
import os
import random
import statistics as st
from collections import defaultdict


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for j, i in enumerate(order):
            r[i] = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) *
                    sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def decompose(cells):
    """cells[(a, f)] = [y_1..y_n]，平衡设计。

    两因子随机效应交叉模型的期望均方:
        E[MS_A]  = var_e + n*var_AF + n*b*var_A
        E[MS_F]  = var_e + n*var_AF + n*a*var_F
        E[MS_AF] = var_e + n*var_AF
        E[MS_e]  = var_e
    """
    A = sorted({a for a, _ in cells})
    F = sorted({f for _, f in cells})
    a, b = len(A), len(F)
    n = min(len(v) for v in cells.values())
    if n < 2 or a < 2 or b < 2:
        return None

    y = {k: v[:n] for k, v in cells.items()}
    grand = st.mean([x for v in y.values() for x in v])
    cell = {k: st.mean(v) for k, v in y.items()}
    mA = {i: st.mean([cell[(i, j)] for j in F]) for i in A}
    mF = {j: st.mean([cell[(i, j)] for i in A]) for j in F}

    ss_a = n * b * sum((mA[i] - grand) ** 2 for i in A)
    ss_f = n * a * sum((mF[j] - grand) ** 2 for j in F)
    ss_af = n * sum((cell[(i, j)] - mA[i] - mF[j] + grand) ** 2
                    for i in A for j in F)
    ss_e = sum((x - cell[(i, j)]) ** 2
               for i in A for j in F for x in y[(i, j)])

    ms_a, ms_f = ss_a / (a - 1), ss_f / (b - 1)
    ms_af = ss_af / ((a - 1) * (b - 1))
    ms_e = ss_e / (a * b * (n - 1))

    v_e = ms_e
    v_af = max(0.0, (ms_af - ms_e) / n)
    v_a = max(0.0, (ms_a - ms_af) / (n * b))
    v_f = max(0.0, (ms_f - ms_af) / (n * a))
    return {"architecture": v_a, "farm": v_f,
            "architecture x farm": v_af, "seed (residual)": v_e,
            "_dims": (a, b, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_v2")
    ap.add_argument("--file", default="in_region.csv")
    ap.add_argument("--metric", default="mAP50")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    rows = load(args.dir, args.file)
    if rows is None:
        print(f"找不到 {os.path.join(args.dir, args.file)}")
        return 1
    if "model" not in rows[0]:
        print("表里没有 model 列 —— 这是单架构的 v1 表，无法分解。")
        return 1

    fold = defaultdict(list)
    rnd = defaultdict(list)
    for r in rows:
        v = fnum(r, args.metric)
        if v is None:
            continue
        m, c = r["model"], r["config"]
        if c.startswith("byfarm"):
            fold[(m, c)].append(v)
        elif c == "random":
            rnd[m].append(v)

    models = sorted({m for m, _ in fold})
    folds = sorted({c for _, c in fold})
    print("=" * 72)
    print(f"metric: {args.metric}    architectures: {models}")
    print(f"folds: {len(folds)}")
    have = [(m, c) for m in models for c in folds if (m, c) in fold]
    miss = [(m, c) for m in models for c in folds if (m, c) not in fold]
    reps = sorted({len(fold[k]) for k in have})
    print(f"cells present: {len(have)}/{len(models)*len(folds)}   "
          f"seeds per cell: {reps}")
    if miss:
        print(f"★ 缺失的格子 ({len(miss)}): {miss[:8]}")
        print("  完全交叉尚未跑满，下面的分解只用已有架构。")
        models = [m for m in models
                  if all((m, c) in fold for c in folds)]
        print(f"  可用架构: {models}")
    if len(models) < 2:
        print("\n只有一个架构跑满全部折，无法分解 architecture 分量。")
        print("跑: colab_run_v2.py --step train --models yolo11n --yes")
        print("    colab_run_v2.py --step train --models rtdetr-l --yes")
        return 1

    cells = {(m, c): fold[(m, c)] for m in models for c in folds}

    # ---------------- 主表 ----------------
    print("\n" + "=" * 72)
    print("[1] 各架构 x 各折的均值")
    print("=" * 72)
    print(f"  {'fold':<16}" + "".join(f"{m:>12}" for m in models))
    for c in folds:
        print(f"  {c:<16}" +
              "".join(f"{st.mean(cells[(m,c)]):>12.4f}" for m in models))
    print(f"  {'mean':<16}" +
          "".join(f"{st.mean([st.mean(cells[(m,c)]) for c in folds]):>12.4f}"
                  for m in models))
    print(f"  {'random split':<16}" +
          "".join(f"{st.mean(rnd[m]):>12.4f}" if rnd.get(m) else f"{'-':>12}"
                  for m in models))

    # ---------------- 方差分解 ----------------
    print("\n" + "=" * 72)
    print("[2] 方差成分  (两因子随机效应，完全交叉)")
    print("=" * 72)
    comp = decompose(cells)
    if comp is None:
        print("  设计不平衡或重复不足。")
        return 1
    a, b, n = comp.pop("_dims")
    tot = sum(comp.values())
    print(f"  设计: {a} architectures x {b} farms x {n} seeds\n")
    print(f"  {'component':<22}{'variance':>12}{'s.d.':>10}{'share':>9}")
    for k in ["farm", "architecture", "architecture x farm",
              "seed (residual)"]:
        v = comp[k]
        print(f"  {k:<22}{v:>12.5f}{math.sqrt(v):>10.4f}"
              f"{100*v/tot:>8.1f}%")
    print(f"  {'total':<22}{tot:>12.5f}{math.sqrt(tot):>10.4f}")

    if comp["architecture"] > 0:
        print(f"\n  Var(farm) / Var(architecture) = "
              f"{comp['farm']/comp['architecture']:.1f}")
    if comp["seed (residual)"] > 0:
        print(f"  Var(farm) / Var(seed)         = "
              f"{comp['farm']/comp['seed (residual)']:.1f}")

    # ---------------- bootstrap ----------------
    if args.boot:
        random.seed(0)
        keep = defaultdict(list)
        for _ in range(args.boot):
            fb = [random.choice(folds) for _ in folds]
            cb = {}
            for i, f in enumerate(fb):
                for m in models:
                    cb[(m, f"b{i}")] = cells[(m, f)]
            c2 = decompose(cb)
            if c2:
                c2.pop("_dims")
                t2 = sum(c2.values())
                for k, v in c2.items():
                    keep[k].append(100 * v / t2 if t2 else 0)
        print(f"\n  share 的 90% bootstrap 区间 (对农场重抽样, "
              f"{args.boot} 次)")
        for k in ["farm", "architecture", "architecture x farm",
                  "seed (residual)"]:
            s = sorted(keep[k])
            lo = s[int(0.05 * len(s))]
            hi = s[int(0.95 * len(s)) - 1]
            print(f"    {k:<22}{lo:>7.1f}% – {hi:.1f}%")

    # ---------------- ranking 一致性 ----------------
    print("\n" + "=" * 72)
    print("[3] 各架构下的农场排名是否一致")
    print("=" * 72)
    order = {m: [st.mean(cells[(m, c)]) for c in folds] for m in models}
    print(f"  {'':<12}" + "".join(f"{m:>12}" for m in models))
    for m in models:
        print(f"  {m:<12}" +
              "".join(f"{spearman(order[m], order[k]):>12.3f}"
                      for k in models))
    offd = [spearman(order[m], order[k])
            for i, m in enumerate(models) for k in models[i+1:]]
    if offd:
        print(f"\n  平均成对 Spearman = {st.mean(offd):+.3f}")
        print("  接近 1 表示同一批农场在每个架构下都难，失败是站点的属性，")
        print("  不是某个检测器的属性。")
        worst = {m: folds[order[m].index(min(order[m]))] for m in models}
        best = {m: folds[order[m].index(max(order[m]))] for m in models}
        print(f"\n  最差折: {worst}")
        print(f"  最好折: {best}")

    # ---------------- 每架构的 split-rule 差距 ----------------
    print("\n" + "=" * 72)
    print("[4] 随机划分与 leave-one-farm-out 的差距，逐架构")
    print("=" * 72)
    print(f"  {'model':<12}{'random':>10}{'by-farm':>10}{'gap':>9}"
          f"{'overstate':>11}{'fold sd':>9}{'seed sd':>9}")
    for m in models:
        if not rnd.get(m):
            continue
        r = st.mean(rnd[m])
        fm = [st.mean(cells[(m, c)]) for c in folds]
        bf = st.mean(fm)
        sd_f = st.stdev(fm)
        sd_s = st.mean([st.stdev(cells[(m, c)]) for c in folds])
        print(f"  {m:<12}{r:>10.4f}{bf:>10.4f}{r-bf:>9.4f}"
              f"{100*(r-bf)/r:>10.1f}%{sd_f:>9.4f}{sd_s:>9.4f}")
    print("\n  若 overstate 在各架构上量级相同，随机划分的高估就不是")
    print("  某个检测器的性质。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
