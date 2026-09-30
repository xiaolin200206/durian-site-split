#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q.py -- 把填补稿件 [[RERUN]] 占位所需的数字一次全部打印出来。

放到 results_v2 目录里跑:
    python q.py

或从项目根目录跑:
    python q.py --dir results_v2

纯标准库，不需要 pandas。缺哪张表就跳过哪一段，不会中断。
"""

import argparse
import csv
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

CLASSES = ["Algal", "Leaf_rot", "Phomopsis",
           "Psyllid", "Psyllid_damage", "leaf_hopper_damage"]
MIN_IMAGES = 5


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        return None
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = (r.get(k) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def hdr(s):
    print("\n" + "=" * 68)
    print(s)
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    d = a.dir

    # ------------------------------------------------ 1. per class ------
    hdr("[1] Per-class AP50  ->  Results 的 per-class 表")
    ir, ci = load(d, "in_region.csv"), load(d, "cross_island.csv")
    fa = load(d, "farm_focal_annotation_scale.csv")
    if ir is None or ci is None:
        print("  缺 in_region.csv 或 cross_island.csv")
    else:
        farms = {}
        if fa:
            g = defaultdict(set)
            for r in fa:
                if (r.get("focal") or "").strip() == "ALL" and \
                        (num(r, "n_images") or 0) >= MIN_IMAGES:
                    g[r["class"]].add(str(r["farm"]).strip())
            farms = {k: len(v) for k, v in g.items()}
        print(f"  {'class':<20}{'farms>=5':>9}{'random':>9}{'sabah':>9}"
              f"{'change':>9}{'n_sabah':>9}")
        for c in CLASSES:
            k = "AP50::" + c
            rv = [num(r, k) for r in ir if r.get("config") == "random"]
            rv = [x for x in rv if x is not None]
            sv = [num(r, k) for r in ci]
            sv = [x for x in sv if x is not None]
            if not rv or not sv:
                print(f"  {c:<20}{farms.get(c,'?'):>9}   (missing column)")
                continue
            r, s = st.mean(rv), st.mean(sv)
            print(f"  {c:<20}{farms.get(c,'?'):>9}{r:>9.3f}{s:>9.3f}"
                  f"{100*(s-r)/r:>8.0f}%{len(sv):>9}")
        print("\n  change = (sabah - random) / random。正=跨岛更好。")

    # ------------------------------------------------ 2. per burst ------
    hdr("[2] Per-burst / per-tree  ->  Results 的站点表与敏感性表")
    settings = [("peninsula_by_burst_60s_min5.csv", "60 s / >=5"),
                ("peninsula_by_burst_30s_min5.csv", "30 s / >=5"),
                ("peninsula_by_burst_15s_min5.csv", "15 s / >=5"),
                ("peninsula_by_burst_15s_min3.csv", "15 s / >=3"),
                ("peninsula_by_burst_15s.csv", "(unsuffixed)")]
    pool = 827
    sa = load(d, "split_assignment.csv")
    if sa:
        pool = len(sa)
    primary = None
    print(f"  {'setting':<16}{'bursts':>7}{'imgs':>7}{'cover':>8}"
          f"{'mean':>8}{'sd':>8}{'CV':>7}{'min':>8}{'max':>8}")
    for fn, label in settings:
        rows = load(d, fn)
        if rows is None:
            continue
        g = defaultdict(list)
        n_img = {}
        for r in rows:
            v = num(r, "mAP50")
            if v is not None:
                g[r["burst"]].append(v)
                n_img[r["burst"]] = num(r, "n_images") or 0
        m = {b: st.mean(v) for b, v in g.items()}
        vals = list(m.values())
        if len(vals) < 2:
            continue
        sd = st.stdev(vals)
        mu = st.mean(vals)
        cov = sum(n_img.values())
        print(f"  {label:<16}{len(vals):>7}{int(cov):>7}"
              f"{100*cov/pool:>7.1f}%{mu:>8.3f}{sd:>8.3f}"
              f"{sd/mu:>7.3f}{min(vals):>8.3f}{max(vals):>8.3f}")
        if fn.startswith("peninsula_by_burst_60s"):
            primary = (m, n_img)

    if primary:
        m, n_img = primary
        zeros = [b for b, v in m.items() if v == 0]
        print(f"\n  60 s 设置: {len(zeros)} 个 burst 得分为零 {zeros}")
        by_farm = defaultdict(list)
        for b, v in m.items():
            by_farm[b.split("b")[0].lstrip("f")].append(v)
        print(f"\n  {'farm':<6}{'bursts':>7}{'mean':>8}{'sd':>8}"
              f"{'min':>8}{'max':>8}")
        fm = {}
        for f in sorted(by_farm, key=lambda x: (len(x), x)):
            v = by_farm[f]
            fm[f] = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"  {f:<6}{len(v):>7}{fm[f]:>8.3f}{sd:>8.3f}"
                  f"{min(v):>8.3f}{max(v):>8.3f}")
        if len(fm) > 1 and min(fm.values()) > 0:
            print(f"\n  农场 burst 均值极差: {max(fm.values())/min(fm.values()):.2f}x")

    # sabah trees
    st_rows = load(d, "sabah_by_tree.csv")
    tree_vals = None
    if st_rows:
        g, n_img = defaultdict(list), {}
        for r in st_rows:
            v = num(r, "mAP50")
            if v is not None:
                g[r["group"]].append(v)
                n_img[r["group"]] = num(r, "n_images") or 0
        m = {k: st.mean(v) for k, v in g.items() if n_img[k] >= MIN_IMAGES}
        tree_vals = list(m.values())
        sd, mu = st.stdev(tree_vals), st.mean(tree_vals)
        print(f"\n  Sabah trees (>={MIN_IMAGES} imgs): n={len(tree_vals)} "
              f"mean {mu:.3f} sd {sd:.3f} CV {sd/mu:.3f} "
              f"range {min(tree_vals):.3f}-{max(tree_vals):.3f}")
        if primary:
            pv = list(primary[0].values())
            cvp = st.stdev(pv) / st.mean(pv)
            print(f"  CV ratio peninsula / Sabah = {cvp/(sd/mu):.2f}")
            print(f"  per-site mean difference   = "
                  f"{abs(st.mean(pv)-mu):.3f}")

    so = load(d, "sabah_by_orchard.csv")
    if so:
        g = defaultdict(list)
        for r in so:
            v = num(r, "mAP50")
            if v is not None:
                g[r["group"]].append(v)
        om = {k: st.mean(v) for k, v in g.items()}
        print(f"  Sabah orchards: " +
              ", ".join(f"{k} {v:.3f}" for k, v in sorted(om.items())) +
              (f"  ratio {max(om.values())/min(om.values()):.2f}x"
               if len(om) > 1 else ""))

    # ------------------------------------------------ 3. resampling -----
    hdr("[3] Resampling  ->  Fig. 2 与样本量要求")
    random.seed(0)

    def curve(vals, label, ks):
        for k in ks:
            if k > len(vals):
                continue
            means = sorted(st.mean(random.sample(vals, k))
                           for _ in range(4000))
            lo, hi = means[200], means[3799]
            print(f"  {label:<12} k={k:<3} 90% interval "
                  f"{lo:.3f}-{hi:.3f}  width {hi-lo:.3f}")

    if primary:
        pv = list(primary[0].values())
        curve(pv, "peninsula", [1, 5, 10, 20])
        for tol in (0.10, 0.05):
            n = math.ceil((1.645 * st.stdev(pv) / tol) ** 2)
            print(f"  capture sites needed for +/-{tol}: {n}")
    if tree_vals:
        curve(tree_vals, "sabah", [1, 5])

    ir2 = load(d, "in_region.csv")
    if ir2:
        g = defaultdict(list)
        for r in ir2:
            v = num(r, "mAP50")
            if v is not None:
                g[r["config"]].append(v)
        fm = [st.mean(v) for k, v in g.items() if k.startswith("byfarm")]
        if len(fm) > 1:
            s = st.stdev(fm)
            for tol in (0.10, 0.05):
                print(f"  farms needed for +/-{tol}: "
                      f"{math.ceil((1.645*s/tol)**2)}  (sd {s:.4f})")

    # ------------------------------------------------ 4. coverage -------
    p = os.path.join(d, "per_site_coverage.json")
    if os.path.isfile(p):
        hdr("[4] per_site_coverage.json (最后一次运行的设置)")
        print(" ", json.dumps(json.load(open(p, encoding="utf-8"))))

    print("\n把以上全部输出发回去。")


if __name__ == "__main__":
    main()
