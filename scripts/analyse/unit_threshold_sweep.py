#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unit_threshold_sweep.py -- 两个对照，零 GPU。

[1] 最小单元阈值扫描
    每单元至少几个 item 才计入分析是一个选择（主文用 5）。单元越小，
    平均精度越容易跑到 0 或 1，所以「单元间差异大」有可能只是度量噪声。
    把阈值提到 10、20、30 再看离散度还在不在，就把两者分开了。

    读法：如果 s.d. 随阈值下降得不多，那么异质性是真的；如果一路塌下去，
    主文那句「单元间跨度从接近失败到接近完美」就得改写成关于小单元的话。

[2] BreaKHis 的类别对照
    patient fold 是按 item 数贪心装箱的，没有做类别平衡，所以病人身份
    和类别构成在方差分解里是混在一起的。durian 有单类对照，BreaKHis
    需要同样的东西：在良性、恶性各自内部重做，看单元分量还在不在。

用法：
    python unit_threshold_sweep.py --root .
"""

import argparse
import os

import numpy as np
import pandas as pd

SETS = [
    ("Durian bursts", "results_durian/peninsula_by_burst_60s_min5.csv",
     "burst", "mAP50", "n_images", "yolo11s"),
    ("GWHD sessions", "results_gwhd/gwhd_by_domain.csv",
     "group", "mAP50", "n_images", "yolo11s"),
    ("BreaKHis patients", "results_breakhis/breakhis_by_patient.csv",
     "group", "top1", "n_images", "yolo11s-cls"),
    ("HAR subjects", "results_har/har_by_subject.csv",
     "group", "macro_f1", "n_windows", "mlp256"),
]
THRESHOLDS = [5, 10, 20, 30, 50]


def sweep(root):
    print("=" * 74)
    print("[1] 最小单元阈值扫描（等权口径）")
    print("=" * 74)
    rows = []
    for name, rel, ukey, metric, nkey, model in SETS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"缺表，跳过: {rel}")
            continue
        df = pd.read_csv(path)
        df = df[df.model == model].dropna(subset=[metric])
        g = df.groupby(ukey).agg(**{"score": (metric, "mean"),
                                    "n": (nkey, "first")})
        print(f"\n{name} ({model})")
        print(f"  {'min items':>10}{'units':>7}{'kept':>7}{'mean':>8}"
              f"{'s.d.':>8}{'CV':>7}{'min':>7}{'max':>7}")
        base_sd = None
        for t in THRESHOLDS:
            sel = g[g.n >= t]
            if len(sel) < 5:
                print(f"  {t:>10}{len(sel):>7}   單元不足 5 个，停止")
                break
            sd = sel.score.std(ddof=1)
            if base_sd is None:
                base_sd = sd
            print(f"  {t:>10}{len(sel):>7}{len(sel) / len(g):>7.0%}"
                  f"{sel.score.mean():>8.3f}{sd:>8.3f}"
                  f"{sd / sel.score.mean():>7.2f}"
                  f"{sel.score.min():>7.3f}{sel.score.max():>7.3f}")
            rows.append([name, t, len(sel), round(sel.score.mean(), 4),
                         round(sd, 4), round(sel.score.min(), 4),
                         round(sel.score.max(), 4)])
        if base_sd:
            last = rows[-1]
            print(f"    s.d. {base_sd:.3f} -> {last[4]:.3f} "
                  f"({last[4] / base_sd:.0%} of the min-5 value) "
                  f"at min items = {last[1]}")
    return rows


def breakhis_class_control(root):
    print("\n" + "=" * 74)
    print("[2] BreaKHis 类别对照：在良性、恶性内部各自重做")
    print("=" * 74)
    path = os.path.join(root, "results_breakhis", "breakhis_by_patient.csv")
    if not os.path.isfile(path):
        print("缺表，跳过")
        return []
    df = pd.read_csv(path).dropna(subset=["top1"])
    rows = []
    for label in ["all"] + sorted(df.label.unique()):
        sub = df if label == "all" else df[df.label == label]
        # 折内单元 vs 折间，与 nested_decomposition 同口径的简化版
        fold_mean = sub.groupby("fold").top1.transform("mean")
        unit_mean = sub.groupby(["fold", "group"]).top1.transform("mean")
        grand = sub.top1.mean()
        ss_tot = ((sub.top1 - grand) ** 2).sum()
        ss_fold = ((fold_mean - grand) ** 2).sum()
        ss_unit = ((unit_mean - fold_mean) ** 2).sum()
        ss_res = ss_tot - ss_fold - ss_unit
        n_units = sub.groupby(["fold", "group"]).ngroups
        per_unit = sub.groupby("group").top1.mean()
        print(f"\n  {label:<12}{n_units:>4} patients, "
              f"mean {per_unit.mean():.3f}, s.d. {per_unit.std(ddof=1):.3f}, "
              f"range {per_unit.min():.3f}–{per_unit.max():.3f}")
        print(f"    sums of squares: fold {100*ss_fold/ss_tot:5.1f}%   "
              f"unit within fold {100*ss_unit/ss_tot:5.1f}%   "
              f"residual {100*ss_res/ss_tot:5.1f}%")
        rows.append([label, n_units, round(per_unit.mean(), 4),
                     round(per_unit.std(ddof=1), 4),
                     round(100 * ss_unit / ss_tot, 1)])
    print("\n  读法：若两个类别内部的 unit 分量都仍然大，那么 BreaKHis 的")
    print("       单元分量就不是折的类别构成造成的，与 durian 的单类对照同理。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results_unit_threshold_sweep.csv")
    a = ap.parse_args()
    rows = sweep(a.root)
    breakhis_class_control(a.root)
    if rows:
        out = os.path.join(a.root, a.out)
        pd.DataFrame(rows, columns=["dataset", "min_items", "units", "mean",
                                    "sd", "min", "max"]).to_csv(
            out, index=False, encoding="utf-8-sig")
        print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
