#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkpoint_bias_bkh.py -- BreaKHis 的 checkpoint 选择偏差，只做推理。

留一组患者的折里，验证集就是被留出的那批患者本身。训练循环每轮在它上面
算 fitness、保留最好的一轮、并据此早停。于是被留出的患者不只是拿到了一个
分数，它还挑了产生这个分数的权重。

最后一轮的权重没有被任何验证信号选过，所以它给出的是无选择的估计。本脚本
对每个 run 的 last.pt 重跑一次验证，与现有的 best 结果并排输出。

★ 不训练。100 个 checkpoint，224 px 分类，4090 上二十分钟左右。

★ 走的是 bkh_run.val_once，与主表逐字相同的代码路径。这一点是刻意的：
  弃权分析当初自己写了一套匹配逻辑，结果同一个农场的分数从 0.440 变成
  0.130，两张表对不上。这里不重复那个错误。

★ imgsz 必须与训练时一致。bkh_run.py 顶部的 IMGSZ = 640 是过时常量，
  args.yaml 记录的实际值是 224。本脚本默认 224，并逐个 run 核对 args.yaml，
  不一致就报错退出，而不是安静地算出一批错数字。

用法：
    cd /root/autodl-tmp/breakhis
    screen -S bkh
    python checkpoint_bias_bkh.py --dry-run      # 先看清单，不跑模型
    python checkpoint_bias_bkh.py                # 正式跑
    python checkpoint_bias_bkh.py --models yolo11s-cls --configs patient_fold0

输出：
    results_v2/checkpoint_bias.csv
    列与 durian / GWHD 的同名表一致，可直接进 verify_claims.py
"""

import argparse
import csv
import os
import re
import sys
import time

ROOT = "/root/autodl-tmp/breakhis"
MODELS = ["yolo11n-cls", "yolo11s-cls"]
SEEDS = [42, 1, 2, 3, 4]
CONFIGS = ([f"patient_fold{i}" for i in range(5)] +
           [f"random_fold{i}" for i in range(5)])


def read_trained_imgsz(run_dir):
    """从 args.yaml 读训练时的真实 imgsz。读不到返回 None。"""
    f = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(f):
        return None
    for line in open(f, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*imgsz:\s*([0-9]+)", line)
        if m:
            return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="只清点权重与 imgsz，不加载模型")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    runs = os.path.join(root, "runs")
    splits = os.path.join(root, "splits")
    out = a.out or os.path.join(root, "results_v2", "checkpoint_bias.csv")

    sys.path.insert(0, root)          # 为了 import bkh_run
    if not a.dry_run:
        from bkh_run import val_once

    jobs, missing, mismatched = [], [], []
    for m in a.models:
        for cfg in a.configs:
            for s in a.seeds:
                name = f"{m}_{cfg}_s{s}"
                d = os.path.join(runs, name)
                best = os.path.join(d, "weights", "best.pt")
                last = os.path.join(d, "weights", "last.pt")
                if not (os.path.isfile(best) and os.path.isfile(last)):
                    missing.append(name)
                    continue
                trained = read_trained_imgsz(d)
                if trained is not None and trained != a.imgsz:
                    mismatched.append((name, trained))
                    continue
                jobs.append((m, cfg, s, name, best, last))

    print(f"待验证 {len(jobs)} 个 run（每个跑 best 与 last 两次）")
    if missing:
        print(f"  缺权重 {len(missing)} 个: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if mismatched:
        print("\n训练时的 imgsz 与 --imgsz 不一致，已跳过："
              f"{len(mismatched)} 个")
        for n, t in mismatched[:6]:
            print(f"    {n}: args.yaml 记录 {t}, 本次传入 {a.imgsz}")
        sys.exit("用 --imgsz 改成 args.yaml 里的值再跑。"
                 "用错分辨率验证会得到一批看起来正常的错数字。")
    if a.dry_run:
        print("\n--dry-run，未加载模型。")
        return 0

    rows, t0 = [], time.time()
    for i, (m, cfg, s, name, best, last) in enumerate(jobs, 1):
        rb = val_once(best, os.path.join(splits, cfg),
                      name + "_cb_best", a.imgsz, runs)
        rl = val_once(last, os.path.join(splits, cfg),
                      name + "_cb_last", a.imgsz, runs)
        rows.append({"model": m, "config": cfg, "seed": s,
                     "best_top1": rb["top1"], "last_top1": rl["top1"],
                     "delta": rb["top1"] - rl["top1"],
                     "best_top5": rb["top5"], "last_top5": rl["top5"]})
        el = time.time() - t0
        print(f"  [{i}/{len(jobs)}] {name}  best {rb['top1']:.4f}  "
              f"last {rl['top1']:.4f}  delta {rb['top1']-rl['top1']:+.4f}  "
              f"({el/i:.0f}s/run, 剩 {el/i*(len(jobs)-i)/60:.0f} min)",
              flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {out}  ({len(rows)} 行)")

    # ---- 摘要：两种划分下选择偏差的大小 ----
    import statistics as st
    print(f"\n{'model':<14}{'regime':<10}{'best':>9}{'last':>9}{'delta':>9}")
    summary = {}
    for m in sorted({r["model"] for r in rows}):
        for reg, pref in (("item-level", "random"), ("unit-level", "patient")):
            sel = [r for r in rows if r["model"] == m
                   and r["config"].startswith(pref)]
            if not sel:
                continue
            b = st.mean(r["best_top1"] for r in sel)
            l = st.mean(r["last_top1"] for r in sel)
            summary[(m, reg)] = (b, l)
            print(f"{m:<14}{reg:<10}{b:>9.4f}{l:>9.4f}{b-l:>+9.4f}")

    print("\n高估幅度（item-level 相对 unit-level）：")
    for m in sorted({r["model"] for r in rows}):
        if (m, "item-level") in summary and (m, "unit-level") in summary:
            ib, il = summary[(m, "item-level")]
            ub, ul = summary[(m, "unit-level")]
            sel_ = (ib - ub) / ib * 100
            fin_ = (il - ul) / il * 100
            print(f"  {m:<14}选择后 {sel_:.1f}%  ->  无选择 {fin_:.1f}%  "
                  f"（升高 {fin_-sel_:.1f} 点）")

    print("\n预期方向：item-level 的 best 与 last 几乎不动（验证集在训练集里"
          "近似重复），unit-level 的 last 明显更低。若 item-level 也大幅下降，"
          "先查 imgsz 与 split 目录，不要直接写进稿子。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
