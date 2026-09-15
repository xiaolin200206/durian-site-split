#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gwhd_run.py -- GWHD 2021 上的外部验证：训练、评估、逐 domain 评估。

要回答的只有一个问题：

    评估单元的方差是否同样支配报告值，在另一个作物、另一个任务、
    另一个采集团队的数据上？

设计与榴莲实验对齐，但有两处必要的差异，都在 Methods 里说明：

  1. 47 个 domain 无法逐一留出（47 折 x 5 seed 跑不完），所以沿用你已
     建好的 domain-disjoint 5 折，每折留出 9-10 个 domain。
  2. 训练集是榴莲的 6.3 倍，150 epoch 是浪费，默认降到 100 / patience 30。

关键的一步是 --step per_domain：折分数本身是 9-10 个 domain 的池化平均，
而本文的主张恰恰是"池化掩盖离散度"。所以我们拿每个 domain-fold 的权重，
对它留出的每个 domain 单独评估。五折加起来覆盖全部 47 个 domain，每个都
由一个没见过它的模型评分。零额外训练成本，得到的正是榴莲那边
per-burst / per-tree 的对应物。

用法：
    cd /root/autodl-tmp/gwhd2021
    python gwhd_run.py --step splits
    nohup python gwhd_run.py --step train --models yolo11s --yes \\
        > train.log 2>&1 &
    python gwhd_run.py --step eval       --models yolo11s
    python gwhd_run.py --step per_domain --models yolo11s

然后（复用榴莲的脚本）：
    python variance_decomposition.py --dir results_v2
    python q.py --dir results_v2
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys
import time
from collections import defaultdict

ROOT = "/root/autodl-tmp/gwhd2021"

MODELS = {
    "yolo11n":  {"weights": "yolo11n.pt",  "batch": 32},
    "yolo11s":  {"weights": "yolo11s.pt",  "batch": 32},
    "yolo11m":  {"weights": "yolo11m.pt",  "batch": 16},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "batch": 8},
}
SEEDS = [42, 1, 2, 3, 4]

IMGSZ = 640
EPOCHS = 100          # 榴莲是 150；这里数据多 6.3 倍，收敛快
PATIENCE = 30
WORKERS = 8
MIN_PER_UNIT = 5      # 与榴莲一致：单元少于 5 张不评估

MIN_PER_RUN = {"yolo11n": 28, "yolo11s": 40, "yolo11m": 75, "rtdetr-l": 120}


def P(root):
    return {"root": root,
            "proc": os.path.join(root, "processed"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_v2"),
            "splits": os.path.join(root, "splits")}


def manifest(p):
    f = os.path.join(p["proc"], "manifest_clean.csv")
    if not os.path.isfile(f):
        f = os.path.join(p["proc"], "manifest.csv")
    if not os.path.isfile(f):
        sys.exit(f"找不到 manifest：{p['proc']}")
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"manifest: {f}  ({len(rows)} 行)")
    return rows


def write_yaml(d, train, val):
    os.makedirs(d, exist_ok=True)
    for name, paths in (("train", train), ("val", val)):
        with open(os.path.join(d, name + ".txt"), "w") as fh:
            for x in paths:
                fh.write(x + "\n")
    y = os.path.join(d, "data.yaml")
    with open(y, "w") as fh:
        fh.write(f"train: {os.path.join(d,'train.txt')}\n")
        fh.write(f"val: {os.path.join(d,'val.txt')}\n")
        fh.write("nc: 1\nnames:\n- wheat_head\n")
    return y


# ------------------------------------------------------------------ splits --
def step_splits(a, p):
    rows = manifest(p)
    have = [r for r in rows if os.path.isfile(r["image_path"])]
    if len(have) < len(rows):
        print(f"★ {len(rows)-len(have)} 张图像文件缺失，已跳过")
    doms = sorted({r["domain"] for r in have})
    print(f"images {len(have)}   domains {len(doms)}")

    made = []
    for regime, col in (("random", "random_fold"), ("domain", "domain_fold")):
        folds = sorted({r[col] for r in have}, key=lambda x: int(x))
        for f in folds:
            tr = [r["image_path"] for r in have if r[col] != f]
            va = [r["image_path"] for r in have if r[col] == f]
            name = f"{regime}_fold{f}"
            write_yaml(os.path.join(p["splits"], name), tr, va)
            vd = len({r["domain"] for r in have if r[col] == f})
            made.append((name, len(tr), len(va), vd))

    print(f"\n  {'split':<18}{'train':>8}{'val':>8}{'val domains':>13}")
    for n, t, v, vd in made:
        print(f"  {n:<18}{t:>8}{v:>8}{vd:>13}")

    # 完整性：domain 折不得有 domain 跨界
    bad = 0
    for f in sorted({r["domain_fold"] for r in have}, key=lambda x: int(x)):
        tr = {r["domain"] for r in have if r["domain_fold"] != f}
        va = {r["domain"] for r in have if r["domain_fold"] == f}
        if tr & va:
            bad += 1
            print(f"  ★ domain_fold{f}: {len(tr&va)} 个 domain 跨界")
    print(f"\n  domain 泄漏检查: {'通过' if bad==0 else '★ 失败'}")
    print(f"  写到 {p['splits']}")


# ------------------------------------------------------------------- train --
def build(w):
    from ultralytics import YOLO
    if "rtdetr" in w.lower():
        try:
            from ultralytics import RTDETR
            return RTDETR(w)
        except ImportError:
            pass
    return YOLO(w)


def cfgs_of(p):
    return sorted(d for d in os.listdir(p["splits"])
                  if os.path.isfile(os.path.join(p["splits"], d, "data.yaml")))


def step_train(a, p):
    if not os.path.isdir(p["splits"]):
        sys.exit("先跑 --step splits")
    cfgs = cfgs_of(p)
    seeds = a.seeds or SEEDS
    jobs = [(m, c, s) for m in a.models for s in seeds for c in cfgs]
    budget = sum(MIN_PER_RUN.get(m, 40) for m, _, _ in jobs)
    print(f"配置 {len(cfgs)}: {cfgs}")
    for m in a.models:
        print(f"  {m:<10} seeds {seeds}  batch {MODELS[m]['batch']}  "
              f"{len(cfgs)*len(seeds)} run  "
              f"约 {len(cfgs)*len(seeds)*MIN_PER_RUN.get(m,40)/60:.1f} 小时")
    print(f"\n合计 {len(jobs)} run，估计 {budget/60:.1f} 小时")
    print(f"epochs {a.epochs} / patience {a.patience}  (榴莲用的是 150/50)")
    print("顺序：种子优先。已完成的自动跳过。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return

    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"{model}_{cfg}_s{seed}"
        if glob.glob(os.path.join(p["runs"], name, "weights", "best.pt")):
            continue
        el = (time.time() - t0) / 60
        eta = (el / done * (len(jobs) - i) / 60) if done else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        spec = MODELS[model]
        try:
            build(spec["weights"]).train(
                data=os.path.join(p["splits"], cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                batch=a.batch or spec["batch"], workers=a.workers,
                patience=a.patience, project=p["runs"], name=name,
                exist_ok=True, deterministic=True, plots=False, verbose=False)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
            if "out of memory" in str(e).lower():
                print("    显存不足：--batch 调小")
    print(f"\n训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# -------------------------------------------------------------------- eval --
def val_once(weights, data_yaml, tag, imgsz, runs):
    from ultralytics import YOLO
    m = (build(weights) if "rtdetr" in os.path.basename(weights).lower()
         else YOLO(weights))
    r = m.val(data=data_yaml, imgsz=imgsz, split="val", batch=8, workers=2,
              project=runs, name=tag, exist_ok=True, verbose=False,
              plots=False)
    return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr)}


def dump(rows, path, head):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    keys = head + [k for k in keys if k not in head]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {path}  ({len(rows)} 行)")


def step_eval(a, p):
    cfgs = cfgs_of(p)
    seeds = a.seeds or SEEDS
    rows = []
    for m in a.models:
        for c in cfgs:
            for s in seeds:
                name = f"{m}_{c}_s{s}"
                best = os.path.join(p["runs"], name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                r = val_once(best, os.path.join(p["splits"], c, "data.yaml"),
                             name + "_val", a.imgsz, p["runs"])
                r.update({"model": m, "config": c, "seed": s,
                          "eval_on": "gwhd"})
                rows.append(r)
                print(f"  {name:<34} mAP50 {r['mAP50']:.4f}", flush=True)
    if not rows:
        sys.exit("没有找到权重")
    dump(rows, os.path.join(p["results"], "in_region.csv"),
         ["model", "config", "seed", "eval_on"])

    print("\n" + "=" * 70)
    print("  随机划分 vs domain-disjoint")
    print("=" * 70)
    for m in sorted({r["model"] for r in rows}):
        for regime in ("random", "domain"):
            v = defaultdict(list)
            for r in rows:
                if r["model"] == m and r["config"].startswith(regime):
                    v[r["config"]].append(r["mAP50"])
            if not v:
                continue
            fm = [st.mean(x) for x in v.values()]
            sd = st.stdev(fm) if len(fm) > 1 else 0.0
            per_seed = [st.stdev(x) for x in v.values() if len(x) > 1]
            within = st.mean(per_seed) if per_seed else float("nan")
            wtxt = f"{within:.4f}" if per_seed else "n/a (1 seed)"
            print(f"  {m:<10} {regime:<8} mean {st.mean(fm):.4f}  "
                  f"fold sd {sd:.4f}  seed sd {wtxt}  n={len(fm)}")
        rnd = [st.mean(x) for k, x in
               [(k, [r["mAP50"] for r in rows
                     if r["model"] == m and r["config"] == k])
                for k in {r["config"] for r in rows
                          if r["config"].startswith("random")}]]
        dom = [st.mean(x) for k, x in
               [(k, [r["mAP50"] for r in rows
                     if r["model"] == m and r["config"] == k])
                for k in {r["config"] for r in rows
                          if r["config"].startswith("domain")}]]
        rnd = [x for x in rnd if x == x]
        dom = [x for x in dom if x == x]
        if rnd and dom:
            print(f"  {m:<10} overstate "
                  f"{100*(st.mean(rnd)-st.mean(dom))/st.mean(rnd):.1f}%"
                  f"   (durian: 44.3%)")


# -------------------------------------------------------------- per domain --
def step_per_domain(a, p):
    """每个 domain 单独评估，用没见过它的那一折的权重。

    这是与榴莲 per-burst / per-tree 对应的分析。折分数是 9-10 个 domain
    的池化平均；本文的主张是池化会掩盖离散度，所以必须拆开。
    """
    rows_m = manifest(p)
    have = [r for r in rows_m if os.path.isfile(r["image_path"])]
    by_dom = defaultdict(list)
    fold_of = {}
    for r in have:
        by_dom[r["domain"]].append(r["image_path"])
        fold_of[r["domain"]] = r["domain_fold"]

    doms = sorted(d for d in by_dom if len(by_dom[d]) >= a.min_images)
    skipped = [d for d in by_dom if len(by_dom[d]) < a.min_images]
    print(f"domains {len(by_dom)}；>= {a.min_images} 张的 {len(doms)}")
    if skipped:
        print(f"  跳过（图像不足）: {skipped}")
    print(f"覆盖 {sum(len(by_dom[d]) for d in doms)} / {len(have)} 张")

    seeds = a.seeds or SEEDS
    scratch = os.path.join(p["root"], "domain_subsets")
    rows = []
    for i, d in enumerate(doms, 1):
        cfg = f"domain_fold{fold_of[d]}"
        y = write_yaml(os.path.join(scratch, d.replace("/", "_")),
                       by_dom[d], by_dom[d])
        for m in a.models:
            for s in seeds:
                name = f"{m}_{cfg}_s{s}"
                best = os.path.join(p["runs"], name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                r = val_once(best, y, f"dom_{d}_{m}_s{s}", a.imgsz, p["runs"])
                r.update({"group": d, "n_images": len(by_dom[d]),
                          "fold": cfg, "model": m, "seed": s,
                          "country": next((x["country"] for x in have
                                           if x["domain"] == d), "")})
                rows.append(r)
        if i % 5 == 0:
            print(f"  {i}/{len(doms)}", flush=True)

    if not rows:
        sys.exit("没有找到 domain 折的权重；先跑 --step train")
    dump(rows, os.path.join(p["results"], "gwhd_by_domain.csv"),
         ["group", "model", "fold", "seed", "n_images", "country"])

    per = defaultdict(list)
    n_img = {}
    for r in rows:
        per[r["group"]].append(r["mAP50"])
        n_img[r["group"]] = r["n_images"]
    m_ = {g: st.mean(v) for g, v in per.items()}
    vals = list(m_.values())
    mu, sd = st.mean(vals), st.stdev(vals)
    print("\n" + "=" * 70)
    print(f"  逐 domain（{len(vals)} 个单元，每个由未见过它的模型评分）")
    print("=" * 70)
    print(f"  均值 {mu:.4f}  sd {sd:.4f}  CV {sd/mu:.3f}  "
          f"范围 {min(vals):.3f}-{max(vals):.3f}  "
          f"极差 {max(vals)/min(vals):.1f}x" if min(vals) > 0 else "")
    print(f"\n  {'domain':<20}{'n':>6}{'mAP50':>9}")
    for g in sorted(m_, key=lambda x: -m_[x]):
        print(f"  {g:<20}{n_img[g]:>6}{m_[g]:>9.4f}")
    print("\n  durian 对照：per-burst CV 0.75，per-tree CV 0.17")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["splits", "train", "eval", "per_domain"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--models", nargs="+", default=["yolo11s"],
                    choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--min-images", type=int, default=MIN_PER_UNIT)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(a.root)
    {"splits": step_splits, "train": step_train,
     "eval": step_eval, "per_domain": step_per_domain}[a.step](a, p)


if __name__ == "__main__":
    main()
