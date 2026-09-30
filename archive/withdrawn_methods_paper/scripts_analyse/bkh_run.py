#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bkh_run.py -- BreaKHis 上的训练、评估与逐患者评估。

与前两个数据集的对应关系：

    榴莲      capture burst   58 个   per_site_v2.py
    GWHD      domain          47 个   gwhd_run.py --step per_domain
    BreaKHis  patient         82 个   本脚本 --step per_patient

任务是二分类（benign / malignant），所以用 YOLO11s-cls，指标是 top-1
准确率与 macro F1。其余设定与前两个数据集对齐：640 px、5 seed、
patience 50。分类比检测快得多，全套跑得完。

用法：
    python bkh_run.py --step splits
    nohup python bkh_run.py --step train --yes > bkh.log 2>&1 &
    python bkh_run.py --step eval
    python bkh_run.py --step per_patient
"""

import argparse
import csv
import glob
import os
import shutil
import statistics as st
import sys
import time
from collections import Counter, defaultdict

ROOT = "/root/autodl-tmp/breakhis"
MODELS = {"yolo11n-cls": ("yolo11n-cls.pt", 64),
          "yolo11s-cls": ("yolo11s-cls.pt", 64)}
SEEDS = [42, 1, 2, 3, 4]
IMGSZ = 640
EPOCHS = 100
PATIENCE = 50
MIN_PER_UNIT = 5


def P(root):
    root = os.path.abspath(root)
    return {"root": root, "proc": os.path.join(root, "processed"),
            "splits": os.path.join(root, "splits"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_v2")}


def manifest(p):
    f = os.path.join(p["proc"], "manifest_clean.csv")
    if not os.path.isfile(f):
        sys.exit(f"找不到 {f}，先跑 prepare_breakhis.py")
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"manifest: {len(rows)} 行")
    return rows


def link_tree(dst, rows, split_of):
    """Ultralytics 分类要求 <root>/<split>/<class>/<img> 的目录结构。
    用软链，不复制，省磁盘。"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    n = Counter()
    for r in rows:
        s = split_of(r)
        if s not in ("train", "val"):
            continue
        d = os.path.join(dst, s, r["label"])
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, r["image_id"] + ".png")
        if not os.path.exists(link):
            try:
                os.symlink(r["image_path"], link)
            except OSError:
                shutil.copy2(r["image_path"], link)
        n[s] += 1
    return n


def step_splits(a, p):
    rows = manifest(p)
    made = []
    for regime, col in (("random", "random_fold"),
                        ("patient", "patient_fold")):
        for f in sorted({r[col] for r in rows}, key=int):
            name = f"{regime}_fold{f}"
            n = link_tree(os.path.join(p["splits"], name), rows,
                          lambda r, c=col, k=f:
                          "val" if r[c] == k else "train")
            vp = len({r["patient"] for r in rows if r[col] == f})
            made.append((name, n["train"], n["val"], vp))
    print(f"\n  {'split':<18}{'train':>8}{'val':>8}{'val patients':>14}")
    for nm, t, v, vp in made:
        print(f"  {nm:<18}{t:>8}{v:>8}{vp:>14}")

    bad = 0
    for f in sorted({r["patient_fold"] for r in rows}, key=int):
        tr = {r["patient"] for r in rows if r["patient_fold"] != f}
        va = {r["patient"] for r in rows if r["patient_fold"] == f}
        if tr & va:
            bad += 1
    print(f"\n  患者跨折: {'无' if bad == 0 else f'★ {bad}'}")
    print(f"  写到 {p['splits']}")


def step_train(a, p):
    from ultralytics import YOLO
    cfgs = sorted(d for d in os.listdir(p["splits"])
                  if os.path.isdir(os.path.join(p["splits"], d, "train")))
    seeds = a.seeds or SEEDS
    jobs = [(m, c, s) for m in a.models for s in seeds for c in cfgs]
    print(f"配置 {len(cfgs)}  x {len(seeds)} seed x {len(a.models)} model "
          f"= {len(jobs)} run")
    if not a.yes:
        print("加 --yes 开始")
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
        w, batch = MODELS[model]
        try:
            YOLO(w).train(data=os.path.join(p["splits"], cfg),
                          imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                          batch=a.batch or batch, workers=a.workers,
                          patience=a.patience, project=p["runs"], name=name,
                          exist_ok=True, deterministic=True, plots=False,
                          verbose=False)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
    print(f"\n结束，用时 {(time.time()-t0)/60:.1f} 分钟")


def val_once(weights, data_dir, tag, imgsz, runs):
    from ultralytics import YOLO
    r = YOLO(weights).val(data=data_dir, imgsz=imgsz, split="val", batch=32,
                          workers=2, project=runs, name=tag, exist_ok=True,
                          verbose=False, plots=False)
    return {"top1": float(r.top1), "top5": float(getattr(r, "top5", 0.0))}


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
    cfgs = sorted(d for d in os.listdir(p["splits"])
                  if os.path.isdir(os.path.join(p["splits"], d, "train")))
    seeds = a.seeds or SEEDS
    rows = []
    for m in a.models:
        for c in cfgs:
            for s in seeds:
                name = f"{m}_{c}_s{s}"
                best = os.path.join(p["runs"], name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                r = val_once(best, os.path.join(p["splits"], c),
                             name + "_val", a.imgsz, p["runs"])
                r.update({"model": m, "config": c, "seed": s})
                rows.append(r)
                print(f"  {name:<36} top1 {r['top1']:.4f}", flush=True)
    if not rows:
        sys.exit("没有权重")
    dump(rows, os.path.join(p["results"], "in_region.csv"),
         ["model", "config", "seed"])

    print("\n" + "=" * 68)
    print("  随机划分 vs 患者互斥")
    print("=" * 68)
    for m in sorted({r["model"] for r in rows}):
        means = {}
        for reg in ("random", "patient"):
            v = defaultdict(list)
            for r in rows:
                if r["model"] == m and r["config"].startswith(reg):
                    v[r["config"]].append(r["top1"])
            if not v:
                continue
            fm = [st.mean(x) for x in v.values()]
            per = [st.stdev(x) for x in v.values() if len(x) > 1]
            means[reg] = st.mean(fm)
            print(f"  {m:<12}{reg:<9} mean {st.mean(fm):.4f}  "
                  f"fold sd {st.stdev(fm) if len(fm)>1 else 0:.4f}  "
                  f"seed sd {st.mean(per) if per else float('nan'):.4f}  "
                  f"n={len(fm)}")
        if len(means) == 2:
            o = 100 * (means["random"] - means["patient"]) / means["random"]
            print(f"  {m:<12}overstate {o:.1f}%   "
                  f"(durian 44.3% / 42.0%, GWHD 17.6%)")


def step_per_patient(a, p):
    """每个患者单独评估，用没见过他的那一折的权重。

    与榴莲的 per-burst、GWHD 的 per-domain 对应。患者折的分数是十几个
    患者的池化平均；本文的主张是池化掩盖离散度，所以要拆开。
    """
    rows_m = manifest(p)
    by_pat = defaultdict(list)
    fold_of, lab = {}, {}
    for r in rows_m:
        by_pat[r["patient"]].append(r)
        fold_of[r["patient"]] = r["patient_fold"]
        lab[r["patient"]] = r["label"]
    pats = sorted(x for x in by_pat if len(by_pat[x]) >= a.min_images)
    print(f"患者 {len(by_pat)}；>= {a.min_images} 张的 {len(pats)}")

    seeds = a.seeds or SEEDS
    scratch = os.path.join(p["root"], "patient_subsets")
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    rows = []
    for i, pat in enumerate(pats, 1):
        d = os.path.join(scratch, pat.replace("/", "_"))
        link_tree(d, by_pat[pat], lambda r: "val")
        # Ultralytics 的分类校验会检查 train/ 里确实有图，空目录会报
        # "no training images found"。放几张该患者自己的图占位；它们
        # 不参与任何计算，只是让检查通过。
        for cls in {r["label"] for r in rows_m}:
            td = os.path.join(d, "train", cls)
            os.makedirs(td, exist_ok=True)
        for r in by_pat[pat][:2]:
            link = os.path.join(d, "train", r["label"],
                                "placeholder_" + r["image_id"] + ".png")
            if not os.path.exists(link):
                try:
                    os.symlink(r["image_path"], link)
                except OSError:
                    shutil.copy2(r["image_path"], link)
        cfg = f"patient_fold{fold_of[pat]}"
        for m in a.models:
            for s in seeds:
                name = f"{m}_{cfg}_s{s}"
                best = os.path.join(p["runs"], name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                r = val_once(best, d, f"pat_{pat}_{m}_s{s}", a.imgsz,
                             p["runs"])
                r.update({"group": pat, "n_images": len(by_pat[pat]),
                          "label": lab[pat], "fold": cfg,
                          "model": m, "seed": s})
                rows.append(r)
        if i % 10 == 0:
            print(f"  {i}/{len(pats)}", flush=True)

    if not rows:
        sys.exit("没有患者折的权重")
    dump(rows, os.path.join(p["results"], "breakhis_by_patient.csv"),
         ["group", "label", "model", "fold", "seed", "n_images"])

    per = defaultdict(list)
    nimg, plab = {}, {}
    for r in rows:
        per[r["group"]].append(r["top1"])
        nimg[r["group"]] = r["n_images"]
        plab[r["group"]] = r["label"]
    m_ = {g: st.mean(v) for g, v in per.items()}
    vals = list(m_.values())
    mu, sd = st.mean(vals), st.stdev(vals)
    print("\n" + "=" * 68)
    print(f"  逐患者（{len(vals)} 个单元，每个由未见过他的模型评分）")
    print("=" * 68)
    print(f"  均值 {mu:.4f}  sd {sd:.4f}  CV {sd/mu:.3f}  "
          f"范围 {min(vals):.3f}-{max(vals):.3f}")
    zero = [g for g, v in m_.items() if v == 0]
    print(f"  完全答错的患者: {len(zero)}  {zero[:8]}")
    print(f"\n  {'patient':<16}{'label':<11}{'n':>5}{'top1':>9}")
    for g in sorted(m_, key=lambda x: m_[x]):
        print(f"  {g:<16}{plab[g]:<11}{nimg[g]:>5}{m_[g]:>9.4f}")
    print("\n  对照: durian per-burst CV 0.75, GWHD per-domain CV 0.31")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["splits", "train", "eval", "per_patient"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--models", nargs="+", default=["yolo11s-cls"],
                    choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-images", type=int, default=MIN_PER_UNIT)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(a.root)
    {"splits": step_splits, "train": step_train, "eval": step_eval,
     "per_patient": step_per_patient}[a.step](a, p)


if __name__ == "__main__":
    main()
