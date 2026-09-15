#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_unit_clean.py -- 干净协议下的逐单元评估（GWHD 的 47 个 session、
BreaKHis 的 81 个病人）。

为什么必须单独做：现有的 in_region_clean.csv 只有折层面的分数（GWHD 10 个、
BreaKHis 10 个）。但论文里有四样东西要的是**每个单元各自的分数**：

    Fig 1 的每单元分布
    嵌套方差分解（unit within fold）
    评价单元重抽曲线
    模型比较的配对差与翻转率

每个单元都用留出它的那一折的 clean 权重评分，五折加起来覆盖全部单元，
零额外训练。这与旧协议下 gwhd_run.py --step per_domain 和
bkh_run.py --step per_patient 做的是同一件事，只是换成 clean 权重。

★ 与主表同口径：走 Ultralytics 的 val()，不自己写指标。当初弃权分析
  自己写了一套 AP 就和主表差了 0.31，这里不重犯。

用法：
    # GWHD
    python per_unit_clean.py --dataset gwhd --root /root/autodl-tmp/gwhd2021 \
        --models yolo11s yolo11n --seeds 42 1 2 --dry-run
    python per_unit_clean.py --dataset gwhd --root /root/autodl-tmp/gwhd2021 \
        --models yolo11s yolo11n --seeds 42 1 2

    # BreaKHis（含跨族模型则另见脚本内说明）
    python per_unit_clean.py --dataset breakhis --root /root/autodl-tmp/breakhis \
        --models yolo11s-cls yolo11n-cls --seeds 42 1 2 3 4
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict

MIN_PER_UNIT = 5          # 与主文一致：单元少于 5 张不评估


def read_lines(f):
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------ GWHD --
def gwhd_units(root):
    """图片路径 -> domain，以及每折留出了哪些 domain。"""
    man = None
    for f in ("manifest_clean.csv", "manifest.csv"):
        c = os.path.join(root, "processed", f)
        if os.path.isfile(c):
            man = c
            break
    if not man:
        sys.exit(f"找不到 manifest：{root}/processed")
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    dom = {}
    for r in rows:
        for k in ("image_path", "image_id"):
            if r.get(k):
                dom[os.path.splitext(os.path.basename(r[k]))[0]] = r["domain"]
    return dom


def run_gwhd(a):
    import yaml
    from ultralytics import YOLO
    dom = gwhd_units(a.root)
    splits = os.path.join(a.root, "splits")
    outdir = os.path.abspath(os.path.join(a.root, "results_clean"))
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.abspath(os.path.join(outdir, "_perunit"))
    cfgs = sorted(d for d in os.listdir(splits)
                  if d.startswith("domain_fold")
                  and os.path.isfile(os.path.join(splits, d, "val.txt")))

    # 每折 -> {domain: [images]}
    plan = {}
    for cfg in cfgs:
        by = defaultdict(list)
        for ip in read_lines(os.path.join(splits, cfg, "val.txt")):
            d = dom.get(os.path.splitext(os.path.basename(ip))[0])
            if d:
                by[d].append(ip)
        plan[cfg] = {k: v for k, v in by.items() if len(v) >= MIN_PER_UNIT}

    n_units = sum(len(v) for v in plan.values())
    print(f"GWHD: {len(cfgs)} 折, {n_units} 个 domain 达到 {MIN_PER_UNIT} 张门槛")
    for cfg in cfgs:
        print(f"  {cfg:<16}{len(plan[cfg]):>3} domains, "
              f"{sum(len(v) for v in plan[cfg].values()):>5} images")
    jobs = len(a.models) * len(a.seeds) * n_units
    print(f"\n共 {jobs} 次评估")
    if a.dry_run:
        print("--dry-run，未加载模型。")
        return

    rows = []
    done = 0
    for model in a.models:
        for cfg in cfgs:
            for seed in a.seeds:
                w = os.path.join(a.root, "runs",
                                 f"clean_{model}_{cfg}_s{seed}",
                                 "weights", "best.pt")
                if not os.path.isfile(w):
                    print(f"  跳过（无权重）clean_{model}_{cfg}_s{seed}")
                    continue
                m = YOLO(w)
                for unit, imgs in sorted(plan[cfg].items()):
                    d = os.path.abspath(os.path.join(tmp, cfg, unit))
                    os.makedirs(d, exist_ok=True)
                    lst = os.path.join(d, "val.txt")
                    with open(lst, "w", encoding="utf-8") as fh:
                        # ★ 绝对路径。相对路径会被 Ultralytics 再拼一次前缀，
                        #   变成 .../Arvalis_10/gwhd2021/results_clean/... 而找不到。
                        fh.write("\n".join(os.path.abspath(x) for x in imgs)
                                 + "\n")
                    yml = os.path.join(d, "data.yaml")
                    with open(yml, "w", encoding="utf-8") as fh:
                        fh.write(f"path: {d}\ntrain: {lst}\nval: {lst}\n")
                        fh.write("nc: 1\nnames:\n- wheat_head\n")
                    r = m.val(data=yml, imgsz=a.imgsz, split="val",
                              batch=a.batch, workers=a.workers,
                              project=tmp, name="v", exist_ok=True,
                              verbose=False, plots=False)
                    rows.append({"group": unit, "model": model, "fold": cfg,
                                 "seed": seed, "n_images": len(imgs),
                                 "mAP50": float(r.box.map50),
                                 "mAP50_95": float(r.box.map),
                                 "precision": float(r.box.mp),
                                 "recall": float(r.box.mr)})
                    done += 1
                    if done % 25 == 0:
                        print(f"  [{done}/{jobs}]", flush=True)
    out = os.path.join(outdir, "gwhd_by_domain_clean.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print(f"\n写出 {out}  ({len(rows)} 行)")
    shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------- BreaKHis --
def run_breakhis(a):
    from ultralytics import YOLO
    man = os.path.join(a.root, "processed", "manifest_clean.csv")
    if not os.path.isfile(man):
        sys.exit(f"找不到 {man}")
    rows_m = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    outdir = os.path.join(a.root, "results_clean")
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "_perunit")

    by_pat = defaultdict(list)
    fold_of = {}
    for r in rows_m:
        by_pat[r["patient"]].append(r)
        fold_of[r["patient"]] = f"patient_fold{r['patient_fold']}"
    pats = {k: v for k, v in by_pat.items() if len(v) >= MIN_PER_UNIT}
    print(f"BreaKHis: {len(pats)} 个病人达到 {MIN_PER_UNIT} 张门槛")
    jobs = len(a.models) * len(a.seeds) * len(pats)
    print(f"共 {jobs} 次评估")
    if a.dry_run:
        ex = sorted(pats)[0]
        print(f"  示例 {ex} -> {fold_of[ex]}, {len(pats[ex])} 张")
        print("--dry-run，未加载模型。")
        return

    rows, done = [], 0
    cache = {}
    for model in a.models:
        for seed in a.seeds:
            for pat in sorted(pats):
                cfg = fold_of[pat]
                key = f"clean_{model}_{cfg}_s{seed}"
                w = os.path.join(a.root, "runs", key, "weights", "best.pt")
                if not os.path.isfile(w):
                    continue
                if key not in cache:
                    cache[key] = YOLO(w)
                d = os.path.join(tmp, pat)
                if os.path.exists(d):
                    shutil.rmtree(d)
                # 同一批图既当 train 也当 val，只读 val 的指标
                for split in ("train", "val"):
                    for r in pats[pat]:
                        dd = os.path.join(d, split, r["label"])
                        os.makedirs(dd, exist_ok=True)
                        link = os.path.join(dd, r["image_id"] + ".png")
                        if not os.path.exists(link):
                            try:
                                os.symlink(r["image_path"], link)
                            except OSError:
                                shutil.copy2(r["image_path"], link)
                # 分类需要两个类目录都存在
                labs = {r["label"] for r in rows_m}
                for split in ("train", "val"):
                    for lab in labs:
                        os.makedirs(os.path.join(d, split, lab),
                                    exist_ok=True)
                res = cache[key].val(data=d, imgsz=a.imgsz, split="val",
                                     batch=a.batch, workers=a.workers,
                                     project=tmp, name="v", exist_ok=True,
                                     verbose=False, plots=False)
                rows.append({"group": pat, "model": model, "fold": cfg,
                             "seed": seed, "n_images": len(pats[pat]),
                             "label": pats[pat][0]["label"],
                             "top1": float(res.top1)})
                done += 1
                if done % 25 == 0:
                    print(f"  [{done}/{jobs}]", flush=True)
    out = os.path.join(outdir, "breakhis_by_patient_clean.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print(f"\n写出 {out}  ({len(rows)} 行)")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gwhd", "breakhis"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dataset == "gwhd":
        a.imgsz = a.imgsz or 640
        a.batch = a.batch or 16
        run_gwhd(a)
    else:
        a.imgsz = a.imgsz or 224
        a.batch = a.batch or 64
        run_breakhis(a)


if __name__ == "__main__":
    main()
