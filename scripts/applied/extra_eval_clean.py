#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extra_eval_clean.py -- 两个补充评估，只用已有的 clean 协议权重，不训练。

1. Sabah 分果园：每个留一果园模型（clean_{model}_byfarm_fold*_s*）在 Sabah 两个果园
   上各评一次（原来只有合并的 Sabah 分数）。
2. 随机切分按果园评：随机切分模型（clean_{model}_random_s*）在它的测试集里，
   按果园分开评，再按果园等权平均——和留一果园的汇总方式完全一样，
   这样"随机切分 vs 新果园"的比较不再混着两种平均方式。

用法（AutoDL，/root/autodl-tmp/durian 下，runs/ 里有 clean_* 权重）：
    python extra_eval_clean.py
    # 结果: results_applied/extra_eval_clean.csv -> 发回来
约 30-40 分钟（4090）。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_budget as fb  # noqa: E402

MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "rtdetr-l"]
SEEDS = [42, 1, 2, 3, 4]


def build(w):
    from ultralytics import YOLO
    if "rtdetr" in w.lower():
        from ultralytics import RTDETR
        return RTDETR(w)
    return YOLO(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=fb.ROOT)
    ap.add_argument("--manifest", default="split_assignment.csv")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    runs = os.path.join(a.root, "runs")
    work = os.path.join(a.root, "extra_eval")
    out = os.path.join(a.root, "results_applied", "extra_eval_clean.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    names = fb.class_names(a.root)
    p = {"splits": work}
    sab = fb.sabah_yamls(a, p, names)

    # random split test set, by farm
    rows = fb.load_manifest(a.root, a.manifest)
    farm_of = {os.path.basename(ip): farm for _, farm, _, ip in rows}
    rv = os.path.join(a.root, "splits", "random", "val.txt")
    test = [l.strip() for l in open(rv, encoding="utf-8") if l.strip()]
    by_farm = {}
    for ip in test:
        f = farm_of.get(os.path.basename(ip))
        if f is not None:
            by_farm.setdefault(f, []).append(ip)
    rand_yaml = {f: fb.write_eval_yaml(work, v, names, f"random_farm{f}")
                 for f, v in sorted(by_farm.items(), key=lambda t: int(t[0]))}
    print("random test images by farm:", {f: len(v) for f, v in sorted(by_farm.items())})

    done, res = set(), []
    if os.path.isfile(out):
        res = list(csv.DictReader(open(out, encoding="utf-8")))
        done = {(r["run"], r["eval_on"]) for r in res}
    keys = ["run", "model", "config", "seed", "eval_on", "n_images", "mAP50"] + \
           [f"AP50::{c}" for c in names]

    def flush():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(res)

    jobs = []
    for m in MODELS:
        for s in SEEDS:
            for f in range(8):
                jobs.append((m, f"byfarm_fold{f}", s,
                             [(k, y) for k, y in sab.items() if k != "sabah"]))
            jobs.append((m, "random", s, [(f"random_farm{f}", y) for f, y in rand_yaml.items()]))
    for i, (m, cfg, s, evals) in enumerate(jobs, 1):
        run = f"clean_{m}_{cfg}_s{s}"
        w = os.path.join(runs, run, "weights", "best.pt")
        if not os.path.isfile(w):
            print(f"  缺权重 {run}")
            continue
        model = None
        for tag, yml in evals:
            if (run, tag) in done:
                continue
            model = model or build(w)
            r = fb.val_one(model, yml, a, os.path.join(work, "_val"), f"{run}_{tag}", names)
            n = len(open(yml.replace(".yaml", ".txt")).read().split())
            r.update({"run": run, "model": m, "config": cfg, "seed": s, "eval_on": tag,
                      "n_images": n})
            res.append(r)
        if model is not None:
            flush()
            print(f"[{i}/{len(jobs)}] {run}", flush=True)
    flush()
    print(f"写出 {out}（{len(res)} 行）")


if __name__ == "__main__":
    main()
