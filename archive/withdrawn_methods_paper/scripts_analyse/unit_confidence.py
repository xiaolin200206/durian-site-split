#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unit_confidence.py -- 每个评价单元的模型置信度，只做推理。

榴莲上八个农场的结论是：平均最高置信度与农场得分的相关是 +0.09，沉默率与
得分是 +0.30（正相关，即模型在做得好的地方更常弃答）。八个点太少，只能算
迹象。把同一分析扩到 GWHD 的 47 个 session 和 BreaKHis 的 81 个病人，如果
零相关在三个学科上都成立，就是跨域结论。

不训练。每个单元用「没见过它的那一折」的权重推理一次。

指标：
  分类（BreaKHis）  mean max prob、mean entropy、mean margin（top1 − top2）
  检测（GWHD）      mean max confidence、silence rate（最高分 < 阈值的图占比）

★ 与 durian 那次的教训一致：路径不猜。BreaKHis 从 manifest_clean.csv 读
  image_path 与 patient；GWHD 从 splits/<cfg>/val.txt 读图，再用 manifest
  把图映射到 domain。找不到就报告并退出，不会静默产出错数字。

用法：
    python unit_confidence.py --dataset breakhis --root /root/autodl-tmp/breakhis --dry-run
    python unit_confidence.py --dataset breakhis --root /root/autodl-tmp/breakhis
    python unit_confidence.py --dataset gwhd --root /root/autodl-tmp/gwhd2021 --dry-run
"""

import argparse
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

SILENCE_T = 0.25


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    return pearson(rank(x), rank(y))


def report(name, units, key, score, extra=""):
    xs = [u[key] for u in units if not math.isnan(u[key])]
    ys = [u["score"] for u in units if not math.isnan(u[key])]
    r, rho = pearson(xs, ys), spearman(xs, ys)
    print(f"  {name:<22} r = {r:+.3f}   rho = {rho:+.3f}   (n={len(xs)}){extra}")
    return r, rho


# ------------------------------------------------------------- BreaKHis --
def run_breakhis(a):
    man = os.path.join(a.root, "processed", "manifest_clean.csv")
    if not os.path.isfile(man):
        sys.exit(f"找不到 {man}")
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    need = {"image_path", "patient", "patient_fold", "label"}
    if not need <= set(rows[0]):
        sys.exit(f"manifest 缺列，需要 {need}，实际 {set(rows[0])}")

    by_unit = defaultdict(list)
    fold_of = {}
    for r in rows:
        by_unit[r["patient"]].append(r["image_path"])
        fold_of[r["patient"]] = f"patient_fold{r['patient_fold']}"

    print(f"BreaKHis: {len(by_unit)} 个病人, {len(rows)} 张图")
    missing = [u for u in by_unit
               if not os.path.isfile(os.path.join(
                   a.root, "runs",
                   f"{a.model}_{fold_of[u]}_s{a.seeds[0]}", "weights",
                   "best.pt"))]
    if missing:
        print(f"  ★ {len(missing)} 个病人没有对应折的权重，例如 "
              f"{missing[0]} -> {fold_of[missing[0]]}")
    if a.dry_run:
        ex = sorted(by_unit)[0]
        print(f"  示例: {ex} -> {fold_of[ex]}, {len(by_unit[ex])} 张")
        print(f"  第一张: {by_unit[ex][0]}")
        print(f"  存在: {os.path.isfile(by_unit[ex][0])}")
        return None

    from ultralytics import YOLO
    cache = {}
    out = []
    for i, unit in enumerate(sorted(by_unit), 1):
        cfg = fold_of[unit]
        probs_all, ents, margins = [], [], []
        for seed in a.seeds:
            name = f"{a.model}_{cfg}_s{seed}"
            w = os.path.join(a.root, "runs", name, "weights", "best.pt")
            if not os.path.isfile(w):
                continue
            if name not in cache:
                cache[name] = YOLO(w)
            m = cache[name]
            imgs = by_unit[unit]
            for j in range(0, len(imgs), 64):
                for res in m.predict(imgs[j:j + 64], imgsz=a.imgsz,
                                     verbose=False):
                    p = res.probs.data.tolist()
                    p = sorted(p, reverse=True)
                    probs_all.append(p[0])
                    margins.append(p[0] - (p[1] if len(p) > 1 else 0.0))
                    ents.append(-sum(q * math.log(q + 1e-12) for q in p))
        if not probs_all:
            continue
        out.append({"unit": unit, "n": len(by_unit[unit]),
                    "mean_max_prob": st.mean(probs_all),
                    "mean_entropy": st.mean(ents),
                    "mean_margin": st.mean(margins)})
        if i % 10 == 0:
            print(f"  [{i}/{len(by_unit)}]", flush=True)
    return out


# ----------------------------------------------------------------- GWHD --
def _stem(p):
    """去掉目录与扩展名，供 image_id 与 val.txt 的路径互相比对。"""
    return os.path.splitext(os.path.basename(p))[0]


def gwhd_domain_map(a):
    """图片 -> domain。优先 manifest_clean.csv（去重后的那张）。

    ★ image_id 是 'test_000000'，val.txt 里是 'test_000000.jpg'，所以按
      文件名直接比会全部落空。这里统一去掉扩展名再比，并且同时索引
      image_path 与 image_id，哪一列对得上都行。
    """
    proc = os.path.join(a.root, "processed")
    if not os.path.isdir(proc):
        sys.exit(f"找不到 {proc}")
    cands = [f for f in os.listdir(proc) if f.endswith(".csv")]
    order = ([f for f in cands if "clean" in f.lower()]
             + [f for f in cands if "clean" not in f.lower()])
    for f in order:
        path = os.path.join(proc, f)
        try:
            rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        except Exception:
            continue
        if not rows:
            continue
        cols = set(rows[0])
        dom = next((c for c in cols
                    if c.lower() in ("domain", "group", "session",
                                     "acquisition_domain")), None)
        if not dom:
            continue
        imgcols = [c for c in ("image_path", "path", "image", "file",
                               "filename", "image_id") if c in cols]
        if not imgcols:
            continue
        mp = {}
        for r in rows:
            for c in imgcols:
                if r[c]:
                    mp[_stem(r[c])] = r[dom]
        print(f"  用 {f}: 图片列 {imgcols}, domain 列 '{dom}', "
              f"{len(rows)} 行, {len(mp)} 个可比对的键")
        return mp, f
    sys.exit(f"processed/ 下没有同时含 domain 与图片列的表。看到: {cands}")


def run_gwhd(a):
    mp, src = gwhd_domain_map(a)
    splits = os.path.join(a.root, "splits")
    cfgs = sorted(d for d in os.listdir(splits)
                  if d.startswith("domain_fold")
                  and os.path.isfile(os.path.join(splits, d, "val.txt")))
    by_unit = defaultdict(list)
    fold_of = {}
    unmapped = 0
    for cfg in cfgs:
        for line in open(os.path.join(splits, cfg, "val.txt"),
                         encoding="utf-8"):
            ip = line.strip()
            if not ip:
                continue
            d = mp.get(_stem(ip))
            if d is None:
                unmapped += 1
                continue
            by_unit[d].append(ip)
            fold_of[d] = cfg
    print(f"GWHD: {len(cfgs)} 折, {len(by_unit)} 个 domain, "
          f"{sum(len(v) for v in by_unit.values())} 张图"
          + (f", {unmapped} 张映射不到 domain" if unmapped else ""))
    if unmapped:
        print("  ★ 有图映射不到 domain，先查 " + src + " 的图片列是否与 "
              "val.txt 的文件名一致，不要带着这个跑下去")
    if not by_unit:
        sys.exit("没有任何图片映射到 domain，不要继续。")
    if a.dry_run:
        ex = sorted(by_unit)[0]
        print(f"  示例: {ex} -> {fold_of[ex]}, {len(by_unit[ex])} 张")
        print(f"  第一张: {by_unit[ex][0]}  存在: "
              f"{os.path.isfile(by_unit[ex][0])}")
        sizes = sorted(len(v) for v in by_unit.values())
        print(f"  每个 domain 的图片数: {sizes[0]} 到 {sizes[-1]}")
        return None

    from ultralytics import YOLO
    cache = {}
    out = []
    for i, unit in enumerate(sorted(by_unit), 1):
        cfg = fold_of[unit]
        confs, silent = [], []
        for seed in a.seeds:
            name = f"{a.model}_{cfg}_s{seed}"
            w = os.path.join(a.root, "runs", name, "weights", "best.pt")
            if not os.path.isfile(w):
                continue
            if name not in cache:
                cache[name] = YOLO(w)
            m = cache[name]
            imgs = by_unit[unit]
            for j in range(0, len(imgs), 32):
                for res in m.predict(imgs[j:j + 32], imgsz=a.imgsz,
                                     conf=0.001, verbose=False):
                    b = res.boxes
                    c = (float(b.conf.max()) if b is not None and len(b)
                         else 0.0)
                    confs.append(c)
                    silent.append(1.0 if c < SILENCE_T else 0.0)
        if not confs:
            continue
        out.append({"unit": unit, "n": len(by_unit[unit]),
                    "mean_max_conf": st.mean(confs),
                    "silent_rate": st.mean(silent)})
        if i % 10 == 0:
            print(f"  [{i}/{len(by_unit)}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["breakhis", "gwhd"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.dataset == "breakhis":
        a.model = a.model or "yolo11s-cls"
        a.imgsz = a.imgsz or 224
        units = run_breakhis(a)
        keys = ["mean_max_prob", "mean_entropy", "mean_margin"]
        score_file = ("results_v2", "breakhis_by_patient.csv", "group",
                      "top1", a.model)
    else:
        a.model = a.model or "yolo11s"
        a.imgsz = a.imgsz or 640
        units = run_gwhd(a)
        keys = ["mean_max_conf", "silent_rate"]
        score_file = ("results_v2", "gwhd_by_domain.csv", "group",
                      "mAP50", a.model)

    if units is None:
        print("\n--dry-run，未加载模型。")
        return 0

    # 接上每个单元的得分
    sub, fname, ukey, metric, mval = score_file
    path = os.path.join(a.root, sub, fname)
    scores = defaultdict(list)
    if os.path.isfile(path):
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            if r.get("model") not in (None, mval):
                continue
            try:
                scores[r[ukey]].append(float(r[metric]))
            except (ValueError, KeyError):
                pass
    else:
        print(f"\n★ 找不到 {path}，只写出置信度，相关性请离线算")
    for u in units:
        v = scores.get(u["unit"], [])
        u["score"] = st.mean(v) if v else float("nan")

    out = a.out or os.path.join(a.root,
                                f"unit_confidence_{a.dataset}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(units[0].keys()))
        w.writeheader()
        w.writerows(units)
    print(f"\n写出 {out}  ({len(units)} 个单元)")

    have = [u for u in units if not math.isnan(u["score"])]
    if len(have) < 3:
        print("没有足够的单元得分，相关性跳过")
        return 0
    print(f"\n与每单元得分的相关（n={len(have)}）：")
    for k in keys:
        report(k, have, k, "score")
    print("\n榴莲上的对照：mean max confidence r = +0.09，"
          "silence rate r = +0.30（n=8）。")
    print("若这里也接近零，就不是八个农场的偶然。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
