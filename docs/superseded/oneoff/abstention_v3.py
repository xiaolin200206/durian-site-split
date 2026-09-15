#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abstention_v3.py -- v2 的修正版。

v2 有两个问题，都会让它的 AP50 与 in_region.csv 里的 mAP50 对不上：

  [A] 类无关合并。v2 在匹配时用了类别（gc != cls 会跳过），但匹配完
      就把类别丢了：matched.append((conf, ok))。ap50() 于是把六个类的
      检测混成一条置信度排序曲线，算出来的是一个实例加权的
      class-agnostic AP，而 Ultralytics 的 mAP50 是各类 AP 的平均。
      在 farm 6 上差别最大：四个有标注的类 AP 是 0.00/0.00/0.02/0.42，
      mAP=0.13，但 Phomopsis 占了绝大多数框，合并曲线被拉到 0.30。

  [B] 单种子。v2 默认 --seeds 42，主表是五种子均值。这一项影响很小
      （rtdetr-l 上最大 0.041），但没有理由不做对。

v3 的改动：
  - matched 里保留类别；AP50 按类计算，再对「该农场出现过的类」取平均，
    与 Ultralytics 的口径一致（未出现的类不进平均）。
  - 默认跑全部五个种子，逐农场对种子取平均。
  - 新增 --check，把结果与 results_durian/in_region.csv 的 mAP50 逐农场
    对照并打印差值。这一列应当在 0.02 以内；如果不是，说明还有第三处
    口径差异，不要把结果写进稿子。
  - 置信度与沉默率是类无关的量，照旧计算，但也跨种子平均。

风险—覆盖曲线同样改为按类计算再平均，所以稿子里
「弃答 50% 使 AP50 从 0.193 升到 0.242」这两个数会变，需要重跑后更新。

用法（在 durian 目录，需要 runs/ 的权重与 splits/<cfg>/val.txt）：
    python abstention_v3.py --root . --model rtdetr-l --check --seeds 42
    python abstention_v3.py --root . --model rtdetr-l --check
"""

import argparse
import csv
import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict

IOU_T = 0.5
CLASS_NAMES = ["Algal", "Leaf_rot", "Psyllid", "Psyllid_damage",
               "leaf_hopper_damage", "Phomopsis"]


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def load_gt(label_path, W, H):
    out = []
    if not os.path.isfile(label_path):
        return out
    for line in open(label_path, encoding="utf-8", errors="replace"):
        v = line.split()
        if len(v) < 5:
            continue
        try:
            c = int(float(v[0]))
            cx, cy, w, h = (float(x) for x in v[1:5])
        except ValueError:
            continue
        out.append((c, [(cx - w / 2) * W, (cy - h / 2) * H,
                        (cx + w / 2) * W, (cy + h / 2) * H]))
    return out


def ap_single_class(dets, n_gt):
    """dets: [(conf, is_tp)] 单一类别。全点插值 AP。"""
    if n_gt == 0:
        return None            # 该类不在这个农场出现，不进平均
    if not dets:
        return 0.0
    dets = sorted(dets, key=lambda t: -t[0])
    tp = fp = 0
    prec, rec = [], []
    for _, is_tp in dets:
        tp += int(is_tp)
        fp += int(not is_tp)
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap, prev = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev)
        prev = r
    return ap


def map50(dets_by_class, gt_by_class):
    """★ 与 v2 的关键差异：按类算 AP，再对出现过的类取平均。"""
    per = {}
    for c in set(list(dets_by_class) + list(gt_by_class)):
        a = ap_single_class(dets_by_class.get(c, []), gt_by_class.get(c, 0))
        if a is not None:
            per[c] = a
    if not per:
        return float("nan"), {}
    return sum(per.values()) / len(per), per


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    return pearson(rank(x), rank(y))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--root", default=".")
    ap_.add_argument("--model", default="rtdetr-l")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap_.add_argument("--conf", type=float, default=0.001)
    ap_.add_argument("--imgsz", type=int, default=640)
    ap_.add_argument("--check", action="store_true",
                     help="与 results_durian/in_region.csv 对照")
    ap_.add_argument("--ref", default=None,
                     help="对照表的路径，默认 <root>/results_durian/in_region.csv")
    a = ap_.parse_args()

    from ultralytics import YOLO   # 延迟导入，--check 之外才需要

    runs = os.path.join(a.root, "runs")
    res = os.path.join(a.root, "results_v3")
    os.makedirs(res, exist_ok=True)

    folds = sorted(d for d in os.listdir(os.path.join(a.root, "splits"))
                   if d.startswith("byfarm_fold")
                   and os.path.isfile(os.path.join(a.root, "splits", d,
                                                   "val.txt")))
    if not folds:
        sys.exit(f"{os.path.join(a.root, 'splits')} 下没有 byfarm_fold*/val.txt")

    rows = []
    for cfg in folds:
        # ★ 图片清单在 splits/<cfg>/val.txt，一行一个路径（与 abstention_v2
        #   一致）。早先这里假设了 splits/<cfg>/val/images/ 的目录树，那是
        #   分类任务的结构，在这个检测语料上会读到 0 张。
        val_txt = os.path.join(a.root, "splits", cfg, "val.txt")
        if not os.path.isfile(val_txt):
            print(f"  跳过 {cfg}（没有 {val_txt}）")
            continue
        imgs = [l.strip() for l in open(val_txt, encoding="utf-8")
                if l.strip()]
        imgs = [p_ if os.path.isabs(p_) else os.path.join(a.root, p_)
                for p_ in imgs]
        imgs = sorted(p_ for p_ in imgs if os.path.isfile(p_))
        for seed in a.seeds:
            name = f"{a.model}_{cfg}_s{seed}"
            best = os.path.join(runs, name, "weights", "best.pt")
            if not os.path.isfile(best):
                print(f"  跳过 {name}（无权重）")
                continue
            m = YOLO(best)
            print(f"\n{name}: {len(imgs)} 张", flush=True)
            for i in range(0, len(imgs), 32):
                batch = imgs[i:i + 32]
                preds = m.predict(batch, conf=a.conf, imgsz=a.imgsz,
                                  verbose=False)
                for ip, r in zip(batch, preds):
                    H, W = r.orig_shape
                    lp = (ip.replace("/images/", "/labels/")
                            .rsplit(".", 1)[0] + ".txt")
                    gt = load_gt(lp, W, H)
                    b = r.boxes
                    dets = []
                    if b is not None and len(b):
                        dets = sorted(zip(b.conf.tolist(),
                                          [int(x) for x in b.cls.tolist()],
                                          b.xyxy.tolist()),
                                      key=lambda t: -t[0])
                    used = set()
                    matched = []          # ★ 保留类别
                    for conf, cls, box in dets:
                        bi, bv = None, 0.0
                        for j, (gc, gb) in enumerate(gt):
                            if j in used or gc != cls:
                                continue
                            v = iou_xyxy(box, gb)
                            if v > bv:
                                bv, bi = v, j
                        ok = bv >= IOU_T and bi is not None
                        if ok:
                            used.add(bi)
                        matched.append((conf, cls, ok))
                    gt_count = defaultdict(int)
                    for gc, _ in gt:
                        gt_count[gc] += 1
                    rows.append({
                        "model": a.model, "config": cfg, "seed": seed,
                        "image": os.path.basename(ip),
                        "n_gt": len(gt), "n_det": len(matched),
                        "max_conf": max((c for c, _, _ in matched), default=0.0),
                        "gt_by_class": ";".join(f"{c}:{n}" for c, n in
                                                sorted(gt_count.items())),
                        # ★ 类别写进 dets，下游才能按类重算
                        "dets": ";".join(f"{c:.4f}:{cl}:{int(ok)}"
                                         for c, cl, ok in matched),
                    })

    if not rows:
        sys.exit("没有读到任何图像。检查 splits/<cfg>/val.txt 里的路径是否"
                 "相对于 --root，以及 runs/ 下是否有对应权重。")

    out = os.path.join(res, "abstention_per_image.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {out}  ({len(rows)} 行)")

    # ---- 每农场：按类 AP -> mAP50，跨种子平均 ----
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["seed"]].append(r)

    summary = {}
    for cfg, seeds in by.items():
        maps, confs, silents = [], [], []
        for seed, rs in seeds.items():
            d, g = defaultdict(list), defaultdict(int)
            for r in rs:
                for tok in filter(None, r["dets"].split(";")):
                    c, cl, ok = tok.split(":")
                    d[int(cl)].append((float(c), ok == "1"))
                for tok in filter(None, r["gt_by_class"].split(";")):
                    cl, n = tok.split(":")
                    g[int(cl)] += int(n)
            mp, _ = map50(d, g)
            maps.append(mp)
            confs.append(st.mean(float(r["max_conf"]) for r in rs))
            silents.append(st.mean(1.0 if float(r["max_conf"]) < 0.25 else 0.0
                                   for r in rs))
        summary[cfg] = {"map50": st.mean(maps), "map50_sd":
                        st.stdev(maps) if len(maps) > 1 else 0.0,
                        "mean_conf": st.mean(confs),
                        "silent": st.mean(silents),
                        "n_images": len(next(iter(seeds.values())))}

    print(f"\n{'fold':<16}{'imgs':>6}{'meanconf':>10}{'silent@.25':>12}"
          f"{'mAP50':>9}{'sd':>7}")
    for k in sorted(summary):
        s = summary[k]
        print(f"{k:<16}{s['n_images']:>6}{s['mean_conf']:>10.3f}"
              f"{s['silent']:>12.3f}{s['map50']:>9.3f}{s['map50_sd']:>7.3f}")

    ks = sorted(summary)
    mc = [summary[k]["mean_conf"] for k in ks]
    sr = [summary[k]["silent"] for k in ks]
    mp = [summary[k]["map50"] for k in ks]
    print(f"\n  平均最高置信度 vs mAP50 : r = {pearson(mc, mp):+.3f}  "
          f"rho = {spearman(mc, mp):+.3f}  (n={len(ks)})")
    print(f"  沉默率         vs mAP50 : r = {pearson(sr, mp):+.3f}  "
          f"rho = {spearman(sr, mp):+.3f}")

    # ---- 与主表对照 ----
    if a.check:
        path = a.ref or os.path.join(a.root, "results_durian",
                                     "in_region.csv")
        if not os.path.isfile(path):
            print(f"\n--check 跳过：找不到 {path}")
            print("  这张表在论文仓库里，不在训练机上。把它拷过来，或用")
            print("  --ref /path/to/in_region.csv 指定，或把上面那张 mAP50")
            print("  表直接和仓库里的 in_region.csv 逐格对照。")
            path = None
    if a.check and path:
        ref = defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            if r["model"] == a.model and r["config"].startswith("byfarm"):
                ref[r["config"]].append(float(r["mAP50"]))
        print(f"\n{'fold':<16}{'v3':>9}{'in_region':>11}{'diff':>8}")
        worst = 0.0
        for k in sorted(summary):
            if k not in ref:
                continue
            r5 = st.mean(ref[k])
            d = summary[k]["map50"] - r5
            worst = max(worst, abs(d))
            print(f"{k:<16}{summary[k]['map50']:>9.3f}{r5:>11.3f}{d:>+8.3f}")
        print(f"\n  最大差值 {worst:.3f} —— "
              f"{'口径一致，可以用' if worst < 0.02 else '仍有第三处差异，不要写进稿子'}")

    with open(os.path.join(res, "abstention_summary.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_images", "mean_max_conf",
                    "silent_rate_at_0.25", "map50", "map50_sd_across_seeds"])
        for k in sorted(summary):
            s = summary[k]
            w.writerow([k, s["n_images"], round(s["mean_conf"], 4),
                        round(s["silent"], 4), round(s["map50"], 4),
                        round(s["map50_sd"], 4)])


if __name__ == "__main__":
    main()
