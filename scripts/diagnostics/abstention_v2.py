#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
abstention_v2.py -- 置信度能否携带那些协变量都没有的信号？

Discussion 里的主张是：既然站点级表现无法由任何记录的协变量预测，那么一个
总是给出自信答案的系统就是设定错误的；弃答（把低置信度的样本转给人工）是
让不可预测的站点级失效变得可见的机制。

那是论证。本脚本把它变成结果，而且不需要重新训练。

三件事：

  [1] 每个农场的置信度画像
      用留出该农场的那一折的权重，在该农场上以 conf=0.001 预测，记录每张
      图的最高置信度、检测数、以及 IoU>=0.5 的匹配情况。

  [2] ★ 置信度能否预测哪个农场差
      把「每农场平均最高置信度」与「该农场的 mAP50」做相关。八个协变量
      都失败了（拍摄时间 +0.73 靠单点、病斑尺寸 +0.29、图片数 -0.23）。
      如果置信度也不行，那结论是"连模型自己都不知道"；如果行，那弃答
      就是可用的预警机制。两个方向都是结果。

  [3] 风险—覆盖曲线
      按最高置信度从低到高弃答 x% 的图像，看剩下的 mAP50 怎么变。斜率
      陡说明弃答有效，平说明模型在错的时候一样自信。

用法（在 durian 目录，需要 runs/ 里的 by-farm 权重）：
    python abstention_v2.py --root . --model yolo11s
    python abstention_v2.py --root . --model yolo11s --seeds 42 1
    python abstention_v2.py --root . --model rtdetr-l      # 换架构对照

输出：
    results_v2/abstention_per_image.csv
    results_v2/abstention_summary.csv
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
    """YOLO 归一化 -> 像素 xyxy。"""
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


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def ap50(records):
    """records: [(conf, is_tp)]，加上正样本总数，算 AP50（全точ插值）。"""
    dets, n_gt = records
    if n_gt == 0 or not dets:
        return 0.0
    dets = sorted(dets, key=lambda t: -t[0])
    tp = fp = 0
    prec, rec = [], []
    for _, is_tp in dets:
        tp += int(is_tp)
        fp += int(not is_tp)
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    # 单调化后按召回积分
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap, prev = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev)
        prev = r
    return ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--root", default=".")
    ap_.add_argument("--model", default="yolo11s")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap_.add_argument("--conf", type=float, default=0.001,
                     help="默认 0.25 会先把低置信尾部丢掉，而那正是本分析的对象")
    ap_.add_argument("--imgsz", type=int, default=640)
    a = ap_.parse_args()

    from ultralytics import YOLO
    root = os.path.abspath(a.root)
    splits = os.path.join(root, "splits")
    runs = os.path.join(root, "runs")
    res = os.path.join(root, "results_v2")
    os.makedirs(res, exist_ok=True)

    cfgs = sorted(d for d in os.listdir(splits) if d.startswith("byfarm_fold"))
    if not cfgs:
        sys.exit(f"{splits} 下没有 byfarm_fold*")
    print(f"{len(cfgs)} 折: {cfgs}")

    rows = []
    for cfg in cfgs:
        val_txt = os.path.join(splits, cfg, "val.txt")
        imgs = [l.strip() for l in open(val_txt, encoding="utf-8")
                if l.strip()]
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
                        xy = b.xyxy.tolist()
                        cf = b.conf.tolist()
                        cl = [int(x) for x in b.cls.tolist()]
                        dets = sorted(zip(cf, cl, xy), key=lambda t: -t[0])
                    used = set()
                    matched = []
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
                        matched.append((conf, ok))
                    rows.append({
                        "model": a.model, "config": cfg, "seed": seed,
                        "image": os.path.basename(ip),
                        "n_gt": len(gt), "n_det": len(matched),
                        "max_conf": max((c for c, _ in matched), default=0.0),
                        "mean_conf": (st.mean([c for c, _ in matched])
                                      if matched else 0.0),
                        "n_tp": sum(1 for _, ok in matched if ok),
                        "dets": ";".join(f"{c:.4f}:{int(ok)}"
                                         for c, ok in matched),
                    })
                if (i // 32) % 5 == 0:
                    print(f"    {min(i+32,len(imgs))}/{len(imgs)}", flush=True)

    if not rows:
        sys.exit("没有产出任何预测")

    out = os.path.join(res, "abstention_per_image.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {out}  ({len(rows)} 行)")

    # ---------------- [1] 每农场画像 ----------------
    print("\n" + "=" * 74)
    print("[1] 每农场的置信度画像")
    print("=" * 74)
    by_farm = defaultdict(list)
    for r in rows:
        by_farm[r["config"]].append(r)

    farm_stat = {}
    print(f"  {'fold':<16}{'imgs':>6}{'mean maxconf':>14}"
          f"{'silent@.25':>12}{'AP50':>8}")
    for cfg in sorted(by_farm):
        v = by_farm[cfg]
        mc = st.mean([r["max_conf"] for r in v])
        silent = sum(1 for r in v if r["max_conf"] < 0.25) / len(v)
        dets, n_gt = [], 0
        for r in v:
            n_gt += r["n_gt"]
            for d in filter(None, r["dets"].split(";")):
                c, ok = d.split(":")
                dets.append((float(c), ok == "1"))
        a50 = ap50((dets, n_gt))
        farm_stat[cfg] = {"mean_conf": mc, "silent": silent, "ap50": a50,
                          "n": len(v)}
        print(f"  {cfg:<16}{len(v):>6}{mc:>14.4f}"
              f"{100*silent:>11.1f}%{a50:>8.4f}")

    # ---------------- [2] 置信度能否预测 ----------------
    print("\n" + "=" * 74)
    print("[2] ★ 置信度能否预测哪个农场差")
    print("=" * 74)
    ks = sorted(farm_stat)
    r_conf = pearson([farm_stat[k]["mean_conf"] for k in ks],
                     [farm_stat[k]["ap50"] for k in ks])
    r_sil = pearson([farm_stat[k]["silent"] for k in ks],
                    [farm_stat[k]["ap50"] for k in ks])
    print(f"  平均最高置信度  vs  该农场 AP50 :  r = {r_conf:+.3f}  (n={len(ks)})")
    print(f"  低置信沉默率    vs  该农场 AP50 :  r = {r_sil:+.3f}")
    print("""
  对照（八个记录的协变量，均不可用）：
    拍摄时间 +0.73（靠单个农场支撑）  病斑尺寸 +0.29
    图片数 -0.23   正午占比 -0.35   太阳高度角 -0.32

  |r| 大且方向为正 -> 模型在它会失败的站点上确实不自信，
                      弃答可以作为部署时的预警信号。
  |r| 接近零       -> 连模型的置信度也不携带这个信号，
                      失效在部署时不可见，这是更强的负面结论。""")

    # ---------------- [3] 风险—覆盖曲线 ----------------
    print("\n" + "=" * 74)
    print("[3] 风险—覆盖：按最高置信度从低到高弃答")
    print("=" * 74)
    order = sorted(rows, key=lambda r: r["max_conf"])
    n = len(order)
    print(f"  {'弃答比例':>10}{'保留图像':>10}{'保留AP50':>10}"
          f"{'弃答图的GT占比':>16}")
    for frac in (0.0, 0.1, 0.2, 0.3, 0.5):
        keep = order[int(n * frac):]
        drop = order[:int(n * frac)]
        dets, n_gt = [], 0
        for r in keep:
            n_gt += r["n_gt"]
            for d in filter(None, r["dets"].split(";")):
                c, ok = d.split(":")
                dets.append((float(c), ok == "1"))
        gt_dropped = sum(r["n_gt"] for r in drop)
        gt_all = sum(r["n_gt"] for r in rows)
        print(f"  {100*frac:>9.0f}%{len(keep):>10}"
              f"{ap50((dets, n_gt)):>10.4f}"
              f"{100*gt_dropped/max(gt_all,1):>15.1f}%")
    print("""
  斜率陡  -> 弃答换回来的精度多，代价是漏掉的病斑少，机制有效。
  斜率平  -> 模型在错的时候一样自信，弃答换不来什么。
  最后一列是代价：被弃答的图里含有多少真实病斑，那些是漏检。""")

    out2 = os.path.join(res, "abstention_summary.csv")
    with open(out2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_images", "mean_max_conf",
                    "silent_rate_at_0.25", "ap50"])
        for k in ks:
            s = farm_stat[k]
            w.writerow([k, s["n"], round(s["mean_conf"], 4),
                        round(s["silent"], 4), round(s["ap50"], 4)])
    print(f"\n写出 {out2}")


if __name__ == "__main__":
    main()
