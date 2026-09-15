#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_protocol_gwhd.py -- 外层留出 domain 完全不参与任何训练决策。

现有协议的问题：每折的 val.txt 就是被留出的 9-10 个 domain，而 Ultralytics
每个 epoch 都在它上面算 fitness、据此早停、据此保留 best.pt。所以外层单元
没进梯度，但进了模型选择。这个脚本把它彻底拿出去。

做法：

    原 splits/<cfg>/train.txt   ->  clean_splits/<cfg>/train.txt  (90%)
                                    clean_splits/<cfg>/val.txt    (10%, 内层)
    原 splits/<cfg>/val.txt     ->  只在训练全部结束后评估一次

内层验证集从**外层训练图像**里随机切 10%，两个 regime（random_fold 与
domain_fold）用同一条规则。这一点是刻意的：主表的核心是两个 arm 的差，
如果内层规则不同，那个差就不能归给划分规则。

代价：每折训练数据少 10%。两个 arm 都少，比较仍然成立，但绝对分数会比
现有结果低一点，Methods 要写明。

★ 与 durian/BreaKHis 的同名脚本保持同一套约定：
  - 新目录、新 run 名一律加 clean_ 前缀，绝不覆盖现有结果
  - best.pt 与 last.pt 都保留
  - 外层评估同时存逐图预测，供以后做单元内 bootstrap（现在做不了是因为
    已发布的表只有分数和张数，没有逐图结果）

用法：
    cd /root/autodl-tmp/gwhd2021
    python clean_protocol_gwhd.py --step splits
    python clean_protocol_gwhd.py --step train --models yolo11s yolo11n   # 先看清单
    screen -S gwhd
    python clean_protocol_gwhd.py --step train --models yolo11s yolo11n --yes
    python clean_protocol_gwhd.py --step eval  --models yolo11s yolo11n
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time

ROOT = "/root/autodl-tmp/gwhd2021"
INNER_FRAC = 0.10
INNER_SEED = 20260911          # 内层划分固定，与训练种子无关


def P(root):
    return {"root": root,
            "splits": os.path.join(root, "splits"),
            "clean_splits": os.path.join(root, "clean_splits"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_clean")}


def read_lines(f):
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip()]


def write_split(d, train, val, nc=1, names=("wheat_head",)):
    os.makedirs(d, exist_ok=True)
    for name, paths in (("train", train), ("val", val)):
        with open(os.path.join(d, name + ".txt"), "w") as fh:
            fh.write("\n".join(paths) + "\n")
    y = os.path.join(d, "data.yaml")
    with open(y, "w") as fh:
        fh.write(f"train: {os.path.join(d, 'train.txt')}\n")
        fh.write(f"val: {os.path.join(d, 'val.txt')}\n")
        fh.write(f"nc: {nc}\nnames:\n")
        for n in names:
            fh.write(f"- {n}\n")
    return y


# ----------------------------------------------------------------- splits --
def step_splits(a, p):
    src = p["splits"]
    if not os.path.isdir(src):
        sys.exit(f"找不到 {src}")
    cfgs = sorted(d for d in os.listdir(src)
                  if os.path.isfile(os.path.join(src, d, "train.txt")))
    if not cfgs:
        sys.exit(f"{src} 下没有 train.txt")

    print(f"{len(cfgs)} 个折，内层验证取训练图像的 {INNER_FRAC:.0%}"
          f"（固定种子 {INNER_SEED}）\n")
    print(f"  {'fold':<18}{'outer val':>10}{'inner train':>13}"
          f"{'inner val':>11}")
    meta = {}
    for cfg in cfgs:
        tr = read_lines(os.path.join(src, cfg, "train.txt"))
        outer = read_lines(os.path.join(src, cfg, "val.txt"))
        rng = random.Random(INNER_SEED)
        shuffled = tr[:]
        rng.shuffle(shuffled)
        n_in = max(1, int(round(len(shuffled) * INNER_FRAC)))
        inner_val, inner_train = shuffled[:n_in], shuffled[n_in:]

        # 外层图像绝不能出现在内层任何一侧
        outer_set = set(outer)
        assert not (outer_set & set(inner_train)), f"{cfg}: 外层图进了内层训练"
        assert not (outer_set & set(inner_val)), f"{cfg}: 外层图进了内层验证"

        write_split(os.path.join(p["clean_splits"], cfg),
                    inner_train, inner_val)
        meta[cfg] = {"outer_val": len(outer), "inner_train": len(inner_train),
                     "inner_val": len(inner_val)}
        print(f"  {cfg:<18}{len(outer):>10}{len(inner_train):>13}"
              f"{len(inner_val):>11}")

    os.makedirs(p["clean_splits"], exist_ok=True)
    with open(os.path.join(p["clean_splits"], "split_meta.json"), "w") as fh:
        json.dump({"inner_frac": INNER_FRAC, "inner_seed": INNER_SEED,
                   "folds": meta}, fh, indent=1)
    print(f"\n写到 {p['clean_splits']}")
    print("外层 val.txt 仍在原 splits/ 下，训练时完全不引用。")


# ------------------------------------------------------------------ train --
def build(w):
    from ultralytics import YOLO
    if "rtdetr" in w.lower():
        try:
            from ultralytics import RTDETR
            return RTDETR(w)
        except ImportError:
            pass
    return YOLO(w)


def step_train(a, p):
    if not os.path.isdir(p["clean_splits"]):
        sys.exit("先跑 --step splits")
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isfile(os.path.join(p["clean_splits"], d,
                                                 "data.yaml")))
    jobs = [(m, c, s) for m in a.models for s in a.seeds for c in cfgs]
    todo = [j for j in jobs
            if not os.path.isfile(os.path.join(
                p["runs"], f"clean_{j[0]}_{j[1]}_s{j[2]}", "weights",
                "best.pt"))]
    print(f"配置 {len(cfgs)}，模型 {a.models}，种子 {a.seeds}")
    print(f"共 {len(jobs)} run，待跑 {len(todo)}")
    print(f"epochs {a.epochs}  patience {a.patience}  imgsz {a.imgsz}  "
          f"batch {a.batch}")
    print("★ 训练只看内层验证集；外层 domain 一次都不出现。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return

    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"clean_{model}_{cfg}_s{seed}"
        if os.path.isfile(os.path.join(p["runs"], name, "weights",
                                       "best.pt")):
            continue
        el = (time.time() - t0) / 60
        eta = (el / done * (len(todo) - done) / 60) if done else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        try:
            build(f"{model}.pt").train(
                data=os.path.join(p["clean_splits"], cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed, batch=a.batch,
                workers=a.workers, patience=a.patience, project=p["runs"],
                name=name, exist_ok=True, deterministic=True, plots=False,
                verbose=False, save_period=-1)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
            if "out of memory" in str(e).lower():
                print("    显存不足：--batch 调小")
    print(f"\n训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# ------------------------------------------------------------------- eval --
def step_eval(a, p):
    """外层评估：训练结束后第一次、也是唯一一次看外层 domain。"""
    from ultralytics import YOLO
    os.makedirs(p["results"], exist_ok=True)
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isfile(os.path.join(p["clean_splits"], d,
                                                 "data.yaml")))
    rows, pred_rows = [], []
    for model in a.models:
        for cfg in cfgs:
            for seed in a.seeds:
                name = f"clean_{model}_{cfg}_s{seed}"
                for tag, wf in (("best", "best.pt"), ("last", "last.pt")):
                    w = os.path.join(p["runs"], name, "weights", wf)
                    if not os.path.isfile(w):
                        continue
                    # 外层 data.yaml：内层的目录，但 val 指向原始外层 val.txt
                    outer_dir = os.path.join(p["results"], "_outer", cfg)
                    os.makedirs(outer_dir, exist_ok=True)
                    ov = os.path.join(p["splits"], cfg, "val.txt")
                    yml = os.path.join(outer_dir, "outer.yaml")
                    with open(yml, "w") as fh:
                        fh.write(f"train: {ov}\nval: {ov}\n")
                        fh.write("nc: 1\nnames:\n- wheat_head\n")
                    m = YOLO(w) if "rtdetr" not in model else build(w)
                    r = m.val(data=yml, imgsz=a.imgsz, split="val",
                              batch=a.batch, workers=a.workers,
                              project=p["runs"], name=f"{name}_outer_{tag}",
                              exist_ok=True, verbose=False, plots=False)
                    rows.append({"model": model, "config": cfg, "seed": seed,
                                 "checkpoint": tag,
                                 "mAP50": float(r.box.map50),
                                 "mAP50_95": float(r.box.map),
                                 "precision": float(r.box.mp),
                                 "recall": float(r.box.mr)})
                    print(f"  {name} [{tag}]  mAP50 {float(r.box.map50):.4f}",
                          flush=True)

                    # 逐图预测只对 best 存一次，供以后做单元内 bootstrap
                    if tag == "best" and a.save_preds:
                        for ip in read_lines(ov):
                            res = m.predict(ip, imgsz=a.imgsz, conf=0.001,
                                            verbose=False)[0]
                            b = res.boxes
                            n = 0 if b is None else len(b)
                            pred_rows.append({
                                "model": model, "config": cfg, "seed": seed,
                                "image": os.path.basename(ip), "n_det": n,
                                "max_conf": (float(b.conf.max()) if n else 0.0),
                                "boxes": ("" if not n else ";".join(
                                    f"{c:.4f}:{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"
                                    for c, (x0, y0, x1, y1) in
                                    zip(b.conf.tolist(), b.xyxy.tolist())))})

    if rows:
        out = os.path.join(p["results"], "in_region_clean.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w_.writeheader()
            w_.writerows(rows)
        print(f"\n写出 {out}  ({len(rows)} 行)")
    if pred_rows:
        out = os.path.join(p["results"], "outer_predictions.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
            w_.writeheader()
            w_.writerows(pred_rows)
        print(f"写出 {out}  ({len(pred_rows)} 行)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True,
                    choices=["splits", "train", "eval"])
    ap.add_argument("--models", nargs="+", default=["yolo11s", "yolo11n"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-preds", action="store_true", default=True)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(os.path.abspath(a.root))
    {"splits": step_splits, "train": step_train, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
