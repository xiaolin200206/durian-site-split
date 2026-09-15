#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_protocol_durian.py -- 外层留出农场完全不参与任何训练决策。

与 clean_protocol_gwhd.py 同一套约定，但有一处必须不同：

★ 内层验证集按主类分层。
  durian 六个类在八个农场之间极不均衡（Phomopsis 只出现在三个农场却占
  6,627 个框，Leaf_rot 出现在七个农场只有 570 个）。外层训练集 680-770 张，
  取 10% 只有 68-77 张；随机切很可能某个类一张都没有，早停的 fitness 就
  不稳，而且各折不稳的方式还不一样。所以按 split_assignment.csv 的
  primary 列分层，每个类至少留一张进内层验证。
  GWHD 是单类，没有这个问题，那边随机切即可。

其余与 GWHD 版本一致：
    原 splits/<cfg>/train.txt  ->  clean_splits/<cfg>/  (90% 训练 / 10% 内层)
    原 splits/<cfg>/val.txt    ->  训练全部结束后评估一次
  新目录、新 run 名一律 clean_ 前缀，不覆盖现有结果；best 与 last 都留；
  外层评估存逐图预测，供以后做单元内 bootstrap。

用法：
    cd /root/autodl-tmp/durian
    python clean_protocol_durian.py --step splits
    python clean_protocol_durian.py --step train            # 先看清单
    screen -S durian
    python clean_protocol_durian.py --step train --yes
    python clean_protocol_durian.py --step eval
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

ROOT = "/root/autodl-tmp/durian"
INNER_FRAC = 0.10
INNER_SEED = 20260911

BATCH = {"yolo11n": 32, "yolo11s": 32, "yolo11m": 16,
         "yolo11l": 8, "rtdetr-l": 24}
DEFAULT_MODELS = ["yolo11n", "yolo11s", "yolo11m", "rtdetr-l"]
DEFAULT_SEEDS = [42, 1, 2, 3, 4]


def P(root):
    return {"root": root,
            "splits": os.path.join(root, "splits"),
            "clean_splits": os.path.join(root, "clean_splits"),
            "runs": os.path.join(root, "runs"),
            "results": os.path.join(root, "results_clean")}


def read_lines(f):
    return [l.strip() for l in open(f, encoding="utf-8") if l.strip()]


def class_names(root):
    import yaml
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns
                  if d not in ("runs", "splits", "clean_splits",
                               "results_v2", "results_clean")]
        if "merged_peninsula" in dns:
            y = os.path.join(dp, "merged_peninsula", "data.yaml")
            if os.path.isfile(y):
                n = yaml.safe_load(open(y, encoding="utf-8"))["names"]
                return [n[k] for k in sorted(n)] if isinstance(n, dict) else n
    sys.exit("找不到 merged_peninsula/data.yaml")


def primary_of(root):
    """stem -> primary class。用于内层验证集的分层。"""
    for cand in (os.path.join(root, "split_assignment.csv"),):
        if os.path.isfile(cand):
            rows = list(csv.DictReader(open(cand, encoding="utf-8-sig",
                                            errors="replace")))
            if "primary" not in rows[0]:
                sys.exit("split_assignment.csv 没有 primary 列，无法分层")
            return {r["stem"].lower(): (r.get("primary") or "none").strip()
                    for r in rows}
    sys.exit(f"找不到 {root}/split_assignment.csv")


def write_split(d, train, val, names):
    import yaml
    os.makedirs(d, exist_ok=True)
    for which, paths in (("train", train), ("val", val)):
        with open(os.path.join(d, which + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(paths) + "\n")
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"train": os.path.join(d, "train.txt"),
                        "val": os.path.join(d, "val.txt"),
                        "nc": len(names), "names": names},
                       fh, sort_keys=False, allow_unicode=True)


def step_splits(a, p):
    names = class_names(p["root"])
    prim = primary_of(p["root"])
    print(f"classes ({len(names)}): {names}")

    cfgs = sorted(d for d in os.listdir(p["splits"])
                  if os.path.isfile(os.path.join(p["splits"], d, "train.txt"))
                  and (d.startswith("byfarm_fold") or d == "random"))
    if not cfgs:
        sys.exit(f"{p['splits']} 下没有 byfarm_fold*/random")

    print(f"\n{len(cfgs)} 个折，内层验证按 primary 分层取 {INNER_FRAC:.0%}"
          f"（固定种子 {INNER_SEED}）\n")
    print(f"  {'fold':<18}{'outer val':>10}{'inner train':>13}"
          f"{'inner val':>11}{'classes in inner':>18}")
    meta = {}
    for cfg in cfgs:
        tr = read_lines(os.path.join(p["splits"], cfg, "train.txt"))
        outer = set(read_lines(os.path.join(p["splits"], cfg, "val.txt")))

        by_cls = defaultdict(list)
        for ip in tr:
            stem = os.path.splitext(os.path.basename(ip))[0].lower()
            by_cls[prim.get(stem, "none")].append(ip)

        rng = random.Random(INNER_SEED)
        inner_val = []
        for cls in sorted(by_cls):
            g = by_cls[cls][:]
            rng.shuffle(g)
            k = max(1, int(round(len(g) * INNER_FRAC))) if len(g) >= 2 else 0
            inner_val += g[:k]
        inner_val_set = set(inner_val)
        inner_train = [x for x in tr if x not in inner_val_set]

        assert not (outer & set(inner_train)), f"{cfg}: 外层图进了内层训练"
        assert not (outer & inner_val_set), f"{cfg}: 外层图进了内层验证"

        write_split(os.path.join(p["clean_splits"], cfg),
                    inner_train, inner_val, names)
        n_cls = len({prim.get(os.path.splitext(os.path.basename(x))[0].lower())
                     for x in inner_val})
        meta[cfg] = {"outer_val": len(outer), "inner_train": len(inner_train),
                     "inner_val": len(inner_val), "inner_classes": n_cls}
        print(f"  {cfg:<18}{len(outer):>10}{len(inner_train):>13}"
              f"{len(inner_val):>11}{n_cls:>18}")

    os.makedirs(p["clean_splits"], exist_ok=True)
    with open(os.path.join(p["clean_splits"], "split_meta.json"), "w") as fh:
        json.dump({"inner_frac": INNER_FRAC, "inner_seed": INNER_SEED,
                   "stratified_by": "primary", "folds": meta}, fh, indent=1)
    print(f"\n写到 {p['clean_splits']}")
    print("外层 val.txt 仍在原 splits/ 下，训练时完全不引用。")


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
    todo = [j for j in jobs if not os.path.isfile(os.path.join(
        p["runs"], f"clean_{j[0]}_{j[1]}_s{j[2]}", "weights", "best.pt"))]
    print(f"配置 {len(cfgs)}: {cfgs}")
    print(f"模型 {a.models}，种子 {a.seeds}")
    print(f"共 {len(jobs)} run，待跑 {len(todo)}")
    print(f"epochs {a.epochs}  patience {a.patience}  imgsz {a.imgsz}")
    print(f"batch: " + ", ".join(f"{m} {a.batch or BATCH[m]}"
                                 for m in a.models))
    print("★ 训练只看内层验证集；外层农场一次都不出现。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return

    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"clean_{model}_{cfg}_s{seed}"
        if os.path.isfile(os.path.join(p["runs"], name, "weights", "best.pt")):
            continue
        el = (time.time() - t0) / 60
        eta = (el / done * (len(todo) - done) / 60) if done else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        try:
            build(f"{model}.pt").train(
                data=os.path.join(p["clean_splits"], cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                batch=a.batch or BATCH[model], workers=a.workers,
                patience=a.patience, project=p["runs"], name=name,
                exist_ok=True, deterministic=True, plots=False,
                verbose=False, save_period=-1)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
            if "out of memory" in str(e).lower():
                print("    显存不足：--batch 调小")
    print(f"\n训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")


def find_sabah(root):
    """merged_sabah 的 images 目录，用于跨区域评估。"""
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns
                  if d not in ("runs", "splits", "clean_splits",
                               "results_v2", "results_clean")]
        if "merged_sabah" in dns:
            d = os.path.join(dp, "merged_sabah")
            if os.path.isdir(os.path.join(d, "images")):
                return d
    return None


def step_eval(a, p):
    import glob as _glob
    import yaml
    from ultralytics import YOLO
    names = class_names(p["root"])
    os.makedirs(p["results"], exist_ok=True)

    # Sabah：从未参与任何训练，用干净权重再评一次，使跨区域结论与主表同协议
    sabah = find_sabah(p["root"])
    sabah_yaml = None
    if sabah:
        imgs = sorted(x for x in _glob.glob(os.path.join(sabah, "images", "*"))
                      if x.lower().endswith((".jpg", ".jpeg", ".png")))
        sd = os.path.join(p["results"], "_sabah")
        os.makedirs(sd, exist_ok=True)
        lst = os.path.join(sd, "val.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(imgs) + "\n")
        sabah_yaml = os.path.join(sd, "sabah.yaml")
        with open(sabah_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"train": lst, "val": lst, "nc": len(names),
                            "names": names}, fh, sort_keys=False,
                           allow_unicode=True)
        print(f"Sabah: {len(imgs)} 张，将用每个 clean 权重各评一次")
    else:
        print("★ 找不到 merged_sabah，跳过跨区域评估")
    cfgs = sorted(d for d in os.listdir(p["clean_splits"])
                  if os.path.isfile(os.path.join(p["clean_splits"], d,
                                                 "data.yaml")))
    rows, preds = [], []
    for model in a.models:
        for cfg in cfgs:
            for seed in a.seeds:
                name = f"clean_{model}_{cfg}_s{seed}"
                ov = os.path.join(p["splits"], cfg, "val.txt")
                od = os.path.join(p["results"], "_outer", cfg)
                os.makedirs(od, exist_ok=True)
                yml = os.path.join(od, "outer.yaml")
                with open(yml, "w", encoding="utf-8") as fh:
                    yaml.safe_dump({"train": ov, "val": ov,
                                    "nc": len(names), "names": names},
                                   fh, sort_keys=False, allow_unicode=True)
                for tag, wf in (("best", "best.pt"), ("last", "last.pt")):
                    w = os.path.join(p["runs"], name, "weights", wf)
                    if not os.path.isfile(w):
                        continue
                    m = build(w)
                    r = m.val(data=yml, imgsz=a.imgsz, split="val",
                              batch=a.batch or BATCH[model],
                              workers=a.workers, project=p["runs"],
                              name=f"{name}_outer_{tag}", exist_ok=True,
                              verbose=False, plots=False)
                    row = {"model": model, "config": cfg, "seed": seed,
                           "checkpoint": tag, "eval_on": "outer",
                           "mAP50": float(r.box.map50),
                           "mAP50_95": float(r.box.map),
                           "precision": float(r.box.mp),
                           "recall": float(r.box.mr)}
                    try:
                        for ci, cname in enumerate(names):
                            if ci in r.box.ap_class_index:
                                j = list(r.box.ap_class_index).index(ci)
                                row[f"AP50::{cname}"] = float(r.box.ap50[j])
                    except Exception:
                        pass
                    rows.append(row)
                    print(f"  {name} [{tag}]  mAP50 {row['mAP50']:.4f}",
                          flush=True)
                    if sabah_yaml and tag == "best":
                        rs = m.val(data=sabah_yaml, imgsz=a.imgsz,
                                   split="val",
                                   batch=a.batch or BATCH[model],
                                   workers=a.workers, project=p["runs"],
                                   name=f"{name}_sabah", exist_ok=True,
                                   verbose=False, plots=False)
                        rows.append({"model": model, "config": cfg,
                                     "seed": seed, "checkpoint": "best",
                                     "eval_on": "sabah",
                                     "mAP50": float(rs.box.map50),
                                     "mAP50_95": float(rs.box.map),
                                     "precision": float(rs.box.mp),
                                     "recall": float(rs.box.mr)})
                    if tag == "best" and a.save_preds:
                        for ip in read_lines(ov):
                            res = m.predict(ip, imgsz=a.imgsz, conf=0.001,
                                            verbose=False)[0]
                            b = res.boxes
                            n = 0 if b is None else len(b)
                            preds.append({
                                "model": model, "config": cfg, "seed": seed,
                                "image": os.path.basename(ip), "n_det": n,
                                "max_conf": (float(b.conf.max()) if n else 0.0),
                                "dets": ("" if not n else ";".join(
                                    f"{c:.4f}:{int(k)}"
                                    for c, k in zip(b.conf.tolist(),
                                                    b.cls.tolist())))})
    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        out = os.path.join(p["results"], "in_region_clean.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=keys)
            w_.writeheader()
            w_.writerows(rows)
        print(f"\n写出 {out}  ({len(rows)} 行)")
    if preds:
        out = os.path.join(p["results"], "outer_predictions.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as fh:
            w_ = csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
            w_.writeheader()
            w_.writerows(preds)
        print(f"写出 {out}  ({len(preds)} 行)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True,
                    choices=["splits", "train", "eval"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="yolo11l 也支持，batch 自动用 8")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None,
                    help="留空则按模型用 32/32/16/24")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-preds", action="store_true", default=True)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(os.path.abspath(a.root))
    {"splits": step_splits, "train": step_train, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
