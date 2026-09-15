#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autodl_run.py -- colab_run_v2.py 的 AutoDL 版本。

差别只在路径与持久化：
  - 没有 Drive 挂载。数据与产出都放在 --root 下（AutoDL 上用
    /root/autodl-tmp，那是数据盘，重启不丢；系统盘 /root 只有 30 GB
    且会随实例释放）。
  - 解压后不再复制到别处，直接在数据盘上训练。
  - 断点续跑逻辑不变：有 best.pt 的 run 自动跳过。

准备（在 AutoDL 终端里）：
    mkdir -p /root/autodl-tmp/durian && cd /root/autodl-tmp/durian
    # 上传 durian_colab.zip 和 split_assignment.csv 到这个目录
    pip install ultralytics

跑：
    python autodl_run.py --step unpack
    python autodl_run.py --step splits
    python autodl_run.py --step train --models yolo11n rtdetr-l --yes
    python autodl_run.py --step eval  --models yolo11n rtdetr-l
    python autodl_run.py --step cross --models yolo11n rtdetr-l

后台跑（关掉终端也不停）：
    nohup python autodl_run.py --step train --models yolo11n rtdetr-l --yes \
        > train.log 2>&1 &
    tail -f train.log
"""

import argparse
import csv as _csv
import glob
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict

# ---------------------------------------------------------------- settings --
ROOT = "/root/autodl-tmp/durian"          # 数据盘，重启不丢

MODELS = {
    "yolo11n":  {"weights": "yolo11n.pt",  "batch": 32},
    "yolo11s":  {"weights": "yolo11s.pt",  "batch": 32},
    "yolo11m":  {"weights": "yolo11m.pt",  "batch": 16},
    "yolo11l":  {"weights": "yolo11l.pt",  "batch": 8},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "batch": 8},
}
SEEDS = [42, 1, 2, 3, 4]      # 完全交叉：每个架构每折都要 5 个重复

IMGSZ = 640
EPOCHS = 150
PATIENCE = 50
WORKERS = 8

# 每 run 粗略耗时（分钟，A100）。只用于开跑前给个预算。
MIN_PER_RUN = {"yolo11n": 4.5, "yolo11s": 6.3, "yolo11m": 12,
               "yolo11l": 18, "rtdetr-l": 19}
# ---------------------------------------------------------------------------


def paths(root):
    return {
        "work": root,
        "runs": os.path.join(root, "runs"),
        "results": os.path.join(root, "results_v2"),
        "splits": os.path.join(root, "splits"),
    }


def find_one(root, *names):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("runs", "splits", "results_v2")]
        for n in names:
            if n in dns:
                return os.path.join(dp, n)
            if n in fns:
                return os.path.join(dp, n)
    return None


def load_names(merged):
    import yaml
    with open(os.path.join(merged, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def gpu_line():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name}, {p.total_memory/2**30:.0f} GB"
    except Exception:
        pass
    return "no CUDA visible"


# ------------------------------------------------------------------ unpack --
def step_unpack(a, P):
    z = a.archive or os.path.join(P["work"], "durian_colab.zip")
    if not os.path.isfile(z):
        sys.exit(f"找不到 {z}\n把 durian_colab.zip 上传到 {P['work']}")
    print(f"extracting {z} -> {P['work']}")
    t0 = time.time()
    with zipfile.ZipFile(z) as zf:
        zf.extractall(P["work"])
    print(f"done in {time.time()-t0:.0f}s")
    for key in ("merged_peninsula", "merged_sabah"):
        p = find_one(P["work"], key)
        if p:
            print(f"  {key}: images "
                  f"{len(glob.glob(os.path.join(p,'images','*.*')))}, "
                  f"labels {len(glob.glob(os.path.join(p,'labels','*.txt')))}")
        else:
            print(f"  {key}: NOT FOUND")


# ------------------------------------------------------------------ splits --
def step_splits(a, P):
    """不重算划分。读 split_assignment.csv，只把路径改写成本机的。"""
    import yaml

    merged = find_one(P["work"], "merged_peninsula")
    if not merged:
        sys.exit("先跑 --step unpack")
    names = load_names(merged)
    print(f"classes ({len(names)}): {names}")

    sa = a.split_assignment
    if not sa or not os.path.isfile(sa):
        for c in (os.path.join(P["work"], "split_assignment.csv"),
                  find_one(P["work"], "split_assignment.csv")):
            if c and os.path.isfile(c):
                sa = c
                break
    if not sa:
        sys.exit(f"找不到 split_assignment.csv，放到 {P['work']} 下")
    print(f"split_assignment: {sa}")

    with open(sa, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(_csv.DictReader(fh))
    cols = [c for c in rows[0] if c.startswith("split_")]
    cols.sort(key=lambda c: (not c.endswith("random"), c))
    print(f"划分列 ({len(cols)}): {cols}")

    local = {}
    for p in glob.glob(os.path.join(merged, "images", "*.*")):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            local[os.path.splitext(os.path.basename(p))[0].lower()] = p
    print(f"本地图片 {len(local)}")
    missing = [r["stem"] for r in rows if r["stem"].lower() not in local]
    if missing:
        print(f"★ {len(missing)} 个 stem 找不到，例: {missing[:5]}")
        if len(missing) > len(rows) * 0.02:
            sys.exit("缺失过多，archive 与 split_assignment 不是同一批数据")

    if os.path.exists(P["splits"]):
        shutil.rmtree(P["splits"])
    made = []
    for col in cols:
        part = {}
        for r in rows:
            v = (r.get(col) or "").strip()
            if v in ("train", "val"):
                p = local.get(r["stem"].lower())
                if p:
                    part[p] = v
        if not any(v == "val" for v in part.values()):
            continue
        name = (col.replace("split_farm_", "byfarm_")
                   .replace("split_gps_", "gps_only_")
                   .replace("split_random", "random"))
        d = os.path.join(P["splits"], name)
        os.makedirs(d)
        for which in ("train", "val"):
            with open(os.path.join(d, which + ".txt"), "w",
                      encoding="utf-8") as fh:
                for p, v in part.items():
                    if v == which:
                        fh.write(p + "\n")
        with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump({"train": os.path.join(d, "train.txt"),
                            "val": os.path.join(d, "val.txt"),
                            "nc": len(names), "names": names},
                           fh, sort_keys=False, allow_unicode=True)
        made.append((name,
                     sum(1 for v in part.values() if v == "train"),
                     sum(1 for v in part.values() if v == "val")))

    print("\n  {:<18}{:>8}{:>8}".format("split", "train", "val"))
    for n, tr, va in made:
        print(f"  {n:<18}{tr:>8}{va:>8}")

    bad = 0
    for name, _, _ in made:
        d = os.path.join(P["splits"], name)
        t = set(open(os.path.join(d, "train.txt"), encoding="utf-8").read().split())
        v = set(open(os.path.join(d, "val.txt"), encoding="utf-8").read().split())
        if t & v:
            bad += 1
            print(f"  ★ {name}: {len(t & v)} 张同时在 train 和 val")
    print(f"\n  train/val 重叠检查: {'通过' if bad == 0 else '★ 失败'}")
    print(f"  写到 {P['splits']}")


# ------------------------------------------------------------------- train --
def _build(weights):
    from ultralytics import YOLO
    if "rtdetr" in weights.lower():
        try:
            from ultralytics import RTDETR
            return RTDETR(weights)
        except ImportError:
            pass
    return YOLO(weights)


def _configs(P, include_gps):
    cfgs = sorted(d for d in os.listdir(P["splits"])
                  if os.path.isfile(os.path.join(P["splits"], d, "data.yaml")))
    if not include_gps:
        cfgs = [c for c in cfgs if not c.startswith("gps_only")]
    cfgs.sort(key=lambda c: (c != "random", c))
    return cfgs


def step_train(a, P):
    if not os.path.isdir(P["splits"]):
        sys.exit("先跑 --step splits")
    cfgs = _configs(P, a.include_gps)
    seeds = a.seeds or SEEDS

    # 种子优先：断在任何一点，手上都是完整的一轮。
    jobs = [(m, c, s) for m in a.models for s in seeds for c in cfgs]
    budget = sum(MIN_PER_RUN.get(m, 8) for m, _, _ in jobs)
    print(f"GPU: {gpu_line()}")
    print(f"配置 {len(cfgs)}: {cfgs}")
    for m in a.models:
        print(f"  {m:<10} seeds {seeds}  batch {MODELS[m]['batch']}  "
              f"{len(cfgs)*len(seeds)} run   "
              f"约 {len(cfgs)*len(seeds)*MIN_PER_RUN.get(m,8)/60:.1f} 小时")
    print(f"\n合计 {len(jobs)} run，估计 {budget/60:.1f} 小时")
    print("顺序: 种子优先。已完成的 run 自动跳过。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return

    os.makedirs(P["runs"], exist_ok=True)
    t0 = time.time()
    done = 0
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"{model}_{cfg}_s{seed}"
        if glob.glob(os.path.join(P["runs"], name, "weights", "best.pt")):
            continue
        el = (time.time() - t0) / 60
        rate = el / done if done else 0
        eta = rate * (len(jobs) - i) / 60 if rate else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        spec = MODELS[model]
        try:
            _build(spec["weights"]).train(
                data=os.path.join(P["splits"], cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                batch=a.batch or spec["batch"], workers=a.workers,
                patience=a.patience, project=P["runs"], name=name,
                exist_ok=True, deterministic=True, plots=False, verbose=False)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
            if "out of memory" in str(e).lower():
                print(f"    显存不足：--batch 调小，或改 MODELS['{model}']")
    print(f"\n训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# -------------------------------------------------------------------- eval --
def _val(weights, data_yaml, imgsz, tag, names, runs):
    from ultralytics import YOLO
    m = (_build(weights) if "rtdetr" in os.path.basename(weights).lower()
         else YOLO(weights))
    r = m.val(data=data_yaml, imgsz=imgsz, split="val", batch=8, workers=2,
              project=runs, name=tag, exist_ok=True, verbose=False,
              plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            cn = names[int(c)] if int(c) < len(names) else str(c)
            row[f"AP50::{cn}"] = float(r.box.ap50[i])
    except Exception:
        pass
    return row


def step_eval(a, P, cross=False):
    import yaml

    cfgs = _configs(P, a.include_gps)
    merged = find_one(P["work"], "merged_peninsula")
    names = load_names(merged)
    seeds = a.seeds or SEEDS

    sabah_yaml = None
    if cross:
        sabah = find_one(P["work"], "merged_sabah")
        if not sabah:
            sys.exit("merged_sabah not found")
        sabah_yaml = os.path.join(P["work"], "sabah_eval.yaml")
        with open(sabah_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"path": sabah, "train": "images", "val": "images",
                            "nc": len(names), "names": names},
                           fh, sort_keys=False, allow_unicode=True)

    out = os.path.join(P["results"],
                       "cross_island.csv" if cross else "in_region.csv")
    rows = []
    if a.append and os.path.isfile(out):
        with open(out, newline="", encoding="utf-8-sig") as fh:
            rows = list(_csv.DictReader(fh))
        have = {(r["model"], r["config"], str(r["seed"])) for r in rows}
        print(f"沿用已有 {len(rows)} 行，只补新的")
    else:
        have = set()

    for model in a.models:
        for cfg in cfgs:
            for seed in seeds:
                if (model, cfg, str(seed)) in have:
                    continue
                name = f"{model}_{cfg}_s{seed}"
                best = os.path.join(P["runs"], name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                if cross:
                    r = _val(best, sabah_yaml, a.imgsz, name + "_sabah",
                             names, P["runs"])
                    r["eval_on"] = "sabah"
                else:
                    r = _val(best, os.path.join(P["splits"], cfg, "data.yaml"),
                             a.imgsz, name + "_val", names, P["runs"])
                    r["eval_on"] = "peninsula"
                r.update({"model": model, "config": cfg, "seed": seed})
                rows.append(r)
                print(f"  {name:<34} mAP50 {r['mAP50']:.4f}", flush=True)

    if not rows:
        sys.exit("没有找到已训练的权重")

    os.makedirs(P["results"], exist_ok=True)
    keys, head = [], ["model", "config", "seed", "eval_on"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    keys = head + [k for k in keys if k not in head]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["config"])].append(float(r["mAP50"]))
    print("\n" + "=" * 66)
    print(f"  {'model':<12}{'config':<18}{'n':>4}{'mAP50':>9}{'sd':>9}")
    for (m, c), v in sorted(agg.items()):
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 \
            if len(v) > 1 else 0.0
        print(f"  {m:<12}{c:<18}{len(v):>4}{mu:>9.4f}{sd:>9.4f}")

    if not cross:
        print("\n" + "=" * 66)
        print("  随机划分 vs leave-one-farm-out，逐架构")
        print("=" * 66)
        for m in sorted({r["model"] for r in rows}):
            rnd = [float(r["mAP50"]) for r in rows
                   if r["model"] == m and r["config"] == "random"]
            fold = defaultdict(list)
            for r in rows:
                if r["model"] == m and r["config"].startswith("byfarm"):
                    fold[r["config"]].append(float(r["mAP50"]))
            if not rnd or not fold:
                continue
            fm = [sum(v) / len(v) for v in fold.values()]
            rm, bm = sum(rnd) / len(rnd), sum(fm) / len(fm)
            sd = (sum((x - bm) ** 2 for x in fm) / (len(fm) - 1)) ** 0.5
            print(f"  {m:<12} random {rm:.4f}  by-farm {bm:.4f}  "
                  f"overstate {100*(rm-bm)/rm:.1f}%  "
                  f"fold sd {sd:.4f}  folds {len(fm)}")
    print(f"\n写出 {out}")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["unpack", "splits", "train", "eval", "cross"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--archive", default=None)
    ap.add_argument("--split-assignment", default=None)
    ap.add_argument("--models", nargs="+", default=["yolo11s"],
                    choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--include-gps", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="eval/cross 时保留表中已有的行，只补新架构")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    P = paths(a.root)
    {"unpack": step_unpack, "splits": step_splits, "train": step_train,
     "eval": lambda x, p: step_eval(x, p, False),
     "cross": lambda x, p: step_eval(x, p, True)}[a.step](a, P)


if __name__ == "__main__":
    main()
