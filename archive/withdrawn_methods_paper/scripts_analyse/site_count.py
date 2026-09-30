#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
site_count.py -- 训练时覆盖多少个独立站点，决定在一个没见过的站点上表现如何？

这是全文核心主张的直接测量。现有的三个点已经指向它：

    榴莲 yolo11s    每折训练 7 个农场      高估 44.3%
    榴莲 rtdetr-l   每折训练 7 个农场      高估 42.0%
    GWHD  yolo11s   每折训练 38 个 domain  高估 17.6%

但那是三个观测，不是一条曲线，而且榴莲与 GWHD 之间还差着作物、任务和图像
总量。本实验在单一数据集内把站点数当自变量，**并固定训练图像总数**，所以
"多站点更好"不会被"更多图像更好"解释掉。

设计：
  - 固定一组 held-out 站点，全程不参与训练，作为评估集。
  - 从剩余站点中抽取 k 个（k = 2, 4, 8, 16, 32 ...），每个 k 抽 --draws 组
    不同的站点组合，每组 --seeds 个种子。
  - 每次训练都从抽中的站点里采样到同一个图像总数 N（默认取最小 k 也能满足
    的量），所以 k 变而 N 不变。
  - 报告 held-out 站点上的 mAP50 随 k 的变化，以及站点间离散度随 k 的变化。

用法（GWHD）：
    python site_count.py --step build --root /root/autodl-tmp/gwhd2021 \\
        --ks 2 4 8 16 32 --draws 2 --holdout 8
    nohup python site_count.py --step train --root /root/autodl-tmp/gwhd2021 \\
        --yes > sc.log 2>&1 &
    python site_count.py --step eval --root /root/autodl-tmp/gwhd2021

用法（榴莲，站点少，k 只能到 6）：
    python site_count.py --step build --root /root/autodl-tmp/durian \\
        --manifest split_assignment.csv --site-col farm \\
        --ks 1 2 3 4 6 --draws 3 --holdout 2
"""

import argparse
import csv
import glob
import json
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict

MODELS = {"yolo11n": ("yolo11n.pt", 32), "yolo11s": ("yolo11s.pt", 32),
          "rtdetr-l": ("rtdetr-l.pt", 24)}


def P(root):
    root = os.path.abspath(root)
    return {"root": root,
            "splits": os.path.join(root, "sc_splits"),
            "runs": os.path.join(root, "sc_runs"),
            "results": os.path.join(root, "results_v2")}


def load_rows(root, manifest, site_col, images_dir=None):
    """返回 [(image_path, site)]。兼容 GWHD 的 manifest_clean.csv 与
    榴莲的 split_assignment.csv。

    split_assignment.csv 里的 image 列是产出它的那台机器上的绝对路径
    （Colab 的 /content/durian/...），换机器后必然失效，所以路径解析
    分三步：原路径 -> 指定的 images_dir -> 在 root 下按 stem 搜。"""
    cands = [os.path.join(root, manifest),
             os.path.join(root, "processed", manifest),
             os.path.join(root, "results_v2", manifest)]
    p = next((c for c in cands if os.path.isfile(c)), None)
    if p is None:
        sys.exit(f"找不到 {manifest}（试过 {cands}）")
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"manifest: {p}  ({len(rows)} 行)")
    if site_col not in rows[0]:
        sys.exit(f"没有列 {site_col}；可用列 {list(rows[0])[:12]}")

    path_col = next((c for c in ("image_path", "image", "path")
                     if c in rows[0]), None)
    stem_col = next((c for c in ("stem", "image_id") if c in rows[0]), None)

    # 建 stem -> 本地路径的索引，作为回退
    index = {}
    search = [images_dir] if images_dir else [root]
    for base in search:
        if not base or not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns
                      if d not in ("sc_splits", "sc_runs", "runs", "splits")]
            for f in fns:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    index.setdefault(os.path.splitext(f)[0].lower(), 
                                     os.path.join(dp, f))

    out, via_orig, via_stem, miss = [], 0, 0, 0
    for r in rows:
        site = str(r[site_col]).strip()
        if not site:
            continue
        ip = None
        if path_col and r.get(path_col) and os.path.isfile(r[path_col]):
            ip, via_orig = r[path_col], via_orig + 1
        else:
            key = None
            if stem_col and r.get(stem_col):
                key = str(r[stem_col]).strip().lower()
            elif path_col and r.get(path_col):
                key = os.path.splitext(os.path.basename(r[path_col]))[0].lower()
            if key and key in index:
                ip, via_stem = index[key], via_stem + 1
        if ip:
            out.append((os.path.abspath(ip), site))
        else:
            miss += 1
    print(f"图像定位: 原路径 {via_orig} | 按 stem 找到 {via_stem} | 缺失 {miss}")
    if not out:
        sys.exit("没有一张图能定位。用 --images-dir 指定图片所在目录。")
    return out


def write_yaml(d, train, val, names):
    # 一律写绝对路径。相对路径会被 Ultralytics 再拼一次 dataset 根目录，
    # 变成 sc_splits/k38_d6/sc_splits/k38_d6/val.txt 这样的重复前缀。
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    for nm, ps in (("train", train), ("val", val)):
        with open(os.path.join(d, nm + ".txt"), "w") as fh:
            fh.write("\n".join(os.path.abspath(x) for x in ps) + "\n")
    with open(os.path.join(d, "data.yaml"), "w") as fh:
        fh.write(f"path: {d}\n")
        fh.write(f"train: {os.path.join(d,'train.txt')}\n")
        fh.write(f"val: {os.path.join(d,'val.txt')}\n")
        fh.write(f"nc: {len(names)}\nnames:\n")
        for n in names:
            fh.write(f"- {n}\n")
    return os.path.join(d, "data.yaml")


def read_names(root):
    for pat in ("**/data.yaml", "**/merged_peninsula/data.yaml"):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            if "sc_splits" in f or "splits" in f:
                continue
            txt = open(f, encoding="utf-8").read()
            names, seen = [], False
            for line in txt.splitlines():
                if line.startswith("names:"):
                    seen = True
                    continue
                if seen:
                    s = line.strip()
                    if s.startswith("- "):
                        names.append(s[2:].strip().strip("'\""))
                    elif s and not s.startswith("#"):
                        break
            if names:
                return names
    return ["wheat_head"]


# ------------------------------------------------------------------ build --
def step_build(a, p):
    rows = load_rows(a.root, a.manifest, a.site_col, a.images_dir)
    names = read_names(a.root)
    by_site = defaultdict(list)
    for ip, s in rows:
        by_site[s].append(ip)
    sites = sorted(by_site, key=lambda s: -len(by_site[s]))
    print(f"站点 {len(sites)} 个，图像 {len(rows)} 张，类别 {names}")

    rnd = random.Random(a.seed)
    # held-out：从中位规模的站点里抽，避免全挑最大或最小的
    mid = sites[len(sites) // 4: 3 * len(sites) // 4] or sites
    holdout = rnd.sample(mid, min(a.holdout, len(mid)))
    pool_sites = [s for s in sites if s not in holdout]
    val_paths = [x for s in holdout for x in by_site[s]]
    print(f"\nheld-out 站点 {len(holdout)} 个 / {len(val_paths)} 张: {holdout}")
    print(f"可抽训练站点 {len(pool_sites)} 个")

    ks = [k for k in a.ks if k <= len(pool_sites)]
    if len(ks) < len(a.ks):
        print(f"★ k 超过可用站点数，截断为 {ks}")

    # 固定训练图像总数：最小的 k 也要能凑够
    kmin = min(ks)
    feas = []
    for _ in range(200):
        sub = rnd.sample(pool_sites, kmin)
        feas.append(sum(len(by_site[s]) for s in sub))
    n_train = a.n_train or int(min(feas) * 0.9)
    print(f"\n固定训练图像总数 N = {n_train}"
          f"  (k={kmin} 时最小可得 {min(feas)})")

    jobs = []
    for k in ks:
        for d in range(a.draws):
            sub = rnd.sample(pool_sites, k)
            avail = [x for s in sub for x in by_site[s]]
            if len(avail) < n_train:
                print(f"  k={k} draw={d}: 只有 {len(avail)} 张，跳过")
                continue
            # 各站点等比例采样，避免大站点独占
            per = max(1, n_train // k)
            picked = []
            for s in sub:
                v = by_site[s][:]
                rnd.shuffle(v)
                picked += v[:per]
            rnd.shuffle(picked)
            picked = picked[:n_train]
            if len(picked) < n_train:
                extra = [x for x in avail if x not in set(picked)]
                rnd.shuffle(extra)
                picked += extra[:n_train - len(picked)]
            name = f"k{k:02d}_d{d}"
            write_yaml(os.path.join(p["splits"], name), picked, val_paths,
                       names)
            jobs.append({"name": name, "k": k, "draw": d,
                         "n_train": len(picked),
                         "sites": ";".join(sorted(sub))})

    os.makedirs(p["results"], exist_ok=True)
    meta = os.path.join(p["results"], "site_count_design.json")
    with open(meta, "w") as fh:
        json.dump({"holdout": holdout, "n_val": len(val_paths),
                   "n_train": n_train, "ks": ks, "draws": a.draws,
                   "jobs": jobs}, fh, indent=2)

    print(f"\n  {'config':<12}{'k':>4}{'train':>8}")
    for j in jobs:
        print(f"  {j['name']:<12}{j['k']:>4}{j['n_train']:>8}")
    print(f"\n{len(jobs)} 个配置，写到 {p['splits']}")
    print(f"设计记录 {meta}")
    print(f"训练轮次: {len(jobs)} x {len(a.seeds)} seed = "
          f"{len(jobs)*len(a.seeds)} run")


# ------------------------------------------------------------------- train --
def build_model(w):
    from ultralytics import YOLO
    if "rtdetr" in w.lower():
        try:
            from ultralytics import RTDETR
            return RTDETR(w)
        except ImportError:
            pass
    return YOLO(w)


def step_train(a, p):
    cfgs = sorted(d for d in os.listdir(p["splits"])
                  if os.path.isfile(os.path.join(p["splits"], d, "data.yaml")))
    jobs = [(c, s) for s in a.seeds for c in cfgs]
    w, batch = MODELS[a.model]
    print(f"{len(cfgs)} 配置 x {len(a.seeds)} seed = {len(jobs)} run")
    if not a.yes:
        print("加 --yes 开始")
        return
    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (cfg, seed) in enumerate(jobs, 1):
        name = f"{a.model}_{cfg}_s{seed}"
        if glob.glob(os.path.join(p["runs"], name, "weights", "best.pt")):
            continue
        el = (time.time() - t0) / 60
        eta = (el / done * (len(jobs) - i) / 60) if done else 0
        print(f"\n[{i}/{len(jobs)}] {name}   已用 {el:.0f} 分钟"
              + (f"，剩余约 {eta:.1f} 小时" if eta else ""), flush=True)
        try:
            build_model(w).train(
                data=os.path.join(p["splits"], cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                batch=a.batch or batch, workers=a.workers,
                patience=a.patience, project=p["runs"], name=name,
                exist_ok=True, deterministic=True, plots=False, verbose=False)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
    print(f"\n结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# -------------------------------------------------------------------- eval --
def step_eval(a, p):
    from ultralytics import YOLO
    meta = json.load(open(os.path.join(p["results"],
                                       "site_count_design.json")))
    by_name = {j["name"]: j for j in meta["jobs"]}
    rows = []
    for cfg, j in sorted(by_name.items()):
        y = os.path.join(p["splits"], cfg, "data.yaml")
        for seed in a.seeds:
            name = f"{a.model}_{cfg}_s{seed}"
            best = os.path.join(p["runs"], name, "weights", "best.pt")
            if not os.path.isfile(best):
                continue
            m = (build_model(best) if "rtdetr" in a.model else YOLO(best))
            r = m.val(data=y, imgsz=a.imgsz, split="val", batch=8, workers=2,
                      project=p["runs"], name=name + "_val", exist_ok=True,
                      verbose=False, plots=False)
            rows.append({"model": a.model, "config": cfg, "k": j["k"],
                         "draw": j["draw"], "seed": seed,
                         "n_train": j["n_train"],
                         "mAP50": float(r.box.map50),
                         "mAP50_95": float(r.box.map),
                         "sites": j["sites"]})
            print(f"  {name:<28} k={j['k']:<3} mAP50 {r.box.map50:.4f}",
                  flush=True)
    if not rows:
        sys.exit("没有权重")
    out = os.path.join(p["results"], "site_count.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 66)
    print(f"  held-out 站点表现 vs 训练站点数"
          f"   (训练图像数固定为 {meta['n_train']})")
    print("=" * 66)
    print(f"  {'k':>4}{'runs':>6}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    byk = defaultdict(list)
    for r in rows:
        byk[r["k"]].append(r["mAP50"])
    ks = sorted(byk)
    for k in ks:
        v = byk[k]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"  {k:>4}{len(v):>6}{st.mean(v):>9.4f}{sd:>9.4f}"
              f"{min(v):>9.4f}{max(v):>9.4f}")
    if len(ks) > 1:
        lo, hi = st.mean(byk[ks[0]]), st.mean(byk[ks[-1]])
        print(f"\n  k={ks[0]} -> k={ks[-1]}: {lo:.4f} -> {hi:.4f}"
              f"  ({100*(hi-lo)/lo:+.1f}%)")
        print("  训练图像总数不变，唯一变的是这些图像来自几个站点。")
    print(f"\n写出 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["build", "train", "eval"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default="manifest_clean.csv")
    ap.add_argument("--site-col", default="domain")
    ap.add_argument("--images-dir", default=None,
                    help="图片所在目录；不给则在 --root 下递归搜索")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--model", default="yolo11s", choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    p = P(a.root)
    {"build": step_build, "train": step_train, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
