#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
farm_budget.py -- 数据预算实验：多跑几个果园，还是每个果园多拍几张？

问的是一个采集决策：模型要拿到一个从没见过的果园上用，训练数据该
怎么花——去更多果园，还是在每个果园多拍？

设计（外层 = 留一果园，全程不碰训练）：
  held-out 果园 h ∈ 8 个
  训练果园数 k ∈ {1, 2, 4, 7}      从其余 7 个里抽，k=7 就是全部
  每果园张数 m ∈ {15, 50, all}     每个抽中的果园随机取 m 张
  抽签 draws: k<7 抽 2 组不同果园组合；k=7 只有 1 组
  种子 seeds: 42, 1
  模型: yolo11n（部署在 Pi 5 上的那个）

  同一 (h, k, draw) 下三个 m 用同一组果园、嵌套的图片（15 ⊂ 50 ⊂ all），
  所以 m 之间是配对比较。

训练协议（和论文的 clean 协议同一原则，但更彻底）：
  - held-out 果园一次都不出现在训练里，包括验证、早停、选 checkpoint。
  - 不留内层验证集：小预算下（15 张）留 10% 等于再砍掉一截，
    而且 2 张图做早停没有意义。改为固定优化步数、不做任何选择，
    评估 last.pt。
  - 固定步数：每个 run 约 ITERS 个 gradient step，
    epochs = clamp(ITERS * batch / n_train, 100, 300)。
    这样 15 张和 700 张的模型都被训到差不多的步数，比较的是数据而不是
    谁训得更久。
  - 每个 run 评两次：held-out 果园（全部图）+ Sabah（若存在）。

规模：8 果园 × 21 配置 × 2 种子 = 336 run。yolo11n 每个 run 约 3-6 分钟
（4090），合计约 20-30 GPU 小时。--seeds 42 先跑一半（168 run）就能出图。

用法（AutoDL）：
    cd /root/autodl-tmp/durian
    python farm_budget.py --step build
    python farm_budget.py --step train                 # 先看清单
    screen -S fb
    python farm_budget.py --step train --yes
    python farm_budget.py --step eval
    # 结果: results_applied/farm_budget.csv  -> 发回来做分析

需要：root 下的 split_assignment.csv（仓库 results_durian/ 里那份），
      merged_peninsula/{images,labels,data.yaml}，可选 merged_sabah/。
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time

ROOT = "/root/autodl-tmp/durian"
MODEL = "yolo11n"
KS = [1, 2, 4, 7]
MS = ["15", "50", "all"]
DRAWS = 2
SEEDS = [42, 1]
ITERS = 2000
EPOCH_MIN, EPOCH_MAX = 100, 300
BATCH = 32
DESIGN_SEED = 20260928
MIN_FARM_IMAGES = 20
SKIP_DIRS = {"runs", "splits", "clean_splits", "fb_splits", "fb_runs",
             "nf_splits", "nf_runs", "sc_splits", "sc_runs",
             "results_v2", "results_clean", "results_applied"}


def P(root):
    return {"root": root,
            "splits": os.path.join(root, "fb_splits"),
            "runs": os.path.join(root, "fb_runs"),
            "results": os.path.join(root, "results_applied")}


# --------------------------------------------------------------- shared --
def find_dir(root, name):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        if name in dns:
            d = os.path.join(dp, name)
            if os.path.isdir(os.path.join(d, "images")):
                return d
    return None


def class_names(root):
    import yaml
    d = find_dir(root, "merged_peninsula")
    if not d or not os.path.isfile(os.path.join(d, "data.yaml")):
        sys.exit("找不到 merged_peninsula/data.yaml")
    n = yaml.safe_load(open(os.path.join(d, "data.yaml"), encoding="utf-8"))["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)


def load_manifest(root, manifest):
    """[(stem, farm, primary, abs_image_path)]，按 stem 在 merged_peninsula 下找图。"""
    cands = [os.path.join(root, manifest),
             os.path.join(root, "results_durian", manifest)]
    mp = next((c for c in cands if os.path.isfile(c)), None)
    if not mp:
        sys.exit(f"找不到 {manifest}（试过 {cands}）")
    rows = list(csv.DictReader(open(mp, encoding="utf-8-sig", errors="replace")))
    pen = find_dir(root, "merged_peninsula")
    if not pen:
        sys.exit("找不到 merged_peninsula/images")
    index = {}
    for f in os.listdir(os.path.join(pen, "images")):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            index[os.path.splitext(f)[0].lower()] = os.path.join(pen, "images", f)
    out, miss = [], []
    for r in rows:
        ip = index.get(r["stem"].strip().lower())
        if ip:
            out.append((r["stem"], str(r["farm"]).strip(),
                        (r.get("primary") or "-1").strip(), os.path.abspath(ip)))
        else:
            miss.append(r["stem"])
    # 只留分析池：farm 是非负整数且至少 MIN_FARM_IMAGES 张（论文的 8 个果园）
    from collections import Counter
    cnt = Counter(f for _, f, _, _ in out)
    keep = {f for f, n in cnt.items()
            if f.lstrip("-").isdigit() and int(f) >= 0 and n >= MIN_FARM_IMAGES}
    dropped = {f: n for f, n in cnt.items() if f not in keep}
    out = [r for r in out if r[1] in keep]
    if dropped:
        print(f"  不进分析池的 farm（未归属或少于 {MIN_FARM_IMAGES} 张）: {dropped}")
    print(f"manifest {mp}: {len(rows)} 行，找到图 {len(out) + sum(dropped.values())}，"
          f"缺 {len(miss)}，分析池 {len(out)} 张 / {len(keep)} 个果园")
    if miss:
        print("  缺的前几个:", miss[:5])
        if len(miss) > 0.02 * len(rows):
            sys.exit("缺图超过 2%，检查 merged_peninsula/images")
    return out


def write_yaml(d, train, val, names):
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    for nm, ps in (("train", train), ("val", val)):
        with open(os.path.join(d, nm + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(ps) + "\n")
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {d}\ntrain: {os.path.join(d, 'train.txt')}\n"
                 f"val: {os.path.join(d, 'val.txt')}\nnc: {len(names)}\nnames:\n")
        for n in names:
            fh.write(f"- {n}\n")
    return os.path.join(d, "data.yaml")


def write_eval_yaml(d, imgs, names, tag):
    """评估专用的 yaml，训练时不引用。"""
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    lst = os.path.join(d, f"{tag}.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(imgs) + "\n")
    y = os.path.join(d, f"{tag}.yaml")
    with open(y, "w", encoding="utf-8") as fh:
        fh.write(f"path: {d}\ntrain: {lst}\nval: {lst}\nnc: {len(names)}\nnames:\n")
        for n in names:
            fh.write(f"- {n}\n")
    return y


def epochs_for(n, batch):
    b = min(batch, n)
    return max(EPOCH_MIN, min(EPOCH_MAX, round(ITERS * b / n))), b


# ---------------------------------------------------------------- build --
def step_build(a, p):
    names = class_names(a.root)
    rows = load_manifest(a.root, a.manifest)
    by_farm = {}
    for stem, farm, prim, ip in rows:
        by_farm.setdefault(farm, []).append(ip)
    farms = sorted(by_farm, key=lambda x: int(x))
    print(f"果园 {len(farms)} 个: " +
          ", ".join(f"{f}:{len(by_farm[f])}" for f in farms))
    print(f"类别 {names}")

    jobs = []
    for h in farms:
        others = [f for f in farms if f != h]
        heldout = sorted(by_farm[h])
        for k in a.ks:
            if k > len(others):
                continue
            combos = list(itertools.combinations(others, k))
            rnd = random.Random(f"{DESIGN_SEED}-{h}-{k}")
            rnd.shuffle(combos)
            ndraw = 1 if k == len(others) else a.draws
            for d, combo in enumerate(combos[:ndraw]):
                # 每个抽中的果园打乱一次；三个 m 取前缀 -> 嵌套
                order = {}
                for f in combo:
                    v = sorted(by_farm[f])
                    random.Random(f"{DESIGN_SEED}-{h}-{k}-{d}-{f}").shuffle(v)
                    order[f] = v
                for m in a.ms:
                    train = []
                    for f in combo:
                        train += order[f] if m == "all" else order[f][:int(m)]
                    assert not set(train) & set(heldout), "held-out 果园进了训练"
                    ep, b = epochs_for(len(train), a.batch)
                    name = f"fb_h{h}_k{k}_m{m}_d{d}"
                    # 训练用的 yaml：val 指向训练图本身（只为让 Ultralytics
                    # 最后一轮的日志不报错），held-out 果园不出现在里面
                    write_yaml(os.path.join(p["splits"], name), train,
                               train[:50], names)
                    write_eval_yaml(os.path.join(p["splits"], name), heldout,
                                    names, "heldout")
                    jobs.append({"name": name, "heldout": h, "k": k, "m": m,
                                 "draw": d, "train_farms": ";".join(combo),
                                 "n_train": len(train), "n_heldout": len(heldout),
                                 "epochs": ep, "batch": b})
    os.makedirs(p["results"], exist_ok=True)
    meta = os.path.join(p["splits"], "design.json")
    json.dump({"model": MODEL, "iters": ITERS, "epoch_range": [EPOCH_MIN, EPOCH_MAX],
               "design_seed": DESIGN_SEED, "ks": a.ks, "ms": a.ms,
               "draws": a.draws, "jobs": jobs}, open(meta, "w"), indent=1)
    with open(os.path.join(p["results"], "farm_budget_design.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jobs[0]))
        w.writeheader(); w.writerows(jobs)

    print(f"\n{len(jobs)} 个配置 × {len(a.seeds)} 种子 = "
          f"{len(jobs) * len(a.seeds)} run")
    print(f"  {'k':>3}{'m':>5}{'配置':>6}{'训练张数':>12}{'epochs':>10}")
    for k in a.ks:
        for m in a.ms:
            js = [j for j in jobs if j["k"] == k and j["m"] == m]
            if js:
                ns = [j["n_train"] for j in js]; es = [j["epochs"] for j in js]
                print(f"  {k:>3}{m:>5}{len(js):>6}{min(ns):>7}-{max(ns):<5}"
                      f"{min(es):>5}-{max(es)}")
    print(f"\n设计写到 {meta}")


# ---------------------------------------------------------------- train --
def load_jobs(p):
    meta = os.path.join(p["splits"], "design.json")
    if not os.path.isfile(meta):
        sys.exit("先跑 --step build")
    return json.load(open(meta))["jobs"]


def step_train(a, p):
    jobs = load_jobs(p)
    runs = [(j, s) for s in a.seeds for j in jobs]
    todo = [(j, s) for j, s in runs if not os.path.isfile(
        os.path.join(p["runs"], f"{j['name']}_s{s}", "weights", "last.pt"))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(runs)} run，待跑 {len(todo)}（模型 {MODEL}，种子 {a.seeds}）")
    print("★ 不留内层验证、不早停、不选 checkpoint；held-out 果园不参与训练。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return
    from ultralytics import YOLO
    os.makedirs(p["runs"], exist_ok=True)
    t0, done = time.time(), 0
    for i, (j, s) in enumerate(todo, 1):
        name = f"{j['name']}_s{s}"
        el = (time.time() - t0) / 60
        eta = el / done * (len(todo) - done) / 60 if done else 0
        print(f"\n[{i}/{len(todo)}] {name}  n={j['n_train']} "
              f"ep={min(j['epochs'], a.epochs_cap or 10**9)}"
              f"  已用 {el:.0f} 分" + (f"，剩约 {eta:.1f} 小时" if eta else ""),
              flush=True)
        try:
            YOLO(f"{MODEL}.pt").train(
                data=os.path.join(p["splits"], j["name"], "data.yaml"),
                imgsz=a.imgsz, epochs=min(j["epochs"], a.epochs_cap or 10**9),
                batch=j["batch"], seed=s,
                patience=0, val=False, workers=a.workers,
                project=p["runs"], name=name, exist_ok=True,
                deterministic=True, plots=False, verbose=False, save_period=-1)
            done += 1
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
    print(f"\n训练结束，{(time.time() - t0) / 60:.0f} 分钟")


# ----------------------------------------------------------------- eval --
def val_one(model, yml, a, project, name, names):
    r = model.val(data=yml, imgsz=a.imgsz, split="val", batch=16,
                  workers=a.workers, project=project, name=name,
                  exist_ok=True, verbose=False, plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    idx = list(r.box.ap_class_index)
    for ci, cn in enumerate(names):
        row[f"AP50::{cn}"] = float(r.box.ap50[idx.index(ci)]) if ci in idx else ""
    return row


def sabah_yaml(a, p, names):
    s = find_dir(a.root, "merged_sabah")
    if not s:
        return None
    imgs = sorted(os.path.join(s, "images", f) for f in os.listdir(os.path.join(s, "images"))
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return write_eval_yaml(os.path.join(p["splits"], "_sabah"), imgs, names, "sabah")


def step_eval(a, p):
    from ultralytics import YOLO
    names = class_names(a.root)
    jobs = load_jobs(p)
    sab = sabah_yaml(a, p, names) if a.sabah else None
    out = os.path.join(p["results"], "farm_budget.csv")
    done = set()
    rows = []
    if os.path.isfile(out):
        rows = list(csv.DictReader(open(out, encoding="utf-8")))
        done = {(r["run"], r["eval_on"]) for r in rows}
    keys = ["run", "config", "heldout", "k", "m", "draw", "seed", "train_farms",
            "n_train", "eval_on", "mAP50", "mAP50_95", "precision", "recall"] + \
           [f"AP50::{c}" for c in names]

    def flush():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    for s in a.seeds:
        for j in jobs:
            run = f"{j['name']}_s{s}"
            w = os.path.join(p["runs"], run, "weights", "last.pt")
            if not os.path.isfile(w):
                continue
            m = None
            for tag, yml in (("heldout", os.path.join(p["splits"], j["name"], "heldout.yaml")),
                             ("sabah", sab)):
                if yml is None or (run, tag) in done:
                    continue
                m = m or YOLO(w)
                r = val_one(m, yml, a, os.path.join(p["runs"], "_eval"),
                            f"{run}_{tag}", names)
                r.update({"run": run, "config": j["name"], "heldout": j["heldout"],
                          "k": j["k"], "m": j["m"], "draw": j["draw"], "seed": s,
                          "train_farms": j["train_farms"], "n_train": j["n_train"],
                          "eval_on": tag})
                rows.append(r)
                print(f"  {run} [{tag}] mAP50 {r['mAP50']:.4f}", flush=True)
            if m is not None:
                flush()
    flush()
    print(f"\n写出 {out}（{len(rows)} 行）——把这个文件发回来")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True, choices=["build", "train", "eval"])
    ap.add_argument("--manifest", default="split_assignment.csv")
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--ms", nargs="+", default=MS)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-sabah", dest="sabah", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个待跑 run（试跑用）")
    ap.add_argument("--epochs-cap", type=int, default=0,
                    help="试跑用：把 epochs 截到这个数。正式跑不要加")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    p = P(a.root)
    if a.epochs_cap:            # 试跑写到单独目录，不会被正式跑当成已完成
        p["runs"] += "_trial"
        print(f"★ 试跑模式：epochs 截到 {a.epochs_cap}，写到 {p['runs']}")
    {"build": step_build, "train": step_train, "eval": step_eval}[a.step](a, p)


if __name__ == "__main__":
    main()
