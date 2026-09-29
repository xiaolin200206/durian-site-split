#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_farm_calibration.py -- 到一个新果园，先拍几张做校准，能补回多少？

接在 farm_budget.py 之后跑。基座模型 = farm_budget 里 k=7、m=all 的 run
（用其余 7 个果园全部图训练、固定步数、last.pt），每个 held-out 果园、
每个种子各一个。

新果园内部怎么切（关键：同一棵树的连拍不能一半校准一半测试）：
  - 该果园的图按拍摄顺序排（有 --meta 时间戳就用时间戳，否则用
    相机计数器 IMG_xxxx；计数器回绕 9999→0000 会自动处理）。
  - 前一半 = 校准池，后一半 = 测试集，中间丢掉 BUFFER 张作缓冲。
  - 再反过来做一次（后一半校准、前一半测试），两个方向 = 两个 draw，
    抵消"先拍的和后拍的不一样"。
  - 同一方向下测试集固定，所有 m、所有 arm 都在同一批图上评。

三个 arm（协议 v2，2026-09-28：每个 m 的参数更新次数相同）：
  base  基座模型直接用，不校准（m=0）
  ft    在基座上用 m 张校准图微调：冻结 backbone（freeze=10），AdamW lr0=5e-4，
        无 warmup，batch 16、nbs 16（不做梯度累积，一次迭代 = 一次更新），
        训练列表重复到每 epoch >= 20 个 batch，正好 FT_ITERS 次更新，不选 checkpoint
  rt    从 COCO 权重重训：其余 7 个果园全部 + m 张校准图，
        和 farm_budget v2 同一固定迭代协议（相当于"下一版模型把新果园加进去"）
  m ∈ {5, 10, 20, all}，all = 整个校准池（28-73 张）

规模（默认）：
  ft  8 果园 × 2 方向 × 4 m × 2 种子 = 128 run，每个约 1 分钟
  rt  8 × 2 × 4 × 1 种子（42）      =  64 run，每个约 3 分钟
  合计约 5 GPU 小时。

用法（AutoDL，farm_budget 的 k=7 m=all 跑完之后）：
    cd /root/autodl-tmp/durian
    python new_farm_calibration.py --step build
    python new_farm_calibration.py --step train            # 看清单
    python new_farm_calibration.py --step train --yes
    python new_farm_calibration.py --step eval
    # 结果: results_applied/new_farm_calibration.csv -> 发回来
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_budget as fb  # noqa: E402  共用读 manifest / 写 yaml / 评估

ROOT = fb.ROOT
MS = ["5", "10", "20", "all"]
FT_SEEDS = [42, 1]
RT_SEEDS = [42]
BUFFER = 3
FT_ITERS = 300
FT_BATCH = 16
FT_LR = 5e-4
FT_FREEZE = 10
DESIGN_SEED = 20260929
PROTOCOL = "v2-calibration-fixed-steps"


def P(root):
    return {"root": root,
            "splits": os.path.join(root, "nf2_splits"),
            "runs": os.path.join(root, "nf2_runs"),
            "fb_runs": os.path.join(root, "fb2_runs"),
            "results": os.path.join(root, "results_applied")}


def capture_order(stems, meta):
    """按拍摄顺序排 stem。meta: stem -> 可排序的时间字符串。"""
    if meta:
        hit = [s for s in stems if norm(s) in meta]
        if len(hit) >= 0.95 * len(stems):
            # 少数没有时间戳的图放到按计数器最接近的有时间戳图旁边
            def num(s):
                m = re.search(r"(\d{3,5})", s)
                return int(m.group(1)) if m else 0
            timed = sorted(hit, key=lambda s: num(s))
            def key(s):
                if norm(s) in meta:
                    return (meta[norm(s)], s)
                near = min(timed, key=lambda t: abs(num(t) - num(s)))
                return (meta[norm(near)], s)
            return sorted(stems, key=key), f"timestamp {len(hit)}/{len(stems)}"

    def num(s):
        m = re.search(r"(\d{3,5})", s)
        return int(m.group(1)) if m else 0
    ns = [num(s) for s in stems]
    wrap = max(ns) - min(ns) > 5000       # 9999 -> 0000 回绕
    key = {s: (n - 10000 if wrap and n > 5000 else n) for s, n in zip(stems, ns)}
    return sorted(stems, key=lambda s: (key[s], s)), "counter" + ("+wrap" if wrap else "")


def load_meta(path):
    if not path:
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", errors="replace")))
    tcol = next((c for c in ("datetime", "DateTimeOriginal", "capture_time",
                             "timestamp", "time") if c in rows[0]), None)
    scol = next((c for c in ("stem", "file", "filename", "image") if c in rows[0]), None)
    if not tcol or not scol:
        sys.exit(f"--meta 需要 stem 列和时间列；现有 {list(rows[0])}")
    return {norm(os.path.splitext(os.path.basename(r[scol]))[0]): r[tcol].strip()
            for r in rows if (r.get(tcol) or "").strip()}


def norm(stem):
    """IMG_2395-HEIC / IMG_2395.HEIC / img_2395 -> img_2395"""
    s = stem.strip().lower()
    for suf in ("-heic", "_heic", ".heic"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def step_build(a, p):
    names = fb.class_names(a.root)
    rows = fb.load_manifest(a.root, a.manifest)
    meta = load_meta(a.meta)
    by_farm = {}
    for stem, farm, prim, ip in rows:
        by_farm.setdefault(farm, {})[stem] = ip
    farms = sorted(by_farm, key=int)

    jobs, splits = [], {}
    print(f"  {'farm':>4}  {'排序依据':<22}{'校准池':>8}{'测试':>6}  (两个方向)")
    for h in farms:
        order, how = capture_order(list(by_farm[h]), meta)
        half = len(order) // 2
        first = order[:half - BUFFER // 2]
        second = order[half + (BUFFER - BUFFER // 2):]
        others = [ip for f in farms if f != h for ip in by_farm[f].values()]
        for direc, (calib, test) in enumerate(((first, second), (second, first))):
            calib_p = [by_farm[h][s] for s in calib]
            test_p = [by_farm[h][s] for s in test]
            assert not set(calib_p) & set(test_p)
            base = f"nf_h{h}_dir{direc}"
            d = os.path.join(p["splits"], base)
            fb.write_eval_yaml(d, test_p, names, "test")
            splits[base] = {"farm": h, "dir": direc, "order": how,
                            "n_calib_pool": len(calib_p), "n_test": len(test_p)}
            pool = sorted(calib_p)
            random.Random(f"{DESIGN_SEED}-{h}-{direc}").shuffle(pool)
            for m in a.ms:
                c = pool if m == "all" else pool[:int(m)]
                if m != "all" and len(pool) < int(m):
                    continue
                # ft: 只用校准图
                ft_name = f"{base}_m{m}_ft"
                r, ep, per, cm = fb.schedule(len(c), FT_BATCH, FT_ITERS)
                fb.write_yaml(os.path.join(p["splits"], ft_name), c * r, c, names)
                jobs.append({"name": ft_name, "arm": "ft", "farm": h, "dir": direc,
                             "m": m, "n_calib": len(c), "n_train": len(c), "repeat": r,
                             "epochs": ep, "iterations": ep * per, "close_mosaic": cm,
                             "batch": FT_BATCH, "test": base})
                # rt: 其余 7 个果园 + 校准图，从 COCO 重训
                rt_name = f"{base}_m{m}_rt"
                tr = others + c
                assert not set(tr) & set(test_p)
                r, ep, per, cm = fb.schedule(len(tr), fb.BATCH)
                fb.write_yaml(os.path.join(p["splits"], rt_name), tr * r, tr[:50], names)
                jobs.append({"name": rt_name, "arm": "rt", "farm": h, "dir": direc,
                             "m": m, "n_calib": len(c), "n_train": len(tr), "repeat": r,
                             "epochs": ep, "iterations": ep * per, "close_mosaic": cm,
                             "batch": fb.BATCH, "test": base})
        s0 = splits[f"nf_h{h}_dir0"]
        print(f"  {h:>4}  {how:<22}{s0['n_calib_pool']:>8}{s0['n_test']:>6}")

    os.makedirs(p["splits"], exist_ok=True)
    json.dump({"protocol": "v2-fixed-iterations", "splits": splits, "jobs": jobs,
               "ft": {"iters": FT_ITERS, "batch": FT_BATCH, "nbs": FT_BATCH, "lr0": FT_LR,
                      "freeze": FT_FREEZE}, "buffer": BUFFER},
              open(os.path.join(p["splits"], "design.json"), "w"), indent=1)
    nft = sum(j["arm"] == "ft" for j in jobs) * len(a.ft_seeds)
    nrt = sum(j["arm"] == "rt" for j in jobs) * len(a.rt_seeds)
    print(f"\nft {nft} run（种子 {a.ft_seeds}），rt {nrt} run（种子 {a.rt_seeds}）")
    if meta is None:
        print("★ 没给 --meta，按相机计数器排序。有 EXIF 时间表的话加 --meta 更准。")


def base_weights(p, h, seed):
    return os.path.join(p["fb_runs"], f"fb_h{h}_k7_mall_d0_s{seed}", "weights", "last.pt")


def load_design(p):
    f = os.path.join(p["splits"], "design.json")
    if not os.path.isfile(f):
        sys.exit("先跑 --step build")
    return json.load(open(f))


def planned(a, p):
    D = load_design(p)
    out = []
    for j in D["jobs"]:
        for s in (a.ft_seeds if j["arm"] == "ft" else a.rt_seeds):
            out.append((j, s))
    return out


def step_train(a, p):
    runs = planned(a, p)
    todo = [(j, s) for j, s in runs if not os.path.isfile(
        os.path.join(p["runs"], f"{j['name']}_s{s}", "weights", "last.pt"))]
    missing = sorted({(j["farm"], s) for j, s in todo if j["arm"] == "ft"
                      and not os.path.isfile(base_weights(p, j["farm"], s))})
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(runs)} run，待跑 {len(todo)}")
    if missing:
        print(f"★ 缺基座权重（farm_budget k=7 m=all）: {missing}")
        print("  先跑 farm_budget.py；缺基座的 ft run 会跳过。")
    if not a.yes:
        print("\n加 --yes 开始。")
        return
    from ultralytics import YOLO
    os.makedirs(p["runs"], exist_ok=True)
    t0 = time.time()
    for i, (j, s) in enumerate(todo, 1):
        name = f"{j['name']}_s{s}"
        data = os.path.join(p["splits"], j["name"], "data.yaml")
        ep = min(j["epochs"], a.epochs_cap or 10**9)
        print(f"\n[{i}/{len(todo)}] {name}  n={j['n_train']} ep={ep}"
              f"  已用 {(time.time() - t0) / 60:.0f} 分", flush=True)
        try:
            if j["arm"] == "ft":
                w = base_weights(p, j["farm"], s)
                if not os.path.isfile(w):
                    print("  跳过：没有基座权重")
                    continue
                YOLO(w).train(data=data, imgsz=a.imgsz, epochs=ep, batch=j["batch"],
                              nbs=j["batch"], close_mosaic=j["close_mosaic"],
                              seed=s, patience=0, val=False, workers=a.workers,
                              optimizer="AdamW", lr0=FT_LR, warmup_epochs=0,
                              freeze=FT_FREEZE, project=p["runs"], name=name,
                              exist_ok=True, deterministic=True, plots=False,
                              verbose=False, save_period=-1)
            else:
                YOLO(f"{fb.MODEL}.pt").train(
                    data=data, imgsz=a.imgsz, epochs=ep, batch=j["batch"], seed=s,
                    close_mosaic=j["close_mosaic"],
                    patience=0, val=False, workers=a.workers, project=p["runs"],
                    name=name, exist_ok=True, deterministic=True, plots=False,
                    verbose=False, save_period=-1)
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
    print(f"\n训练结束，{(time.time() - t0) / 60:.0f} 分钟")


def step_eval(a, p):
    from ultralytics import YOLO
    names = fb.class_names(a.root)
    D = load_design(p)
    out = os.path.join(p["results"], "new_farm_calibration.csv")
    rows = fb.load_results(out, PROTOCOL)
    done = {r["run"] for r in rows}
    keys = ["protocol", "run", "arm", "farm", "dir", "m", "n_calib", "seed", "n_test",
            "mAP50", "mAP50_95", "precision", "recall"] + [f"AP50::{c}" for c in names]
    os.makedirs(p["results"], exist_ok=True)

    def flush():
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    todo = []
    # base：每个 (farm, dir, seed) 一次
    for base, sp in D["splits"].items():
        for s in sorted(set(a.ft_seeds) | set(a.rt_seeds)):
            todo.append((f"{base}_base_s{s}", "base", sp["farm"], sp["dir"], "0", 0, s,
                         base, base_weights(p, sp["farm"], s)))
    for j, s in planned(a, p):
        todo.append((f"{j['name']}_s{s}", j["arm"], j["farm"], j["dir"], j["m"],
                     j["n_calib"], s, j["test"],
                     os.path.join(p["runs"], f"{j['name']}_s{s}", "weights", "last.pt")))
    for run, arm, farm, direc, m, nc, s, test, w in todo:
        if run in done or not os.path.isfile(w):
            continue
        yml = os.path.join(p["splits"], test, "test.yaml")
        r = fb.val_one(YOLO(w), yml, a, os.path.join(p["runs"], "_eval"), run, names)
        r.update({"run": run, "arm": arm, "farm": farm, "dir": direc, "m": m,
                  "n_calib": nc, "seed": s, "n_test": D["splits"][test]["n_test"],
                  "protocol": PROTOCOL})
        rows.append(r)
        print(f"  {run}  mAP50 {r['mAP50']:.4f}", flush=True)
        flush()
    flush()
    print(f"\n写出 {out}（{len(rows)} 行）——把这个文件发回来")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--step", required=True, choices=["build", "train", "eval"])
    ap.add_argument("--manifest", default="split_assignment.csv")
    ap.add_argument("--meta", default=None,
                    help="可选：含 stem 和拍摄时间列的 CSV（如 metadata_backup.csv）")
    ap.add_argument("--ms", nargs="+", default=MS)
    ap.add_argument("--ft-seeds", type=int, nargs="+", default=FT_SEEDS)
    ap.add_argument("--rt-seeds", type=int, nargs="+", default=RT_SEEDS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs-cap", type=int, default=0)
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
