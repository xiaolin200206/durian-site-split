"""
Colab driver v2 -- 多架构 + 外部划分。

与 v1 的关键差别:

  1. splits 步骤不再在 Colab 里重算。v1 用 metadata.csv 按文件名 join farm
     再跑 GroupKFold —— 正是造成 8.9% 误标的那条路径 (farm 0 与 farm 6
     重号 IMG_98xx，(1) 后缀被抹，fold 0 验证集 29.7% 实为 farm 6)。
     v2 直接读 make_splits_v2.py 产出的 split_assignment.csv，
     只把路径改写成 Linux 的。划分逻辑只存在于一处。

  2. 多模型。runs / 结果表都带 model 列，互不覆盖。

  3. 主模型跑满 5 seed，其余模型默认只跑 2 seed —— 架构检验回答的是
     "这是不是 YOLO 特有的"，不需要与主结果同精度。

用法 (Colab):
    !pip -q install ultralytics
    from google.colab import drive; drive.mount('/content/drive')
    !python colab_run_v2.py --step unpack
    !python colab_run_v2.py --step splits
    !python colab_run_v2.py --step train --models yolo11s
    !python colab_run_v2.py --step eval  --models yolo11s
    !python colab_run_v2.py --step cross --models yolo11s
    # 主模型跑完再补架构对照
    !python colab_run_v2.py --step train --models rtdetr-l yolo11n

上传到 Drive 的 durian_for_nature_food/ 下需要有:
    durian_colab.zip                     (merged_peninsula / merged_sabah)
    split_assignment.csv                 (splits_B 里那个)
"""

import os
import sys
import glob
import time
import shutil
import zipfile
import argparse
import csv as _csv
from collections import Counter, defaultdict

# ---------------------------------------------------------------- settings --
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
ARCHIVE = os.path.join(DRIVE, "durian_colab.zip")

WORK = "/content/durian"
RUNS = os.path.join(DRIVE, "runs_colab_v2")
RESULTS = os.path.join(DRIVE, "results_v2")

# 每个模型的权重与 batch。rtdetr 显存吃得多，batch 要降。
MODELS = {
    "yolo11n":  {"weights": "yolo11n.pt",  "batch": 32},
    "yolo11s":  {"weights": "yolo11s.pt",  "batch": 32},
    "yolo11m":  {"weights": "yolo11m.pt",  "batch": 16},
    "yolo11l":  {"weights": "yolo11l.pt",  "batch": 8},
    "rtdetr-l": {"weights": "rtdetr-l.pt", "batch": 8},
}
PRIMARY = "yolo11s"                 # 跑满 seed 的主模型
SEEDS = [42, 1, 2, 3, 4]
ARCH_SEEDS = [42, 1, 2, 3, 4]       # 完全交叉：方差分解需要每个格子都有重复

IMGSZ = 640
EPOCHS = 150
PATIENCE = 50
WORKERS = 8

# 每 run 的粗略耗时(分钟, A100)，只用于开跑前给个预算
MIN_PER_RUN = {"yolo11n": 4.5, "yolo11s": 6.3, "yolo11m": 12,
               "yolo11l": 18, "rtdetr-l": 19}
# ---------------------------------------------------------------------------


def find_one(root, *names):
    for dp, dns, fns in os.walk(root):
        for n in names:
            if n in dns:
                return os.path.join(dp, n)
            if n in fns:
                return os.path.join(dp, n)
    return None


def seeds_for(model, a):
    if a.seeds:
        return a.seeds
    return SEEDS if model == PRIMARY else ARCH_SEEDS


def load_names(merged):
    import yaml
    with open(os.path.join(merged, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


# ------------------------------------------------------------------ unpack --
def step_unpack(a):
    if not os.path.isfile(a.archive):
        sys.exit(f"archive not found: {a.archive}")
    os.makedirs(WORK, exist_ok=True)
    print(f"extracting {a.archive} -> {WORK}")
    t0 = time.time()
    with zipfile.ZipFile(a.archive) as zf:
        zf.extractall(WORK)
    print(f"done in {time.time()-t0:.0f}s")
    for key in ("merged_peninsula", "merged_sabah"):
        p = find_one(WORK, key)
        if p:
            print(f"  {key}: images "
                  f"{len(glob.glob(os.path.join(p,'images','*.*')))}, "
                  f"labels {len(glob.glob(os.path.join(p,'labels','*.txt')))}")
        else:
            print(f"  {key}: NOT FOUND")


# ------------------------------------------------------------------ splits --
def step_splits(a):
    """不重算划分。读 split_assignment.csv，只改写路径。"""
    import yaml

    merged = find_one(WORK, "merged_peninsula")
    if not merged:
        sys.exit("先跑 --step unpack")
    names = load_names(merged)
    nc = len(names)
    print(f"classes ({nc}): {names}")

    sa_p = a.split_assignment
    if not sa_p or not os.path.isfile(sa_p):
        for c in (os.path.join(DRIVE, "split_assignment.csv"),
                  os.path.join(DRIVE, "results", "split_assignment.csv"),
                  find_one(WORK, "split_assignment.csv")):
            if c and os.path.isfile(c):
                sa_p = c
                break
    if not sa_p or not os.path.isfile(sa_p):
        sys.exit("找不到 split_assignment.csv。把 splits_B 里那个上传到 Drive，"
                 "或用 --split-assignment 指定。")
    print(f"split_assignment: {sa_p}")

    with open(sa_p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        sys.exit("split_assignment.csv 是空的")
    cols = [c for c in rows[0] if c.startswith("split_")]
    cols.sort(key=lambda c: (not c.endswith("random"), c))
    print(f"划分列 ({len(cols)}): {cols}")

    # stem -> 本地图片路径
    local = {}
    for p in glob.glob(os.path.join(merged, "images", "*.*")):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            local[os.path.splitext(os.path.basename(p))[0].lower()] = p
    print(f"本地图片 {len(local)}")

    missing = [r["stem"] for r in rows
               if r["stem"].lower() not in local]
    if missing:
        print(f"★ {len(missing)} 个 stem 在本地找不到，例: {missing[:5]}")
        if len(missing) > len(rows) * 0.02:
            sys.exit("缺失过多，archive 与 split_assignment 不是同一批数据")

    out = os.path.join(WORK, "splits")
    if os.path.exists(out):
        shutil.rmtree(out)
    made = []
    for col in cols:
        part_of = {}
        for r in rows:
            v = (r.get(col) or "").strip()
            if v in ("train", "val"):
                p = local.get(r["stem"].lower())
                if p:
                    part_of[p] = v
        if not any(v == "val" for v in part_of.values()):
            print(f"  {col}: 无 val，跳过")
            continue
        name = col.replace("split_farm_", "byfarm_").replace(
            "split_gps_", "gps_only_").replace("split_random", "random")
        d = os.path.join(out, name)
        os.makedirs(d)
        for part in ("train", "val"):
            with open(os.path.join(d, part + ".txt"), "w",
                      encoding="utf-8") as fh:
                for p, v in part_of.items():
                    if v == part:
                        fh.write(p + "\n")
        with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump({"train": os.path.join(d, "train.txt"),
                            "val": os.path.join(d, "val.txt"),
                            "nc": nc, "names": names},
                           fh, sort_keys=False, allow_unicode=True)
        tr = sum(1 for v in part_of.values() if v == "train")
        va = sum(1 for v in part_of.values() if v == "val")
        made.append((name, tr, va))

    print("\n  {:<18}{:>8}{:>8}".format("split", "train", "val"))
    for n, tr, va in made:
        print(f"  {n:<18}{tr:>8}{va:>8}")

    # 完整性: 同一张图不得同时在 train 和 val
    bad = 0
    for name, _, _ in made:
        d = os.path.join(out, name)
        t = set(open(os.path.join(d, "train.txt"), encoding="utf-8").read().split())
        v = set(open(os.path.join(d, "val.txt"), encoding="utf-8").read().split())
        if t & v:
            bad += 1
            print(f"  ★ {name}: {len(t & v)} 张同时在 train 和 val")
    print(f"\n  train/val 重叠检查: {'通过' if bad == 0 else '★ 失败'}")
    print(f"  写到 {out}")


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


def step_train(a):
    splits = os.path.join(WORK, "splits")
    if not os.path.isdir(splits):
        sys.exit("先跑 --step splits")
    cfgs = sorted(d for d in os.listdir(splits)
                  if os.path.isfile(os.path.join(splits, d, "data.yaml")))
    if not a.include_gps:
        cfgs = [c for c in cfgs if not c.startswith("gps_only")]
    cfgs.sort(key=lambda c: (c != "random", c))

    # 种子优先: seed 42 先跑完全部配置，再跑 seed 1，依此类推。
    # 这样任何时候中断，手上都是完整的一轮（全部配置各一个种子），
    # 头条数字当场可算，只是种子数少。配置优先的话，断在中途
    # 可能只有 random 和 fold0，什么都算不出来。
    jobs = []
    if a.config_major:
        for m in a.models:
            for c in cfgs:
                for s in seeds_for(m, a):
                    jobs.append((m, c, s))
    else:
        for m in a.models:
            for si, s in enumerate(seeds_for(m, a)):
                for c in cfgs:
                    jobs.append((m, c, s))

    budget = sum(MIN_PER_RUN.get(m, 8) for m, _, _ in jobs)
    print(f"配置 {len(cfgs)}: {cfgs}")
    for m in a.models:
        ns = len(seeds_for(m, a))
        per_round = len(cfgs) * MIN_PER_RUN.get(m, 8)
        print(f"  {m:<10} seeds {seeds_for(m, a)}  "
              f"batch {MODELS[m]['batch']}  "
              f"{len(cfgs)*ns} run   一轮约 {per_round/60:.1f} 小时")
    print(f"\n合计 {len(jobs)} run，估计 {budget/60:.1f} 小时")
    if not a.config_major:
        print("顺序: 种子优先 —— 每跑完一轮就有一套完整结果，中断不亏")
    if budget > 60 * 5 and not a.yes:
        print("★ 超过 5 小时。种子优先的顺序下中断是安全的，加 --yes 开始。")
        print("  已完成的 run 会自动跳过，断了重跑同一条命令即可。")
        return

    os.makedirs(RUNS, exist_ok=True)
    t0 = time.time()
    for i, (model, cfg, seed) in enumerate(jobs, 1):
        name = f"{model}_{cfg}_s{seed}"
        if glob.glob(os.path.join(RUNS, name, "weights", "best.pt")):
            print(f"[{i}/{len(jobs)}] {name}: 已完成，跳过")
            continue
        print(f"\n[{i}/{len(jobs)}] {name}   "
              f"已用 {(time.time()-t0)/60:.0f} 分钟")
        spec = MODELS[model]
        try:
            _build(spec["weights"]).train(
                data=os.path.join(splits, cfg, "data.yaml"),
                imgsz=a.imgsz, epochs=a.epochs, seed=seed,
                batch=a.batch or spec["batch"], workers=a.workers,
                patience=a.patience, project=RUNS, name=name,
                exist_ok=True, deterministic=True, plots=False, verbose=True)
        except Exception as e:
            print(f"  ★ {name} 失败: {e}")
            if "out of memory" in str(e).lower():
                print(f"    显存不足，把 MODELS['{model}']['batch'] 调小")
    print(f"\n训练结束，用时 {(time.time()-t0)/60:.1f} 分钟")


# -------------------------------------------------------------------- eval --
def _val(model_path, data_yaml, imgsz, tag, names):
    from ultralytics import YOLO
    m = _build(model_path) if "rtdetr" in os.path.basename(model_path).lower() \
        else YOLO(model_path)
    r = m.val(data=data_yaml, imgsz=imgsz, split="val", batch=8, workers=2,
              project=RUNS, name=tag, exist_ok=True, verbose=False, plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            cn = names[int(c)] if int(c) < len(names) else str(c)
            row[f"AP50::{cn}"] = float(r.box.ap50[i])
    except Exception:
        pass
    return row


def step_eval(a, cross=False):
    import yaml

    splits = os.path.join(WORK, "splits")
    cfgs = sorted(d for d in os.listdir(splits)
                  if os.path.isfile(os.path.join(splits, d, "data.yaml")))
    merged = find_one(WORK, "merged_peninsula")
    names = load_names(merged)

    sabah_yaml = None
    if cross:
        sabah = find_one(WORK, "merged_sabah")
        if not sabah:
            sys.exit("merged_sabah not found")
        sabah_yaml = os.path.join(WORK, "sabah_eval.yaml")
        with open(sabah_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"path": sabah, "train": "images", "val": "images",
                            "nc": len(names), "names": names},
                           fh, sort_keys=False, allow_unicode=True)

    rows = []
    for model in a.models:
        for cfg in cfgs:
            for seed in seeds_for(model, a):
                name = f"{model}_{cfg}_s{seed}"
                best = os.path.join(RUNS, name, "weights", "best.pt")
                if not os.path.isfile(best):
                    continue
                if cross:
                    r = _val(best, sabah_yaml, a.imgsz, name + "_sabah", names)
                    r.update({"eval_on": "sabah"})
                else:
                    r = _val(best, os.path.join(splits, cfg, "data.yaml"),
                             a.imgsz, name + "_val", names)
                    r.update({"eval_on": "peninsula"})
                r.update({"model": model, "config": cfg, "seed": seed})
                rows.append(r)
                print(f"  {name:<34} mAP50 {r['mAP50']:.4f}")

    if not rows:
        sys.exit("没有找到已训练的权重")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS,
                       "cross_island.csv" if cross else "in_region.csv")
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    head = ["model", "config", "seed", "eval_on"]
    keys = head + [k for k in keys if k not in head]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 72)
    print("  按模型 x 配置")
    print("=" * 72)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["config"])].append(r["mAP50"])
    print(f"  {'model':<10}{'config':<18}{'n':>4}{'mAP50':>9}{'sd':>9}")
    for (m, c), v in sorted(agg.items()):
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 \
            if len(v) > 1 else 0.0
        print(f"  {m:<10}{c:<18}{len(v):>4}{mean:>9.4f}{sd:>9.4f}")

    # 核心对比: 随机划分 vs 按农场
    print("\n" + "=" * 72)
    print("  随机划分 vs 按农场  (论文的头条数字)")
    print("=" * 72)
    for m in a.models:
        rnd = [r["mAP50"] for r in rows
               if r["model"] == m and r["config"] == "random"]
        folds = defaultdict(list)
        for r in rows:
            if r["model"] == m and r["config"].startswith("byfarm"):
                folds[r["config"]].append(r["mAP50"])
        if not rnd or not folds:
            continue
        fm = [sum(v) / len(v) for v in folds.values()]
        rm = sum(rnd) / len(rnd)
        bm = sum(fm) / len(fm)
        sd = (sum((x - bm) ** 2 for x in fm) / (len(fm) - 1)) ** 0.5 \
            if len(fm) > 1 else 0.0
        print(f"  {m}")
        print(f"    随机划分      {rm:.4f}")
        print(f"    按农场均值    {bm:.4f}   (sd {sd:.4f}, {len(fm)} 折)")
        print(f"    高估          {100*(rm-bm)/rm:.1f}%      (旧版 48.5%)")
        print(f"    折间极差      {max(fm)/min(fm):.2f}x   (旧版 2.84x)")
    print(f"\n写出 {out}")


# -------------------------------------------------------------------- main --
def _parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="splits",
                    choices=["unpack", "splits", "train", "eval", "cross"])
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--split-assignment", default=None)
    ap.add_argument("--models", nargs="+", default=[PRIMARY],
                    choices=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="覆盖所有模型的 seed；不给则每个模型 5 个（完全交叉）")
    ap.add_argument("--include-gps", action="store_true",
                    help="连 gps_only 折一起跑（稳健性检验，可后补）")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--config-major", action="store_true",
                    help="按配置排序而非种子优先（不推荐，中断会拿不到完整一轮）")

    if argv is None:
        argv = sys.argv[1:]
    # Jupyter/Colab 会往 argv 塞 -f kernel.json，忽略掉
    a, unknown = ap.parse_known_args(argv)
    noise = [u for u in unknown if u.startswith("-f") or u.endswith(".json")]
    real = [u for u in unknown if u not in noise]
    if real:
        print(f"忽略了无法识别的参数: {real}")
    return a


STEPS = {"unpack": step_unpack, "splits": step_splits, "train": step_train,
         "eval": lambda x: step_eval(x, False),
         "cross": lambda x: step_eval(x, True)}


def run(step="splits", **kw):
    """在 Colab 单元格里直接调用，不必用 !python。

        import colab_run_v2 as R
        R.run("splits")
        R.run("train", models=["yolo11s"])
        R.run("train", models=["rtdetr-l"], seeds=[42, 1], yes=True)
    """
    argv = ["--step", step]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return STEPS[step](_parse(argv))


def main():
    a = _parse()
    STEPS[a.step](a)


if __name__ == "__main__":
    main()
