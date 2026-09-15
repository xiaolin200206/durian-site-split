"""
per_site_v2.py -- 用新的农场归属和新权重重算 per-site 分数。

替代 peninsula_by_burst.py 与 sabah_by_tree.py。三处变更:

  1. 农场归属改读 attribution_by_content.csv 的 farm_new（按内容匹配），
     不再读 metadata_backup.csv 的 farm 列（那条路径造成了 8.9% 误标）。
  2. fold 映射改读 splits_B/split_assignment.csv 的 split_farm_foldN。
  3. 权重目录 runs_colab_v2，命名 {model}_{cfg}_s{seed}。

同时修掉一个未披露的过滤: 旧版硬编码 MIN_IMAGES=5，导致 34 个 burst
只覆盖 560 张里的 279 张(49.8%)，而正文与图注都没写。本版把
覆盖率明确打印出来，并把该参数写进输出表，供 Methods 引用。

在 Colab 跑（需要先 unpack）:
    !python per_site_v2.py --step peninsula
    !python per_site_v2.py --step sabah
    !python per_site_v2.py --step both --gap 15 --min-images 5

输出:
    results_v2/peninsula_by_burst_{gap}s_min{min}.csv
    results_v2/sabah_by_tree.csv
    results_v2/sabah_by_orchard.csv
    results_v2/per_site_coverage.json     ← 覆盖率，Methods 要引用
"""

import os
import re
import sys
import csv
import glob
import json
import shutil
import argparse
from collections import defaultdict, Counter
from datetime import datetime

WORK = "/content/durian"
DRIVE = "/content/drive/MyDrive/durian_for_nature_food"
RUNS = os.path.join(DRIVE, "runs_colab_v2")
RESULTS = os.path.join(DRIVE, "results_v2")
SCRATCH = os.path.join(WORK, "site_subsets")
IMGSZ = 640


def find_one(root, *names):
    for dp, dns, fns in os.walk(root):
        for n in names:
            if n in dns:
                return os.path.join(dp, n)
            if n in fns:
                return os.path.join(dp, n)
    return None


def read_names(merged):
    import yaml
    with open(os.path.join(merged, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def norm_stem(x):
    return os.path.splitext(os.path.basename(str(x)))[0].lower()


def norm_farm(x):
    x = (x or "").strip()
    if not x:
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def parse_ts(s, sub=None):
    if not isinstance(s, str) or not s.strip():
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            t = datetime.strptime(s.strip(), fmt).timestamp()
            if sub:
                d = "".join(c for c in str(sub) if c.isdigit())
                if d:
                    t += float("0." + d)
            return t
        except ValueError:
            continue
    return None


def read_csv_rows(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.DictReader(fh))


def evaluate(weights, data_yaml, tag, names):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=data_yaml, imgsz=IMGSZ, split="val", batch=8, workers=2,
              project=os.path.join(WORK, "site_eval_runs"), name=tag,
              exist_ok=True, verbose=False, plots=False)
    row = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    try:
        for i, c in enumerate(r.box.ap_class_index):
            cn = names[int(c)] if int(c) < len(names) else str(c)
            row[f"AP50::{cn}"] = float(r.box.ap50[i])
    except Exception:
        pass
    return row


def write_yaml(d, paths, names):
    import yaml
    os.makedirs(d, exist_ok=True)
    txt = os.path.join(d, "val.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        for p in paths:
            fh.write(p + "\n")
    y = os.path.join(d, "data.yaml")
    with open(y, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"train": txt, "val": txt, "nc": len(names),
                        "names": names}, fh, sort_keys=False,
                       allow_unicode=True)
    return y


def dump(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def summarise(rows, key, label):
    per = defaultdict(list)
    n_img = {}
    for r in rows:
        per[r[key]].append(r["mAP50"])
        n_img[r[key]] = r["n_images"]
    means = {g: sum(v) / len(v) for g, v in per.items()}
    if len(means) < 2:
        return
    xs = list(means.values())
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    print(f"\n  {label}: {len(means)} 个单元")
    print(f"    均值 {mu:.4f}  sd {sd:.4f}  CV {sd/mu:.3f}")
    print(f"    范围 {min(xs):.4f} – {max(xs):.4f}   极差 "
          f"{max(xs)/min(xs) if min(xs) > 0 else float('inf'):.2f}x")
    print(f"    {'unit':<16}{'n_img':>7}{'mAP50':>9}")
    for g in sorted(means, key=lambda x: -means[x]):
        print(f"    {g:<16}{n_img[g]:>7}{means[g]:>9.4f}")


# ------------------------------------------------------------- peninsula --
def step_peninsula(a, merged, names, coverage):
    def locate(given, *names):
        """依次尝试: 显式给定 -> WORK 目录内 -> Drive 根目录。"""
        if given and os.path.isfile(given):
            return given
        p = find_one(WORK, *names)
        if p and os.path.isfile(p):
            return p
        for n in names:
            p = os.path.join(DRIVE, n)
            if os.path.isfile(p):
                return p
        return None

    attr_p = locate(a.attribution, "attribution_by_content.csv")
    meta_p = locate(a.meta, "metadata.csv")
    sa_p = locate(a.split_assignment, "split_assignment.csv")
    for p, w in ((attr_p, "attribution_by_content.csv"),
                 (meta_p, "metadata.csv"), (sa_p, "split_assignment.csv")):
        if not p:
            sys.exit(f"找不到 {w}。用 --{w.split('.')[0].replace('_','-')} "
                     f"指定，或放到 {DRIVE}/ 下")
    print(f"attribution : {attr_p}")
    print(f"metadata    : {meta_p}")
    print(f"splits      : {sa_p}")

    farm_of, matched_of = {}, {}
    for r in read_csv_rows(attr_p):
        if (r.get("status") or "").strip() == "ok":
            f = norm_farm(r.get("farm_new"))
            if f:
                s = norm_stem(r.get("stem"))
                farm_of[s] = f
                matched_of[s] = norm_stem(r.get("matched_file") or "")
    print(f"农场归属 {len(farm_of)} 张  <- {os.path.basename(attr_p)}")

    ts_meta = {}
    for r in read_csv_rows(meta_p):
        s = norm_stem(r.get("stem") or r.get("file") or "")
        t = parse_ts(r.get("DateTimeOriginal") or r.get("DateTime"),
                     r.get("SubsecTimeOriginal"))
        if s and t:
            ts_meta[s] = t
    # 标注图可能被平台改名 (00006 这类), metadata 里没有该 stem;
    # 其时间戳要经 matched_file 指向的原图去取。
    ts_of, via_self, via_match = {}, 0, 0
    for s in farm_of:
        if s in ts_meta:
            ts_of[s] = ts_meta[s]; via_self += 1
        elif matched_of.get(s) in ts_meta:
            ts_of[s] = ts_meta[matched_of[s]]; via_match += 1
    print(f"可解析时间戳 {len(ts_of)} 张  (同名 {via_self} | 经 matched_file {via_match})")

    fold_of_farm = {}
    for r in read_csv_rows(sa_p):
        for k, v in r.items():
            if (k or "").startswith("split_farm_fold") and \
                    (v or "").strip().lower() == "val":
                fold_of_farm[norm_farm(r.get("farm"))] = \
                    k.replace("split_farm_", "byfarm_")
    print(f"farm -> fold: {dict(sorted(fold_of_farm.items()))}")

    present = {}
    for p in glob.glob(os.path.join(merged, "images", "*.*")):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            present[norm_stem(p)] = p
    pool = [s for s in present if s in farm_of]
    print(f"\n池 {len(pool)} 张")

    # 分 burst：农场内按时间排序
    by_farm = defaultdict(list)
    no_ts = 0
    for s in pool:
        if s in ts_of:
            by_farm[farm_of[s]].append((ts_of[s], s))
        else:
            no_ts += 1
    if no_ts:
        print(f"  ★ {no_ts} 张无时间戳，无法分 burst")

    burst_of, sizes = {}, Counter()
    for f, items in by_farm.items():
        items.sort()
        idx, prev = 0, None
        for t, s in items:
            if prev is not None and (t - prev) > a.gap:
                idx += 1
            bid = f"f{f}b{idx:03d}"
            burst_of[s] = bid
            sizes[bid] += 1
            prev = t

    groups = defaultdict(list)
    for s, b in burst_of.items():
        groups[b].append(present[s])
    all_bursts = len(groups)
    all_imgs = sum(len(v) for v in groups.values())
    groups = {g: ps for g, ps in groups.items() if len(ps) >= a.min_images}
    kept_imgs = sum(len(v) for v in groups.values())

    print(f"\n  gap {a.gap}s: burst 共 {all_bursts} 个，覆盖 {all_imgs} 张")
    print(f"  >= {a.min_images} 张的 burst: {len(groups)} 个，"
          f"覆盖 {kept_imgs} 张 ({100*kept_imgs/max(len(pool),1):.1f}% of pool)")
    print("  ★ 这个覆盖率必须写进 Methods 和图注。旧版只写了 Sabah 那边的过滤。")
    coverage["peninsula"] = {
        "gap_seconds": a.gap, "min_images": a.min_images,
        "pool_images": len(pool), "bursts_all": all_bursts,
        "bursts_kept": len(groups), "images_covered": kept_imgs,
        "coverage_fraction": round(kept_imgs / max(len(pool), 1), 4)}

    if not groups:
        print("  没有可评估的 burst")
        return []

    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    rows = []
    for bi, (g, ps) in enumerate(sorted(groups.items()), 1):
        farm = g.split("b")[0][1:]
        cfg = fold_of_farm.get(farm)
        if not cfg:
            print(f"  {g}: farm {farm} 无对应 fold，跳过")
            continue
        y = write_yaml(os.path.join(SCRATCH, g), ps, names)
        for seed in a.seeds:
            w = os.path.join(RUNS, f"{a.model}_{cfg}_s{seed}",
                             "weights", "best.pt")
            if not os.path.isfile(w):
                continue
            r = evaluate(w, y, f"{g}_s{seed}", names)
            r.update({"burst": g, "farm": farm, "n_images": len(ps),
                      "fold": cfg, "seed": seed, "model": a.model})
            rows.append(r)
        if bi % 10 == 0:
            print(f"    {bi}/{len(groups)}")
    summarise(rows, "burst", "半岛 burst")
    dump(rows, os.path.join(RESULTS, f"peninsula_by_burst_{a.gap}s_min{a.min_images}.csv"))
    return rows


# ----------------------------------------------------------------- sabah --
def step_sabah(a, names, coverage):
    sabah = find_one(WORK, "merged_sabah")
    if not sabah:
        sys.exit("merged_sabah not found")
    pat = re.compile(r"^(o\d+)(t\d+)_", re.I)
    trees, orchards = defaultdict(list), defaultdict(list)
    unmatched = 0
    for p in glob.glob(os.path.join(sabah, "images", "*.*")):
        m = pat.match(os.path.basename(p))
        if not m:
            unmatched += 1
            continue
        trees[m.group(1).lower() + m.group(2).lower()].append(p)
        orchards[m.group(1).lower()].append(p)
    print(f"\nSabah: {len(trees)} 棵树, {len(orchards)} 个园")
    if unmatched:
        print(f"  {unmatched} 张文件名不含树编号")

    kept = {t: ps for t, ps in trees.items() if len(ps) >= a.min_images}
    print(f"  >= {a.min_images} 张的树: {len(kept)}")
    coverage["sabah"] = {"trees_all": len(trees), "trees_kept": len(kept),
                         "orchards": len(orchards),
                         "min_images": a.min_images}

    cfgs = [f"byfarm_fold{i}" for i in range(a.n_folds)] + ["random"]
    jobs = []
    for cfg in cfgs:
        for seed in a.seeds:
            w = os.path.join(RUNS, f"{a.model}_{cfg}_s{seed}",
                             "weights", "best.pt")
            if os.path.isfile(w):
                jobs.append((cfg, seed, w))
    print(f"  用 {len(jobs)} 个模型评估（Sabah 对所有模型都是留出集）")

    out = []
    for unit, table, label in (("group", kept, "tree"),
                               ("group", orchards, "orchard")):
        rows = []
        for g, ps in sorted(table.items()):
            y = write_yaml(os.path.join(SCRATCH, "sabah_" + g), ps, names)
            for cfg, seed, w in jobs:
                r = evaluate(w, y, f"sabah_{g}_{cfg}_s{seed}", names)
                r.update({"group": g, "n_images": len(ps), "config": cfg,
                          "seed": seed, "model": a.model, "unit": label})
                rows.append(r)
        summarise(rows, "group", f"Sabah {label}")
        dump(rows, os.path.join(RESULTS, f"sabah_by_{label}.csv"))
        out += rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="both",
                    choices=["peninsula", "sabah", "both"])
    ap.add_argument("--gap", type=int, default=15)
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 1, 2, 3, 4])
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--attribution", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--split-assignment", default=None)
    a, unknown = ap.parse_known_args(
        [x for x in sys.argv[1:] if not x.startswith("-f")
         and not x.endswith(".json")])

    merged = find_one(WORK, "merged_peninsula")
    if not merged:
        sys.exit("先跑 colab_run_v2.py --step unpack")
    names = read_names(merged)
    print(f"classes: {names}")

    coverage = {}
    if a.step in ("peninsula", "both"):
        step_peninsula(a, merged, names, coverage)
    if a.step in ("sabah", "both"):
        step_sabah(a, names, coverage)

    os.makedirs(RESULTS, exist_ok=True)
    cp = os.path.join(RESULTS, "per_site_coverage.json")
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(coverage, fh, indent=2)
    print(f"\n覆盖率写入 {cp} —— Methods 要引用这两个数")


if __name__ == "__main__":
    main()
