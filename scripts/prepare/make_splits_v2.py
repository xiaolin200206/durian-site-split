#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_splits_v2.py -- 用按内容重建的农场归属重做划分。

与 make_splits.py 的差别:

  1. 农场归属来自 attribution_by_content.csv (感知哈希匹配原图)，
     不再按文件名 join metadata。按文件名 join 在旧池里造成 8.9% 误标:
     farm 0 与 farm 6 两次走访拍出重号 IMG_98xx，导入时后到的加了 (1)
     后缀，缩放环节又抹掉，于是 fold 0 的验证集有 29.7% 实为 farm 6，
     而 farm 6 在该折是训练集 —— 直接洩漏。

  2. 留一农场交叉验证 (每折恰好一个农场)，取代原来的 GroupKFold(5)。
     可用农场从 5 个增至 8 个，不必再把 farm 6/7/8 塞进同一折凑数。

  3. 新增跨折完整性断言: 匹配到同一张原图的多个标注图不得分到不同折。

输出与旧版格式一致，下游脚本无需改动:
    <OUT>/random/            data.yaml, train.txt, val.txt
    <OUT>/byfarm_fold0..k/   data.yaml, train.txt, val.txt
    <OUT>/gps_only_fold0..k/ data.yaml, train.txt, val.txt
    <OUT>/split_assignment.csv

纯标准库。默认写到 splits_B，不覆盖 splits_A。

用法:
    python make_splits_v2.py
    python make_splits_v2.py --min-farm 20 --dry-run
"""

import argparse
import csv
import glob
import os
import random
import re
import sys
from collections import Counter, defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png")


def stem_of(p):
    return os.path.splitext(os.path.basename(str(p)))[0]


def read_names(data_yaml):
    """不依赖 PyYAML 解析 names (块状列表 / 行内列表 / 字典)。"""
    with open(data_yaml, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    m = re.search(r"^names\s*:\s*$", txt, re.M)
    if m:
        out = []
        for line in txt[m.end():].splitlines():
            if not line.strip():
                continue
            it = re.match(r"^\s*-\s*(.+?)\s*$", line)
            if it:
                out.append(it.group(1).strip().strip("'\""))
                continue
            kv = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if kv:
                out.append(kv.group(2).strip().strip("'\""))
                continue
            break
        if out:
            return out
    m = re.search(r"^names\s*:\s*\[(.*?)\]", txt, re.M | re.S)
    if m:
        return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    pairs = re.findall(r"^\s+(\d+)\s*:\s*(.+?)\s*$", txt, re.M)
    if pairs:
        return [n.strip().strip("'\"")
                for _, n in sorted(pairs, key=lambda x: int(x[0]))]
    sys.exit(f"无法从 {data_yaml} 读出 names")


def load_classes(merged, stem):
    p = os.path.join(merged, "labels", stem + ".txt")
    if not os.path.isfile(p):
        return []
    out = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            v = line.split()
            if v:
                try:
                    out.append(int(float(v[0])))
                except ValueError:
                    pass
    return out


def write_split(out_dir, name, rows, col, names):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    for part in ("train", "val"):
        with open(os.path.join(d, part + ".txt"), "w", encoding="utf-8") as fh:
            for r in rows:
                if r[col] == part:
                    fh.write(r["image"].replace("\\", "/") + "\n")
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("train: " + os.path.join(d, "train.txt").replace("\\", "/") + "\n")
        fh.write("val: " + os.path.join(d, "val.txt").replace("\\", "/") + "\n")
        fh.write("nc: %d\n" % len(names))
        fh.write("names:\n")
        for n in names:
            fh.write("- " + str(n) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--merged", default="dataset_store/merged_peninsula")
    ap.add_argument("--attribution", default="attribution_by_content.csv")
    ap.add_argument("--out", default="dataset_store/splits_B")
    ap.add_argument("--min-farm", type=int, default=20,
                    help="农场至少这么多张图才单独成折")
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    merged = os.path.join(base, *re.split(r"[\\/]", args.merged))
    attr_p = os.path.join(base, args.attribution)
    out_dir = os.path.join(base, *re.split(r"[\\/]", args.out))
    for p, what in ((merged, "merged 目录"), (attr_p, "attribution csv")):
        if not os.path.exists(p):
            sys.exit(f"找不到{what}: {p}")

    names = read_names(os.path.join(merged, "data.yaml"))
    print(f"类别 {len(names)}: {names}")

    by_stem_path = {}
    for p in sorted(glob.glob(os.path.join(merged, "images", "*.*"))):
        if p.lower().endswith(IMG_EXT):
            by_stem_path[stem_of(p).lower()] = p
    print(f"标注集图片 {len(by_stem_path)}")

    # ---------------- 读入按内容的归属 ----------------
    rows, skipped = [], Counter()
    with open(attr_p, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            s = (r.get("stem") or "").strip().lower()
            farm = (r.get("farm_new") or "").strip()
            st = (r.get("status") or "").strip()
            if st != "ok" or not farm:
                skipped[st or "无farm"] += 1
                continue
            p = by_stem_path.get(s)
            if p is None:
                skipped["图片不在盘上"] += 1
                continue
            cls = load_classes(merged, stem_of(p))
            rows.append({
                "stem": stem_of(p), "image": p, "farm": farm,
                "matched_file": r.get("matched_file", ""),
                "farm_source": (r.get("farm_source") or "").strip(),
                "classes": ",".join(str(c) for c in sorted(set(cls))),
                "primary": (Counter(cls).most_common(1)[0][0] if cls else -1),
                "n_boxes": len(cls),
            })
    print(f"进入划分 {len(rows)} 张   跳过 {dict(skipped)}")

    farm_n = Counter(r["farm"] for r in rows)
    farms = sorted((f for f, n in farm_n.items() if n >= args.min_farm),
                   key=lambda x: (len(x), x))
    small = {f: n for f, n in farm_n.items() if n < args.min_farm}
    print(f"\n农场规模: {dict(sorted(farm_n.items(), key=lambda kv: -kv[1]))}")
    print(f"成折的农场 ({len(farms)} 个, >={args.min_farm} 张): {farms}")
    if small:
        print(f"过小、并入训练集: {small}")

    pool = [r for r in rows if r["farm"] in farms]
    print(f"\n最终池: {len(pool)} 张 / {len(farms)} 个农场   (旧版 560 / 5)")

    # ---------------- 留一农场 ----------------
    for i, f in enumerate(farms):
        col = "split_farm_fold%d" % i
        for r in pool:
            r[col] = "val" if r["farm"] == f else "train"

    # ---------------- gps_only 敏感性 ----------------
    gps_pool = [r for r in pool if r["farm_source"] == "gps"]
    gfarm_n = Counter(r["farm"] for r in gps_pool)
    gfarms = [f for f in farms if gfarm_n.get(f, 0) >= args.min_farm]
    print(f"gps_only 子集: {len(gps_pool)} 张 / {len(gfarms)} 个农场 -> {gfarms}")
    for i, f in enumerate(gfarms):
        col = "split_gps_fold%d" % i
        for r in pool:
            r[col] = ("" if r not in gps_pool else
                      ("val" if r["farm"] == f else "train"))

    # ---------------- 随机划分 (同池, 按主类分层) ----------------
    rnd = random.Random(args.seed)
    byc = defaultdict(list)
    for r in pool:
        byc[r["primary"]].append(r)
    for r in pool:
        r["split_random"] = "train"
    n_val = 0
    for c, group in byc.items():
        g = group[:]
        rnd.shuffle(g)
        k = max(1, int(round(len(g) * args.val_fraction))) if len(g) >= 5 \
            else int(len(g) * args.val_fraction)
        for r in g[:k]:
            r["split_random"] = "val"
        n_val += k
    print(f"随机划分: train {len(pool)-n_val} / val {n_val}")

    # ---------------- 完整性断言 ----------------
    print("\n" + "=" * 74)
    print("完整性检查")
    print("=" * 74)
    bad = 0
    bysrc = defaultdict(list)
    for r in pool:
        if r["matched_file"]:
            bysrc[r["matched_file"]].append(r)
    dup_src = {k: v for k, v in bysrc.items() if len(v) > 1}
    print(f"  匹配到同一张原图的标注图组: {len(dup_src)}")
    for i in range(len(farms)):
        col = "split_farm_fold%d" % i
        for k, v in dup_src.items():
            if len({r[col] for r in v}) > 1:
                bad += 1
                if bad <= 5:
                    print(f"    ★ {col}: {k} 的多个标注图跨折 "
                          f"{[r['stem'] for r in v]}")
    print(f"  跨折违例: {bad}   " + ("(通过)" if bad == 0 else "★ 需处理"))

    for i, f in enumerate(farms):
        col = "split_farm_fold%d" % i
        tr = {r["farm"] for r in pool if r[col] == "train"}
        assert f not in tr, f"fold{i} 的训练集里出现了留出农场 {f}"
    print("  每折的留出农场均未出现在训练集: 通过")

    # ---------------- 各折构成 ----------------
    print("\n" + "=" * 74)
    print("各折构成 (论文需要报告每折覆盖哪些类)")
    print("=" * 74)
    print(f"  {'fold':<8}{'留出农场':<10}{'val图':>7}{'train图':>8}   val 中出现的类")
    for i, f in enumerate(farms):
        col = "split_farm_fold%d" % i
        v = [r for r in pool if r[col] == "val"]
        t = len(pool) - len(v)
        cs = sorted({int(x) for r in v for x in r["classes"].split(",") if x})
        miss = [names[c] for c in range(len(names)) if c not in cs]
        print(f"  fold{i:<3}{'':<1}{f:<10}{len(v):>7}{t:>8}   "
              f"{[names[c] for c in cs]}")
        if miss:
            print(f"  {'':<19}{'':<15}   缺: {miss}")

    if args.dry_run:
        print("\n--dry-run，未写文件。")
        return 0

    # ---------------- 写出 ----------------
    os.makedirs(out_dir, exist_ok=True)
    write_split(out_dir, "random", pool, "split_random", names)
    for i in range(len(farms)):
        write_split(out_dir, "byfarm_fold%d" % i, pool,
                    "split_farm_fold%d" % i, names)
    for i in range(len(gfarms)):
        col = "split_gps_fold%d" % i
        sub = [r for r in pool if r[col] in ("train", "val")]
        write_split(out_dir, "gps_only_fold%d" % i, sub, col, names)

    cols = (["stem", "image", "farm", "farm_source", "matched_file",
             "n_boxes", "classes", "primary", "split_random"]
            + ["split_farm_fold%d" % i for i in range(len(farms))]
            + ["split_gps_fold%d" % i for i in range(len(gfarms))])
    with open(os.path.join(out_dir, "split_assignment.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pool)

    print(f"\n写出到 {out_dir}")
    print(f"  random/ + byfarm_fold0..{len(farms)-1}/ + "
          f"gps_only_fold0..{len(gfarms)-1}/ + split_assignment.csv")
    print(f"\n训练轮次: (1 随机 + {len(farms)} 折) x 5 seed = "
          f"{(1+len(farms))*5} run"
          f"   + gps_only {len(gfarms)*5} run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
