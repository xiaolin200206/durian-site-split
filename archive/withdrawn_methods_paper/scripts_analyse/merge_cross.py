#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_cross.py -- 把两个架构的 cross_island 结果合成一张表。

起因：autodl_run.py 的 --step cross 那一步没有生效 --append，
rtdetr-l 的 45 行覆盖掉了 yolo11s 的 45 行。两份原始文件都还在，
合并即可，不必重跑。

放在项目根目录（与 results_v2/ 同级）运行：
    python merge_cross.py

它会找 results_v2/ 下所有 cross_island*.csv，按 (model, config, seed)
去重后合并成 cross_island.csv，并报告每个架构有多少行。
"""

import csv
import glob
import os
import shutil
from collections import Counter, OrderedDict

D = "results_v2"
OUT = os.path.join(D, "cross_island.csv")


def main():
    if not os.path.isdir(D):
        print(f"找不到 {D}/，请在项目根目录运行")
        return 1

    files = sorted(glob.glob(os.path.join(D, "cross_island*.csv")))
    if not files:
        print(f"{D}/ 下没有 cross_island*.csv")
        return 1
    print("找到:")
    for f in files:
        n = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
        print(f"  {os.path.basename(f):<34} {n} 行")

    rows, keys, seen, dup = [], [], set(), 0
    for f in files:
        with open(f, newline="", encoding="utf-8-sig", errors="replace") as fh:
            for r in csv.DictReader(fh):
                k = (r.get("model", ""), r.get("config", ""),
                     str(r.get("seed", "")))
                if k in seen:
                    dup += 1
                    continue
                seen.add(k)
                for c in r:
                    if c not in keys:
                        keys.append(c)
                rows.append(r)

    if not rows:
        print("没有读到任何行")
        return 1

    # 备份现有的 cross_island.csv，避免又一次覆盖
    if os.path.isfile(OUT):
        bak = OUT.replace(".csv", "_backup.csv")
        if not os.path.isfile(bak):
            shutil.copy2(OUT, bak)
            print(f"\n原文件已备份为 {os.path.basename(bak)}")

    head = ["model", "config", "seed", "eval_on"]
    keys = [k for k in head if k in keys] + [k for k in keys if k not in head]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    c = Counter(r.get("model", "?") for r in rows)
    print(f"\n写出 {OUT}")
    print(f"  {len(rows)} 行   {dict(c)}")
    if dup:
        print(f"  跳过重复 {dup} 行")

    if len(c) < 2:
        print("\n★ 只有一个架构。确认另一份是不是被命名成了别的名字，")
        print("  或者仍在服务器上没下载。")
    else:
        cfg = Counter((r.get("model"), r.get("config")) for r in rows)
        bad = [k for k, v in cfg.items() if v != 5]
        if bad:
            print(f"\n★ 这些 (model, config) 不是 5 个 seed: {bad[:6]}")
        else:
            print("  每个 (架构 × 配置) 都是 5 个 seed，完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
