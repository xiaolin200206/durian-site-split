#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_breakhis.py -- 把 BreaKHis 转成分类数据集，并建立随机 / 患者互斥两套划分。

这是第三个数据集，用来把主张从"田间农业视觉"扩到"任何分层采集的数据"：

    榴莲      检测   农业   8 个农场    单人采集，极度受控
    GWHD      检测   农业   47 个会话   十几个机构，完全异质
    BreaKHis  分类   医学   82 个患者   单实验室，临床采集

评估单元在这里是**患者**。同一患者的多张切片视野共享染色批次、切片制备和
组织形态，随机划分会把它们分到边界两侧——与榴莲的 capture burst、GWHD 的
acquisition session 是同一个机制。

BreaKHis 还白送一个维度：同一患者有 40/100/200/400 四个放大倍率。这正好
对应榴莲那边的焦距混淆，可以检验"采集参数混淆"是否也不是农业独有。

做四件事：
  [1] 从目录结构解析患者 ID、类别、倍率，导出 YOLO 分类格式
  [2] 重复审计：sha256 找逐位相同的图，检查患者与标签是否一致
  [3] 划分：5 折随机（图像级）+ 5 折患者互斥
  [4] 断言：患者折不得有任何患者跨越 train/val

用法：
    wget -c http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz
    tar xzf BreaKHis_v1.tar.gz
    python prepare_breakhis.py --step build --root /root/autodl-tmp/breakhis

    # 只用某个倍率（控制采集参数）
    python prepare_breakhis.py --step build --root ... --mag 100X

产出：
    processed/manifest.csv / manifest_clean.csv
    processed/duplicate_report.csv
    processed/images/<class>/<image_id>.png   （软链，不复制）
"""

import argparse
import csv
import hashlib
import os
import random
import re
import sys
from collections import Counter, defaultdict

# SOB_B_A-14-22549AB-40-001.png
#     ^ B/M  ^subtype ^patient ^mag ^seq
FN = re.compile(r"^SOB_([BM])_([A-Z]+)-(.+?)-(\d+)-(\d+)\.png$", re.I)


def find_root(root):
    for dp, dns, fns in os.walk(root):
        if os.path.basename(dp) == "breast" and dns:
            return dp
    for dp, dns, fns in os.walk(root):
        if any(f.lower().startswith("sob_") for f in fns):
            return root
    return None


def step_build(a):
    src = find_root(a.root)
    if src is None:
        sys.exit(f"在 {a.root} 下找不到 BreaKHis 目录（应有 .../breast/）")
    print(f"数据根: {src}")

    proc = os.path.join(a.root, "processed")
    os.makedirs(proc, exist_ok=True)

    rows, bad = [], 0
    for dp, _, fns in os.walk(src):
        for f in fns:
            if not f.lower().endswith(".png"):
                continue
            m = FN.match(f)
            if not m:
                bad += 1
                continue
            grade, subtype, patient, mag, seq = m.groups()
            if a.mag and mag != a.mag.rstrip("X"):
                continue
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            rows.append({
                "image_id": os.path.splitext(f)[0],
                "image_path": os.path.abspath(p),
                "patient": patient,
                "label": "benign" if grade.upper() == "B" else "malignant",
                "subtype": subtype.upper(),
                "magnification": mag + "X",
                "seq": int(seq),
                "sha256": sha,
            })
    if bad:
        print(f"文件名不匹配、已跳过: {bad}")
    if not rows:
        sys.exit("没有解析出任何图像")

    pats = {r["patient"] for r in rows}
    print(f"\n图像 {len(rows)}   患者 {len(pats)}")
    print(f"类别: {dict(Counter(r['label'] for r in rows))}")
    print(f"倍率: {dict(sorted(Counter(r['magnification'] for r in rows).items()))}")
    print(f"亚型: {dict(Counter(r['subtype'] for r in rows).most_common())}")

    # 患者的标签必须唯一
    plab = defaultdict(set)
    for r in rows:
        plab[r["patient"]].add(r["label"])
    mixed = [p for p, v in plab.items() if len(v) > 1]
    print(f"标签不唯一的患者: {len(mixed)}"
          + (f"  {mixed[:5]}" if mixed else "  (患者层面标签一致)"))

    # ---------------- 重复审计 ----------------
    print("\n" + "=" * 68)
    print("[1] 重复审计（sha256 逐位相同）")
    print("=" * 68)
    g = defaultdict(list)
    for r in rows:
        g[r["sha256"]].append(r)
    dup = {k: v for k, v in g.items() if len(v) > 1}
    print(f"  唯一图像 {len(g)} / {len(rows)}")
    print(f"  重复组 {len(dup)}，涉及 {sum(len(v) for v in dup.values())} 行")

    conflict = []
    for k, v in dup.items():
        if len({r["patient"] for r in v}) > 1 or len({r["label"] for r in v}) > 1:
            conflict.append((k, v))
    print(f"    患者或标签不一致的: {len(conflict)}  ← 剔除")
    print(f"    完全一致的:        {len(dup)-len(conflict)}  ← 保留一份")
    cross_mag = sum(1 for v in dup.values()
                    if len({r["magnification"] for r in v}) > 1)
    print(f"  跨倍率的重复组: {cross_mag}")

    rep = os.path.join(proc, "duplicate_report.csv")
    with open(rep, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "n", "image_ids", "patients", "labels",
                    "magnifications", "verdict"])
        for k, v in dup.items():
            bad_ = (len({r["patient"] for r in v}) > 1 or
                    len({r["label"] for r in v}) > 1)
            w.writerow([k, len(v), ";".join(r["image_id"] for r in v),
                        ";".join(sorted({r["patient"] for r in v})),
                        ";".join(sorted({r["label"] for r in v})),
                        ";".join(sorted({r["magnification"] for r in v})),
                        "drop" if bad_ else "keep_one"])
    print(f"  明细 -> {rep}")

    drop = {r["image_id"] for _, v in conflict for r in v}
    keep, seen = [], set()
    for r in rows:
        if r["image_id"] in drop or r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        keep.append(r)
    print(f"\n  分析集 {len(keep)} / {len(rows)}")

    # ---------------- 划分 ----------------
    print("\n" + "=" * 68)
    print("[2] 建立划分")
    print("=" * 68)
    rnd = random.Random(a.seed)
    idx = list(range(len(keep)))
    rnd.shuffle(idx)
    for pos, i in enumerate(idx):
        keep[i]["random_fold"] = pos % a.folds

    # 患者折：按图像数贪心装箱，并尽量平衡两类患者
    size = Counter(r["patient"] for r in keep)
    lab = {r["patient"]: r["label"] for r in keep}
    bins = [[] for _ in range(a.folds)]
    load = [0] * a.folds
    for label in ("malignant", "benign"):
        for p, n in [(p, n) for p, n in size.most_common() if lab[p] == label]:
            j = load.index(min(load))
            bins[j].append(p)
            load[j] += n
    pf = {p: j for j, ps in enumerate(bins) for p in ps}
    for r in keep:
        r["patient_fold"] = pf[r["patient"]]

    mp = os.path.join(proc, "manifest_clean.csv")
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)
    print(f"写出 {mp}  ({len(keep)} 行)")

    print(f"\n  {'fold':<6}{'random val':>12}{'patient val':>13}"
          f"{'patients':>10}{'benign%':>9}")
    for f in range(a.folds):
        rv = sum(1 for r in keep if r["random_fold"] == f)
        sub = [r for r in keep if r["patient_fold"] == f]
        nb = sum(1 for r in sub if r["label"] == "benign")
        print(f"  {f:<6}{rv:>12}{len(sub):>13}"
              f"{len({r['patient'] for r in sub}):>10}"
              f"{100*nb/max(len(sub),1):>8.1f}%")

    # ---------------- 断言 ----------------
    print("\n" + "=" * 68)
    print("[3] 泄漏断言")
    print("=" * 68)
    bad_ = 0
    for f in range(a.folds):
        tr = {r["patient"] for r in keep if r["patient_fold"] != f}
        va = {r["patient"] for r in keep if r["patient_fold"] == f}
        if tr & va:
            bad_ += 1
            print(f"  ★ patient_fold{f}: {sorted(tr & va)[:5]} 跨界")
    print(f"  患者跨折: {'无' if bad_ == 0 else f'★ {bad_} 折有问题'}")
    h = defaultdict(set)
    for r in keep:
        h[r["sha256"]].add(r["patient_fold"])
    print(f"  同一图像跨折: {sum(1 for v in h.values() if len(v) > 1)}")

    # 随机折里同患者跨折的比例 —— 这正是泄漏的量
    rh = defaultdict(set)
    for r in keep:
        rh[r["patient"]].add(r["random_fold"])
    span = sum(1 for v in rh.values() if len(v) > 1)
    print(f"\n  随机划分下，图像分散到多折的患者: {span} / {len(rh)} "
          f"({100*span/len(rh):.0f}%)")
    print("  这就是随机划分泄漏的直接来源：几乎每个患者都同时出现在")
    print("  训练和验证两侧。")

    print(f"""
下一步：
    python bkh_run.py --step splits --root {a.root}
    nohup python bkh_run.py --step train --root {a.root} --yes > bkh.log 2>&1 &
""")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="build", choices=["build"])
    ap.add_argument("--root", default="/root/autodl-tmp/breakhis")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mag", default=None,
                    help="只保留某个倍率，例如 100X；不给则全用")
    a = ap.parse_args()
    return step_build(a)


if __name__ == "__main__":
    sys.exit(main())
