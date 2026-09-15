#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_gwhd.py -- 把 GWHD 2021 转成 YOLO 格式，并建立随机 / domain-disjoint 两套划分。

外部验证要回答的只有一个问题：评估单元的方差是否同样支配报告值，在另一个
作物、另一个任务、另一个采集团队的数据上。所以这里刻意与榴莲实验对齐：
同样的 5 折结构、同样的重复审计、同样的泄漏断言。

做四件事：
  [1] 从 HuggingFace 的 parquet 解出图像与框，写成 YOLO 标注
  [2] 重复审计：按 sha256 找逐位相同的图像，检查它们的 domain 与标注是否
      一致。榴莲那边就是靠内容比对查出 8.9% 的农场误标，同一道检查在这里
      也要做，而且不裁决冲突，整组剔除。
  [3] 建划分：5 折随机（图像级）+ 5 折 domain-disjoint
  [4] 断言：domain 折不得有任何 domain 跨越 train/val

用法：
    export HF_ENDPOINT=https://hf-mirror.com
    python prepare_gwhd.py --step download
    python prepare_gwhd.py --step build
    python prepare_gwhd.py --step audit        # 可单独重跑

产出：
    processed/images/*.jpg
    processed/labels/*.txt
    processed/manifest.csv            全部图像
    processed/manifest_clean.csv      去掉冲突重复后的分析集
    processed/duplicate_report.csv    重复组明细
"""

import argparse
import csv
import hashlib
import io
import os
import random
import sys
from collections import Counter, defaultdict

HF_REPO = "Etienne-David/GlobalWheatHeadDataset2021"


def step_download(a):
    from huggingface_hub import snapshot_download
    d = os.path.join(a.root, "hf_dataset")
    print(f"downloading {HF_REPO} -> {d}")
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=d,
                      allow_patterns=["data/*", "*.md"])
    for dp, _, fns in os.walk(d):
        p = [f for f in fns if f.endswith(".parquet")]
        if p:
            print(f"  {dp}: {len(p)} parquet")
    return 0


# ------------------------------------------------------------------ build --
def load_parquet(root):
    import pyarrow.parquet as pq
    files = []
    for dp, _, fns in os.walk(os.path.join(root, "hf_dataset")):
        for f in sorted(fns):
            if f.endswith(".parquet"):
                files.append(os.path.join(dp, f))
    if not files:
        sys.exit("找不到 parquet，先跑 --step download")
    print(f"parquet 文件 {len(files)} 个")
    for f in files:
        t = pq.read_table(f)
        split = ("train" if "train" in os.path.basename(f)
                 else "validation" if "validation" in os.path.basename(f)
                 else "test")
        yield split, t


def step_build(a):
    from PIL import Image

    proc = os.path.join(a.root, "processed")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(proc, sub), exist_ok=True)

    rows = []
    idx_in_split = Counter()
    for split, table in load_parquet(a.root):
        cols = table.column_names
        print(f"\n{split}: {table.num_rows} rows   columns={cols}")
        d = table.to_pylist()
        for r in d:
            i = idx_in_split[split]
            idx_in_split[split] += 1
            iid = f"{split}_{i:06d}"

            img = r.get("image")
            raw = img.get("bytes") if isinstance(img, dict) else img
            if raw is None:
                continue
            sha = hashlib.sha256(raw).hexdigest()

            try:
                im = Image.open(io.BytesIO(raw))
                im = im.convert("RGB")
                W, H = im.size
            except Exception as e:
                print(f"  解码失败 {iid}: {e}")
                continue

            ip = os.path.join(proc, "images", iid + ".jpg")
            if not os.path.isfile(ip):
                im.save(ip, quality=95)

            obj = r.get("objects") or {}
            boxes = obj.get("boxes") or []
            n_bad = 0
            lines = []
            for b in boxes:
                if len(b) != 4:
                    n_bad += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in b)
                x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
                y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    n_bad += 1
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f" % (
                    (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                    (x2 - x1) / W, (y2 - y1) / H))
            lp = os.path.join(proc, "labels", iid + ".txt")
            with open(lp, "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))

            rows.append({
                "image_id": iid, "image_path": os.path.abspath(ip),
                "label_path": os.path.abspath(lp),
                "source_split": split, "source_index": i,
                "domain": str(r.get("domain", "")).strip(),
                "country": str(r.get("country", "")).strip(),
                "location": str(r.get("location", "")).strip(),
                "development_stage": str(r.get("development_stage", "")).strip(),
                "width": W, "height": H,
                "n_boxes": len(lines), "bad_boxes_removed": n_bad,
                "sha256": sha,
            })
        print(f"  写出 {idx_in_split[split]} 张")

    if not rows:
        sys.exit("没有解出任何图像")
    write_manifest(os.path.join(proc, "manifest.csv"), rows)
    print(f"\n合计 {len(rows)} 张，"
          f"{sum(r['n_boxes'] for r in rows)} 个框，"
          f"{len({r['domain'] for r in rows})} 个 domain")
    bad = sum(r["bad_boxes_removed"] for r in rows)
    if bad:
        print(f"剔除退化框 {bad} 个（宽或高不足 1 px）")
    return step_audit(a)


def write_manifest(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {path}  ({len(rows)} 行)")


# ------------------------------------------------------------------ audit --
def step_audit(a):
    proc = os.path.join(a.root, "processed")
    mp = os.path.join(proc, "manifest.csv")
    if not os.path.isfile(mp):
        sys.exit("先跑 --step build")
    with open(mp, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    print("\n" + "=" * 70)
    print("[1] 重复审计（按 sha256 逐位相同）")
    print("=" * 70)
    g = defaultdict(list)
    for r in rows:
        g[r["sha256"]].append(r)
    dup = {k: v for k, v in g.items() if len(v) > 1}
    print(f"  唯一图像 {len(g)} / {len(rows)}")
    print(f"  重复组 {len(dup)}，涉及 {sum(len(v) for v in dup.values())} 行")

    conflict, benign = [], []
    for k, v in dup.items():
        doms = {r["domain"] for r in v}
        nb = {r["n_boxes"] for r in v}
        splits = {r["source_split"] for r in v}
        (conflict if (len(doms) > 1 or len(nb) > 1) else benign).append(
            (k, v, doms, nb, splits))
    print(f"    domain 或标注不一致的: {len(conflict)}  ← 剔除")
    print(f"    完全一致的:           {len(benign)}  ← 保留一份")

    cross = [x for x in dup.values()
             if len({r["source_split"] for r in x}) > 1]
    print(f"  跨 split 的重复组: {len(cross)}  "
          f"(官方划分本身的泄漏)")

    rep = os.path.join(proc, "duplicate_report.csv")
    with open(rep, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "n_copies", "image_ids", "domains",
                    "n_boxes_values", "source_splits", "verdict"])
        for k, v, doms, nb, sp in conflict + [
                (k, v, {r["domain"] for r in v}, {r["n_boxes"] for r in v},
                 {r["source_split"] for r in v}) for k, v, _, _, _ in benign]:
            w.writerow([k, len(v), ";".join(r["image_id"] for r in v),
                        ";".join(sorted(doms)),
                        ";".join(sorted(str(x) for x in nb)),
                        ";".join(sorted(sp)),
                        "drop" if (len(doms) > 1 or len(nb) > 1) else "keep_one"])
    print(f"  明细 -> {rep}")
    if conflict:
        print("\n  冲突组样例（不裁决哪一边对，整组剔除）:")
        for k, v, doms, nb, sp in conflict[:5]:
            print(f"    {k[:12]}  domains={sorted(doms)}  "
                  f"n_boxes={sorted(nb)}  splits={sorted(sp)}")

    drop_ids = {r["image_id"] for _, v, _, _, _ in conflict for r in v}
    keep, seen = [], set()
    for r in rows:
        if r["image_id"] in drop_ids:
            continue
        if r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        keep.append(r)
    print(f"\n  分析集 {len(keep)} / {len(rows)} "
          f"(剔除冲突 {len(drop_ids)}，去重 {len(rows)-len(drop_ids)-len(keep)})")

    # ---------------- 划分 ----------------
    print("\n" + "=" * 70)
    print("[2] 建立划分")
    print("=" * 70)
    rnd = random.Random(a.seed)
    doms = sorted({r["domain"] for r in keep})
    print(f"  domain {len(doms)} 个")

    # 随机 5 折：图像级
    idx = list(range(len(keep)))
    rnd.shuffle(idx)
    for pos, i in enumerate(idx):
        keep[i]["random_fold"] = pos % a.folds

    # domain 5 折：按图像数贪心装箱，使各折规模接近
    size = Counter()
    for r in keep:
        size[r["domain"]] += 1
    bins = [[] for _ in range(a.folds)]
    load = [0] * a.folds
    for d, n in size.most_common():
        j = load.index(min(load))
        bins[j].append(d)
        load[j] += n
    dom_fold = {d: j for j, ds in enumerate(bins) for d in ds}
    for r in keep:
        r["domain_fold"] = dom_fold[r["domain"]]

    write_manifest(os.path.join(proc, "manifest_clean.csv"), keep)

    print(f"\n  {'fold':<6}{'random val':>12}{'domain val':>12}"
          f"{'domains':>9}")
    for f in range(a.folds):
        rv = sum(1 for r in keep if r["random_fold"] == f)
        dv = sum(1 for r in keep if r["domain_fold"] == f)
        nd = len({r["domain"] for r in keep if r["domain_fold"] == f})
        print(f"  {f:<6}{rv:>12}{dv:>12}{nd:>9}")

    # ---------------- 断言 ----------------
    print("\n" + "=" * 70)
    print("[3] 泄漏断言")
    print("=" * 70)
    bad = 0
    for f in range(a.folds):
        tr = {r["domain"] for r in keep if r["domain_fold"] != f}
        va = {r["domain"] for r in keep if r["domain_fold"] == f}
        if tr & va:
            bad += 1
            print(f"  ★ domain_fold{f}: {sorted(tr & va)} 跨界")
    print(f"  domain 跨折: {'无' if bad == 0 else f'★ {bad} 折有问题'}")

    h = defaultdict(set)
    for r in keep:
        h[r["sha256"]].add(r["domain_fold"])
    dupfold = sum(1 for v in h.values() if len(v) > 1)
    print(f"  同一图像跨折: {'无' if dupfold == 0 else f'★ {dupfold} 组'}")

    rf = defaultdict(set)
    for r in keep:
        rf[r["sha256"]].add(r["random_fold"])
    print(f"  随机折中同一图像跨折: "
          f"{sum(1 for v in rf.values() if len(v) > 1)}  "
          f"(去重后应为 0)")

    print(f"""
下一步：
    python gwhd_run.py --step splits --root {a.root}
    nohup python gwhd_run.py --step train --models yolo11s --seeds 42 \\
        --epochs 150 --patience 50 --yes > pilot.log 2>&1 &
""")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["download", "build", "audit"])
    ap.add_argument("--root", default="/root/autodl-tmp/gwhd2021")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    return {"download": step_download, "build": step_build,
            "audit": step_audit}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
