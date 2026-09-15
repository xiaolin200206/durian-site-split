#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_duplicates.py (v2) -- metadata.csv 里同一张照片是否被归到了不同的农场？

v1 的漏洞: 只比对"存在的原图文件"，且排除派生副本。但像 IMG_9885 这种，
原图已不在磁盘上，仅存 640x640 的派生拷贝，于是它整个没进比对，
和 IMG_9885(1) 的冲突就查不出来。

v2 改为以 **metadata 的行** 为单位:
    每条 metadata 记录取一个哈希，来源优先级 = 原图 > 派生副本(640x640)。
    dHash 对缩放不敏感, 两者距离通常为 0, 所以派生副本可以安全兜底。
然后把 metadata 行按哈希聚成"同一张照片"的组, 检查组内 farm 是否一致。

复用 match_by_phash 生成的 .phash_cache.csv。

用法:
    python audit_duplicates.py
    python audit_duplicates.py --tol 0
    python audit_duplicates.py --split-assignment github/results/split_assignment.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
           ".heic", ".heif"}
DERIVED = ("merged_peninsula", "merged_sabah", "zenodo_upload",
           "splits_a", "peninsula_subsets", "runs")
COPY_PAT = re.compile(r"\(\d+\)\s*$|[-_]copy\s*$|\s+copy\s*$", re.I)


def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def norm_stem(x):
    return Path(str(x)).stem.lower()


def is_derived(p, root):
    try:
        rel = p.relative_to(root).as_posix().lower()
    except ValueError:
        rel = p.as_posix().lower()
    return any(d in rel for d in DERIVED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--cache", default=".phash_cache.csv")
    ap.add_argument("--split-assignment", default=None)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--out", default="duplicate_farm_conflicts.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache_p = root / args.cache
    if not cache_p.is_file():
        print(f"找不到 {cache_p}，先跑 match_by_phash.py")
        return 1

    cache = {}
    with open(cache_p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) == 2:
                try:
                    cache[r[0]] = int(r[1])
                except ValueError:
                    pass

    meta = {}
    with open(root / args.meta, newline="", encoding="utf-8-sig",
              errors="replace") as f:
        for r in csv.DictReader(f):
            s = norm_stem(r.get("stem") or r.get("file") or "")
            if s:
                meta.setdefault(s, r)
    print(f"缓存 {len(cache)} 条 | metadata {len(meta)} 个唯一 stem")

    # ---------- split_assignment ----------
    pool, fold_of = set(), {}
    cands = []
    if args.split_assignment:
        p = root.joinpath(*re.split(r"[\\/]", args.split_assignment))
        if p.is_file():
            cands = [p]
    else:
        cands = sorted(root.rglob("split_assignment.csv"))
    if not cands:
        print("未找到 split_assignment.csv (可用 --split-assignment 指定)")
    for sa in cands[:1]:
        with open(sa, newline="", encoding="utf-8-sig", errors="replace") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            print(f"split_assignment: {sa}")
            print(f"  列名: {cols}")
            foldcols = [c for c in cols if c and "fold" in c.lower()]
            vals = Counter()
            for r in rd:
                s = norm_stem(r.get("stem") or r.get("image") or "")
                if not s:
                    continue
                pool.add(s)
                for c in foldcols:
                    v = (r.get(c) or "").strip().lower()
                    vals[v] += 1
                    if v in ("val", "valid", "validation", "1", "true"):
                        fold_of[s] = c
            print(f"  fold 列: {foldcols}")
            print(f"  这些列的取值分布: {dict(vals.most_common(6))}")
            print(f"  池 {len(pool)} 张, 解析出 fold 归属 {len(fold_of)} 张")
    if len(cands) > 1:
        print(f"  (另找到 {len(cands)-1} 个同名文件, 未使用)")

    # ---------- 每条 metadata 行取一个哈希 ----------
    by_stem_paths = defaultdict(list)
    for k in cache:
        p = Path(k)
        if p.suffix.lower() in IMG_EXT:
            by_stem_paths[norm_stem(p.name)].append(p)

    items, src_kind = [], Counter()
    no_hash = []
    for s, m in meta.items():
        paths = by_stem_paths.get(s, [])
        orig = [p for p in paths if not is_derived(p, root)]
        der = [p for p in paths if is_derived(p, root)]
        if orig:
            p, kind = orig[0], "原图"
        elif der:
            p, kind = der[0], "派生副本"
        else:
            no_hash.append(s)
            continue
        h = cache.get(str(p))
        if h is None:
            no_hash.append(s)
            continue
        src_kind[kind] += 1
        items.append((s, p, h, m, kind))

    print(f"\n参与审计的 metadata 行: {len(items)}   哈希来源 {dict(src_kind)}")
    if no_hash:
        print(f"  无可用哈希, 已跳过: {len(no_hash)} 行  例: {no_hash[:5]}")

    # ---------- 分组 ----------
    print("\n" + "=" * 78)
    print(f"[1] 按感知哈希聚成'同一张照片'的组  (阈值 <= {args.tol})")
    print("=" * 78)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    n = len(items)
    if args.tol == 0:
        by_h = defaultdict(list)
        for i in range(n):
            by_h[items[i][2]].append(i)
        for idxs in by_h.values():
            for j in idxs[1:]:
                union(idxs[0], j)
    else:
        for i in range(n):
            hi = items[i][2]
            for j in range(i + 1, n):
                if popcount(hi ^ items[j][2]) <= args.tol:
                    union(i, j)
            if n > 1500 and (i + 1) % 500 == 0:
                print(f"    比对中 {i+1}/{n}", flush=True)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  组数 {len(groups)} | 含多条的组 {len(multi)} | "
          f"涉及 metadata 行 {sum(len(v) for v in multi.values())}")
    if multi:
        sz = Counter(len(v) for v in multi.values())
        print(f"  组大小分布: {dict(sorted(sz.items()))}")

    # ---------- 冲突 ----------
    print("\n" + "=" * 78)
    print("[2] ★ 组内 farm 标注冲突")
    print("=" * 78)
    conflicts, rows = [], []
    for k, idxs in multi.items():
        farms = {(items[i][3].get("farm") or "").strip() for i in idxs}
        farms.discard("")
        if len(farms) > 1:
            conflicts.append(idxs)
    print(f"  冲突组数: {len(conflicts)}  涉及 metadata 行 "
          f"{sum(len(v) for v in conflicts)}")

    in_pool = [items[i][0] for idxs in conflicts for i in idxs
               if items[i][0] in pool]
    print(f"  其中落在论文 560 池里的: {len(in_pool)} 张")
    if fold_of:
        print(f"  按 fold 分布: "
              f"{dict(Counter(fold_of.get(s, '<不在val>') for s in in_pool))}")

    if conflicts:
        print(f"\n  明细 (最多 {args.show} 组):")
        for idxs in conflicts[:args.show]:
            print()
            for i in idxs:
                s, p, h, m, kind = items[i]
                tag = "  <-在560池" if s in pool else ""
                if s in fold_of:
                    tag += f" [{fold_of[s]}]"
                if COPY_PAT.search(Path(p.name).stem):
                    tag += " ★副本名"
                print(f"    {p.name:<30} farm={str(m.get('farm')):<4}"
                      f" src={str(m.get('farm_source')):<14}"
                      f" {str(m.get('DateTimeOriginal'))[:19]:<20}"
                      f" gps={'有' if (m.get('lat') or '').strip() else '无'}"
                      f" [{kind}]{tag}")
        for idxs in conflicts:
            for i in idxs:
                s, p, h, m, kind = items[i]
                rows.append({
                    "group": idxs[0], "stem": s, "file": p.name,
                    "hash_source": kind,
                    "farm": m.get("farm", ""),
                    "farm_source": m.get("farm_source", ""),
                    "DateTimeOriginal": m.get("DateTimeOriginal", ""),
                    "has_gps": bool((m.get("lat") or "").strip()),
                    "copy_name": bool(COPY_PAT.search(Path(p.name).stem)),
                    "in_pool": s in pool,
                    "fold": fold_of.get(s, ""),
                })

        print("\n" + "=" * 78)
        print("[3] 哪一边可能是对的")
        print("=" * 78)
        srcs = Counter(items[i][3].get("farm_source") or "<空>"
                       for idxs in conflicts for i in idxs)
        tot = sum(len(v) for v in conflicts)
        print(f"  farm_source 分布: {dict(srcs.most_common())}")
        print(f"  带 GPS 的: "
              f"{sum(1 for idxs in conflicts for i in idxs if (items[i][3].get('lat') or '').strip())} / {tot}")
        print(f"  文件名带 (1)/copy 的: "
              f"{sum(1 for idxs in conflicts for i in idxs if COPY_PAT.search(Path(items[i][1].name).stem))}")
        print("""
  判据 (按可信度排序):
    1. farm_source=gps 且带 GPS 的那条最可信
    2. 文件名不带 (1)/copy 的原件优先
    3. 两边都靠时间推定 -> 整组存疑, 建议剔除

  最要紧的是标了 <-在560池 的行: 它们已进入论文的训练/验证划分。""")
    else:
        print("\n  未发现冲突。")
        print("  注意: 若上面 '哈希来源' 里派生副本占比很高, 说明大量原图已不在磁盘,")
        print("  结论仍然成立(dHash 对缩放不敏感), 但值得确认 verify_phash 报的")
        print("  那几例 (IMG_9885 / IMG_2122) 是否出现在本次分组里。")

    if rows:
        out = root / args.out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n写出: {out}")

    # ---------- 定点复核 ----------
    print("\n" + "=" * 78)
    print("[4] 定点复核 verify_phash 报出的那几例")
    print("=" * 78)
    idx_of = {items[i][0]: i for i in range(n)}
    for base in ("img_9885", "img_9886", "img_2122"):
        pair = [s for s in (base, base + "(1)")]
        print(f"\n  {base}:")
        for s in pair:
            m = meta.get(s)
            if m is None:
                print(f"    {s:<16} 不在 metadata")
                continue
            i = idx_of.get(s)
            if i is None:
                print(f"    {s:<16} farm={m.get('farm')}  "
                      f"src={m.get('farm_source')}  ★ 无哈希, 未参与分组")
                continue
            gid = find(i)
            print(f"    {s:<16} farm={str(m.get('farm')):<4} "
                  f"src={str(m.get('farm_source')):<14} "
                  f"组={gid} 组内 {len(groups[gid])} 条 [{items[i][4]}]")
    print("\n把 [2][3][4] 发回来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
