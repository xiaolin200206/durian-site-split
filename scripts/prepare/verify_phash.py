#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_phash.py -- 核实 match_by_phash 的 25 个"错误"到底是不是错误，
                   并用更稳健的判据重新裁定 463 张的 farm 归属。

两件事:

[1] 误判诊断
    怀疑那 25 个不是匹配错，是项目树里存在同一张照片的改名副本
    (annotation_scale_check\\Algal\\f00_a0000041_n018_IMG_9813.jpg 就是
    IMG_9813 的拷贝)。逐个打印: 汉明距离、匹配到的文件名、来自哪个目录、
    真 stem 是否被包含在匹配名里、两者的 farm 是否一致。
    若这些距离都是 0 且 farm 一致，它们就不是错误。

[2] 农场共识判据
    对论文而言真正要紧的不是"匹配到哪一张原图"，而是"归到哪个农场"。
    所以改用: 取距离最优值 +TOL 以内的全部近邻，看它们的 farm 是否一致。
    全体一致才接受。这比"最近邻唯一"稳健得多，且直接针对
    "归错农场 -> fold 间泄漏"这个真正的风险。

复用 match_by_phash 生成的 .phash_cache.csv，不重新解码图片，几秒跑完。

用法:
    python verify_phash.py
    python verify_phash.py --tol 2 --max-dist 8
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RASTER = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIC = {".heic", ".heif"}
IMG_EXT = RASTER | HEIC
DERIVED = ("merged_peninsula", "merged_sabah", "zenodo_upload",
           "splits_a", "peninsula_subsets", "runs")


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
    ap.add_argument("--merged", default="dataset_store/merged_peninsula")
    ap.add_argument("--cache", default=".phash_cache.csv")
    ap.add_argument("--tol", type=int, default=2,
                    help="共识半径: 最优距离 +tol 以内的都算近邻")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--show", type=int, default=30, help="最多打印多少条误判明细")
    ap.add_argument("--out", default="recovered_farms_verified.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    merged = root.joinpath(*re.split(r"[\\/]", args.merged))
    cache_p = root / args.cache
    if not cache_p.is_file():
        print(f"找不到 {cache_p}。先跑一遍 match_by_phash.py 生成缓存。")
        return 1

    cache = {}
    with open(cache_p, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) == 2:
                try:
                    cache[r[0]] = int(r[1])
                except ValueError:
                    pass
    print(f"缓存哈希 {len(cache)} 条")

    meta = {}
    with open(root / args.meta, newline="", encoding="utf-8-sig",
              errors="replace") as f:
        for r in csv.DictReader(f):
            s = norm_stem(r.get("stem") or r.get("file") or "")
            if s:
                meta.setdefault(s, r)

    # 参考集 = 缓存里所有非派生副本的路径
    ref = []
    for k, h in cache.items():
        p = Path(k)
        if p.suffix.lower() not in IMG_EXT:
            continue
        if is_derived(p, root):
            continue
        ref.append((p, h))
    print(f"参考原图 {len(ref)} 张")

    src = merged / "images" if (merged / "images").is_dir() else merged
    merged_imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    known = [p for p in merged_imgs if norm_stem(p.name) in meta]
    unknown = [p for p in merged_imgs if norm_stem(p.name) not in meta]
    print(f"标注集 {len(merged_imgs)} | 已知 {len(known)} | 待定 {len(unknown)}\n")

    def neighbours(h):
        """返回 (best_dist, [(path, dist), ...] 在 best+tol 以内)"""
        best = 65
        ds = []
        for p, rh in ref:
            d = popcount(h ^ rh)
            ds.append((p, d))
            if d < best:
                best = d
        near = [(p, d) for p, d in ds if d <= best + args.tol]
        return best, near

    def farm_of(p):
        m = meta.get(norm_stem(p.name))
        return (m or {}).get("farm", "") or None

    def dirname(p):
        try:
            return p.relative_to(root).parts[0]
        except ValueError:
            return p.parent.name

    # ---------------- [1] 误判诊断 ----------------
    print("=" * 78)
    print("[1] 误判诊断 —— 那些'匹配错'的到底是什么")
    print("=" * 78)
    strict_ok = same_photo = farm_ok = 0
    detail = []
    for p in known:
        h = cache.get(str(p))
        if h is None:
            continue
        s = norm_stem(p.name)
        best, near = neighbours(h)
        bp = min(near, key=lambda t: t[1])[0]
        bs = norm_stem(bp.name)
        if bs == s:
            strict_ok += 1
            continue
        # 严格判据下算"错"。看看是不是同一张照片的改名副本
        embedded = s in bs or bs in s
        f_true, f_match = farm_of(p), farm_of(bp)
        if embedded or best == 0:
            same_photo += 1
        if f_true and f_match and f_true == f_match:
            farm_ok += 1
        detail.append((s, bp.name, dirname(bp), best, embedded, f_true, f_match))

    n = strict_ok + len(detail)
    print(f"  检验 {n} 对")
    print(f"    文件名严格相同        : {strict_ok}  ({100*strict_ok/max(n,1):.1f}%)")
    print(f"    名字不同但是同一张照片 : {same_photo}")
    print(f"    名字不同且 farm 也一致 : {farm_ok}")
    eff = strict_ok + max(same_photo, farm_ok)
    print(f"    -> 有效准确率约        : {100*eff/max(n,1):.1f}%")

    if detail:
        print(f"\n  明细 (最多 {args.show} 条):")
        print(f"    {'标注集 stem':<24}{'匹配到':<40}{'目录':<24}"
              f"{'距离':>5}{'含真名':>7}{'真farm':>8}{'匹farm':>8}")
        for d in detail[:args.show]:
            print(f"    {d[0][:23]:<24}{d[1][:39]:<40}{d[2][:23]:<24}"
                  f"{d[3]:>5}{('是' if d[4] else '否'):>7}"
                  f"{str(d[5]):>8}{str(d[6]):>8}")
        dd = Counter(d[2] for d in detail)
        print(f"\n  误判来源目录分布: {dict(dd.most_common())}")
        print("  判读: 若这些距离多为 0、且来自 annotation_scale_check /")
        print("        lesion_scale_check 这类副本目录，就不是匹配错误，")
        print("        原来的 93.8% 应读作接近 100%。")

    # ---------------- [2] 农场共识判据 ----------------
    print("\n" + "=" * 78)
    print(f"[2] 农场共识判据重裁 {len(unknown)} 张  (共识半径 = 最优 +{args.tol})")
    print("=" * 78)
    rows = []
    cnt = Counter()
    nb_hist = Counter()
    for p in unknown:
        h = cache.get(str(p))
        if h is None:
            cnt["no_hash"] += 1
            continue
        best, near = neighbours(h)
        farms = {farm_of(q) for q, _ in near}
        farms.discard(None)
        nb_hist[len(near)] += 1
        if best > args.max_dist:
            status = "too_far"
        elif not farms:
            status = "no_farm"
        elif len(farms) > 1:
            status = "farm_conflict"
        else:
            status = "ok"
        cnt[status] += 1
        bp = min(near, key=lambda t: t[1])[0]
        rows.append({"merged_stem": norm_stem(p.name),
                     "matched_original": bp.name,
                     "matched_dir": dirname(bp),
                     "best_dist": best,
                     "n_neighbours": len(near),
                     "farms_in_consensus": "|".join(sorted(farms)),
                     "status": status,
                     "farm": (list(farms)[0] if len(farms) == 1 else "")})

    print(f"  一致接受 {cnt['ok']} | 农场冲突 {cnt['farm_conflict']} | "
          f"近邻无归属 {cnt['no_farm']} | 距离过大 {cnt['too_far']}")
    print(f"  近邻数分布 (1 表示唯一): "
          f"{dict(sorted(nb_hist.items())[:6])}")
    if cnt["farm_conflict"]:
        print(f"\n  ★ {cnt['farm_conflict']} 张的近邻分属不同农场，已拒绝。")
        print("    这些正是会造成 fold 间泄漏的那类，宁可丢掉。")

    # ---------------- [3] farm 分布 ----------------
    print("\n" + "=" * 78)
    print("[3] 通过共识判据后的 farm 分布")
    print("=" * 78)
    old = {"0": 165, "2": 154, "3": 59, "5": 141, "6": 39, "7": 1, "8": 1}
    c = Counter(r["farm"] for r in rows if r["status"] == "ok" and r["farm"])
    print(f"    {'farm':<10}{'新增':>8}{'原有':>8}{'合计':>8}   说明")
    new_total = 0
    for f, k in sorted(c.items(), key=lambda kv: -kv[1]):
        o = old.get(f, 0)
        note = "论文已用" if o >= 20 else "★ 新可用农场"
        print(f"    {f:<10}{k:>8}{o:>8}{k+o:>8}   {note}")
        new_total += k
    for f, o in sorted(old.items()):
        if f not in c:
            print(f"    {f:<10}{0:>8}{o:>8}{o:>8}")
    print(f"\n    分析池: 560 -> {560 + new_total}")
    allf = set(c) | set(old)
    usable = sorted(f for f in allf if c.get(f, 0) + old.get(f, 0) >= 20)
    newf = [f for f in usable if old.get(f, 0) < 20]
    print(f"    >=20 张的农场: {len(usable)} 个 -> {usable}")
    if newf:
        print(f"    ★★ 新增可用农场 {newf}，n 从 5 提到 {len(usable)}")

    if rows:
        out = root / args.out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n写出: {out}")
    print("\n把 [1] 的明细和 [3] 发回来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
