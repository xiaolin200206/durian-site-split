#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_pool_join.py -- 标注集里的每张图，是不是真的对应它被 join 到的那条 metadata？

已确认的问题:
    标注集的 IMG_9885.jpg 与 IMG_9885(1).HEIC 感知哈希距离 0，
    与同名的 IMG_9885.HEIC 距离 >2。两个原图是不同照片、各自带 GPS、
    分属 farm 6 和 farm 0。
    -> 标注图的真身是 farm 6 那张，但按文件名 join 后被标成了 farm 0。
    起因是导入时重名产生 (1) 后缀，导出/缩放环节又把后缀丢了。

后果: 该图进入了错误农场的 fold。farm 0 在 fold 0，farm 6 在 fold 4，
      于是一张 farm 6 的照片进了 fold 0 的训练集，而它的同伴在
      fold 4 的验证集里 —— 正是本文所论证的那种泄漏。

本脚本对池内每张图做:
    d_claim = dist(标注图, 与其同名的原图)
    d_best  = dist(标注图, 全体原图中最近的一张)
    若 d_claim 明显大于 d_best，则 join 错了；再看真身的 farm 是否不同。

复用 .phash_cache.csv。只读，不改任何文件；更正清单单独写成 csv。

用法:
    python audit_pool_join.py
    python audit_pool_join.py --tol 2 --show 60
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
    ap.add_argument("--split-assignment",
                    default="dataset_store/splits_A/split_assignment.csv")
    ap.add_argument("--tol", type=int, default=2,
                    help="d_claim 超过此值即认为 join 可疑")
    ap.add_argument("--show", type=int, default=60)
    ap.add_argument("--out", default="pool_join_corrections.csv")
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

    # 池与 fold
    pool, fold_of, farm_claim = set(), {}, {}
    sa = root.joinpath(*re.split(r"[\\/]", args.split_assignment))
    if not sa.is_file():
        print(f"找不到 {sa}")
        return 1
    with open(sa, newline="", encoding="utf-8-sig", errors="replace") as f:
        rd = csv.DictReader(f)
        foldcols = [c for c in (rd.fieldnames or [])
                    if c and re.match(r"split_farm_fold\d+$", c)]
        for r in rd:
            s = norm_stem(r.get("stem") or "")
            if not s:
                continue
            pool.add(s)
            farm_claim[s] = (r.get("farm") or "").strip()
            for c in foldcols:
                if (r.get(c) or "").strip().lower() == "val":
                    fold_of[s] = c
    print(f"池 {len(pool)} 张 | fold 列 {foldcols} | 有 val 归属 {len(fold_of)}")

    # 原图哈希（非派生）
    orig = []
    orig_by_stem = defaultdict(list)
    merged_hash = {}
    src = merged / "images" if (merged := root.joinpath(
        *re.split(r"[\\/]", args.merged))) and (merged / "images").is_dir() \
        else merged
    for k, h in cache.items():
        p = Path(k)
        if p.suffix.lower() not in IMG_EXT:
            continue
        if is_derived(p, root):
            try:
                if src.resolve() in p.resolve().parents or p.parent == src:
                    merged_hash[norm_stem(p.name)] = h
            except OSError:
                pass
            continue
        orig.append((p, h))
        orig_by_stem[norm_stem(p.name)].append((p, h))
    print(f"原图哈希 {len(orig)} | 标注集哈希 {len(merged_hash)}\n")

    def farm_of_path(p):
        m = meta.get(norm_stem(p.name))
        return (m or {}).get("farm", "") or ""

    print("=" * 78)
    print("[1] 逐张校验池内图片与其 metadata 记录是否对应")
    print("=" * 78)
    bad, ok, nocheck = [], 0, []
    for i, s in enumerate(sorted(pool), 1):
        hm = merged_hash.get(s)
        if hm is None:
            nocheck.append((s, "标注图无哈希"))
            continue
        claim = orig_by_stem.get(s)
        if not claim:
            nocheck.append((s, "同名原图不在盘上"))
            continue
        d_claim = min(popcount(hm ^ h) for _, h in claim)
        if d_claim <= args.tol:
            ok += 1
            continue
        # join 可疑，找真身
        best_p, best_d = None, 65
        for p, h in orig:
            d = popcount(hm ^ h)
            if d < best_d:
                best_d, best_p = d, p
        bad.append((s, d_claim, best_p, best_d))
        if i % 200 == 0:
            print(f"    校验中 {i}/{len(pool)}", flush=True)

    print(f"\n  对应正确 (d<= {args.tol}) : {ok}")
    print(f"  join 可疑              : {len(bad)}")
    if nocheck:
        print(f"  无法校验               : {len(nocheck)}  "
              f"例: {[x[0] for x in nocheck[:4]]}")

    # ---------- 分类 ----------
    print("\n" + "=" * 78)
    print("[2] 可疑项里，有多少真的改变了 farm")
    print("=" * 78)
    changed, samefarm, unresolved = [], 0, 0
    for s, d_claim, bp, bd in bad:
        if bp is None or bd > args.tol:
            unresolved += 1
            continue
        f_true = farm_of_path(bp)
        f_claim = farm_claim.get(s, "")
        if f_true and f_claim and f_true != f_claim:
            changed.append((s, f_claim, f_true, bp, d_claim, bd))
        else:
            samefarm += 1
    print(f"  真身 farm 与标注 farm 不同 : {len(changed)}   ★ 这些是误标")
    print(f"  真身不同但 farm 相同       : {samefarm}   (无害)")
    print(f"  找不到可信真身             : {unresolved}")

    if changed:
        fc = Counter(f"{a}->{b}" for _, a, b, _, _, _ in changed)
        print(f"\n  farm 迁移方向: {dict(fc.most_common())}")
        cross = Counter()
        for s, f_claim, f_true, _, _, _ in changed:
            cross[fold_of.get(s, "<在train>")] += 1
        print(f"  这些图当前所在的 val fold: {dict(cross.most_common())}")
        print(f"\n  明细 (最多 {args.show} 条):")
        print(f"    {'池内 stem':<18}{'标注farm':>9}{'真实farm':>9}"
              f"{'d_claim':>8}{'d_best':>7}   真身文件")
        for s, f_claim, f_true, bp, d_claim, bd in changed[:args.show]:
            print(f"    {s:<18}{f_claim:>9}{f_true:>9}"
                  f"{d_claim:>8}{bd:>7}   {bp.name}")

        out = root / args.out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "farm_claimed", "farm_true", "true_file",
                        "d_claim", "d_best", "current_val_fold"])
            for s, a, b, bp, dc, bd in changed:
                w.writerow([s, a, b, bp.name, dc, bd, fold_of.get(s, "")])
        print(f"\n  更正清单: {out}")

    # ---------- 影响评估 ----------
    print("\n" + "=" * 78)
    print("[3] 对论文的影响")
    print("=" * 78)
    n = len(pool)
    print(f"  池内误标率: {len(changed)}/{n} = {100*len(changed)/max(n,1):.1f}%")
    if changed:
        aff = Counter()
        for s, f_claim, f_true, _, _, _ in changed:
            aff[f_claim] += 1
        print(f"  受影响的农场(按当前标注): {dict(aff.most_common())}")
        print("""
  每一张误标的图, 都让某个农场的照片出现在另一个农场的 fold 里。
  若该图进的是训练集, 而它真正的农场是某折的验证集, 那一折的
  留出就不干净 —— 即本文所论证的泄漏。

  处理选项:
    A. 按更正清单改 metadata 的 farm 列, 重建划分, 重跑。
    B. 整体剔除这些图 (最保守, 但会缩小本已很小的池)。
  两种都要在 Methods 里如实交代发现过程。""")
    else:
        print("  未发现误标。池内的 join 是干净的。")

    # ---------- (1) 兄弟扫描 ----------
    print("\n" + "=" * 78)
    print("[4] metadata 里存在 '(1)' 兄弟且 farm 不同的 stem")
    print("=" * 78)
    sib = []
    for s in meta:
        for suf in ("(1)", "(2)"):
            t = s + suf
            if t in meta:
                fa = (meta[s].get("farm") or "").strip()
                fb = (meta[t].get("farm") or "").strip()
                if fa and fb and fa != fb:
                    sib.append((s, fa, t, fb, s in pool))
    print(f"  找到 {len(sib)} 对")
    for s, fa, t, fb, inpool in sib[:args.show]:
        print(f"    {s:<18} farm={fa:<4} | {t:<20} farm={fb:<4}"
              f"{'   <-在560池' if inpool else ''}")
    if sib:
        print("\n  这些是最高危的 stem: 文件名只差一个后缀, 拍摄地点却不同,")
        print("  任何按文件名做的 join 都可能接错。")
    print("\n把 [1][2][3][4] 发回来。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
