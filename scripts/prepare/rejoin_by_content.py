#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rejoin_by_content.py -- 放弃按文件名 join，改由感知哈希决定每张标注图的农场。

背景: audit_pool_join 确认池内 8.9% (50/560) 的农场标注是错的。
成因是 farm 0 与 farm 6 两次走访拍出重号的 IMG_98xx/99xx，导入时
后到的一批被加 (1) 后缀，缩放/导出环节又把后缀抹掉，于是按文件名
join 接到了另一张照片的 metadata 记录上。fold 0 的验证集因此有
29.7% 实为 farm 6 的照片，而 farm 6 在该折是训练集 —— 直接洩漏。

本脚本不去逐条裁决哪一边对，而是换掉判据:
    每张标注图的农场 = 与它感知哈希最接近的那张原图的农场。

同时检查同名覆盖是否造成丢图: 若 IMG_x.HEIC 和 IMG_x(1).HEIC
都被标注过，缩放后同名相撞，只会剩下一张。

复用 .phash_cache.csv。只读，更正结果写入新 csv，不改动任何原始文件。

用法:
    python rejoin_by_content.py
    python rejoin_by_content.py --max-dist 8 --min-margin 3
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
COPY_PAT = re.compile(r"^(.*?)\((\d+)\)$")


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
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--min-margin", type=int, default=3,
                    help="最近邻与'最近的异农场邻居'之间的最小距离差")
    ap.add_argument("--out", default="attribution_by_content.csv")
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

    old_farm, pool = {}, set()
    sa = root.joinpath(*re.split(r"[\\/]", args.split_assignment))
    if sa.is_file():
        with open(sa, newline="", encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                s = norm_stem(r.get("stem") or "")
                if s:
                    pool.add(s)
                    old_farm[s] = (r.get("farm") or "").strip()
    print(f"metadata {len(meta)} | 旧池 {len(pool)}")

    merged = root.joinpath(*re.split(r"[\\/]", args.merged))
    src = merged / "images" if (merged / "images").is_dir() else merged

    orig, merged_hash = [], {}
    for k, h in cache.items():
        p = Path(k)
        if p.suffix.lower() not in IMG_EXT:
            continue
        if is_derived(p, root):
            if p.parent == src:
                merged_hash[norm_stem(p.name)] = h
            continue
        m = meta.get(norm_stem(p.name))
        farm = (m or {}).get("farm", "").strip() if m else ""
        orig.append((p, h, farm, m))
    print(f"原图 {len(orig)} | 标注图 {len(merged_hash)}\n")

    def norm_farm(x):
        """'0.0' 和 '0' 统一。"""
        x = (x or "").strip()
        if not x:
            return ""
        try:
            return str(int(float(x)))
        except ValueError:
            return x

    # ---------- 逐张按内容定农场 ----------
    print("=" * 78)
    print("[1] 按内容重建农场归属")
    print("=" * 78)
    rows, cnt = [], Counter()
    for i, (s, hm) in enumerate(sorted(merged_hash.items()), 1):
        best_d, best_p, best_f = 65, None, ""
        other_d = 65                      # 最近的"异农场"邻居
        for p, h, farm, m in orig:
            d = popcount(hm ^ h)
            if d < best_d:
                if norm_farm(best_f) and norm_farm(farm) != norm_farm(best_f):
                    other_d = min(other_d, best_d)
                best_d, best_p, best_f = d, p, farm
            elif norm_farm(farm) and norm_farm(farm) != norm_farm(best_f):
                other_d = min(other_d, d)
        margin = other_d - best_d
        nf = norm_farm(best_f)
        if best_d > args.max_dist:
            st = "too_far"
        elif not nf:
            st = "no_farm"
        elif margin < args.min_margin:
            st = "ambiguous"
        else:
            st = "ok"
        cnt[st] += 1
        of = norm_farm(old_farm.get(s, ""))
        rows.append({"stem": s, "farm_new": nf if st == "ok" else "",
                     "farm_old": of, "changed": bool(of and nf and of != nf),
                     "matched_file": best_p.name if best_p else "",
                     "dist": best_d, "margin_to_other_farm": margin,
                     "status": st, "in_old_pool": s in pool,
                     "farm_source": (meta.get(norm_stem(best_p.name), {})
                                     .get("farm_source", "") if best_p else "")})
        if i % 200 == 0:
            print(f"    {i}/{len(merged_hash)}", flush=True)

    print(f"\n  可信归属 {cnt['ok']} | 跨农场歧义 {cnt['ambiguous']} | "
          f"近邻无归属 {cnt['no_farm']} | 距离过大 {cnt['too_far']}")

    # ---------- 与旧归属对比 ----------
    print("\n" + "=" * 78)
    print("[2] 与旧的按文件名 join 对比")
    print("=" * 78)
    ok = [r for r in rows if r["status"] == "ok"]
    inpool = [r for r in ok if r["in_old_pool"]]
    chg = [r for r in inpool if r["changed"]]
    print(f"  旧池 {len(pool)} 张中，本次给出可信归属的: {len(inpool)}")
    print(f"  其中农场改变: {len(chg)}  "
          f"({100*len(chg)/max(len(inpool),1):.1f}%)")
    if chg:
        moves = Counter(r["farm_old"] + "->" + r["farm_new"] for r in chg)
        print(f"  迁移方向: {dict(moves.most_common())}")

    print(f"\n  {'farm':<8}{'旧池':>8}{'新归属':>8}{'变化':>8}")
    of_c = Counter(norm_farm(v) for v in old_farm.values())
    nf_c = Counter(r["farm_new"] for r in ok)
    for f in sorted(set(of_c) | set(nf_c), key=lambda x: (len(x), x)):
        if not f:
            continue
        a, b = of_c.get(f, 0), nf_c.get(f, 0)
        print(f"  {f:<8}{a:>8}{b:>8}{b-a:>+8}")
    usable = sorted(f for f, n in nf_c.items() if f and n >= 20)
    print(f"\n  新池合计: {len(ok)} 张   (旧池 560)")
    print(f"  >=20 张的农场: {len(usable)} 个 -> {usable}")

    # ---------- 同名覆盖丢图 ----------
    print("\n" + "=" * 78)
    print("[3] 同名覆盖是否造成丢图")
    print("=" * 78)
    sib = defaultdict(list)
    for s in meta:
        m = COPY_PAT.match(s)
        base = m.group(1) if m else s
        sib[base].append(s)
    coll = {b: v for b, v in sib.items() if len(v) > 1}
    print(f"  metadata 里存在同基名多版本的组: {len(coll)}")
    lost = 0
    both_ann = 0
    for b, members in coll.items():
        present = [s for s in members if s in merged_hash]
        if b in merged_hash and len(members) > 1:
            both_ann += 1
            if len(present) < len(members):
                lost += len(members) - len(present)
    print(f"  这些组里，标注集只保留了一张的: {both_ann}")
    print(f"  据此推算可能被同名覆盖掉的图: 最多 {lost} 张")
    print("""
  判读: 缩放输出统一写成 <stem>.jpg，若 IMG_x 与 IMG_x(1) 都被标注，
        后写的会覆盖先写的。上面这个数是上界，不是确证 —— 也可能
        当初就只标注了其中一张。要确证，比对标注平台的导出清单。""")

    out = root / args.out
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出: {out}")
    print("""
下一步 (确认数字合理后):
  1. 用本表的 farm_new 重建 GroupKFold 划分
  2. 重跑训练 (旧的 30 run 约 190 分钟)
  3. Methods 里写明: 农场归属由感知哈希对原图匹配确定，不按文件名，
     并说明是按文件名 join 的碰撞促成了这个改动
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
