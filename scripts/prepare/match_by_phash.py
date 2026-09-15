#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_by_phash.py -- 用感知哈希把被平台重命名的标注图匹配回原图，恢复 farm 归属。

v2 变更:
  - 支持 HEIC/HEIF（iPhone 原生格式）。原图大部分是 .heic，v1 全漏掉了。
  - 原图定位改为在整个项目树建索引，不再只扫一个目录，
    并自动排除 merged_* / zenodo_upload 这些重采样过的派生副本。
  - 开头打印扩展名分布，这类「格式没收进来」的问题一眼可见。

依赖:
    pip install Pillow pillow-heif
  （缺 pillow-heif 时脚本仍能跑，但会跳过所有 .heic 并明确告警）

用法:
    python match_by_phash.py
    python match_by_phash.py --max-dist 10
"""

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("需要 Pillow:  pip install Pillow")
    sys.exit(1)

HEIC_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    pass

RASTER = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIC = {".heic", ".heif"}
IMG_EXT = RASTER | HEIC
HASH_W = 9                      # dHash: 缩到 9x8 -> 64 bit
DERIVED = ("merged_peninsula", "merged_sabah", "zenodo_upload",
           "splits_a", "peninsula_subsets", "runs")


def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def dhash(path):
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("L").resize((HASH_W, HASH_W - 1), Image.BILINEAR)
            px = im.tobytes()
    except Exception:
        return None
    bits = 0
    for row in range(HASH_W - 1):
        base = row * HASH_W
        for col in range(HASH_W - 1):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def norm_stem(x):
    return Path(str(x)).stem.lower()


def is_derived(p, root):
    try:
        rel = p.relative_to(root).as_posix().lower()
    except ValueError:
        rel = p.as_posix().lower()
    return any(d in rel for d in DERIVED)


def index_root(root, merged):
    """在整个项目树建索引: stem -> [候选原图路径]，排除派生副本。"""
    idx = defaultdict(list)
    ext_all, ext_kept = Counter(), Counter()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        e = p.suffix.lower()
        if e not in IMG_EXT:
            continue
        ext_all[e] += 1
        if is_derived(p, root):
            continue
        ext_kept[e] += 1
        idx[norm_stem(p.name)].append(p)
    return idx, ext_all, ext_kept


def hash_all(paths, label, cache):
    out, skipped = {}, Counter()
    t0, n = time.time(), len(paths)
    for i, p in enumerate(paths, 1):
        key = str(p)
        h = cache.get(key)
        if h is None:
            h = dhash(p)
            if h is not None:
                cache[key] = h
        if h is None:
            skipped[p.suffix.lower()] += 1
        else:
            out[p] = h
        if i % 200 == 0 or i == n:
            print(f"    {label}: {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    if skipped:
        print(f"    {label}: 解码失败 {sum(skipped.values())} 个 -> {dict(skipped)}")
    return out


def load_cache(p):
    c = {}
    if p.is_file():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    try:
                        c[r[0]] = int(r[1])
                    except ValueError:
                        pass
    return c


def save_cache(p, c):
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(c.items())


def best_match(h, ref):
    b1 = b2 = 65
    bp = None
    for p, rh in ref:
        d = popcount(h ^ rh)
        if d < b1:
            b2, b1, bp = b1, d, p
        elif d < b2:
            b2 = d
    return bp, b1, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--merged", default="dataset_store/merged_peninsula")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--min-margin", type=int, default=2)
    ap.add_argument("--validate", type=int, default=400)
    ap.add_argument("--out", default="recovered_farms.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    merged = root.joinpath(*re.split(r"[\\/]", args.merged))
    meta_p = root / args.meta
    for p, name in ((meta_p, "metadata.csv"), (merged, "merged 目录")):
        if not p.exists():
            print(f"找不到 {name}: {p}")
            return 1

    print("=" * 76)
    if HEIC_OK:
        print("HEIC 支持: 已启用 (pillow-heif)")
    else:
        print("HEIC 支持: ★ 未安装。原图多为 .heic，强烈建议先:")
        print("             pip install pillow-heif")

    meta = {}
    with open(meta_p, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            s = norm_stem(r.get("stem") or r.get("file") or "")
            if s:
                meta.setdefault(s, r)

    src = merged / "images" if (merged / "images").is_dir() else merged
    merged_imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]

    print("\n扫描项目树建立原图索引 ...")
    idx, ext_all, ext_kept = index_root(root, merged)
    print(f"  全树图片扩展名分布 : {dict(ext_all.most_common())}")
    print(f"  排除派生副本后     : {dict(ext_kept.most_common())}")
    n_heic = ext_all[".heic"] + ext_all[".heif"]
    if not HEIC_OK and n_heic:
        print(f"  ★ 其中 {n_heic} 个 HEIC 无法解码，装 pillow-heif 后重跑")

    orig_paths, seen = [], set()
    for plist in idx.values():
        for p in plist:
            if p not in seen:
                seen.add(p)
                orig_paths.append(p)
    print(f"  候选原图 : {len(orig_paths)}   （metadata 有 {len(meta)} 行）")
    print(f"  其中能对上 metadata 的 stem : {sum(1 for s in idx if s in meta)}")

    known = [p for p in merged_imgs if norm_stem(p.name) in meta]
    unknown = [p for p in merged_imgs if norm_stem(p.name) not in meta]
    print(f"\n标注集 {len(merged_imgs)} 张 | 文件名已能对上 {len(known)} | "
          f"需哈希找回 {len(unknown)}")
    print("=" * 76)

    cache_p = root / ".phash_cache.csv"
    cache = load_cache(cache_p)
    print(f"\n计算哈希（缓存 {len(cache)} 条）")
    H_orig = hash_all(orig_paths, "原图", cache)
    H_merged = hash_all(merged_imgs, "标注集", cache)
    save_cache(cache_p, cache)

    ref = list(H_orig.items())
    orig_stems = defaultdict(list)
    for p in H_orig:
        orig_stems[norm_stem(p.name)].append(p)
    print(f"\n可比对的原图哈希 : {len(ref)}")
    if len(ref) < 500:
        print("  ★ 数量偏少。若上面显示大量 HEIC 未解码，先装 pillow-heif。")

    # ---------- 正确性检验 ----------
    print("\n" + "-" * 76)
    print("[1] 正确性检验 —— 在文件名已知的配对上验证")
    print("-" * 76)
    have_gt = [p for p in known if norm_stem(p.name) in orig_stems]
    print(f"    可用于检验的配对: {len(have_gt)} / {len(known)}")
    if len(have_gt) < 30:
        print("    ★ 样本太少，检验结果不可信。多半是原图仍未被正确读入。")
    ok = wrong = 0
    dists = []
    for i, p in enumerate(have_gt[:args.validate], 1):
        h = H_merged.get(p)
        if h is None:
            continue
        bp, d1, d2 = best_match(h, ref)
        dists.append(d1)
        if bp is not None and norm_stem(bp.name) == norm_stem(p.name):
            ok += 1
        else:
            wrong += 1
        if i % 100 == 0:
            print(f"    检验中 {i}/{min(len(have_gt), args.validate)}", flush=True)
    tot = ok + wrong
    if tot:
        dists.sort()
        acc = 100 * ok / tot
        print(f"\n    检验 {tot} 对: 正确 {ok}  错误 {wrong}   准确率 {acc:.1f}%")
        print(f"    汉明距离: 中位 {dists[len(dists)//2]}, "
              f"p95 {dists[int(len(dists)*0.95)]}, 最大 {dists[-1]}")
        if acc < 95:
            print("\n    ★ 准确率不足 95%，下面的结果不可信。")
            print("      可能标注集经过裁剪或增强，不只是缩放。")
    else:
        print("    !! 无法检验。")

    # ---------- 匹配 ----------
    print("\n" + "-" * 76)
    print(f"[2] 匹配 {len(unknown)} 张被重命名的图")
    print("-" * 76)
    rows, cnt = [], Counter()
    for i, p in enumerate(unknown, 1):
        h = H_merged.get(p)
        if h is None:
            continue
        bp, d1, d2 = best_match(h, ref)
        margin = d2 - d1
        status = ("too_far" if d1 > args.max_dist
                  else "ambiguous" if margin < args.min_margin else "ok")
        cnt[status] += 1
        m = meta.get(norm_stem(bp.name)) if bp else None
        rows.append({"merged_stem": norm_stem(p.name),
                     "matched_original": bp.name if bp else "",
                     "hamming": d1, "runner_up": d2, "margin": margin,
                     "status": status,
                     "farm": (m or {}).get("farm", ""),
                     "farm_source": (m or {}).get("farm_source", ""),
                     "FocalLength": (m or {}).get("FocalLength", ""),
                     "Model": (m or {}).get("Model", ""),
                     "DateTimeOriginal": (m or {}).get("DateTimeOriginal", "")})
        if i % 100 == 0:
            print(f"    匹配中 {i}/{len(unknown)}", flush=True)
    print(f"\n    接受 {cnt['ok']} | 歧义 {cnt['ambiguous']} | 距离过大 {cnt['too_far']}")
    if rows and cnt["too_far"] > len(unknown) * 0.5:
        d = sorted(r["hamming"] for r in rows)
        print(f"    最优距离分布: 中位 {d[len(d)//2]}, p25 {d[len(d)//4]}, 最小 {d[0]}")
        print("    若中位数很大(>16)，说明这些图的原图根本不在项目树里。")

    # ---------- farm 分布 ----------
    print("\n" + "-" * 76)
    print("[3] ★ 恢复出来的 farm 分布")
    print("-" * 76)
    c = Counter(r["farm"] or "<原图也无归属>" for r in rows if r["status"] == "ok")
    old = {"0": 165, "2": 154, "3": 59, "5": 141, "6": 39, "7": 1, "8": 1}
    print(f"    {'farm':<16}{'新增':>8}{'原有':>8}{'合计':>8}   说明")
    total_new = 0
    for f, n in sorted(c.items(), key=lambda kv: -kv[1]):
        o = old.get(f, 0)
        note = ("论文已用" if f in old
                else "★ 论文未用的新农场" if f.isdigit() else "")
        print(f"    {f:<16}{n:>8}{o:>8}{n+o:>8}   {note}")
        if f.isdigit():
            total_new += n
    print(f"\n    分析池: 560 -> {560 + total_new}")
    allf = {f for f in c if f.isdigit()} | set(old)
    usable = {f: c.get(f, 0) + old.get(f, 0) for f in allf}
    usable = {f: v for f, v in usable.items() if v >= 20}
    new_farms = sorted(f for f in usable if old.get(f, 0) < 20)
    print(f"    >=20 张的农场数: 5 -> {len(usable)}")
    if len(usable) > 5:
        print(f"\n    ★★ n 从 5 提到 {len(usable)}，新增农场: {new_farms}")
        print("        代价: 30 个 run 全部重跑(约 190 分钟)，正文每个数字重算。")
    elif total_new:
        print("\n    未扩出新农场，但分析池变大，fold 内样本更厚，")
        print("    且 Limitations 里 46% 排除率那条可以改写。")

    if rows:
        out = root / args.out
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n逐图匹配结果: {out}")
    print("\n先看 [1] 的可用配对数和准确率，够高再看 [3]。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
