#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_conditions_v2.py -- 重算 farm_capture_conditions，八个农场，图片级焦距。

两处变更:

  1. 农场归属改读 attribution_by_content.csv 的 farm_new。
  2. 焦距不再只报中位数。旧表的 focal_median_mm 让正文写出了
     "farm 6 用 2.22mm，其余农场都用 6.76mm"，而实测 2.22mm 出现在
     全部农场，farm 3 的中位数是 6.76 只因为 30 比 29 多一张。
     本版输出每个农场的完整焦距构成与占比，verify_focal_block 据此
     锁住图片级事实，正文再也无法退回中位数的说法。

纯标准库。只读。在本机跑。

用法:
    python capture_conditions_v2.py
"""

import argparse
import csv
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm_stem(x):
    return Path(str(x)).stem.lower()


def norm_farm(x):
    x = (x or "").strip()
    if not x:
        return ""
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--attribution", default="attribution_by_content.csv")
    ap.add_argument("--out", default="results_v2/farm_capture_conditions.csv")
    ap.add_argument("--out-focal",
                    default="results_v2/farm_focal_composition.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    meta_p, attr_p = root / args.meta, root / args.attribution
    for p, w in ((meta_p, "metadata.csv"), (attr_p, "attribution csv")):
        if not p.is_file():
            sys.exit(f"找不到 {w}: {p}")

    meta = {}
    with open(meta_p, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            s = norm_stem(r.get("stem") or r.get("file") or "")
            if s:
                meta.setdefault(s, r)

    farm_of, matched_of = {}, {}
    with open(attr_p, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() == "ok":
                fa = norm_farm(r.get("farm_new"))
                if fa:
                    s = norm_stem(r.get("stem"))
                    farm_of[s] = fa
                    matched_of[s] = norm_stem(r.get("matched_file") or "")

    # 标注图的文件名可能是平台顺序号（00006 这种），metadata 里没有这个 stem。
    # 它的 EXIF 要经由 matched_file 指向的原图去取 —— 那才是有元数据的记录。
    rows = defaultdict(list)
    via_self = via_matched = miss = 0
    for s, fa in farm_of.items():
        m = meta.get(s)
        if m is not None:
            via_self += 1
        else:
            m = meta.get(matched_of.get(s, ""))
            if m is not None:
                via_matched += 1
        if m is None:
            miss += 1
            continue
        rows[fa].append(m)
    print(f"元数据来源: 同名 {via_self} | 经 matched_file {via_matched} "
          f"| 找不到 {miss}")
    farms = sorted(rows, key=lambda x: (len(x), x))
    print(f"农场 {len(farms)} 个，共 {sum(len(v) for v in rows.values())} 张"
          f"   (应为 8 个 / 827 张)\n")

    # ---------- 主表 ----------
    out, focal_rows = [], []
    for fa in farms:
        rs = rows[fa]
        n = len(rs)
        hours = [fnum(r.get("hour")) for r in rs]
        hours = [h for h in hours if h is not None]
        solar = [fnum(r.get("solar_elev_deg")) for r in rs]
        solar = [x for x in solar if x is not None]
        focs = [fnum(r.get("FocalLength")) for r in rs]
        focs_ok = [round(x, 2) for x in focs if x is not None]
        n_nofocal = sum(1 for x in focs if x is None)
        models = Counter((r.get("Model") or "<空>").strip() for r in rs)
        dates = Counter((r.get("date") or "").strip() for r in rs if r.get("date"))
        srcs = Counter((r.get("farm_source") or "<空>").strip() for r in rs)
        gps = sum(1 for r in rs if (r.get("lat") or "").strip())

        fc = Counter(focs_ok)
        dom, dom_n = (fc.most_common(1)[0] if fc else ("", 0))
        out.append({
            "farm": fa, "n_images": n,
            "hour_median": round(st.median(hours), 3) if hours else "",
            "midday_share": (round(sum(1 for h in hours if 10 <= h < 15)
                                   / len(hours), 4) if hours else ""),
            "solar_median": round(st.median(solar), 2) if solar else "",
            "focal_median_mm": round(st.median(focs_ok), 2) if focs_ok else "",
            "focal_dominant_mm": dom,
            "focal_dominant_share": round(dom_n / n, 4) if n else "",
            "focal_n_distinct": len(fc),
            "focal_missing": n_nofocal,
            "focal_mixed": len(fc) > 1,
            "devices": ";".join(sorted(models)),
            "n_devices": len(models),
            "dates": len(dates),
            "gps_share": round(gps / n, 4) if n else "",
            "farm_source": ";".join(f"{k}:{v}" for k, v in srcs.most_common()),
        })
        for fv, k in sorted(fc.items()):
            focal_rows.append({"farm": fa, "focal_mm": fv, "n_images": k,
                               "share": round(k / n, 4)})
        if n_nofocal:
            focal_rows.append({"farm": fa, "focal_mm": "MISSING",
                               "n_images": n_nofocal,
                               "share": round(n_nofocal / n, 4)})

    # ---------- 打印 ----------
    print(f"{'farm':<6}{'n':>5}{'中位焦距':>9}{'主焦距':>8}{'占比':>7}"
          f"{'种类':>5}{'缺失':>5}{'设备数':>6}{'GPS':>7}")
    for r in out:
        print(f"{r['farm']:<6}{r['n_images']:>5}"
              f"{str(r['focal_median_mm']):>9}{str(r['focal_dominant_mm']):>8}"
              f"{r['focal_dominant_share']:>7}"
              f"{r['focal_n_distinct']:>5}{r['focal_missing']:>5}"
              f"{r['n_devices']:>6}{r['gps_share']:>7}")

    mixed = [r["farm"] for r in out if r["focal_mixed"]]
    nofoc = [r["farm"] for r in out if r["focal_missing"] == r["n_images"]]
    print(f"\n焦距混合的农场: {mixed}")
    if nofoc:
        print(f"★ 完全没有焦距记录的农场: {nofoc}")
        print("  这些农场进不了任何协变量分析，正文须单列说明。")

    print(f"\n完整焦距构成:")
    print(f"  {'farm':<6}{'焦距':>10}{'n':>6}{'占比':>8}")
    for r in focal_rows:
        print(f"  {r['farm']:<6}{str(r['focal_mm']):>10}"
              f"{r['n_images']:>6}{r['share']:>8.2%}")

    wide = defaultdict(int)
    for r in focal_rows:
        if r["focal_mm"] == 2.22:
            wide[r["farm"]] = r["n_images"]
    if len(wide) > 1:
        print(f"\n★ 2.22mm 出现在 {len(wide)} 个农场: {dict(sorted(wide.items()))}")
        print("  正文不得再称其为 farm 6 独有。")

    for path, data in ((args.out, out), (args.out_focal, focal_rows)):
        p = root.joinpath(*re.split(r"[\\/]", path))
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\n写出 {p}  ({len(data)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
