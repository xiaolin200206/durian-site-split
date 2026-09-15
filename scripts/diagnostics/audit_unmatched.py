#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_unmatched.py -- 那 204 张进不了分析的图，到底是什么？

rejoin_by_content 的结果里:
    too_far  160   最近的原图也差太远 —— 原图不在项目树里
    no_farm   44   匹配上了，但那张原图本身没有 farm 归属

204 / 1033 = 19.7%。要回答三个问题:

  [1] 它们是不是集中在某一批？
      按文件名前缀、命名风格、标注框数分组。若集中在某次导出，
      多半是那批原图被移走或删了 —— 那么 Methods 里
      "originals are at full phone resolution and available from the
      author" 这句话需要修改。

  [2] 它们和能匹配上的图有没有系统差别？
      比较标注框数、类别构成。若 too_far 的图在类别上有偏，
      Limitations 里"排除的图不是语料的随机样本"那条要给出方向。

  [3] 距离分布长什么样？
      若 too_far 的最优距离集中在 10-16，可能只是阈值偏严，
      放宽还能捞回一些；若集中在 25 以上，原图是真的不在。

纯标准库。只读。

用法:
    python audit_unmatched.py
    python audit_unmatched.py --relax 12      # 试探放宽阈值能捞回多少
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm_stem(x):
    return Path(str(x)).stem.lower()


def name_style(s):
    """把文件名归纳成风格，用来看是不是同一批导出。"""
    if re.fullmatch(r"\d+", s):
        return "纯数字 (平台顺序号)"
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-.*", s):
        return "UUID (平台导出)"
    if re.match(r"^img_\d+", s):
        return "IMG_xxxx (相机原名)"
    if re.match(r"^f\d+_a\d+", s):
        return "f00_a0000_ (标注检查副本)"
    if re.match(r"^o\d+t\d+", s):
        return "o1t1_ (Sabah 树编号)"
    return "其他"


def prefix(s, n=4):
    m = re.match(r"^([a-z_]*)", s)
    p = m.group(1) if m else ""
    return p if p else s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--attribution", default="attribution_by_content.csv")
    ap.add_argument("--merged", default="dataset_store/merged_peninsula")
    ap.add_argument("--relax", type=int, default=None,
                    help="试探: 若把 max-dist 放宽到这个值，能多捞回多少")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ap_p = root / args.attribution
    if not ap_p.is_file():
        print(f"找不到 {ap_p}，先跑 rejoin_by_content.py")
        return 1
    merged = root.joinpath(*re.split(r"[\\/]", args.merged))

    rows = []
    with open(ap_p, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                r["dist"] = int(float(r.get("dist") or 99))
            except ValueError:
                r["dist"] = 99
            rows.append(r)
    st = Counter(r["status"] for r in rows)
    print("=" * 76)
    print(f"总计 {len(rows)} 张   状态分布 {dict(st.most_common())}")
    print("=" * 76)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]

    # 标注框数与类别
    def load_classes(stem):
        p = merged / "labels" / (stem + ".txt")
        out = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                v = line.split()
                if v:
                    try:
                        out.append(int(float(v[0])))
                    except ValueError:
                        pass
        return out

    boxes = {}
    for r in rows:
        boxes[r["stem"]] = load_classes(r["stem"])

    # ---------------- [1] 是不是集中在某一批 ----------------
    print("\n[1] 命名风格分布（看是不是同一批导出）")
    print(f"    {'风格':<26}{'可用':>8}{'不可用':>8}{'不可用占比':>12}")
    styles = sorted({name_style(r["stem"].lower()) for r in rows})
    for s in styles:
        a = sum(1 for r in ok if name_style(r["stem"].lower()) == s)
        b = sum(1 for r in bad if name_style(r["stem"].lower()) == s)
        tot = a + b
        print(f"    {s:<26}{a:>8}{b:>8}{100*b/max(tot,1):>11.1f}%")

    print("\n    不可用图的文件名前缀 Top:")
    pc = Counter(prefix(r["stem"].lower()) for r in bad)
    for p, n in pc.most_common(8):
        print(f"      {p or '<无字母前缀>':<24} {n}")

    # 数字型文件名的编号区间
    nums = sorted(int(r["stem"]) for r in bad if r["stem"].isdigit())
    if nums:
        print(f"\n    纯数字文件名的不可用图: {len(nums)} 张, "
              f"编号 {nums[0]}–{nums[-1]}")
        gaps, run_start, prev = [], nums[0], nums[0]
        for x in nums[1:]:
            if x - prev > 5:
                gaps.append((run_start, prev))
                run_start = x
            prev = x
        gaps.append((run_start, prev))
        print(f"    连续区段 {len(gaps)} 个: "
              f"{['%d-%d' % g for g in gaps[:8]]}")
        print("    若集中在少数几个连续区段，说明是整批原图缺失，不是零星丢失。")

    # ---------------- [2] 与可用图的系统差别 ----------------
    print("\n[2] 与可用图的差别")
    def stats(group, label):
        nb = [len(boxes.get(r["stem"], [])) for r in group]
        empty = sum(1 for x in nb if x == 0)
        nb_nz = [x for x in nb if x > 0]
        med = sorted(nb_nz)[len(nb_nz) // 2] if nb_nz else 0
        print(f"    {label:<10} n={len(group):<5} 空标注 {empty:<5} "
              f"框数中位 {med:<5} 框数均值 {sum(nb)/max(len(nb),1):.1f}")
    stats(ok, "可用")
    stats(bad, "不可用")

    print(f"\n    类别构成（占该组总框数的比例）")
    def cls_share(group):
        c = Counter()
        for r in group:
            for x in boxes.get(r["stem"], []):
                c[x] += 1
        t = sum(c.values()) or 1
        return {k: 100 * v / t for k, v in c.items()}, t
    a_sh, a_t = cls_share(ok)
    b_sh, b_t = cls_share(bad)
    print(f"    {'类别索引':<10}{'可用%':>9}{'不可用%':>10}{'差':>8}")
    for k in sorted(set(a_sh) | set(b_sh)):
        x, y = a_sh.get(k, 0), b_sh.get(k, 0)
        flag = "  ★" if abs(x - y) > 10 else ""
        print(f"    {k:<10}{x:>8.1f}%{y:>9.1f}%{y-x:>+8.1f}{flag}")
    print(f"    (可用组共 {a_t} 框，不可用组共 {b_t} 框)")

    # ---------------- [3] 距离分布 ----------------
    print("\n[3] 最优匹配距离分布")
    tf = [r for r in rows if r["status"] == "too_far"]
    if tf:
        d = sorted(r["dist"] for r in tf)
        print(f"    too_far {len(d)} 张: 最小 {d[0]}, p25 {d[len(d)//4]}, "
              f"中位 {d[len(d)//2]}, p75 {d[3*len(d)//4]}, 最大 {d[-1]}")
        hist = Counter((x // 4) * 4 for x in d)
        for lo in sorted(hist):
            bar = "#" * min(hist[lo], 50)
            print(f"      {lo:>2}-{lo+3:<3} {hist[lo]:>4} {bar}")
        print("""
    判读:
      距离集中在 25 以上 -> 原图确实不在项目树里，放宽阈值也没用
      距离集中在 10-16   -> 阈值偏严，可能捞得回，但要重新验证准确率""")
        if args.relax:
            more = [r for r in tf if r["dist"] <= args.relax]
            print(f"\n    放宽到 {args.relax}: 可多捞 {len(more)} 张")
            if more:
                fc = Counter(r.get("farm_old") or "?" for r in more)
                print(f"    它们的旧农场标注: {dict(fc.most_common(8))}")
                print("    ★ 放宽阈值等于降低归属可信度，会重新引入误标风险。")

    nf = [r for r in rows if r["status"] == "no_farm"]
    if nf:
        print(f"\n    no_farm {len(nf)} 张: 匹配上了原图，但那张原图无 farm 归属")
        d = sorted(r["dist"] for r in nf)
        print(f"      距离: 中位 {d[len(d)//2]}, 最大 {d[-1]}")
        mf = Counter(r.get("matched_file", "")[:20] for r in nf)
        print(f"      匹配到的原图样例: {[k for k,_ in mf.most_common(5)]}")
        print("      这些是当年 exif_audit 就没能定位的图，与本次匹配无关。")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 76)
    print("对 Methods 的影响")
    print("=" * 76)
    n_far = len(tf)
    print(f"""
  排除率 {len(bad)}/{len(rows)} = {100*len(bad)/max(len(rows),1):.1f}%  (旧版 46%)

  措辞要改: 旧版的排除理由是"无法恢复拍摄位置"。现在是两个不同的理由:
    - too_far {n_far} 张: 标注图无法匹配回任何原图
    - no_farm {len(nf)} 张: 匹配到的原图本身没有位置信息

  若 [1] 显示 too_far 集中在少数连续编号区段, 说明是整批原图不在了。
  那么 Methods 里这句需要修正:
      "the originals are at full phone resolution and are available
       from the author"
  改成如实说明哪一部分的原图仍然持有。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
