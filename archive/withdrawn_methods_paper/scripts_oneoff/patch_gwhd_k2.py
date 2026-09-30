#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_gwhd_k2.py -- 一次性修正：GWHD 的 k=2 只有一个抽样。

放在仓库根目录（与 verify_claims.py 同级）运行一次：

    python patch_gwhd_k2.py

做两件事。

[1] verify_claims.py
    GWHD 的剂量曲线里，k=2 只有一个 draw 存活（其余 k 有 3 到 8 个）。
    单个 draw 的"抽样间标准差"是未定义的，pandas 返回 NaN，断言因此失败。
    改成先丢掉 NaN，从第一个有两个以上抽样的 k 读起。

[2] manuscript.md 与 supplementary.md
    正文原本把 GWHD 的塌陷写成 0.032 -> 0.008，起点取的正是 k=2 那个值。
    但那个 0.0319 是两个种子之间的标准差，不是抽样之间的标准差，两者不是
    同一个量。曲线的起点改为 k=4（0.0366），并说明 k=2 不参与。

跑完请重跑 verify_claims.py 与 make_figures.py。
"""

import io
import os
import sys

VC_OLD = '''        draw = sc.groupby(["k", "config"])["mAP50"].mean().groupby("k").std()
        ks = sorted(draw.index)
        check("GWHD: draw spread falls from 0.032 to 0.008",
              draw[ks[0]] > 0.030 and draw[ks[-1]] < 0.010, True,
              note=f"{draw[ks[0]]:.4f} -> {draw[ks[-1]]:.4f}")'''

VC_NEW = '''        draw = sc.groupby(["k", "config"])["mAP50"].mean().groupby("k").std()
        # k=2 survives with a single draw, so its across-draw spread is
        # undefined. The curve is read from the first k with two or more
        # draws; the single k=2 point is not an estimate of dispersion.
        n_draws = sc.groupby("k")["config"].nunique()
        check("GWHD: k=2 has a single surviving draw",
              int(n_draws.min()), 1, 0,
              note="its spread is undefined and excluded from the curve")
        draw = draw.dropna()
        ks = sorted(draw.index)
        check("GWHD: draw spread falls from 0.037 to 0.008",
              draw[ks[0]] > 0.030 and draw[ks[-1]] < 0.010, True,
              note=f"k={ks[0]} {draw[ks[0]]:.4f} -> k={ks[-1]} "
                   f"{draw[ks[-1]]:.4f}")'''

MS = [
    ("The spread across which units were drawn falls in all three, by "
     "factors of 3.8, 4.0 and 6.4.",
     "The spread across which units were drawn falls in all three, by "
     "factors of 3.8, 4.8 and 6.4. The GWHD factor is measured from k = 4, "
     "the smallest unit count with more than one surviving draw; at k = 2 "
     "only one draw could reach the item budget, so no across-draw spread "
     "is defined there."),
]

SU = [
    ("Draws that could not reach the item budget were skipped, which is why "
     "durian starts with four draws at k = 1 and GWHD has one at k = 2; the "
     "GWHD k = 2 point should not be read as an estimate.",
     "Draws that could not reach the item budget were skipped, which is why "
     "durian starts with four draws at k = 1 and GWHD has one at k = 2. "
     "With a single draw there is no across-draw spread to compute, so the "
     "GWHD k = 2 row reports a mean only and the collapse is measured from "
     "k = 4 onwards."),
    ("| 2 | 1 | 2 | 0.304 | 0.0319 |",
     "| 2 | 1 | 2 | 0.304 | – |"),
]


def patch(path, pairs, label):
    if not os.path.isfile(path):
        print(f"  {label}: 找不到 {path}，跳过")
        return 0
    s = io.open(path, encoding="utf-8").read()
    n = 0
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new)
            n += 1
        elif new in s:
            print(f"  {label}: 已经改过，跳过一处")
        else:
            print(f"  {label}: 未找到 -> {old[:60]}...")
    if n:
        io.open(path, "w", encoding="utf-8").write(s)
    print(f"  {label}: 应用 {n} 处")
    return n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print("在", here)

    total = 0
    total += patch("verify_claims.py", [(VC_OLD, VC_NEW)], "verify_claims.py")
    total += patch("manuscript.md", MS, "manuscript.md")
    total += patch("supplementary.md", SU, "supplementary.md")

    print(f"\n合计 {total} 处。接着跑：")
    print("    python verify_claims.py")
    print("    python scripts/analyse/make_figures.py --out figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
