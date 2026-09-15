#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_unitcheck.py -- 把 unitcheck.py 写进论文、补充材料与 README。

论文原本只在补充材料里给了一段"应当报告哪两个数字"的建议。建议不是产出。
unitcheck.py 把这三个问题做成了任何人都能在自己数据上跑的东西，所以正文
需要一节交代它，补充材料的 checklist 需要给出用法，README 需要把它放在
"复现本文数字"之前——读者拿走的应该是尺子，不是我们的读数。

先把 unitcheck.py 放到仓库根目录，再运行：

    python patch_unitcheck.py
"""

import io
import os
import sys

MS_ANCHOR = "### What follows for reporting"

MS_NEW = """### A tool, not only a finding

The three questions this paper asks of four datasets can be asked of any
dataset that records which unit each item came from, and none of them needs a
model to be retrained. We release `unitcheck.py`, which takes a table of item
to unit assignments and, optionally, a table of per-unit scores, and reports:
how many units straddle the train/validation boundary under a given split;
the dispersion, range and error-rate ratio across units; and how many units
the observed dispersion implies are needed for a target precision. It reads
neither images nor weights, depends only on the standard library, and its
output ends with the two numbers we argue should accompany any aggregate
score.

The first question is answerable before any training happens, from the
assignment table alone. On all four datasets here it returns 100%: every farm,
session, patient and subject appears on both sides of an item-level split. A
practitioner who runs that check on their own data and sees a high number
knows the reported figure is inflated without having to measure by how much.

### What follows for reporting"""

MS_CODE_OLD = ("Every quantitative claim in this paper is recomputed from "
               "the released tables by a single script that runs on every "
               "commit;")
MS_CODE_NEW = ("The audit described above is packaged as `unitcheck.py`, "
               "which runs on any dataset with a unit identifier and depends "
               "only on the standard library. Every quantitative claim in "
               "this paper is recomputed from the released tables by a "
               "single script that runs on every commit;")

SU_OLD = "## Supplementary Note 3 | Reporting checklist"
SU_NEW = """## Supplementary Note 3 | Reporting checklist, and a tool that produces it

`unitcheck.py`, released with this paper, produces everything in this note
from two tables and no model. Minimal use:

```
python unitcheck.py --assignment items.csv --unit patient --split fold
```

Add `--scores per_unit.csv --metric accuracy` once per-unit scores exist, and
`--grouped-split patient_fold` to contrast the two partitions directly. It
reports the fraction of units straddling the boundary, the dispersion and
range across units, the error-rate ratio between best and worst, the
correlation between a unit's item count and its score, and the number of
units the observed dispersion implies for a target precision."""

RM_OLD = "## Reproduce every number"
RM_NEW = """## The tool

`unitcheck.py` runs the audit in this paper on any dataset with a unit
identifier. Standard library only, no model, no images.

```bash
# before training: does an item-level split leak?
python unitcheck.py --assignment items.csv --unit patient --split fold

# after training: what does the reported figure hide?
python unitcheck.py --assignment items.csv --unit patient --split fold \\
    --scores per_patient.csv --score-unit patient --metric accuracy \\
    --n-col n_images

# contrast the two partitions directly
python unitcheck.py --assignment items.csv --unit patient \\
    --split random_fold --grouped-split patient_fold
```

It ends with the two numbers this paper argues should accompany any aggregate
score: how many independent units contributed, and the dispersion across
withheld units. The first is answerable before a single model is trained.

On this repository's own tables:

```bash
python unitcheck.py --assignment results_durian/split_assignment.csv \\
    --unit farm --split split_random --grouped-split split_farm_fold0
```

reports that 8 of 8 farms straddle the item-level boundary and none straddles
the farm-level one.

## Reproduce every number"""


def patch(path, pairs, label):
    if not os.path.isfile(path):
        print(f"  {label}: 找不到 {path}")
        return 0
    s = io.open(path, encoding="utf-8").read()
    n = 0
    for old, new in pairs:
        if new.split("\n")[0] in s and old != new.split("\n")[0]:
            print(f"  {label}: 已改过，跳过一处")
            continue
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        else:
            print(f"  {label}: 未找到 -> {old[:55]}...")
    if n:
        io.open(path, "w", encoding="utf-8").write(s)
    print(f"  {label}: 应用 {n} 处")
    return n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if not os.path.isfile("unitcheck.py"):
        print("★ 仓库根目录下没有 unitcheck.py，请先放进来再跑本补丁。")
        return 1

    total = 0
    total += patch("manuscript.md",
                   [(MS_ANCHOR, MS_NEW), (MS_CODE_OLD, MS_CODE_NEW)],
                   "manuscript.md")
    total += patch("supplementary.md", [(SU_OLD, SU_NEW)], "supplementary.md")
    total += patch("README.md", [(RM_OLD, RM_NEW)], "README.md")

    print(f"\n合计 {total} 处。自检一下工具本身：")
    print("    python unitcheck.py --assignment results_durian/"
          "split_assignment.csv \\")
    print("        --unit farm --split split_random "
          "--grouped-split split_farm_fold0")
    print("\n应当报出 8/8 跨界与 0/8 跨界。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
