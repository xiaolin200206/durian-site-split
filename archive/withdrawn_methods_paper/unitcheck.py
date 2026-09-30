#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unitcheck.py -- audit the evaluation protocol of a bout-structured dataset.

This is the reusable form of the analysis run on four datasets in "Reported
accuracy is a property of the evaluation sample". It trains nothing and reads
neither images nor weights. Give it a table saying which unit each item came
from, and optionally a table of per-unit scores, and it answers three
questions.

  1. Does an item-level split leak?  How many units straddle the boundary.
  2. What does the reported figure hide?  Dispersion, range, error-rate ratio.
  3. Is the reported figure stable?  How many units the observed dispersion
     implies for a target precision.

Question 1 is answerable before any model is trained.

Usage 1, assignment table only, before training:

    python unitcheck.py --assignment items.csv --unit patient --split fold

Usage 2, with per-unit scores, after training:

    python unitcheck.py --assignment items.csv --unit patient --split fold \\
        --scores per_patient.csv --score-unit patient --metric accuracy

Usage 3, contrast two partitions directly:

    python unitcheck.py --assignment items.csv --unit patient \\
        --split random_fold --grouped-split patient_fold

Column names need not match these; name yours with the flags. Output is plain
text, optionally also JSON with --json. Standard library only.

Lin Ding Shan. Released with the paper. MIT licence; use it freely.
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BAR = "=" * 70


def read(path):
    if not os.path.isfile(path):
        sys.exit(f"cannot find {path}")
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def need(rows, col, path):
    if col not in rows[0]:
        sys.exit(f"{path} has no column {col!r}; "
                 f"available: {list(rows[0])[:15]}")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(v):
    return st.mean(v) if v else float("nan")


def sd(v):
    return st.stdev(v) if len(v) > 1 else float("nan")


# ------------------------------------------------------------ 1. leakage --
def leakage(rows, unit_col, split_col, out):
    print("\n" + BAR)
    print("[1] Does the split leak?")
    print(BAR)

    by_unit = defaultdict(set)
    n_items = Counter()
    for r in rows:
        u = str(r[unit_col]).strip()
        s = str(r[split_col]).strip()
        if not u or not s:
            continue
        by_unit[u].add(s)
        n_items[u] += 1
    if not by_unit:
        sys.exit(f"columns {unit_col!r} and {split_col!r} yielded no values")

    span = {u: v for u, v in by_unit.items() if len(v) > 1}
    frac = len(span) / len(by_unit)
    items_in_span = sum(n_items[u] for u in span)

    print(f"  items {len(rows)}   units {len(by_unit)}   "
          f"split values {sorted({s for v in by_unit.values() for s in v})}")
    print(f"  items per unit: min {min(n_items.values())}  "
          f"median {int(st.median(list(n_items.values())))}  "
          f"max {max(n_items.values())}")
    print()
    print(f"  units straddling the boundary: {len(span)} / {len(by_unit)} "
          f"({100*frac:.0f}%)")
    print(f"  items they contribute:         {items_in_span} / {len(rows)} "
          f"({100*items_in_span/len(rows):.0f}%)")

    if frac > 0.5:
        print("\n  This split is item-level. Items from one unit appear on")
        print("  both sides of the boundary, so a model can score well by")
        print("  recognising the unit rather than the phenomenon. All four")
        print("  datasets in the paper return 100% here.")
    elif frac > 0:
        print("\n  Most units do not straddle, but some do. Check whether")
        print("  those are duplicate items, or one acquisition split across")
        print("  two identifiers.")
    else:
        print("\n  No unit straddles the boundary. This split is unit-level.")

    out["leakage"] = {"units": len(by_unit), "items": len(rows),
                      "units_spanning": len(span),
                      "fraction_spanning": frac,
                      "items_in_spanning_units": items_in_span}
    return by_unit


# ----------------------------------------------------- 2. what is hidden --
def dispersion(rows, unit_col, metric_col, n_col, min_items, out):
    print("\n" + BAR)
    print("[2] What does the reported figure hide?")
    print(BAR)

    g, n = defaultdict(list), {}
    for r in rows:
        u = str(r[unit_col]).strip()
        v = fnum(r.get(metric_col))
        if not u or v is None:
            continue
        g[u].append(v)
        if n_col and n_col in r:
            k = fnum(r[n_col])
            if k is not None:
                n[u] = int(k)
    if not g:
        sys.exit(f"no values from {unit_col!r} or {metric_col!r} "
                 f"in the scores table")

    kept = {u: mean(v) for u, v in g.items()
            if not n or n.get(u, min_items) >= min_items}
    dropped = len(g) - len(kept)
    v = sorted(kept.values())
    if len(v) < 2:
        sys.exit("fewer than two usable units; cannot compute dispersion")

    mu, s = mean(v), sd(v)
    lo, hi = v[0], v[-1]
    print(f"  units {len(v)}" +
          (f"  ({dropped} with fewer than {min_items} items skipped)"
           if dropped else ""))
    print(f"  equal-weight mean {mu:.4f}   s.d. {s:.4f}   CV {s/mu:.3f}")
    print(f"  range {lo:.4f} to {hi:.4f}")
    if hi < 1.0 and lo < hi:
        e_lo, e_hi = 1 - hi, 1 - lo
        if e_lo > 0:
            print(f"  as error rates: best {100*e_lo:.2f}%  "
                  f"worst {100*e_hi:.2f}%  ratio {e_hi/e_lo:.0f}x")
            print("  Near the ceiling a coefficient of variation compresses")
            print("  the gap; the error rate does not.")
    q1, q3 = v[len(v)//4], v[3*len(v)//4]
    print(f"  interquartile {q1:.4f} to {q3:.4f}")

    print("\n  Five worst units:")
    for u, x in sorted(kept.items(), key=lambda t: t[1])[:5]:
        print(f"      {u:<24}{x:>9.4f}" + (f"   n={n[u]}" if u in n else ""))
    print("  Five best:")
    for u, x in sorted(kept.items(), key=lambda t: -t[1])[:5]:
        print(f"      {u:<24}{x:>9.4f}" + (f"   n={n[u]}" if u in n else ""))

    if n:
        xs = [n[u] for u in kept if u in n]
        ys = [kept[u] for u in kept if u in n]
        if len(xs) > 2:
            mx, my = mean(xs), mean(ys)
            den = math.sqrt(sum((a-mx)**2 for a in xs) *
                            sum((b-my)**2 for b in ys))
            r = (sum((a-mx)*(b-my) for a, b in zip(xs, ys)) / den
                 if den else float("nan"))
            print(f"\n  items per unit against that unit's score: r = {r:+.3f}")
            print("  Near zero means the amount of data from a unit does not")
            print("  say whether that unit will be hard. That held in all four")
            print("  datasets in the paper, as did eight capture covariates")
            print("  and the model's own confidence.")

    out["dispersion"] = {"units": len(v), "mean": mu, "sd": s, "cv": s/mu,
                         "min": lo, "max": hi, "q1": q1, "q3": q3,
                         "dropped_small": dropped}
    return kept


# ------------------------------------------------------- 3. how many now --
def sample_size(scores, out, tols=(0.10, 0.05, 0.02)):
    print("\n" + BAR)
    print("[3] How many units before the figure holds still?")
    print(BAR)

    v = list(scores.values())
    s = sd(v)
    if s != s:
        return
    print(f"  At the observed between-unit s.d. of {s:.4f} "
          f"(normal approximation, 90%):\n")
    print(f"      {'target':<14}{'units needed':>13}")
    for t in tols:
        print(f"      +/- {t:<10.2f}{math.ceil((1.645*s/t)**2):>13}")
    print(f"\n  You have {len(v)} units, which gives about "
          f"+/- {1.645*s/math.sqrt(len(v)):.3f}.")
    print("\n  This is a first-order estimate that treats units as")
    print("  exchangeable, not a guarantee. The point is to report the")
    print("  dispersion alongside the unit count rather than to adopt a")
    print("  threshold: in the paper's three dose-response experiments the")
    print("  draw-to-draw spread fell below training noise at six farms,")
    print("  sixteen acquisition sessions and twenty-four subjects.")

    out["sample_size"] = {"sd_between_units": s, "units_available": len(v),
                          "precision_now": 1.645*s/math.sqrt(len(v)),
                          "needed": {str(t): math.ceil((1.645*s/t)**2)
                                     for t in tols}}


# --------------------------------------------------- 4. two split regimes --
def compare(rows, unit_col, split_a, split_b, out):
    print("\n" + BAR)
    print("[4] The two partitions side by side")
    print(BAR)
    res = {}
    for col, name in ((split_a, "item-level"), (split_b, "unit-level")):
        by_unit = defaultdict(set)
        for r in rows:
            u, s_ = str(r[unit_col]).strip(), str(r.get(col, "")).strip()
            if u and s_:
                by_unit[u].add(s_)
        span = sum(1 for v in by_unit.values() if len(v) > 1)
        res[col] = span / len(by_unit) if by_unit else float("nan")
        print(f"  {name:<12} column {col:<22} straddling "
              f"{span}/{len(by_unit)} ({100*res[col]:.0f}%)")
    print("\n  If the first row is near 100% and the second is zero, the two")
    print("  partitions are not measuring the same thing. Across the paper's")
    print("  four datasets, moving to a unit-level partition lowered the")
    print("  reported score by between 3.7% and 44.3%.")
    out["split_comparison"] = res


def main():
    ap = argparse.ArgumentParser(
        description="Audit the evaluation protocol of a bout-structured "
                    "dataset. No model, no images.")
    ap.add_argument("--assignment", required=True,
                    help="one row per item, with a unit column and a "
                         "split column")
    ap.add_argument("--unit", required=True,
                    help="unit column, e.g. patient, subject, site, session")
    ap.add_argument("--split", help="split column, e.g. fold or random_fold")
    ap.add_argument("--grouped-split",
                    help="a unit-level split column, for the contrast")
    ap.add_argument("--scores", help="table of per-unit scores")
    ap.add_argument("--score-unit",
                    help="unit column in the scores table; defaults to --unit")
    ap.add_argument("--metric", default="score", help="score column")
    ap.add_argument("--n-col",
                    help="item-count column in the scores table, used to "
                         "skip units with too few items")
    ap.add_argument("--min-items", type=int, default=5)
    ap.add_argument("--json", help="also write results as JSON")
    a = ap.parse_args()

    out = {"assignment": a.assignment, "unit_column": a.unit}
    rows = read(a.assignment)
    need(rows, a.unit, a.assignment)

    print(BAR)
    print("unitcheck  |  evaluation audit for bout-structured data")
    print(BAR)
    print(f"  assignment {a.assignment}   {len(rows)} rows")

    if a.split:
        need(rows, a.split, a.assignment)
        leakage(rows, a.unit, a.split, out)
    if a.split and a.grouped_split:
        need(rows, a.grouped_split, a.assignment)
        compare(rows, a.unit, a.split, a.grouped_split, out)

    if a.scores:
        srows = read(a.scores)
        ucol = a.score_unit or a.unit
        need(srows, ucol, a.scores)
        need(srows, a.metric, a.scores)
        sc = dispersion(srows, ucol, a.metric, a.n_col, a.min_items, out)
        sample_size(sc, out)
    else:
        print("\n  (No --scores given, so sections 2 and 3 are skipped.")
        print("   Run again once per-unit scores exist.)")

    print("\n" + BAR)
    print("The two numbers to report alongside any aggregate score")
    print(BAR)
    if "dispersion" in out:
        d = out["dispersion"]
        print(f"  independent units evaluated:  {d['units']}")
        print(f"  dispersion across them:       s.d. {d['sd']:.4f}, "
              f"range {d['min']:.3f} to {d['max']:.3f}")
    else:
        print("  independent units:            see above")
        print("  dispersion across them:       needs per-unit scores")
    print("\n  Both are free to obtain from any dataset that records where")
    print("  its items came from.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False, default=float)
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
