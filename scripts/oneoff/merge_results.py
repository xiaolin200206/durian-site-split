#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_results.py -- merge result tables that were overwritten instead of appended.

Three tables lost rows because a step wrote its output without keeping what
was already there:

    results_durian/cross_island.csv        the yolo11s block was replaced
    results_har/site_count.csv             the mlp256 block was replaced
    results_breakhis/breakhis_by_patient.csv   the yolo11s-cls block was replaced

In each case the missing rows still exist in a copy, so nothing has to be
re-run. This script finds the pieces, tags any that predate the `model`
column, merges on (model, config-or-group, seed) and writes the combined
table back. It is safe to run more than once.

    python merge_results.py            # report what it would do
    python merge_results.py --apply    # write the merged tables

Run it from the repository root, next to verify_claims.py.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# directory, output file, glob for the pieces, model name to assume for
# rows that have no model column, and the columns that identify a row
JOBS = [
    ("results_durian", "cross_island.csv", "cross_island*.csv",
     "yolo11s", ("model", "config", "seed")),
    ("results_har", "site_count.csv", "site_count*.csv",
     "mlp256", ("model", "k", "draw", "seed")),
    ("results_breakhis", "breakhis_by_patient.csv",
     "breakhis_by_patient*.csv", "yolo11s-cls",
     ("model", "group", "seed")),
]


def read(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.DictReader(fh))


def key_of(row, keys):
    return tuple(str(row.get(k, "")) for k in keys)


def do_job(d, out_name, pattern, default_model, keys, apply):
    out = os.path.join(d, out_name)
    if not os.path.isdir(d):
        print(f"  {d}/ not found, skipping")
        return
    pieces = sorted(glob.glob(os.path.join(d, pattern)))
    # a merged output from a previous run is itself a valid piece
    if not pieces:
        print(f"  {d}/{pattern}: nothing found")
        return

    print(f"\n  {d}/{out_name}")
    rows, seen, dup, tagged = [], set(), 0, 0
    for p in pieces:
        r = read(p)
        models = Counter(x.get("model") or "(none)" for x in r)
        print(f"      {os.path.basename(p):<38} {len(r):>5} rows  "
              f"{dict(models)}")
        for x in r:
            if not x.get("model"):
                x["model"] = default_model
                tagged += 1
            k = key_of(x, keys)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            rows.append(x)

    if not rows:
        print("      nothing to merge")
        return
    after = Counter(x["model"] for x in rows)
    print(f"      -> {len(rows)} rows  {dict(after)}"
          + (f"   ({tagged} tagged {default_model}, {dup} duplicates dropped)"
             if tagged or dup else ""))
    if len(after) < 2:
        print(f"      NOTE only one model present; the other piece may be "
              f"missing")

    if not apply:
        return
    if os.path.isfile(out):
        bak = out.replace(".csv", "_backup.csv")
        if not os.path.isfile(bak):
            shutil.copy2(out, bak)
    cols = []
    for x in rows:
        for c in x:
            if c not in cols:
                cols.append(c)
    head = [c for c in ("model", "config", "group", "k", "draw", "seed")
            if c in cols]
    cols = head + [c for c in cols if c not in head]
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow({c: x.get(c, "") for c in cols})
    print(f"      wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print(f"in {here}")
    print("(dry run; add --apply to write)" if not a.apply else "(writing)")

    for job in JOBS:
        do_job(*job, apply=a.apply)

    if a.apply:
        print("""
Done. Delete the leftover pieces once the merged tables look right:

    del results_durian\\cross_island_A.csv results_durian\\cross_island_*backup*.csv
    del results_har\\site_count_mlp.csv results_har\\site_count_rf.csv
    del results_breakhis\\breakhis_by_patient_*.csv

Then run:  python verify_claims.py""")
    else:
        print("\nAdd --apply to write the merged tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
