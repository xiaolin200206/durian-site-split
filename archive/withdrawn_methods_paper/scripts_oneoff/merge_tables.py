#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_tables.py -- rebuild the result tables that were overwritten across
machines.

Three tables lost rows because a step wrote its output without merging what
was already there. In every case the missing rows survive in another copy;
nothing needs re-running. This script finds the copies, tags them where a
model column is absent, merges by (model, config, seed) or (model, group,
seed), and writes the union back.

Run from the repository root, next to results_durian/ and results_har/:

    python merge_tables.py            # report only
    python merge_tables.py --apply    # write

What it fixes:

  results_durian/cross_island.csv     the server copy lost yolo11s when
                                      --append did not take effect; the local
                                      copy has it
  results_har/site_count.csv          the mlp256 rows were overwritten by the
                                      rf run; the local copy has them and
                                      predates the model column
  results_breakhis/breakhis_by_patient.csv
                                      the yolo11s-cls rows were overwritten by
                                      the yolo11n-cls run

Any table already complete is left alone.
"""

import argparse
import csv
import glob
import io
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (directory, final name, glob for candidate copies, key columns,
#  model to assume when a copy has no model column)
JOBS = [
    ("results_durian", "cross_island.csv", "cross_island*.csv",
     ("model", "config", "seed"), None),
    ("results_har", "site_count.csv", "site_count*.csv",
     ("model", "k", "draw", "seed"), "mlp256"),
    ("results_breakhis", "breakhis_by_patient.csv",
     "breakhis_by_patient*.csv", ("model", "group", "seed"), "yolo11s-cls"),
]


def read(p):
    with io.open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.DictReader(fh))


def merge(dirname, final, pattern, keys, assume, apply_):
    if not os.path.isdir(dirname):
        print(f"  {dirname}/ not present, skipping")
        return
    files = sorted(glob.glob(os.path.join(dirname, pattern)))
    if not files:
        print(f"  {dirname}/{pattern}: nothing found")
        return

    rows, seen, dup, tagged = [], set(), 0, 0
    for f in files:
        r = read(f)
        if not r:
            continue
        has_model = "model" in r[0] and any(x.get("model") for x in r)
        for x in r:
            if not x.get("model"):
                if assume:
                    x["model"] = assume
                    tagged += 1
                elif not has_model:
                    x["model"] = "(unlabelled)"
            k = tuple(str(x.get(c, "")) for c in keys)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            rows.append(x)
        print(f"    {os.path.basename(f):<38}{len(r):>5} rows"
              + ("" if has_model else f"   (no model column"
                                      + (f", tagged {assume}" if assume else "")
                                      + ")"))

    if not rows:
        print(f"  {dirname}/{final}: no rows")
        return
    c = Counter(r.get("model", "?") for r in rows)
    print(f"  -> {len(rows)} rows   {dict(c)}"
          + (f"   ({dup} duplicates dropped)" if dup else ""))
    if len(c) < 2:
        print(f"     only one model present; check whether a copy is missing")

    if not apply_:
        return
    keys_out = []
    for r in rows:
        for k in r:
            if k not in keys_out:
                keys_out.append(k)
    head = [k for k in ("model", "config", "group", "k", "draw", "seed")
            if k in keys_out]
    keys_out = head + [k for k in keys_out if k not in head]
    out = os.path.join(dirname, final)
    with io.open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys_out})
    print(f"     wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    print(f"in {here}")
    print("(dry run; add --apply to write)\n" if not a.apply else "")
    for dirname, final, pattern, keys, assume in JOBS:
        print(f"{dirname}/{final}")
        merge(dirname, final, pattern, keys, assume, a.apply)
        print()
    if a.apply:
        print("Now delete the leftover copies, then re-run verify_claims.py:")
        print("    del results_durian\\cross_island_?.csv")
        print("    del results_har\\site_count_*.csv")
        print("    del results_breakhis\\breakhis_by_patient_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
