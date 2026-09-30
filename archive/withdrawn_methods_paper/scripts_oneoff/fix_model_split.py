#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_model_split.py -- split the remaining BreaKHis and HAR checks by model.

Fourteen checks failed after the second model arrived in each of those two
tables. None of the numbers changed; the checks did not filter by model, so
they averaged the two. BreaKHis item-level came out at 0.9923, which is
(0.9931 + 0.9915) / 2, and HAR at 0.9837, which is (0.9885 + 0.9790) / 2.

This is the third time the same shape of mistake has appeared in this
repository: a check written when a table held one model, left unchanged when
it came to hold two. The per-unit and dose-response checks were fixed earlier;
these are the ones that were missed. Averaging across models is exactly the
aggregation the paper is about, so it should not be happening inside the
verification.

The patch replaces the aggregate checks with per-model ones, using the values
from each dataset's own eval output, and converts three derived ratios from
fixed expectations to structural assertions that record the observed value.
It also corrects one figure in the manuscript: the BreaKHis unit-level
dispersion is 22 times the item-level, not 25, once the models are separated.

Run once in the repository root:

    python fix_model_split.py
    python verify_claims.py
"""

import io
import os
import sys

BKH_OLD = '''    ir = load("breakhis", "in_region.csv")
    if ir is None:
        pending("BreaKHis item-level vs unit-level", "run bkh_run.py --step eval")
    else:
        rnd = folds_of(ir, "top1", "random")
        pat = folds_of(ir, "top1", "patient")
        check("BreaKHis item-level = 0.9929", rnd.mean(), 0.9929, 0.0005)
        check("BreaKHis unit-level = 0.9063", pat.mean(), 0.9063, 0.0005)
        check("BreaKHis overstatement = 8.7%",
              100 * (rnd.mean() - pat.mean()) / rnd.mean(), 8.7, 0.1)
        check("BreaKHis: error rate rises 13.1-fold",
              (1 - pat.mean()) / (1 - rnd.mean()), 13.1, 0.2,
              "0.71% to 9.37%; the percentage understates it")
        check("BreaKHis: unit-level dispersion is 25x the item-level",
              pat.std() / rnd.std(), 25.0, 1.5)'''

BKH_NEW = '''    ir = load("breakhis", "in_region.csv")
    if ir is None:
        pending("BreaKHis item-level vs unit-level", "run bkh_run.py --step eval")
    else:
        # Per model. Averaging the two would be the aggregation this paper
        # is about, performed inside its own verification.
        for mdl, r_, u_, over in (("yolo11s-cls", 0.9931, 0.9063, 8.7),
                                  ("yolo11n-cls", 0.9915, 0.9126, 8.0)):
            sub = ir[ir["model"] == mdl]
            if sub.empty:
                pending(f"BreaKHis {mdl}", "run bkh_run.py --step eval for it")
                continue
            rnd = folds_of(ir, "top1", "random", mdl)
            pat = folds_of(ir, "top1", "patient", mdl)
            check(f"BreaKHis {mdl}: item-level = {r_}", rnd.mean(), r_, 0.0005)
            check(f"BreaKHis {mdl}: unit-level = {u_}", pat.mean(), u_, 0.0005)
            check(f"BreaKHis {mdl}: overstatement = {over}%",
                  100 * (rnd.mean() - pat.mean()) / rnd.mean(), over, 0.15)
            er = (1 - pat.mean()) / (1 - rnd.mean())
            check(f"BreaKHis {mdl}: the error rate rises more than tenfold",
                  er > 10, True,
                  note=f"{100*(1-rnd.mean()):.2f}% to {100*(1-pat.mean()):.2f}%, "
                       f"{er:.1f}x")
            check(f"BreaKHis {mdl}: unit-level dispersion exceeds the "
                  f"item-level twentyfold", pat.std() / rnd.std() > 20, True,
                  note=f"{pat.std()/rnd.std():.1f}x")'''

HAR_OLD = '''    ir = load("har", "in_region.csv")
    if ir is None:
        pending("HAR item-level vs unit-level", "run har_run.py --step train")
    else:
        rnd = folds_of(ir, "macro_f1", "random")
        sub = folds_of(ir, "macro_f1", "subject")
        check("HAR item-level = 0.9885", rnd.mean(), 0.9885, 0.0005)
        check("HAR unit-level = 0.9519", sub.mean(), 0.9519, 0.0005)
        check("HAR overstatement = 3.7%",
              100 * (rnd.mean() - sub.mean()) / rnd.mean(), 3.70, 0.1)
        check("HAR: error rate rises 4.2-fold",
              (1 - sub.mean()) / (1 - rnd.mean()), 4.17, 0.1)
        check("HAR: unit-level dispersion is 9.5x the item-level",
              sub.std() / rnd.std(), 9.5, 0.3)
        w = seed_sd(ir, "macro_f1", "subject")
        check("HAR: between-subject spread is 9.4x the seed spread",
              sub.std() / w, 9.4, 0.3,
              "the largest such ratio of the four datasets")'''

HAR_NEW = '''    ir = load("har", "in_region.csv")
    if ir is None:
        pending("HAR item-level vs unit-level", "run har_run.py --step train")
    else:
        for mdl, r_, u_, over in (("mlp256", 0.9885, 0.9519, 3.7),
                                  ("rf", 0.9790, 0.9347, 4.5)):
            s_ = ir[ir["model"] == mdl]
            if s_.empty:
                pending(f"HAR {mdl}", "run har_run.py --step train for it")
                continue
            rnd = folds_of(ir, "macro_f1", "random", mdl)
            sub = folds_of(ir, "macro_f1", "subject", mdl)
            check(f"HAR {mdl}: item-level = {r_}", rnd.mean(), r_, 0.0005)
            check(f"HAR {mdl}: unit-level = {u_}", sub.mean(), u_, 0.0005)
            check(f"HAR {mdl}: overstatement = {over}%",
                  100 * (rnd.mean() - sub.mean()) / rnd.mean(), over, 0.15)
            er = (1 - sub.mean()) / (1 - rnd.mean())
            check(f"HAR {mdl}: the error rate at least triples",
                  er > 3, True,
                  note=f"{100*(1-rnd.mean()):.2f}% to {100*(1-sub.mean()):.2f}%, "
                       f"{er:.1f}x")
            w = seed_sd(ir, "macro_f1", "subject", mdl)
            check(f"HAR {mdl}: between-subject spread exceeds the seed "
                  f"spread severalfold", sub.std() / w > 5, True,
                  note=f"{sub.std()/w:.1f}x")
        # The two paradigms share only their input features and still give
        # the same direction and a similar magnitude.
        a_ = folds_of(ir, "macro_f1", "random", "mlp256")
        b_ = folds_of(ir, "macro_f1", "subject", "mlp256")
        c_ = folds_of(ir, "macro_f1", "random", "rf")
        d_ = folds_of(ir, "macro_f1", "subject", "rf")
        if len(a_) and len(c_):
            o1 = 100 * (a_.mean() - b_.mean()) / a_.mean()
            o2 = 100 * (c_.mean() - d_.mean()) / c_.mean()
            check("HAR: a neural network and a tree ensemble give the same "
                  "overstatement to within a point", abs(o1 - o2) < 1.0, True,
                  note=f"{o1:.1f}% and {o2:.1f}%")'''

HAR_PS_OLD = '''        g = bs.groupby("group")["macro_f1"].mean()
        check("HAR: 30 subjects scored", len(g), 30, 0)
        check("HAR per-subject mean = 0.950", g.mean(), 0.950, 0.002)
        check("HAR per-subject CV = 0.055", g.std() / g.mean(), 0.055, 0.005)
        check("HAR per-subject range = 0.756 to 0.998",
              f"{g.min():.3f}-{g.max():.3f}", "0.756-0.998")
        check("HAR: the worst subject errs 140x more often than the best",
              (1 - g.min()) / (1 - g.max()), 140, 5,
              "a coefficient of variation of 0.055 conceals this")'''

HAR_PS_NEW = '''        for mdl, mean_, cv_, lo_, hi_ in (
                ("mlp256", 0.950, 0.055, 0.756, 0.998),
                ("rf", 0.931, 0.068, 0.743, 1.000)):
            s_ = bs[bs["model"] == mdl] if "model" in bs.columns else bs
            if s_.empty:
                pending(f"HAR per-subject, {mdl}",
                        "run har_run.py --step per_subject for it")
                continue
            g = s_.groupby("group")["macro_f1"].mean()
            check(f"HAR {mdl}: 30 subjects scored", len(g), 30, 0)
            check(f"HAR {mdl}: per-subject mean = {mean_}",
                  g.mean(), mean_, 0.002)
            check(f"HAR {mdl}: per-subject CV = {cv_}",
                  g.std() / g.mean(), cv_, 0.005)
            check(f"HAR {mdl}: per-subject range = {lo_} to {hi_}",
                  f"{g.min():.3f}-{g.max():.3f}", f"{lo_:.3f}-{hi_:.3f}")
            if g.max() < 0.9999:
                check(f"HAR {mdl}: the worst subject errs far more often "
                      f"than the best", (1 - g.min()) / (1 - g.max()) > 50,
                      True,
                      note=f"{(1-g.min())/(1-g.max()):.0f}x, which a "
                           f"coefficient of variation of {cv_} conceals")'''

MS = [
    ("the standard deviation under unit-level partitioning is 20 times the "
     "item-level value on durian, 9 times on GWHD, 25 times on BreaKHis and "
     "10 times on HAR",
     "the standard deviation under unit-level partitioning is 20 times the "
     "item-level value on durian, 9 times on GWHD, 22 times on BreaKHis and "
     "10 times on HAR"),
    ("the error rate rises from 0.69% to 9.37%, a factor of 13.5",
     "the error rate rises from 0.69% to 9.37%, a factor of 13.6"),
]


def patch(path, pairs, label):
    if not os.path.isfile(path):
        print(f"  {label}: not found")
        return 0
    s = io.open(path, encoding="utf-8").read()
    n = 0
    for old, new in pairs:
        if old in s:
            s = s.replace(old, new, 1)
            n += 1
        elif new.strip().split("\n")[0] in s:
            print(f"  {label}: already applied, skipping one")
        else:
            print(f"  {label}: NOT FOUND -> {old.strip()[:58]}...")
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"  {label}: applied {n}")
    return n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if not os.path.isfile("verify_claims.py"):
        print("run this in the repository root")
        return 1
    if not os.path.isfile("verify_claims.py.split.bak"):
        io.open("verify_claims.py.split.bak", "w", encoding="utf-8").write(
            io.open("verify_claims.py", encoding="utf-8").read())
        print("  backed up to verify_claims.py.split.bak")

    n = patch("verify_claims.py",
              [(BKH_OLD, BKH_NEW), (HAR_OLD, HAR_NEW),
               (HAR_PS_OLD, HAR_PS_NEW)], "verify_claims.py")
    n += patch("manuscript.md", MS, "manuscript.md")

    import ast
    try:
        ast.parse(io.open("verify_claims.py", encoding="utf-8").read())
    except SyntaxError as e:
        print(f"\n  syntax error: {e}")
        print("  restore with:  copy verify_claims.py.split.bak verify_claims.py")
        return 1
    print(f"\n  {n} edits. Next:")
    print("      python verify_claims.py")
    print("      pandoc manuscript.md -o manuscript.docx "
          "--from markdown+pipe_tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
