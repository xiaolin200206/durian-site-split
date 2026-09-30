#!/usr/bin/env python3
"""
verify_applied.py -- check the applied manuscript against the released data.

Three layers:
  A. independent recomputation: key numbers are recomputed from the raw per-run
     CSVs with code that shares nothing with scripts/applied/paper_analysis.py,
     and compared with results_applied/paper_numbers.json
  B. the manuscript is exactly what paper/render.py produces from the template
     and the JSON (no hand edits, no unresolved placeholders)
  C. submission rules and internal consistency: abstract length, highlights,
     keywords, citations <-> reference list, figure and table numbering

Usage:  python verify_applied.py [--full] [--verbose]
        --full first re-runs scripts/applied/paper_analysis.py (about a minute)
        and checks that it reproduces results_applied/paper_numbers.json exactly.
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_applied")
PAPER = os.path.join(ROOT, "paper")
V = "--verbose" in sys.argv
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    if V or not ok:
        print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v)


N = json.load(open(os.path.join(RES, "paper_numbers.json")))
if "--full" in sys.argv:
    print("0. re-deriving paper_numbers.json from the raw tables")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "applied", "paper_analysis.py"),
                    "--root", ROOT], check=True, stdout=subprocess.DEVNULL)
    N2 = json.load(open(os.path.join(RES, "paper_numbers.json")))
    check("paper_numbers.json is reproduced exactly by paper_analysis.py", N2 == N)
    N = N2

# ------------------------------------------------------------ A. recompute --
print("A. independent recomputation")
man = rows(os.path.join(ROOT, "results_durian", "split_assignment.csv"))
farms = sorted({r["farm"] for r in man}, key=int)
check("8 farms, 827 images", len(farms) == 8 and len(man) == 827, f"{len(farms)}, {len(man)}")
check("JSON farm and image counts", N["farms"]["n_farms"] == len(farms) and
      N["farms"]["images"] == len(man))
check("JSON box count", N["farms"]["boxes"] == sum(int(r["n_boxes"]) for r in man))
cls = {r["stem"]: {int(c) for c in (r["classes"] or "").replace(";", ",").split(",") if c.strip()}
       for r in man}
farm_of = {r["stem"]: r["farm"] for r in man}
farm_cls = defaultdict(set)
for s, c in cls.items():
    farm_cls[farm_of[s]] |= c
check("Phomopsis on 3 farms", sum(2 in farm_cls[f] for f in farms) == 3 ==
      N["farms"]["farms_per_class"]["Phomopsis"])

clean = rows(os.path.join(ROOT, "results_clean", "durian_in_region_clean.csv")) + \
    rows(os.path.join(ROOT, "results_clean", "durian_in_region_clean_yolo11l.csv"))
frc = rows(os.path.join(ROOT, "results_clean", "durian_in_region_clean_frcnn.csv"))
allr = [r for r in clean + frc if r["checkpoint"] == "best" and r["eval_on"] == "outer"]


def headline(model):
    rr = [r for r in allr if r["model"] == model]
    rnd = mean([float(r["mAP50"]) for r in rr if r["config"] == "random"])
    per = [mean([float(r["mAP50"]) for r in rr if r["config"] == f"byfarm_fold{f}"]) for f in farms]
    return rnd, mean(per), per


per_farm_all = defaultdict(list)
for m in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "rtdetr-l", "frcnn-r50"]:
    rnd, un, per = headline(m)
    h = N["headline"][m]
    check(f"{m}: random split {rnd:.4f}", close(rnd, h["random_split"], 6e-4), h["random_split"])
    check(f"{m}: unseen farm {un:.4f}", close(un, h["unseen_farm"], 6e-4), h["unseen_farm"])
    check(f"{m}: loss {100 * (1 - un / rnd):.1f}%", close(100 * (1 - un / rnd), h["drop_pct"], 0.15))
    for f, v in zip(farms, per):
        per_farm_all[f].append(v)
pf = {f: mean(v) for f, v in per_farm_all.items()}
check("per-farm range", close(min(pf.values()), N["per_farm"]["min"], 6e-4) and
      close(max(pf.values()), N["per_farm"]["max"], 6e-4),
      f"{min(pf.values()):.3f}-{max(pf.values()):.3f}")
worst = sorted(pf, key=pf.get)[:2]


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


per_model = {}
for m in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "rtdetr-l", "frcnn-r50"]:
    per_model[m] = headline(m)[2]
ms_ = list(per_model)
rhos = [spearman(per_model[a], per_model[b]) for i, a in enumerate(ms_) for b in ms_[i + 1:]]
check(f"mean pairwise Spearman {mean(rhos):.3f}, min {min(rhos):.3f}",
      close(mean(rhos), N["per_farm"]["mean_pairwise_spearman"], 0.006) and
      close(min(rhos), N["per_farm"]["min_pairwise_spearman"], 0.006))
check("two hardest farms", sorted(int(x) for x in worst) == sorted(N["per_farm"]["two_worst"]), worst)

sab = [r for r in clean if r["eval_on"] == "sabah" and r["checkpoint"] == "best"
       and r["config"].startswith("byfarm")]
sv = {}
for m in {r["model"] for r in sab}:
    sv[m] = mean([mean([float(r["mAP50"]) for r in sab if r["model"] == m and r["config"] == f"byfarm_fold{f}"])
                  for f in farms])
check("Sabah uses leave-one-farm-out models only", all(close(sv[m], N["sabah_clean"][m], 6e-4) for m in sv))
check("Sabah range", close(min(sv.values()), N["headline_range"]["sabah_min"], 6e-4) and
      close(max(sv.values()), N["headline_range"]["sabah_max"], 6e-4),
      f"{min(sv.values()):.3f}-{max(sv.values()):.3f}")

# per class retained, psyllid classes (five detectors with per-class AP)
names = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid", "Psyllid_damage", "leaf_hopper_damage"]
for ci, c in enumerate(names):
    carriers = [f for f in farms if ci in farm_cls[f]]
    ret = []
    for m in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "rtdetr-l"]:
        rr = [r for r in clean if r["model"] == m and r["checkpoint"] == "best" and r["eval_on"] == "outer"]
        rnd = mean([float(r[f"AP50::{c}"]) for r in rr if r["config"] == "random" and r[f"AP50::{c}"]])
        nf = mean([mean([float(r[f"AP50::{c}"]) for r in rr if r["config"] == f"byfarm_fold{f}"
                         and r[f"AP50::{c}"]]) for f in carriers])
        ret.append(100 * nf / rnd)
    check(f"{c}: retained {mean(ret):.1f}%", close(mean(ret), N["per_class"][c]["retained_pct"], 0.6))

# RQ2
fb = rows(os.path.join(RES, "farm_budget.csv"))
held = [r for r in fb if r["eval_on"] == "heldout"]
check("RQ2 run count", len({r["run"] for r in held}) == N["rq2_design"]["runs"])
cell = defaultdict(list)
for r in held:
    cell[(int(r["k"]), r["m"], r["heldout"])].append(float(r["mAP50"]))
for (k, m) in [(1, "all"), (2, "all"), (4, "all"), (7, "15"), (7, "50"), (7, "all")]:
    v = mean([mean(cell[(k, m, f)]) for f in farms])
    check(f"grid k={k} m={m}: {v:.4f}", close(v, N["grid"][f"k{k}_m{m}"]["mAP50"], 6e-4))
for key, (a, b) in {"k7m15_vs_k2m50": ((7, "15"), (2, "50")),
                    "k7m50_vs_k4mall": ((7, "50"), (4, "all")),
                    "k4m50_vs_k2mall": ((4, "50"), (2, "all"))}.items():
    d = [mean(cell[a + (f,)]) - mean(cell[b + (f,)]) for f in farms]
    wins = sum(x > 0 for x in d)
    check(f"{key}: diff {mean(d):+.4f}, wins {wins}/8",
          close(mean(d), N["equal_budget"][key]["diff"], 6e-4) and
          N["equal_budget"][key]["wins"] == f"{wins}/8")

# regression by within-farm demeaning (Frisch-Waugh), independent of the dummy-variable fit
def within(vals, groups):
    g = defaultdict(list)
    for v, k in zip(vals, groups):
        g[k].append(v)
    mu = {k: mean(v) for k, v in g.items()}
    return [v - mu[k] for v, k in zip(vals, groups)]


def slopes(sub):
    grp = [r["heldout"] for r in sub]
    y = within([float(r["mAP50"]) for r in sub], grp)
    x1 = within([math.log2(int(r["k"])) for r in sub], grp)
    x2 = within([math.log2(int(r["n_train"]) / int(r["k"])) for r in sub], grp)
    s11 = sum(a * a for a in x1); s22 = sum(b * b for b in x2); s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * c for a, c in zip(x1, y)); s2y = sum(b * c for b, c in zip(x2, y))
    det = s11 * s22 - s12 * s12
    return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


b1, b2 = slopes(held)
check(f"regression, all runs: farms {b1:.4f} photos {b2:.4f}",
      close(b1, N["regression"]["all runs"]["farms"], 6e-4) and
      close(b2, N["regression"]["all runs"]["photos"], 6e-4))


# coverage is computed from the images actually drawn for each training set
_drawn = defaultdict(set)
for _r in csv.DictReader(open(os.path.join(RES, "farm_budget_train_lists.csv"), encoding="utf-8")):
    _drawn[_r["config"]] |= cls.get(_r["stem"], set())


def coverage(r):
    seen = _drawn[r["config"]]
    imgs = [s for s in cls if farm_of[s] == r["heldout"] and cls[s]]
    return mean([1.0 if cls[s] <= seen else 0.0 for s in imgs])


full = [r for r in held if coverage(r) >= 0.999]
b1, b2 = slopes(full)
check(f"regression, full coverage ({len(full)} runs): farms {b1:.4f} photos {b2:.4f}",
      len(full) == N["regression"]["full class coverage only"]["n_runs"] and
      close(b1, N["regression"]["full class coverage only"]["farms"], 6e-4) and
      close(b2, N["regression"]["full class coverage only"]["photos"], 6e-4))
k1cov = mean([coverage(r) for r in held if r["k"] == "1"])
check(f"class coverage, one farm {k1cov:.3f}", close(k1cov, N["coverage"]["k1_mean"], 6e-3))

# ------------------------------------------------------------ B. rendering --
print("B. manuscript is the rendered template")
cur = {f: open(os.path.join(PAPER, f), encoding="utf-8").read()
       for f in ("manuscript.md", "highlights.md", "cover_letter.md")}
subprocess.run([sys.executable, os.path.join(PAPER, "render.py")], check=True,
               stdout=subprocess.DEVNULL)
for f, old in cur.items():
    new = open(os.path.join(PAPER, f), encoding="utf-8").read()
    check(f"{f} matches a fresh render", new == old)
    check(f"{f} has no unresolved placeholders", "{{" not in new and "MISSING" not in new
          and "[[TABLE" not in new)
ms = cur["manuscript.md"]

# ------------------------------------------------------------ C. rules --
print("C. submission rules and consistency")
abstract = ms.split("## Abstract", 1)[1].split("**Keywords", 1)[0]
nw = len(abstract.split())
check(f"abstract {nw} words <= 250", nw <= 250)
kw = ms.split("**Keywords:**", 1)[1].split("\n", 1)[0].split(";")
check(f"{len(kw)} keywords (1-7)", 1 <= len(kw) <= 7)
hl = [l[2:].strip() for l in cur["highlights.md"].splitlines() if l.startswith("- ")]
check(f"{len(hl)} highlights (3-5)", 3 <= len(hl) <= 5)
for h in hl:
    check(f"highlight {len(h)} chars <= 85", len(h) <= 85, h)

body, refs = ms.split("## References", 1)
body = body.split("## CRediT", 1)[0]
cited = set()
for grp in re.findall(r"\(([^()]*\d{4}[a-z]?)\)", body):
    for part in grp.split(";"):
        m = re.match(r"^\s*(?:e\.g\.\s*)?(.+?)(?: et al\.| and [^,]+)?, "
                     r"(\d{4}[a-z]?(?:, \d{4}[a-z]?)*)\s*$", part)
        if m:
            for y in m.group(2).split(", "):
                cited.add((m.group(1).strip(), y))
for m in re.finditer(r"([A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)?)(?: et al\.| and [A-Z][\w'’\-]+)? "
                     r"\((\d{4}[a-z]?(?:, \d{4}[a-z]?)*)\)", body):
    for y in m.group(2).split(", "):
        cited.add((m.group(1), y))
reflist = [l for l in refs.strip().split("\n\n") if l.strip()]
refkeys = set()
for r in reflist:
    refkeys.add((r.split(",")[0].strip(), re.search(r", (\d{4}[a-z]?)\. ", r).group(1)))
cited_norm = cited
missing = sorted(k for k in cited_norm if k not in refkeys)
unused = sorted(k for k in refkeys if k not in cited_norm)
check("every citation is in the reference list", not missing, missing)
check("every reference is cited", not unused, unused)
check("references in alphabetical order",
      [r.split(",")[0] for r in reflist] == sorted([r.split(",")[0] for r in reflist],
                                                   key=lambda s: s.lower()))

figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d)\.\*\*", ms)]
check("figures 1..n captioned in order", figs == list(range(1, len(figs) + 1)), figs)
tabs = [int(x) for x in re.findall(r"\*\*Table (\d)\.\*\*", ms)]
check("tables 1..n in order", tabs == list(range(1, len(tabs) + 1)), tabs)
first_mention = [int(x) for x in re.findall(r"Table (\d)(?!\.\*\*)", body)]
seen = []
for t in first_mention:
    if t not in seen:
        seen.append(t)
check("tables first cited in numerical order", seen == sorted(seen), seen)
fm = []
for t in re.findall(r"Fig\. (\d)[a-z]?(?!\.\*\*)", body):
    if int(t) not in fm:
        fm.append(int(t))
check("figures first cited in numerical order", fm == sorted(fm), fm)
for f in ("fig1_data", "fig2_new_farm_cost", "fig3_budget", "fig4_class_by_k"):
    check(f"{f}.png and .pdf exist", all(os.path.isfile(os.path.join(RES, f"{f}.{e}"))
                                         for e in ("png", "pdf")))
check("no drafting markers", not re.search(r"TODO|TBD|XXX|\?\?", ms))
for f in ("prose_blocks.md", "highlights_template.md", "cover_letter_template.md"):
    txt = open(os.path.join(PAPER, f), encoding="utf-8").read()
    check(f"{f} is final (no DRAFT marker)", "DRAFT" not in txt)
meta = json.load(open(os.path.join(PAPER, "metadata.json")))
check("metadata.json DOIs filled in", all("TODO" not in str(v) for v in meta.values()))
for t in range(1, 8):
    check(f"Table S{t} cited in the manuscript", f"Table S{t}" in ms)
sup = open(os.path.join(PAPER, "supplementary.md"), encoding="utf-8").read()
for t in range(1, 8):
    check(f"Table S{t} exists in the supplementary", f"## Table S{t}." in sup)

# qualitative claims written in prose_blocks.md; each must still hold for the current numbers
rq3 = N["rq3"]; rg_ = N["regression"]; st = N["steps"]; eb = N["equal_budget"]
check("claim: fine-tuning lowered mAP50 at every size",
      all(rq3["ft"][m]["gain"] < 0 for m in rq3["ft"]))
check("claim: retraining never lost accuracy", all(rq3["rt"][m]["gain"] >= 0 for m in rq3["rt"]))
check("claim: fine-tuning loss mostly on leafhopper damage and leaf rot",
      rq3["change"]["ft"]["5"]["worst"] == ["leaf_hopper_damage", "Leaf_rot"])
check("claim: fine-tuning raised precision and cut recall (5 images)",
      rq3["change"]["ft"]["5"]["precision"] > 0 > rq3["change"]["ft"]["5"]["recall"])
check("claim: retraining raised precision and recall (whole half)",
      rq3["change"]["rt"]["all"]["precision"] > 0 and rq3["change"]["rt"]["all"]["recall"] > 0)
check("claim: slope difference within uncertainty (all runs, excl. k=7) but not excl. k=1",
      all(rg_[m]["diff_ci"][0] <= 0 <= rg_[m]["diff_ci"][1] for m in ("all runs", "excluding k = 7"))
      and rg_["excluding k = 1"]["diff_ci"][0] > 0)
check("claim: every images-per-farm step positive",
      all(v["diff"] > 0 for k, v in st.items() if k.startswith("photos")))
check("claim: matched budgets, more farms won twice and lost once",
      eb["k7m15_vs_k2m50"]["diff"] > 0 and eb["k7m50_vs_k4mall"]["diff"] > 0 and
      eb["k4m50_vs_k2mall"]["diff"] < 0)
check("claim: still rising from four to seven farms", st["farms 4->7 @ all per farm"]["diff"] > 0)
v1s = N["v1_schedule"]
check("claim: same-seed schedules share the farm slope; capped photo slope lower; capped diff excludes 0",
      abs(v1s["farms"] - v1s["v2_seed42"]["farms"]) <= 0.003 and
      v1s["photos"] < v1s["v2_seed42"]["photos"] and v1s["diff_ci"][0] > 0 and
      v1s["v2_seed42"]["diff_ci"][0] <= 0)
check("claim: whole-half fine-tuning loss within uncertainty, smaller sizes not",
      rq3["ft"]["all"]["ci"][1] >= 0 and all(rq3["ft"][m]["ci"][1] < 0 for m in ("5", "10", "20")))
check("claim: every class higher at seven farms than one",
      all(v["7"] > v["1"] for v in N["class_by_k"].values()))

# README quotes a few headline numbers; they must match the JSON
rd_ = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
h = N["headline"]["yolo11n"]; hr = N["headline_range"]; rg = N["regression"]
want = [f"{h['random_split']:.3f} → {h['unseen_farm']:.3f} mAP50 (−{h['drop_pct']:.1f}%)",
        f"−{hr['drop_min']:.1f}% to −{hr['drop_max']:.1f}%",
        f"{N['per_farm']['min']:.3f} – {N['per_farm']['max']:.3f}",
        f"{N['model_pairs']['max_abs_mean_diff']:.3f} mAP50",
        f"psyllid classes {N['per_class_summary']['psyllid_retained_range'][0]:.0f}–"
        f"{N['per_class_summary']['psyllid_retained_range'][1]:.0f}%",
        f"+{rg['all runs']['farms']:.3f} vs +{rg['all runs']['photos']:.3f}",
        f"+{rg['full class coverage only']['farms']:.3f} vs +{rg['full class coverage only']['photos']:.3f}",
        f"{N['equal_budget']['k7m50_vs_k4mall']['mAP'][0]:.3f} vs {N['equal_budget']['k7m50_vs_k4mall']['mAP'][1]:.3f}",
        f"seed-{N['seed_list'].replace(' and ', '/')} data-budget run"]
for w_ in want:
    check(f"README states '{w_}'", w_ in rd_)

n_ok = sum(ok for _, ok, _ in results)
print("-" * 60)
print(f"{n_ok} of {len(results)} checks pass")
sys.exit(0 if n_ok == len(results) else 1)
