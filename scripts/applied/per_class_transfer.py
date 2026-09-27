#!/usr/bin/env python3
"""
per_class_transfer.py -- which diseases and pests survive the move to a new farm?

No training. Reads the clean-protocol leave-one-farm-out results that already
exist (results_clean/durian_in_region_clean*.csv) and the farm manifest
(results_durian/split_assignment.csv), and reports, per class:

  * random-split AP      what an 80/20 split of the pooled images reports
  * new-farm AP          AP on a farm the model never saw, averaged over the
                         farms where that class actually occurs
  * how many farms carry the class, and how many of them the model trained on

Class ids follow docs/dataset_README.md:
  0 Algal  1 Leaf_rot  2 Phomopsis  3 Psyllid  4 Psyllid_damage  5 leaf_hopper_damage

Outputs (results_applied/):
  per_class_transfer.csv        one row per class x model
  per_class_by_farm.csv         one row per class x held-out farm x model
Usage:
  python scripts/applied/per_class_transfer.py --root .
"""
import argparse, csv, os, statistics as st
from collections import defaultdict

NAMES = ["Algal", "Leaf_rot", "Phomopsis", "Psyllid", "Psyllid_damage",
         "leaf_hopper_damage"]


def read(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def farm_presence(root):
    """farm -> class -> number of images containing that class."""
    rows = read(os.path.join(root, "results_durian", "split_assignment.csv"))
    pres = defaultdict(lambda: defaultdict(int))
    size = defaultdict(int)
    for r in rows:
        size[r["farm"]] += 1
        for c in (r["classes"] or "").replace(";", ",").split(","):
            c = c.strip()
            if c:
                pres[r["farm"]][NAMES[int(c)]] += 1
    return pres, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--models", nargs="+",
                    default=["yolo11n", "yolo11s", "yolo11m", "yolo11l",
                             "rtdetr-l"])
    a = ap.parse_args()
    R = os.path.abspath(a.root)
    rows = []
    for f in ("durian_in_region_clean.csv", "durian_in_region_clean_yolo11l.csv"):
        p = os.path.join(R, "results_clean", f)
        if os.path.isfile(p):
            rows += read(p)
    rows = [r for r in rows if r["checkpoint"] == "best" and r["eval_on"] == "outer"]
    pres, size = farm_presence(R)
    farms = sorted(size, key=int)

    # (model, config, class) -> [AP over seeds]
    ap_ = defaultdict(list)
    for r in rows:
        for c in NAMES:
            v = r.get("AP50::" + c, "")
            if v not in ("", None):
                ap_[(r["model"], r["config"], c)].append(float(v))

    by_farm, summary = [], []
    for m in a.models:
        for c in NAMES:
            carriers = [f for f in farms if pres[f].get(c, 0) > 0]
            rnd = ap_.get((m, "random", c))
            per_farm = []
            for f in carriers:
                v = ap_.get((m, f"byfarm_fold{f}", c))
                if not v:
                    continue
                trained_on = len(carriers) - 1
                row = {"model": m, "class": c, "heldout_farm": f,
                       "images_with_class": pres[f][c],
                       "train_farms_with_class": trained_on,
                       "train_images_with_class":
                           sum(pres[g].get(c, 0) for g in farms if g != f),
                       "new_farm_AP50": round(st.mean(v), 4),
                       "n_seeds": len(v)}
                by_farm.append(row)
                per_farm.append(st.mean(v))
            if not per_farm or not rnd:
                continue
            nf = st.mean(per_farm)
            summary.append({
                "model": m, "class": c, "farms_with_class": len(carriers),
                "random_split_AP50": round(st.mean(rnd), 4),
                "new_farm_AP50": round(nf, 4),
                "new_farm_min": round(min(per_farm), 4),
                "new_farm_max": round(max(per_farm), 4),
                "retained_pct": round(100 * nf / st.mean(rnd), 1)})

    out = os.path.join(R, "results_applied")
    os.makedirs(out, exist_ok=True)
    for name, data in (("per_class_transfer.csv", summary),
                       ("per_class_by_farm.csv", by_farm)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)

    # console: mean over models, which is what the text quotes
    print(f"{'class':<20}{'farms':>6}{'random':>9}{'new farm':>10}"
          f"{'range':>16}{'kept':>7}")
    for c in NAMES:
        s = [x for x in summary if x["class"] == c]
        if not s:
            continue
        print(f"{c:<20}{s[0]['farms_with_class']:>6}"
              f"{st.mean(x['random_split_AP50'] for x in s):>9.3f}"
              f"{st.mean(x['new_farm_AP50'] for x in s):>10.3f}"
              f"{min(x['new_farm_min'] for x in s):>8.3f}-"
              f"{max(x['new_farm_max'] for x in s):<7.3f}"
              f"{st.mean(x['retained_pct'] for x in s):>6.0f}%")

    # does training-farm coverage of a class track its new-farm AP?
    xs = [r["train_farms_with_class"] for r in by_farm]
    ys = [r["new_farm_AP50"] for r in by_farm]
    if len(set(xs)) > 1:
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i]); rk = [0]*len(v)
            i = 0
            while i < len(o):
                j = i
                while j+1 < len(o) and v[o[j+1]] == v[o[i]]:
                    j += 1
                for t in range(i, j+1):
                    rk[o[t]] = (i+j)/2
                i = j+1
            return rk
        rx, ry = rank(xs), rank(ys)
        mx, my = st.mean(rx), st.mean(ry)
        num = sum((p-mx)*(q-my) for p, q in zip(rx, ry))
        den = (sum((p-mx)**2 for p in rx)*sum((q-my)**2 for q in ry))**.5
        print(f"\nSpearman(train farms carrying the class, new-farm AP) = "
              f"{num/den:+.2f}  over {len(xs)} class x farm x model cells")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
