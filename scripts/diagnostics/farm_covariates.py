"""
Recompute the per-farm covariates on the current six-class taxonomy.

A reviewer will say that five farms differing simultaneously in cultivar,
season, device, capture time and annotation session do not demonstrate
anything about evaluation protocol, only that heterogeneous samples give
heterogeneous results. That objection cannot be fully answered with five
farms, but it can be narrowed: if two measurable, non-geographic covariates
line up with the fold that fails, then part of what looks like domain shift
has a named mechanism.

The two covariates are:

  capture conditions   hour of day, solar elevation, device, lens, ISO.
                       Farm 2 was the worst fold. If it is also the only
                       farm photographed almost entirely under midday sun,
                       that is a concrete alternative to distance.

  annotation scale     median box area per class per farm. If the same
                       class is boxed at very different scales at different
                       farms, part of the cross-farm recall loss is a
                       target-size mismatch, not a failure to recognise the
                       lesion. Growers describe early and spread stages of
                       the same foliar disease as different problems, and
                       stage is not recorded in any dataset, so it can only
                       appear here as box size.

Nothing is trained. Everything comes from metadata.csv and the label files
of merged_peninsula, so the numbers correspond to the taxonomy actually
used in the reported experiments.

Setup:
    pip install pandas numpy pyyaml

Usage:
    python farm_covariates.py
    python farm_covariates.py --base /content/durian     # Colab
"""

import os
import sys
import glob
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- settings --
DEFAULT_BASE = (r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper"
                r"\durian_for_nature_food")

MIN_IMAGES = 5        # per class per farm, below this the cell is not shown
# ---------------------------------------------------------------------------


def find(base, *rel):
    for r in rel:
        p = os.path.join(base, *r) if isinstance(r, (list, tuple)) \
            else os.path.join(base, r)
        if os.path.exists(p):
            return p
    return None


def load_names(merged):
    with open(os.path.join(merged, "data.yaml"), encoding="utf-8") as fh:
        n = yaml.safe_load(fh)["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def load_boxes(merged, stem):
    p = os.path.join(merged, "labels", stem + ".txt")
    out = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                v = line.split()
                if len(v) >= 5:
                    out.append((int(float(v[0])),
                                float(v[3]) * float(v[4])))   # class, area
    return out


def fmt_px(area_frac, imgsz=640):
    """normalised area -> approximate side length in pixels at imgsz"""
    if not np.isfinite(area_frac) or area_frac <= 0:
        return "-"
    return f"{np.sqrt(area_frac) * imgsz:.0f}px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES)
    a = ap.parse_args()

    merged = find(a.base, os.path.join("dataset_store", "merged_peninsula"))
    meta_p = find(a.base, "metadata.csv", "metadata_backup.csv")
    split_p = find(a.base,
                   os.path.join("results", "split_assignment.csv"),
                   os.path.join("dataset_store", "splits_A",
                                "split_assignment.csv"))

    for p, what in [(merged, "merged_peninsula"), (meta_p, "metadata csv")]:
        if not p:
            sys.exit(f"{what} not found under {a.base}")

    names = load_names(merged)
    print(f"classes: {names}\n")

    # ---------------------------------------------------------- assemble ---
    meta = pd.read_csv(meta_p)
    if "stem" not in meta.columns:
        meta["stem"] = meta["file"].astype(str).str.replace(
            r"\.[^.]+$", "", regex=True)
    meta = meta.drop_duplicates("stem").set_index("stem")

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(merged, "images", "*.*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))

    rows = []
    for s in stems:
        if s not in meta.index:
            continue
        m = meta.loc[s]
        if pd.isna(m.get("farm")):
            continue
        b = load_boxes(merged, s)
        rows.append({
            "stem": s,
            "farm": int(m["farm"]),
            "farm_source": m.get("farm_source"),
            "date": m.get("date"),
            "hour": pd.to_numeric(m.get("hour"), errors="coerce"),
            "solar": pd.to_numeric(m.get("solar_elev_deg"), errors="coerce"),
            "model": m.get("Model"),
            "lens": m.get("LensModel"),
            "iso": pd.to_numeric(m.get("ISOSpeedRatings"), errors="coerce"),
            "focal": pd.to_numeric(m.get("FocalLength"), errors="coerce"),
            "n_boxes": len(b),
            "boxes": b,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no images matched metadata with a farm")

    print(f"images with a farm: {len(df)} across {df['farm'].nunique()} farms")

    # attach fold performance if we have it
    perf = {}
    fold_of = {}
    res = find(a.base, os.path.join("results", "in_region.csv"))
    if res and split_p:
        sp = pd.read_csv(split_p)
        for col in [c for c in sp.columns if c.startswith("byfarm_fold")]:
            fs = sorted(set(sp.loc[sp[col] == "val", "farm"].dropna()
                            .astype(int)))
            for f in fs:
                fold_of[f] = col
        ir = pd.read_csv(res)
        g = ir.groupby("config")["mAP50"].mean()
        for f, c in fold_of.items():
            if c in g.index:
                perf[f] = float(g[c])

    farms = sorted(df["farm"].unique())

    # ------------------------------------------------ capture conditions ---
    print("\n" + "=" * 78)
    print("  CAPTURE CONDITIONS BY FARM")
    print("=" * 78)
    hdr = (f"  {'farm':>4s} {'n':>5s} {'mAP50':>7s} {'hour med':>9s} "
           f"{'IQR':>13s} {'solar med':>10s} {'midday%':>8s} {'dates':>6s}")
    print(hdr)
    for f in farms:
        s = df[df["farm"] == f]
        h = s["hour"].dropna()
        sol = s["solar"].dropna()
        midday = (100 * ((h >= 12) & (h < 15)).sum() / len(h)) if len(h) else np.nan
        p = f"{perf[f]:7.3f}" if f in perf else f"{'-':>7s}"
        print(f"  {f:4d} {len(s):5d} {p} "
              f"{h.median() if len(h) else np.nan:9.2f} "
              f"{(f'{h.quantile(.25):.1f}-{h.quantile(.75):.1f}' if len(h) else '-'):>13s} "
              f"{sol.median() if len(sol) else np.nan:10.1f} "
              f"{midday:8.0f} {s['date'].nunique():6d}")

    print("\n  devices:")
    for f in farms:
        s = df[df["farm"] == f]
        c = s["model"].fillna("?").value_counts()
        parts = ", ".join(f"{k} {v}" for k, v in c.items())
        print(f"    farm {f}: {parts}")

    print("\n  ISO and focal length (median):")
    for f in farms:
        s = df[df["farm"] == f]
        print(f"    farm {f}: ISO {s['iso'].median():6.0f}   "
              f"focal {s['focal'].median():5.2f} mm")

    # correlation between midday share and fold score, if we have both
    if perf:
        rows_c = []
        for f in farms:
            if f not in perf:
                continue
            h = df.loc[df["farm"] == f, "hour"].dropna()
            sol = df.loc[df["farm"] == f, "solar"].dropna()
            if len(h) < a.min_images:
                continue
            rows_c.append({"farm": f, "mAP50": perf[f],
                           "midday_share": ((h >= 12) & (h < 15)).mean(),
                           "solar_med": sol.median() if len(sol) else np.nan})
        cc = pd.DataFrame(rows_c)
        if len(cc) >= 3:
            print("\n  farm-level association (n = %d farms, descriptive "
                  "only):" % len(cc))
            for col in ("midday_share", "solar_med"):
                v = cc[[col, "mAP50"]].dropna()
                if len(v) >= 3:
                    r = v[col].corr(v["mAP50"], method="spearman")
                    print(f"    Spearman rho({col}, mAP50) = {r:+.2f}")
            print("    with five farms this cannot be a test; it is reported")
            print("    to show the direction, not to claim significance.")

    # ------------------------------------------------- annotation scale ----
    print("\n" + "=" * 78)
    print("  ANNOTATION SCALE: MEDIAN BOX AREA PER CLASS PER FARM")
    print("=" * 78)

    per = defaultdict(lambda: defaultdict(list))     # class -> farm -> areas
    imgs = defaultdict(lambda: defaultdict(int))
    for _, r in df.iterrows():
        seen = set()
        for c, area in r["boxes"]:
            per[c][r["farm"]].append(area)
            seen.add(c)
        for c in seen:
            imgs[c][r["farm"]] += 1

    print(f"  {'class':22s}" + "".join(f"{('f'+str(f)):>12s}" for f in farms)
          + f"{'ratio':>8s}")
    scale_rows = []
    for i, n in enumerate(names):
        cells, vals = "", {}
        for f in farms:
            areas = per[i].get(f, [])
            if imgs[i].get(f, 0) >= a.min_images and areas:
                med = float(np.median(areas))
                vals[f] = med
                cells += f"{med:12.5f}"
            else:
                cells += f"{'-':>12s}"
        ratio = (max(vals.values()) / min(vals.values())
                 if len(vals) >= 2 and min(vals.values()) > 0 else np.nan)
        rs = f"{ratio:8.1f}" if np.isfinite(ratio) else f"{'-':>8s}"
        print(f"  {n:22s}{cells}{rs}")
        scale_rows.append({"class": n, "ratio": ratio, **{f"f{f}": vals.get(f)
                                                          for f in farms}})

    print(f"\n  the same, as an approximate box side at {a.imgsz} px input:")
    print(f"  {'class':22s}" + "".join(f"{('f'+str(f)):>12s}" for f in farms))
    for i, n in enumerate(names):
        cells = ""
        for f in farms:
            areas = per[i].get(f, [])
            if imgs[i].get(f, 0) >= a.min_images and areas:
                cells += f"{fmt_px(float(np.median(areas)), a.imgsz):>12s}"
            else:
                cells += f"{'-':>12s}"
        print(f"  {n:22s}{cells}")

    print(f"\n  images contributing to each cell (blank cells below "
          f"{a.min_images}):")
    print(f"  {'class':22s}" + "".join(f"{('f'+str(f)):>12s}" for f in farms))
    for i, n in enumerate(names):
        print(f"  {n:22s}" + "".join(f"{imgs[i].get(f,0):12d}"
                                     for f in farms))

    print("\n  boxes per image, within class and farm (median):")
    print(f"  {'class':22s}" + "".join(f"{('f'+str(f)):>12s}" for f in farms))
    for i, n in enumerate(names):
        cells = ""
        for f in farms:
            sub = df[df["farm"] == f]
            counts = [sum(1 for c, _ in r["boxes"] if c == i)
                      for _, r in sub.iterrows()]
            counts = [c for c in counts if c > 0]
            cells += (f"{np.median(counts):12.1f}"
                      if len(counts) >= a.min_images else f"{'-':>12s}")
        print(f"  {n:22s}{cells}")

    # ------------------------------------------------------- class share ---
    print("\n" + "=" * 78)
    print("  CLASS COMPOSITION: share of a farm's images carrying a class")
    print("=" * 78)
    print(f"  {'class':22s}" + "".join(f"{('f'+str(f)):>12s}" for f in farms))
    tot = df.groupby("farm").size()
    for i, n in enumerate(names):
        print(f"  {n:22s}" + "".join(
            f"{100*imgs[i].get(f,0)/tot[f]:11.1f}%" for f in farms))

    # ----------------------------------------------------------- outputs ---
    out_dir = os.path.join(a.base, "results")
    os.makedirs(out_dir, exist_ok=True)

    cap = []
    for f in farms:
        s = df[df["farm"] == f]
        h = s["hour"].dropna()
        cap.append({
            "farm": f, "images": len(s), "mAP50_fold": perf.get(f),
            "hour_median": float(h.median()) if len(h) else None,
            "solar_median": float(s["solar"].dropna().median())
            if s["solar"].notna().any() else None,
            "midday_share": float(((h >= 12) & (h < 15)).mean())
            if len(h) else None,
            "dates": int(s["date"].nunique()),
            "devices": ";".join(sorted(set(s["model"].dropna().astype(str)))),
            "iso_median": float(s["iso"].median())
            if s["iso"].notna().any() else None,
        })
    pd.DataFrame(cap).to_csv(
        os.path.join(out_dir, "farm_capture_conditions.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(scale_rows).to_csv(
        os.path.join(out_dir, "farm_annotation_scale.csv"),
        index=False, encoding="utf-8-sig")

    print(f"\nwrote {out_dir}/farm_capture_conditions.csv")
    print(f"wrote {out_dir}/farm_annotation_scale.csv")

    print("""
  How to use this. Two numbers decide whether the subsection is worth
  writing. First, whether the worst fold is also an outlier on capture
  conditions: a farm photographed almost entirely under midday sun is
  experiencing a light domain the training set barely contains, which is a
  named mechanism rather than an appeal to distance. Second, whether the
  same class is boxed at very different scales at different farms: a
  several-fold ratio means the detector is asked to find targets an order of
  magnitude apart under one label, which is a target-size mismatch and not a
  failure of recognition.

  Report both as descriptive. With five farms no correlation is a test, and
  saying so in the text is stronger than implying otherwise.""")


if __name__ == "__main__":
    main()
