"""
Is the 92.8-fold Leaf_rot size range a difference in lesions or in distance?

Median Leaf_rot box area is 0.02242 at farm 6 and 0.00024 at farm 2, which
at a 640 px input is roughly 96 px against 10 px on a side. The manuscript
reads that as disease stage: an early lesion at a leaf tip against a spread
necrotic region. There is a competing reading that would undermine the
claim, namely that the lesions are the same size and the photographer simply
stood further back at farm 2, so the lesion occupies a smaller fraction of
the frame.

The two readings are separable by looking. If farm 2's red boxes sit at leaf
tips as small discrete spots while a whole leaf fills the frame, it is
stage. If farm 2's frames are wide shots containing several leaves at a
distance, it is camera distance.

This renders annotated samples from the extreme farms into one folder,
named so the farms sort together and the size ordering is visible without
opening anything, and prints the numbers that bear on the same question:
how much of the frame the leaf occupies, and how far the camera was.

Nothing is trained.

Usage:
    python check_lesion_scale.py
    python check_lesion_scale.py --class Algal --farms 0 2
    python check_lesion_scale.py --n 8
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- settings --
DEFAULT_BASE = (r"C:\path\to\workdir\Durian project and paper"
                r"\durian_for_nature_food")

TARGET = "Leaf_rot"
FARMS = None            # None = the two farms with the most extreme medians
N_PER_FARM = 6
MAX_SIDE = 1400
IMGSZ = 640
# ---------------------------------------------------------------------------


def find(base, *rel):
    for r in rel:
        p = os.path.join(base, r)
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
                    out.append((int(float(v[0])), float(v[1]), float(v[2]),
                                float(v[3]), float(v[4])))
    return out


def find_image(merged, stem):
    hits = glob.glob(os.path.join(merged, "images", stem + ".*"))
    return hits[0] if hits else None


def draw(merged, stem, cls_idx, names, out_path, farm, med_area):
    ip = find_image(merged, stem)
    if not ip:
        return
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, int(min(W, H) / 350))

    n_t = 0
    for c, x, y, w, h in load_boxes(merged, stem):
        x1, y1 = (x - w / 2) * W, (y - h / 2) * H
        x2, y2 = (x + w / 2) * W, (y + h / 2) * H
        if c == cls_idx:
            d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=lw * 2)
            n_t += 1
        else:
            d.rectangle([x1, y1, x2, y2], outline=(80, 150, 255), width=lw)

    try:
        font = ImageFont.truetype("arial.ttf", max(20, int(W / 42)))
    except Exception:
        font = ImageFont.load_default()
    side = np.sqrt(med_area) * IMGSZ if med_area > 0 else 0
    cap = (f"farm {farm}   {names[cls_idx]}   {n_t} box   "
           f"median {med_area:.5f}  ~{side:.0f}px at {IMGSZ}")
    d.rectangle([0, 0, W, int(W / 26)], fill=(0, 0, 0))
    d.text((8, 4), cap, fill=(255, 255, 255), font=font)

    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)),
                       Image.BILINEAR)
    im.save(out_path, "JPEG", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--class", dest="cls", default=TARGET)
    ap.add_argument("--farms", type=int, nargs="+", default=FARMS)
    ap.add_argument("--n", type=int, default=N_PER_FARM)
    a = ap.parse_args()

    merged = find(a.base, os.path.join("dataset_store", "merged_peninsula"))
    meta_p = find(a.base, "metadata.csv", "metadata_backup.csv")
    if not merged or not meta_p:
        sys.exit(f"merged_peninsula or metadata not found under {a.base}")

    names = load_names(merged)
    if a.cls not in names:
        sys.exit(f"{a.cls} not in {names}")
    ci = names.index(a.cls)

    meta = pd.read_csv(meta_p)
    if "stem" not in meta.columns:
        meta["stem"] = meta["file"].astype(str).str.replace(
            r"\.[^.]+$", "", regex=True)
    meta = meta.drop_duplicates("stem").set_index("stem")

    # gather per farm
    per = defaultdict(list)     # farm -> [(stem, n_boxes, median_area, W, H)]
    for p in sorted(glob.glob(os.path.join(merged, "images", "*.*"))):
        if not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in meta.index:
            continue
        m = meta.loc[stem]
        if pd.isna(m.get("farm")):
            continue
        areas = [w * h for c, x, y, w, h in load_boxes(merged, stem)
                 if c == ci]
        if not areas:
            continue
        per[int(m["farm"])].append(
            (stem, len(areas), float(np.median(areas)),
             m.get("focal"), m.get("hour")))

    if not per:
        sys.exit(f"no {a.cls} boxes found")

    med = {f: float(np.median([t[2] for t in v])) for f, v in per.items()
           if len(v) >= 5}
    if not med:
        sys.exit("no farm has at least 5 images of this class")

    farms = a.farms
    if not farms:
        lo = min(med, key=med.get)
        hi = max(med, key=med.get)
        farms = [hi, lo]
    farms = [f for f in farms if f in per]

    print(f"class {a.cls}")
    print(f"  {'farm':>5s} {'imgs':>5s} {'median area':>12s} "
          f"{'~px@640':>8s} {'boxes/img':>10s}")
    for f in sorted(med, key=med.get, reverse=True):
        v = per[f]
        print(f"  {f:5d} {len(v):5d} {med[f]:12.5f} "
              f"{np.sqrt(med[f])*IMGSZ:8.0f} "
              f"{np.median([t[1] for t in v]):10.1f}")

    # ---------------------------------------------- distance surrogates ---
    print("\n  camera distance surrogates, "
          "the competing explanation for the size gap:")
    print(f"  {'farm':>5s} {'focal med':>10s} {'largest box':>12s} "
          f"{'frame share':>12s}")
    for f in sorted(med, key=med.get, reverse=True):
        v = per[f]
        # the biggest box on each image approximates how much of the frame
        # the nearest subject occupies, whatever class it belongs to
        big = []
        for stem, _, _, _, _ in v:
            allb = [w * h for c, x, y, w, h in load_boxes(merged, stem)]
            if allb:
                big.append(max(allb))
        foc = [t[3] for t in v if pd.notna(t[3])]
        print(f"  {f:5d} {np.median(foc) if foc else np.nan:10.2f} "
              f"{np.median(big) if big else np.nan:12.5f} "
              f"{np.median(big)*100 if big else np.nan:11.2f}%")
    print("""
  If the farm with small lesions also has a much smaller largest-box share,
  the photographer was standing further back and the size gap is distance.
  If the largest-box share is similar while the target boxes differ by an
  order of magnitude, the frames are comparable and the lesions are not.""")

    # -------------------------------------------------------- render ------
    out = os.path.join(a.base, "lesion_scale_check", a.cls)
    os.makedirs(out, exist_ok=True)
    for f in farms:
        v = sorted(per[f], key=lambda t: t[2])
        idx = (np.linspace(0, len(v) - 1, min(a.n, len(v))).astype(int)
               if len(v) > a.n else range(len(v)))
        for k in idx:
            stem, nb, ma, _, _ = v[k]
            name = f"f{f:02d}_a{int(ma*1e6):07d}_n{nb:03d}_{stem}.jpg"
            draw(merged, stem, ci, names, os.path.join(out, name), f, ma)
        print(f"\n  farm {f}: rendered {len(list(idx))} samples")

    print(f"\nwritten to {out}")
    print("""
  Open the folder at largest icons, sorted by name. Filenames start with the
  farm, then the median box area, so the two farms group together and the
  ordering is visible before opening anything. Red boxes are the class in
  question, blue are the other classes on the same image.

  The question to answer in ten seconds: at the farm with the small median,
  is a whole leaf filling the frame with a small lesion at its tip, or is
  the frame a wide shot of several leaves? The first is stage. The second is
  distance, and the manuscript claim would have to change.""")


if __name__ == "__main__":
    main()
