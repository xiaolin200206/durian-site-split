"""
Merge each region's Roboflow exports, then check that the two label spaces
line up.

Every class lives in its own Roboflow project, so each export carries its
own class indices, and the same photograph appears in several exports when
it holds several diseases. Roboflow also appends a per-project hash to
filenames, so one photograph has a different filename in every export.

For each region this script:
  - recovers the original filename by stripping the '.rf.<hash>' suffix
  - merges the label files of every export holding that photograph
  - remaps each export's local class index onto one shared index
  - drops boxes that are duplicated across exports
  - flattens train/valid/test, since the splits are rebuilt afterwards

It then compares the two regions. If the class lists differ in content or
in order, the cross-island evaluation is meaningless, because the same
integer would mean different things on the two sides. The script says so
loudly rather than proceeding.

Setup:
    pip install pyyaml

Usage:
    python merge_regions.py
    python merge_regions.py --bg-limit 120
"""

import os
import re
import sys
import glob
import shutil
import zipfile
import hashlib
import tempfile
import argparse
from collections import defaultdict, Counter

import yaml

# ---------------------------------------------------------------- settings --
BASE = (r"C:\path\to\workdir\Durian project and paper"
        r"\durian_for_nature_food")

REGIONS = {
    "peninsula": {
        "store": os.path.join(BASE, "freeze_peninsula_labelled_dataset"),
        "out":   os.path.join(BASE, "dataset_store", "merged_peninsula"),
        "background": "Background.v4i.yolov8.zip",
    },
    "sabah": {
        "store": os.path.join(BASE, "freeze_sabah_lebeling_dataset"),
        "out":   os.path.join(BASE, "dataset_store", "merged_sabah"),
        "background": None,
    },
}

BG_LIMIT = 120          # cap on negatives, peninsula only; None for all
IOU_DEDUP = 0.90

# One class list, one order, for both regions. The integer in a label file
# means nothing on its own, so the two regions must agree on this list
# exactly or every box is silently relabelled. Exports whose class names
# differ only in case or in separators are folded in automatically; anything
# else has to be added to ALIASES or the run stops.
CANONICAL_CLASSES = [
    "Algal",
    "Leaf_rot",
    "Phomopsis",
    "Psyllid",
    "Psyllid_damage",
    "leaf_hopper_damage",
]

# map an awkward export name onto a canonical one
ALIASES = {
    "leafhopper_damage": "leaf_hopper_damage",
    "leafhopper damage": "leaf_hopper_damage",
    "leaf hopper damage": "leaf_hopper_damage",
}
# ---------------------------------------------------------------------------

RF_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)
TREE_RE = re.compile(r"^(o\d+t\d+|trunk\d+)_", re.I)


def original_stem(fname):
    return RF_SUFFIX.sub("", os.path.splitext(fname)[0])


def sha1_of(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def iou_xywhn(a, b):
    def xyxy(v):
        x, y, w, h = v
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def canon_key(name):
    """case- and separator-insensitive key"""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


CANON_LOOKUP = {}


def build_canon_lookup():
    CANON_LOOKUP.clear()
    for c in CANONICAL_CLASSES:
        CANON_LOOKUP[canon_key(c)] = c
    for src_name, dst in ALIASES.items():
        if dst not in CANONICAL_CLASSES:
            sys.exit(f"alias target not in CANONICAL_CLASSES: {dst}")
        CANON_LOOKUP[canon_key(src_name)] = dst


def to_canonical(name, where):
    c = CANON_LOOKUP.get(canon_key(name))
    if c is None:
        sys.exit(
            f"\nunknown class {name!r} in {where}.\n"
            f"known: {CANONICAL_CLASSES}\n"
            f"add it to CANONICAL_CLASSES, or map it in ALIASES.")
    return c


def read_names(root):
    for cand in ("data.yaml", "data.yml"):
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                y = yaml.safe_load(fh)
            n = y.get("names")
            if isinstance(n, dict):
                n = [n[k] for k in sorted(n)]
            return n
    return None


def walk_export(root):
    for split in ("train", "valid", "test", "."):
        img_dir = (os.path.join(root, "images") if split == "."
                   else os.path.join(root, split, "images"))
        lbl_dir = (os.path.join(root, "labels") if split == "."
                   else os.path.join(root, split, "labels"))
        if not os.path.isdir(img_dir):
            continue
        for img in sorted(os.listdir(img_dir)):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ip = os.path.join(img_dir, img)
            lp = os.path.join(lbl_dir, os.path.splitext(img)[0] + ".txt")
            yield ip, (lp if os.path.isfile(lp) else None)


def merge_region(tag, cfg, bg_limit):
    store = cfg["store"]
    out = cfg["out"]
    if not os.path.isdir(store):
        sys.exit(f"[{tag}] store not found: {store}")
    if os.path.exists(out):
        sys.exit(f"[{tag}] output exists, refusing to overwrite: {out}")

    zips = sorted(z for z in os.listdir(store) if z.lower().endswith(".zip"))
    bg_zip = cfg.get("background")
    label_zips = [z for z in zips if z != bg_zip]

    print("\n" + "=" * 74)
    print(f"  {tag.upper()}   {store}")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix=f"rf_{tag}_")
    unified = list(CANONICAL_CLASSES)
    renamed = []
    boxes = defaultdict(list)
    img_src, img_hash = {}, {}
    seen_in = defaultdict(set)
    clash = []

    for z in label_zips:
        root = os.path.join(tmp, os.path.splitext(z)[0])
        with zipfile.ZipFile(os.path.join(store, z)) as zf:
            zf.extractall(root)

        names = read_names(root)
        if not names:
            print(f"  {z}: no data.yaml, skipped")
            continue

        local2global = {}
        for i, n in enumerate(names):
            canonical = to_canonical(n, f"{tag}/{z}")
            local2global[i] = CANONICAL_CLASSES.index(canonical)
            if canonical != n:
                renamed.append((z, n, canonical))

        n_img = n_box = 0
        for ip, lp in walk_export(root):
            stem = original_stem(os.path.basename(ip))
            seen_in[stem].add(z)
            n_img += 1

            h = sha1_of(ip)
            if stem in img_hash:
                if img_hash[stem] != h:
                    clash.append((stem, z))
            else:
                img_hash[stem] = h
                img_src[stem] = ip

            if lp is None:
                continue
            with open(lp, "r", encoding="utf-8") as fh:
                for line in fh:
                    v = line.split()
                    if len(v) < 5:
                        continue
                    boxes[stem].append(
                        (local2global.get(int(float(v[0])), int(float(v[0]))),
                         float(v[1]), float(v[2]), float(v[3]), float(v[4])))
                    n_box += 1

        print(f"  {z:38s} {str(names):40s} {n_img:5d} img {n_box:6d} box")

    if renamed:
        print("\n  class names folded onto the canonical list:")
        for z, was, now in renamed:
            print(f"    {z:38s} {was!r} -> {now!r}")

    # de-duplicate boxes that appear in more than one export
    before = sum(len(v) for v in boxes.values())
    for stem, bs in boxes.items():
        kept = []
        for b in bs:
            if not any(k[0] == b[0] and iou_xywhn(b[1:], k[1:]) >= IOU_DEDUP
                       for k in kept):
                kept.append(b)
        boxes[stem] = kept
    after = sum(len(v) for v in boxes.values())

    # negatives
    bg_stems = []
    if bg_zip and os.path.isfile(os.path.join(store, bg_zip)):
        root = os.path.join(tmp, "background")
        with zipfile.ZipFile(os.path.join(store, bg_zip)) as zf:
            zf.extractall(root)
        cands = []
        for ip, _ in walk_export(root):
            stem = original_stem(os.path.basename(ip))
            if stem not in img_src:
                cands.append((stem, ip))
        if bg_limit is not None and len(cands) > bg_limit:
            step = len(cands) / bg_limit
            cands = [cands[int(i * step)] for i in range(bg_limit)]
        for stem, ip in cands:
            img_src[stem] = ip
            boxes[stem] = []
            bg_stems.append(stem)
        print(f"  {bg_zip:38s} {len(bg_stems)} negatives kept")

    # write
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "labels"))
    for stem in sorted(img_src):
        ext = os.path.splitext(img_src[stem])[1].lower()
        shutil.copy2(img_src[stem], os.path.join(out, "images", stem + ext))
        with open(os.path.join(out, "labels", stem + ".txt"), "w",
                  encoding="utf-8") as fh:
            for c, x, y, w, h in boxes[stem]:
                fh.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": out.replace("\\", "/"),
                        "train": "images", "val": "images",
                        "nc": len(unified), "names": unified},
                       fh, sort_keys=False, allow_unicode=True)

    # report
    cnt = Counter(c for bs in boxes.values() for c, *_ in bs)
    imgs = Counter()
    trees = defaultdict(set)
    for stem, bs in boxes.items():
        m = TREE_RE.match(stem)
        t = m.group(1).lower() if m else None
        for c in {b[0] for b in bs}:
            imgs[c] += 1
            if t:
                trees[c].add(t)

    print(f"\n  classes ({len(unified)}, canonical order): {unified}")
    print(f"  images  : {len(img_src)}  "
          f"({len(img_src)-len(bg_stems)} labelled, {len(bg_stems)} negative)")
    print(f"  boxes   : {after}  (deduped {before-after})")
    print(f"\n  {'class':26s} {'boxes':>7s} {'images':>7s} {'trees':>6s}")
    for i, n in enumerate(unified):
        flag = "   <- absent in this region" if cnt.get(i, 0) == 0 else ""
        print(f"  {n:26s} {cnt.get(i,0):7d} {imgs.get(i,0):7d} "
              f"{len(trees.get(i, set())):6d}{flag}")

    multi = [s for s in seen_in if len(seen_in[s]) > 1]
    print(f"\n  images in more than one export: {len(multi)}")
    if clash:
        print(f"  WARNING same stem different content: {len(clash)}")
        for s, z in clash[:8]:
            print(f"    {s} in {z}")

    shutil.rmtree(tmp, ignore_errors=True)
    return unified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-limit", type=int, default=BG_LIMIT)
    ap.add_argument("--all-background", action="store_true")
    a = ap.parse_args()

    bg = None if a.all_background else a.bg_limit

    build_canon_lookup()

    spaces = {}
    for tag, cfg in REGIONS.items():
        spaces[tag] = merge_region(tag, cfg, bg)

    # ------------------------------------------------- label space check ---
    print("\n" + "=" * 74)
    print("  LABEL SPACE")
    print("=" * 74)
    for i, n in enumerate(CANONICAL_CLASSES):
        print(f"  {i}  {n}")

    ok = all(v == CANONICAL_CLASSES for v in spaces.values())
    if ok:
        print("\n  both regions written against this list, same order.")
        print("  a class index means the same thing on both sides, so the")
        print("  cross-region evaluation is sound.")
    else:
        print("\n  MISMATCH - this should not happen; check the script.")

    print("\n  next:")
    for tag, cfg in REGIONS.items():
        print(f"    {tag:12s} {cfg['out']}")


if __name__ == "__main__":
    main()
