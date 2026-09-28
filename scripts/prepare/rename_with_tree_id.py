"""
Embed tree_id into filenames before uploading to Roboflow.

Expected layout:
    New folder (2)/
        algal/o1-t1/IMG_5183.jpg
        algal/o1-t3/IMG_5340.jpg
        scale_insect/o1-t7/IMG_5633.jpg
        leaf_lesion/o2-t1/IMG_5975.jpg
        ...

Produces:
    algal/o1-t1/o1t1_IMG_5183.jpg
    scale_insect/o1-t7/o1t7_IMG_5633.jpg
    ...

Why: Roboflow does not preserve your folder structure. Without the tree id
in the filename you cannot group by tree after export, and a random split
will put photos of the same tree on both sides of the train/test boundary.

Note: the class name is deliberately NOT put in the filename. The same
image may sit in several class folders, and Roboflow merges their
annotations into one label file only if the filenames match exactly.

Setup:
    pip install pandas

Usage:
    1. run once with DRY_RUN = True, read the report
    2. set DRY_RUN = False and run again
"""

import os
import re
import csv
import sys
import shutil
from collections import defaultdict

# ---------------------------------------------------------------- settings --
ROOT = r"C:\path\to\workdir\New folder (2)"
DRY_RUN = True                 # <-- set to False to actually rename
MAP_CSV = os.path.join(ROOT, "rename_map.csv")
IMG_EXT = (".jpg", ".jpeg", ".png", ".heic")
# ---------------------------------------------------------------------------


def normalise_tree(folder):
    """
    'o1-t3'  -> 'o1t3'
    'o1_t3'  -> 'o1t3'
    'o2-t1'  -> 'o2t1'
    'trunk7' -> 'trunk7'
    anything else -> sanitised lowercase, no separators
    """
    f = folder.strip()

    m = re.match(r"^(o\d+)[\s\-_]*t(\d+)$", f, flags=re.I)
    if m:
        return f"{m.group(1).lower()}t{m.group(2)}"

    m = re.match(r"^trunk[\s\-_]*(\d+)$", f, flags=re.I)
    if m:
        return f"trunk{m.group(1)}"

    return re.sub(r"[^0-9a-zA-Z]", "", f).lower()


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"Folder not found: {ROOT}")

    plan = []                       # (src, dst, class, tree, oldname, newname)
    unknown_folders = set()
    name_to_new = {}                # original basename -> new basename
    inconsistent = []
    collisions = []
    already = 0
    per_class = defaultdict(lambda: defaultdict(int))

    for class_dir in sorted(os.listdir(ROOT)):
        class_path = os.path.join(ROOT, class_dir)
        if not os.path.isdir(class_path):
            continue

        for tree_dir in sorted(os.listdir(class_path)):
            tree_path = os.path.join(class_path, tree_dir)
            if not os.path.isdir(tree_path):
                continue

            tree = normalise_tree(tree_dir)
            if not re.match(r"^(o\d+t\d+|trunk\d+)$", tree):
                unknown_folders.add(f"{class_dir}/{tree_dir} -> {tree}")

            for name in sorted(os.listdir(tree_path)):
                if not name.lower().endswith(IMG_EXT):
                    continue

                src = os.path.join(tree_path, name)

                # already carries a tree prefix?
                if re.match(r"^(o\d+t\d+|trunk\d+)_", name):
                    already += 1
                    per_class[class_dir][tree] += 1
                    continue

                new_name = f"{tree}_{name}"
                dst = os.path.join(tree_path, new_name)

                # same original name must always map to the same new name
                prev = name_to_new.get(name)
                if prev is None:
                    name_to_new[name] = new_name
                elif prev != new_name:
                    inconsistent.append((name, prev, new_name))

                if os.path.exists(dst):
                    collisions.append(dst)
                    continue

                plan.append((src, dst, class_dir, tree, name, new_name))
                per_class[class_dir][tree] += 1

    # ------------------------------------------------------------- report ---
    print(f"Root : {ROOT}")
    print(f"Mode : {'DRY RUN (nothing will change)' if DRY_RUN else 'RENAMING'}")
    print(f"\nTo rename        : {len(plan)}")
    print(f"Already prefixed : {already}")
    print(f"Name collisions  : {len(collisions)}")

    print("\nPer class / tree:")
    for cls in sorted(per_class):
        trees = per_class[cls]
        total = sum(trees.values())
        print(f"  {cls:24s} {total:5d} imgs, {len(trees):2d} trees")
        for t in sorted(trees):
            print(f"      {t:10s} {trees[t]:4d}")

    if unknown_folders:
        print("\nFolder names that did not match the expected pattern:")
        for u in sorted(unknown_folders):
            print(f"  {u}")
        print("  -> check these before running for real")

    if inconsistent:
        print("\nWARNING - same filename mapped to different new names:")
        for name, a, b in inconsistent:
            print(f"  {name}: {a}  vs  {b}")
        print("  -> the same image sits under two different trees. Fix this,")
        print("     otherwise Roboflow will treat them as two separate images")
        print("     and their annotations will not merge.")

    if collisions:
        print("\nWARNING - target filename already exists:")
        for c in collisions[:20]:
            print(f"  {c}")
        if len(collisions) > 20:
            print(f"  ... and {len(collisions) - 20} more")

    if not plan:
        print("\nNothing to do.")
        return

    # -------------------------------------------------------------- apply ---
    if DRY_RUN:
        print("\nDry run only. Set DRY_RUN = False to apply.")
        return

    if inconsistent:
        sys.exit("\nRefusing to run: resolve the inconsistent mappings first.")

    done, failed = 0, []
    for src, dst, cls, tree, old, new in plan:
        try:
            os.rename(src, dst)
            done += 1
        except Exception as e:
            failed.append((src, repr(e)))

    with open(MAP_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "tree_id", "old_name", "new_name", "path"])
        for src, dst, cls, tree, old, new in plan:
            w.writerow([cls, tree, old, new,
                        os.path.relpath(dst, ROOT).replace("\\", "/")])

    print(f"\nRenamed : {done}")
    print(f"Failed  : {len(failed)}")
    for src, err in failed[:20]:
        print(f"  {src} -> {err}")
    print(f"\nMapping written to {MAP_CSV}")
    print("\nAfter export from Roboflow, recover the group with:")
    print("  df['tree_id'] = df['filename'].str.extract(r'^(o\\d+t\\d+|trunk\\d+)_')")


if __name__ == "__main__":
    main()
