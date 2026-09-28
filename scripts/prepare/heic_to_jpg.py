"""
HEIC -> JPEG converter for field photos.

Preserves EXIF (GPS, timestamp, camera settings) and folder structure.
Does NOT touch the original files.

Setup (run once in cmd/PowerShell):
    pip install pillow pillow-heif

Usage:
    python heic_to_jpg.py
"""

import os
import glob
import sys

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ---------------------------------------------------------------- settings --
SRC = r"C:\path\to\workdir\sabah"
DST = r"C:\path\to\workdir\sabah_jpg"
QUALITY = 95          # keep high; low quality destroys fine disease texture
SUBSAMPLING = 0       # 4:4:4, no chroma subsampling
# ---------------------------------------------------------------------------


def find_heic(root):
    """Case-insensitive search for .heic / .heif under root."""
    found = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".heic", ".heif")):
                full = os.path.join(dirpath, name)
                found[full.lower()] = full
    return sorted(found.values())


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"Source folder not found: {SRC}")

    files = find_heic(SRC)
    if not files:
        sys.exit(f"No HEIC files found under {SRC}")

    print(f"Found {len(files)} HEIC files.")
    print(f"Writing JPEGs to {DST}\n")

    ok, failed = 0, []

    for i, src_path in enumerate(files, 1):
        rel = os.path.relpath(src_path, SRC)
        out_path = os.path.join(DST, os.path.splitext(rel)[0] + ".jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            print(f"[{i}/{len(files)}] skip (exists): {rel}")
            ok += 1
            continue

        try:
            img = Image.open(src_path)
            exif = img.info.get("exif")

            if img.mode != "RGB":
                img = img.convert("RGB")

            save_args = {
                "format": "JPEG",
                "quality": QUALITY,
                "subsampling": SUBSAMPLING,
            }
            if exif:
                save_args["exif"] = exif

            img.save(out_path, **save_args)
            ok += 1
            print(f"[{i}/{len(files)}] {rel}")

        except Exception as e:
            failed.append((rel, repr(e)))
            print(f"[{i}/{len(files)}] FAILED: {rel}  ->  {e}")

    # -------------------------------------------------------------- summary --
    print("\n" + "-" * 60)
    print(f"Converted OK : {ok}")
    print(f"Failed       : {len(failed)}")

    if failed:
        log = os.path.join(DST, "_failed.txt")
        with open(log, "w", encoding="utf-8") as fh:
            for name, err in failed:
                fh.write(f"{name}\t{err}\n")
        print(f"Failure list written to {log}")

    print("\nNext: verify EXIF survived, e.g.")
    print(r'  exiftool -GPSLatitude -DateTimeOriginal -Model "some_file.jpg"')


if __name__ == "__main__":
    main()
