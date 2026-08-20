#!/usr/bin/env python3
"""
exif_audit.py - inventory the environmental metadata sitting in your photos,
and assign every image to a farm.

    pip install pillow pillow-heif
    python exif_audit.py /path/to/photos -o metadata.csv

What changed from the previous version
--------------------------------------
1. Writes `path` (relative to the scan root) alongside `file`, so images that
   share a basename can still be told apart. IMG_9885 exists twice in this
   library with different content; the old csv keyed on basename alone and
   silently conflated them.

2. Writes `sha1` and `stem`, so the csv can be joined to a Roboflow export
   (whose filenames carry a per-project hash suffix) and so exact duplicates
   are visible.

3. Assigns a `farm` to every image, not only the ones with GPS:
       - images with GPS are clustered at FARM_KM
       - each capture date is then mapped to whichever farm its GPS-bearing
         images belong to
       - images with a date but no GPS inherit that date's farm
   In this library the second rule recovers most of what was missing, because
   each farm was visited on a single day. `farm_source` records which rule
   fired, so the inherited assignments can be excluded if a reviewer objects.

4. Reports device-by-farm and date-by-farm crosstabs, because a camera or a
   date that appears at exactly one farm is confounded with it.

5. Flags duplicate basenames and duplicate content explicitly.
"""
import os
import sys
import csv
import math
import hashlib
import argparse
from collections import defaultdict, Counter
from datetime import datetime

try:
    from PIL import Image, ExifTags
except ImportError:
    sys.exit("pip install pillow")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".dng"}
GPSTAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
TAGS = {v: k for k, v in ExifTags.TAGS.items()}

FARM_KM = 1.5          # two photos within this distance belong to one farm
NEIGHBOUR_MIN = 30     # a GPS-less photo inherits a farm only if the
                       # nearest GPS photo that day is within this many
                       # minutes on both sides, or unambiguous on one
BORNEO_LON = 109.0     # east of this is Sabah / Sarawak


# ----------------------------------------------------------------- helpers --
def _rat(x):
    try:
        return float(x)
    except Exception:
        try:
            return x[0] / x[1]
        except Exception:
            return None


def dms_to_deg(dms, ref):
    try:
        d, m, s = (_rat(v) for v in dms)
        v = d + m / 60.0 + s / 3600.0
        return -v if ref in ("S", "W") else v
    except Exception:
        return None


def sha1_of(path, chunk=1 << 20):
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def haversine(a, b):
    R = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(a[0]))
         * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def solar_elevation(dt_local, lat, lon, tz_offset=8):
    """Solar elevation. Malaysia is UTC+8; exact enough for a covariate."""
    doy = dt_local.timetuple().tm_yday
    hour = dt_local.hour + dt_local.minute / 60 + dt_local.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g)
                       - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g)
                       - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    toff = eqtime + 4 * lon - 60 * tz_offset
    tst = hour * 60 + toff
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = (math.sin(la) * math.sin(decl)
          + math.cos(la) * math.cos(decl) * math.cos(ha))
    return 90 - math.degrees(math.acos(max(-1, min(1, cz))))


# -------------------------------------------------------------- extraction --
def read_one(path, root, want_hash=True):
    rel = os.path.relpath(path, root).replace("\\", "/")
    base = os.path.basename(path)
    out = {
        "file": base,
        "path": rel,
        "stem": os.path.splitext(base)[0],
        "folder": os.path.dirname(rel),
    }
    if want_hash:
        out["sha1"] = sha1_of(path)

    try:
        im = Image.open(path)
        ex = im.getexif()
        out["ImageWidth"], out["ImageLength"] = im.size
    except Exception as e:
        out["error"] = type(e).__name__
        return out

    for name in ("Make", "Model", "DateTimeOriginal", "DateTime", "Software"):
        t = TAGS.get(name)
        if t and t in ex:
            out[name] = str(ex[t])

    try:
        ifd = ex.get_ifd(0x8769)
        for name in ("DateTimeOriginal", "ISOSpeedRatings", "FNumber",
                     "ExposureTime", "FocalLength", "WhiteBalance",
                     "LensModel", "SubsecTimeOriginal"):
            t = TAGS.get(name)
            if t and t in ifd:
                out[name] = str(ifd[t])
    except Exception:
        pass

    try:
        g = ex.get_ifd(0x8825)
    except Exception:
        g = {}
    if g:
        lat = dms_to_deg(g.get(GPSTAGS["GPSLatitude"]),
                         g.get(GPSTAGS["GPSLatitudeRef"], "N"))
        lon = dms_to_deg(g.get(GPSTAGS["GPSLongitude"]),
                         g.get(GPSTAGS["GPSLongitudeRef"], "E"))
        if lat is not None and lon is not None:
            out["lat"], out["lon"] = lat, lon
        alt = _rat(g.get(GPSTAGS["GPSAltitude"]))
        if alt is not None:
            if g.get(GPSTAGS.get("GPSAltitudeRef"), 0) in (1, b"\x01"):
                alt = -alt
            out["altitude_m"] = round(alt, 1)
        for name in ("GPSImgDirection", "GPSSpeed", "GPSHPositioningError",
                     "GPSDateStamp", "GPSTimeStamp"):
            k = GPSTAGS.get(name)
            if k and k in g:
                v = g[k]
                if name == "GPSTimeStamp":
                    try:
                        v = ":".join("%02d" % int(_rat(x)) for x in v)
                    except Exception:
                        pass
                elif name != "GPSDateStamp":
                    v = _rat(v)
                out[name] = str(v)

    ts = out.get("DateTimeOriginal") or out.get("DateTime")
    if ts:
        try:
            dt = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S")
            out["date"] = dt.date().isoformat()
            out["hour"] = round(dt.hour + dt.minute / 60, 3)
            out["doy"] = dt.timetuple().tm_yday
            ang = 2 * math.pi * out["doy"] / 365.25
            out["doy_sin"] = round(math.sin(ang), 4)
            out["doy_cos"] = round(math.cos(ang), 4)
            if out.get("lat") is not None:
                out["solar_elev_deg"] = round(
                    solar_elevation(dt, out["lat"], out["lon"]), 2)
        except Exception:
            pass

    if out.get("lon") is not None:
        out["region"] = "Borneo" if out["lon"] > BORNEO_LON else "Peninsular"
    return out


# ------------------------------------------------------------ farm assign --
def assign_farms(rows, km=FARM_KM):
    """
    GPS-bearing rows are clustered; every date is then mapped to the farm its
    GPS-bearing images sit in, and dateless-but-GPS-less rows inherit nothing.
    Sets 'farm' and 'farm_source' on every row.
    """
    gps_rows = [r for r in rows if r.get("lat") is not None]

    # single-link clustering on the distinct coordinates
    clusters = []                       # list of list of (lat, lon)
    for r in gps_rows:
        p = (r["lat"], r["lon"])
        hit = None
        for c in clusters:
            if any(haversine(p, q) < km for q in c):
                hit = c
                break
        if hit is None:
            clusters.append([p])
        else:
            hit.append(p)

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if any(haversine(p, q) < km
                       for p in clusters[i] for q in clusters[j]):
                    clusters[i] += clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    clusters.sort(key=len, reverse=True)
    point2farm = {}
    for idx, c in enumerate(clusters):
        for p in c:
            point2farm[p] = idx

    for r in gps_rows:
        r["farm"] = point2farm[(r["lat"], r["lon"])]
        r["farm_source"] = "gps"

    # ---- rule 2: a date that maps to exactly one farm ----------------------
    date_farms = defaultdict(set)
    for r in gps_rows:
        if r.get("date"):
            date_farms[r["date"]].add(r["farm"])
    date2farm = {d: list(f)[0] for d, f in date_farms.items() if len(f) == 1}
    ambiguous = {d for d, f in date_farms.items() if len(f) > 1}

    # ---- rule 3: nearest GPS photo in time, same day -----------------------
    # On a multi-farm day the visits are sequential, so a photo taken between
    # two GPS photos of the same farm belongs to that farm. This is what
    # recovers the images shot on the second phone, which had location off.
    anchors = defaultdict(list)          # date -> [(minutes, farm), ...]
    for r in gps_rows:
        if r.get("date") and r.get("hour") is not None:
            anchors[r["date"]].append((r["hour"] * 60.0, r["farm"]))
    for d in anchors:
        anchors[d].sort()

    def infer_by_time(r):
        d, h = r.get("date"), r.get("hour")
        if not d or h is None or d not in anchors:
            return None, None
        t = h * 60.0
        seq = anchors[d]
        before = [(tt, f) for tt, f in seq if tt <= t]
        after = [(tt, f) for tt, f in seq if tt >= t]
        pb = before[-1] if before else None
        pa = after[0] if after else None

        if pb and pa:
            if pb[1] == pa[1]:
                return pb[1], "time_between"        # bracketed, same farm
            # straddles a move between farms: take the closer one if it is
            # clearly closer, otherwise refuse
            db, da = t - pb[0], pa[0] - t
            if min(db, da) <= NEIGHBOUR_MIN and abs(db - da) > NEIGHBOUR_MIN:
                return (pb[1] if db < da else pa[1]), "time_nearest"
            return None, "time_straddle"
        p = pb or pa
        if p and abs(t - p[0]) <= NEIGHBOUR_MIN:
            return p[1], "time_edge"
        return None, "time_far"

    for r in rows:
        if r.get("farm") is not None:
            continue
        d = r.get("date")
        if d and d in date2farm:
            r["farm"] = date2farm[d]
            r["farm_source"] = "date"
            continue
        f, why = infer_by_time(r)
        if f is not None:
            r["farm"] = f
            r["farm_source"] = why
        else:
            r["farm"] = None
            r["farm_source"] = why or ("ambiguous_date" if d in ambiguous
                                       else "none")

    return clusters, date2farm, ambiguous


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha1 (faster on a big library)")
    a = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(a.root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in EXTS:
                files.append(os.path.join(dp, fn))
    if not files:
        sys.exit("no images found under " + a.root)

    print(f"scanning {len(files)} files ...")
    if not HEIC:
        print("  (pillow-heif not installed; HEIC files will fail)")
    rows = [read_one(p, a.root, want_hash=not a.no_hash) for p in files]

    clusters, date2farm, ambiguous = assign_farms(rows)

    n = len(rows)
    gps = [r for r in rows if r.get("lat") is not None]
    tim = [r for r in rows if r.get("date")]
    farmed = [r for r in rows if r.get("farm") is not None]

    print("\n" + "=" * 68)
    print("  COVERAGE")
    print("=" * 68)
    print(f"  images                    {n}")
    print(f"  with GPS                  {len(gps):5d}  ({len(gps)/n:.0%})")
    print(f"  with capture time         {len(tim):5d}  ({len(tim)/n:.0%})")
    print(f"  assigned to a farm        {len(farmed):5d}  ({len(farmed)/n:.0%})")
    src = Counter(r.get("farm_source") for r in rows)
    for k in ("gps", "date", "time_between", "time_nearest", "time_edge",
              "time_straddle", "time_far", "ambiguous_date", "none"):
        if src.get(k):
            print(f"      via {k:16s}{src[k]:5d}")

    # ------------------------------------------------------------- farms ----
    print("\n" + "=" * 68)
    print(f"  FARMS (single-link at {FARM_KM*1000:.0f} m)")
    print("=" * 68)
    byfarm = defaultdict(list)
    for r in farmed:
        byfarm[r["farm"]].append(r)
    print(f"  {'farm':>4s} {'images':>7s} {'gps':>5s} {'date':>5s} "
          f"{'lat':>9s} {'lon':>10s}  {'region':<11s} dates")
    for f in sorted(byfarm, key=lambda k: -len(byfarm[k])):
        rs = byfarm[f]
        gl = [r for r in rs if r.get("lat") is not None]
        la = sum(r["lat"] for r in gl) / len(gl) if gl else float("nan")
        lo = sum(r["lon"] for r in gl) / len(gl) if gl else float("nan")
        ds = sorted({r["date"] for r in rs if r.get("date")})
        reg = "Borneo" if lo > BORNEO_LON else "Peninsular"
        print(f"  {f:4d} {len(rs):7d} "
              f"{sum(1 for r in rs if r.get('farm_source')=='gps'):5d} "
              f"{sum(1 for r in rs if r.get('farm_source')=='date'):5d} "
              f"{la:9.4f} {lo:10.4f}  {reg:<11s} "
              f"{','.join(ds[:3])}{'...' if len(ds)>3 else ''}")

    pen = sum(1 for r in gps if r["lon"] <= BORNEO_LON)
    bor = sum(1 for r in gps if r["lon"] > BORNEO_LON)
    print(f"\n  Peninsular {pen}   Borneo {bor}")
    if bor == 0:
        print("  No Borneo images in this scan - point the script at the Sabah")
        print("  folder too if the cross-island set should be included.")

    # ------------------------------------------------------- confounding ----
    print("\n" + "=" * 68)
    print("  CONFOUNDING CHECKS")
    print("=" * 68)

    dm = defaultdict(Counter)
    unassigned = Counter()
    for r in rows:
        m = r.get("Model", "?")
        if r.get("farm") is not None:
            dm[m][r["farm"]] += 1
        else:
            unassigned[m] += 1
    print("  device by farm:")
    allmodels = set(dm) | set(unassigned)
    for m in sorted(allmodels,
                    key=lambda x: -(sum(dm[x].values()) + unassigned[x])):
        c = dm.get(m, Counter())
        fs = sorted(c)
        print(f"    {str(m):22s} farms {fs}  assigned={sum(c.values())}  "
              f"unassigned={unassigned.get(m,0)}")
        if len(fs) == 1:
            print("        ^ appears at one farm only: confounded with it")
        if unassigned.get(m, 0) and not fs:
            print("        ^ no image from this device has a farm: it carried")
            print("          no GPS and could not be inferred")

    print("\n  GPS presence by device:")
    gpsdev = defaultdict(lambda: [0, 0])
    for r in rows:
        gpsdev[r.get("Model", "?")][0 if r.get("lat") is None else 1] += 1
    for m, (no, yes) in sorted(gpsdev.items(), key=lambda t: -sum(t[1])):
        print(f"    {str(m):22s} with GPS {yes:5d}   without {no:5d}")

    dd = defaultdict(set)
    for r in farmed:
        if r.get("date"):
            dd[r["farm"]].add(r["date"])
    single = [f for f, s in dd.items() if len(s) == 1]
    print(f"\n  farms visited on exactly one day: {len(single)} of {len(dd)}")
    if single:
        print("    farm and date are then the same grouping; a difference")
        print("    between farms cannot be separated from a difference")
        print("    between days.")

    if ambiguous:
        print(f"\n  dates spanning more than one farm: {sorted(ambiguous)}")
        print("    images from these dates were NOT given an inherited farm")

    # -------------------------------------------------------- duplicates ----
    print("\n" + "=" * 68)
    print("  DUPLICATES")
    print("=" * 68)
    by_stem = defaultdict(list)
    for r in rows:
        by_stem[r["stem"]].append(r)
    dup_names = {k: v for k, v in by_stem.items() if len(v) > 1}
    print(f"  repeated basenames        {len(dup_names)}")
    for k, v in list(dup_names.items())[:10]:
        same = len({r.get("sha1") for r in v}) == 1
        tag = "same content" if same else "DIFFERENT CONTENT"
        print(f"    {k:34s} x{len(v)}  {tag}")
        if not same:
            for r in v[:3]:
                print(f"        {r['path']}")

    if not a.no_hash:
        by_hash = defaultdict(list)
        for r in rows:
            if r.get("sha1"):
                by_hash[r["sha1"]].append(r)
        dup_content = {k: v for k, v in by_hash.items() if len(v) > 1}
        extra = sum(len(v) - 1 for v in dup_content.values())
        print(f"  identical files           {len(dup_content)} groups, "
              f"{extra} redundant copies")

    # ------------------------------------------------------------- write ----
    if a.out:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        for k in ("farm", "farm_source"):
            if k not in keys:
                keys.append(k)
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {a.out}  ({len(rows)} rows, {len(keys)} columns)")
        print("  join to a Roboflow export on 'stem' after stripping the")
        print("  '.rf.<hash>' suffix from the exported filenames.")


if __name__ == "__main__":
    main()
