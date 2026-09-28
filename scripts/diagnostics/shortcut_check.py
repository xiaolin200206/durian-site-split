"""
!! TAXONOMY WARNING -- READ BEFORE CITING ANY OUTPUT OF THIS SCRIPT !!

CLASS_NAMES below is the superseded twelve-class taxonomy (Pink_disease,
Root_disease, Scale_insect ...). The paper reports a six-class model. This
script therefore does NOT test the model the paper describes, and no result
from it may be cited in the manuscript until it is re-run on the six-class
weights. A sentence claiming a shortcut diagnostic "following Noyan's
design" was drafted for Related Work on the strength of this script and was
removed for exactly this reason.

Shortcut-learning diagnostic on healthy-trunk negatives.

Runs the unified 12-class detector over the trunk1..trunk9 folders, which
contain healthy durian trunks only. Any detection is a false positive by
construction. The question of interest is whether Pink_disease fires on
bark that carries no disease at all -- which would indicate the class is
keyed on capture-source characteristics rather than on the lesion.

Reports, per class:
  - how many images produce at least one detection
  - how many detections in total
  - the maximum and the distribution of confidence
  - the same broken down per tree (trunk1..trunk9)

Screenshots taken in the field as run separators (phone screen
resolution) are excluded: they contain no bark and the model produces
fixed-size boxes on their UI elements.

Setup:
    pip install onnxruntime pillow numpy pandas

Usage:
    python shortcut_check.py
"""

import os
import glob
import sys

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort

# ---------------------------------------------------------------- settings --
ROOT = r"C:\path\to\workdir\sabah"
MODEL = os.path.join(ROOT, "best.onnx")
TRUNK_GLOB = os.path.join(ROOT, "trunk*")
OUT_CSV = os.path.join(ROOT, "shortcut_check_detections.csv")

IMGSZ = 640
CONF_FLOOR = 0.001        # collect everything, threshold afterwards
IOU_NMS = 0.7
REPORT_AT = [0.001, 0.10, 0.25, 0.50]   # thresholds to report

# Field run-separator screenshots: phone screen resolution, not bark.
# These are excluded -- they are not trunk images at all.
SCREENSHOT_SIZES = {(1320, 2868), (2868, 1320)}

# VERIFY THIS ORDER against your data.yaml before trusting the output.
CLASS_NAMES = [
    "Algal",
    "Leaf_rot",
    "Phomopsis",
    "Pink_disease",
    "Root_disease",
    "Psyllid",
    "Psyllid_damage",
    "Scale_insect",
    "Stem_borer",
    "leafhopper_damage",
    "weevil",
    "weevil_damage",
]
# ---------------------------------------------------------------------------


def letterbox(im, new=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to square. Returns array + scale/pad."""
    w, h = im.size
    r = min(new / w, new / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new, new), color)
    dw, dh = (new - nw) // 2, (new - nh) // 2
    canvas.paste(im, (dw, dh))
    return canvas, r, dw, dh


def nms(boxes, scores, iou_thr):
    """boxes: (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(out, nc):
    """
    Ultralytics YOLOv8/v11 ONNX head: (1, 4+nc, A) or (1, A, 4+nc).
    Returns boxes (xyxy, letterbox space), scores, class ids.
    """
    p = out[0]
    if p.ndim == 3:
        p = p[0]
    # orient so rows are anchors
    if p.shape[0] == 4 + nc:
        p = p.T
    elif p.shape[1] != 4 + nc:
        raise ValueError(f"unexpected head shape {out[0].shape}, nc={nc}")

    xywh = p[:, :4]
    cls_scores = p[:, 4:]
    conf = cls_scores.max(axis=1)
    cid = cls_scores.argmax(axis=1)

    m = conf > CONF_FLOOR
    xywh, conf, cid = xywh[m], conf[m], cid[m]

    cx, cy, w, h = xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, conf, cid


def main():
    if not os.path.isfile(MODEL):
        sys.exit(f"Model not found: {MODEL}")

    dirs = sorted(d for d in glob.glob(TRUNK_GLOB) if os.path.isdir(d))
    if not dirs:
        sys.exit(f"No trunk folders under {ROOT}")

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    print(f"Model  : {os.path.basename(MODEL)}")
    print(f"Input  : {inp.name} {inp.shape}")
    print(f"Output : {out_meta.name} {out_meta.shape}")

    nc = len(CLASS_NAMES)
    print(f"Classes assumed: {nc}  -- VERIFY against your data.yaml\n")

    rows = []
    n_images = 0
    n_skipped = 0

    for d in dirs:
        tree = os.path.basename(d)
        all_files = [f for f in sorted(glob.glob(os.path.join(d, "*")))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files, skipped = [], []
        for f in all_files:
            if f.lower().endswith(".png"):
                skipped.append(f)          # field run-separator screenshots
                continue
            with Image.open(f) as probe:
                if probe.size in SCREENSHOT_SIZES:
                    skipped.append(f)
                    continue
            files.append(f)
        n_skipped += len(skipped)

        for path in files:
            im = Image.open(path).convert("RGB")
            canvas, r, dw, dh = letterbox(im, IMGSZ)
            x = np.asarray(canvas, dtype=np.float32) / 255.0
            x = x.transpose(2, 0, 1)[None]

            out = sess.run(None, {inp.name: x})
            boxes, conf, cid = postprocess(out, nc)

            # class-wise NMS
            kept = []
            for c in np.unique(cid):
                m = cid == c
                idx = np.flatnonzero(m)
                k = nms(boxes[m], conf[m], IOU_NMS)
                kept.extend(idx[k].tolist())

            n_images += 1
            for i in kept:
                bw = (boxes[i, 2] - boxes[i, 0]) / r
                bh = (boxes[i, 3] - boxes[i, 1]) / r
                rows.append({
                    "tree": tree,
                    "file": os.path.basename(path),
                    "class_id": int(cid[i]),
                    "class": CLASS_NAMES[int(cid[i])],
                    "conf": float(conf[i]),
                    "box_w": float(bw),
                    "box_h": float(bh),
                    "box_area": float(bw * bh),
                })

        print(f"  {tree:8s} {len(files):4d} images"
              f"   (skipped {len(skipped)} screenshots)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------ report ----
    print(f"\nImages scanned    : {n_images}")
    print(f"Screenshots skipped: {n_skipped}")
    print(f"Raw detections above {CONF_FLOOR} : {len(df)}")
    print(f"Detections written to {OUT_CSV}\n")

    if df.empty:
        print("No detections at all. Nothing fires on healthy bark.")
        return

    print("=" * 70)
    print("FALSE POSITIVES ON HEALTHY TRUNKS (all detections here are wrong)")
    print("=" * 70)

    for thr in REPORT_AT:
        sub = df[df["conf"] >= thr]
        if sub.empty:
            print(f"\n-- conf >= {thr}: no detections")
            continue
        print(f"\n-- conf >= {thr}")
        g = (sub.groupby("class")
                .agg(dets=("conf", "size"),
                     images=("file", "nunique"),
                     trees=("tree", "nunique"),
                     max_conf=("conf", "max"),
                     mean_conf=("conf", "mean"))
                .sort_values("dets", ascending=False))
        g["img_rate_%"] = (100 * g["images"] / n_images).round(1)
        print(g.to_string())

    pd_rows = df[df["class"] == "Pink_disease"]
    print("\n" + "=" * 70)
    print("PINK DISEASE SPECIFICALLY")
    print("=" * 70)
    if pd_rows.empty:
        print("Zero Pink_disease detections on healthy trunks at any confidence.")
        print("No evidence of a source shortcut from this test.")
    else:
        print(f"Detections      : {len(pd_rows)}")
        print(f"Images affected : {pd_rows['file'].nunique()} / {n_images} "
              f"({100 * pd_rows['file'].nunique() / n_images:.1f}%)")
        print(f"Trees affected  : {pd_rows['tree'].nunique()} / {len(dirs)}")
        print(f"Max confidence  : {pd_rows['conf'].max():.3f}")
        print("\nConfidence quantiles:")
        print(pd_rows["conf"].describe(
            percentiles=[.5, .75, .9, .95, .99]).to_string())
        print("\nPer tree:")
        print(pd_rows.groupby("tree")
                     .agg(dets=("conf", "size"),
                          images=("file", "nunique"),
                          max_conf=("conf", "max")).to_string())
        print("\nHighest-confidence examples:")
        print(pd_rows.nlargest(15, "conf")[
            ["tree", "file", "conf", "box_w", "box_h"]].to_string(index=False))


if __name__ == "__main__":
    main()
