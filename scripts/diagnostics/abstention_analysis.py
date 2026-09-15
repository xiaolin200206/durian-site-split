"""
Abstention analysis — does confidence carry the signal that no covariate does?

The claim in the Discussion is that a system which always returns a confident
class is misspecified, and that abstention makes unpredictable site-level
failure visible. That is currently an argument. This makes it a result.

What it needs: the raw per-detection outputs from the by-farm folds, i.e. for
each predicted box, its confidence and whether it matched a ground-truth box
at IoU >= 0.5. Ultralytics gives you this without retraining:

    from ultralytics import YOLO
    m = YOLO('runs/fold1/weights/best.pt')
    r = m.val(data='fold1.yaml', conf=0.001, save_json=True)

conf=0.001 matters — the default 0.25 has already thrown away the low-confidence
tail, which is exactly the tail this analysis is about.

Point PRED_JSON at the resulting predictions.json and GT_JSON at the COCO-format
ground truth for the same split. Run once per fold; the interesting sites are
the low scorers (farm 2, fold 1).
"""

import json
from collections import defaultdict
import numpy as np

PRED_JSON = "runs/fold1/predictions.json"
GT_JSON = "data/fold1_gt.json"
IOU_T = 0.5


def iou(a, b):
    """COCO xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def match(preds, gts):
    """Greedy match by descending confidence. Returns (conf, tp) per prediction."""
    by_img = defaultdict(list)
    for g in gts:
        by_img[g["image_id"]].append(g)

    out = []
    used = defaultdict(set)
    for p in sorted(preds, key=lambda d: -d["score"]):
        best, best_j = 0.0, None
        for j, g in enumerate(by_img[p["image_id"]]):
            if j in used[p["image_id"]] or g["category_id"] != p["category_id"]:
                continue
            v = iou(p["bbox"], g["bbox"])
            if v > best:
                best, best_j = v, j
        hit = best >= IOU_T
        if hit:
            used[p["image_id"]].add(best_j)
        out.append((p["score"], hit))
    return out


def sweep(matched, n_gt):
    """Precision and coverage as the confidence floor rises."""
    rows = []
    for t in np.arange(0.0, 1.0, 0.05):
        kept = [(c, h) for c, h in matched if c >= t]
        if not kept:
            break
        tp = sum(h for _, h in kept)
        rows.append({
            "threshold": round(float(t), 2),
            "kept": len(kept),
            "abstained_frac": round(1 - len(kept) / len(matched), 3),
            "precision": round(tp / len(kept), 3),
            "recall": round(tp / n_gt, 3),
        })
    return rows


if __name__ == "__main__":
    preds = json.load(open(PRED_JSON))
    gt = json.load(open(GT_JSON))
    gts = gt["annotations"] if isinstance(gt, dict) else gt

    matched = match(preds, gts)
    print(f"{len(matched)} predictions, {len(gts)} ground-truth boxes\n")
    print(f"{'thr':>5} {'kept':>6} {'abstain':>8} {'prec':>6} {'recall':>7}")
    for r in sweep(matched, len(gts)):
        print(f"{r['threshold']:>5} {r['kept']:>6} {r['abstained_frac']:>8} "
              f"{r['precision']:>6} {r['recall']:>7}")

    print("""
What to look for
----------------
The claim is supported if, at the worst site, discarding a large fraction of
low-confidence predictions raises the precision of what remains substantially
— i.e. the model is not uniformly wrong there, it is wrong in a way its own
confidence already flags. If precision is flat across the sweep, confidence
carries no site-level signal and the abstention argument stays an argument.
Report the curve either way; a flat curve is a real finding and belongs in
Limitations rather than being dropped.""")
