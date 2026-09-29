#!/usr/bin/env python3
"""
render.py -- build paper/manuscript.md (and .docx) from manuscript_template.md.

Every number in the text is a placeholder {{path/to/value:fmt}} resolved from
results_applied/paper_numbers.json; every table is built from the CSVs that
paper_analysis.py writes. Nothing numeric is typed by hand.

Usage:  python paper/render.py            (from the repository root)
"""
import json
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.environ.get("PAPER_RES", os.path.join(ROOT, "results_applied"))
PAPER = os.path.join(ROOT, "paper")
LABEL = {"Algal": "Algal spot", "Leaf_rot": "Leaf rot", "Phomopsis": "*Phomopsis*",
         "Psyllid": "Psyllid", "Psyllid_damage": "Psyllid damage",
         "leaf_hopper_damage": "Leafhopper damage"}
NAMES = list(LABEL)
DET = {"yolo11n": "YOLO11n (deployed)", "yolo11s": "YOLO11s", "yolo11m": "YOLO11m",
       "yolo11l": "YOLO11l", "rtdetr-l": "RT-DETR-L", "frcnn-r50": "Faster R-CNN R50-FPN"}


def minus(s):
    return re.sub(r"(?<![\w.])-(?=\d)", "−", s)


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}


def fmt(v, spec):
    if spec and spec.startswith("|"):     # magnitude, for "fell by" / "below" phrasing
        v, spec = abs(float(v)), spec[1:]
    if spec == "w":
        return WORDS.get(int(v), str(v))
    if spec == "W":
        return WORDS.get(int(v), str(v)).capitalize()
    if spec:
        if spec.endswith("f") or spec.endswith("d") or "," in spec:
            v = format(float(v) if "f" in spec else int(v), spec)
        else:
            v = format(v, spec)
    return minus(str(v))


def lookup(N, path):
    cur = N
    for part in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif part in cur:
            cur = cur[part]
        elif part.isdigit() and int(part) in cur:
            cur = cur[int(part)]
        else:
            raise KeyError(path)
    return cur


def f3(x):
    return minus(f"{x:.3f}")


def table1():
    t = pd.read_csv(os.path.join(RES, "table1_farms.csv"))
    head = "| Farm | Images | Boxes | " + " | ".join(LABEL[c] for c in NAMES) + " |"
    sep = "|" + "---|" * (3 + len(NAMES))
    rows = [f"| {int(r.farm)} | {int(r.images)} | {int(r.boxes):,} | " +
            " | ".join(str(int(r[c])) if r[c] else "–" for c in NAMES) + " |"
            for _, r in t.iterrows()]
    tot = (f"| All | {int(t.images.sum())} | {int(t.boxes.sum()):,} | " +
           " | ".join(str(int(t[c].sum())) for c in NAMES) + " |")
    return ("**Table 1.** The eight peninsular farms: images, annotated boxes, and the number of "
            "images containing each class (a dash marks a class absent from the farm).\n\n" +
            "\n".join([head, sep] + rows + [tot]))


def table2(N):
    fw = N.get("random_split_farm_weighted", {})
    so = N.get("sabah_by_orchard", {})
    head = ("| Detector | Random split, pooled | Random split, farm-weighted | Unseen farm | "
            "Loss (%) | Sabah orchard 1 | Sabah orchard 2 |")
    sep = "|---|---|---|---|---|---|---|"
    rows = []
    for m in DET:
        h = N["headline"][m]
        o = so.get(m, {})
        cell = lambda x: f3(x) if x is not None else "–"
        rows.append(f"| {DET[m]} | {f3(h['random_split'])} | {cell(fw.get(m))} | "
                    f"{f3(h['unseen_farm'])} | {h['drop_pct']:.1f} | {cell(o.get('sabah_o1'))} | "
                    f"{cell(o.get('sabah_o2'))} |")
    return ("**Table 2.** mAP50 on a random 80/20 split and on farms never seen in training "
            "(leave-one-farm-out, farms weighted equally), and on the two Sabah orchards "
            f"({N['sabah_orchard_images']['sabah_o1']} and {N['sabah_orchard_images']['sabah_o2']} "
            "images). The pooled random-split figure is computed over all test images; the "
            "farm-weighted figure scores each farm's test images separately and averages farms "
            "with equal weight, as for the unseen-farm figure. Loss compares the pooled random "
            "split with the unseen farm and is computed from unrounded values. Faster R-CNN was "
            "not scored on Sabah or per farm on the random split.\n\n" + "\n".join([head, sep] + rows))


def table3(N):
    t = pd.read_csv(os.path.join(RES, "table2_per_class.csv")).set_index("class")
    head = ("| Class | Farms carrying it | Random split | Unseen farm (range over farms) | "
            "Retained (%) | Retained, farms with ≥ 10 images (%) |")
    sep = "|---|---|---|---|---|---|"
    rows = []
    for c in NAMES:
        r = t.loc[c]
        s = N["per_class_sensitivity_min10"][c]
        rows.append(f"| {LABEL[c]} | {int(r.farms)} | {f3(r.random_split)} | {f3(r.unseen_farm)} "
                    f"({f3(r.unseen_min)}–{f3(r.unseen_max)}) | {r.retained_pct:.0f} | "
                    f"{s['retained_pct']:.0f} ({s['farms']} farms) |")
    return ("**Table 3.** Per-class AP50 on a random split and on unseen farms, averaged over the "
            "five detectors that report per-class AP. Unseen-farm AP is averaged over the farms "
            "that carry the class; the range is over those farms. Retained is the mean over "
            "detectors of each detector's unseen-farm/random-split ratio, so it can differ "
            "slightly from the ratio of the column means; the last column repeats the calculation "
            "using only farms with at least 10 images of the class.\n\n" + "\n".join([head, sep] + rows))


def table4():
    t = pd.read_csv(os.path.join(RES, "table3_budget_grid.csv"), dtype={"m": str})
    head = ("| Training farms | Photos per farm | Training images | Unseen-farm mAP50 [90% CI] | "
            "Worst–best farm | Class coverage | Sabah mAP50 |")
    sep = "|---|---|---|---|---|---|---|"
    rows = [f"| {int(r.k)} | {r.m} | {int(r.train_images)} | {f3(r.unseen_farm_mAP50)} "
            f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}] | {f3(r.worst_farm)}–{f3(r.best_farm)} | "
            f"{r.class_coverage:.2f} | {f3(r.sabah_mAP50)} |" for r in t.itertuples()]
    return ("**Table 4.** Data-budget experiment (YOLO11n, about 2,000 iterations per run). Each "
            "row averages two farm draws (one for seven farms) and the seeds within each held-out "
            "farm, and then the eight held-out farms with equal weight. Class coverage is the share "
            "of the held-out farm's annotated images whose classes all occur in the images drawn "
            "for training. Sabah is the mean of the two orchards' scores.\n\n" + "\n".join([head, sep] + rows))


def table5():
    e = pd.read_csv(os.path.join(RES, "table4_equal_budget.csv"))
    r = pd.read_csv(os.path.join(RES, "table5_regression.csv"))
    a = ["| More farms | Fewer farms | Images | mAP50 | Difference [90% CI] | Held-out farms where more farms won |",
         "|---|---|---|---|---|---|"]
    for x in e.itertuples():
        a.append(f"| {x.more_farms.replace(' x ', ' × ')} | {x.fewer_farms.replace(' x ', ' × ')} | "
                 f"{x.images_more} / {x.images_fewer} | {f3(x.mAP_more)} / {f3(x.mAP_fewer)} | "
                 f"{f3(x.diff)} [{f3(x.ci_lo)}, {f3(x.ci_hi)}] | {x.farms_more_wins} of {x.farms} |")
    b = ["| Runs included | n | Doubling farms [90% CI] | Doubling images per farm [90% CI] | Difference [90% CI] |",
         "|---|---|---|---|---|"]
    for x in r.itertuples():
        b.append(f"| {x.model} | {x.n_runs} | {f3(x.doubling_farms)} [{f3(x.farms_lo)}, {f3(x.farms_hi)}] | "
                 f"{f3(x.doubling_photos)} [{f3(x.photos_lo)}, {f3(x.photos_hi)}] | "
                 f"{f3(x.diff)} [{f3(x.diff_lo)}, {f3(x.diff_hi)}] |")
    return ("**Table 5.** (A) Matched-budget comparisons, paired by held-out farm. (B) Change in "
            "unseen-farm mAP50 per doubling of training farms and of images per farm, from a "
            "regression with held-out-farm fixed effects; intervals resample held-out farms.\n\n"
            "*(A) Matched budgets*\n\n" + "\n".join(a) + "\n\n*(B) Regression slopes*\n\n" + "\n".join(b))


def table6(N):
    path = os.path.join(RES, "table6_calibration.csv")
    if not os.path.isfile(path):
        return "**Table 6.** [MISSING calibration results]"
    t = pd.read_csv(path, dtype={"m": str})
    head = ("| Calibration | Images from the new farm | mAP50 on the test half | Change [90% CI] | "
            "Farms improved |")
    sep = "|---|---|---|---|---|"
    name = {"ft": "Fine-tune", "rt": "Retrain with images added"}
    rows = [f"| None | 0 | {f3(N['rq3']['base_mAP50'])} | – | – |"]
    for r in t.itertuples():
        lo, hi = N["rq3"].get("all_images_range") or (0, 0)
        img = f"all ({lo}–{hi}; mean {r.images:.0f})" if str(r.m) == "all" else f"{r.images:.0f}"
        rows.append(f"| {name[r.arm]} | {img} | {f3(r.mAP50)} | {f3(r.gain)} "
                    f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}] | {r.farms_improved} of {r.farms} |")
    return ("**Table 6.** New-farm calibration (YOLO11n). Each farm's images are split by capture "
            "time into a calibration half and a test half, in both directions; changes are paired "
            "with the uncalibrated model on the same test half and averaged over directions and "
            "seeds within a farm, then over farms.\n\n" + "\n".join([head, sep] + rows))


def prose_blocks():
    """Result-dependent paragraphs live in prose_blocks.md, one per <!-- block: name -->."""
    p = os.path.join(PAPER, "prose_blocks.md")
    if not os.path.isfile(p):
        return {}
    txt = open(p, encoding="utf-8").read()
    parts = re.split(r"<!-- block: (\w+) -->\n", txt)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def render():
    N = json.load(open(os.path.join(RES, "paper_numbers.json")))
    meta = os.path.join(PAPER, "metadata.json")
    if os.path.isfile(meta):
        N.update(json.load(open(meta)))
    s = open(os.path.join(PAPER, "manuscript_template.md"), encoding="utf-8").read()
    blocks = prose_blocks()
    for _ in range(2):
        s = re.sub(r"\{\{(\w+)\}\}", lambda m: blocks.get(m.group(1), m.group(0)), s)
    missing = []

    def sub(m):
        path, spec = m.group(1).strip(), m.group(2)
        try:
            return fmt(lookup(N, path), spec)
        except (KeyError, IndexError, TypeError):
            missing.append(path)
            return "**[MISSING " + path + "]**"
    s = re.sub(r"\{\{([^}:]+?)(?::([^}]+))?\}\}", sub, s)
    tables = {"table1": table1(), "table2": table2(N), "table3": table3(N),
              "table4": table4(), "table5": table5(), "table6": table6(N) if N.get("rq3") else
              "**Table 6.** [MISSING calibration results]"}
    s = re.sub(r"\[\[TABLE:(\w+)\]\]", lambda m: tables[m.group(1)], s)
    if missing:
        sys.exit(f"unresolved placeholders: {missing}")
    out = os.path.join(PAPER, "manuscript.md")
    open(out, "w", encoding="utf-8").write(s)
    return out


def to_docx(md, name):
    ref = os.path.join(PAPER, "reference.docx")
    cmd = ["pandoc", md, "-o", os.path.join(PAPER, name), "--resource-path", PAPER,
           "--from", "markdown+pipe_tables"]
    if os.path.isfile(ref):
        cmd += ["--reference-doc", ref]
    subprocess.run(cmd, check=True)
    postprocess(os.path.join(PAPER, name))


def postprocess(path):
    """Times New Roman 12 pt, 1.5 line spacing, continuous line numbers, page numbers."""
    import docx
    from docx.enum.text import WD_LINE_SPACING
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    d = docx.Document(path)
    for st in d.styles:
        try:
            if st.type == 1:
                st.font.name = "Times New Roman"
                rpr = st.element.get_or_add_rPr()
                rf = rpr.find(qn("w:rFonts"))
                if rf is None:
                    rf = OxmlElement("w:rFonts"); rpr.append(rf)
                for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                    rf.set(qn(a), "Times New Roman")
        except Exception:
            pass
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in [s.name for s in d.styles]:
            st = d.styles[name]
            st.font.size = docx.shared.Pt(12 if name != "Compact" else 10)
            st.paragraph_format.line_spacing = 1.5 if name != "Compact" else 1.0
    for sec in d.sections:
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
        sec._sectPr.append(ln)
        p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        p.alignment = 1
        run = p.add_run()
        for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); run._r.append(e)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
                run._r.append(e)
    d.save(path)


def render_simple(template, outname, docx_name=None):
    N = json.load(open(os.path.join(RES, "paper_numbers.json")))
    meta = os.path.join(PAPER, "metadata.json")
    if os.path.isfile(meta):
        N.update(json.load(open(meta)))
    s = open(os.path.join(PAPER, template), encoding="utf-8").read()
    blocks = prose_blocks()
    for _ in range(2):
        s = re.sub(r"\{\{(\w+)\}\}", lambda m: blocks.get(m.group(1), m.group(0)), s)
    missing = []

    def sub(m):
        try:
            return fmt(lookup(N, m.group(1).strip()), m.group(2))
        except (KeyError, IndexError, TypeError):
            missing.append(m.group(1))
            return "[MISSING]"
    s = re.sub(r"\{\{([^}:]+?)(?::([^}]+))?\}\}", sub, s)
    if missing:
        sys.exit(f"{template}: unresolved {missing}")
    p = os.path.join(PAPER, outname)
    open(p, "w", encoding="utf-8").write(s)
    if docx_name:
        subprocess.run(["pandoc", p, "-o", os.path.join(PAPER, docx_name)], check=True)
    return p


if __name__ == "__main__":
    md = render()
    to_docx(md, "manuscript.docx")
    render_simple("highlights_template.md", "highlights.md", "highlights.docx")
    render_simple("cover_letter_template.md", "cover_letter.md", "cover_letter.docx")
    subprocess.run(["pandoc", os.path.join(PAPER, "declaration_of_interest.md"), "-o",
                    os.path.join(PAPER, "declaration_of_interest.docx")], check=True)
    print("wrote manuscript, highlights, cover letter, declaration")
