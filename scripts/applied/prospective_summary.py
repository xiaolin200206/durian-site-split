"""Summarise the prospective test from prospective/score.log and the Seremban labels.

Run from the repository root:
    python scripts/applied/prospective_summary.py
Writes results_applied/prospective_seremban_summary.json.
"""
import re, statistics as st, collections, pathlib, json
log = pathlib.Path("prospective/score.log").read_text(encoding="utf-8", errors="ignore")
rows = re.findall(r"\[(\d+)/200\]\s+(\S+)\s+fold(\d+)\s+s(\d+)\s+mAP50=([\d.]+)", log)
print("parsed", len(rows))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for i, fam, fold, seed, m in rows:
    by[fam][int(fold)].append(float(m))
PAPER = {"yolo11n": (0.428, 0.223), "yolo11s": (0.462, 0.214), "yolo11m": (0.465, 0.233),
         "yolo11l": (0.472, 0.236), "rtdetr-l": (0.499, 0.221)}
out = {}
for fam in ["yolo11n","yolo11s","yolo11m","yolo11l","rtdetr-l"]:
    folds = {f: st.mean(v) for f, v in by[fam].items()}
    allv = [x for v in by[fam].values() for x in v]
    mean = st.mean(folds.values())
    seed_sd = st.mean(st.pstdev(v) for v in by[fam].values())
    out[fam] = dict(mean=mean, fold_min=min(folds.values()), fold_max=max(folds.values()),
                    n_models=len(allv), fold_sd=st.pstdev(list(folds.values())), seed_sd_within_fold=seed_sd,
                    folds={f: round(v,4) for f,v in sorted(folds.items())})
    rp, lofo = PAPER[fam]
    print(f"{fam:9s} mean={mean:.4f} folds {min(folds.values()):.3f}-{max(folds.values()):.3f} n={len(allv)} | LOFO {lofo} random {rp} | ratio random/seremban={rp/mean:.2f} | seremban-LOFO={mean-lofo:+.3f} | fold sd {out[fam]['fold_sd']:.3f} seed sd {seed_sd:.3f}")
means = {f: o["mean"] for f, o in out.items()}
print("detector gap", max(means.values())-min(means.values()), "five-detector mean", st.mean(means.values()))
# per-fold table across families
print("fold-by-family:")
for f in range(8):
    print(f, [round(out[fam]['folds'][f],3) for fam in out])
json.dump(out, open("results_applied/prospective_seremban_summary.json","w"), indent=1)

# labels
lab = pathlib.Path("prospective/seremban_640/labels/test")
names = ["Algal","Leaf_rot","Phomopsis","Psyllid","Psyllid_damage","leaf_hopper_damage"]
img_per_cls = collections.Counter(); box_per_cls = collections.Counter(); nbox=0; nimg=0; empty=0
trees = collections.Counter()
for p in sorted(lab.glob("*.txt")):
    nimg += 1
    trees[p.name.split("_")[0]] += 1
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if not lines: empty += 1
    cls = set()
    for l in lines:
        c = int(float(l.split()[0])); box_per_cls[c]+=1; nbox+=1; cls.add(c)
    for c in cls: img_per_cls[c]+=1
print("images", nimg, "boxes", nbox, "empty", empty, "trees", dict(trees))
for i,n in enumerate(names): print(f"  {n:20s} images {img_per_cls[i]:3d} boxes {box_per_cls[i]}")
imgs = list(pathlib.Path("prospective/seremban_640/images/test").glob("*.jpg"))
print("image files", len(imgs))
