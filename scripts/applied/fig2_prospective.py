"""Figure 2 with the Seremban prospective farm added to panel (a).

Run from the repository root after scripts/applied/prospective_summary.py:
    python scripts/applied/fig2_prospective.py
"""
import sys, os, json
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "applied"))
import paper_analysis as PA
from paper_analysis import style, panel, ACCENT, ACCENT2, INK, INK2, MUTED, GRID, SURF, LABEL
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(REPO, "results_applied")
N = json.load(open(os.path.join(R, "paper_numbers.json")))
w = pd.read_csv(os.path.join(R, "tableS_per_farm_detectors.csv")).set_index("farm")
t2 = pd.read_csv(os.path.join(R, "table2_per_class.csv")).set_index("class")
S = json.load(open(os.path.join(R, "prospective_seremban_summary.json")))
per_farm = w.mean(axis=1)
rs = np.mean([v["random_split"] for v in N["headline"].values()])
rfw = np.mean(list(N["random_split_farm_weighted"].values()))

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
fig.patch.set_facecolor(SURF); style(a); style(b, ygrid=False)
order = per_farm.sort_values().index
x = np.arange(len(order))
a.axhline(rs, color=INK2, lw=1, ls="--", label=f"random split, pooled ({rs:.2f})")
a.axhline(rfw, color=INK2, lw=1, ls=":", label=f"random split, farm-weighted ({rfw:.2f})")
for m in w.columns:
    a.plot(x, w.loc[order, m], "o", ms=3.5, color=MUTED, alpha=.8, mec="none", zorder=2)
a.plot(x, per_farm[order], "D", ms=6.5, color=ACCENT, mec=SURF, mew=1.5, zorder=3, label="held-out farm, mean of six detectors")
a.plot([], [], "o", ms=3.5, color=MUTED, label="each detector")
# Seremban prospective farm
xs = len(order) + 0.7
sv = [S[f]["mean"] for f in ["yolo11n","yolo11s","yolo11m","yolo11l","rtdetr-l"]]
a.plot([xs]*5, sv, "o", ms=3.5, color=MUTED, alpha=.8, mec="none", zorder=2)
a.plot([xs], [np.mean(sv)], "*", ms=10, color=ACCENT2, mec=SURF, mew=1.0, zorder=4,
       label="Seremban, prospective, mean of five detectors")
a.axvline(len(order) - 0.15, color=GRID, lw=0.8)
a.set_xticks(list(x) + [xs], [f"F{f}" for f in order] + ["Ser."], fontsize=7.5)
a.set_xlim(-0.6, xs + 0.6)
a.set_ylim(0, 0.6)
a.set_ylabel("mAP50", fontsize=8, color=INK2)
a.set_xlabel("Held-out farm (ordered by difficulty); Ser. = farm collected afterwards", fontsize=7.5, color=INK2)
a.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=1.0, fontsize=6.4, loc="upper left", labelcolor=INK2, handlelength=1.8, borderaxespad=0.2)
panel(a, "a", "Each farm, scored by models that never saw it")

t = t2.sort_values("retained_pct"); y = np.arange(len(t))
for i, (c, r) in enumerate(t.iterrows()):
    b.plot([r.unseen_farm, r.random_split], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
b.plot(t.random_split, y, "o", ms=6, mfc=SURF, mec=INK2, mew=1.3, zorder=2, label="random split")
b.plot(t.unseen_farm, y, "o", ms=6.5, color=ACCENT, mec=SURF, mew=1.2, zorder=3, label="unseen farm")
for i, (c, r) in enumerate(t.iterrows()):
    b.annotate(f"{r.retained_pct:.0f}%", (r.random_split, i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
b.set_yticks(y, [LABEL[c] for c in t.index], fontsize=7.5)
b.set_xlim(0, 0.72); b.set_xlabel("AP50 (mean of five detectors)", fontsize=8, color=INK2)
b.grid(axis="x", color=GRID, lw=0.7)
b.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK2)
panel(b, "b", "AP kept on a new farm, by class")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(R, f"fig2_new_farm_cost.{ext}"), facecolor=SURF, dpi=300)
print("ok", rs, rfw, np.mean(sv))
