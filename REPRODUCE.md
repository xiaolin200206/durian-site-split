# Reproduce in one command

```bash
pip install -r requirements.txt     # and pandoc (apt install pandoc / brew install pandoc)
python verify_applied.py --full
```

Expected last line: `N of N checks pass`. About a minute on one CPU core; no GPU, images or weights.

What it checks, in order:

0. `scripts/applied/paper_analysis.py` re-derives `results_applied/paper_numbers.json`, every table
   CSV and every figure from the raw per-run tables, and the JSON must come out identical.
A. The headline numbers are recomputed a second time by independent code inside `verify_applied.py`
   (per-detector random-split and unseen-farm mAP50, per-farm range, Sabah, per-class retention,
   the data-budget grid, matched-budget differences, regression slopes by within-farm demeaning,
   class coverage).
B. `paper/render.py` re-renders the manuscript, highlights and cover letter from their templates;
   the result must be byte-identical to the committed files, with no unresolved placeholder.
C. Journal rules and internal consistency: abstract ≤ 250 words, 3–5 highlights ≤ 85 characters,
   1–7 keywords, every in-text citation in the reference list and every reference cited,
   references alphabetical, figures and tables numbered and first cited in order, README numbers.

Training is not part of this path. The scripts are in `scripts/applied/` (data-budget and
calibration experiments) and `scripts/clean/` (leave-one-farm-out runs); they need a GPU.
