# Reproduce in one command

For editors and reviewers, and for the Code Ocean capsule.

```bash
pip install -r requirements.txt
python verify_claims.py
```

Expected final lines:

```
--------------------------------------------------------------
185 of 185 claims reproduce
every quantitative claim in the manuscript matches the released tables.
```

A few minutes on one CPU core. No GPU, no model weights, no image data.

## What it does

Each check names a claim as it appears in the manuscript, recomputes it from
`results_clean/summary_*.csv`, and compares. Use `--verbose` to print every
check rather than only failures. Use `--todo` to list claims whose input table
is absent.

The tables are themselves derived, and that derivation is also released:

```bash
python scripts/analyse/recompute_clean.py          # raw runs -> summary tables
python scripts/analyse/decomposition_robustness.py # intervals, fixed effects, roster subsets
python scripts/analyse/uncertainty_propagation.py  # effect-size and noise uncertainty
python scripts/analyse/make_figures_clean.py --root . --out figures/
```

Running `recompute_clean.py` overwrites the summary tables from the per-run
result files in `results_clean/`, so the chain from raw evaluation output to
printed number is complete and auditable. The figures are drawn from the same
summaries the text quotes.

## What is not in the capsule

Training. The 512 model runs behind `results_clean/` need GPUs and several
days. Their scripts are in `scripts/clean/` and the split manifests they
consume are released, so the runs can be reproduced by anyone with the
hardware, but nothing in the verification path depends on them.

Trained weights are not deposited: 512 checkpoints exceed practical archive
limits. They can be regenerated from the released splits and scripts, or
supplied on request.

## A second check, on the manuscript rather than the numbers

```bash
python scripts/analyse/cross_audit.py --root . \
    --manuscript manuscript_ncs_submission.md --supplementary supplementary.md
```

This one does not recompute anything. It checks that every cross-reference
resolves, that no quantity is stated with two different values in two places,
that no drafting markers survive, and that the length limits hold. It exists
because several errors in earlier drafts were of a kind `verify_claims.py`
cannot catch: not a wrong number, but the same number written differently in
the main text and the supplementary information.

`docs/checking.md` describes the three layers and what each one misses.
