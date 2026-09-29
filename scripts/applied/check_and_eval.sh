#!/usr/bin/env bash
# Check what actually ran on the server, then evaluate the fixed-iteration (v2) weights.
# Usage (from the folder holding the three .py files):
#   bash check_and_eval.sh            # diagnose + evaluate
#   bash check_and_eval.sh --check    # diagnose only
set -u
cd "$(dirname "$0")"
ROOT=/root/autodl-tmp/durian
R=$ROOT/results_applied
REPORT=$R/diag_v2.txt
mkdir -p "$R"

{
echo "=== $(date)"
echo "--- scripts in use"
grep -H '^PROTOCOL' farm_budget.py new_farm_calibration.py
grep -c 'def load_results' farm_budget.py | sed 's/^/load_results defined: /'

echo "--- trained weights (last.pt)"
for d in fb_runs fb2_runs nf_runs nf2_runs; do
  n=$(ls "$ROOT/$d"/*/weights/last.pt 2>/dev/null | wc -l)
  echo "$d: $n"
done

echo "--- design used for fb2 (should say v2-fixed-iterations, 168 jobs)"
python3 - <<EOF
import json, os
f = "$ROOT/fb2_splits/design.json"
if os.path.isfile(f):
    d = json.load(open(f)); j = d["jobs"]
    print("protocol:", d.get("protocol"), "| jobs:", len(j),
          "| iterations", min(x["iterations"] for x in j), "-", max(x["iterations"] for x in j))
else:
    print("MISSING", f)
EOF

echo "--- epochs actually trained, smallest and largest budget (v2: ~95 and ~91 epochs, batch 32; old capped: 300 and 100)"
for run in fb_h0_k1_m15_d0_s42 fb_h0_k7_mall_d0_s42; do
  for d in fb_runs fb2_runs; do
    a=$ROOT/$d/$run/args.yaml
    [ -f "$a" ] && echo "$d/$run: $(grep -E '^(epochs|batch|close_mosaic):' "$a" | tr '\n' ' ')"
  done
done

echo "--- results files and whether rows carry a protocol"
for f in farm_budget.csv new_farm_calibration.csv extra_eval_clean.csv farm_budget_train_lists.csv; do
  if [ -f "$R/$f" ]; then
    echo "$f: $(($(wc -l < "$R/$f") - 1)) rows | header: $(head -1 "$R/$f" | cut -c1-80)"
  else
    echo "$f: MISSING"
  fi
done

echo "--- Sabah file names (orchard split needs names starting o1t / o2t)"
S=$(find "$ROOT" -type d -name merged_sabah -not -path '*runs*' | head -1)
if [ -n "$S" ]; then
  ls "$S/images" | head -3
  echo "o1: $(ls "$S/images" | grep -c '^o1t')  o2: $(ls "$S/images" | grep -c '^o2t')  total: $(ls "$S/images" | wc -l)"
else
  echo "merged_sabah not found"
fi
} 2>&1 | tee "$REPORT"

[ "${1:-}" = "--check" ] && { echo; echo "Diagnosis written to $REPORT"; exit 0; }

n=$(ls "$ROOT"/fb2_runs/*/weights/last.pt 2>/dev/null | wc -l)
if [ "$n" -lt 336 ]; then
  echo; echo "Only $n of 336 fixed-iteration runs have weights. Not evaluating."
  echo "Send $REPORT back before doing anything else."
  exit 1
fi

echo; echo "=== evaluating fixed-iteration runs (old-protocol rows are moved aside automatically)"
python3 farm_budget.py --root "$ROOT" --step eval          || exit 1
python3 new_farm_calibration.py --root "$ROOT" --step eval || exit 1
python3 extra_eval_clean.py --root "$ROOT"                  || exit 1

cp "$ROOT/fb2_splits/design.json" "$R/farm_budget_design.json" 2>/dev/null
cd "$R" && zip -q -j results_v2_bundle.zip farm_budget.csv farm_budget_train_lists.csv \
    farm_budget_design.csv farm_budget_design.json new_farm_calibration.csv \
    extra_eval_clean.csv diag_v2.txt 2>/dev/null
echo; echo "Done. Send back: $R/results_v2_bundle.zip"
