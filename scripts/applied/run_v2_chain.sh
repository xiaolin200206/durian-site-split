#!/usr/bin/env bash
# Full fixed-iteration (v2) chain, seed 42. Run inside tmux or with nohup:
#   nohup bash run_v2_chain.sh > v2_chain.log 2>&1 &
#   tail -f v2_chain.log
# Safe to restart: every step skips work that is already finished under the v2 protocol.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=/root/autodl-tmp/durian
R=$ROOT/results_applied
mkdir -p "$R"
META=$(find "$ROOT" -maxdepth 3 -name metadata_backup.csv | head -1 || true)

step() { echo; echo "=================== $(date '+%F %T')  $*"; }

step "1/7 re-evaluate clean leave-one-farm-out models on each Sabah orchard (independent, quick)"
python3 extra_eval_clean.py --root "$ROOT"

step "2/7 build data-budget design (fixed iterations)"
python3 farm_budget.py --root "$ROOT" --step build 2>&1 | tee "$R/fb2_build.log"

step "3/7 train data-budget runs, seed 42 (the long step)"
python3 farm_budget.py --root "$ROOT" --step train --seeds 42 --yes

step "4/7 evaluate data-budget runs"
python3 farm_budget.py --root "$ROOT" --step eval --seeds 42

step "5/7 build new-farm calibration design  (meta: ${META:-none, camera counter used})"
if [ -n "$META" ]; then
  python3 new_farm_calibration.py --root "$ROOT" --step build --meta "$META"
else
  python3 new_farm_calibration.py --root "$ROOT" --step build
fi

step "6/7 train calibration runs from the seven-farm v2 models"
python3 new_farm_calibration.py --root "$ROOT" --step train --ft-seeds 42 --rt-seeds 42 --yes
python3 new_farm_calibration.py --root "$ROOT" --step eval  --ft-seeds 42 --rt-seeds 42

step "7/7 bundle results"
cp "$ROOT/fb2_splits/design.json" "$R/farm_budget_design.json"
cp "$ROOT/nf2_splits/design.json" "$R/new_farm_calibration_design.json" 2>/dev/null || true
cd "$R"
zip -q -j results_v2_bundle.zip farm_budget.csv farm_budget_train_lists.csv farm_budget_design.csv \
    farm_budget_design.json fb2_build.log new_farm_calibration.csv \
    new_farm_calibration_design.json extra_eval_clean.csv
echo; echo "ALL DONE. Send back: $R/results_v2_bundle.zip"
