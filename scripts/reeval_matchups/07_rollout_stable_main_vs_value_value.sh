#!/usr/bin/env bash
# Row 7 — Us: rollout_value_stable + MainPlayAgent | Them: Value + Value
export RUN_NAME="${RUN_NAME:-reeval_m07}"
source "$(dirname "$0")/_lib.sh"
reeval_run "07 rollout_value_stable + RL main vs ValueFunctionPlayer" \
  --placement-strategy rollout_value_stable \
  --load "${MAIN_MODEL}" \
  --enemy value
