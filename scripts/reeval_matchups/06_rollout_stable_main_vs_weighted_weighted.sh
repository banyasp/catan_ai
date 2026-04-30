#!/usr/bin/env bash
# Row 6 — Us: rollout_value_stable + MainPlayAgent | Them: WeightedRandom + WeightedRandom
export RUN_NAME="${RUN_NAME:-reeval_m06}"
source "$(dirname "$0")/_lib.sh"
reeval_run "06 rollout_value_stable + RL main vs WeightedRandom" \
  --placement-strategy rollout_value_stable \
  --load "${MAIN_MODEL}" \
  --enemy weighted
