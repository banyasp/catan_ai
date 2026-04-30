#!/usr/bin/env bash
# Row 5 — Us: rollout_value_stable + MainPlayAgent | Them: Random + Random
export RUN_NAME="${RUN_NAME:-reeval_m05}"
source "$(dirname "$0")/_lib.sh"
reeval_run "05 rollout_value_stable + RL main vs Random" \
  --placement-strategy rollout_value_stable \
  --load "${MAIN_MODEL}" \
  --enemy random
