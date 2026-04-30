#!/usr/bin/env bash
# Row 8 — Us: rollout_value_stable + MainPlayAgent | Them: AlphaBeta(2) + AlphaBeta(2)
export RUN_NAME="${RUN_NAME:-reeval_m08}"
source "$(dirname "$0")/_lib.sh"
reeval_run "08 rollout_value_stable + RL main vs AlphaBeta(2)" \
  --placement-strategy rollout_value_stable \
  --load "${MAIN_MODEL}" \
  --enemy alphabeta \
  --enemy-ab-depth 2
