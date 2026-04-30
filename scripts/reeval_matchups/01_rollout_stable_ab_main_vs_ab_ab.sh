#!/usr/bin/env bash
# Row 1 — Us: rollout_value_stable + AlphaBeta(2) main | Them: AlphaBeta(2) + AlphaBeta(2)
export RUN_NAME="${RUN_NAME:-reeval_m01}"
source "$(dirname "$0")/_lib.sh"
reeval_run "01 rollout_value_stable + AB(2) main vs AlphaBeta(2)" \
  --placement-strategy rollout_value_stable \
  --main-play-mode alphabeta \
  --main-play-ab-depth 2 \
  --enemy alphabeta \
  --enemy-ab-depth 2
