#!/usr/bin/env bash
# Us: Random + MainPlay | Them: WeightedRandom + WeightedRandom
export RUN_NAME="${RUN_NAME:-reeval_m03}"
source "$(dirname "$0")/_lib.sh"
reeval_run "03 random placement + RL main vs WeightedRandom" \
  --placement-strategy random \
  --load "${MAIN_MODEL}" \
  --enemy weighted
