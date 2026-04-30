#!/usr/bin/env bash
# Us: Random + MainPlay | Them: Value + Value
export RUN_NAME="${RUN_NAME:-reeval_m04}"
source "$(dirname "$0")/_lib.sh"
reeval_run "04 random placement + RL main vs ValueFunctionPlayer" \
  --placement-strategy random \
  --load "${MAIN_MODEL}" \
  --enemy value
