#!/usr/bin/env bash
# Us: Random placement + MainPlayAgent | Them: Random + Random
export RUN_NAME="${RUN_NAME:-reeval_m02}"
source "$(dirname "$0")/_lib.sh"
reeval_run "02 random placement + RL main vs Random+Random" \
  --placement-strategy random \
  --load "${MAIN_MODEL}" \
  --enemy random
