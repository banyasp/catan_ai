#!/usr/bin/env bash
# Shared defaults for matchup re-evaluation. Source from sibling scripts:
#   source "$(dirname "$0")/_lib.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

: "${GAMES:=400}"
: "${MAIN_MODEL:=capstone_agent/models/capstone_model.pt}"
: "${PLACEMENT_MODEL:=capstone_agent/models/placement_model.pt}"
: "${ENEMY_MAIN_MODEL:=${MAIN_MODEL}}"
: "${BENCHMARK_CSV:=capstone_agent/benchmarks/reeval_matchups.csv}"
# Each matchup script should ``export RUN_NAME="${RUN_NAME:-reeval_mNN}"`` before sourcing.
: "${RUN_NAME:=reeval_matchups}"
: "${MAP_MODE:=fixed}"
: "${MAP_TEMPLATE:=TOURNAMENT}"
: "${FIXED_MAP_SEED:=0}"

reeval_run() {
  local title="$1"
  shift
  echo ""
  echo "================================================================"
  echo " ${title}"
  echo " Games=${GAMES}  run_name=${RUN_NAME}"
  echo "================================================================"
  python capstone_agent/run_simulation.py \
    --games "${GAMES}" \
    --run-name "${RUN_NAME}" \
    --benchmark-csv "${BENCHMARK_CSV}" \
    --map-mode "${MAP_MODE}" \
    --map-template "${MAP_TEMPLATE}" \
    --fixed-map-seed "${FIXED_MAP_SEED}" \
    "$@"
}
