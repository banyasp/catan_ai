#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
cd "${ROOT}"
for s in \
  01_rollout_stable_ab_main_vs_ab_ab.sh \
  02_random_main_vs_random_random.sh \
  03_random_main_vs_weighted_weighted.sh \
  04_random_main_vs_value_value.sh \
  05_rollout_stable_main_vs_random_random.sh \
  06_rollout_stable_main_vs_weighted_weighted.sh \
  07_rollout_stable_main_vs_value_value.sh \
  08_rollout_stable_main_vs_ab_ab.sh
do
  bash "${DIR}/${s}"
done
echo "All matchup scripts finished."
BC="${BENCHMARK_CSV:-capstone_agent/benchmarks/reeval_matchups.csv}"
OUT_DIR="${REEVAL_OUT_DIR:-}"
if [[ -n "${OUT_DIR}" ]]; then
  python "${DIR}/generate_report.py" --csv "${BC}" --out-dir "${OUT_DIR}"
else
  python "${DIR}/generate_report.py" --csv "${BC}"
fi
