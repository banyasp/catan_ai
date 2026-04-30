Re-evaluation scripts for the Capstone 1v1 benchmark table (8 matchups).

Prereqs: run from repo root (scripts call python capstone_agent/run_simulation.py).

Environment variables (all optional):
  GAMES              default 400
  MAIN_MODEL         default capstone_agent/models/capstone_model.pt
  PLACEMENT_MODEL    default capstone_agent/models/placement_model.pt (unused for
                     rollout_value_stable / random / alphabeta placement rows)
  ENEMY_MAIN_MODEL   default same as MAIN_MODEL (only for rl-capstone opponents;
                     none of the 8 table scripts use rl-capstone)
  BENCHMARK_CSV      default capstone_agent/benchmarks/reeval_matchups.csv
  RUN_NAME           Each script defaults to a distinct value (reeval_m01 … reeval_m08)
                     so rows stay separable in the CSV and in plots. Override to force
                     one name for all runs, e.g. export RUN_NAME=my_batch (not recommended).
  REEVAL_OUT_DIR     If set, run_all.sh passes --out-dir here for generate_report.py
  MAP_MODE / MAP_TEMPLATE / FIXED_MAP_SEED  map controls

Eight rows (Us = Blue, Them = Red env opponent):

  Script  Us (placement + main)                         Them (opening + midgame)
  ------  -------------------------------------------  --------------------------
  01      rollout_value_stable + AlphaBeta(2) main     AlphaBeta(2) + AlphaBeta(2)
  02      random + MainPlayAgent (PPO)                 Random + Random
  03      random + MainPlayAgent                       WeightedRandom + WeightedRandom
  04      random + MainPlayAgent                       Value + Value
  05      rollout_value_stable + MainPlayAgent         Random + Random
  06      rollout_value_stable + MainPlayAgent         WeightedRandom + WeightedRandom
  07      rollout_value_stable + MainPlayAgent         Value + Value
  08      rollout_value_stable + MainPlayAgent         AlphaBeta(2) + AlphaBeta(2)

Simulator mapping:
- MainPlayAgent: pass --load "${MAIN_MODEL}" and --main-play-mode model (default).
- AlphaBeta main (row 1): --main-play-mode alphabeta --main-play-ab-depth 2; do not pass --load.
- Opponent bots: --enemy random | weighted | value | alphabeta (with --enemy-ab-depth 2 for AB).

Reports and plots
-----------------
After ``run_all.sh`` finishes, ``generate_report.py`` runs automatically. It reads
``BENCHMARK_CSV`` (default path above), discovers ``run_name`` values matching
``reeval_m##``, and writes under ``<csv_directory>/reeval_report/``:

  matchup_summary.json   Per-matchup wins, Wilson 95% CI, mean reward/steps
  matchup_summary.csv    Same data as CSV
  matchup_win_rates.png  Bar chart of win rates with error bars
  matchup_cumulative_win_rate.png   Cumulative win rate vs game index (one line per matchup)
  matchup_reward_trailing.png       Trailing mean reward (smoothed)

Run the report alone (e.g. after manual games):

  python scripts/reeval_matchups/generate_report.py \\
    --csv capstone_agent/benchmarks/reeval_matchups.csv \\
    --out-dir capstone_agent/benchmarks/reeval_report

Legacy CSVs that used a single ``run_name`` for every matchup:

  python scripts/reeval_matchups/generate_report.py --csv path.csv \\
    --legacy-run-name reeval_matchups

Smoke test (requires matplotlib + numpy + pytest + pytest-benchmark for repo pytest.ini):

  python -m pytest tests/test_reeval_report.py -q

Single matchup:
  ./scripts/reeval_matchups/01_rollout_stable_ab_main_vs_ab_ab.sh

All eight + report:
  ./scripts/reeval_matchups/run_all.sh
