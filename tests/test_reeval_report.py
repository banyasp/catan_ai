"""Smoke test for scripts/reeval_matchups/generate_report.py (no catanatron game run)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "scripts" / "reeval_matchups" / "generate_report.py"
HEADER = [
    "timestamp_utc",
    "run_name",
    "mode",
    "loaded_model_path",
    "game_index",
    "games_total",
    "status",
    "won",
    "terminated",
    "truncated",
    "steps",
    "reward",
    "cum_wins",
    "cum_losses",
    "cum_truncations",
    "cum_win_rate",
    "self_seat",
    "went_first",
]


@pytest.mark.skipif(not REPORT_SCRIPT.is_file(), reason="generate_report.py missing")
def test_generate_report_writes_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")

    csv_path = tmp_path / "bench.csv"
    out_dir = tmp_path / "out"
    rows = []
    for m in ("01", "02"):
        rn = f"reeval_m{m}"
        for gi in range(1, 21):
            won = 1 if (gi + int(m)) % 3 != 0 else 0
            rows.append(
                {
                    "timestamp_utc": "2026-01-01T00:00:00Z",
                    "run_name": rn,
                    "mode": "eval",
                    "loaded_model_path": "",
                    "game_index": gi,
                    "games_total": 20,
                    "status": "WON" if won else "LOST",
                    "won": won,
                    "terminated": 1,
                    "truncated": 0,
                    "steps": 100 + gi,
                    "reward": 1.0 if won else -1.0,
                    "cum_wins": 0,
                    "cum_losses": 0,
                    "cum_truncations": 0,
                    "cum_win_rate": 0.0,
                    "self_seat": 0,
                    "went_first": 1,
                }
            )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})

    r = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--csv",
            str(csv_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    assert (out_dir / "matchup_summary.csv").is_file()
    assert (out_dir / "matchup_win_rates.png").is_file()
    with open(out_dir / "matchup_summary.csv", newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2
    names = {row["run_name"] for row in got}
    assert names == {"reeval_m01", "reeval_m02"}
