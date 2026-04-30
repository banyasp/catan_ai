#!/usr/bin/env python3
"""Summarize re-eval matchup benchmark CSVs and write plots + tables.

Expects per-matchup ``run_name`` values (default ``reeval_m01`` … ``reeval_m08``)
written by the matchup shell scripts. Falls back to a single ``run_name`` if only
one series is present (see ``--legacy-run-name``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Default labels aligned with scripts/reeval_matchups/README.txt
DEFAULT_LABELS: dict[str, str] = {
    "reeval_m01": "01 rollout+AB(2) main vs AB+AB",
    "reeval_m02": "02 random+PPO vs Random+Random",
    "reeval_m03": "03 random+PPO vs Weighted+Weighted",
    "reeval_m04": "04 random+PPO vs Value+Value",
    "reeval_m05": "05 rollout+PPO vs Random+Random",
    "reeval_m06": "06 rollout+PPO vs Weighted+Weighted",
    "reeval_m07": "07 rollout+PPO vs Value+Value",
    "reeval_m08": "08 rollout+PPO vs AB+AB",
}


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (low, high, point_estimate) Wilson score interval for Binomial(wins, n)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denom
    rad = (z / denom) * math.sqrt(
        max(0.0, phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    )
    return (max(0.0, centre - rad), min(1.0, centre + rad), phat)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return list(header), rows


def _discover_run_names(rows: list[dict[str, str]], prefix: str) -> list[str]:
    names = sorted({str(r.get("run_name", "")).strip() for r in rows if r.get("run_name")})
    pat = re.compile(rf"^{re.escape(prefix)}\d{{2}}$")
    picked = [n for n in names if pat.match(n)]
    if picked:
        return sorted(picked, key=lambda s: int(s[len(prefix) :]))
    return []


def _aggregate_matchup(sub: list[dict[str, str]], run_name: str) -> dict[str, Any]:
    def _int(x: str, default: int = 0) -> int:
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default

    def _float(x: str) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    sub = sorted(sub, key=lambda r: _int(r.get("game_index", "0"), 0))
    n = len(sub)
    wins = sum(_int(r.get("won", "0")) for r in sub)
    losses = 0
    truncs = 0
    rewards: list[float] = []
    steps: list[float] = []
    for r in sub:
        rewards.append(_float(r.get("reward", "0")))
        steps.append(_float(r.get("steps", "0")))
        t = _int(r.get("terminated", "0"))
        w = _int(r.get("won", "0"))
        if t and not w:
            losses += 1
        truncs += _int(r.get("truncated", "0"))

    lo, hi, p = _wilson_ci(wins, n)
    mean_r = sum(rewards) / n if n and not any(math.isnan(x) for x in rewards) else float("nan")
    mean_s = sum(steps) / n if n and not any(math.isnan(x) for x in steps) else float("nan")
    return {
        "run_name": run_name,
        "games": n,
        "wins": wins,
        "losses": losses,
        "truncations": truncs,
        "win_rate": p,
        "ci_low": lo,
        "ci_high": hi,
        "mean_reward": mean_r,
        "mean_steps": mean_s,
    }


def _plot_bar(rows: list[dict[str, Any]], labels: dict[str, str], out_path: Path) -> None:
    plt = _require_matplotlib()
    names = [r["run_name"] for r in rows]
    ys = [r["win_rate"] for r in rows]
    yerr_lo = [r["win_rate"] - r["ci_low"] for r in rows]
    yerr_hi = [r["ci_high"] - r["win_rate"] for r in rows]
    xlabs = [labels.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(max(10.0, len(names) * 1.2), 5.5))
    import numpy as np

    x = np.arange(len(names))
    ax.bar(x, ys, yerr=[yerr_lo, yerr_hi], capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, rotation=22, ha="right")
    ax.set_ylabel("Blue win rate (95% Wilson CI)")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1, alpha=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Re-eval matchup summary (eval games)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_cumulative(
    frames: dict[str, list[dict[str, str]]], labels: dict[str, str], out_path: Path
) -> None:
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(sorted(frames.keys())):
        g = sorted(frames[name], key=lambda r: int(float(r.get("game_index", 0) or 0)))
        gi = [int(float(r.get("game_index", 0) or 0)) for r in g]
        w = [int(float(r.get("won", 0) or 0)) for r in g]
        cum = np.cumsum(np.array(w, dtype=float)) / np.arange(1, len(w) + 1)
        ax.plot(
            gi,
            cum,
            label=labels.get(name, name),
            color=cmap(i % 10),
            linewidth=1.6,
            alpha=0.9,
        )
    ax.set_xlabel("Game index")
    ax.set_ylabel("Cumulative win rate")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Cumulative win rate by matchup")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_reward_trailing(
    frames: dict[str, list[dict[str, str]]],
    labels: dict[str, str],
    window: int,
    out_path: Path,
) -> None:
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    minp = max(3, window // 10)
    for i, name in enumerate(sorted(frames.keys())):
        g = sorted(frames[name], key=lambda r: int(float(r.get("game_index", 0) or 0)))
        gi = np.array([int(float(r.get("game_index", 0) or 0)) for r in g], dtype=float)
        r = np.array([float(r.get("reward", 0) or 0) for r in g], dtype=float)
        roll = np.empty_like(r)
        for j in range(len(r)):
            lo = max(0, j - window + 1)
            chunk = r[lo : j + 1]
            roll[j] = float(chunk.mean()) if len(chunk) >= minp else float("nan")
        ax.plot(
            gi,
            roll,
            label=labels.get(name, name),
            color=cmap(i % 10),
            linewidth=1.4,
            alpha=0.9,
        )
    ax.set_xlabel("Game index")
    ax.set_ylabel(f"Trailing mean reward (window={window})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Reward smoothing by matchup")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="capstone_agent/benchmarks/reeval_matchups.csv",
        help="Benchmark CSV from run_simulation.py",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: <csv_dir>/reeval_report)",
    )
    p.add_argument(
        "--mode",
        choices=["eval", "train", "all"],
        default="eval",
        help="Which rows to include (matchup scripts use eval).",
    )
    p.add_argument(
        "--run-name-prefix",
        default="reeval_m",
        help="Prefix for auto-discovered matchup run_names.",
    )
    p.add_argument(
        "--run-names",
        default="",
        help="Comma-separated run_names (overrides discovery).",
    )
    p.add_argument(
        "--legacy-run-name",
        default="",
        help="If set and no prefix matches, aggregate this single run_name.",
    )
    p.add_argument("--reward-window", type=int, default=50)
    args = p.parse_args(list(argv) if argv is not None else None)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "reeval_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    header, rows = _read_csv_rows(csv_path)
    required = {"run_name", "mode", "won", "game_index", "reward", "steps"}
    missing = required - set(header)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    if args.mode != "all":
        rows = [r for r in rows if str(r.get("mode", "")).strip() == args.mode]
    if not rows:
        raise SystemExit(f"No rows after mode filter ({args.mode}).")

    if args.run_names.strip():
        run_names = [x.strip() for x in args.run_names.split(",") if x.strip()]
    else:
        run_names = _discover_run_names(rows, args.run_name_prefix)
        if not run_names and args.legacy_run_name:
            if any(str(r.get("run_name", "")).strip() == args.legacy_run_name for r in rows):
                run_names = [args.legacy_run_name]
        if not run_names:
            available = sorted({str(r.get("run_name", "")).strip() for r in rows if r.get("run_name")})
            raise SystemExit(
                "No run_names matched prefix "
                f"{args.run_name_prefix!r}. Available: {available}. "
                "Set per-script RUN_NAME (reeval_m01 …) or pass --run-names / --legacy-run-name."
            )

    by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        rn = str(r.get("run_name", "")).strip()
        if rn in run_names:
            by_name[rn].append(r)

    summary_rows: list[dict[str, Any]] = []
    for rn in run_names:
        sub = by_name.get(rn, [])
        if not sub:
            continue
        summary_rows.append(_aggregate_matchup(sub, rn))

    if not summary_rows:
        raise SystemExit("Nothing to plot after filtering run_names.")

    summary_path = out_dir / "matchup_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    csv_out = out_dir / "matchup_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r[k] for k in summary_rows[0].keys()})

    labels = dict(DEFAULT_LABELS)
    _plot_bar(summary_rows, labels, out_dir / "matchup_win_rates.png")
    if by_name:
        _plot_cumulative(dict(by_name), labels, out_dir / "matchup_cumulative_win_rate.png")
        _plot_reward_trailing(
            dict(by_name),
            labels,
            max(5, args.reward_window),
            out_dir / "matchup_reward_trailing.png",
        )

    print(f"Wrote report to {out_dir.resolve()}")
    print(f"  summary: {summary_path.name}, {csv_out.name}")
    print("  plots: matchup_win_rates.png, matchup_cumulative_win_rate.png,")
    print("         matchup_reward_trailing.png")
    for r in summary_rows:
        print(
            f"  {r['run_name']}: n={r['games']} wins={r['wins']} "
            f"win_rate={r['win_rate']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
