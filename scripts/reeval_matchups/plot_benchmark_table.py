#!/usr/bin/env python3
"""Render the 8-row benchmark table (Us / Results / Them) from matchup_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# Match README / evaluation sheet (Them = same policy opening + main).
ROWS = [
    ("rollout_value_stable", "AlphaBeta(2)", "AlphaBeta(2)", "AlphaBeta(2)"),
    ("Random", "MainPlayAgent", "Random", "Random"),
    ("Random", "MainPlayAgent", "Random", "WeightedRandom"),
    ("Random", "MainPlayAgent", "Random", "Value"),
    ("rollout_value_stable", "MainPlayAgent", "Random", "Random"),
    ("rollout_value_stable", "MainPlayAgent", "WeightedRandom", "WeightedRandom"),
    ("rollout_value_stable", "MainPlayAgent", "Value", "Value"),
    ("rollout_value_stable", "MainPlayAgent", "AlphaBeta(2)", "AlphaBeta(2)"),
]

RUN_NAMES = [f"reeval_m{i:02d}" for i in range(1, 9)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("capstone_agent/benchmarks/reeval_vab123_simple_report/matchup_summary.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "capstone_agent/benchmarks/reeval_vab123_simple_report/benchmark_table_win_pct.png"
        ),
    )
    p.add_argument(
        "--title-suffix",
        default="",
        help="Extra subtitle text (e.g. model name).",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing capstone_model.pt and placement_model.pt; "
            "footer lists this folder (basename) and full path."
        ),
    )
    args = p.parse_args()

    data = json.loads(args.summary_json.read_text(encoding="utf-8"))
    by_run = {d["run_name"]: d for d in data}

    col_labels = [
        "Us\nplacement",
        "Us\nmain",
        "Us\nwin %",
        "Them\nwin %",
        "Them\nplacement",
        "Them\nmain",
    ]
    cell_text = []
    games_n = None
    for i, rn in enumerate(RUN_NAMES):
        r = by_run.get(rn)
        if not r:
            raise SystemExit(f"Missing {rn} in summary JSON")
        g = int(r["games"])
        games_n = g
        w = int(r["wins"])
        tr = int(r.get("truncations", 0) or 0)
        us_pct = 100.0 * w / g
        # Opponent wins = games not won by Us (excludes truncations from denominator sense).
        them_pct = 100.0 * (g - w - tr) / g if g else 0.0
        up, um, tp, tm = ROWS[i]
        cell_text.append(
            [
                up,
                um,
                f"{us_pct:.1f}%",
                f"{them_pct:.1f}%",
                tp,
                tm,
            ]
        )

    plt = _require_matplotlib()
    fig_w = 14.0
    fig_h = 6.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    title = "Win percentage (Blue = Us, Red = Them)"
    if games_n is not None:
        title += f"\n{games_n} completed games per row"
    if args.title_suffix:
        title += f" — {args.title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=[f"  {i + 1}  " for i in range(8)],
        loc="center",
        cellLoc="center",
        colColours=["#cfe8fc"] * 2 + ["#d4e8d4"] * 2 + ["#fcd4d4"] * 2,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_height(0.09)
        else:
            cell.set_height(0.085)

    note_lines = [
        "Them win % = (games − Us wins − truncations) / games. "
        "Sheet target is often 1000 games; use GAMES=1000 for tighter estimates.",
    ]
    if args.model_dir is not None:
        md = args.model_dir.expanduser().resolve()
        folder = md.name
        parent_folder = md.parent.name
        main_pt = md / "capstone_model.pt"
        place_pt = md / "placement_model.pt"
        note_lines.append(
            f"Weights folder: {folder}  (parent of this folder: {parent_folder})"
        )
        note_lines.append(f"Full path: {md}")
        note_lines.append(f"Main checkpoint: {main_pt.name}  |  Paired placement: {place_pt.name}")
    note = "\n".join(note_lines)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="0.25", linespacing=1.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.92])
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
