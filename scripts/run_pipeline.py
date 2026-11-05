"""Run every stage end to end and reproduce all results (PLAN.md step 7).

Built up step by step; currently: events -> D1/D2 pass labels -> results/d1_label_counts.csv, d2_set_piece_counts.csv.

Run:  uv run python scripts/run_pipeline.py
"""

from pathlib import Path

import pandas as pd

from passmodel.clean import label_counts, label_passes, set_piece_counts
from passmodel.load import MATCH_IDS, load_events

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    events = pd.concat([load_events(m) for m in MATCH_IDS], ignore_index=True)
    labelled = label_passes(events)
    for name, table in [("d1_label_counts", label_counts(labelled)), ("d2_set_piece_counts", set_piece_counts(labelled))]:
        table.to_csv(RESULTS / f"{name}.csv")
        print(f"\n{name}\n{table.to_string()}")


if __name__ == "__main__":
    main()
