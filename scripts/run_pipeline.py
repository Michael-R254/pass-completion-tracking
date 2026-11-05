"""Run every stage end to end and reproduce all results (PLAN.md step 7).

Built up step by step; currently: events -> D1 pass labels -> results/d1_label_counts.csv.

Run:  uv run python scripts/run_pipeline.py
"""

from pathlib import Path

import pandas as pd

from passmodel.clean import label_counts, label_passes
from passmodel.load import MATCH_IDS, load_events

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    events = pd.concat([load_events(m) for m in MATCH_IDS], ignore_index=True)
    counts = label_counts(label_passes(events))
    counts.to_csv(RESULTS / "d1_label_counts.csv")
    print(counts.to_string())


if __name__ == "__main__":
    main()
