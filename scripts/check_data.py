"""Step 0 sanity check: confirm tracking and events load for all 3 Metrica games.

Prints, per game: tracking summary (frames, frame rate, ...) and first rows; then
event Type x Subtype counts per game. Audit only, to inform decision D1.
The same checks, viewable in Jupyter: notebooks/00_setup_check.ipynb.

Run:  uv run python scripts/check_data.py
"""

import pandas as pd
from kloppy import metrica

from passmodel.load import MATCH_IDS, event_type_table, game_dir, load_tracking, tracking_summary


def main() -> None:
    pd.set_option("display.width", 200, "display.max_columns", 12, "display.max_rows", 200)

    for m in MATCH_IDS:
        ds = load_tracking(m)
        df = ds.to_df()
        print(f"\n=== Game {m} tracking ===")
        print(tracking_summary(ds))
        print(f"wide df shape: {df.shape}")
        print(df.iloc[:5, :8])

    ev3 = metrica.load_event(
        event_data=game_dir(3) / "Sample_Game_3_events.json",
        meta_data=game_dir(3) / "Sample_Game_3_metadata.xml",
    )
    print(f"\nGame 3 events via kloppy.load_event: {len(ev3):,} events")

    print("\n=== Event Type x Subtype counts per game ===")
    print(event_type_table().to_string())


if __name__ == "__main__":
    main()
