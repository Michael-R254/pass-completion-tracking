"""Step 0 sanity check: confirm tracking and events load for all 3 Metrica games.

Prints, per game: tracking frame count, frame rate, first rows; then a table of
event Type x Subtype counts per game. This is only an audit to inform decision D1;
it makes no labelling decisions.

Tracking is loaded from the local files in data/raw/ with kloppy (not
``load_open_data``, which would re-download from GitHub).

Events: games 1-2 are CSV, read with pandas. Game 3 is JSON. We read it with the
stdlib ``json`` so the table shows Metrica's *raw* type names, the same
vocabulary as games 1-2. kloppy's ``load_event`` translates types into its
own generic event model, which would hide the raw subtypes we need for D1. We
also load game 3 with kloppy to confirm that loader works.

Run:  uv run python scripts/check_data.py
"""

import json
from pathlib import Path

import pandas as pd
from kloppy import metrica

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def game_dir(g: int) -> Path:
    return RAW / f"Sample_Game_{g}"


def load_tracking(g: int):
    d = game_dir(g)
    if g in (1, 2):
        return metrica.load_tracking_csv(
            home_data=d / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv",
            away_data=d / f"Sample_Game_{g}_RawTrackingData_Away_Team.csv",
        )
    return metrica.load_tracking_epts(
        meta_data=d / "Sample_Game_3_metadata.xml",
        raw_data=d / "Sample_Game_3_tracking.txt",
    )


def raw_events(g: int) -> pd.DataFrame:
    """Raw event Type/Subtype per game, in the same vocabulary for CSV and JSON."""
    d = game_dir(g)
    if g in (1, 2):
        df = pd.read_csv(d / f"Sample_Game_{g}_RawEventsData.csv")
        return df[["Type", "Subtype"]]
    events = json.loads((d / "Sample_Game_3_events.json").read_text())["data"]

    def subtype(e):
        s = e.get("subtypes")
        if s is None:
            return None
        s = s if isinstance(s, list) else [s]
        return "-".join(x["name"] for x in s)  # same "A-B" form as the CSV games

    return pd.DataFrame({"Type": [e["type"]["name"] for e in events],
                         "Subtype": [subtype(e) for e in events]})


def main() -> None:
    pd.set_option("display.width", 200, "display.max_columns", 12, "display.max_rows", 200)

    for g in (1, 2, 3):
        ds = load_tracking(g)
        df = ds.to_df()
        print(f"\n=== Game {g} tracking ===")
        print(f"frames: {len(ds):,}   frame rate: {ds.metadata.frame_rate} Hz   "
              f"periods: {[p.id for p in ds.metadata.periods]}   "
              f"players: {sum(len(t.players) for t in ds.metadata.teams)}   "
              f"pitch: {ds.metadata.pitch_dimensions.pitch_length} x {ds.metadata.pitch_dimensions.pitch_width}")
        print(f"wide df shape: {df.shape}")
        print(df.iloc[:5, :8])

    ev3 = metrica.load_event(
        event_data=game_dir(3) / "Sample_Game_3_events.json",
        meta_data=game_dir(3) / "Sample_Game_3_metadata.xml",
    )
    print(f"\nGame 3 events via kloppy.load_event: {len(ev3):,} events")

    events = pd.concat({g: raw_events(g) for g in (1, 2, 3)}, names=["game"]).reset_index("game")
    events["Subtype"] = events["Subtype"].fillna("(none)")
    table = pd.crosstab([events["Type"], events["Subtype"]], events["game"], margins=True, margins_name="total")
    print("\n=== Event Type x Subtype counts per game ===")
    print(table.to_string())


if __name__ == "__main__":
    main()
