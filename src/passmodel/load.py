"""Unified loaders for Metrica events and tracking (PLAN.md step 1).

Games 1-2 are CSV, game 3 is EPTS + JSON. Everything comes out in one schema.

Step 0 provides the raw loaders below (used by scripts/check_data.py and
notebooks/00_setup_check.ipynb); step 1 builds the unified schema on top of them.
"""

import json
from pathlib import Path

import pandas as pd
from kloppy import metrica

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MATCH_IDS = (1, 2, 3)


def game_dir(match_id: int) -> Path:
    return RAW_DIR / f"Sample_Game_{match_id}"


def load_tracking(match_id: int):
    """Load one match's tracking as a kloppy TrackingDataset from data/raw/.

    Games 1-2 are two CSVs (home, away); game 3 is EPTS (XML metadata + txt).
    We read the local files instead of ``metrica.load_open_data`` so nothing is re-downloaded.
    """
    d = game_dir(match_id)
    if match_id in (1, 2):
        return metrica.load_tracking_csv(
            home_data=d / f"Sample_Game_{match_id}_RawTrackingData_Home_Team.csv",
            away_data=d / f"Sample_Game_{match_id}_RawTrackingData_Away_Team.csv",
        )
    return metrica.load_tracking_epts(
        meta_data=d / "Sample_Game_3_metadata.xml",
        raw_data=d / "Sample_Game_3_tracking.txt",
    )


def raw_event_types(match_id: int) -> pd.DataFrame:
    """Raw Metrica event Type and Subtype for one match, same vocabulary for CSV and JSON.

    Why raw and not kloppy's ``load_event``: kloppy maps events into its own
    generic model, which hides the provider subtypes we need to define a failed
    pass (decision D1). Game 3's JSON stores subtypes as null, one dict, or a
    list of dicts; we join them as "A-B" to match the CSV's "GROUND-LOST" form.
    """
    d = game_dir(match_id)
    if match_id in (1, 2):
        return pd.read_csv(d / f"Sample_Game_{match_id}_RawEventsData.csv")[["Type", "Subtype"]]

    events = json.loads((d / "Sample_Game_3_events.json").read_text())["data"]

    def subtype(e):
        s = e.get("subtypes")
        if s is None:
            return None
        s = s if isinstance(s, list) else [s]
        return "-".join(x["name"] for x in s)

    return pd.DataFrame({"Type": [e["type"]["name"] for e in events],
                         "Subtype": [subtype(e) for e in events]})


def event_type_table() -> pd.DataFrame:
    """Counts of every event Type x Subtype, one column per match plus a total."""
    events = pd.concat({m: raw_event_types(m) for m in MATCH_IDS}, names=["match_id"]).reset_index("match_id")
    events["Subtype"] = events["Subtype"].fillna("(none)")
    return pd.crosstab([events["Type"], events["Subtype"]], events["match_id"],
                       margins=True, margins_name="total")


def tracking_summary(ds) -> dict:
    """One-line summary of a TrackingDataset: frames, frame rate, periods, players, pitch size."""
    pitch = ds.metadata.pitch_dimensions
    return {
        "frames": len(ds),
        "frame_rate_hz": ds.metadata.frame_rate,
        "periods": [p.id for p in ds.metadata.periods],
        "players_in_metadata": sum(len(t.players) for t in ds.metadata.teams),
        "pitch_m": f"{pitch.pitch_length} x {pitch.pitch_width}",
    }
