"""Download the 3 Metrica Sports sample games into data/raw/ (PLAN.md step 0).

Source: https://github.com/metrica-sports/sample-data (anonymised, public).
Files already on disk are skipped, so re-running is cheap. Each file is
downloaded to a ``.part`` file and renamed only when complete, so an
interrupted download is never mistaken for a finished one.

Run:  uv run python scripts/download_data.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = [
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
]


def main() -> None:
    for rel in FILES:
        dest = RAW_DIR / rel
        if dest.exists():
            print(f"skip  {rel} (already downloaded)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"get   {rel} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{rel}", tmp)
        tmp.rename(dest)
        print(f"      {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
