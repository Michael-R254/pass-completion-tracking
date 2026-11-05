"""Checks on the processed data: direction flip and event/tracking sync.

The flip test runs on a tiny synthetic table. The data tests need the pipeline output
(`uv run python scripts/run_pipeline.py`) and are skipped if it isn't there.
"""

import numpy as np
import pandas as pd
import pytest

from passmodel.clean import FPS, clean_tracking
from passmodel.load import MATCH_IDS, REPO_ROOT, tracking_path

PASSES = REPO_ROOT / "data" / "processed" / "passes.parquet"
needs_data = pytest.mark.skipif(not PASSES.exists(), reason="run scripts/run_pipeline.py first")


def _kickoff_mean_x(tracking: pd.DataFrame) -> pd.DataFrame:
    """Mean x per period and team over the first second of each period."""
    first = tracking.groupby("period")["frame"].transform("min")
    start = tracking[(tracking["frame"] < first + FPS) & (tracking["team"] != "ball")]
    return start.groupby(["period", "team"], observed=True)["x"].mean().unstack()


def test_flip_applied_exactly_once_per_half():
    """Home starts on the right in half 1 and the left in half 2 (raw 0-1 units).

    After cleaning, home must defend -x in BOTH halves: flipping zero times or twice in a half would leave
    home on the +x side there.
    """
    rows = []
    for period, home_x, frame0 in [(1, 0.8, 0), (2, 0.2, 100)]:
        for f in range(frame0, frame0 + 30):
            rows += [(period, f, "Player1", "home", home_x, 0.5), (period, f, "Player2", "away", 1 - home_x, 0.5)]
    raw = pd.DataFrame(rows, columns=["period", "frame", "player_id", "team", "x", "y"])
    mean_x = _kickoff_mean_x(clean_tracking(raw, match_id=0))
    assert (mean_x["home"] < 0).all() and (mean_x["away"] > 0).all()
    assert np.allclose(mean_x["home"], -0.3 * 105)


@needs_data
@pytest.mark.parametrize("match_id", MATCH_IDS)
def test_stored_tracking_home_attacks_plus_x_in_both_halves(match_id):
    first_seconds = pd.read_parquet(tracking_path(match_id), filters=[("frame", "<", 25 * 60 * 100)])
    mean_x = _kickoff_mean_x(first_seconds)
    assert (mean_x["home"] < 0).all() and (mean_x["away"] > 0).all(), mean_x


@needs_data
@pytest.mark.parametrize("match_id", MATCH_IDS)
def test_event_start_matches_tracked_ball(match_id):
    """Events and tracking are synced and in the same coordinates: median gap under 1 m per game."""
    p = pd.read_parquet(PASSES, filters=[("match_id", "==", match_id)])
    assert p["ball_to_event_m"].median() < 1.0


@needs_data
@pytest.mark.parametrize("match_id", MATCH_IDS)
def test_passer_is_at_the_ball(match_id):
    """Guards player-ID mapping (incl. the game 3 ID swap): after drops no passer is > 5 m from the ball,
    and the median is well under 1 m."""
    p = pd.read_parquet(PASSES, filters=[("match_id", "==", match_id)])
    assert p["passer_to_ball_m"].max() <= 5.0
    assert p["passer_to_ball_m"].median() < 0.5
