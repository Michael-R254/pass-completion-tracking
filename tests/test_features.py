"""Geometry helpers and leakage checks for features.py."""

import numpy as np
import pandas as pd
import pytest

from passmodel.clean import add_velocities
from passmodel.features import (FEATURES, PITCH_CONTROL, angle_deg, pitch_control, build_features, candidate_features, fold_targets,
                                pass_frame_arrays, point_to_segment, read_pass_frames)
from passmodel.load import REPO_ROOT

PASSES = REPO_ROOT / "data" / "processed" / "passes.parquet"
needs_data = pytest.mark.skipif(not PASSES.exists(), reason="run scripts/run_pipeline.py first")


def test_point_to_segment():
    # segment from (0,0) to (10,0)
    d, t = point_to_segment(np.array([5, -3, 13]), np.array([2, 0, 4]), 0, 0, 10, 0)
    assert np.allclose(d, [2, 3, 5])        # above the middle; behind A; beyond B (3-4-5 triangle)
    assert np.allclose(t, [0.5, -0.3, 1.3])  # only the first is "between" passer and target


def test_angle_deg():
    assert np.allclose(angle_deg(np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])), [0, 90, 180, 90])


def test_velocity_uses_only_past_frames():
    """Changing positions AFTER frame t must not change the velocity AT frame t."""
    frames = np.arange(20)
    t = pd.DataFrame({"period": 1, "frame": frames, "player_id": "Player1", "team": "home",
                      "x": frames * 0.2, "y": 0.0})             # 0.2 m per frame = 5 m/s
    v1 = add_velocities(t).set_index("frame")["vx"]
    future_changed = t.assign(x=np.where(t["frame"] > 10, 99.0, t["x"]))
    v2 = add_velocities(future_changed).set_index("frame")["vx"]
    assert np.isclose(v1[10], 5.0) and v1[:10].equals(v2[:10]) and v1[10] == v2[10]


def test_pitch_control_sanity():
    """Mirror-image players -> 0.5; an attacker alone at the target -> ~1; a defender alone there -> ~0."""
    one = lambda x: {"x": np.array([[x]]), "y": np.array([[0.0]]), "vx": np.zeros((1, 1)), "vy": np.zeros((1, 1))}
    ball, tx, ty, no_gk = np.array([[0.0, 20.0]]), np.array([0.0]), np.array([0.0]), np.array([[False]])
    assert np.isclose(pitch_control(ball, tx, ty, one(-5.0), one(5.0), no_gk), 0.5)
    assert pitch_control(ball, tx, ty, one(0.0), one(30.0), no_gk) > 0.99
    assert pitch_control(ball, tx, ty, one(30.0), one(0.0), no_gk) < 0.01


@needs_data
def test_features_ignore_frames_after_the_pass():
    """Adding garbage tracking rows just after each pass start must not change any feature."""
    p = pd.read_parquet(PASSES).groupby("match_id").head(150).reset_index(drop=True)  # every match, so folds can fit
    frames = read_pass_frames(p)
    later = frames.assign(frame=frames["frame"] + 1, x=frames["x"] + 30, y=-frames["y"])
    starts = set(zip(p["match_id"], p["start_frame"]))
    later = later[[(m, f) not in starts for m, f in zip(later["match_id"], later["frame"])]]  # only non-pass frames
    f1 = build_features(p, frames)
    f2 = build_features(p, pd.concat([frames, later], ignore_index=True))
    pd.testing.assert_frame_equal(f1[FEATURES + [PITCH_CONTROL]], f2[FEATURES + [PITCH_CONTROL]])


@needs_data
@pytest.mark.parametrize("test_match", [1, 2, 3])
def test_targets_never_learned_from_own_match(test_match):
    """Scrambling the receivers of match m must not change match m's intended targets in any fold."""
    p = pd.read_parquet(PASSES)
    a = pass_frame_arrays(p, read_pass_frames(p))
    X = candidate_features(a)
    base = fold_targets(p, a, X, test_match)
    for m in (1, 2, 3):
        scrambled = p.copy()
        rows = scrambled["match_id"] == m
        scrambled.loc[rows, "receiver"] = scrambled.loc[rows, "receiver"].sample(frac=1, random_state=1).to_numpy()
        changed = fold_targets(scrambled, a, X, test_match)
        assert base.loc[rows.to_numpy(), "target_player"].equals(changed.loc[rows.to_numpy(), "target_player"])
