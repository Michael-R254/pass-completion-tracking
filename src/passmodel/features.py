"""Feature engineering at the pass start frame (PLAN.md step 2).

Rule: every feature uses only frames <= the pass start frame. We read ONLY the tracking
rows at each pass's start frame; velocities there were computed from backward differences
(clean.add_velocities), so nothing after the pass is ever seen.

Everything works on per-pass arrays built once by ``pass_frame_arrays``: for N passes,
teammates are an (N, 10) grid and opponents an (N, 11) grid, all rotated so the passing
team attacks +x. Features are NumPy broadcasting over those grids: no Python loop over
passes or players.

Intended target (decision D3): a learned model scores each teammate as the likely receiver,
fitted only on matches other than the pass's own (see ``fold_targets``).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from passmodel.load import MATCH_IDS, read_frames
from passmodel.profiling import benchmark

N_MATES, N_OPPS = 10, 11
GOAL = np.array([52.5, 0.0])   # the goal the passing team attacks, in the rotated frame
LANE_WIDTH_M = 2.0              # an opponent within 2 m of the passer->target segment is "in the lane"
SEED = 0


# ---------------------------------------------------------------- geometry helpers

def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment AB, and t = position of P's projection along AB (0 at A, 1 at B).

    Tracking idea: a defender only blocks a pass if he is near the line BETWEEN passer and target,
    not beyond the target or behind the passer. t in (0, 1) says the closest point is inside the segment.
    All arguments broadcast, so one call handles every pass x every opponent.
    """
    abx, aby = bx - ax, by - ay
    length_sq = abx ** 2 + aby ** 2
    t = ((px - ax) * abx + (py - ay) * aby) / np.where(length_sq == 0, np.nan, length_sq)
    tc = np.clip(t, 0, 1)
    return np.hypot(px - (ax + tc * abx), py - (ay + tc * aby)), t


def angle_deg(dx, dy):
    """Absolute angle of a vector from straight forward (+x), in degrees: 0 = forward, 180 = backward."""
    return np.degrees(np.abs(np.arctan2(dy, dx)))


def _nanmin(a, axis):
    """nanmin that returns NaN (not a warning) for all-NaN slices."""
    return np.where(np.isnan(a).all(axis=axis), np.nan, np.nanmin(np.where(np.isnan(a), np.inf, a), axis=axis))


# ---------------------------------------------------------------- per-pass arrays

def read_pass_frames(passes: pd.DataFrame) -> pd.DataFrame:
    """Tracking rows at every pass start frame, all matches, reading only those frames from Parquet."""
    return pd.concat([read_frames(match_id=m, frames=passes.loc[passes["match_id"] == m, "start_frame"])
                      for m in MATCH_IDS], ignore_index=True)


def _grid(rows: pd.DataFrame, n_passes: int, width: int, col: str, fill=np.nan) -> np.ndarray:
    """Scatter one column of long rows (with pass_idx, slot) into an (n_passes, width) array."""
    out = np.full((n_passes, width), fill, dtype=object if fill is None else float)
    out[rows["pass_idx"].to_numpy(), rows["slot"].to_numpy()] = rows[col].to_numpy()
    return out


def pass_frame_arrays(passes: pd.DataFrame, frames: pd.DataFrame) -> dict:
    """Per-pass position/velocity arrays at the start frame, in the passing team's frame (attacks +x).

    Returns a dict of arrays, row i = passes.iloc[i]:
      passer_xy (N,2), passer_v (N,2)
      mate_x, mate_y, mate_vx, mate_vy (N,10), mate_id (N,10) object  -- teammates, passer excluded
      opp_x, opp_y, opp_vx, opp_vy (N,11)                              -- opponents
    """
    p = passes.reset_index(drop=True)[["match_id", "start_frame", "team", "attack_sign", "passer"]]
    p = p.assign(pass_idx=np.arange(len(p)))
    f = frames[frames["team"] != "ball"].astype({"player_id": str, "team": str, "match_id": int})
    rows = p.merge(f, left_on=["match_id", "start_frame"], right_on=["match_id", "frame"], suffixes=("_pass", ""))
    for c in ("x", "y", "vx", "vy"):
        rows[c] = rows[c].astype(float) * rows["attack_sign"]  # rotate: passing team attacks +x
    is_passer = rows["player_id"] == rows["passer"]
    mates = rows[(rows["team"] == rows["team_pass"]) & ~is_passer].copy()
    opps = rows[rows["team"] != rows["team_pass"]].copy()
    for g in (mates, opps):
        g["slot"] = g.groupby("pass_idx").cumcount()
    n = len(p)
    passer = rows[is_passer].set_index("pass_idx").reindex(range(n))
    out = {"passer_xy": passer[["x", "y"]].to_numpy(), "passer_v": passer[["vx", "vy"]].to_numpy(),
           "mate_id": _grid(mates, n, N_MATES, "player_id", fill=None)}
    for prefix, g, width in (("mate", mates, N_MATES), ("opp", opps, N_OPPS)):
        for c in ("x", "y", "vx", "vy"):
            out[f"{prefix}_{c}"] = _grid(g, n, width, c)
    return out


# ---------------------------------------------------------------- D3: intended target

TARGET_FEATURES = ["dist", "log_dist", "cos_run_dir", "cos_run_dir_x_speed", "forward_m", "openness",
                   "openness_capped", "mate_vx"]


def candidate_features(a: dict) -> np.ndarray:
    """(N, 10, F) start-frame features describing each teammate as a possible receiver.

    dist / log_dist: passer -> teammate distance. cos_run_dir: alignment with the passer's running
    direction (0 if stationary), and the same scaled by passer speed (capped at 5 m/s).
    forward_m: how far ahead of the passer. openness: teammate's distance to nearest opponent
    (and capped at 10 m). mate_vx: teammate running toward goal (+) or back (-).
    """
    dx = a["mate_x"] - a["passer_xy"][:, [0]]
    dy = a["mate_y"] - a["passer_xy"][:, [1]]
    dist = np.hypot(dx, dy)
    vx, vy = a["passer_v"][:, [0]], a["passer_v"][:, [1]]
    speed = np.hypot(vx, vy)
    cos_run = np.nan_to_num((dx * vx + dy * vy) / (dist * speed))
    openness = _nanmin(np.hypot(a["mate_x"][:, :, None] - a["opp_x"][:, None, :],
                                a["mate_y"][:, :, None] - a["opp_y"][:, None, :]), axis=2)
    X = np.stack([dist, np.log1p(dist), cos_run, cos_run * np.minimum(np.nan_to_num(speed), 5), dx,
                  openness, np.minimum(openness, 10), a["mate_vx"]], axis=2)
    return np.nan_to_num(X)


def fit_target_model(X: np.ndarray, is_receiver: np.ndarray):
    """Logistic regression on teammate rows: 'is this teammate the receiver?'.

    Trained on completed passes only (the only ones with a known receiver). At prediction time the
    highest-scoring teammate in each pass is the intended target.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    return model.fit(X.reshape(-1, X.shape[2]), is_receiver.ravel())


def predict_target(model, X: np.ndarray, a: dict, rows: np.ndarray) -> pd.DataFrame:
    """Intended target for the passes in `rows`: argmax teammate, its position, and the model's confidence.

    target_confidence = softmax share of the chosen teammate among the 10: high when the pass is
    'obvious', low when several options look alike. Uses start-frame information only.
    """
    scores = model.decision_function(X[rows].reshape(-1, X.shape[2])).reshape(-1, N_MATES)
    scores = np.where(np.isnan(a["mate_x"][rows]), -np.inf, scores)  # never pick a missing teammate
    slot = scores.argmax(axis=1)
    soft = np.exp(scores - scores.max(axis=1, keepdims=True))
    i, r = np.arange(len(slot)), np.flatnonzero(rows)
    q = soft / soft.sum(axis=1, keepdims=True)  # P(teammate j is the intended target), sums to 1 per pass
    out = pd.DataFrame({"target_player": a["mate_id"][r, slot], "target_slot": slot,
                        "target_confidence": q[i, slot]}, index=r)
    return out.join(pd.DataFrame(q, columns=[f"q_{j}" for j in range(N_MATES)], index=r))


def fold_targets(passes: pd.DataFrame, a: dict, X: np.ndarray, test_match: int) -> pd.DataFrame:
    """Intended targets for one leave-one-match-out fold, so no pass's target was learned from its own match.

    - test match: model fitted on both training matches.
    - each training match: model fitted on the OTHER training match only (cross-fitting). If it were fitted
      on its own match it would have memorised those receivers, making targets agree with the receiver far
      more on completed passes than on failed ones -- i.e. the target would leak the label into training.
    Training data for every model: completed open-play passes of the matches it is allowed to see.
    """
    p = passes.reset_index(drop=True)
    is_receiver = a["mate_id"] == p["receiver"].to_numpy()[:, None]
    trainable = (p["completed"] & ~p["is_set_piece"]).to_numpy()
    train_matches = [m for m in MATCH_IDS if m != test_match]
    parts = []
    for score_match, fit_matches in [(test_match, train_matches)] + [(m, [o for o in train_matches if o != m])
                                                                       for m in train_matches]:
        fit_rows = trainable & p["match_id"].isin(fit_matches).to_numpy()
        model = fit_target_model(X[fit_rows], is_receiver[fit_rows])
        parts.append(predict_target(model, X, a, (p["match_id"] == score_match).to_numpy()))
    return pd.concat(parts).sort_index()


# ---------------------------------------------------------------- stretch: pitch control

# Spearman (2018) physics-based pitch control, parameters as in Laurie Shaw's Friends of Tracking
# tutorial (github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, Metrica_PitchControl.py).
PC_MAX_SPEED = 5.0      # m/s, player top speed when running to the ball
PC_REACTION_S = 0.7     # s, players keep their current velocity for this long before turning
PC_SIGMA = 0.45         # s, uncertainty in each player's time to intercept
PC_LAMBDA = 4.3         # 1/s, control rate once a player has reached the ball (both teams; kappa_def = 1)
PC_LAMBDA_GK = 3 * PC_LAMBDA  # goalkeepers can use their hands
PC_BALL_SPEED = 15.0    # m/s, average pass speed
PC_DT, PC_MAX_T = 0.04, 10.0  # s, integration step and horizon


def _intercept_prob(x, y, vx, vy, tx, ty, T):
    """P(player has reached point (tx, ty) by time T). Arrays broadcast: (N, players) with (N, 1).

    Time to intercept: the player keeps running at their current velocity for the reaction time, then
    runs straight at the point at top speed. Arrival is uncertain, so P is a logistic curve around it.
    Missing players (NaN position) get probability 0.
    """
    rx, ry = x + vx * PC_REACTION_S, y + vy * PC_REACTION_S
    tti = PC_REACTION_S + np.hypot(tx - rx, ty - ry) / PC_MAX_SPEED
    p = 1 / (1 + np.exp(-np.pi / np.sqrt(3) / PC_SIGMA * (T - tti)))
    return np.nan_to_num(p)


def pitch_control(ball_xy, tx, ty, att, opp, opp_is_gk):
    """Attacking team's probability of controlling the ball at (tx, ty), one value per pass.

    Tracking idea: pitch control asks, for a point on the pitch, "if the ball were played there now, which
    team would get it first?" The ball takes distance / 15 m/s to arrive. From that moment on, at each small
    time step, every player who has reached the point gets a chance to win the ball, at rate lambda per
    second, out of whatever probability is still uncontrolled. Integrating gives P(attack) + P(defence) -> 1.

    att, opp: dicts of (N, k) arrays x, y, vx, vy (passing team attacks +x). opp_is_gk: (N, 11) bool.
    The loop is over integration time steps (250), not passes or players: each step updates all N passes
    x 22 players at once with broadcasting.
    """
    tx, ty = tx[:, None], ty[:, None]
    att_args = [att["x"], att["y"], np.nan_to_num(att["vx"]), np.nan_to_num(att["vy"])]  # NaN velocity = standing
    opp_args = [opp["x"], opp["y"], np.nan_to_num(opp["vx"]), np.nan_to_num(opp["vy"])]
    opp_lambda = np.where(opp_is_gk, PC_LAMBDA_GK, PC_LAMBDA)
    t_ball = np.hypot(tx[:, 0] - ball_xy[:, 0], ty[:, 0] - ball_xy[:, 1]) / PC_BALL_SPEED
    p_att, p_def = np.zeros(len(tx)), np.zeros(len(tx))
    for step in range(int(PC_MAX_T / PC_DT)):
        T = (t_ball + step * PC_DT)[:, None]
        rate_att = (PC_LAMBDA * _intercept_prob(*att_args, tx, ty, T)).sum(axis=1)
        rate_def = (opp_lambda * _intercept_prob(*opp_args, tx, ty, T)).sum(axis=1)
        free = 1 - p_att - p_def
        p_att, p_def = p_att + free * rate_att * PC_DT, p_def + free * rate_def * PC_DT
    # ponytail: no offside check (Shaw excludes offside attackers); add it if pitch control earns its place.
    return p_att / (p_att + p_def)  # renormalise the small mass left uncontrolled after 10 s


def target_pitch_control(a: dict, tx, ty) -> np.ndarray:
    """Pitch control at the intended target's position, at the pass start frame (start-frame data only).

    Attackers = passer + 10 teammates; the ball starts at the passer. Goalkeeper = the defender nearest
    their own goal (52.5, 0) at that frame, a one-line rule since the data has no position labels.
    """
    passer = {"x": a["passer_xy"][:, 0], "y": a["passer_xy"][:, 1], "vx": a["passer_v"][:, 0], "vy": a["passer_v"][:, 1]}
    att = {c: np.column_stack([passer[c], a[f"mate_{c}"]]) for c in passer}
    opp = {c: a[f"opp_{c}"] for c in ("x", "y", "vx", "vy")}
    d_goal = np.where(np.isnan(opp["x"]), np.inf, np.hypot(GOAL[0] - opp["x"], GOAL[1] - opp["y"]))
    is_gk = np.arange(N_OPPS)[None, :] == d_goal.argmin(axis=1)[:, None]
    return pitch_control(a["passer_xy"], tx, ty, att, opp, is_gk)


# ---------------------------------------------------------------- pass features

FEATURE_GROUPS = {
    "geometry": ["pass_length", "pass_angle", "is_forward", "progress_to_goal"],
    "location": ["start_x", "start_y", "target_x", "target_y", "zone_third", "zone_channel"],
    "pressure": ["passer_nearest_opp", "passer_opps_3m", "passer_opps_5m"],
    "target": ["target_nearest_opp", "target_closing_speed", "target_confidence"],
    "lane": ["lane_opps", "lane_min_opp_dist"],
    "speed": ["passer_speed", "target_speed", "passer_nearest_opp_speed"],
    "event": ["is_cross", "is_header"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
# Stretch: stored in features.parquet but NOT in FEATURES, so the headline model is unchanged
# unless pitch control beats it on held-out log loss.
PITCH_CONTROL = "target_pitch_control"


def pass_features(passes: pd.DataFrame, a: dict, targets: pd.DataFrame) -> pd.DataFrame:
    """All model features for each pass, at its start frame, given its intended target (D3).

    Every array below is (N,) per pass or (N, 11) per pass x opponent; nothing loops.
    """
    p = passes.reset_index(drop=True)
    i, slot = np.arange(len(p)), targets["target_slot"].to_numpy()
    px, py = a["passer_xy"][:, 0], a["passer_xy"][:, 1]
    tx, ty = a["mate_x"][i, slot], a["mate_y"][i, slot]
    tvx, tvy = a["mate_vx"][i, slot], a["mate_vy"][i, slot]
    ox, oy, ovx, ovy = a["opp_x"], a["opp_y"], a["opp_vx"], a["opp_vy"]
    dx, dy = tx - px, ty - py

    # pressure on the passer: (N, 11) passer-to-opponent distances
    d_passer = np.hypot(ox - px[:, None], oy - py[:, None])
    nearest_to_passer = np.nanargmin(np.where(np.isnan(d_passer), np.inf, d_passer), axis=1)

    # target openness and closing speed of the nearest defender to the target
    d_target = np.hypot(ox - tx[:, None], oy - ty[:, None])
    k = np.nanargmin(np.where(np.isnan(d_target), np.inf, d_target), axis=1)
    rel_x, rel_y = ox[i, k] - tx, oy[i, k] - ty                     # target -> defender
    rel_vx, rel_vy = ovx[i, k] - tvx, ovy[i, k] - tvy
    # closing speed = how fast the gap shrinks = -d/dt |rel|; positive means the defender is closing in
    with np.errstate(invalid="ignore", divide="ignore"):  # defender exactly on the target -> NaN
        closing = -(rel_x * rel_vx + rel_y * rel_vy) / np.hypot(rel_x, rel_y)

    # lane: opponents near the passer->target segment and strictly between them (0 < t < 1)
    d_lane, t = point_to_segment(ox, oy, px[:, None], py[:, None], tx[:, None], ty[:, None])
    between = (t > 0) & (t < 1)

    goal_dist = lambda x, y: np.hypot(GOAL[0] - x, GOAL[1] - y)
    f = pd.DataFrame({
        "pass_length": np.hypot(dx, dy),
        "pass_angle": angle_deg(dx, dy),
        "is_forward": dx > 0,
        "progress_to_goal": goal_dist(px, py) - goal_dist(tx, ty),
        "start_x": px, "start_y": py, "target_x": tx, "target_y": ty,
        "zone_third": np.digitize(px, [-17.5, 17.5]),               # 0 defensive, 1 middle, 2 attacking third
        "zone_channel": np.digitize(np.abs(py), [9.16, 20.16]),      # 0 central, 1 half-space, 2 wide
        "passer_nearest_opp": _nanmin(d_passer, axis=1),
        "passer_opps_3m": (d_passer < 3).sum(axis=1),
        "passer_opps_5m": (d_passer < 5).sum(axis=1),
        "target_nearest_opp": d_target[i, k],
        "target_closing_speed": closing,
        "target_confidence": targets["target_confidence"].to_numpy(),
        "lane_opps": ((d_lane < LANE_WIDTH_M) & between).sum(axis=1),
        "lane_min_opp_dist": _nanmin(np.where(between, d_lane, np.nan), axis=1),
        "passer_speed": np.hypot(a["passer_v"][:, 0], a["passer_v"][:, 1]),
        "target_speed": np.hypot(tvx, tvy),
        "passer_nearest_opp_speed": np.hypot(ovx[i, nearest_to_passer], ovy[i, nearest_to_passer]),
        "is_cross": p["is_cross"].to_numpy(),
        "is_header": p["is_header"].to_numpy(),
    })
    f["lane_min_opp_dist"] = f["lane_min_opp_dist"].fillna(LANE_WIDTH_M * 10)  # nobody between: lane wide open
    f[PITCH_CONTROL] = target_pitch_control(a, tx, ty)
    return f[FEATURES + [PITCH_CONTROL]]


@benchmark("build features (all folds)")
def build_features(passes: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Features for every pass in every leave-one-match-out fold, one row per (fold, pass).

    Only the intended target (and features that depend on it) differ between folds, because each fold's
    target models may only see that fold's training matches. `fold` = the fold's test match.
    """
    p = passes.reset_index(drop=True)
    a = pass_frame_arrays(p, frames)
    X = candidate_features(a)
    keep = ["match_id", "event_id", "period", "start_frame", "team", "passer", "receiver", "completed",
            "is_set_piece", "set_piece_type"]
    out = []
    for test_match in MATCH_IDS:
        targets = fold_targets(p, a, X, test_match)
        feats = pass_features(p, a, targets)
        out.append(pd.concat([p[keep], targets[["target_player"]], feats], axis=1).assign(fold=test_match))
    return pd.concat(out, ignore_index=True)


@benchmark("build target-mixture features (all folds x 10 teammates)")
def build_target_mixture_features(passes: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Features for every (fold, pass, teammate): the pass scored as if aimed at each of the 10 teammates.

    Instead of betting on one intended target, keep all of them with weight q = P(teammate is the target)
    from the same cross-fitted target models as build_features (so no pass's weights were learned from its
    own match). The completion model is trained on these rows weighted by q, and predicts
    P(complete) = sum_j q_j * P(complete | aimed at j)  (models.run_lomo_target_mixture).
    The loop is over the 10 teammate slots; each iteration is vectorised over all passes.
    Rows with q = 0 (teammate not tracked) are dropped.
    """
    p = passes.reset_index(drop=True)
    a = pass_frame_arrays(p, frames)
    X = candidate_features(a)
    keep = ["match_id", "event_id", "completed", "is_set_piece"]
    out = []
    for test_match in MATCH_IDS:
        targets = fold_targets(p, a, X, test_match)
        for j in range(N_MATES):
            q = targets[f"q_{j}"].to_numpy()
            slot_j = pd.DataFrame({"target_slot": j, "target_confidence": q}, index=targets.index)
            feats = pass_features(p, a, slot_j)
            out.append(pd.concat([p[keep], feats], axis=1).assign(fold=test_match, slot=j, weight=q)[q > 0])
    return pd.concat(out, ignore_index=True)


def target_agreement(features: pd.DataFrame) -> pd.DataFrame:
    """How often the intended target is the actual receiver, on completed open-play passes.

    Per fold, split into the held-out test match and the cross-fitted training matches. Every
    number here comes from a target model that did not see the match being scored.
    """
    c = features[features["completed"] & ~features["is_set_piece"]]
    c = c.assign(role=np.where(c["match_id"] == c["fold"], "test match", "training matches (cross-fitted)"),
                 agree=c["target_player"] == c["receiver"])
    return c.groupby(["fold", "role", "match_id"])["agree"].agg(agreement="mean", n="size").round(3)
