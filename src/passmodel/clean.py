"""Units (metres), playing-direction flip, velocities and pass labelling (PLAN.md step 1).

Coordinate convention (tracking and events after cleaning):
  - metres, origin at the centre spot; x along the pitch length (-52.5..52.5),
    y across it (-34..34). Metrica's raw 0-1 units have origin top-left with y
    pointing down, so x_m = (x - 0.5) * 105 and y_m = (0.5 - y) * 68.
  - tracking is rotated per period so the HOME team always attacks +x (towards
    x = +52.5) in both halves. Teams swap ends at half time, so without this a
    pass "towards goal" would mean opposite things in each half.
  - passes are stored with an extra flip so the PASSING team attacks +x
    (attack_sign = +1 for home, -1 for away). Features in step 2 multiply tracking
    by the same attack_sign, so "forward" always means +x.
  The flip is a 180 degree rotation (both x and y change sign), so it is a pure
  change of viewpoint: distances, and left/right relative to the attacking direction, are preserved.
"""

import numpy as np
import pandas as pd

from passmodel.profiling import benchmark

PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0  # from the kloppy metadata of all three games
FPS = 25
VELOCITY_LAG = 5          # frames: backward difference over 0.2 s
MAX_PLAYER_SPEED = 12.0   # m/s; faster is tracking noise (Usain Bolt peaks ~12.4 m/s)
MAX_PASSER_TO_BALL_M = 5.0  # drop a pass if the named passer is further than this from the tracked ball

# Known tracking/event player-ID disagreements, applied to the tracking (match_id -> period -> {old: new}).
# Game 3, 2nd half: events and tracking give Player6 and Player13 (both home) opposite IDs. Evidence:
# all 144 second-half events by either player (passes, carries, challenges, recoveries, shots, ...)
# have the named player 14-33 m (median) from the event and the other one ~0 m; 0 of 69 in the 1st half.
# We can't tell which source is wrong; swapping the tracking makes the two agree.
# tests/test_data.py fails if passers stop being at the ball, so a wrong swap would be caught.
ID_SWAPS = {3: {2: {"Player6": "Player13", "Player13": "Player6"}}}

# Subtype tokens that mean "not a pass attempt we can judge", whatever the event type.
# Applied to PASS, BALL LOST and BALL OUT alike, so exclusions never depend on the outcome.
# Order matters only for naming the reason when a subtype has two of them (e.g. FORCED-END HALF).
EXCLUDE_TOKENS = ("THEFT", "FORCED", "CLEARANCE", "OFFSIDE", "END HALF", "WOODWORK", "HAND BALL", "REFEREE HIT")


# SET PIECE subtype token -> set_piece_type (D2). Goal kicks are detected from the pass's own subtype.
SET_PIECE_TYPES = {"CORNER KICK": "corner", "FREE KICK": "free kick", "THROW IN": "throw-in",
                   "KICK OFF": "kick-off", "PENALTY": "penalty"}


def _tokens(subtype: pd.Series) -> pd.Series:
    """'HEAD-INTERCEPTION' -> {'HEAD', 'INTERCEPTION'}; missing -> empty set."""
    return subtype.fillna("").map(lambda s: set(s.split("-")) - {""})


def label_passes(events: pd.DataFrame) -> pd.DataFrame:
    """Apply decision D1 (see CLAUDE.md): add pass_bucket, exclude_reason and flags to a load_events() table.

    pass_bucket:
      completed  PASS row
      failed     BALL LOST row (lost in play: intercepted, headed or crossed to an opponent, ...)
                 or BALL OUT row (player in possession puts the ball out of play); `type` tells which
      excluded   see exclude_reason
      NaN        not a pass-type event
    exclude_reason (first that applies):
      subtype token in EXCLUDE_TOKENS, e.g. "THEFT", "FORCED" -- any pass-type row
      "ball out: no From player"
      "ball out: after SHOT" / "after CLEARANCE" / "after tackle or THEFT"
      "ball out: after BALL LOST" -- the same loose ball already counted as a failed pass,
                                     so it must not be counted twice
    Flags: is_cross, is_header, and (decision D2) set_piece_type / is_set_piece:
    corner, free kick, throw-in, kick-off, penalty (from the SET PIECE row directly
    before the pass, same player) or goal kick (the pass's own subtype). NaN = open play.

    "Previous event" skips CARRY rows, which only game 3 records, so the rules mean
    the same thing in all three games.
    """
    ev = events.copy()
    tok = _tokens(ev["subtype"])
    bucket = ev["type"].map({"PASS": "completed", "BALL LOST": "failed", "BALL OUT": "failed"})

    seq = ev[ev["type"] != "CARRY"]
    prev = seq.groupby("match_id")[["type", "subtype", "team", "from_player"]].shift(1).reindex(ev.index)
    prev_tok = _tokens(prev["subtype"])

    token_reason = tok.map(lambda t: next((x for x in EXCLUDE_TOKENS if x in t), None))
    is_out = ev["type"] == "BALL OUT"
    reasons = [
        (bucket.notna() & token_reason.notna(), token_reason),
        (is_out & ev["from_player"].isna(), "ball out: no From player"),
        (is_out & (prev["type"] == "SHOT"), "ball out: after SHOT"),
        (is_out & prev_tok.map(lambda t: "CLEARANCE" in t), "ball out: after CLEARANCE"),
        (is_out & (((prev["type"] == "CHALLENGE") & prev_tok.map(lambda t: "TACKLE" in t))
                   | prev_tok.map(lambda t: "THEFT" in t)), "ball out: after tackle or THEFT"),
        (is_out & (prev["type"] == "BALL LOST"), "ball out: after BALL LOST"),
    ]
    ev["exclude_reason"] = None
    for cond, reason in reversed(reasons):  # reversed so the first rule listed wins
        ev.loc[cond, "exclude_reason"] = reason if isinstance(reason, str) else reason[cond]
    ev["pass_bucket"] = bucket.mask(ev["exclude_reason"].notna(), "excluded")

    ev["is_cross"] = tok.map(lambda t: "CROSS" in t)
    ev["is_header"] = tok.map(lambda t: "HEAD" in t)

    # D2: Metrica writes a set piece as its own SET PIECE row, then the restart pass as the
    # next event by the same player (checked: 100% same player, 91.5% same frame).
    # Goal kicks have no SET PIECE row; the pass itself carries the GOAL KICK subtype.
    after_set_piece = (prev["type"] == "SET PIECE") & (prev["from_player"] == ev["from_player"])
    sp_type = prev_tok.map(lambda t: next((v for k, v in SET_PIECE_TYPES.items() if k in t), None))
    sp_type = sp_type.where(after_set_piece)
    sp_type = sp_type.mask(tok.map(lambda t: "GOAL KICK" in t), "goal kick")
    ev["set_piece_type"] = sp_type.where(bucket.notna())
    ev["is_set_piece"] = ev["set_piece_type"].notna()
    return ev


def label_counts(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-game counts of every pass bucket and exclusion reason, plus completion rate.

    Written to results/d1_label_counts.csv so the README's label and limitations
    numbers come from a script output (e.g. how many FORCED rows were dropped per game).
    """
    p = labelled[labelled["pass_bucket"].notna()]
    row = p["pass_bucket"].where(p["pass_bucket"] != "excluded", "excluded: " + p["exclude_reason"].astype(str))
    t = pd.crosstab(row, p["match_id"], margins=True, margins_name="all")
    rate = t.loc["completed"] / (t.loc["completed"] + t.loc["failed"])
    return pd.concat([t, rate.round(3).to_frame("completion_rate").T])


def set_piece_counts(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per game and overall: pass count and completion rate for open play and each set_piece_type (D2).

    Written to results/d2_set_piece_counts.csv.
    """
    k = labelled[labelled["pass_bucket"].isin(["completed", "failed"])]
    k = k.assign(completed=k["pass_bucket"].eq("completed"), kind=k["set_piece_type"].fillna("open play"))
    k = pd.concat([k, k.assign(match_id="all")])
    k = pd.concat([k, k[k["is_set_piece"]].assign(kind="all set pieces")])
    g = k.groupby(["kind", "match_id"])["completed"].agg(n="size", completion_rate="mean")
    return g.unstack("match_id").round(3)


def to_metres(x, y):
    """Metrica 0-1 coordinates (origin top-left, y down) -> metres from the centre spot (y up)."""
    return (x - 0.5) * PITCH_LENGTH, (0.5 - y) * PITCH_WIDTH


def home_attack_sign(tracking: pd.DataFrame) -> pd.Series:
    """Per period: +1 if home already attacks +x, else -1. Input: tracking in metres.

    How: at kick-off every player is in his own half, so the home players' mean x over the
    first second of a period tells us which end home defends. Home defending x < 0 means home attacks +x.
    """
    first = tracking.groupby("period")["frame"].transform("min")
    start = tracking[(tracking["team"] == "home") & (tracking["frame"] < first + FPS)]
    mean_x = start.groupby("period")["x"].mean()
    return np.sign(-mean_x).astype(int).rename("home_attack_sign")


def add_velocities(tracking: pd.DataFrame) -> pd.DataFrame:
    """Add vx, vy, speed (m/s) for every object at every frame, from backward differences only.

    v(t) = (pos(t) - pos(t - 5 frames)) / 0.2 s, per player and period. Backward only, so a
    feature at the pass frame never uses positions after the pass (CLAUDE.md rule 3).
    Averaging over 0.2 s instead of 1 frame smooths tracking jitter; the cost is the velocity
    describes the last 0.2 s (centred 0.1 s in the past).
    If the frame 5 rows back isn't exactly 5 frames earlier (player just appeared), velocity is NaN.
    Player speeds above MAX_PLAYER_SPEED are scaled down to it (direction kept); the ball is not capped.
    One grouped shift over the whole match: no loops over players or frames.
    """
    t = tracking.sort_values(["period", "player_id", "frame"], ignore_index=True)
    prev = t.groupby(["period", "player_id"], observed=True)[["frame", "x", "y"]].shift(VELOCITY_LAG)
    dt = (t["frame"] - prev["frame"]) / FPS
    ok = (t["frame"] - prev["frame"]) == VELOCITY_LAG
    vx = ((t["x"] - prev["x"]) / dt).where(ok)
    vy = ((t["y"] - prev["y"]) / dt).where(ok)
    speed = np.hypot(vx, vy)
    scale = np.where((t["team"] != "ball") & (speed > MAX_PLAYER_SPEED), MAX_PLAYER_SPEED / speed, 1.0)
    t["vx"], t["vy"] = (vx * scale).astype(np.float32), (vy * scale).astype(np.float32)
    t["speed"] = (speed * scale).astype(np.float32)
    t.attrs["n_speed_capped"] = int((scale < 1).sum())
    return t


def apply_id_swaps(tracking: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Relabel tracking player IDs listed in ID_SWAPS for this match (see the evidence there)."""
    t = tracking.copy()
    for period, mapping in ID_SWAPS.get(match_id, {}).items():
        in_period = t["period"] == period
        masks = {old: in_period & (t["player_id"] == old) for old in mapping}  # computed before any change
        for old, new in mapping.items():
            t.loc[masks[old], "player_id"] = new
    return t


@benchmark("clean tracking (ID fixes, metres, flip, velocities)")
def clean_tracking(tracking: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Known ID fixes + metres + home-attacks-+x rotation + velocities for one match's long tracking table."""
    t = apply_id_swaps(tracking, match_id)
    t["x"], t["y"] = to_metres(t["x"], t["y"])
    sign = t["period"].map(home_attack_sign(t)).to_numpy(np.float32)
    t["x"], t["y"] = t["x"] * sign, t["y"] * sign
    t = add_velocities(t)
    t.insert(0, "match_id", np.int8(match_id))
    return t


@benchmark("build passes (read pass frames only)")
def build_passes(labelled: pd.DataFrame, match_id: int, home_sign: pd.Series, frames: pd.DataFrame) -> pd.DataFrame:
    """One row per attempted pass (D1 completed/failed, set pieces kept and flagged) for one match.

    labelled:  label_passes() output for this match.
    home_sign: home_attack_sign() per period, the rotation applied to this match's tracking.
    frames:    tracking at the pass start frames only (load.read_frames), already in metres/home-attacks-+x.

    Coordinates are rotated so the passing team attacks +x (attack_sign: +1 home, -1 away).
    end_x/end_y and receiver are kept for checks and for scoring the intended-target rule
    (D3), but are outcome-dependent and must never be model features (CLAUDE.md rule 4).

    Adds data-quality columns at the start frame, used to decide which passes to drop:
    ball_present, passer_present, n_home / n_away players tracked,
    ball_to_event_m (event start vs tracked ball: the event/tracking sync error),
    passer_to_ball_m (tracked passer vs tracked ball).
    """
    p = labelled[labelled["pass_bucket"].isin(["completed", "failed"])].copy()
    p["completed"] = p["pass_bucket"].eq("completed")
    p["attack_sign"] = np.where(p["team"] == "home", 1, -1)
    sign = p["period"].map(home_sign).to_numpy() * p["attack_sign"]  # raw metres -> passing team attacks +x
    for end in ("start", "end"):
        x, y = to_metres(p[f"{end}_x"], p[f"{end}_y"])
        p[f"{end}_x"], p[f"{end}_y"] = x * sign, y * sign

    # Tracking at the start frame is already rotated so home attacks +x; only attack_sign is left to apply.
    team_sign = p["attack_sign"].to_numpy()[:, None]
    f = frames.set_index("frame")
    ball = f[f["player_id"] == "ball"][["x", "y"]]
    ball = ball.reindex(p["start_frame"]).to_numpy() * team_sign
    passer = f[["player_id", "x", "y"]].reset_index().merge(
        p[["start_frame", "from_player"]].reset_index(), left_on=["frame", "player_id"],
        right_on=["start_frame", "from_player"]).set_index("index")[["x", "y"]].reindex(p.index)
    passer = passer.to_numpy() * team_sign
    counts = f[f["team"] != "ball"].groupby(["frame", "team"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(p["start_frame"]).fillna(0)

    p["ball_present"] = ~np.isnan(ball[:, 0])
    p["passer_present"] = ~np.isnan(passer[:, 0])
    p["n_home"], p["n_away"] = counts.get("home", 0).to_numpy(int), counts.get("away", 0).to_numpy(int)
    p["ball_to_event_m"] = np.hypot(ball[:, 0] - p["start_x"], ball[:, 1] - p["start_y"])
    p["passer_to_ball_m"] = np.hypot(passer[:, 0] - ball[:, 0], passer[:, 1] - ball[:, 1])

    cols = ["match_id", "event_id", "period", "start_frame", "end_frame", "team", "attack_sign",
            "from_player", "to_player", "type", "subtype", "completed", "is_set_piece", "set_piece_type",
            "is_cross", "is_header", "start_x", "start_y", "end_x", "end_y",
            "ball_present", "passer_present", "n_home", "n_away", "ball_to_event_m", "passer_to_ball_m"]
    return p[cols].rename(columns={"from_player": "passer", "to_player": "receiver"}).reset_index(drop=True)


def drop_bad_passes(passes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop passes whose start frame can't be trusted; return (kept passes, drop counts per game and reason).

    Reasons, first that applies:
      "ball not tracked at start frame"
      "passer > 5 m from ball at start frame"  (event/tracking out of sync, or a player-ID error)
    Missing end coordinates are not a reason: end location is never a feature.
    """
    reason = pd.Series(None, index=passes.index, dtype=object)
    far = passes["passer_to_ball_m"] > MAX_PASSER_TO_BALL_M
    reason[far & reason.isna()] = f"passer > {MAX_PASSER_TO_BALL_M:g} m from ball at start frame"
    reason[~passes["ball_present"]] = "ball not tracked at start frame"
    dropped = passes[reason.notna()].assign(reason=reason[reason.notna()])
    counts = pd.crosstab(dropped["reason"], dropped["match_id"]).reindex(columns=sorted(passes["match_id"].unique()), fill_value=0)
    counts.loc["total dropped"] = counts.sum()
    counts.loc["passes before"] = passes.groupby("match_id").size()
    counts["all"] = counts.sum(axis=1)
    return passes[reason.isna()].reset_index(drop=True), counts


def pass_summary(passes: pd.DataFrame) -> pd.DataFrame:
    """Per game: passes and completion rate, all passes and open play only (D2: the main model's data)."""
    both = pd.concat([passes, passes.assign(match_id="all")])
    open_play = both[~both["is_set_piece"]]
    return pd.DataFrame({
        "passes": both.groupby("match_id").size(),
        "completion_rate": both.groupby("match_id")["completed"].mean().round(3),
        "open_play_passes": open_play.groupby("match_id").size(),
        "open_play_completion_rate": open_play.groupby("match_id")["completed"].mean().round(3),
    })
