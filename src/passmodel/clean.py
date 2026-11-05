"""Units (metres), playing-direction flip and pass labelling (PLAN.md step 1).

Coordinate convention is documented here once decided.
"""

import numpy as np
import pandas as pd

# Subtype tokens that mean "not a pass attempt we can judge", whatever the event type.
# Applied to PASS, BALL LOST and BALL OUT alike, so exclusions never depend on the outcome.
# Order matters only for naming the reason when a subtype has two of them (e.g. FORCED-END HALF).
EXCLUDE_TOKENS = ("THEFT", "FORCED", "CLEARANCE", "OFFSIDE", "END HALF", "WOODWORK", "HAND BALL", "REFEREE HIT")


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
    Flags: is_cross, is_header, is_set_piece. A set piece is a separate SET PIECE row
    by the same team immediately before the pass; goal kicks have no SET PIECE row,
    so the GOAL KICK subtype also counts.

    "Previous event" skips CARRY rows, which only game 3 records, so the rules mean
    the same thing in all three games.
    """
    ev = events.copy()
    tok = _tokens(ev["subtype"])
    bucket = ev["type"].map({"PASS": "completed", "BALL LOST": "failed", "BALL OUT": "failed"})

    seq = ev[ev["type"] != "CARRY"]
    prev = seq.groupby("match_id")[["type", "subtype", "team"]].shift(1).reindex(ev.index)
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
    after_set_piece = (prev["type"] == "SET PIECE") & (prev["team"] == ev["team"])
    ev["is_set_piece"] = after_set_piece | tok.map(lambda t: "GOAL KICK" in t)
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
