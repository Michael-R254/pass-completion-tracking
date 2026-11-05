"""Units (metres), playing-direction flip and pass labelling (PLAN.md step 1).

Coordinate convention is documented here once decided.
"""

import numpy as np
import pandas as pd

# Subtype tokens that mean "not a pass attempt we can judge", whatever the event type.
# Applied to PASS, BALL LOST and BALL OUT alike, so completed and failed passes are filtered by the same rule.
EXCLUDE_TOKENS = {"THEFT", "FORCED", "CLEARANCE", "OFFSIDE", "END HALF", "WOODWORK", "HAND BALL", "REFEREE HIT"}


def _tokens(subtype: pd.Series) -> pd.Series:
    """'HEAD-INTERCEPTION' -> {'HEAD', 'INTERCEPTION'}; missing -> empty set."""
    return subtype.fillna("").map(lambda s: set(s.split("-")) - {""})


def label_passes(events: pd.DataFrame) -> pd.DataFrame:
    """Add a D1 pass bucket and flags to a load_events() table.

    pass_bucket:
      completed  PASS row
      failed     BALL LOST row (lost in play: intercepted, headed or crossed to an opponent, ...)
      failed_out BALL OUT row (player in possession puts the ball out of play) -- pending D1
      excluded   any of the above whose subtype has an EXCLUDE_TOKENS token
      NaN        not a pass-type event
    Flags: is_cross, is_header, is_set_piece. A set piece is a separate SET PIECE row
    by the same team immediately before the pass (CARRY rows, only recorded in game 3,
    are skipped so the rule means the same in all games); goal kicks have no SET PIECE
    row, so the GOAL KICK subtype also counts.
    """
    ev = events.copy()
    tok = _tokens(ev["subtype"])
    bucket = ev["type"].map({"PASS": "completed", "BALL LOST": "failed", "BALL OUT": "failed_out"})
    excluded = bucket.notna() & tok.map(lambda t: bool(t & EXCLUDE_TOKENS))
    ev["pass_bucket"] = bucket.mask(excluded, "excluded")

    ev["is_cross"] = tok.map(lambda t: "CROSS" in t)
    ev["is_header"] = tok.map(lambda t: "HEAD" in t)

    seq = ev[ev["type"] != "CARRY"]
    prev = seq.groupby("match_id")[["type", "team"]].shift(1)
    after_set_piece = (prev["type"] == "SET PIECE") & (prev["team"] == seq["team"])
    ev["is_set_piece"] = after_set_piece.reindex(ev.index, fill_value=False) | tok.map(lambda t: "GOAL KICK" in t)
    return ev
