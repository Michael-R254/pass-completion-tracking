"""D1 pass labelling: buckets, exclusions applied to all types, BALL OUT rules, set-piece flag skipping CARRY rows."""

import pandas as pd

from passmodel.clean import label_passes


def test_label_passes():
    rows = [  # team, type, subtype, from_player
        ("A", "SET PIECE", "THROW IN", "P1"),
        ("A", "CARRY", None, "P1"),
        ("A", "PASS", None, "P1"),                  # completed, set piece (CARRY skipped)
        ("B", "BALL LOST", "HEAD-INTERCEPTION", "P2"),  # failed
        ("A", "BALL OUT", None, "P3"),              # excluded: after BALL LOST (no double count)
        ("A", "PASS", "CLEARANCE", "P3"),           # excluded: CLEARANCE, even though completed
        ("A", "BALL OUT", None, "P3"),              # excluded: after CLEARANCE
        ("A", "BALL OUT", None, "P4"),              # failed (previous is a plain BALL OUT)
        ("B", "BALL LOST", "THEFT", "P5"),          # excluded: THEFT
        ("B", "BALL OUT", None, None),              # excluded: no From player (beats "after THEFT")
        ("B", "SHOT", "OFF TARGET-OUT", "P6"),
        ("B", "BALL OUT", None, "P6"),              # excluded: after SHOT
        ("B", "PASS", "GOAL KICK", "P7"),           # completed, set piece
    ]
    ev = pd.DataFrame(rows, columns=["team", "type", "subtype", "from_player"]).assign(match_id=1)
    out = label_passes(ev)
    assert out["pass_bucket"].fillna("-").tolist() == [
        "-", "-", "completed", "failed", "excluded", "excluded", "excluded", "failed",
        "excluded", "excluded", "-", "excluded", "completed"]
    assert out["exclude_reason"].fillna("-").tolist() == [
        "-", "-", "-", "-", "ball out: after BALL LOST", "CLEARANCE", "ball out: after CLEARANCE", "-",
        "THEFT", "ball out: no From player", "-", "ball out: after SHOT", "-"]
    assert out["is_set_piece"].tolist() == [False, False, True] + [False] * 9 + [True]
    assert out["is_header"].tolist()[3] and not out["is_header"].tolist()[2]
