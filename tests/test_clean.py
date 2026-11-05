"""D1 pass labelling: buckets, exclusions applied to all types, set-piece flag skipping CARRY rows."""

import pandas as pd

from passmodel.clean import label_passes


def test_label_passes():
    ev = pd.DataFrame({
        "match_id": 1,
        "team": ["A", "A", "A", "B", "A", "A", "B", "B", "B"],
        "type": ["SET PIECE", "CARRY", "PASS", "BALL LOST", "PASS", "BALL OUT", "BALL LOST", "PASS", "RECOVERY"],
        "subtype": ["THROW IN", None, None, "HEAD-INTERCEPTION", "CLEARANCE", None, "THEFT", "GOAL KICK", None],
    })
    out = label_passes(ev)
    assert out["pass_bucket"].fillna("-").tolist() == ["-", "-", "completed", "failed", "excluded", "failed_out",
                                                       "excluded", "completed", "-"]
    assert out["is_set_piece"].tolist() == [False, False, True, False, False, False, False, True, False]
    assert out["is_header"].tolist()[3] and not out["is_header"].tolist()[2]
