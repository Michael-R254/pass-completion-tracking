"""The leave-one-match-out split never puts the test match (or set pieces) into training."""

import pandas as pd

from passmodel.models import fold_split


def test_fold_split_holds_out_whole_match():
    rows = [(fold, match, sp) for fold in (1, 2, 3) for match in (1, 2, 3) for sp in (False, True)]
    f = pd.DataFrame(rows, columns=["fold", "match_id", "is_set_piece"])
    for fold in (1, 2, 3):
        train, test = fold_split(f, fold)
        assert set(train["match_id"]) == {1, 2, 3} - {fold}
        assert set(test["match_id"]) == {fold}
        assert (train["fold"] == fold).all() and (test["fold"] == fold).all()
        assert not train["is_set_piece"].any() and not test["is_set_piece"].any()
