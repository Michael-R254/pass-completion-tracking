"""The benchmark decorator writes one correct row per call and returns the result unchanged."""

import csv

import numpy as np

from passmodel import profiling


def test_benchmark_logs_one_row(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "BENCHMARK_CSV", tmp_path / "bench.csv")

    @profiling.benchmark("double")
    def double(arr, match_id):
        return np.concatenate([arr, arr])

    out = double(np.zeros(1_000_000), match_id=2)
    assert len(out) == 2_000_000

    rows = list(csv.DictReader((tmp_path / "bench.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "double"
    assert row["match_id"] == "2"
    assert row["rows_in"] == "1000000"
    assert row["rows_out"] == "2000000"
    assert float(row["peak_mb"]) >= 16  # the 2M float64 output alone is 16 MB
