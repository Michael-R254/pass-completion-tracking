"""Tests never write to the real results/benchmarks.csv."""

import pytest

from passmodel import profiling


@pytest.fixture(autouse=True)
def _benchmarks_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "BENCHMARK_CSV", tmp_path / "benchmarks.csv")
