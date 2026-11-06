"""Timing and peak-memory logging per pipeline stage -> results/benchmarks.csv (PLAN.md step 0).

Why: the README's "Scaling" section must quote measured numbers, not guesses.
Every pipeline stage is wrapped in ``@benchmark("stage name")`` and appends one
row per call to ``results/benchmarks.csv``. Per-match rows let us project the
cost of a 380-match season (time per match x 380).

Memory is measured with the stdlib ``tracemalloc``: it counts allocations made
through Python's allocator, which includes NumPy and pandas arrays. It does NOT
see memory allocated inside Arrow's C++ pool (e.g. inside ``pyarrow.parquet``),
so Parquet read/write peaks are a lower bound. Stated in the README.
Each row therefore also logs process_peak_mb: the operating system's high-water mark
for the whole process so far (includes Arrow and kloppy), via process_peak_mb().
"""

from __future__ import annotations

import csv
import functools
import inspect
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CSV = REPO_ROOT / "results" / "benchmarks.csv"
COLUMNS = ["timestamp", "stage", "match_id", "wall_s", "peak_mb", "process_peak_mb", "rows_in", "rows_out"]


def process_peak_mb() -> float:
    """Peak memory of this whole process so far, in MB, from the OS (stdlib only).

    Unix: getrusage max RSS. Windows: PeakWorkingSetSize via the Win32 psapi call
    (the stdlib `resource` module doesn't exist on Windows).
    """
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KB on Linux
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        current_process = wintypes.HANDLE(-1)
        ctypes.windll.psapi.GetProcessMemoryInfo(current_process, ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e6


def _rows(obj) -> int | None:
    """Row count of a DataFrame / Series / array (anything with ``shape``), else None."""
    shape = getattr(obj, "shape", None)
    return int(shape[0]) if shape else None


def benchmark(stage: str):
    """Decorator: log wall time, peak memory and rows in/out of one pipeline stage.

    - rows_in  = total rows of all DataFrame/array arguments (blank if none, e.g. a loader).
    - rows_out = rows of the returned DataFrame/array (blank if it returns something else).
    - match_id = the function's ``match_id`` argument if it has one, so costs are per match.

    If stages are nested, the inner stage's peak is the peak since the outer
    stage started (an upper bound), because tracemalloc has one global peak.
    """

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            in_counts = [r for r in map(_rows, bound.arguments.values()) if r is not None]

            started_here = not tracemalloc.is_tracing()
            if started_here:
                tracemalloc.start()
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                wall = time.perf_counter() - t0
                peak = tracemalloc.get_traced_memory()[1]
                if started_here:
                    tracemalloc.stop()

            _append_row({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "match_id": bound.arguments.get("match_id", ""),
                "wall_s": round(wall, 3),
                "peak_mb": round(peak / 1e6, 1),
                "process_peak_mb": round(process_peak_mb(), 1),
                "rows_in": sum(in_counts) if in_counts else "",
                "rows_out": _rows(result) if _rows(result) is not None else "",
            })
            return result

        return wrapper

    return decorator


def _append_row(row: dict) -> None:
    """Append one row to the benchmarks CSV, writing the header if the file is new."""
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not BENCHMARK_CSV.exists()
    with BENCHMARK_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def season_projection(benchmarks_csv: Path, tracking_files_csv: Path, n_measured: int = 3,
                      n_season: int = 380):
    """Scale the measured per-stage cost of `n_measured` matches up to a season of `n_season` matches.

    Every stage runs one match at a time (or is linear in passes), so season time = measured time x 380 / 3,
    and peak memory stays at the single-match peak rather than growing with the season.
    An estimate: assumes season matches look like these three (similar length, ~23 tracked objects).
    Returns one row per stage plus a storage row for the tracking Parquet.
    """
    import pandas as pd

    b = pd.read_csv(benchmarks_csv)
    t = b.groupby("stage", sort=False).agg(rows_out=("rows_out", "sum"), wall_s=("wall_s", "sum"),
                                           peak_mb=("peak_mb", "max"), process_peak_mb=("process_peak_mb", "max"))
    t["season_wall_min"] = t["wall_s"] * n_season / n_measured / 60
    total = t[["wall_s", "season_wall_min"]].sum().to_frame("all stages").T
    total[["peak_mb", "process_peak_mb"]] = t[["peak_mb", "process_peak_mb"]].max().to_numpy()
    files = pd.read_csv(tracking_files_csv)
    storage = pd.DataFrame({"rows_out": [files["rows"].sum()],
                            "season_parquet_gb": [files["parquet_mb"].sum() * n_season / n_measured / 1000]},
                           index=["tracking Parquet on disk"])
    return pd.concat([t, total, storage]).rename_axis("stage").round(1)
