"""Long-running workload for sampling issue #28304 predicate plans."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import polars as pl


def query(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path, parallel="prefiltered")
        .filter(pl.col("l_shipmode").is_in(["AIR", "AIR REG"]))
        .filter(pl.col("l_shipinstruct") == "DELIVER IN PERSON")
        .filter(pl.col("l_quantity").is_between(1, 30))
        .select(pl.col("l_quantity").sum())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--stage-spec")
    args = parser.parse_args()

    if args.stage_spec is None:
        os.environ.pop("POLARS_ISSUE_28304_STAGES", None)
    else:
        os.environ["POLARS_ISSUE_28304_STAGES"] = args.stage_spec
    os.environ.pop("POLARS_ISSUE_28304_ADAPTIVE", None)

    workload = query(args.data_path)
    expected = workload.collect(engine="streaming").item()
    for _ in range(args.warmups):
        assert workload.collect(engine="streaming").item() == expected
    if args.ready_file is not None:
        args.ready_file.touch()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for _ in range(args.iterations):
        assert workload.collect(engine="streaming").item() == expected

    print(
        json.dumps(
            {
                "cpu_seconds": time.process_time() - cpu_start,
                "expected": expected,
                "iterations": args.iterations,
                "stage_spec": args.stage_spec,
                "wall_seconds": time.perf_counter() - wall_start,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
