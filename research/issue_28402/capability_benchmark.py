"""Benchmark mixed-capability Parquet predicate staging for issue #28402.

Research runner; not production policy. It compares current global fallback
with environment-gated capability partition and slice-blocked reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import polars as pl


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = round((len(ordered) - 1) * probability)
    return ordered[position]


def summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_median_difference(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    candidate_array = np.asarray(candidate)
    baseline_array = np.asarray(baseline)
    if candidate_array.shape != baseline_array.shape:
        msg = "candidate and baseline sample counts differ"
        raise ValueError(msg)
    if np.array_equal(candidate_array, baseline_array):
        return {"median": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        candidate_array.size,
        size=(resamples, candidate_array.size),
    )
    differences = np.median(
        candidate_array[indices] - baseline_array[indices],
        axis=1,
    )
    return {
        "median": float(np.median(candidate_array - baseline_array)),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def generate_data(
    path: Path,
    *,
    rows: int,
    row_group_size: int,
    prefix_retention: float,
    relationship: str,
    payload_width: int,
    seed: int,
) -> dict[str, float]:
    if not 0 < prefix_retention < 1:
        msg = "--prefix-retention must be between zero and one"
        raise ValueError(msg)

    generator = np.random.default_rng(seed)
    first = generator.random(rows)
    second = generator.random(rows)
    if relationship == "independent":
        marginal = prefix_retention**0.5
        first_mask = first < marginal
        second_mask = second < marginal
    elif relationship == "nested":
        first_mask = first < prefix_retention
        second_mask = first < min(0.99, max(prefix_retention, prefix_retention**0.5))
    elif relationship == "negative":
        marginal = (1.0 + prefix_retention) / 2.0
        first_mask = first < marginal
        second_mask = first >= 1.0 - marginal
    else:
        msg = f"unknown relationship: {relationship}"
        raise ValueError(msg)

    residual = generator.integers(1, 51, size=rows, dtype=np.int32)
    columns: dict[str, Any] = {
        "_first": first_mask,
        "_second": second_mask,
        "c_i64": residual.astype(np.int64),
        "c_f64": residual.astype(np.float64),
    }
    for index in range(payload_width):
        columns[f"payload_{index}"] = generator.integers(
            0,
            10_000,
            size=rows,
            dtype=np.int64,
        )

    frame = (
        pl.DataFrame(columns)
        .with_columns(
            pl.when("_first").then(pl.lit("C")).otherwise(pl.lit("A")).alias("c1"),
            pl.when("_second").then(pl.lit("W")).otherwise(pl.lit("Z")).alias("c2"),
        )
        .drop("_first", "_second")
        .select("c1", "c2", "c_i64", "c_f64", pl.selectors.starts_with("payload_"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(
        path,
        compression="zstd",
        row_group_size=row_group_size,
        statistics=True,
    )
    return measure_data(path)


def predicates(residual_dtype: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    residual_column = "c_f64" if residual_dtype == "float" else "c_i64"
    return (
        pl.col("c1") == "C",
        pl.col("c2") == "W",
        pl.col(residual_column).is_between(1, 30),
    )


def measure_data(path: Path) -> dict[str, float]:
    frame = pl.scan_parquet(path)
    first, second, residual = predicates("float")
    values = frame.select(
        first.mean().alias("first"),
        second.mean().alias("second"),
        (first & second).mean().alias("prefix"),
        residual.mean().alias("residual"),
        (first & second & residual).mean().alias("final"),
    ).collect()
    return {name: float(value) for name, value in values.row(0, named=True).items()}


def make_query(
    path: Path,
    *,
    residual_dtype: str,
    blocked: bool,
    residual_projected: bool,
) -> pl.LazyFrame:
    first, second, residual = predicates(residual_dtype)
    residual_column = "c_f64" if residual_dtype == "float" else "c_i64"
    query = pl.scan_parquet(path, parallel="prefiltered").filter(first).filter(second)
    if blocked:
        query = query.slice(0, None)
    query = query.filter(residual)

    aggregations = [
        pl.col(name).sum().alias(name)
        for name in query.collect_schema().names()
        if name.startswith("payload_")
    ]
    if residual_projected:
        aggregations.append(pl.col(residual_column).sum().alias(residual_column))
    return query.select(aggregations)


def configure_environment(
    *, capability_mode: str | None, stage_spec: str | None
) -> None:
    if capability_mode is not None:
        os.environ["POLARS_ISSUE_28402_CAPABILITY_STAGING"] = capability_mode
    else:
        os.environ.pop("POLARS_ISSUE_28402_CAPABILITY_STAGING", None)
    if stage_spec is None:
        os.environ.pop("POLARS_ISSUE_28304_STAGES", None)
    else:
        os.environ["POLARS_ISSUE_28304_STAGES"] = stage_spec
    os.environ.pop("POLARS_ISSUE_28304_ADAPTIVE", None)
    os.environ.pop("POLARS_ISSUE_28304_ROLLING_POLICY", None)


def execute(
    query: pl.LazyFrame,
    *,
    capability_mode: str | None = None,
    stage_spec: str | None = None,
) -> tuple[tuple[Any, ...], float, float]:
    configure_environment(capability_mode=capability_mode, stage_spec=stage_spec)
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = query.collect(engine="streaming").row(0)
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1e9
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1e9
    return result, wall_seconds, cpu_seconds


def run(
    path: Path,
    *,
    residual_dtype: str,
    residual_projected: bool,
    warmups: int,
    iterations: int,
    seed: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    blocked_query = make_query(
        path,
        residual_dtype=residual_dtype,
        blocked=True,
        residual_projected=residual_projected,
    )
    pushed_query = make_query(
        path,
        residual_dtype=residual_dtype,
        blocked=False,
        residual_projected=residual_projected,
    )
    if residual_dtype == "float":
        specifications = {
            "global_fallback": (pushed_query, None, None),
            "capability_selected": (pushed_query, "selected", None),
            "capability_materialized": (pushed_query, "materialized", None),
            "slice_blocked": (blocked_query, None, None),
        }
        baseline_name = "global_fallback"
    else:
        specifications = {
            "all_at_once": (pushed_query, None, None),
            "forced_staged": (pushed_query, None, "c1,c2|c_i64"),
            "slice_blocked": (blocked_query, None, None),
        }
        baseline_name = "all_at_once"

    plans = {}
    for name, (query, capability_mode, stage_spec) in specifications.items():
        configure_environment(
            capability_mode=capability_mode,
            stage_spec=stage_spec,
        )
        plans[name] = query.explain(optimized=True, engine="streaming")

    expected: tuple[Any, ...] | None = None
    for name, (query, capability_mode, stage_spec) in specifications.items():
        result, _, _ = execute(
            query,
            capability_mode=capability_mode,
            stage_spec=stage_spec,
        )
        if expected is None:
            expected = result
        elif result != expected:
            msg = f"query {name!r} returned {result!r}; expected {expected!r}"
            raise AssertionError(msg)

    for _ in range(warmups):
        for query, capability_mode, stage_spec in specifications.values():
            execute(
                query,
                capability_mode=capability_mode,
                stage_spec=stage_spec,
            )

    samples: dict[str, dict[str, list[float]]] = {
        name: {"wall_seconds": [], "cpu_seconds": []} for name in specifications
    }
    names = list(specifications)
    generator = random.Random(seed)
    for _ in range(iterations):
        generator.shuffle(names)
        for name in names:
            gc.collect()
            query, capability_mode, stage_spec = specifications[name]
            _, wall_seconds, cpu_seconds = execute(
                query,
                capability_mode=capability_mode,
                stage_spec=stage_spec,
            )
            samples[name]["wall_seconds"].append(wall_seconds)
            samples[name]["cpu_seconds"].append(cpu_seconds)

    query_summaries = {
        name: {metric: summarize(values) for metric, values in metrics.items()}
        for name, metrics in samples.items()
    }
    baseline_wall = query_summaries[baseline_name]["wall_seconds"]["median"]
    comparisons = {}
    for name in specifications:
        if name == baseline_name:
            continue
        candidate_wall = query_summaries[name]["wall_seconds"]["median"]
        comparisons[name] = {
            "change_vs_baseline": candidate_wall / baseline_wall - 1,
            "paired_median_wall_difference_seconds": bootstrap_median_difference(
                samples[name]["wall_seconds"],
                samples[baseline_name]["wall_seconds"],
                resamples=bootstrap_resamples,
                seed=seed,
            ),
        }

    return (
        {
            "expected_result": expected,
            "baseline": baseline_name,
            "queries": query_summaries,
            "comparisons": comparisons,
            "samples": samples,
        },
        plans,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--prefix-retention", type=float, default=0.05)
    parser.add_argument(
        "--relationship",
        choices=["independent", "nested", "negative"],
        default="independent",
    )
    parser.add_argument("--residual-dtype", choices=["float", "int"], default="float")
    parser.add_argument("--payload-width", type=int, default=1)
    parser.add_argument(
        "--residual-projection",
        choices=["projected", "filter_only"],
        default="projected",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=28402)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.row_group_size < 1 or args.payload_width < 1:
        msg = "row counts, row-group size, and payload width must be positive"
        raise ValueError(msg)
    if args.iterations < 1 or args.bootstrap_resamples < 1:
        msg = "iterations and bootstrap resamples must be positive"
        raise ValueError(msg)

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.data_path is None:
        if args.reuse_data:
            msg = "--reuse-data requires --data-path"
            raise ValueError(msg)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="polars-issue-28402-capability-"
        )
        data_path = Path(temporary_directory.name) / "mixed-capability.parquet"
    else:
        data_path = args.data_path

    if args.reuse_data:
        rates = measure_data(data_path)
    else:
        rates = generate_data(
            data_path,
            rows=args.rows,
            row_group_size=args.row_group_size,
            prefix_retention=args.prefix_retention,
            relationship=args.relationship,
            payload_width=args.payload_width,
            seed=args.seed,
        )

    result, plans = run(
        data_path,
        residual_dtype=args.residual_dtype,
        residual_projected=args.residual_projection == "projected",
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "polars_version": pl.__version__,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
            "polars_thread_pool_size": pl.thread_pool_size(),
        },
        "configuration": {
            "rows": args.rows,
            "row_group_size": args.row_group_size,
            "row_groups": (args.rows + args.row_group_size - 1) // args.row_group_size,
            "prefix_retention": args.prefix_retention,
            "relationship": args.relationship,
            "residual_dtype": args.residual_dtype,
            "payload_width": args.payload_width,
            "residual_projection": args.residual_projection,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "observed_selectivity": rates,
        **result,
        "plans": plans,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if not args.quiet:
        print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    configure_environment(capability_mode=None, stage_spec=None)
    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
