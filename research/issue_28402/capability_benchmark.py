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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import polars as pl


@dataclass(frozen=True)
class Specification:
    query: pl.LazyFrame
    capability_mode: str | None = None
    stage_spec: str | None = None
    fanout_min_task_values: int | None = None
    auto_max_speculative_values: int | None = None
    auto_min_file_rows: int | None = None


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
    payload_dtype: str,
    seed: int,
) -> None:
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
        values = generator.integers(
            0,
            10_000,
            size=rows,
            dtype=np.int64,
        )
        if payload_dtype == "int":
            payload = values
        elif payload_dtype == "float":
            payload = values.astype(np.float64)
        elif payload_dtype == "string":
            payload = np.char.add("v", values.astype(str))
        else:
            msg = f"unknown payload dtype: {payload_dtype}"
            raise ValueError(msg)
        columns[f"payload_{index}"] = payload

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


def predicates(
    residual_dtype: str,
    residual_retention: float,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    residual_column = "c_f64" if residual_dtype == "float" else "c_i64"
    residual_upper = round(50 * residual_retention)
    residual_upper = max(1, min(49, residual_upper))
    return (
        pl.col("c1") == "C",
        pl.col("c2") == "W",
        pl.col(residual_column).is_between(1, residual_upper),
    )


def measure_data(path: Path, *, residual_retention: float) -> dict[str, float]:
    frame = pl.scan_parquet(path)
    first, second, residual = predicates("float", residual_retention)
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
    residual_retention: float,
    blocked: bool,
    residual_projected: bool,
) -> pl.LazyFrame:
    first, second, residual = predicates(residual_dtype, residual_retention)
    residual_column = "c_f64" if residual_dtype == "float" else "c_i64"
    query = pl.scan_parquet(path, parallel="prefiltered").filter(first).filter(second)
    if blocked:
        query = query.slice(0, None)
    query = query.filter(residual)

    schema = query.collect_schema()
    aggregations = []
    for name, dtype in schema.items():
        if not name.startswith("payload_"):
            continue
        expression = pl.col(name)
        if dtype == pl.String:
            expression = expression.str.len_bytes()
        aggregations.append(expression.sum().alias(name))
    if residual_projected:
        aggregations.append(pl.col(residual_column).sum().alias(residual_column))
    return query.select(aggregations)


def configure_environment(
    *,
    capability_mode: str | None,
    stage_spec: str | None,
    fanout_min_task_values: int | None = None,
    auto_max_speculative_values: int | None = None,
    auto_min_file_rows: int | None = None,
) -> None:
    if capability_mode is not None:
        os.environ["POLARS_ISSUE_28402_CAPABILITY_STAGING"] = capability_mode
    else:
        os.environ.pop("POLARS_ISSUE_28402_CAPABILITY_STAGING", None)
    if stage_spec is None:
        os.environ.pop("POLARS_ISSUE_28304_STAGES", None)
    else:
        os.environ["POLARS_ISSUE_28304_STAGES"] = stage_spec
    if fanout_min_task_values is None:
        os.environ.pop("POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES", None)
    else:
        os.environ["POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES"] = str(
            fanout_min_task_values
        )
    if auto_max_speculative_values is None:
        os.environ.pop("POLARS_ISSUE_28402_AUTO_MAX_SPECULATIVE_VALUES", None)
    else:
        os.environ["POLARS_ISSUE_28402_AUTO_MAX_SPECULATIVE_VALUES"] = str(
            auto_max_speculative_values
        )
    if auto_min_file_rows is None:
        os.environ.pop("POLARS_ISSUE_28402_AUTO_MIN_FILE_ROWS", None)
    else:
        os.environ["POLARS_ISSUE_28402_AUTO_MIN_FILE_ROWS"] = str(auto_min_file_rows)
    os.environ.pop("POLARS_ISSUE_28304_ADAPTIVE", None)
    os.environ.pop("POLARS_ISSUE_28304_ROLLING_POLICY", None)


def execute(
    query: pl.LazyFrame,
    *,
    capability_mode: str | None = None,
    stage_spec: str | None = None,
    fanout_min_task_values: int | None = None,
    auto_max_speculative_values: int | None = None,
    auto_min_file_rows: int | None = None,
) -> tuple[tuple[Any, ...], float, float]:
    configure_environment(
        capability_mode=capability_mode,
        stage_spec=stage_spec,
        fanout_min_task_values=fanout_min_task_values,
        auto_max_speculative_values=auto_max_speculative_values,
        auto_min_file_rows=auto_min_file_rows,
    )
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
    residual_retention: float,
    residual_projected: bool,
    warmups: int,
    iterations: int,
    seed: int,
    bootstrap_resamples: int,
    fanout_min_task_values: Sequence[int],
    auto_max_speculative_values: Sequence[int],
    auto_fanout_min_task_values: int | None,
    auto_min_file_rows: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    blocked_query = make_query(
        path,
        residual_dtype=residual_dtype,
        residual_retention=residual_retention,
        blocked=True,
        residual_projected=residual_projected,
    )
    pushed_query = make_query(
        path,
        residual_dtype=residual_dtype,
        residual_retention=residual_retention,
        blocked=False,
        residual_projected=residual_projected,
    )
    if residual_dtype == "float":
        specifications = {
            "global_fallback": Specification(pushed_query),
            "capability_selected": Specification(
                pushed_query,
                capability_mode="selected",
            ),
            "capability_materialized": Specification(
                pushed_query,
                capability_mode="materialized",
            ),
            "capability_fanout": Specification(
                pushed_query,
                capability_mode="fanout",
            ),
            **{
                f"capability_fanout_local_{task_values}": Specification(
                    pushed_query,
                    capability_mode="fanout",
                    fanout_min_task_values=task_values,
                )
                for task_values in fanout_min_task_values
            },
            **{
                f"capability_auto_{max_values}": Specification(
                    pushed_query,
                    capability_mode="auto",
                    fanout_min_task_values=auto_fanout_min_task_values,
                    auto_max_speculative_values=max_values,
                    auto_min_file_rows=auto_min_file_rows,
                )
                for max_values in auto_max_speculative_values
            },
            "slice_blocked": Specification(blocked_query),
        }
        baseline_name = "global_fallback"
    else:
        specifications = {
            "all_at_once": Specification(pushed_query),
            "forced_staged": Specification(
                pushed_query,
                stage_spec="c1,c2|c_i64",
            ),
            "slice_blocked": Specification(blocked_query),
        }
        baseline_name = "all_at_once"

    plans = {}
    for name, specification in specifications.items():
        configure_environment(
            capability_mode=specification.capability_mode,
            stage_spec=specification.stage_spec,
            fanout_min_task_values=specification.fanout_min_task_values,
            auto_max_speculative_values=specification.auto_max_speculative_values,
            auto_min_file_rows=specification.auto_min_file_rows,
        )
        plans[name] = specification.query.explain(optimized=True, engine="streaming")

    expected: tuple[Any, ...] | None = None
    for name, specification in specifications.items():
        result, _, _ = execute(
            specification.query,
            capability_mode=specification.capability_mode,
            stage_spec=specification.stage_spec,
            fanout_min_task_values=specification.fanout_min_task_values,
            auto_max_speculative_values=specification.auto_max_speculative_values,
            auto_min_file_rows=specification.auto_min_file_rows,
        )
        if expected is None:
            expected = result
        elif result != expected:
            msg = f"query {name!r} returned {result!r}; expected {expected!r}"
            raise AssertionError(msg)

    for _ in range(warmups):
        for specification in specifications.values():
            execute(
                specification.query,
                capability_mode=specification.capability_mode,
                stage_spec=specification.stage_spec,
                fanout_min_task_values=specification.fanout_min_task_values,
                auto_max_speculative_values=specification.auto_max_speculative_values,
                auto_min_file_rows=specification.auto_min_file_rows,
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
            specification = specifications[name]
            _, wall_seconds, cpu_seconds = execute(
                specification.query,
                capability_mode=specification.capability_mode,
                stage_spec=specification.stage_spec,
                fanout_min_task_values=specification.fanout_min_task_values,
                auto_max_speculative_values=specification.auto_max_speculative_values,
                auto_min_file_rows=specification.auto_min_file_rows,
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
    pairwise_comparisons = {}
    if residual_dtype == "float":
        for name in specifications:
            if name.startswith("capability_fanout_local_"):
                reference_names = ("capability_fanout", "slice_blocked")
            elif name.startswith("capability_auto_"):
                reference_names = ("capability_fanout", "capability_materialized")
            else:
                continue
            pairwise_comparisons[name] = {}
            for reference_name in reference_names:
                candidate_wall = query_summaries[name]["wall_seconds"]["median"]
                reference_wall = query_summaries[reference_name]["wall_seconds"][
                    "median"
                ]
                pairwise_comparisons[name][reference_name] = {
                    "change_vs_reference": candidate_wall / reference_wall - 1,
                    "paired_median_wall_difference_seconds": (
                        bootstrap_median_difference(
                            samples[name]["wall_seconds"],
                            samples[reference_name]["wall_seconds"],
                            resamples=bootstrap_resamples,
                            seed=seed,
                        )
                    ),
                }

    return (
        {
            "expected_result": expected,
            "baseline": baseline_name,
            "queries": query_summaries,
            "comparisons": comparisons,
            "samples": samples,
            "pairwise_comparisons": pairwise_comparisons,
        },
        plans,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--prefix-retention", type=float, default=0.05)
    parser.add_argument("--residual-retention", type=float, default=0.6)
    parser.add_argument(
        "--relationship",
        choices=["independent", "nested", "negative"],
        default="independent",
    )
    parser.add_argument("--residual-dtype", choices=["float", "int"], default="float")
    parser.add_argument("--payload-width", type=int, default=1)
    parser.add_argument(
        "--payload-dtype",
        choices=["int", "float", "string"],
        default="int",
    )
    parser.add_argument(
        "--residual-projection",
        choices=["projected", "filter_only"],
        default="projected",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument(
        "--fanout-min-task-values",
        default="",
        help="comma-separated fanout-local task sizes to compare in same run",
    )
    parser.add_argument(
        "--auto-max-speculative-values",
        default="",
        help="comma-separated one-thread auto-schedule thresholds",
    )
    parser.add_argument(
        "--auto-fanout-min-task-values",
        type=int,
        help="fanout-local task size used by auto fanout arm",
    )
    parser.add_argument(
        "--auto-min-file-rows",
        type=int,
        default=500_000,
        help="minimum file rows eligible for one-thread auto mode",
    )
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
    if not 0 < args.residual_retention < 1:
        msg = "--residual-retention must be between zero and one"
        raise ValueError(msg)
    if args.iterations < 1 or args.bootstrap_resamples < 1:
        msg = "iterations and bootstrap resamples must be positive"
        raise ValueError(msg)
    fanout_min_task_values = [
        int(value.strip())
        for value in args.fanout_min_task_values.split(",")
        if value.strip()
    ]
    if any(value < 1 for value in fanout_min_task_values):
        msg = "--fanout-min-task-values must contain positive integers"
        raise ValueError(msg)
    if len(fanout_min_task_values) != len(set(fanout_min_task_values)):
        msg = "--fanout-min-task-values must not contain duplicates"
        raise ValueError(msg)
    auto_max_speculative_values = [
        int(value.strip())
        for value in args.auto_max_speculative_values.split(",")
        if value.strip()
    ]
    if any(value < 1 for value in auto_max_speculative_values):
        msg = "--auto-max-speculative-values must contain positive integers"
        raise ValueError(msg)
    if len(auto_max_speculative_values) != len(set(auto_max_speculative_values)):
        msg = "--auto-max-speculative-values must not contain duplicates"
        raise ValueError(msg)
    if (
        args.auto_fanout_min_task_values is not None
        and args.auto_fanout_min_task_values < 1
    ):
        msg = "--auto-fanout-min-task-values must be positive"
        raise ValueError(msg)
    if args.auto_min_file_rows < 1:
        msg = "--auto-min-file-rows must be positive"
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
        rates = measure_data(
            data_path,
            residual_retention=args.residual_retention,
        )
    else:
        generate_data(
            data_path,
            rows=args.rows,
            row_group_size=args.row_group_size,
            prefix_retention=args.prefix_retention,
            relationship=args.relationship,
            payload_width=args.payload_width,
            payload_dtype=args.payload_dtype,
            seed=args.seed,
        )
        rates = measure_data(
            data_path,
            residual_retention=args.residual_retention,
        )

    result, plans = run(
        data_path,
        residual_dtype=args.residual_dtype,
        residual_retention=args.residual_retention,
        residual_projected=args.residual_projection == "projected",
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        fanout_min_task_values=fanout_min_task_values,
        auto_max_speculative_values=auto_max_speculative_values,
        auto_fanout_min_task_values=args.auto_fanout_min_task_values,
        auto_min_file_rows=args.auto_min_file_rows,
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
            "residual_retention": args.residual_retention,
            "relationship": args.relationship,
            "residual_dtype": args.residual_dtype,
            "payload_width": args.payload_width,
            "payload_dtype": args.payload_dtype,
            "residual_projection": args.residual_projection,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "fanout_min_task_values": fanout_min_task_values,
            "auto_max_speculative_values": auto_max_speculative_values,
            "auto_fanout_min_task_values": args.auto_fanout_min_task_values,
            "auto_min_file_rows": args.auto_min_file_rows,
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

    configure_environment(
        capability_mode=None,
        stage_spec=None,
        fanout_min_task_values=None,
        auto_max_speculative_values=None,
        auto_min_file_rows=None,
    )
    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
