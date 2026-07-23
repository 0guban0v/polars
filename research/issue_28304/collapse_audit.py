"""Audit restricted predicate-staging actions against exhaustive oracle.

Research runner for issue #28304; not production policy. The restricted action
set keeps all predicates concurrent or defers exactly one predicate behind all
remaining predicates.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import operator
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import polars as pl

import benchmark as benchmark_28304

THREE_PREDICATE_COLUMNS = (
    "l_shipmode",
    "l_shipinstruct",
    "l_quantity",
)
FOUR_PREDICATE_COLUMNS = (*THREE_PREDICATE_COLUMNS, "l_discount")


def predicate_columns(predicate_count: int) -> tuple[str, ...]:
    if predicate_count == 3:
        return THREE_PREDICATE_COLUMNS
    if predicate_count == 4:
        return FOUR_PREDICATE_COLUMNS
    msg = f"unsupported predicate count: {predicate_count}"
    raise ValueError(msg)


def predicate_expressions(predicate_count: int) -> dict[str, pl.Expr]:
    expressions = {
        "l_shipmode": pl.col("l_shipmode").is_in(["AIR", "AIR REG"]),
        "l_shipinstruct": pl.col("l_shipinstruct") == "DELIVER IN PERSON",
        "l_quantity": pl.col("l_quantity").is_between(1, 30),
    }
    if predicate_count == 4:
        expressions["l_discount"] = pl.col("l_discount") < 400
    return expressions


def conjunction(expressions: Sequence[pl.Expr]) -> pl.Expr:
    return functools.reduce(operator.and_, expressions)


def measure_data(path: Path, predicate_count: int) -> tuple[int, dict[str, float]]:
    expressions = predicate_expressions(predicate_count)
    columns = predicate_columns(predicate_count)
    metrics = [pl.len().alias("rows")]
    metrics.extend(expressions[column].mean().alias(column) for column in columns)
    metrics.append(
        conjunction([expressions[column] for column in columns])
        .mean()
        .alias("all_conjunction")
    )
    for deferred in columns:
        metrics.append(
            conjunction(
                [expressions[column] for column in columns if column != deferred]
            )
            .mean()
            .alias(f"before_{deferred}")
        )

    row = pl.scan_parquet(path).select(metrics).collect().row(0, named=True)
    rows = int(row.pop("rows"))
    return rows, {name: float(value) for name, value in row.items()}


def generate_data(
    path: Path,
    *,
    predicate_count: int,
    rows: int,
    row_group_size: int,
    shipmode_per_mille: int,
    shipinstruct_per_mille: int,
    seed: int,
    correlation: str,
    mask_topology: str,
) -> dict[str, float]:
    benchmark_28304.generate_data(
        path,
        rows=rows,
        row_group_size=row_group_size,
        shipmode_per_mille=shipmode_per_mille,
        shipinstruct_per_mille=shipinstruct_per_mille,
        seed=seed,
        correlation=correlation,
        mask_topology=mask_topology,
    )
    if predicate_count == 4:
        frame = pl.read_parquet(path)
        generator = np.random.default_rng(seed + 1)
        discount = generator.integers(0, 1_000, size=rows, dtype=np.uint16)
        frame.with_columns(
            pl.Series("l_discount", discount).cast(pl.Int32)
        ).write_parquet(
            path,
            compression="zstd",
            row_group_size=row_group_size,
            statistics=True,
        )
    _, rates = measure_data(path, predicate_count)
    return rates


def ordered_partitions(items: Sequence[str]) -> list[tuple[tuple[str, ...], ...]]:
    """Return every ordered set partition of items in deterministic order."""
    partitions = []
    for stage_count in range(1, len(items) + 1):
        for assignment in itertools.product(range(stage_count), repeat=len(items)):
            if set(assignment) != set(range(stage_count)):
                continue
            stages = tuple(
                tuple(item for item, stage in zip(items, assignment) if stage == index)
                for index in range(stage_count)
            )
            partitions.append(stages)
    return partitions


def stage_spec(stages: Sequence[Sequence[str]]) -> str:
    return "|".join(",".join(stage) for stage in stages)


def classify_partition(stages: Sequence[Sequence[str]]) -> str:
    if len(stages) == 1:
        return "all_at_once"
    if len(stages) == 2 and len(stages[-1]) == 1:
        return "defer_one"
    if len(stages) >= 2 and len(stages[0]) >= 2:
        return "concurrent_prefix"
    if len(stages) == 2:
        return "other_two_stage"
    return "multi_stage"


def query_factory(predicate_count: int) -> benchmark_28304.QueryFactory:
    columns = predicate_columns(predicate_count)

    def make(path: Path) -> pl.LazyFrame:
        query = pl.scan_parquet(path, parallel="prefiltered")
        expressions = predicate_expressions(predicate_count)
        for column in columns:
            query = query.filter(expressions[column])
        return query.select(pl.col("l_quantity").sum())

    return make


def oracle_queries(
    predicate_count: int,
) -> tuple[
    dict[str, benchmark_28304.QueryFactory],
    dict[str, str],
    dict[str, str],
]:
    columns = predicate_columns(predicate_count)
    factory = query_factory(predicate_count)
    factories = {"all_pushed": factory}
    specs = {}
    classes = {"all_pushed": "all_at_once"}
    for index, stages in enumerate(ordered_partitions(columns)):
        spec = stage_spec(stages)
        readable = spec.replace(",", "_").replace("|", "__")
        name = f"partition_{index:03d}_{readable}"
        factories[name] = factory
        specs[name] = spec
        classes[name] = classify_partition(stages)
    return factories, specs, classes


def bootstrap_median_difference(
    candidate: Sequence[float],
    oracle: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    candidate_array = np.asarray(candidate)
    oracle_array = np.asarray(oracle)
    if candidate_array.shape != oracle_array.shape:
        msg = "candidate and oracle sample counts differ"
        raise ValueError(msg)
    if np.array_equal(candidate_array, oracle_array):
        return {"median": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        candidate_array.size,
        size=(resamples, candidate_array.size),
    )
    differences = np.median(
        candidate_array[indices] - oracle_array[indices],
        axis=1,
    )
    return {
        "median": float(np.median(candidate_array - oracle_array)),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def analyze_audit(
    result: dict[str, Any],
    classes: dict[str, str],
    *,
    restricted_action_set: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    medians = {
        name: metrics["wall_seconds"]["median"]
        for name, metrics in result["queries"].items()
    }
    oracle_name = min(medians, key=medians.__getitem__)
    if restricted_action_set == "defer_one":
        allowed_classes = {"all_at_once", "defer_one"}
    elif restricted_action_set == "concurrent_prefix":
        allowed_classes = {
            "all_at_once",
            "defer_one",
            "concurrent_prefix",
        }
    else:
        msg = f"unsupported restricted action set: {restricted_action_set}"
        raise ValueError(msg)
    restricted_names = [
        name
        for name, partition_class in classes.items()
        if partition_class in allowed_classes
    ]
    restricted_name = min(restricted_names, key=medians.__getitem__)
    oracle_wall = medians[oracle_name]
    restricted_wall = medians[restricted_name]

    class_best = {}
    for partition_class in sorted(set(classes.values())):
        names = [name for name, value in classes.items() if value == partition_class]
        name = min(names, key=medians.__getitem__)
        class_best[partition_class] = {
            "name": name,
            "median_wall_seconds": medians[name],
            "regret_vs_oracle": medians[name] / oracle_wall - 1,
        }

    samples = result["samples"]
    difference = bootstrap_median_difference(
        samples[restricted_name]["wall_seconds"],
        samples[oracle_name]["wall_seconds"],
        resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "oracle": {
            "name": oracle_name,
            "class": classes[oracle_name],
            "median_wall_seconds": oracle_wall,
        },
        "restricted": {
            "action_set": restricted_action_set,
            "name": restricted_name,
            "class": classes[restricted_name],
            "median_wall_seconds": restricted_wall,
            "regret_vs_oracle": restricted_wall / oracle_wall - 1,
            "median_wall_difference_seconds": difference,
        },
        "oracle_in_restricted_action_set": classes[oracle_name] in allowed_classes,
        "best_by_class": class_best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--shipmode-per-mille", type=int, default=142)
    parser.add_argument("--shipinstruct-per-mille", type=int, default=250)
    parser.add_argument(
        "--correlation",
        choices=["independent", "positive", "negative"],
        default="independent",
    )
    parser.add_argument(
        "--mask-topology",
        choices=["random", "contiguous", "alternating"],
        default="random",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument(
        "--restricted-action-set",
        choices=["defer_one", "concurrent_prefix"],
        default="defer_one",
    )
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        msg = "--iterations must be positive"
        raise ValueError(msg)
    if args.bootstrap_resamples < 1:
        msg = "--bootstrap-resamples must be positive"
        raise ValueError(msg)

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.data_path is None:
        if args.reuse_data:
            msg = "--reuse-data requires --data-path"
            raise ValueError(msg)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="polars-issue-28304-collapse-"
        )
        data_path = Path(temporary_directory.name) / "lineitem-like.parquet"
    else:
        data_path = args.data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_data:
        actual_rows, rates = measure_data(data_path, args.predicate_count)
    else:
        rates = generate_data(
            data_path,
            predicate_count=args.predicate_count,
            rows=args.rows,
            row_group_size=args.row_group_size,
            shipmode_per_mille=args.shipmode_per_mille,
            shipinstruct_per_mille=args.shipinstruct_per_mille,
            seed=args.seed,
            correlation=args.correlation,
            mask_topology=args.mask_topology,
        )
        actual_rows = args.rows

    factories, stage_specs, classes = oracle_queries(args.predicate_count)
    result, plans = benchmark_28304.run(
        data_path,
        factories,
        stage_specs,
        set(),
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        include_samples=True,
    )
    audit = analyze_audit(
        result,
        classes,
        restricted_action_set=args.restricted_action_set,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "polars_version": pl.__version__,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
            "polars_thread_pool_size": pl.thread_pool_size(),
        },
        "data": {
            "path": str(data_path),
            "rows": actual_rows,
            "row_group_size": args.row_group_size,
            "row_groups": (actual_rows + args.row_group_size - 1)
            // args.row_group_size,
            "predicate_count": args.predicate_count,
            "predicate_columns": predicate_columns(args.predicate_count),
            "shipmode_per_mille": args.shipmode_per_mille,
            "shipinstruct_per_mille": args.shipinstruct_per_mille,
            "correlation": args.correlation,
            "mask_topology": args.mask_topology,
            "selectivity": rates,
        },
        "measurement": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "rotated_query_order": True,
            "bootstrap_resamples": args.bootstrap_resamples,
            "restricted_action_set": args.restricted_action_set,
        },
        "partition_count": len(stage_specs),
        "partition_classes": classes,
        "stage_specs": stage_specs,
        "audit": audit,
        **result,
        "plans": plans,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if not args.quiet:
        print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
