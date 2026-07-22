"""Parquet predicate decode placement reproducer for issue #28304.

Research runner; not CI assertion. Use release build and compare wall time with
process CPU time. Generated data uses multiple row groups with possible matches
for every predicate, preventing row-group statistics from eliminating work.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

QueryFactory = Callable[[Path], pl.LazyFrame]

STAGE_ORACLE_SPECS = (
    "l_shipmode,l_shipinstruct,l_quantity",
    "l_shipmode|l_shipinstruct,l_quantity",
    "l_shipinstruct|l_shipmode,l_quantity",
    "l_quantity|l_shipmode,l_shipinstruct",
    "l_shipmode,l_shipinstruct|l_quantity",
    "l_shipmode,l_quantity|l_shipinstruct",
    "l_shipinstruct,l_quantity|l_shipmode",
    "l_shipmode|l_shipinstruct|l_quantity",
    "l_shipmode|l_quantity|l_shipinstruct",
    "l_shipinstruct|l_shipmode|l_quantity",
    "l_shipinstruct|l_quantity|l_shipmode",
    "l_quantity|l_shipmode|l_shipinstruct",
    "l_quantity|l_shipinstruct|l_shipmode",
)


def generate_data(
    path: Path,
    *,
    rows: int,
    row_group_size: int,
    shipmode_per_mille: int,
    shipinstruct_per_mille: int,
    seed: int,
    correlation: str,
    mask_topology: str,
) -> dict[str, float]:
    """Generate deterministic data with issue-like predicate selectivities."""
    if not 0 < shipmode_per_mille < 1_000:
        msg = "shipmode_per_mille must be between 1 and 999"
        raise ValueError(msg)
    if not 0 < shipinstruct_per_mille < 1_000:
        msg = "shipinstruct_per_mille must be between 1 and 999"
        raise ValueError(msg)

    generator = np.random.default_rng(seed)
    ship_codes = generator.integers(0, 1_000, size=rows, dtype=np.uint16)
    if correlation == "independent":
        instruct_codes = generator.integers(0, 1_000, size=rows, dtype=np.uint16)
    elif correlation == "positive":
        instruct_codes = ship_codes.copy()
    elif correlation == "negative":
        instruct_codes = 999 - ship_codes
    else:
        msg = f"unknown correlation: {correlation}"
        raise ValueError(msg)
    frame = (
        pl.DataFrame(
            {
                "_ship_code": ship_codes,
                "_instruct_code": instruct_codes,
                "l_quantity": generator.integers(1, 51, size=rows, dtype=np.uint8),
            }
        )
        .with_columns(
            pl.when(pl.col("_ship_code") < shipmode_per_mille // 2)
            .then(pl.lit("AIR REG"))
            .when(pl.col("_ship_code") < shipmode_per_mille)
            .then(pl.lit("AIR"))
            .otherwise(pl.lit("SHIP"))
            .alias("l_shipmode"),
            pl.when(pl.col("_instruct_code") < shipinstruct_per_mille)
            .then(pl.lit("DELIVER IN PERSON"))
            .otherwise(pl.lit("TAKE BACK RETURN"))
            .alias("l_shipinstruct"),
            pl.col("l_quantity").cast(pl.Int32),
        )
        .select("l_shipmode", "l_shipinstruct", "l_quantity")
    )

    if mask_topology != "random":
        chunks = []
        conjunction = pl.col("l_shipmode").is_in(["AIR", "AIR REG"]) & (
            pl.col("l_shipinstruct") == "DELIVER IN PERSON"
        )
        for offset in range(0, rows, row_group_size):
            chunk = frame.slice(offset, row_group_size)
            mask = chunk.select(conjunction).to_series().to_numpy()
            true_indices = np.flatnonzero(mask)
            false_indices = np.flatnonzero(~mask)
            if mask_topology == "contiguous":
                order = np.concatenate((true_indices, false_indices))
            elif mask_topology == "alternating":
                order = np.empty(chunk.height, dtype=np.int64)
                true_positions = (
                    np.arange(true_indices.size) * chunk.height // true_indices.size
                    if true_indices.size
                    else np.empty(0, dtype=np.int64)
                )
                is_true_position = np.zeros(chunk.height, dtype=bool)
                is_true_position[true_positions] = True
                order[is_true_position] = true_indices
                order[~is_true_position] = false_indices
            else:
                msg = f"unknown mask topology: {mask_topology}"
                raise ValueError(msg)
            chunks.append(chunk[order, :])
        frame = pl.concat(chunks)

    shipmode = pl.col("l_shipmode").is_in(["AIR", "AIR REG"])
    shipinstruct = pl.col("l_shipinstruct") == "DELIVER IN PERSON"
    quantity = pl.col("l_quantity").is_between(1, 30)
    rates = frame.select(
        shipmode.mean().alias("shipmode"),
        shipinstruct.mean().alias("shipinstruct"),
        quantity.mean().alias("quantity"),
        (shipmode & shipinstruct).mean().alias("ship_conjunction"),
        (shipmode & shipinstruct & quantity).mean().alias("all_conjunction"),
    ).row(0, named=True)

    frame.write_parquet(
        path,
        compression="zstd",
        row_group_size=row_group_size,
        statistics=True,
    )
    return {name: float(value) for name, value in rates.items()}


def measure_data(path: Path) -> tuple[int, dict[str, float]]:
    frame = pl.scan_parquet(path)
    exprs = predicates()
    result = frame.select(
        pl.len().alias("rows"),
        exprs["shipmode"].mean().alias("shipmode"),
        exprs["shipinstruct"].mean().alias("shipinstruct"),
        exprs["quantity"].mean().alias("quantity"),
        (exprs["shipmode"] & exprs["shipinstruct"]).mean().alias("ship_conjunction"),
        (exprs["shipmode"] & exprs["shipinstruct"] & exprs["quantity"])
        .mean()
        .alias("all_conjunction"),
    ).collect()
    row = result.row(0, named=True)
    rows = int(row.pop("rows"))
    return rows, {name: float(value) for name, value in row.items()}


def predicates() -> dict[str, pl.Expr]:
    return {
        "shipmode": pl.col("l_shipmode").is_in(["AIR", "AIR REG"]),
        "shipinstruct": pl.col("l_shipinstruct") == "DELIVER IN PERSON",
        "quantity": pl.col("l_quantity").is_between(1, 30),
    }


def all_pushed_factory(order: Sequence[str]) -> QueryFactory:
    def make(path: Path) -> pl.LazyFrame:
        query = pl.scan_parquet(path, parallel="prefiltered")
        exprs = predicates()
        for name in order:
            query = query.filter(exprs[name])
        return query.select(pl.col("l_quantity").sum())

    return make


def quantity_blocked(path: Path) -> pl.LazyFrame:
    exprs = predicates()
    return (
        pl.scan_parquet(path, parallel="prefiltered")
        .filter(exprs["shipmode"])
        .filter(exprs["shipinstruct"])
        .slice(0, None)
        .filter(exprs["quantity"])
        .select(pl.col("l_quantity").sum())
    )


def split_factory(early: frozenset[str]) -> QueryFactory:
    order = ("shipmode", "shipinstruct", "quantity")

    def make(path: Path) -> pl.LazyFrame:
        query = pl.scan_parquet(path, parallel="prefiltered")
        exprs = predicates()
        for name in order:
            if name in early:
                query = query.filter(exprs[name])
        query = query.slice(0, None)
        for name in order:
            if name not in early:
                query = query.filter(exprs[name])
        return query.select(pl.col("l_quantity").sum())

    return make


def query_factories(
    *,
    include_permutations: bool,
    include_splits: bool,
    include_stage_oracle: bool,
    candidate_stage_spec: str | None,
    include_adaptive_candidate: bool,
    all_pushed_only: bool,
) -> tuple[dict[str, QueryFactory], dict[str, str], set[str]]:
    base_order = ("shipmode", "shipinstruct", "quantity")
    factories: dict[str, QueryFactory] = {
        "all_pushed": all_pushed_factory(base_order),
    }
    if not all_pushed_only:
        factories["quantity_blocked"] = quantity_blocked
    if include_permutations:
        for order in itertools.permutations(base_order):
            name = "pushed_" + "_".join(order)
            factories[name] = all_pushed_factory(order)
    if include_splits:
        for early_size in range(len(base_order)):
            for early_tuple in itertools.combinations(base_order, early_size):
                early = frozenset(early_tuple)
                suffix = "_".join(early_tuple) if early_tuple else "none"
                factories[f"split_early_{suffix}"] = split_factory(early)
    stage_specs: dict[str, str] = {}
    adaptive_names: set[str] = set()
    if include_stage_oracle:
        for i, spec in enumerate(STAGE_ORACLE_SPECS):
            name = f"staged_{i:02d}_{spec.replace(',', '_').replace('|', '__')}"
            factories[name] = all_pushed_factory(base_order)
            stage_specs[name] = spec
    if candidate_stage_spec is not None:
        name = "staged_candidate"
        factories[name] = all_pushed_factory(base_order)
        stage_specs[name] = candidate_stage_spec
        if include_adaptive_candidate:
            name = "adaptive_candidate"
            factories[name] = all_pushed_factory(base_order)
            stage_specs[name] = candidate_stage_spec
            adaptive_names.add(name)
    elif include_adaptive_candidate:
        msg = "--include-adaptive-candidate requires --candidate-stage-spec"
        raise ValueError(msg)
    return factories, stage_specs, adaptive_names


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


def execute(
    query: pl.LazyFrame,
    stage_spec: str | None = None,
    *,
    adaptive: bool = False,
) -> tuple[Any, float, float]:
    if stage_spec is None:
        os.environ.pop("POLARS_ISSUE_28304_STAGES", None)
    else:
        os.environ["POLARS_ISSUE_28304_STAGES"] = stage_spec
    if adaptive:
        os.environ["POLARS_ISSUE_28304_ADAPTIVE"] = "1"
    else:
        os.environ.pop("POLARS_ISSUE_28304_ADAPTIVE", None)
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = query.collect(engine="streaming").item()
    cpu_seconds = (time.process_time_ns() - cpu_start) / 1e9
    wall_seconds = (time.perf_counter_ns() - wall_start) / 1e9
    return result, wall_seconds, cpu_seconds


def run(
    path: Path,
    factories: dict[str, QueryFactory],
    stage_specs: dict[str, str],
    adaptive_names: set[str],
    *,
    warmups: int,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    queries = {name: factory(path) for name, factory in factories.items()}
    plans = {
        name: query.explain(optimized=True, engine="streaming")
        for name, query in queries.items()
    }

    expected: Any | None = None
    for name, query in queries.items():
        result, _, _ = execute(
            query, stage_specs.get(name), adaptive=name in adaptive_names
        )
        if expected is None:
            expected = result
        elif result != expected:
            msg = f"query {name!r} returned {result!r}; expected {expected!r}"
            raise AssertionError(msg)

    for _ in range(warmups):
        for name, query in queries.items():
            execute(query, stage_specs.get(name), adaptive=name in adaptive_names)

    samples: dict[str, dict[str, list[float]]] = {
        name: {"wall_seconds": [], "cpu_seconds": []} for name in queries
    }
    names = list(queries)
    generator = random.Random(seed)
    for _ in range(iterations):
        generator.shuffle(names)
        for name in names:
            gc.collect()
            _, wall_seconds, cpu_seconds = execute(
                queries[name],
                stage_specs.get(name),
                adaptive=name in adaptive_names,
            )
            samples[name]["wall_seconds"].append(wall_seconds)
            samples[name]["cpu_seconds"].append(cpu_seconds)

    results = {
        name: {metric: summarize(values) for metric, values in query_samples.items()}
        for name, query_samples in samples.items()
    }
    return {"expected_result": expected, "queries": results}, plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-permutations", action="store_true")
    parser.add_argument("--include-splits", action="store_true")
    parser.add_argument("--include-stage-oracle", action="store_true")
    parser.add_argument("--candidate-stage-spec")
    parser.add_argument("--include-adaptive-candidate", action="store_true")
    parser.add_argument("--all-pushed-only", action="store_true")
    parser.add_argument("--print-plans", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.data_path is None:
        if args.reuse_data:
            msg = "--reuse-data requires --data-path"
            raise ValueError(msg)
        temporary_directory = tempfile.TemporaryDirectory(prefix="polars-issue-28304-")
        data_path = Path(temporary_directory.name) / "lineitem-like.parquet"
    else:
        data_path = args.data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_data:
        actual_rows, rates = measure_data(data_path)
    else:
        rates = generate_data(
            data_path,
            rows=args.rows,
            row_group_size=args.row_group_size,
            shipmode_per_mille=args.shipmode_per_mille,
            shipinstruct_per_mille=args.shipinstruct_per_mille,
            seed=args.seed,
            correlation=args.correlation,
            mask_topology=args.mask_topology,
        )
        actual_rows = args.rows
    factories, stage_specs, adaptive_names = query_factories(
        include_permutations=args.include_permutations,
        include_splits=args.include_splits,
        include_stage_oracle=args.include_stage_oracle,
        candidate_stage_spec=args.candidate_stage_spec,
        include_adaptive_candidate=args.include_adaptive_candidate,
        all_pushed_only=args.all_pushed_only,
    )
    external_stage_spec = os.environ.get("POLARS_ISSUE_28304_STAGES")
    if external_stage_spec is not None and not args.include_stage_oracle:
        if not args.all_pushed_only:
            msg = "external stage specification requires --all-pushed-only"
            raise ValueError(msg)
        stage_specs.update(dict.fromkeys(factories, external_stage_spec))
    result, plans = run(
        data_path,
        factories,
        stage_specs,
        adaptive_names,
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
    )
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "polars_version": pl.__version__,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
            "polars_thread_pool_size": pl.thread_pool_size(),
            "issue_28304_stages": os.environ.get("POLARS_ISSUE_28304_STAGES"),
            "issue_28304_adaptive": os.environ.get("POLARS_ISSUE_28304_ADAPTIVE"),
        },
        "data": {
            "path": str(data_path),
            "rows": actual_rows,
            "row_group_size": args.row_group_size,
            "row_groups": (actual_rows + args.row_group_size - 1)
            // args.row_group_size,
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
        },
        "stage_specs": stage_specs,
        "adaptive_queries": sorted(adaptive_names),
        **result,
        "plans": plans,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if not args.quiet:
        print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    if args.print_plans:
        for name, plan in plans.items():
            print(f"\n--- {name} ---\n{plan}")

    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
