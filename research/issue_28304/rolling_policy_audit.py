"""Measure rolling defer-one policies for Polars issue #28304.

Research runner; not production policy. It compares four cumulative online
policy levels with all-at-once and fixed concurrent-prefix oracle. Policy sees
only completed row groups and applies decisions to later work.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

import benchmark as benchmark_28304
import collapse_audit
import pilot_audit
from collapse_matrix import WORKLOADS

POLICY_LEVELS = ("marginal", "joint", "cost", "task_supply")
TRACE_PATTERN = re.compile(
    r"POLARS_ISSUE_28304_ROLLING "
    r"issued=(?P<issued>\d+) "
    r"level=(?P<level>\w+) "
    r"plan=(?P<plan>\w+) "
    r"deferred=(?P<deferred>\w+) "
    r"predicted_prefix_selectivity=(?P<predicted_prefix_selectivity>[0-9.]+) "
    r"predicted_saved_fraction=(?P<predicted_saved_fraction>[0-9.]+) "
    r"in_flight=(?P<in_flight>\d+) "
    r"observed_groups=(?P<observed_groups>\d+) "
    r"reason=(?P<reason>\w+)"
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_workloads(names: list[str]) -> None:
    unknown = sorted(set(names) - set(WORKLOADS))
    if unknown:
        msg = f"unknown workloads: {', '.join(unknown)}"
        raise ValueError(msg)


def capture_policy_trace(
    query: pl.LazyFrame,
    policy: str,
) -> list[dict[str, Any]]:
    saved_stderr = os.dup(2)
    try:
        with tempfile.TemporaryFile() as trace_file:
            os.dup2(trace_file.fileno(), 2)
            os.environ["POLARS_ISSUE_28304_POLICY_TRACE"] = "1"
            try:
                benchmark_28304.execute(query, rolling_policy=policy)
            finally:
                os.environ.pop("POLARS_ISSUE_28304_POLICY_TRACE", None)
                os.dup2(saved_stderr, 2)
            trace_file.seek(0)
            trace_text = trace_file.read().decode()
    finally:
        os.close(saved_stderr)

    decisions = []
    for line in trace_text.splitlines():
        match = TRACE_PATTERN.search(line)
        if match is None:
            continue
        values = match.groupdict()
        decisions.append(
            {
                "issued": int(values["issued"]),
                "level": values["level"],
                "plan": values["plan"],
                "deferred": values["deferred"],
                "predicted_prefix_selectivity": float(
                    values["predicted_prefix_selectivity"]
                ),
                "predicted_saved_fraction": float(values["predicted_saved_fraction"]),
                "in_flight": int(values["in_flight"]),
                "observed_groups": int(values["observed_groups"]),
                "reason": values["reason"],
            }
        )
    decisions.sort(key=lambda decision: decision["issued"])
    return decisions


def summarize_trace(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_groups": len(decisions),
        "plans": dict(Counter(decision["plan"] for decision in decisions)),
        "deferred": dict(
            Counter(
                decision["deferred"]
                for decision in decisions
                if decision["deferred"] != "none"
            )
        ),
        "reasons": dict(Counter(decision["reason"] for decision in decisions)),
        "decisions": decisions,
    }


def analyze(
    result: dict[str, Any],
    *,
    fixed_names: list[str],
    bootstrap_resamples: int,
    seed: int,
    traces: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    medians = {
        name: metrics["wall_seconds"]["median"]
        for name, metrics in result["queries"].items()
    }
    oracle_name = min(fixed_names, key=medians.__getitem__)
    oracle_wall = medians[oracle_name]
    baseline_wall = medians["all_pushed"]
    policies = {}
    for level in POLICY_LEVELS:
        name = f"rolling_{level}"
        difference_vs_baseline = collapse_audit.bootstrap_median_difference(
            result["samples"][name]["wall_seconds"],
            result["samples"]["all_pushed"]["wall_seconds"],
            resamples=bootstrap_resamples,
            seed=seed,
        )
        difference_vs_oracle = collapse_audit.bootstrap_median_difference(
            result["samples"][name]["wall_seconds"],
            result["samples"][oracle_name]["wall_seconds"],
            resamples=bootstrap_resamples,
            seed=seed,
        )
        policies[level] = {
            "median_wall_seconds": medians[name],
            "change_vs_all_at_once": medians[name] / baseline_wall - 1,
            "regret_vs_fixed_oracle": medians[name] / oracle_wall - 1,
            "median_wall_difference_vs_all_at_once_seconds": difference_vs_baseline,
            "median_wall_difference_vs_fixed_oracle_seconds": difference_vs_oracle,
            "trace": summarize_trace(traces[level]),
        }
    return {
        "all_at_once_median_wall_seconds": baseline_wall,
        "fixed_oracle": {
            "name": oracle_name,
            "median_wall_seconds": oracle_wall,
            "change_vs_all_at_once": oracle_wall / baseline_wall - 1,
        },
        "policies": policies,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--workloads", default="sparse")
    parser.add_argument("--row-group-size", type=int, default=125_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = parse_csv(args.workloads)
    validate_workloads(workloads)
    if not workloads:
        msg = "--workloads cannot be empty"
        raise ValueError(msg)
    if args.row_group_size < 1 or args.iterations < 1:
        msg = "row-group size and iterations must be positive"
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
            prefix="polars-issue-28304-rolling-"
        )
        data_path = Path(temporary_directory.name) / "rolling.parquet"
    else:
        data_path = args.data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.reuse_data:
        pilot_audit.write_row_groups(
            data_path,
            workloads,
            predicate_count=args.predicate_count,
            row_group_size=args.row_group_size,
            seed=args.seed,
        )

    factories, stage_specs, classes = pilot_audit.concurrent_prefix_queries(
        args.predicate_count
    )
    fixed_names = list(factories)
    factory = collapse_audit.query_factory(args.predicate_count)
    rolling_policies = {}
    for level in POLICY_LEVELS:
        name = f"rolling_{level}"
        factories[name] = factory
        rolling_policies[name] = level

    result, plans = benchmark_28304.run(
        data_path,
        factories,
        stage_specs,
        set(),
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        include_samples=True,
        rolling_policies=rolling_policies,
    )
    trace_query = factory(data_path)
    traces = {
        level: capture_policy_trace(trace_query, level) for level in POLICY_LEVELS
    }
    audit = analyze(
        result,
        fixed_names=fixed_names,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        traces=traces,
    )
    _, selectivity = collapse_audit.measure_data(data_path, args.predicate_count)
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "polars_version": pl.__version__,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
            "polars_thread_pool_size": pl.thread_pool_size(),
        },
        "configuration": {
            "predicate_count": args.predicate_count,
            "workloads": workloads,
            "row_groups": len(workloads),
            "row_group_size": args.row_group_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "policy_levels": POLICY_LEVELS,
            "fixed_candidate_count": len(fixed_names),
            "selectivity_prior": 0.5,
            "row_group_discount": 0.5,
            "refresh_period": 8,
            "maximum_predicted_prefix_selectivity": 0.05,
            "minimum_predicted_saved_fraction": 0.05,
        },
        "selectivity": selectivity,
        "stage_specs": stage_specs,
        "partition_classes": classes,
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
