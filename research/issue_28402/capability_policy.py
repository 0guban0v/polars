"""Evaluate one-thread #28402 eligibility and schedule rules."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        msg = "thresholds must contain positive integers"
        raise ValueError(msg)
    return values


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    position = round((len(ordered) - 1) * probability)
    return ordered[position]


def cases_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for run in summary["runs"]:
        configuration = run["configuration"]
        if run["threads"] != 1 or configuration["residual_dtype"] != "float":
            continue
        medians = run["query_medians"]
        if not {
            "global_fallback",
            "capability_fanout",
            "capability_materialized",
        }.issubset(medians):
            continue

        fallback = medians["global_fallback"]["wall_seconds"]
        fanout = medians["capability_fanout"]["wall_seconds"]
        materialized = medians["capability_materialized"]["wall_seconds"]
        prefix_rows = round(
            run["observed_selectivity"]["prefix"] * configuration["row_group_size"]
        )
        cases.append(
            {
                "configuration": configuration,
                "file_rows": configuration["rows"],
                "speculative_values": prefix_rows * configuration["payload_width"],
                "fallback_wall_seconds": fallback,
                "fanout_wall_seconds": fanout,
                "materialized_wall_seconds": materialized,
                "oracle_schedule": min(
                    ("global_fallback", fallback),
                    ("fanout", fanout),
                    ("materialized", materialized),
                    key=lambda item: item[1],
                )[0],
                "query_medians": medians,
            }
        )
    if not cases:
        msg = "summary contains no one-thread Float64 capability cases"
        raise ValueError(msg)
    return cases


def evaluate(
    cases: list[dict[str, Any]],
    threshold: int,
    min_file_rows: int,
) -> dict[str, Any]:
    records = []
    for case in cases:
        eligible = case["file_rows"] >= min_file_rows
        if not eligible:
            schedule = "global_fallback"
        elif case["speculative_values"] <= threshold:
            schedule = "fanout"
        else:
            schedule = "materialized"
        wall = (
            case["fallback_wall_seconds"]
            if schedule == "global_fallback"
            else case[f"{schedule}_wall_seconds"]
        )
        fallback = case["fallback_wall_seconds"]
        oracle = min(
            fallback,
            case["fanout_wall_seconds"],
            case["materialized_wall_seconds"],
        )
        auto_name = f"capability_auto_{threshold}"
        auto = case["query_medians"].get(auto_name)
        records.append(
            {
                "payload_dtype": case["configuration"]["payload_dtype"],
                "payload_width": case["configuration"]["payload_width"],
                "prefix_retention": case["configuration"]["prefix_retention"],
                "residual_retention": case["configuration"]["residual_retention"],
                "row_group_size": case["configuration"]["row_group_size"],
                "file_rows": case["file_rows"],
                "eligible": eligible,
                "speculative_values": case["speculative_values"],
                "schedule": schedule,
                "oracle_schedule": case["oracle_schedule"],
                "change_vs_fallback": wall / fallback - 1,
                "regret_vs_oracle": wall / oracle - 1,
                "actual_auto_change_vs_fallback": (
                    None if auto is None else auto["wall_seconds"] / fallback - 1
                ),
            }
        )

    changes = [record["change_vs_fallback"] for record in records]
    regrets = [record["regret_vs_oracle"] for record in records]
    actual_changes = [
        record["actual_auto_change_vs_fallback"]
        for record in records
        if record["actual_auto_change_vs_fallback"] is not None
    ]
    return {
        "threshold": threshold,
        "min_file_rows": min_file_rows,
        "case_count": len(records),
        "fallback_choices": sum(
            record["schedule"] == "global_fallback" for record in records
        ),
        "fanout_choices": sum(record["schedule"] == "fanout" for record in records),
        "oracle_matches": sum(
            record["schedule"] == record["oracle_schedule"] for record in records
        ),
        "regressions_over_2pct": sum(change > 0.02 for change in changes),
        "median_change_vs_fallback": statistics.median(changes),
        "max_change_vs_fallback": max(changes),
        "median_regret_vs_oracle": statistics.median(regrets),
        "p95_regret_vs_oracle": percentile(regrets, 0.95),
        "max_regret_vs_oracle": max(regrets),
        "actual_auto_median_change_vs_fallback": (
            None if not actual_changes else statistics.median(actual_changes)
        ),
        "records": records,
    }


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records"}


def rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        metrics["regressions_over_2pct"],
        max(0.0, metrics["max_change_vs_fallback"] - 0.02),
        metrics["max_regret_vs_oracle"],
        metrics["p95_regret_vs_oracle"],
        metrics["median_regret_vs_oracle"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--min-file-rows", type=int, default=500_000)
    parser.add_argument("--fit-payload-dtypes", default="int")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_int_csv(args.thresholds)
    if args.min_file_rows < 1:
        msg = "--min-file-rows must be positive"
        raise ValueError(msg)
    fit_payload_dtypes = {
        item.strip() for item in args.fit_payload_dtypes.split(",") if item.strip()
    }
    cases = cases_from_summary(json.loads(args.summary.read_text()))
    fit_cases = [
        case
        for case in cases
        if case["configuration"]["payload_dtype"] in fit_payload_dtypes
    ]
    if not fit_cases:
        msg = "no cases match --fit-payload-dtypes"
        raise ValueError(msg)

    fit_evaluations = {
        threshold: evaluate(fit_cases, threshold, args.min_file_rows)
        for threshold in thresholds
    }
    selected_threshold = min(thresholds, key=lambda value: rank(fit_evaluations[value]))
    holdout_cases = [case for case in cases if case not in fit_cases]
    by_payload_dtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_payload_dtype[case["configuration"]["payload_dtype"]].append(case)

    report = {
        "fit_payload_dtypes": sorted(fit_payload_dtypes),
        "min_file_rows": args.min_file_rows,
        "selected_threshold": selected_threshold,
        "fit": {
            str(threshold): compact(metrics)
            for threshold, metrics in fit_evaluations.items()
        },
        "selected_fit": fit_evaluations[selected_threshold],
        "selected_holdout": (
            None
            if not holdout_cases
            else evaluate(holdout_cases, selected_threshold, args.min_file_rows)
        ),
        "selected_by_payload_dtype": {
            payload_dtype: compact(
                evaluate(dtype_cases, selected_threshold, args.min_file_rows)
            )
            for payload_dtype, dtype_cases in sorted(by_payload_dtype.items())
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
