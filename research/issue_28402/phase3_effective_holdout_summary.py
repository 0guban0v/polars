"""Validate and summarize frozen Phase 3 effective-scan holdout reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


AUTO_PLAN = "capability_auto_1000000"
FALLBACK_PLAN = "global_fallback"
ORACLE_PLANS = (
    FALLBACK_PLAN,
    "capability_fanout_local_1000000",
    "capability_materialized",
)
EXPECTED_SPECULATIVE_VALUES = {900_000, 1_100_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        action="append",
        type=Path,
        required=True,
        help="matrix output directory; repeat for each seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument(
        "--scope",
        required=True,
        help="bounded hardware and experiment scope described by this summary",
    )
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--cache-regime", required=True)
    return parser.parse_args()


def load_reports(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    reports = []
    for root in roots:
        for path in sorted(root.glob("*.json")):
            report = json.loads(path.read_text())
            if {"configuration", "samples", "queries"} <= report.keys():
                reports.append((path, report))
    if len(reports) != 32:
        msg = f"expected 32 raw reports, found {len(reports)}"
        raise ValueError(msg)
    return reports


def paired_ratio_change(
    candidate: list[float],
    baseline: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    candidate_array = np.asarray(candidate)
    baseline_array = np.asarray(baseline)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        candidate_array.size,
        size=(resamples, candidate_array.size),
    )
    changes = (
        np.median(candidate_array[indices], axis=1)
        / np.median(baseline_array[indices], axis=1)
        - 1
    )
    return {
        "estimate": float(np.median(candidate_array) / np.median(baseline_array) - 1),
        "ci95_low": float(np.quantile(changes, 0.025)),
        "ci95_high": float(np.quantile(changes, 0.975)),
    }


def validate_configuration(report: dict[str, Any]) -> int:
    configuration = report["configuration"]
    expected = {
        "rows": 4_000_000,
        "row_group_size": 1_000_000,
        "payload_width": 8,
        "relationship": "independent",
        "residual_dtype": "float",
        "residual_projection": "projected",
        "warmups": 3,
        "iterations": 15,
        "bootstrap_resamples": 5_000,
        "fanout_min_task_values": [1_000_000],
        "auto_max_speculative_values": [1_000_000],
        "auto_fanout_min_task_values": 1_000_000,
        "auto_min_file_rows": 500_000,
    }
    mismatches = {
        key: (configuration.get(key), value)
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if mismatches:
        msg = f"unexpected holdout configuration: {mismatches}"
        raise ValueError(msg)
    if configuration["seed"] not in {38_402, 38_403}:
        msg = f"unexpected seed: {configuration['seed']}"
        raise ValueError(msg)
    if configuration["payload_dtype"] not in {"float", "string"}:
        msg = f"unexpected payload: {configuration['payload_dtype']}"
        raise ValueError(msg)
    if configuration["residual_retention"] not in {0.1, 0.9}:
        msg = f"unexpected residual retention: {configuration['residual_retention']}"
        raise ValueError(msg)
    if configuration["mask_topology"] not in {"random", "clustered"}:
        msg = f"unexpected topology: {configuration['mask_topology']}"
        raise ValueError(msg)
    if report["environment"]["polars_thread_pool_size"] != 1:
        msg = "holdout report did not use one Polars thread"
        raise ValueError(msg)
    if len(report.get("execution_order", [])) != configuration["iterations"]:
        msg = "holdout report does not retain execution order"
        raise ValueError(msg)

    speculative_values = round(
        report["observed_selectivity"]["prefix"]
        * configuration["row_group_size"]
        * configuration["payload_width"]
    )
    if speculative_values not in EXPECTED_SPECULATIVE_VALUES:
        msg = f"unexpected per-row-group speculative values: {speculative_values}"
        raise ValueError(msg)
    return speculative_values


def summarize_record(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    speculative_values = validate_configuration(report)
    configuration = report["configuration"]
    samples = report["samples"]
    auto_comparison = report["comparisons"][AUTO_PLAN]["paired_wall_ratio_change"]
    query_medians = {
        name: report["queries"][name]["wall_seconds"]["median"] for name in ORACLE_PLANS
    }
    oracle_plan = min(query_medians, key=query_medians.__getitem__)
    oracle_comparison = paired_ratio_change(
        samples[AUTO_PLAN]["wall_seconds"],
        samples[oracle_plan]["wall_seconds"],
        resamples=configuration["bootstrap_resamples"],
        seed=configuration["seed"],
    )
    return {
        "seed": configuration["seed"],
        "speculative_values_per_row_group": speculative_values,
        "schedule": "fanout" if speculative_values <= 1_000_000 else "materialized",
        "topology": configuration["mask_topology"],
        "payload": configuration["payload_dtype"],
        "residual_retention": configuration["residual_retention"],
        "observed_prefix_retention": report["observed_selectivity"]["prefix"],
        "auto_change_vs_fallback": auto_comparison["estimate"],
        "paired_lower_vs_fallback": auto_comparison["ci95_low"],
        "paired_upper_vs_fallback": auto_comparison["ci95_high"],
        "oracle_plan": oracle_plan,
        "oracle_regret": max(0.0, oracle_comparison["estimate"]),
        "paired_lower_vs_oracle": oracle_comparison["ci95_low"],
        "paired_upper_vs_oracle": oracle_comparison["ci95_high"],
        "raw_report": f"seed-{configuration['seed']}/{path.name}",
    }


def main() -> None:
    args = parse_args()
    reports = load_reports(args.report_root)
    records = [summarize_record(path, report) for path, report in reports]
    records.sort(
        key=lambda record: (
            record["seed"],
            record["speculative_values_per_row_group"],
            record["topology"],
            record["payload"],
            record["residual_retention"],
        )
    )

    provenance = {
        json.dumps(report["provenance"], sort_keys=True) for _, report in reports
    }
    if len(provenance) != 1:
        msg = f"holdout reports span {len(provenance)} execution source states"
        raise ValueError(msg)
    environments = {
        json.dumps(report["environment"], sort_keys=True) for _, report in reports
    }
    if len(environments) != 1:
        msg = f"holdout reports span {len(environments)} runtime environments"
        raise ValueError(msg)

    regression_bound = 0.02
    passed = [
        record
        for record in records
        if record["paired_upper_vs_fallback"] <= regression_bound
    ]
    failed = [
        record
        for record in records
        if record["paired_lower_vs_fallback"] > regression_bound
    ]
    inconclusive = [
        record for record in records if record not in passed and record not in failed
    ]
    point_wins = sum(record["auto_change_vs_fallback"] < 0 for record in records)
    paired_wins = sum(
        candidate < baseline
        for _, report in reports
        for candidate, baseline in zip(
            report["samples"][AUTO_PLAN]["wall_seconds"],
            report["samples"][FALLBACK_PLAN]["wall_seconds"],
            strict=True,
        )
    )

    summary = {
        "policy": {
            "threads": 1,
            "max_speculative_values_per_row_group": 1_000_000,
            "min_post_statistics_candidate_rows": 500_000,
            "normalized_slices_eligible": False,
            "regression_bound": regression_bound,
        },
        "gate": {
            "scope": args.scope,
            "controlled_cell_count": len(records),
            "point_wins": point_wins,
            "paired_measurement_wins": paired_wins,
            "passed": len(passed),
            "inconclusive": len(inconclusive),
            "failed": len(failed),
            "weakest_point": max(
                records, key=lambda record: record["auto_change_vs_fallback"]
            ),
            "worst_upper_bound": max(
                records, key=lambda record: record["paired_upper_vs_fallback"]
            ),
            "clustered_high_residual_regret": [
                record
                for record in records
                if record["speculative_values_per_row_group"] == 1_100_000
                and record["topology"] == "clustered"
                and record["residual_retention"] == 0.9
            ],
        },
        "source": {
            "base_commit": args.base_commit,
            **json.loads(next(iter(provenance))),
        },
        "environment": {
            **json.loads(next(iter(environments))),
            "hardware": args.hardware,
            "cache_regime": args.cache_regime,
        },
        "matrix": {
            "rows": 4_000_000,
            "row_group_size": 1_000_000,
            "payload_width": 8,
            "speculative_values_per_row_group": [900_000, 1_100_000],
            "seeds": [38_402, 38_403],
            "payloads": ["float", "string"],
            "residual_retentions": [0.1, 0.9],
            "topologies": ["random", "clustered"],
            "warmups": 3,
            "iterations": 15,
            "bootstrap_resamples": 5_000,
            "controlled_cells_are_independent_workloads": False,
        },
        "records": records,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(
        f"{len(passed)}/{len(records)} cells passed; "
        f"{paired_wins} paired measurements favored Auto; "
        f"worst upper bound "
        f"{summary['gate']['worst_upper_bound']['paired_upper_vs_fallback']:.2%}"
    )


if __name__ == "__main__":
    main()
