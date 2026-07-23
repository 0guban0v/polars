"""Audit whether row-group-0 oracle predicts later predicate grouping.

Research runner for issue #28304; not production policy. It exhaustively times
the restricted concurrent-prefix action set on one pilot row group, selects
the pilot winner, and measures that action against restricted oracle on later
row groups. This is an upper bound on pilot usefulness, not deployable policy:
selection uses candidate timings rather than only all-at-once trace features.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

import benchmark as benchmark_28304
import collapse_audit
from collapse_matrix import WORKLOADS


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_remainder_workloads(
    value: str,
    *,
    pilot_workload: str,
    remainder_row_groups: int,
) -> list[str]:
    names = parse_csv(value) if value else [pilot_workload]
    unknown = sorted(set(names) - set(WORKLOADS))
    if unknown:
        msg = f"unknown workloads: {', '.join(unknown)}"
        raise ValueError(msg)
    if len(names) == 1:
        return names * remainder_row_groups
    if len(names) != remainder_row_groups:
        msg = (
            "--remainder-workloads must contain one workload or exactly "
            f"{remainder_row_groups} workloads"
        )
        raise ValueError(msg)
    return names


def write_row_groups(
    path: Path,
    workload_names: list[str],
    *,
    predicate_count: int,
    row_group_size: int,
    seed: int,
) -> None:
    frames = []
    with tempfile.TemporaryDirectory(prefix="polars-issue-28304-pilot-groups-") as tmp:
        temporary_path = Path(tmp)
        for index, workload_name in enumerate(workload_names):
            workload = WORKLOADS[workload_name]
            group_path = temporary_path / f"group-{index:03d}.parquet"
            collapse_audit.generate_data(
                group_path,
                predicate_count=predicate_count,
                rows=row_group_size,
                row_group_size=row_group_size,
                shipmode_per_mille=workload.shipmode_per_mille,
                shipinstruct_per_mille=workload.shipinstruct_per_mille,
                seed=seed + index,
                correlation=workload.correlation,
                mask_topology=workload.mask_topology,
            )
            frames.append(pl.read_parquet(group_path))

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(
        path,
        compression="zstd",
        row_group_size=row_group_size,
        statistics=True,
    )


def concurrent_prefix_queries(
    predicate_count: int,
) -> tuple[
    dict[str, benchmark_28304.QueryFactory],
    dict[str, str],
    dict[str, str],
]:
    factories, stage_specs, classes = collapse_audit.oracle_queries(predicate_count)
    allowed_classes = {"all_at_once", "defer_one", "concurrent_prefix"}
    selected_names = ["all_pushed"]
    selected_names.extend(
        name
        for name, spec in stage_specs.items()
        if "|" in spec and classes[name] in allowed_classes
    )
    return (
        {name: factories[name] for name in selected_names},
        {name: stage_specs[name] for name in selected_names if name in stage_specs},
        {name: classes[name] for name in selected_names},
    )


def medians(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: metrics["wall_seconds"]["median"]
        for name, metrics in result["queries"].items()
    }


def describe_action(
    name: str,
    *,
    values: dict[str, float],
    stage_specs: dict[str, str],
    classes: dict[str, str],
) -> dict[str, Any]:
    return {
        "name": name,
        "class": classes[name],
        "stage_spec": stage_specs.get(name),
        "median_wall_seconds": values[name],
    }


def analyze_pilot(
    pilot_result: dict[str, Any],
    remainder_result: dict[str, Any],
    *,
    stage_specs: dict[str, str],
    classes: dict[str, str],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    pilot_medians = medians(pilot_result)
    remainder_medians = medians(remainder_result)
    pilot_winner = min(pilot_medians, key=pilot_medians.__getitem__)
    remainder_oracle = min(remainder_medians, key=remainder_medians.__getitem__)

    difference = collapse_audit.bootstrap_median_difference(
        remainder_result["samples"][pilot_winner]["wall_seconds"],
        remainder_result["samples"][remainder_oracle]["wall_seconds"],
        resamples=bootstrap_resamples,
        seed=seed,
    )
    remainder_regret = (
        remainder_medians[pilot_winner] / remainder_medians[remainder_oracle] - 1
    )

    pilot_baseline = pilot_medians["all_pushed"]
    remainder_baseline = remainder_medians["all_pushed"]
    policy_sum = pilot_baseline + remainder_medians[pilot_winner]
    baseline_sum = pilot_baseline + remainder_baseline
    static_oracle_name = min(
        pilot_medians,
        key=lambda name: pilot_medians[name] + remainder_medians[name],
    )
    static_oracle_sum = (
        pilot_medians[static_oracle_name] + remainder_medians[static_oracle_name]
    )

    return {
        "pilot_winner": describe_action(
            pilot_winner,
            values=pilot_medians,
            stage_specs=stage_specs,
            classes=classes,
        ),
        "remainder_oracle": describe_action(
            remainder_oracle,
            values=remainder_medians,
            stage_specs=stage_specs,
            classes=classes,
        ),
        "pilot_winner_matches_remainder_oracle": pilot_winner == remainder_oracle,
        "pilot_winner_remainder_regret": remainder_regret,
        "pilot_winner_remainder_wall_difference_seconds": difference,
        "median_sum_proxy": {
            "pilot_baseline_then_selected": policy_sum,
            "all_at_once": baseline_sum,
            "change_vs_all_at_once": policy_sum / baseline_sum - 1,
            "static_oracle_name": static_oracle_name,
            "static_oracle": static_oracle_sum,
            "regret_vs_static_oracle": policy_sum / static_oracle_sum - 1,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--pilot-workload", choices=sorted(WORKLOADS), default="sparse")
    parser.add_argument("--remainder-workloads", default="")
    parser.add_argument("--remainder-row-groups", type=int, default=7)
    parser.add_argument("--row-group-size", type=int, default=125_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.remainder_row_groups < 1:
        msg = "--remainder-row-groups must be positive"
        raise ValueError(msg)
    if args.row_group_size < 1 or args.iterations < 1:
        msg = "row-group size and iterations must be positive"
        raise ValueError(msg)
    if args.bootstrap_resamples < 1:
        msg = "--bootstrap-resamples must be positive"
        raise ValueError(msg)

    remainder_workloads = resolve_remainder_workloads(
        args.remainder_workloads,
        pilot_workload=args.pilot_workload,
        remainder_row_groups=args.remainder_row_groups,
    )

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.data_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="polars-issue-28304-pilot-"
        )
        data_dir = Path(temporary_directory.name)
    else:
        data_dir = args.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

    pilot_path = data_dir / "pilot.parquet"
    remainder_path = data_dir / "remainder.parquet"
    write_row_groups(
        pilot_path,
        [args.pilot_workload],
        predicate_count=args.predicate_count,
        row_group_size=args.row_group_size,
        seed=args.seed,
    )
    write_row_groups(
        remainder_path,
        remainder_workloads,
        predicate_count=args.predicate_count,
        row_group_size=args.row_group_size,
        seed=args.seed + 10_000,
    )

    factories, stage_specs, classes = concurrent_prefix_queries(args.predicate_count)
    pilot_result, _ = benchmark_28304.run(
        pilot_path,
        factories,
        stage_specs,
        set(),
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        include_samples=True,
    )
    remainder_result, _ = benchmark_28304.run(
        remainder_path,
        factories,
        stage_specs,
        set(),
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed + 1,
        include_samples=True,
    )
    audit = analyze_pilot(
        pilot_result,
        remainder_result,
        stage_specs=stage_specs,
        classes=classes,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    _, pilot_selectivity = collapse_audit.measure_data(pilot_path, args.predicate_count)
    _, remainder_selectivity = collapse_audit.measure_data(
        remainder_path, args.predicate_count
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
            "predicate_count": args.predicate_count,
            "candidate_count": len(factories),
            "pilot_workload": args.pilot_workload,
            "remainder_workloads": remainder_workloads,
            "remainder_row_groups": args.remainder_row_groups,
            "row_group_size": args.row_group_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "selectivity": {
            "pilot": pilot_selectivity,
            "remainder": remainder_selectivity,
        },
        "stage_specs": stage_specs,
        "audit": audit,
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
