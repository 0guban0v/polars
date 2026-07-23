"""Run rolling predicate policy audit in fresh Polars processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    pattern: tuple[str, ...]


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario("stationary_sparse", ("sparse",)),
        Scenario("stationary_dense", ("dense",)),
        Scenario("stationary_disjoint", ("disjoint",)),
        Scenario("stationary_nested", ("nested",)),
        Scenario("sparse_to_dense", ("sparse", "dense")),
        Scenario("dense_to_sparse", ("dense", "sparse")),
        Scenario("alternating", ("sparse", "dense")),
    )
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_threads(value: str) -> list[int]:
    threads = [int(item) for item in parse_csv(value)]
    if not threads or any(thread < 1 for thread in threads):
        msg = "--threads must contain positive integers"
        raise ValueError(msg)
    return threads


def expand_scenario(scenario: Scenario, row_groups: int) -> list[str]:
    if len(scenario.pattern) == 1:
        return [scenario.pattern[0]] * row_groups
    if scenario.name == "alternating":
        return [scenario.pattern[index % 2] for index in range(row_groups)]
    split = row_groups // 2
    return [scenario.pattern[0]] * split + [scenario.pattern[1]] * (row_groups - split)


def run_audit(
    *,
    script: Path,
    output: Path,
    data_path: Path,
    workloads: list[str],
    predicate_count: int,
    row_group_size: int,
    threads: int,
    warmups: int,
    iterations: int,
    bootstrap_resamples: int,
    seed: int,
    reuse_data: bool,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--predicate-count",
        str(predicate_count),
        "--workloads",
        ",".join(workloads),
        "--row-group-size",
        str(row_group_size),
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--seed",
        str(seed),
        "--data-path",
        str(data_path),
        "--output",
        str(output),
        "--quiet",
    ]
    if reuse_data:
        command.append("--reuse-data")
    environment = os.environ.copy()
    environment["POLARS_MAX_THREADS"] = str(threads)
    subprocess.run(command, env=environment, check=True)


def summarize(report: dict[str, Any], scenario: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "threads": report["environment"]["polars_thread_pool_size"],
        "row_groups": report["configuration"]["row_groups"],
        "selectivity": report["selectivity"],
        "all_at_once_median_wall_seconds": report["audit"][
            "all_at_once_median_wall_seconds"
        ],
        "fixed_oracle": report["audit"]["fixed_oracle"],
        "policies": report["audit"]["policies"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--threads", default="1,16")
    parser.add_argument("--row-groups", type=int, default=64)
    parser.add_argument("--row-group-size", type=int, default=125_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = parse_csv(args.scenarios)
    unknown = sorted(set(scenario_names) - set(SCENARIOS))
    if unknown:
        msg = f"unknown scenarios: {', '.join(unknown)}"
        raise ValueError(msg)
    threads = parse_threads(args.threads)
    if args.row_groups < 2 or args.row_group_size < 1:
        msg = "row-group count must be at least two and size must be positive"
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("rolling_policy_audit.py")
    runs = []
    for scenario_name in scenario_names:
        workloads = expand_scenario(SCENARIOS[scenario_name], args.row_groups)
        data_path = args.output_dir / f"p{args.predicate_count}-{scenario_name}.parquet"
        for thread_index, thread_count in enumerate(threads):
            result_path = args.output_dir / (
                f"p{args.predicate_count}-{scenario_name}-t{thread_count}.json"
            )
            run_audit(
                script=script,
                output=result_path,
                data_path=data_path,
                workloads=workloads,
                predicate_count=args.predicate_count,
                row_group_size=args.row_group_size,
                threads=thread_count,
                warmups=args.warmups,
                iterations=args.iterations,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
                reuse_data=thread_index > 0,
            )
            runs.append(summarize(json.loads(result_path.read_text()), scenario_name))

    summary = {
        "configuration": {
            "predicate_count": args.predicate_count,
            "scenarios": scenario_names,
            "threads": threads,
            "row_groups": args.row_groups,
            "row_group_size": args.row_group_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "runs": runs,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = args.summary or args.output_dir / (
        f"rolling-policy-matrix-p{args.predicate_count}.json"
    )
    summary_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
