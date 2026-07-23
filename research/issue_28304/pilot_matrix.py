"""Run row-group-0 pilot stability audit in fresh Polars processes."""

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
    pilot: str
    remainder_pattern: tuple[str, ...]


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario("stationary_sparse", "sparse", ("sparse",)),
        Scenario("stationary_dense", "dense", ("dense",)),
        Scenario("sparse_to_dense", "sparse", ("dense",)),
        Scenario("dense_to_sparse", "dense", ("sparse",)),
        Scenario("alternating", "sparse", ("dense", "sparse")),
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


def expand_pattern(pattern: tuple[str, ...], count: int) -> list[str]:
    return [pattern[index % len(pattern)] for index in range(count)]


def run_scenario(
    *,
    script: Path,
    output: Path,
    data_dir: Path,
    scenario: Scenario,
    predicate_count: int,
    remainder_row_groups: int,
    row_group_size: int,
    threads: int,
    warmups: int,
    iterations: int,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    remainder = expand_pattern(scenario.remainder_pattern, remainder_row_groups)
    command = [
        sys.executable,
        str(script),
        "--predicate-count",
        str(predicate_count),
        "--pilot-workload",
        scenario.pilot,
        "--remainder-workloads",
        ",".join(remainder),
        "--remainder-row-groups",
        str(remainder_row_groups),
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
        "--data-dir",
        str(data_dir),
        "--output",
        str(output),
        "--quiet",
    ]
    environment = os.environ.copy()
    environment["POLARS_MAX_THREADS"] = str(threads)
    subprocess.run(command, env=environment, check=True)


def summarize(report: dict[str, Any], scenario: str) -> dict[str, Any]:
    audit = report["audit"]
    return {
        "scenario": scenario,
        "threads": report["environment"]["polars_thread_pool_size"],
        "pilot_workload": report["configuration"]["pilot_workload"],
        "remainder_workloads": report["configuration"]["remainder_workloads"],
        "pilot_winner": audit["pilot_winner"],
        "remainder_oracle": audit["remainder_oracle"],
        "pilot_winner_matches_remainder_oracle": audit[
            "pilot_winner_matches_remainder_oracle"
        ],
        "pilot_winner_remainder_regret": audit["pilot_winner_remainder_regret"],
        "pilot_winner_remainder_wall_difference_seconds": audit[
            "pilot_winner_remainder_wall_difference_seconds"
        ],
        "median_sum_proxy": audit["median_sum_proxy"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--threads", default="1,16")
    parser.add_argument("--remainder-row-groups", type=int, default=7)
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
    if args.remainder_row_groups < 1 or args.row_group_size < 1:
        msg = "row-group counts and size must be positive"
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("pilot_audit.py")
    runs = []
    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        for thread_count in threads:
            stem = f"p{args.predicate_count}-{scenario_name}-t{thread_count}"
            result_path = args.output_dir / f"{stem}.json"
            run_scenario(
                script=script,
                output=result_path,
                data_dir=args.output_dir / f"{stem}-data",
                scenario=scenario,
                predicate_count=args.predicate_count,
                remainder_row_groups=args.remainder_row_groups,
                row_group_size=args.row_group_size,
                threads=thread_count,
                warmups=args.warmups,
                iterations=args.iterations,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=args.seed,
            )
            runs.append(summarize(json.loads(result_path.read_text()), scenario_name))

    summary = {
        "configuration": {
            "predicate_count": args.predicate_count,
            "scenarios": scenario_names,
            "threads": threads,
            "remainder_row_groups": args.remainder_row_groups,
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
        f"pilot-matrix-p{args.predicate_count}.json"
    )
    summary_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
