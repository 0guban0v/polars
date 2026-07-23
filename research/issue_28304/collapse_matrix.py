"""Run collapse audit across controlled workload and thread matrix."""

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
class Workload:
    name: str
    shipmode_per_mille: int
    shipinstruct_per_mille: int
    correlation: str = "independent"
    mask_topology: str = "random"


WORKLOADS = {
    workload.name: workload
    for workload in (
        Workload("sparse", 100, 100),
        Workload("medium", 316, 316),
        Workload("dense", 950, 950),
        Workload("disjoint", 142, 250, correlation="negative"),
        Workload("nested", 142, 250, correlation="positive"),
        Workload("alternating", 142, 250, mask_topology="alternating"),
        Workload("contiguous", 142, 250, mask_topology="contiguous"),
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


def run_audit(
    *,
    script: Path,
    output: Path,
    data_path: Path,
    workload: Workload,
    predicate_count: int,
    rows: int,
    row_group_size: int,
    threads: int,
    warmups: int,
    iterations: int,
    bootstrap_resamples: int,
    restricted_action_set: str,
    seed: int,
    reuse_data: bool,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--predicate-count",
        str(predicate_count),
        "--rows",
        str(rows),
        "--row-group-size",
        str(row_group_size),
        "--shipmode-per-mille",
        str(workload.shipmode_per_mille),
        "--shipinstruct-per-mille",
        str(workload.shipinstruct_per_mille),
        "--correlation",
        workload.correlation,
        "--mask-topology",
        workload.mask_topology,
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--restricted-action-set",
        restricted_action_set,
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


def summarize_run(
    report: dict[str, Any],
    *,
    workload: str,
    layout: str,
    result_path: Path,
) -> dict[str, Any]:
    audit = report["audit"]
    return {
        "workload": workload,
        "layout": layout,
        "threads": report["environment"]["polars_thread_pool_size"],
        "rows": report["data"]["rows"],
        "row_groups": report["data"]["row_groups"],
        "selectivity": report["data"]["selectivity"],
        "oracle": audit["oracle"],
        "restricted": audit["restricted"],
        "oracle_in_restricted_action_set": audit["oracle_in_restricted_action_set"],
        "best_by_class": audit["best_by_class"],
        "result_path": result_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate-count", type=int, choices=[3, 4], default=3)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--layouts", default="balanced,short_tail")
    parser.add_argument("--threads", default="1,8,16")
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--tail-rows", type=int, default=1_215)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument(
        "--restricted-action-set",
        choices=["defer_one", "concurrent_prefix"],
        default="defer_one",
    )
    parser.add_argument("--seed", type=int, default=28304)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload_names = parse_csv(args.workloads)
    unknown = sorted(set(workload_names) - set(WORKLOADS))
    if unknown:
        msg = f"unknown workloads: {', '.join(unknown)}"
        raise ValueError(msg)
    layouts = parse_csv(args.layouts)
    if not layouts or set(layouts) - {"balanced", "short_tail"}:
        msg = "--layouts supports balanced,short_tail"
        raise ValueError(msg)
    threads = parse_threads(args.threads)
    if args.rows < 1 or args.row_group_size < 1 or args.tail_rows < 1:
        msg = "row counts and row-group size must be positive"
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("collapse_audit.py")
    summaries = []
    for workload_name in workload_names:
        workload = WORKLOADS[workload_name]
        for layout in layouts:
            rows = args.rows if layout == "balanced" else args.rows + args.tail_rows
            stem = f"p{args.predicate_count}-{workload_name}-{layout}"
            data_path = args.output_dir / f"{stem}.parquet"
            for thread_index, thread_count in enumerate(threads):
                result_path = args.output_dir / f"{stem}-t{thread_count}.json"
                run_audit(
                    script=script,
                    output=result_path,
                    data_path=data_path,
                    workload=workload,
                    predicate_count=args.predicate_count,
                    rows=rows,
                    row_group_size=args.row_group_size,
                    threads=thread_count,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    bootstrap_resamples=args.bootstrap_resamples,
                    restricted_action_set=args.restricted_action_set,
                    seed=args.seed,
                    reuse_data=thread_index > 0,
                )
                report = json.loads(result_path.read_text())
                summaries.append(
                    summarize_run(
                        report,
                        workload=workload_name,
                        layout=layout,
                        result_path=result_path,
                    )
                )

    summary = {
        "configuration": {
            "predicate_count": args.predicate_count,
            "workloads": workload_names,
            "layouts": layouts,
            "threads": threads,
            "rows": args.rows,
            "row_group_size": args.row_group_size,
            "tail_rows": args.tail_rows,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "restricted_action_set": args.restricted_action_set,
            "seed": args.seed,
        },
        "runs": summaries,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = args.summary or args.output_dir / (
        f"collapse-matrix-p{args.predicate_count}.json"
    )
    summary_path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
