"""Run issue #28402 capability staging controls in fresh Polars processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    values = [int(item) for item in parse_csv(value)]
    if not values or any(item < 1 for item in values):
        msg = "integer lists must contain positive values"
        raise ValueError(msg)
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(item) for item in parse_csv(value)]
    if not values or any(not 0 < item < 1 for item in values):
        msg = "retentions must be between zero and one"
        raise ValueError(msg)
    return values


def run_benchmark(
    *,
    script: Path,
    output: Path,
    data_path: Path,
    rows: int,
    row_group_size: int,
    retention: float,
    relationship: str,
    residual_dtype: str,
    payload_width: int,
    residual_projection: str,
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
        "--rows",
        str(rows),
        "--row-group-size",
        str(row_group_size),
        "--prefix-retention",
        str(retention),
        "--relationship",
        relationship,
        "--residual-dtype",
        residual_dtype,
        "--payload-width",
        str(payload_width),
        "--residual-projection",
        residual_projection,
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


def summarize(report: dict[str, Any], result_path: Path) -> dict[str, Any]:
    return {
        "configuration": report["configuration"],
        "threads": report["environment"]["polars_thread_pool_size"],
        "observed_selectivity": report["observed_selectivity"],
        "baseline": report["baseline"],
        "queries": report["queries"],
        "comparisons": report["comparisons"],
        "result_path": result_path.name,
    }


def curate(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration": summary["configuration"],
        "runs": [
            {
                "configuration": run["configuration"],
                "threads": run["threads"],
                "observed_selectivity": run["observed_selectivity"],
                "baseline": run["baseline"],
                "query_medians": {
                    name: {
                        metric: values["median"] for metric, values in metrics.items()
                    }
                    for name, metrics in run["queries"].items()
                },
                "comparisons": run["comparisons"],
            }
            for run in summary["runs"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retentions", default="0.01,0.05,0.25,0.6,0.9")
    parser.add_argument("--relationships", default="independent,nested,negative")
    parser.add_argument("--residual-dtypes", default="float,int")
    parser.add_argument("--threads", default="1,8,16")
    parser.add_argument("--payload-widths", default="1")
    parser.add_argument("--residual-projections", default="projected,filter_only")
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=28402)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--curated-summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retentions = parse_float_csv(args.retentions)
    relationships = parse_csv(args.relationships)
    residual_dtypes = parse_csv(args.residual_dtypes)
    threads = parse_int_csv(args.threads)
    payload_widths = parse_int_csv(args.payload_widths)
    residual_projections = parse_csv(args.residual_projections)
    if set(relationships) - {"independent", "nested", "negative"}:
        msg = "unknown relationship"
        raise ValueError(msg)
    if set(residual_dtypes) - {"float", "int"}:
        msg = "unknown residual dtype"
        raise ValueError(msg)
    if set(residual_projections) - {"projected", "filter_only"}:
        msg = "unknown residual projection"
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("capability_benchmark.py")
    summaries = []
    for retention in retentions:
        retention_label = str(retention).replace(".", "p")
        for relationship in relationships:
            for payload_width in payload_widths:
                data_stem = f"r{retention_label}-{relationship}-w{payload_width}"
                data_path = args.output_dir / f"{data_stem}.parquet"
                reuse_data = False
                for residual_dtype in residual_dtypes:
                    for residual_projection in residual_projections:
                        for thread_count in threads:
                            result_path = args.output_dir / (
                                f"{data_stem}-{residual_dtype}-"
                                f"{residual_projection}-t{thread_count}.json"
                            )
                            run_benchmark(
                                script=script,
                                output=result_path,
                                data_path=data_path,
                                rows=args.rows,
                                row_group_size=args.row_group_size,
                                retention=retention,
                                relationship=relationship,
                                residual_dtype=residual_dtype,
                                payload_width=payload_width,
                                residual_projection=residual_projection,
                                threads=thread_count,
                                warmups=args.warmups,
                                iterations=args.iterations,
                                bootstrap_resamples=args.bootstrap_resamples,
                                seed=args.seed,
                                reuse_data=reuse_data,
                            )
                            reuse_data = True
                            report = json.loads(result_path.read_text())
                            summaries.append(summarize(report, result_path))

    summary = {
        "configuration": {
            "retentions": retentions,
            "relationships": relationships,
            "residual_dtypes": residual_dtypes,
            "threads": threads,
            "payload_widths": payload_widths,
            "residual_projections": residual_projections,
            "rows": args.rows,
            "row_group_size": args.row_group_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "runs": summaries,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = args.summary or args.output_dir / "capability-matrix.json"
    summary_path.write_text(rendered + "\n")
    if args.curated_summary is not None:
        args.curated_summary.parent.mkdir(parents=True, exist_ok=True)
        args.curated_summary.write_text(
            json.dumps(curate(summary), indent=2, sort_keys=True) + "\n"
        )
    print(rendered)


if __name__ == "__main__":
    main()
