# Mixed predicate capability staging for Polars #28402

This research branch is not intended for merge. It tests whether supported
Parquet column predicates can keep fast pre-filtering when same conjunction
contains Float predicate that disables current global fast path.

Current fallback decodes all predicate columns before evaluating full
conjunction:

```text
[supported strings, residual Float64]
```

Capability fanout decodes supported predicates first, combines their masks,
then decodes residual Float64 and projected payload concurrently for surviving
rows:

```text
[supported strings] -> [Float64 residual, projected payload]
```

Execution is enabled only by research environment variable. Default behavior
is unchanged.

## Result

Native fanout is broad win, but not unconditional rule.

Final screen contained 90 Float64 cases across prefix retention, predicate
relationship, projection, and thread count. Fanout improved all 90 by roughly
7.5–34.5%. Higher-signal sparse/dense, narrow/wide confirmation improved all
12 cases by roughly 2.3–37.3%, with intervals below zero.

Ten-million-row issue-like case improved 48.7% at 16 threads and was within
0.5% of slice-blocked reference. Native fanout median gap to reference was
1.2% in screen.

Work- and row-group-capacity-aware final filtering closes measured dense/wide
8–16-thread gap to roughly 0–1%. Extended 24-case confirmation improved 23
cases against fallback. One case regressed: 90% prefix retention, eight payload
columns, one thread was roughly 8% slower. Sequential materialization improved
that case roughly 10%, so remaining problem is schedule choice at one thread,
not final-filter task supply.

Mixed-capability consumer does not need PR #28485 selected-decode primitive.

See `findings.md` for measurements and limits.

## Reproduce

Build local extension:

```bash
make build-release
```

Run one case:

```bash
POLARS_MAX_THREADS=16 uv run python \
  research/issue_28402/capability_benchmark.py \
  --rows 4000000 \
  --row-group-size 1000000 \
  --prefix-retention 0.05 \
  --relationship independent \
  --residual-dtype float \
  --payload-width 1 \
  --residual-projection projected \
  --warmups 2 \
  --iterations 20 \
  --bootstrap-resamples 5000 \
  --fanout-min-task-values 1000000 \
  --output /private/tmp/issue-28402.json
```

Run controlled screen:

```bash
uv run python research/issue_28402/capability_matrix.py \
  --rows 1000000 \
  --row-group-size 125000 \
  --warmups 1 \
  --iterations 5 \
  --bootstrap-resamples 1000 \
  --output-dir /private/tmp/issue-28402-screen \
  --summary /private/tmp/issue-28402-screen.json \
  --curated-summary /private/tmp/issue-28402-screen-curated.json
```

`capability_benchmark.py` generates seeded Parquet data, verifies identical
results, rotates plan order, and records wall and process CPU samples.
`capability_matrix.py` isolates each thread count in fresh process.

## Research controls

```text
POLARS_ISSUE_28402_CAPABILITY_STAGING=selected
POLARS_ISSUE_28402_CAPABILITY_STAGING=materialized
POLARS_ISSUE_28402_CAPABILITY_STAGING=fanout
POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES=1000000
POLARS_ISSUE_28402_DECODE_TRACE=1
POLARS_ISSUE_28304_TRACE=1
```

- `selected` uses selected-predicate primitive from PR #28485 for residual
  predicate.
- `materialized` decodes residual values under prefix mask, then evaluates
  residual expression against compact DataFrame.
- `fanout` decodes residual and projected payload concurrently under prefix,
  evaluates residual once, then filters compact columns.
- fanout minimum task values enables local filter sizing. Task count is bounded
  by compact values, columns, pool width, and configured row-group prefetch
  capacity. Capacity is upper bound, not active row-group measurement.
- decode trace counts predicate and projected-column decodes.
- structured trace records stage rows, duration, and overlap.

## Artifacts

- `capability_benchmark.py`: single-case benchmark and correctness check.
- `capability_matrix.py`: controlled subprocess matrix and curated-summary
  generator.
- `findings.md`: decision-gate result and next experiment.
- `results/screening-curated.json`: 180-run screen, including 90 Float64
  cases.
- `results/core-confirmation-curated.json`: sparse/dense, narrow/wide
  confirmation.
- `results/row-groups-*.json`: fixed-size row-group supply runs.
- `results/fanout-screening-curated.json`: native fanout full screen.
- `results/fanout-core-confirmation-curated.json`: sparse/dense,
  narrow/wide fanout confirmation.
- `results/fanout-issue-like.json`: 10-million-row issue-like confirmation.
- `results/fanout-local-controls-curated.json`: 24-case work- and
  row-group-capacity-aware filter confirmation.

Generated Parquet files and intermediate matrix output stay outside repository.
