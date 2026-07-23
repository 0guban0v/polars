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

Work- and row-group-capacity-aware final filtering closed measured dense/wide
8–16-thread gap to roughly 0–1% in targeted Phase 2 controls.

Phase 3 implements two research-only heuristics:

```text
eligible scan?
    no  -> current global fallback
    yes -> prefix-mask cardinality proxy selects fanout or materialization
```

Auto currently requires one pipeline and at least 500,000 metadata rows in
file. Inside eligible scan, fanout is used when:

```text
prefix rows * projected payload columns <= 1,000,000
```

Prefix rows and payload-column count are exact, but their product is value-count
proxy, not decode work. It does not capture byte width, encoding, compression,
page locality, nulls, or mask topology. Otherwise Auto uses sequential
materialization. Both thresholds remain explicit research controls.

Two-million-value threshold was rejected. Fixed-total 4M-row sweep exposed
roughly 37–44% regressions with 250k row groups because threshold selected
fanout for dense, wide payload. One-million threshold selected materialization
and stayed within 1% of fallback in same layout. This establishes 1M as
candidate separator, not calibrated transition.

One additional seed covered Int64, Float64, and short String payloads: 36/36
point estimates improved, all paired timing intervals were below zero, and
median change was roughly -16%. These are repeated warm-cache timings over
controlled configurations, not independent workload samples. Same predicate
masks are reused across payload dtypes. Frozen validation did not cover values
near 1M switch.

Five-hundred-thousand-row Float64/String control improved all 16 point
estimates, but 500k is first safe grid point rather than established boundary.
Two-hundred-fifty-thousand-row pre-guard regression was Int64, not String.
Metadata file rows can also overstate effective scan after statistics pruning.

Pre-eligibility Auto forced fanout whenever pipeline count exceeded one.
Its 8/16-thread Phase 2 control therefore tested unconditional local fanout,
not adaptive multi-thread scheduling: 8 of 72 cases regressed over 2%, up to
roughly 48%. Residual retention and payload type are informative, and global
fallback must remain available. Results do not prove residual prior is required
for every bounded multi-thread extension. Current Auto abstains above one
pipeline.

Phase 3 status is candidate one-thread heuristic with unresolved validation,
not closed policy. See `findings.md` for evidence boundaries and falsification
plan.

Mixed-capability consumer does not need PR #28485 selected-decode primitive.

## Reproduce

Build local extension:

```bash
make build-release
```

Run one case:

```bash
POLARS_MAX_THREADS=1 uv run python \
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
  --auto-max-speculative-values 1000000 \
  --auto-min-file-rows 500000 \
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
Current Auto exercises schedule only at one pipeline. Use explicit `fanout`
plus `POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES` to reproduce old multi-thread
local-fanout arm.

## Research controls

```text
POLARS_ISSUE_28402_CAPABILITY_STAGING=selected
POLARS_ISSUE_28402_CAPABILITY_STAGING=materialized
POLARS_ISSUE_28402_CAPABILITY_STAGING=fanout
POLARS_ISSUE_28402_CAPABILITY_STAGING=auto
POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES=1000000
POLARS_ISSUE_28402_AUTO_MAX_SPECULATIVE_VALUES=1000000
POLARS_ISSUE_28402_AUTO_MIN_FILE_ROWS=500000
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
- `auto` first checks metadata file rows and pipeline count. Ineligible scans
  keep global fallback. Eligible row groups use prefix rows times payload-column
  count as schedule proxy.
- decode trace counts predicate and projected-column decodes.
- structured trace records stage rows, duration, and overlap.

## Artifacts

- `capability_benchmark.py`: single-case benchmark and correctness check.
- `capability_matrix.py`: controlled subprocess matrix and curated-summary
  generator.
- `findings.md`: decision-gate result and next validation.
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
- `results/phase3-one-thread-confirmation-curated.json`: additional-seed
  Int64, Float64, and short-String timing control.
- `results/phase3-multithread-rejection-curated.json`: pre-eligibility
  unconditional-fanout control at 8/16 threads. Current Auto no longer takes
  this path.
- `results/phase3-fixed-total-rg-*.json`: fixed-total row-group-size sweep.
- `results/phase3-small-scan-controls-curated.json`: 500k Float64/String
  eligibility control, not boundary proof.
- `results/phase3-final-*.json`: provisional eligibility and 8/16-thread
  fallback checks against current Auto.

Generated Parquet files and intermediate matrix output stay outside repository.
Curated matrix artifacts omit raw timing samples, so exact intervals cannot be
recomputed from committed summaries alone.
