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

Auto requires one pipeline, no normalized slice, and at least 500,000
post-statistics candidate rows. Inside eligible scan, fanout is used when:

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
masks are reused across payload dtypes.

Five-hundred-thousand-row Float64/String control improved all 16 point
estimates, but 500k is first safe grid point rather than established boundary.
Two-hundred-fifty-thousand-row pre-guard regression was Int64, not String.
Auto now computes post-statistics candidate rows before applying guard. Focused
tests prove large file pruned to one 10k row group falls back.

Frozen M4 boundary holdout used exact 0.9M/1.1M speculative values per row
group, two unseen seeds, random/clustered masks, Float64/short-String payloads,
and 10/90% residual retention. All 32 controlled cells passed targeted 2%
schedule-boundary gate. All 480 paired measurements favored Auto. Weakest
point was 4.2% faster and worst paired ratio-of-medians upper bound was 3.4%
faster than fallback.

All four clustered 1.1M/high-residual cells had 9.9–14.4% regret against local
fanout. Residual retention, payload cost, or prefix-mask topology may recover
this opportunity. Result supports measured M4 schedule switch; it does not
validate 500k guard, portable threshold, or production policy. Cells are
controlled configurations, not independent workloads.

Pre-eligibility Auto forced fanout whenever pipeline count exceeded one.
Its 8/16-thread Phase 2 control therefore tested unconditional local fanout,
not adaptive multi-thread scheduling: 8 of 72 cases regressed over 2%, up to
roughly 48%. Residual retention and payload type are informative, and global
fallback must remain available. Results do not prove residual prior is required
for every bounded multi-thread extension. Current Auto abstains above one
pipeline.

Phase 3 status is M4-local one-thread schedule candidate with provisional 500k
eligibility guard. Production policy remains open. See `findings.md`.

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
  --mask-topology bernoulli \
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

Run exact effective-scan holdout from repository root after `make
build-release`:

```bash
for seed in 38402 38403; do
  .venv/bin/python research/issue_28402/capability_matrix.py \
    --rows 4000000 \
    --row-group-size 1000000 \
    --retentions 0.1125,0.1375 \
    --residual-retentions 0.1,0.9 \
    --relationships independent \
    --mask-topologies random,clustered \
    --residual-dtypes float \
    --threads 1 \
    --payload-widths 8 \
    --payload-dtypes float,string \
    --residual-projections projected \
    --warmups 3 \
    --iterations 15 \
    --bootstrap-resamples 5000 \
    --fanout-min-task-values 1000000 \
    --auto-max-speculative-values 1000000 \
    --auto-fanout-min-task-values 1000000 \
    --auto-min-file-rows 500000 \
    --seed "$seed" \
    --output-dir "/private/tmp/phase3-effective-holdout/seed-$seed" \
    --summary "/private/tmp/phase3-effective-holdout/seed-$seed-summary.json" \
    --curated-summary \
      "/private/tmp/phase3-effective-holdout/seed-$seed-curated.json"
done

.venv/bin/python \
  research/issue_28402/phase3_effective_holdout_summary.py \
  --report-root /private/tmp/phase3-effective-holdout/seed-38402 \
  --report-root /private/tmp/phase3-effective-holdout/seed-38403 \
  --base-commit c0dab4ed02a09af59f11a1f1e869004228ad4878 \
  --hardware "Apple M4 Max, 16 cores, 64 GB" \
  --cache-regime "warm-cache repeated scans" \
  --output /private/tmp/phase3-effective-holdout/summary.json
```

`capability_benchmark.py` generates seeded Parquet data, verifies identical
results, rotates and records plan order, and records wall and process CPU
samples. Holdout reports also retain benchmark and loaded Polars runtime
fingerprints.
`capability_matrix.py` isolates each thread count in fresh process.
Current Auto exercises schedule only at one pipeline. Use explicit `fanout`
plus `POLARS_ISSUE_28402_FANOUT_MIN_TASK_VALUES` to reproduce old multi-thread
local-fanout arm.

## Final verification

Unified holdout source state passed on 2026-07-23:

```text
make build-release
  passed

cargo fmt --check
  passed

cargo test -p polars-stream --features parquet issue_28402 --lib
  3 passed

.venv/bin/python -m pytest \
  py-polars/tests/unit/io/test_parquet.py -k "28402" -q --tb=short
  16 passed, 1692 deselected

ruff check + ruff format --check on three research Python files
  passed

git diff --check
  passed

phase3_effective_holdout_summary.py
  32/32 controlled cells passed
  480/480 paired measurements favored Auto
  worst paired ratio-of-medians upper bound: -3.44%
```

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
- `auto` rejects normalized slices, then checks post-statistics candidate rows
  and pipeline count. Ineligible scans keep global fallback. Eligible row
  groups use prefix rows times payload-column count as schedule proxy.
  Historical environment variable name retains `MIN_FILE_ROWS`, but value is
  applied to post-statistics candidate rows.
- decode trace counts predicate and projected-column decodes.
- structured trace records stage rows, duration, and overlap.

## Artifacts

- `capability_benchmark.py`: single-case benchmark and correctness check.
- `capability_matrix.py`: controlled subprocess matrix and curated-summary
  generator.
- `phase3_effective_holdout_summary.py`: validates frozen holdout shape,
  provenance, paired ratio gate, and opportunity regret.
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
- `results/phase3-effective-scan-holdout/`: sanitized gate summary and 32 raw
  per-case timing reports for frozen M4 boundary holdout.

Generated Parquet files and intermediate matrix output stay outside repository.
Older curated matrices omit raw timing samples. Effective-scan holdout retains
raw per-plan samples.
