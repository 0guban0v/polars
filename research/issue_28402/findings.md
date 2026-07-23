# Polars #28402 findings

Date: 2026-07-23
Branch: `research/28402-capability-staging`
Base: `3b96397da8`

## Question

One Float predicate currently disables independent Parquet column-predicate
path for whole conjunction. Can reader preserve supported string predicates as
first concurrent stage, then evaluate Float residual only on surviving rows?

Research executor compares:

```text
global_fallback
capability_selected
capability_materialized
capability_fanout
slice_blocked
```

`slice_blocked` reproduces optimizer-boundary workaround from issue. It is
reference, not proposed reader implementation.

## Implementation

Plan representation retains two maps:

- existing fast-path-eligible single-column conjuncts;
- infallible, elementwise, single-column residual conjuncts excluded by dtype.

Sequential research reader decodes residual before projected payload. Fanout
reader instead decodes residual and projected payload concurrently under
supported-prefix mask, evaluates residual once, then filters compact columns.
Fanout does not reconstruct full-row mask because no later decode needs it.

Optional fanout-local sizing partitions compact-column filtering by observed
work:

```text
work = compact rows * compact columns
work tasks = ceil(work / minimum task values)
row-group task cap = ceil(2 * pool width / configured row-group prefetch capacity)
```

If in-flight row groups already meet pool width, cap is one task per row
group. Actual task count is minimum of work tasks, cap, and column count. This
changes only final in-memory filtering, not decode plan.

OR, multi-column, nested, mapped, row-index, partial-slice, external-constant,
fallible, and non-elementwise cases keep current path. Environment variable is
required; default execution is unchanged.

## Correctness and decode count

Focused suite covers:

- selected, materialized, and fanout modes;
- projected supported and projected residual columns;
- one decode per projected column per row group;
- empty, all-true, mixed, and null-containing masks;
- multi-column expression fallback;
- existing #28304 staged-reader cases.

Final run: 21 passed. Equality and decode trace are both checked because output
equality alone cannot detect duplicate decode.

Structured trace on 25,000-row groups showed supported predicates reading full
groups, Float64 residual receiving roughly 1,200 rows after 5% prefix, and
payload receiving roughly 700–800 rows after final mask. Trace also records
task duration and active decode count.

## Screening

Screen used 1,000,000 rows, eight 125,000-row groups, five rotated
measurements, fixed seed, and paired bootstrap interval. Float64 cases varied:

- prefix retention: 1%, 5%, 25%, 60%, 90%;
- relationship: independent, nested, negative;
- threads: 1, 8, 16;
- residual projected or filter-only.

Materialized split:

| Result | Cases |
|---|---:|
| Lower median wall | 64 / 90 |
| Higher median wall | 26 / 90 |
| 95% interval below zero | 51 / 90 |
| 95% interval above zero | 11 / 90 |
| Observed range | -34% to +12% |

Sequential screen rejects residual-then-payload schedule, not capability
partition itself.

Native fanout:

| Result | Cases |
|---|---:|
| Lower median wall | 90 / 90 |
| Higher median wall | 0 / 90 |
| Observed range | -34.5% to -7.5% |
| Within 2% of slice reference | 62 / 90 |
| Within 5% of slice reference | 83 / 90 |

Median fanout gap to slice reference was 1.2%. Fanout was faster in 19 cases
and slower in 71. Five-iteration screen establishes stable direction, not
exact parity.

## Confirmation

Confirmation used 4,000,000 rows, four 1,000,000-row groups, 20 rotated
measurements, 5,000 bootstrap resamples, and independent supported predicates.

Materialized wall change against global fallback:

| Prefix retained | Payload columns | 1 thread | 8 threads | 16 threads |
|---:|---:|---:|---:|---:|
| 5% | 1 | -24.2% | -20.2% | -8.0% |
| 5% | 8 | -19.5% | -12.9% | -5.3% |
| 60% | 1 | -6.1% | -1.8% | +7.8% |
| 60% | 8 | -7.2% | +2.5% | +9.1% |

Intervals excluded zero except 60%/one payload/eight threads. Same 60%
retention regressed at one thread in smaller-row-group screen, so retention
alone is not stable threshold.

Native fanout wall change against global fallback:

| Prefix retained | Payload columns | 1 thread | 8 threads | 16 threads |
|---:|---:|---:|---:|---:|
| 5% | 1 | -33.6% | -37.3% | -25.1% |
| 5% | 8 | -23.5% | -33.0% | -28.1% |
| 60% | 1 | -24.9% | -30.8% | -21.3% |
| 60% | 8 | -2.3% | -10.7% | -10.9% |

All fanout intervals excluded zero. Dense wide case that made slice reference
2.1% slower at one thread improved 2.3% natively. At 8 and 16 threads fanout
remained 10–13% behind slice while still improving roughly 11% over fallback.

## Fanout-local task sizing

Dense/wide threshold sweep used four 1M-row groups, 60% prefix retention,
eight payload columns, 30 rotated measurements, and 8/16 threads. Minimum task
values from 0.5M through 2M closed default fanout gap. 1M was retained for
controls:

- 8 threads: 6.6% faster than default fanout, 0.1% behind slice;
- 16 threads: 10.1% faster than default fanout, 1.2% behind slice.

Row-group capacity changed useful granularity. At 16 threads, one row group
benefited roughly 13% from local sizing. Expanding every one of 16 in-flight
row groups regressed default fanout. Final cap keeps one task per row group
when configured capacity already fills pool, and otherwise targets roughly two
filter tasks per worker across available row groups. With final cap,
16-row-group local result was statistically indistinguishable from default
fanout and 3% faster than slice.

Configured prefetch capacity is upper bound, not active row-group measurement.
Byte budget, skipped groups, other files, or pipeline timing can reduce actual
supply. Current control establishes mechanism on single-file runs.

Final 24-case confirmation used four 1M-row groups, 1/5/60/90% retention,
one/eight payload columns, 1/8/16 threads, 20 rotated measurements, and 1M
minimum task values:

| Prefix retained | Payload columns | 1 thread | 8 threads | 16 threads |
|---:|---:|---:|---:|---:|
| 1% | 1 | -28.6% | -27.7% | -13.9% |
| 1% | 8 | -22.6% | -26.4% | -14.7% |
| 5% | 1 | -30.4% | -31.7% | -21.6% |
| 5% | 8 | -22.8% | -28.5% | -23.5% |
| 60% | 1 | -25.6% | -29.3% | -21.1% |
| 60% | 8 | -1.9% | -16.1% | -18.9% |
| 90% | 1 | -22.1% | -29.4% | -20.6% |
| 90% | 8 | +7.8% | -5.8% | -17.7% |

Values are wall change against global fallback. Intervals were below zero in
23 of 24 cases. Local sizing had no statistically significant regression
against default fanout; median CPU change was effectively zero. Remaining
regression is fanout schedule itself: at 90%/eight payload/one thread,
sequential materialization improved roughly 10% while fanout regressed.

## Phase 3 candidate automatic schedule

Phase 3 implements two separate research heuristics:

```text
Decision 1: metadata file rows >= 500k and one pipeline?
    no  -> current global fallback
    yes -> capability path

Decision 2: prefix rows * projected payload columns <= 1M?
    yes -> fanout
    no  -> sequential materialization
```

Prefix rows and payload-column count are exact. Their product is speculative
value count, not exact work. It does not model physical bytes, encoding,
compression, page locality, null density, mask topology, or decoder fixed
cost. Both thresholds are required research environment variables; default
execution is unchanged.

### Fixed-total row-group-size control

Initial candidate was 2M speculative values. Fixed-total 4M-row control used
90% prefix retention, eight payload columns, and 250k/500k/1M/2M row groups.
At 250k row groups, value count was roughly 1.8M, so 2M selected fanout.
Fanout regressed roughly 37% for Int64 and 44% for short String payload when
residual retained 10%. Sequential materialization was within 1% of fallback.
At 500k–2M row groups, 2M selected materialization and improved roughly 6–11%.

This decisively rejects 2M for measured family. Revised 1M selected
materialization at 250k layout and changed wall by -0.9% to +0.8% for
low-residual-retention Int64/String cases; intervals included zero.
High-retention cases improved roughly 3.2–3.4%.

One-million value threshold remains candidate separator. Original seed-28402
screen contained nearest values around 0.9M and 1.2M, but only seven timing
iterations and participated in threshold choice. Additional-seed confirmation
did not independently exercise near-switch values: main points were roughly
0.05M/0.4M/0.6M/0.9M and 4.8M/7.2M.

### Additional seed and payload-type control

One-thread 4M-row control used seed 28403, four 1M row groups, prefix retention
5/60/90%, residual retention 10/90%, payload width one/eight, and
Int64/Float64/short-String payload:

| Payload | Point estimates improved | Median wall change | Weakest point |
|---|---:|---:|---:|
| Int64 | 12 / 12 | -17.9% | -4.5% |
| Float64 | 12 / 12 | -9.2% | -0.5% |
| String | 12 / 12 | -16.9% | -2.1% |

All 36 paired timing intervals were below zero. Overall median was -15.9%;
observed range was -32.8% to -0.5%. Auto and explicit arm selected by 1M rule
differed by 0.003% at median, with range -0.67% to +1.50%.

These are controlled warm-cache configurations, not 36 independent workloads.
Payload dtypes reuse same predicate and residual masks. One additional seed
does not establish workload generalization. Worst measured regret against
three-arm minimum was roughly 8% here and 15.5% in original seven-iteration
screen; those figures are diagnostic because minimum of noisy medians has no
uncertainty interval.

### Pre-eligibility multi-thread fanout control

Old Auto implementation forced fanout whenever pipeline count exceeded one.
At 8/16 threads, it therefore tested unconditional fanout with Phase 2 local
filtering, not threshold-based adaptive scheduling. Auto and explicit local
fanout differed by roughly 0.1% at median, as expected from same effective
path.

| Result | Cases |
|---|---:|
| Lower median wall | 63 / 72 |
| 95% interval below zero | 61 / 72 |
| Point regression above 2% | 8 / 72 |
| Significant regressions among those eight | 7 / 8 |
| Worst regression | +48.0% |

Regressions concentrated in dense, wide cases and varied with residual
retention and payload type. In one 16-thread short-String case, both capability
arms regressed: local fanout roughly +32% and materialized roughly +11%.
Global fallback must remain available.

Control rejects unconditional multi-thread fanout. It does not prove residual
prior is required for every bounded extension. On same matrix, in-sample static
rule using pipeline count, prefix cardinality, payload width/type, and fallback
had no point regression above 2%; it is not held-out evidence, but demonstrates
logical non-necessity. Residual estimate may recover opportunity in ambiguous
cells without conservative abstention.

Current Auto falls back above one pipeline. Final dense/short-String guard
check placed Auto within timing noise of fallback at 8 and 16 threads. Explicit
fanout regressed roughly 65% and 31% in same checks. Pre-eligibility Auto result
cannot be reproduced through current Auto mode; use explicit local-fanout arm
or pre-guard revision.

### Provisional small-scan eligibility guard

One-thread file-size sweep tested 250k, 500k, 1M, 2M, and 4M metadata rows.
At 250k, dense Int64 case regressed roughly 3.2% before guard; best explicit
capability arm was within noise of fallback. At 500k and above, all tested
Int64 point estimates improved.

Five-hundred-thousand-row Float64/short-String control improved all 16 point
estimates by roughly 2.0–26.3%; 15 intervals were below zero. This makes 500k
first safe tested grid point, not established boundary. It was selected and
checked with seed 28403, without independent-seed eligibility confirmation or
intermediate file sizes.

Final 250k short-String run verifies guard follows fallback and changed wall by
-0.02%; it does not establish pre-guard String regression. Five-hundred-
thousand-row weak Float64 case improved 3.2%.

Metadata file rows are static proxy for scheduled scan size. Statistics
skipping can make effective scan much smaller. Multi-file scans, different
row-group layouts at same file size, and post-statistics work remain untested.

## Issue-like confirmation

Ten-million-row, ten-row-group, 16-thread run used roughly 7.1% supported
prefix retention, one payload column, five warmups, and 30 rotated samples:

| Plan | Wall change | Median wall |
|---|---:|---:|
| Sequential materialized | -40.2% | 10.512 ms |
| Native fanout | -48.7% | 9.027 ms |
| Slice reference | -48.9% | 8.988 ms |

Fanout was 0.4% behind reference and recovered issue's reported roughly 2x
fallback gap at smaller but structurally equivalent scale.

## Selected decode versus materialized residual

Selected and materialized candidates were close across confirmation matrix.
At 5% retention they differed by 0.1–1.4 percentage points; at 60% by
0.6–2.4 points. Neither dominated.

This supports narrower conclusion: restoring supported-prefix stage and
reducing residual input creates opportunity. Results do not show
selected-predicate API is required for #28402 consumer. Materialized executor
reaches same performance regime without PR #28485.

## Row-group supply

Fixed 1,000,000-row groups, 5% prefix, 16 threads:

| Row groups | Materialized wall | Materialized CPU |
|---:|---:|---:|
| 1 | -4.2% | -25.1% |
| 4 | -12.0% | -34.3% |
| 8 | -30.5% | -41.0% |

More groups coincided with larger wall benefit, but CPU savings changed too.
Result is compatible with task-supply explanation; it does not isolate
row-group supply as sole cause.

## Decision gates

- Semantics: pass for tested scope.
- Decode count: pass.
- Filter task sizing: pass; it closes targeted high-thread gap without
  significant regression against default fanout.
- Auto wiring: partial. Benchmarks exercise both one-thread arms, but automated
  suite has no eligible-Auto integration test proving fanout and materialized
  branches. It tests helper boundaries and ineligible fallback.
- One-thread performance: promising in measured family, not closed. No observed
  regression, but threshold-near frozen holdout, mask topology, filter-only
  residual, encoding, null, and effective-scan controls are missing.
- Eligibility: provisional. 500k is first safe tested point, not characterized
  boundary or independently seeded guard.
- Multi-thread performance: unconditional local fanout rejected. Adaptive
  multi-thread schedule was not tested.
- Native parity: pass for targeted four-row-group dense/wide high-thread case;
  slice is not universal winner when row-group capacity or one-thread payload
  work changes.
- Reproducibility: partial. Curated matrix artifacts omit raw timing samples,
  and pre-eligibility Auto behavior differs from current code.
- Simplicity: value-count proxy is useful candidate. Evidence does not yet
  establish sufficient policy.

Outcome is implemented mechanism plus provisional one-thread heuristic. Phase 3
policy remains open. Do not make production or #28485 dependency claim from
this branch.

## Next validation

Freeze 1M and 500k candidates before collecting new data:

```text
schedule values: 0.75M, 0.9M, 1.0M, 1.1M, 1.25M, 1.5M
file rows:       250k, 375k, 500k, 625k, 750k
```

Use new seeds and independent processes. Cross only controls capable of
falsifying proxy:

- random, clustered, and alternating prefix masks at same cardinality;
- projected and filter-only residual;
- Int64, Float64, short/long String, null-heavy, and different
  encoding/compression cases;
- fixed total rows with different row-group layouts;
- statistics-pruned files where metadata rows exceed scheduled rows;
- eligible Auto integration tests for both schedule arms and multi-thread
  fallback.

Preserve raw per-plan timing samples and exact code revision with curated
summary.

Treat multi-thread policy as separate three-arm experiment: fallback, local
fanout, and materialization. Start with conservative static rule using observed
prefix cardinality, pipeline count, payload width/type, and abstention. Add
residual-rejection or measured payload-cost signal only if frozen holdout shows
static rule unsafe or sacrifices enough opportunity. Paired-block prediction is
later fallback, not current requirement.

## Limits

- Local machine: Apple M4 Max, 64 GB RAM, macOS arm64.
- Synthetic data covers controlled mechanisms, not workload prevalence.
- Screening includes only five timing samples per case.
- Confirmation uses one encoding/compression setup and one additional seed.
- Payload-dtype cases share predicate and residual masks.
- Timing intervals describe repeated warm-cache scans on one machine, not
  workload-family uncertainty; no multiple-comparison adjustment was applied.
- Earlier supply confirmation changed total rows. Phase 3 row-group-size sweep
  held total rows fixed and exposed 2M-threshold failure.
- Eligibility uses file metadata rows, not post-statistics scheduled rows.
- Bootstrap treats repeated scan as paired unit; rows are not independent
  samples.
- Curated Phase 3 matrices omit raw samples required to recompute intervals.
- No production policy or broad speedup claim is established.
