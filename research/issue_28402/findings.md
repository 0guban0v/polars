# Polars #28402 findings

Date: 2026-07-22
Branch: `research/28402-capability-staging`
Base: `cfca393a24`

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

Final run: 19 passed. Equality and decode trace are both checked because output
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
- Performance: pass for tested mixed-capability matrix; fanout won all screen
  and confirmation cases.
- Native parity: pass for issue-like and median screen; incomplete for dense,
  wide, high-thread cases.
- Simplicity: capability fanout needs no learned selector in tested region.

Outcome is broad tested win, not production proof. Do not make #28485
dependency claim from this branch.

## Real next experiment

Profile remaining dense/wide high-thread parity gap before adding policy.
Then test work-conserving frontier: residual decode/eval runs at high priority;
payload tasks use prefix speculatively only when worker would otherwise idle,
or final mask when already available. Each column still decodes once.

Paired-block prediction is later fallback only if optimized scheduling
reintroduces conditional regressions.

## Limits

- Local machine: Apple M4 Max, 64 GB RAM, macOS arm64.
- Synthetic data covers controlled mechanisms, not workload prevalence.
- Screening includes only five timing samples per case.
- Confirmation uses one encoding/compression setup and fixed seed.
- More row groups also means more total rows in supply confirmation.
- Bootstrap treats repeated scan as paired unit; rows are not independent
  samples.
- No production policy or broad speedup claim is established.
