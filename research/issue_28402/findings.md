# Polars #28402 findings

Date: 2026-07-22
Branch: `research/28402-capability-staging`
Base: `62e47a9a0c`

## Question

One Float predicate currently disables independent Parquet column-predicate
path for whole conjunction. Can reader preserve supported string predicates as
first concurrent stage, then evaluate Float residual only on surviving rows?

Research executor compares:

```text
global_fallback
capability_selected
capability_materialized
slice_blocked
```

`slice_blocked` reproduces optimizer-boundary workaround from issue. It is
reference, not proposed reader implementation.

## Implementation

Plan representation retains two maps:

- existing fast-path-eligible single-column conjuncts;
- infallible, elementwise, single-column residual conjuncts excluded by dtype.

Research reader runs supported predicates concurrently, combines exact masks,
decodes residual columns under combined mask, evaluates residual predicates,
restores mask to row-group coordinates, then decodes projected payload under
final mask.

OR, multi-column, nested, mapped, row-index, partial-slice, external-constant,
fallible, and non-elementwise cases keep current path. Environment variable is
required; default execution is unchanged.

## Correctness and decode count

Focused suite covers:

- selected and materialized residual modes;
- projected supported and projected residual columns;
- one decode per projected column per row group;
- empty, all-true, mixed, and null-containing masks;
- multi-column expression fallback;
- existing #28304 staged-reader cases.

Final run: 15 passed. Equality and decode trace are both checked because output
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

Screen rejects unconditional capability rule. Five iterations are enough to
find counterexamples, not estimate production effect.

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
- Performance: fail for unconditional rule; confirmed regressions reach 9–12%.
- Simplicity: fail for static capability-only policy.

Outcome is conditional win. Do not open production consumer PR and do not
make #28485 dependency claim from this branch.

## Real next experiment

Collapse action space to two choices for mixed-capability conjunction:

```text
current global fallback
materialized capability split
```

Run paired row-group blocks instead of row-level model:

1. start with current fallback;
2. collect prefix joint retention, residual rows, per-stage CPU, row-group
   size/count, payload width, and pool width;
3. assign both actions within matched blocks while exploration budget remains;
4. model paired wall difference with uncertainty;
5. stage only when upper predictive bound on
   `staged wall - fallback wall` is below negative safety margin;
6. refresh on drift and fall back when evidence weak.

First objective is not maximize mean speedup. It is test whether observable
features can bound regression on held-out dense, high-thread, wide-payload,
and low-row-group-supply cases. If bound fails, keep production behavior.

## Limits

- Local machine: Apple M4 Max, 64 GB RAM, macOS arm64.
- Synthetic data covers controlled mechanisms, not workload prevalence.
- Screening includes only five timing samples per case.
- Confirmation uses one encoding/compression setup and fixed seed.
- More row groups also means more total rows in supply confirmation.
- Bootstrap treats repeated scan as paired unit; rows are not independent
  samples.
- No production policy or broad speedup claim is established.
