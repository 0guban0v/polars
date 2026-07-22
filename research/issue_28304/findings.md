# Polars #28304 findings

Date: 2026-07-21
Branch: `research/28304-predicate-staging`
Primitive: PR #28485

## Summary

Current streaming Parquet execution decodes every pushed predicate column at
full row-group height. Column tasks run concurrently, then masks combine.
Projected non-predicate columns decode under final mask.

Issue #28304 concerns predicate grouping, not Python expression order. Staging
keeps concurrency inside each group while passing exact
combined mask between groups:

```text
[l_shipmode, l_shipinstruct] -> [l_quantity]
```

This saves decoding and materialization work. It does not save column-chunk
I/O because `RowGroupData` already contains fetched bytes.

Staging is not always faster. Issue-like split helped when first mask was
sparse and regressed when it was dense. Every fully sequential order
regressed. Marginal selectivity alone also failed under correlated predicates.

## Causal trace

One traced row group contained 1,000,000 rows. Current all-pushed plan started
three predicate decodes concurrently, each with 1,000,000 input rows. Slice
proxy started two string decodes concurrently, combined 35,770 matches, then
decoded `l_quantity` for those 35,770 rows.

Observed `l_quantity` decode duration dropped from 1.48–1.62 ms to 0.34–0.39
ms. Peak predicate overlap changed from three concurrent decodes to two,
followed by one sparse decode.

Slice proxy locates cause but does not model staged reader fairly: later work
runs above scan. Results below use explicit staged executor.

## Staged oracle

Commas mean concurrent decode. Pipes mean sequential stages. Three predicates
produce 13 ordered stage partitions: one all-concurrent plan, six two-stage
plans, and six fully sequential plans. All plans returned identical results.
One-stage control matched current execution within noise.

Ten-million-row synthetic workload, five warmups, 30 rotated measurements:

| Threads | Plan | Median wall | Process CPU | Wall change |
|---:|---|---:|---:|---:|
| 1 | current | 71.857 ms | 72.282 ms | baseline |
| 1 | strings then quantity | 66.158 ms | 66.648 ms | -7.9% |
| 16 | current | 8.338 ms | 86.210 ms | baseline |
| 16 | strings then quantity | 8.044 ms | 77.178 ms | -3.5% |

Best fully sequential plan regressed roughly 33% with one thread and 42% with
16 threads. Serial sorting is not viable.

## Density crossover

Same two-stage split crossed from helpful to harmful between roughly 5% and
10% retained rows on tested schema:

| First-stage mask | 1-thread wall | 16-thread wall | 16-thread CPU |
|---:|---:|---:|---:|
| 1.00% | -13.8% | -6.7% | -15.7% |
| 2.49% | -10.2% | -0.6% | -12.8% |
| 5.01% | -6.2% | -1.3% | -8.9% |
| 9.98% | +2.3% | +7.3% | -1.2% |
| 25.01% | +20.9% | +24.2% | +23.4% |
| 90.24% | +38.2% | +47.9% | +38.0% |

Crossover depends on tested type, encoding, projection, implementation, and
thread counts. It is not suitable as fixed threshold.

## Mask layout and correlation

Cardinality did not explain cost by itself. Holding retained rows near 3.54%
while changing mask layout produced:

| Layout | 1-thread wall | 16-thread wall | 16-thread CPU |
|---|---:|---:|---:|
| random | -7.9% | -3.5% | -10.5% |
| alternating | -15.8% | -5.4% | -18.7% |
| contiguous | -29.6% | -16.2% | -30.8% |

Physical row order also changes compression and page-local behavior, so mask
runs, transitions, or page densities matter alongside cardinality.

Correlation tests kept individual marginals near 14.2% and 25.0% while
changing intersection:

| Correlation | Combined mask | 1-thread wall | 16-thread wall |
|---|---:|---:|---:|
| disjoint | 0% | -33.7% | -21.0% |
| nested | 14.2% | +4.5% | +14.0% |

Equal marginals selected opposite plans. Policy needs conditional group masks,
not independent selectivity estimates.

## TPC-H check

`tpchgen-rs` produced SF1 `lineitem` with 6,001,215 rows. Native
`l_quantity` uses Decimal, which Polars excludes from independent column
predicates. Staged comparison used documented three-column derivative with
`l_quantity` cast to `Int32`.

| Threads | Current wall | Staged wall | Staged CPU change |
|---:|---:|---:|---:|
| 1 | 17.174 ms | 14.690 ms | -14.3% |
| 16 | 3.069 ms | 3.070 ms | -15.6% |

At 16 threads, saved CPU did not reduce wall time. Native Decimal slice proxy
was tied with one thread and 3.2% slower with 16 threads. Physical type, scale,
and available row-group concurrency affect outcome.

## Adaptive prototype

Current all-at-once execution already yields exact per-column masks. Initial
row groups can expose decode cost, group intersections, and mask layout without
assuming uniform key distributions.

Research policy compares current execution with one fixed staged candidate. It
samples both twice, chooses lower mean row-group time, and explores opposite
arm every sixteenth issued row group.

| First-stage mask | Threads | Always staged | Adaptive wall |
|---:|---:|---:|---:|
| 3.54% | 1 | -7.1% | -6.2% |
| 3.54% | 16 | -3.1% | +1.3% |
| 90.24% | 1 | +31.5% | +2.7% |
| 90.24% | 16 | +34.3% | +8.5% |

Prototype reduced dense-case loss but missed selective 16-thread win.
Concurrent row-group issue delays feedback, while elapsed samples include
contention from both arms. Production policy needs low-contention work metrics,
immutable decision snapshots, drift tests, and explicit regression bounds.

## Code boundary

PR #28485 adds `column_iter_to_arrays_selected`. It accepts existing
`PredicateFilter` plus input bitmap and returns values plus refined bitmap in
original column coordinates. Existing decode API, `PredicateFilter`, and
default reader execution stay unchanged.

Research branch adds only environment-gated work:

- explicit stage parser and executor in `RowGroupDecoder`;
- decode tracing and overlap counters;
- fixed-candidate two-arm policy;
- integration tests and benchmark runner.

Prototype scope stays limited to flat, independently separable,
sumwise-complete conjunctions already accepted by column-predicate path.
Unsupported expressions keep current execution.

## Limits

- Results do not establish production grouping policy.
- Headline measurements use local macOS arm64 machine; raw files omit CPU model.
- TPC-H staged result uses documented `Int32` derivative, not native Decimal.
- File I/O savings were not measured or claimed.
- Raw timings predate API refactor in PR #28485. Ported code passes focused
  tests, but timing matrix has not been rerun.
