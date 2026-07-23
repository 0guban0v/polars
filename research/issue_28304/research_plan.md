# Polars #28304 research plan

Started: 2026-07-20
Branch: `research/28304-predicate-staging`
Measurement base: `upstream/main` at `0cdafb4657`
Port base: PR #28485 at `b7d34c0b11`

## Goal

Measure when staged Parquet predicate decoding saves work without losing more
cross-column parallelism than it saves. Compare fixed and adaptive grouping
against exhaustive oracle before proposing default behavior.

## Scope

Study CPU, materialization, masks, and task overlap after column chunks reach
`RowGroupData::fetched_bytes`. Keep PageIndex and remote range pruning outside
this work; selected decode cannot retroactively avoid fetched bytes.

Use flat fields and independent single-column conjunctions already eligible for
Polars column predicates. Preserve current path for nested, multi-column,
fallible, order-sensitive, or unsupported predicates.

## Current path

`row_group_data_to_df_prefiltered` currently:

1. Collects predicate fields in `predicate_field_indices`.
2. Splits fields according to `num_pipelines`.
3. Starts predicate decode chunks through `parallelize_first_to_local`.
4. Decodes each predicate column at full row-group height.
5. Combines exact masks after all predicate tasks finish.
6. Decodes non-predicate projected columns under combined mask.

Source files:

- `crates/polars-stream/src/nodes/io_sources/parquet/row_group_decode.rs`
- `crates/polars-stream/src/nodes/io_sources/parquet/init.rs`
- `crates/polars-async/src/primitives/opt_spawned_future.rs`

## Questions

1. How much decode and materialization work disappears under incoming mask?
2. How much wall-time benefit comes from current cross-column concurrency?
3. Where does staged plan cross from useful to harmful?
4. Do mask runs, transitions, or page densities predict cost beyond cardinality?
5. How do type, encoding, compression, nullability, row-group size, and column
   reuse move crossover?
6. Does best grouping remain stable across row groups and files?
7. Can observed masks and costs approach exhaustive oracle without uniform
   distribution prior?
8. How much regret occurs during exploration, drift, or alternating inputs?

## Measurements

Capture wall and process CPU distributions plus:

- rows entering and surviving each predicate or stage;
- mask set bits, runs, and transitions;
- compressed bytes already fetched;
- pages decompressed;
- values decoded and retained;
- decode and predicate-evaluation time;
- predicate task start, finish, and peak overlap;
- thread and pipeline configuration.

Keep tracing disabled for final wall-time comparisons unless overhead is
measured separately.

## Experiments

### Reproducer

Generate seeded flat Parquet data with two string predicates near 14.2% and
25%, numeric predicate near 60%, multiple row groups, and no row-group
statistics elimination. Verify every plan returns same result. Rotate plan
order during timed loops.

Run with one and 16 threads first. Add 2 and 4 threads when policy work starts.
Use warm cache for primary CPU study.

### Causal trace

Record full-height predicate decode under current plan and sparse decode after
combined string mask. Confirm saved work occurs inside Parquet decoder rather
than only outer expression evaluation. Record overlap lost by stage boundary.

### Exhaustive oracle

For three predicates, enumerate all 13 ordered stage partitions:

- one all-concurrent plan;
- six two-stage plans;
- six fully sequential plans.

Treat fastest measured shape for each dataset and thread configuration as
oracle. Compare both wall and process CPU.

### Controlled matrix

Vary retained density, predicate correlation, mask topology, physical type,
encoding, nullability, projection width, predicate reuse, row-group size,
page size, compression, and thread count. Change one dimension per comparison.

Cover 0%, 1%, 5%, 10%, 25%, 60%, 95%, and 100% retained rows. Include random,
contiguous, and alternating masks. Include stationary, drifting, and
alternating row-group distributions.

### Policy comparison

Compare candidate policies with oracle:

- expression or schema order;
- compressed-byte order;
- measured cost per rejected row;
- previous row-group choice;
- bounded partition search;
- runtime two-arm or hill-climbing choice.

Track steady-state time, convergence, cumulative regret, plan changes, and
worst regression. Model group critical path alongside serial predicate order.

## Decision gates

Do not propose default staging until all gates pass:

1. Causal: selected decode materially reduces decoded or retained values.
2. Latency: staged shape wins outside noise across defined region and multiple
   thread counts.
3. Regression: dense and cheap-predicate cases have measured fallback bounds.
4. Policy: candidate choice is compared against exhaustive oracle.
5. Semantics: unsupported expressions always retain current path.

## Current status

- Selected-predicate primitive exists in PR #28485 with six decoder tests.
- Research branch ports staged executor to separate selected API.
- All 13 three-predicate partitions return identical results.
- Synthetic, density, topology, correlation, SF1, and two-arm runs are saved in
  `results/`.
- Nine staged-reader integration tests pass.
- Two-arm elapsed-time policy does not meet 16-thread regression target.
- Targeted TPC-H, thread-count, and row-group controls were rerun after PR
  #28485 API refactor.
- Ported three-predicate collapse audit enumerated all 13 plans across 42 controlled runs;
  concurrent-prefix action set matched exact oracle in every run.
- Four-predicate audit enumerated all 75 plans. Defer-one had 6.2–38.0% regret in confirmation
  runs; concurrent-prefix action set matched exact oracle in all four runs. Pilot policy work
  remains blocked until first-row-group validity is measured.

## Commands

Build release extension:

```bash
make build-release
```

Run all 13 staged plans:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --rows 10000000 --row-group-size 1000000 \
  --warmups 5 --iterations 30 --all-pushed-only \
  --include-stage-oracle \
  --output /private/tmp/issue-28304-oracle.json
```

Compare fixed staged candidate:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --rows 10000000 --row-group-size 1000000 \
  --warmups 3 --iterations 20 --all-pushed-only \
  --candidate-stage-spec 'l_shipmode,l_shipinstruct|l_quantity'
```

Compare current, fixed, and adaptive candidates:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --rows 10000000 --row-group-size 100000 \
  --warmups 5 --iterations 50 --all-pushed-only \
  --candidate-stage-spec 'l_shipmode,l_shipinstruct|l_quantity' \
  --include-adaptive-candidate
```

Reuse typed SF1 derivative:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --data-path /private/tmp/tpch-sf1-28304/lineitem-28304.parquet \
  --reuse-data --row-group-size 1000000 \
  --warmups 5 --iterations 30 --all-pushed-only \
  --include-stage-oracle
```

Trace one row group:

```bash
POLARS_MAX_THREADS=4 POLARS_ISSUE_28304_TRACE=1 \
  .venv/bin/python research/issue_28304/benchmark.py \
  --rows 1000000 --row-group-size 1000000 \
  --warmups 0 --iterations 1
```
