# Critical-path profile for Polars #28304

Date: 2026-07-22
Branch: `research/28304-profile-critical-path`
Base: `c0321debb5`

## Question

Why does staged predicate decode reduce process CPU at 16 threads without reducing wall time?

This profile uses ported `column_iter_to_arrays_selected` API, so it also checks whether
pre-refactor result still holds.

## Workload

- TPC-H SF1 `lineitem`, 6,001,215 rows in seven row groups;
- `l_quantity` cast from Decimal to `Int32`;
- 16-thread Polars pool on Apple M4 Max, 64 GB RAM;
- all-at-once: `[l_shipmode, l_shipinstruct, l_quantity]`;
- staged: `[l_shipmode, l_shipinstruct] -> [l_quantity]`;
- first-stage mask retains about 3.57% of rows.

`xctrace` was unavailable because active developer directory contains Command Line Tools rather
than full Xcode. Profiles use macOS `sample` after 100 warmup queries. Raw `.sample` files are
machine-specific symbol dumps and remain in `/private/tmp`; commands below reproduce them.

## Ported benchmark result

| Threads | Iterations | Plan | Median wall | Median CPU | Wall change | CPU change |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 30 | all-at-once | 16.649 ms | 16.858 ms | baseline | baseline |
| 1 | 30 | staged | 14.270 ms | 14.492 ms | -14.3% | -14.0% |
| 16 | 30 | all-at-once | 3.032 ms | 19.821 ms | baseline | baseline |
| 16 | 30 | staged | 3.035 ms | 16.810 ms | +0.1% | -15.2% |
| 16 | 1,000 | all-at-once | 3.241 ms | 20.500 ms | baseline | baseline |
| 16 | 1,000 | staged | 3.236 ms | 17.438 ms | -0.2% | -14.9% |

Ported API reproduces original result: staging saves about 15% CPU at 16 threads while wall time
stays unchanged. With one thread, similar CPU reduction becomes similar wall-time reduction.

## Thread-count control

Same seven-row-group file was measured across thread counts. Runs at 2–12 threads use 20 warmups
and 500 rotated iterations; 1-thread run uses 30 iterations and 16-thread run uses 1,000.

| Threads | All-at-once wall | Staged wall | Wall change | CPU change |
|---:|---:|---:|---:|---:|
| 1 | 16.649 ms | 14.270 ms | -14.3% | -14.0% |
| 2 | 9.639 ms | 8.465 ms | -12.2% | -12.0% |
| 4 | 5.850 ms | 5.014 ms | -14.3% | -11.6% |
| 8 | 3.548 ms | 3.039 ms | -14.3% | -10.1% |
| 12 | 3.334 ms | 3.045 ms | -8.7% | -10.5% |
| 16 | 3.241 ms | 3.236 ms | -0.2% | -14.9% |

Staged latency benefit is stable through eight threads, declines at 12, and disappears at 16.
CPU reduction remains across sweep. Same-file control isolates thread-pool width from schema,
values, encoding, and row-group layout.

## Row-group control

Same rows were rewritten with smaller row groups and measured at 16 threads for 1,000 rotated
iterations:

| Row groups | Row-group size | All-at-once wall | Staged wall | Wall change | CPU change |
|---:|---:|---:|---:|---:|---:|
| 7 | 1,000,000 | 3.241 ms | 3.236 ms | -0.2% | -14.9% |
| 13 | 500,000 | 2.520 ms | 2.305 ms | -8.5% | -13.2% |
| 25 | 250,000 | 2.411 ms | 2.282 ms | -5.3% | -11.0% |

Staging reduces 16-thread latency once more independent row groups are available. Rewriting also
changes row-group boundaries and per-group overhead, so this control supports concurrency
explanation but does not isolate row-group count by itself.

## Stack samples

Exclusive top-of-stack samples were grouped after removing parked/waiting stacks:

| Active sample category | All-at-once | Staged |
|---|---:|---:|
| Parquet deserialize | 66.9% | 61.3% |
| Parquet encoding/decompression | 13.8% | 11.5% |
| Boolean/scalar filtering | 9.0% | 14.9% |
| Scheduler and other | 10.3% | 12.3% |

Sampling is directional: captures had different effective durations, and wait stacks include
runtime threads unrelated to predicate critical path. Trace concurrency below is stronger
evidence for stage-boundary effect.

## Concurrency trace

Existing `POLARS_ISSUE_28304_TRACE` instrumentation shows:

- all-at-once reaches 16 concurrent predicate tasks;
- staged first stage peaks at 14 tasks: two columns across seven row groups;
- staged quantity stage runs six concurrent full-row-group tasks after first stage completes;
- each full quantity task receives about 35,500–36,000 selected rows instead of 1,000,000;
- all-at-once overlaps quantity decode with string predicates, while staged plan creates a
  dependency between them.

Quantity work is genuinely reduced. At 16 threads, however, stage boundary removes column-level
overlap and caps available work below pool width. Saved CPU and lost parallelism approximately
cancel on wall-time critical path. Thread and row-group controls show saved work reduces latency
when pool width is lower or more independent row groups are available.

This rejects broad hypothesis that predicate decode is simply off critical path at high thread
counts. For this workload, critical path depends on available task parallelism: decode savings
reach wall time until stage serialization prevents executor from using additional threads.

## Policy implication

Per-predicate cost/selectivity metric is insufficient by itself. Grouping policy must also model:

- available row-group parallelism;
- thread-pool width;
- concurrent columns inside each group;
- sequential stage depth;
- critical-path cost of deferred predicates.

Runnable tasks relative to pool width are a policy input, not a hard threshold: staging still won
at 12 threads with only six full quantity tasks. All-at-once should remain fallback unless
predicted saved work exceeds stage-boundary cost with margin.

## Reproduce

Benchmark current port at 16 threads:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --data-path /private/tmp/tpch-sf1-28304/lineitem-28304.parquet \
  --reuse-data --row-group-size 1000000 \
  --warmups 20 --iterations 1000 --all-pushed-only \
  --candidate-stage-spec 'l_shipmode,l_shipinstruct|l_quantity'
```

Collect ready-synchronized stack profile:

```bash
POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/profile_workload.py \
  --data-path /private/tmp/tpch-sf1-28304/lineitem-28304.parquet \
  --iterations 10000 --warmups 100 \
  --ready-file /private/tmp/issue-28304-ready &

profile_pid=$!
while [ ! -e /private/tmp/issue-28304-ready ]; do sleep 0.05; done
/usr/bin/sample "$profile_pid" 15 \
  -file /private/tmp/issue-28304-profile.sample
wait "$profile_pid"
```

Add `--stage-spec 'l_shipmode,l_shipinstruct|l_quantity'` for staged capture. Use a fresh ready
file for each run.

## Next experiment

Repeat thread sweep for dense first-stage mask and for additional row-group counts. This tests
whether concurrency feature predicts crossover across selectivity regimes rather than only this
sparse TPC-H-derived workload. Full Xcode Instruments capture would add scheduler and core-usage
timelines unavailable from `sample`.
