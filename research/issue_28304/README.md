# Predicate staging experiment for Polars #28304

This research branch is not intended for merge. It connects PR #28485
selected-predicate API to `RowGroupDecoder` so I can measure explicit predicate
stages. Without research environment variables, execution stays unchanged.

Current execution decodes all three predicate columns concurrently:

```text
[l_shipmode, l_shipinstruct, l_quantity]
```

Issue-like staged execution decodes two string columns concurrently, then
passes their combined mask to `l_quantity`:

```text
[l_shipmode, l_shipinstruct] -> [l_quantity]
```

I compared all 13 ordered stage partitions for these three predicates. On ten
million issue-like rows, this split improved median wall time by 7.9% with one
thread and 3.5% with 16 threads. At 16 threads, same split regressed 47.9% when
first stage retained 90% of rows. Every fully sequential plan regressed.

Staging can help, but choosing groups matters. Sorting predicates serially by
marginal selectivity is not enough.

## Reproduce

```bash
make build-release

POLARS_MAX_THREADS=16 .venv/bin/python \
  research/issue_28304/benchmark.py \
  --rows 10000000 --row-group-size 1000000 \
  --warmups 5 --iterations 30 --all-pushed-only \
  --include-stage-oracle \
  --output /private/tmp/issue-28304-oracle.json
```

Commas join columns decoded concurrently. Pipes separate sequential stages:

```text
POLARS_ISSUE_28304_STAGES=l_shipmode,l_shipinstruct|l_quantity
```

Three predicates have 13 ordered stage partitions: one all-concurrent plan,
six two-stage plans, and six fully sequential plans.

## Artifacts

- `benchmark.py` generates seeded data, verifies equivalent results, rotates
  execution order, and records wall and process CPU distributions.
- `findings.md` contains causal traces, staged oracle results, density
  crossover, mask topology, correlation, TPC-H, and adaptive experiments.
- `profile_findings.md` explains 16-thread CPU/wall gap using ported benchmarks,
  stack samples, concurrency traces, and thread/row-group controls.
- `profile_workload.py` runs isolated all-at-once or staged plans long enough
  for external profiler attachment.
- `collapse_audit.py` enumerates ordered partitions for three or four predicates and reports
  restricted-action regret with bootstrap interval. It supports defer-one and concurrent-prefix
  action sets.
- `collapse_matrix.py` runs audit in fresh processes across workload, row-group, and thread
  controls.
- `collapse_findings.md` records defer-one failure and concurrent-prefix confirmation.
- Curated collapse summaries are `results/issue-28304-collapse-p3-prefix-curated.json` and
  `results/issue-28304-collapse-p4-prefix-curated.json`.
- `pilot_audit.py` tests whether row-group-0 oracle action transfers to later row groups.
- `pilot_matrix.py` runs stationary, drifting, and alternating pilot controls in fresh processes.
- `pilot_findings.md` records why direct pilot-winner reuse failed.
- Curated pilot summaries are `results/issue-28304-pilot-p3-curated.json` and
  `results/issue-28304-pilot-p4-curated.json`.
- Fixed-size profile controls repeat typed SF1 rows into 7- and 16-group files;
  generated Parquet files remain outside repository, while curated JSON stays in `results/`.
- `research_plan.md` documents hypotheses, decision gates, and reproduction
  commands for full matrix.
- `results/` contains curated raw JSON from final measurement runs. Smoke and
  superseded intermediate runs are excluded.

Raw JSON records OS and architecture, Polars and Python versions, configured
and observed thread-pool size, dataset parameters, warmups, iterations, plans,
and timing distributions. Runs did not record CPU model.

## Research controls

```text
POLARS_ISSUE_28304_STAGES=<comma-and-pipe stage specification>
POLARS_ISSUE_28304_ADAPTIVE=1
POLARS_ISSUE_28304_TRACE=1
```

`POLARS_ISSUE_28304_ADAPTIVE=1` requires an explicit staged candidate. It
chooses only between current all-at-once execution and that candidate. It does
not discover groups.

## Measurement boundary

Original timing matrix predates replacing `PredicateFilter::input_selection`
with separate `column_iter_to_arrays_selected` API from PR #28485. Research
code now uses that API, and nine staged-reader tests pass. Targeted TPC-H,
thread-count, and row-group controls were rerun on ported API. Collapse audit reran full
three-predicate oracle across density, correlation, topology, thread, and row-group controls.
Pilot audit timed concurrent-prefix oracle on first row group and seven later groups. Direct
pilot-winner reuse matched remainder oracle in 5 of 10 three-predicate runs and 4 of 10
four-predicate runs; one-thread drift produced 10–48% remainder regret.
