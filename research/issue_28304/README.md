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

Raw measurements predate replacing `PredicateFilter::input_selection` with
separate `column_iter_to_arrays_selected` API from PR #28485. Research code now
uses that API, and nine staged-reader tests pass. Performance measurements have
not been rerun since that change.
