# Mixed predicate capability staging for Polars #28402

This research branch is not intended for merge. It tests whether supported
Parquet column predicates can keep fast pre-filtering when same conjunction
contains Float predicate that disables current global fast path.

Current fallback decodes all predicate columns before evaluating full
conjunction:

```text
[supported strings, residual Float64]
```

Capability staging decodes supported predicates first, combines their masks,
then decodes residual Float64 values only for surviving rows:

```text
[supported strings] -> [materialized Float64 residual]
```

Execution is enabled only by research environment variable. Default behavior
is unchanged.

## Result

Capability split is useful candidate, not safe unconditional rule.

Final screen contained 90 Float64 cases across prefix retention, predicate
relationship, projection, and thread count. Materialized split improved 64
cases and regressed 26; observed range was roughly -34% to +12%. Higher-signal
confirmation reproduced both outcomes.

Selected-predicate and materialized-residual implementations were close in
confirmed cases. Mixed-capability consumer therefore does not need PR #28485
to establish same candidate execution shape. Static capability alone does not
decide when that shape is safe.

See `findings.md` for measurements and limits.

## Reproduce

Build local extension:

```bash
make build-release
```

Run one case:

```bash
POLARS_MAX_THREADS=16 uv run python \
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

## Research controls

```text
POLARS_ISSUE_28402_CAPABILITY_STAGING=selected
POLARS_ISSUE_28402_CAPABILITY_STAGING=materialized
POLARS_ISSUE_28402_DECODE_TRACE=1
POLARS_ISSUE_28304_TRACE=1
```

- `selected` uses selected-predicate primitive from PR #28485 for residual
  predicate.
- `materialized` decodes residual values under prefix mask, then evaluates
  residual expression against compact DataFrame.
- decode trace counts predicate and projected-column decodes.
- structured trace records stage rows, duration, and overlap.

## Artifacts

- `capability_benchmark.py`: single-case benchmark and correctness check.
- `capability_matrix.py`: controlled subprocess matrix and curated-summary
  generator.
- `findings.md`: decision-gate result and next experiment.
- `results/screening-curated.json`: 180-run screen, including 90 Float64
  cases.
- `results/core-confirmation-curated.json`: sparse/dense, narrow/wide
  confirmation.
- `results/row-groups-*.json`: fixed-size row-group supply runs.

Generated Parquet files and intermediate matrix output stay outside repository.
