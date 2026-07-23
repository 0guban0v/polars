# First-row-group pilot audit for Polars #28304

Date: 2026-07-22
Branch: `research/28304-profile-critical-path`

## Question

Does best concurrent-prefix action on first row group remain best for later row groups?

This audit gives pilot more information than deployable policy would have. It times every
restricted candidate on row group 0 and selects measured winner. It then applies that action to
seven later row groups and compares it with restricted oracle measured on those groups. Failure
here rules out direct winner reuse; success would not establish feature-based policy.

Each row group has 125,000 rows. Results use 30 rotated iterations for three predicates and 15
for four predicates. Regret is selected action's median wall time divided by remainder oracle's
median wall time minus one.

Action abbreviations:

```text
A = l_shipmode
B = l_shipinstruct
C = l_quantity
D = l_discount
all = all predicates decoded concurrently
```

## Three predicates

| Scenario | Threads | Pilot winner | Remainder oracle | Regret |
|---|---:|---|---|---:|
| stationary sparse | 1 | AB -> C | AB -> C | 0% |
| stationary sparse | 16 | all | AB -> C | 1.4% |
| stationary dense | 1 | all | all | 0% |
| stationary dense | 16 | all | all | 0% |
| sparse -> dense | 1 | AB -> C | all | 26.2% |
| sparse -> dense | 16 | all | all | 0% |
| dense -> sparse | 1 | all | AB -> C | 12.5% |
| dense -> sparse | 16 | all | AB -> C | 3.1% |
| alternating remainder | 1 | AB -> C | all | 10.0% |
| alternating remainder | 16 | all | all | 0% |

Pilot and remainder winners matched in 5 of 10 runs. Three-predicate stationary runs matched in
3 of 4; remaining 1.4% difference had bootstrap median-difference interval crossing zero.
One-thread drift mismatches were separated from zero and caused 10.0–26.2% remainder regret.
Dense-to-sparse mismatch at 16 threads was not separated from zero.

## Four predicates

| Scenario | Threads | Pilot winner | Remainder oracle | Regret |
|---|---:|---|---|---:|
| stationary sparse | 1 | AB -> CD | AB -> D -> C | 3.5% |
| stationary sparse | 16 | all | AB -> CD | 14.1% |
| stationary dense | 1 | all | all | 0% |
| stationary dense | 16 | all | all | 0% |
| sparse -> dense | 1 | AB -> D -> C | all | 48.2% |
| sparse -> dense | 16 | all | all | 0% |
| dense -> sparse | 1 | all | AB -> C -> D | 21.1% |
| dense -> sparse | 16 | all | ABC -> D | 9.6% |
| alternating remainder | 1 | AB -> D -> C | all | 20.7% |
| alternating remainder | 16 | all | all | 0% |

Pilot and remainder winners matched in 4 of 10 runs. In stationary sparse one-thread run,
pilot and remainder chose different staged shapes with 3.5% regret; bootstrap interval was just
above zero. Stationary sparse 16-thread mismatch had 14.1% point regret, but median-difference
interval crossed zero. One-thread drift mismatches were separated from zero and caused
20.7–48.2% remainder regret. Dense-to-sparse mismatch at 16 threads was not separated from zero.

## Interpretation

Directly reusing first-row-group winner fails pilot gate. Distribution drift reverses useful
action, and alternating remainder can favor all-at-once even when sparse pilot favors staging.

Stationary four-predicate result exposes separate issue: action timed on one row group need not
be action timed best on seven equivalent row groups. Candidate's dependency graph interacts
with number of row groups and available task concurrency. Pilot masks and per-column costs can
still be inputs, but pilot winner itself is not policy.

This audit does not show that first-group features are insufficient. It shows next selector must
predict remainder execution from observed masks and task costs plus scheduler context. It also
needs fallback or re-observation under drift.

## Next gate

Build offline stage-boundary model, not runtime adaptive path. Use all-at-once pilot trace to
estimate each concurrent-prefix candidate on held-out row groups. Include remaining row groups,
pool width, task durations, and joint masks. Compare predicted action with restricted oracle and
report regret. Add rolling re-observation only if stationary holdout passes.

## Reproduce

```bash
POLARS_MAX_THREADS=1 .venv/bin/python \
  research/issue_28304/pilot_matrix.py \
  --predicate-count 3 \
  --threads 1,16 --remainder-row-groups 7 --row-group-size 125000 \
  --warmups 5 --iterations 30 --bootstrap-resamples 5000 \
  --output-dir /private/tmp/issue-28304-pilot-p3

POLARS_MAX_THREADS=1 .venv/bin/python \
  research/issue_28304/pilot_matrix.py \
  --predicate-count 4 \
  --threads 1,16 --remainder-row-groups 7 --row-group-size 125000 \
  --warmups 3 --iterations 15 --bootstrap-resamples 3000 \
  --output-dir /private/tmp/issue-28304-pilot-p4
```

Matrix runner creates fresh process for each thread count because Polars thread pool initializes
once. Leading `POLARS_MAX_THREADS=1` controls parent only.
