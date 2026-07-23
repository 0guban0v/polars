# Predicate-stage collapse audit for Polars #28304

Date: 2026-07-22
Branch: `research/28304-profile-critical-path`

## Question

Can grouping policy be reduced to a concurrent-prefix action set: all-at-once, defer one
predicate, or evaluate a concurrent first group before a deferred suffix?

Audit enumerates every ordered partition, verifies identical results, rotates execution order,
and compares restricted action set with measured wall-time oracle. Three predicates produce 13
partitions; four produce 75.

## Three-predicate gate

Ported API was tested across seven workload shapes, balanced and short-tail layouts, and 1, 8,
and 16 threads: 42 exhaustive runs total. Workloads cover sparse, medium, dense, disjoint,
nested, alternating, and contiguous masks.

| Oracle class | Runs |
|---|---:|
| all-at-once | 21 |
| defer-one | 21 |
| other two-stage | 0 |
| three-stage | 0 |

The concurrent-prefix action set selected exact oracle winner in every run. Maximum measured
restricted regret was zero.

## Four-predicate gate

Targeted confirmation used 1M rows in eight balanced row groups, 15 rotated iterations, and
5,000 bootstrap resamples. Concurrent-prefix action set selected exact oracle winner in all four
runs.

| Workload | Threads | Defer-one regret | Concurrent-prefix regret |
|---|---:|---:|---:|
| sparse | 1 | 13.6% | 0% |
| sparse | 16 | 6.2% | 0% |
| disjoint | 1 | 38.0% | 0% |
| disjoint | 16 | 10.6% | 0% |

Defer-one restriction fails. Winners include both two-stage and multi-stage deferred suffixes,
for example `[shipmode, instruct] -> [quantity, discount]` and
`[shipmode, instruct] -> quantity -> discount`. Concurrent-prefix grouping captures both.

These results do not establish why deferred suffix grouping wins. Additional mask reduction
before `l_discount` is plausible, especially with one thread, but was not isolated from encoding,
filtering, and scheduler effects.

## Gate decision

Defer-one collapse gate failed. Concurrent-prefix action set passed current 42 three-predicate
runs and four-predicate confirmation runs with zero measured restricted regret. This is a useful
restricted action space, not proof that arbitrary predicate counts or workload families need no
other action.

Smallest observed missing family is:

```text
[concurrent selective group] -> [deferred predicate group]
```

Next experiment is first-row-group pilot validity for concurrent-prefix candidates, followed by
stage-boundary cost measurement. This remains narrower than unrestricted ordered-partition
search.

## Reproduce

Three-predicate matrix:

```bash
.venv/bin/python research/issue_28304/collapse_matrix.py \
  --predicate-count 3 --restricted-action-set concurrent_prefix --threads 1,8,16 \
  --rows 1000000 --row-group-size 125000 \
  --warmups 2 --iterations 10 --bootstrap-resamples 2000 \
  --output-dir /private/tmp/issue-28304-collapse-p3
```

Four-predicate confirmation:

```bash
.venv/bin/python research/issue_28304/collapse_matrix.py \
  --predicate-count 4 --restricted-action-set concurrent_prefix \
  --workloads sparse,disjoint \
  --layouts balanced --threads 1,16 \
  --rows 1000000 --row-group-size 125000 \
  --warmups 2 --iterations 15 --bootstrap-resamples 5000 \
  --output-dir /private/tmp/issue-28304-collapse-p4
```

Set `POLARS_MAX_THREADS` only through matrix runner; each thread configuration requires fresh
process because Polars thread pool initializes once.
