# Rolling predicate policy findings

This experiment implements maintainer-proposed progression inside environment-gated research
executor:

1. start each predicate selectivity at prior `0.5`;
2. update marginal selectivity from completed row groups;
3. add measured joint prefix masks;
4. add observed full-decode cost;
5. add conservative task-supply guard.

Default Parquet execution is unchanged. Policy chooses between all-at-once and one defer-one
plan for each row group. It never searches arbitrary partitions at runtime.

## Policy mechanics

Two completed all-at-once row groups form cold start. Later all-at-once and staged first groups
update estimates with `0.5` row-group discount. Every eighth issued row group uses all-at-once
as refresh. Deferred predicate unconditional cost and selectivity are learned during cold-start
or refresh groups; staged groups immediately update first-group marginals, costs, and observed
joint prefix survival.

All levels require predicted first-group survival at most 5% and predicted saved work above 5%:

- `marginal` estimates first-group survival as product of individual selectivities;
- `joint` uses measured combined mask for that candidate;
- `cost` weights avoided rows by measured full-decode nanoseconds per row;
- `task_supply` also requires current in-flight row groups to reach thread-pool width.

Thresholds are research controls, not fitted production defaults. Cost timing is collected under
all-at-once contention and is only a proxy for isolated predicate work.

## Controlled matrix

Each file has 64 row groups of 125,000 rows. Results are median wall-time change from
all-at-once over 15 rotated iterations after two warmups. Negative is faster. Fixed oracle is
best single concurrent-prefix plan for whole file, not per-row-group oracle.

| Workload | Threads | Fixed oracle | Marginal | Joint | Cost | Task guard |
|---|---:|---:|---:|---:|---:|---:|
| stationary sparse | 1 | -12.2% | -10.1% | -9.7% | -9.6% | -9.4% |
| stationary sparse | 16 | -11.5% | -7.8% | -9.1% | -5.8% | -2.7% |
| stationary dense | 1 | 0.0% | +0.1% | -0.7% | 0.0% | -0.2% |
| stationary dense | 16 | 0.0% | +0.6% | +0.1% | +1.2% | +1.6% |
| stationary disjoint | 1 | -31.7% | -26.9% | -27.0% | -27.1% | -27.2% |
| stationary disjoint | 16 | -21.0% | -12.4% | -12.9% | -13.2% | +0.5% |
| stationary nested | 1 | 0.0% | +5.1% | +0.7% | +0.5% | +0.5% |
| stationary nested | 16 | 0.0% | +7.2% | +1.0% | 0.0% | +0.3% |
| sparse to dense | 1 | 0.0% | -4.3% | -4.2% | -4.1% | -4.3% |
| sparse to dense | 16 | 0.0% | +3.1% | +1.3% | +2.4% | -0.9% |
| dense to sparse | 1 | 0.0% | -4.7% | -4.4% | -4.4% | -4.6% |
| dense to sparse | 16 | 0.0% | -0.9% | -2.1% | -0.4% | -0.1% |
| alternating | 1 | 0.0% | +0.1% | +0.3% | +0.3% | +0.2% |
| alternating | 16 | 0.0% | +0.3% | 0.0% | +3.5% | +0.7% |

Bootstrap intervals use 5,000 resamples of median difference. Important checks:

- marginal nested regressions are outside noise: +2.0 to +2.1 ms at one thread and +0.24
  to +0.36 ms at 16 threads;
- joint sparse-to-dense result at 16 threads still regresses by +0.04 to +0.15 ms;
- task guard sparse-to-dense interval at 16 threads spans -0.09 to +0.13 ms;
- joint stationary sparse remains faster at 16 threads by -0.48 to -0.25 ms.

## What progression established

Marginal selectivity is insufficient. In nested workload, product of individual match rates
predicts selective first group even though same rows satisfy both predicates. Marginal level
stages and regresses 5–7%. Joint prefix measurement observes overlap and falls back.

Immediate staged feedback helps drift but does not make selector safe. Before staged first-group
updates, unguarded sparse-to-dense policies regressed roughly 8% at 16 threads. With updates,
regression falls to 1–3%, but remains measurable. At 16 threads, about 18 row groups are issued
before two all-at-once groups complete, so decisions still trail input regime.

Measured decode cost did not unlock different useful choice in this three-predicate matrix. It
mostly chose same deferred column as joint policy. Timing also perturbs scan: cost level staged
zero alternating row groups yet measured +3.5% in one 16-thread run.

Current task-supply guard is too coarse. It avoids measurable high-thread drift regression, but
also suppresses almost all staging in stationary sparse and disjoint cases, giving up gains up
to 13 percentage points. Current in-flight row-group count does not model counterfactual task
graph after stage boundary.

## Decision

No tested level passes policy gate. Keep all-at-once default.

Next policy work needs bounded uncertainty and better scheduling context, not another point
estimate. Candidate should stage only when lower confidence bound on saved work exceeds upper
bound on dependency cost, while enough unissued work remains to amortize delayed observations.
Validate on workload-family holdouts and more predicate counts before runtime proposal.

## Limits

- synthetic three-predicate files only;
- defer-one action family, not full concurrent-prefix family required by four-predicate audit;
- fixed 5% gates based on prior experiments rather than independent training set;
- exact row-group workload order, not natural file drift;
- current task guard uses in-flight row-group count, not scheduler simulation;
- no separate measurement of metadata row-group pruning before decoder;
- policy trace is representative and completion order can vary under concurrency.

Curated measurements are in
`results/issue-28304-rolling-policy-p3-curated.json`. Full samples and decision traces remain
outside repository under `/private/tmp/issue-28304-rolling-final-3`.
