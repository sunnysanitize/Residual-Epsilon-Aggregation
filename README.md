# Residual-Driven State Aggregation for Multi-Asset Inventory Control

This repository asks whether state aggregation works better when its resolution
responds to the Bellman residual. More specifically, it tests whether a solver
benefits from watching the residual as it refines its state groups, or whether a
fixed refinement schedule works just as well.

On the inventory-control problem studied here, the answer is no. Once the solver
has converged, the residual-driven rule and a rate-matched geometric schedule
produce effectively the same result. Their final paired difference is
`-2.9e-06` on a scale where `||V*||inf = 100`. The practical conclusion is that
the benefit comes from how quickly the aggregation width shrinks, not from using
the residual as feedback.

This is a deliberately narrow conclusion from a preregistered experiment. **It is
an empirical case study on a single problem instance**, not a general claim about
residual feedback. The study covers one multi-asset inventory MDP and 20 paired
seeds. It also includes a preregistered timing endpoint that turned out to be
poorly chosen. The full record, including that mistake and its effect on the
interpretation, is in
[`docs/residual_epsilon_note.md`](docs/residual_epsilon_note.md).

The conclusion is checked against three global/aggregate update mixes rather than
the single preregistered one, which is what rules out the result being a property
of that particular schedule. It is not a claim about other MDPs, other problem
families, or other solvers. Whether feedback helps when the instance itself
changes is untested here and is the obvious next experiment.

## What the study tests

Adaptive state aggregation alternates full Bellman sweeps over every state with
cheaper sweeps over groups of states that have similar values. The aggregation
width, written here as `epsilon`, controls how coarse those groups are. A larger
width produces fewer groups and cheaper updates, but it loses precision. A
smaller width produces more precise groups at a higher computational cost.

The residual-driven method updates the width at the beginning of each aggregate
phase according to this rule:

```text
epsilon = max(epsilon_min, c * span(TV - V))
```

The intuition is straightforward. When the Bellman residual is large, the value
estimate is still crude, so fine groups may be unnecessary. As the residual
contracts, the method gradually creates finer groups. The experiment tests
whether that feedback is actually useful.

## How the comparison works

Every run uses the same MDP, sampling seeds, sweep schedule, iteration count,
trace schedule, and group limit. Only the rule for `epsilon` changes. The fixed
baseline holds `epsilon` at `0.05`, `0.1`, or `0.5`. The residual configuration
uses the feedback rule above. Two geometric configurations shrink `epsilon` on
a preset timetable without reading the residual.

The fast geometric configuration is the most important control. It starts and
finishes at the same widths as the residual rule and refines at approximately
the same rate. The only meaningful difference is that one configuration reads
the Bellman residual while the other follows a schedule. This separates the
effect of feedback from the more ordinary benefit of annealing.

The slow geometric configuration follows the literal schedule in the original
plan and spreads its decay over all 1,429 aggregate cycles. It is useful for
showing why matching the annealing rate matters, but it is not the main test of
the feedback hypothesis. Altogether, the experiment contains six configurations
across 20 paired seeds, for 120 runs.

## Results

The table below reports median infinity-norm error over the 20 seeds for the
three configurations that form the central comparison.

| Configuration | 20 ms | 50 ms | 100 ms | 400 ms | Final |
|:--|--:|--:|--:|--:|--:|
| Fixed, `epsilon = 0.05` | 0.7802 | 0.1342 | 0.1479 | 0.1403 | 0.1379 |
| Residual | 0.6314 | 0.1344 | 0.1486 | 0.1407 | 0.1388 |
| Geometric fast | 0.6762 | 0.1344 | 0.1486 | 0.1405 | 0.1388 |

From 50 ms onward, the three configurations agree to three or four significant
figures. At the final iterate, the median paired difference between the residual
and fast geometric configurations is `-2.9e-06`, with a 95 percent bootstrap
confidence interval of `[-4.116e-04, 2.127e-07]`. That interval is entirely
inside the preregistered null region of `+/-0.02`.

The preregistration selected 20 ms as the primary endpoint, but that endpoint
falls in the steep part of the solver's descent rather than near its final error
floor. Small timing changes therefore select noticeably different iterates.
Even repeated runs on the same machine changed the reported confidence interval
at that budget. The residual-versus-geometric comparison was indeterminate at
20 ms, while every comparison from 50 ms onward and the deterministic final
iterate were null.

The 20 ms result remains in the repository because it was preregistered. It has
not been silently replaced with a more favorable endpoint. The conclusion is
instead based on the stable portion of the experiment, and on the deterministic
final iterate.

## Schedule robustness

A single update mix cannot distinguish a property of the `epsilon` rules from a
property of that schedule, so [`scripts/schedule_sweep.py`](scripts/schedule_sweep.py)
repeats the comparison at three mixes, written `(global_len, agg_len)`, and
indexes the curves by billed backups rather than wall-clock. Backups are a
property of the run; wall-clock is a property of the machine, which is what made
the 20 ms endpoint unstable.

Median final error, 20 seeds:

| Configuration | `(5,2)` | `(2,5)` | `(1,20)` |
|:--|--:|--:|--:|
| Fixed, `epsilon = 0.05` | 0.0500 | 0.1379 | 0.1909 |
| Residual | 0.0500 | 0.1388 | 0.2034 |
| Geometric fast | 0.0437 | 0.1388 | 0.2005 |
| Stopped value iteration | 1.76e-09 | — | — |

Every rule that reaches `epsilon_min = 0.05` finishes in the same place at every
mix, and rules that end elsewhere separate by up to twentyfold. The clearest case
is the slow geometric arm: transplanted unchanged to `(1,20)` its decay is still
running at the horizon, it ends at `epsilon = 0.232`, and it posts 1.59.
Recalibrating only its cycle count so it reaches the floor returns it to 0.209,
beside every other arm. What the final error tracks is the width a schedule ends
at, not the path it takes there.

Residual feedback shapes only the path, which is why it buys nothing. The
stronger statement the data supports is that it is **never better and sometimes
slightly worse**: at `(5,2)` the paired residual-minus-geometric difference is
`+0.0063` with a confidence interval that excludes zero, and at `(1,20)` residual
trails fixed-`0.05` by `+0.0125`, also excluding zero. Both sit inside the
preregistered `+/-0.02` equivalence region, so practical equivalence holds — but
"no measurable difference" would be too strong.

Two honest caveats. This law governs value error and not decision quality: at
`(2,5)` the lowest-error arm carries 2.5 times the policy loss of fixed-`0.05`,
so ranking arms on the sup-norm alone is misleading. And ordinary value iteration
— the `agg_len = 0` corner of the same solver — reaches `1.76e-09` for 92.6M
billed backups against the best adaptive arm's 0.0437 for 67.7M. At this problem
size aggregation is not competitive at any width or mix, which bounds where these
questions are worth asking.

## The inventory problem

The benchmark is a discounted multi-asset market-making inventory MDP. A state
is a signed inventory vector `(q1, ..., qN)`, with each position bounded between
`-Q` and `Q`. The primary instance uses three assets and `Q = 10`, which produces
`21^3 = 9,261` states. Each state has five quote-aggressiveness actions, and one
period has `2N + 1` possible successors representing no fill or a fill on either
side of one asset.

The cost combines an inventory-risk penalty, `lambda * q^T Sigma q`, with the
spread captured from fills. The discount is `gamma = 0.95`, and costs are scaled
so that `||V*||inf = 100`. The covariance matrix is non-diagonal, which makes
the value of an inventory depend on how positions across assets are correlated.

The five-action space stays fixed as the number of assets grows. This is an
important part of the design. Giving every asset five independent action levels
would make the action count grow as `5^N`, limiting the usefulness of compressing
the state space.

## Reproducing the study

The project requires Python 3.11 or newer and a Unix-like shell. Create a virtual
environment and install the development and plotting dependencies with:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
```

Run the full validation suite with:

```bash
make all
```

This command runs linting, type checking, compiled tests, pure-Python tests, and
a cold bounds-checked Numba run. Two exactness checks apply only to compiled
mode and explain why they are skipped in debug mode.

To reproduce the ground truth, policy baselines, and arm comparison, run:

```bash
scripts/reproduce_residual_epsilon.sh
```

The experiment takes a few minutes and writes its output to `results/`. The main
comparison is stored in `results/arms_inventory_n3.json`, while
`results/arms_inventory_n3_curves.json` includes the per-iteration traces. The
fixed-width reference is stored in `results/inventory_fixed_eps.json`, and the
policy baselines are stored in `results/inventory_baselines.json`.

The schedule-robustness comparison and its figure are separate, because they take
about eight minutes:

```bash
.venv/bin/python scripts/schedule_sweep.py configs/inventory_n3.json
.venv/bin/python scripts/plot_schedule.py results/schedule_inventory_n3.json
```

These write `results/schedule_inventory_n3.json` and the two-mode figure under
`results/figures/`. Both are tracked, so the reported numbers can be checked
without rerunning anything.

Wall-clock columns depend on the machine and may differ from the values reported
above. Final-iterate results and every backup-indexed column are deterministic
for the configured seeds.

The repository is released under the [MIT License](LICENSE).
