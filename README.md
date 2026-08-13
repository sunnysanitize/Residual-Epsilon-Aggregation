# Residual-Epsilon Aggregation

Code and artifacts for **“Feedback or Annealing? A Controlled Study of Adaptive
State Aggregation.”**

**[Read the paper (PDF)](paper/feedback_or_annealing.pdf)**

This repository evaluates whether Bellman-residual feedback improves adaptive
state aggregation on a 9,261-state inventory-control MDP.

The result decomposes. Because the Bellman operator contracts the residual span
by `gamma` per sweep, a residual-driven width is bounded by a geometric
envelope, so the matched open-loop control is fixed in advance rather than
fitted after the fact. Across three update mixes, 20 paired seeds, and five
compute budgets, the feedback signal's largest reliable advantage over that
timetable is 0.3% of the control's error and its largest disadvantage is 14.5%.
The schedule feedback induces is worth far more than the feedback: against a
fixed width it moves the error by up to 23.5% at a matched budget, in a
direction that reverses as the budget grows.

## Repository layout

- `src/mdpagg/`: MDP and state-aggregation implementation
- `configs/`: experiment configurations
- `scripts/`: experiment and plotting scripts
- `results/`: reported outputs and figures
- `paper/`: compiled paper
- `docs/`: preregistration record and metric definitions
- `tests/`: automated tests

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,plots]'
```

## Reproduction

Run the validation suite:

```bash
make all
```

Reproduce the main experiment:

```bash
scripts/reproduce_residual_epsilon.sh
```

Reproduce the schedule-robustness experiment and figure:

```bash
.venv/bin/python scripts/schedule_sweep.py configs/inventory_n3.json
.venv/bin/python scripts/plot_schedule.py results/schedule_inventory_n3.json
```

Reproduce the budget-indexed reanalysis (Table 2) and the realized width paths
checked against the span-contraction envelope (Proposition 1):

```bash
PYTHONPATH=scripts .venv/bin/python scripts/anytime.py configs/inventory_n3.json
```

Generated results are written to `results/`. Wall-clock measurements may vary
by machine; final-iterate and backup-indexed results are deterministic for the
configured seeds.

## License

Released under the [MIT License](LICENSE).
