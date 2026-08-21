# Residual-Epsilon Aggregation

Code and artifacts for **“Feedback or Annealing? A Controlled Study of Adaptive
State Aggregation.”**

**[Read the paper (PDF)](paper/feedback_or_annealing.pdf)**

This repository evaluates whether Bellman-residual feedback improves adaptive
state aggregation on a 9,261-state inventory-control MDP.

On this instance, the residual-driven widths follow a roughly geometric path.
I use the seed-0 feedback run to match an open-loop geometric schedule, then
hold that schedule fixed. Across three update mixes, 20 paired seeds, and five
compute budgets, feedback's largest advantage among unadjusted 95% intervals
that exclude zero is 0.2% of the control's error. Its largest disadvantage is
5.7%. Annealing matters more: matched geometric versus fixed width changes
the error by up to 22.7%, and the direction changes with the budget.

## Repository layout

- `src/mdpagg/`: MDP and state-aggregation implementation
- `configs/`: experiment configurations
- `scripts/`: experiment and plotting scripts
- `results/`: reported outputs and figures
- `paper/`: compiled paper
- `docs/`: analysis record and metric definitions
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

Reproduce the exact budget-indexed reanalysis (Table 2) and compare the
realized width paths with the descriptive geometric reference:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/anytime.py configs/inventory_n3.json
```

Generated results are written to `results/`. Wall-clock measurements may vary
by machine; final-iterate and backup-indexed results are deterministic for the
configured seeds.

## License

Released under the [MIT License](LICENSE).
