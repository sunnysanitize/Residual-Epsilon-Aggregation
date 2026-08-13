# Residual-Epsilon Aggregation

Code and artifacts for **“Feedback or Annealing? A Controlled Study of Adaptive
State Aggregation.”**

**[Read the paper (PDF)](paper/feedback_or_annealing.pdf)**

This repository evaluates whether Bellman-residual feedback improves adaptive
state aggregation on a 9,261-state inventory-control MDP. Across three update
mixes and 20 paired seeds, a residual-driven aggregation width provides no
practical improvement over a matched geometric schedule. The results suggest
that the refinement timetable, not the feedback signal, is the useful component
on this instance.

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

Generated results are written to `results/`. Wall-clock measurements may vary
by machine; final-iterate and backup-indexed results are deterministic for the
configured seeds.

## Citation

```bibtex
@misc{zhang2026feedback,
  author = {Sunny Zhang},
  title  = {Feedback or Annealing?
            A Controlled Study of Adaptive State Aggregation},
  year   = {2026},
  note   = {MLxOR workshop manuscript},
  url    = {https://github.com/sunnysanitize/Residual-Epsilon-Aggregation}
}
```

## License

Released under the [MIT License](LICENSE).
