"""Budget-indexed reanalysis, and the schedule-path diagnostic behind it.

`schedule_sweep.py` answers the preregistered question at the final iterate.
That is the right endpoint for "does feedback change where the solver ends up",
and the answer is no. It is the wrong endpoint for "does refinement scheduling
do anything at all", because by 10,000 iterations every scheduled arm has been
sitting at `eps_min` for almost the whole run: the schedule is a vanishing
fraction of the trajectory it is being judged on. At (5,2) the residual arm and
the fixed-0.05 arm end 3.8e-8 apart, which is not a null result about feedback
so much as a comparison that has nothing left to compare.

This script re-reads the same runs at intermediate compute budgets, where the
schedules are still live, and separates the two contrasts the final iterate
collapses together:

    residual - fixed_0.05        does refinement scheduling help at all?
    residual - geometric_matched does the feedback signal help, given annealing?

`geometric_matched` re-derives its starting width and its entries-to-floor from
the residual arm's own path at each mix; `geometric_fast` holds the
preregistered (2, 5) values at every mix and is carried alongside so the frozen
comparison stays visible. The two coincide at (2, 5) by construction.

Differences are reported relative to the control as well as in absolute value.
The preregistered +/-0.02 region is an absolute margin calibrated to
final-iterate errors of 0.05-0.20; at a budget where both arms still have error
of order 10 it is not a meaningful yardstick and a ratio is.

The second half records the realized `eps` path of the residual arm against a
geometric reference curve. `T` is a gamma-contraction in span seminorm across
consecutive global sweeps, but aggregate phases can increase the residual, so
the reference is an empirical diagnostic rather than a bound on the alternating
algorithm.

Exploratory. Added after the preregistered analysis and after the schedule
sweep; not a preregistered endpoint.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from schedule_sweep import (
    MIXES,
    SEEDS,
    arms_for,
    at_mix,
    err_at_backups,
    with_arm,
)
from sweep import ARMS, median_ci

from mdpagg.config import RunCfg, TraceCfg, load
from mdpagg.run import RESULTS_ROOT, execute
from mdpagg.solve import CACHE_ROOT
from mdpagg.trace import environment

## Only the arms the decomposition needs. The remaining preregistered arms are
## unchanged in schedule_sweep.py and are not re-run here.
ARM_NAMES = ("residual", "geometric_matched", "geometric_fast", "fixed_0.05")

## (treatment, control, what the contrast isolates).
CONTRASTS = (
    ("geometric_matched", "fixed_0.05", "annealing vs no annealing"),
    ("residual", "geometric_matched", "feedback vs matched open loop"),
    ("residual", "geometric_fast", "feedback vs the preregistered open loop"),
)

## Round decades, fixed before the differences were looked at. The ceiling is
## 3e6 because the cheapest mix, (1,20), only bills 9.2M backups in total and a
## ladder that runs past an arm's own horizon compares held terminal values
## rather than work. The final iterate is carried along as the last rung so the
## anytime table and Table 1 can be read on one scale.
BUDGETS = (1e5, 3e5, 1e6, 3e6)

NULL_REGION = 0.02


def at_exact_budget_trace(
    base: RunCfg, global_len: int, agg_len: int
) -> RunCfg:
    """Return a run config that records every completed update.

    A backup budget can fall between two ordinary trace checkpoints.  Reading
    a stride-10 trace would then return the latest *recorded* update rather
    than the latest completed update, and the error can change sharply across
    the intervening global and aggregate phases.  Budget-indexed comparisons
    therefore require a row after every update.
    """
    cfg = at_mix(base, global_len, agg_len)
    return cfg.model_copy(update={"trace": TraceCfg(fine_stride=1)})


def rows_at_budgets(
    base: RunCfg, root: Path, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    ## Same configs, same seeds, same pairing as schedule_sweep. The traces are
    ## kept this time, which is the only reason for the re-run: the sweep drops
    ## them before writing and the medians it keeps cannot be paired.
    rows: list[dict[str, Any]] = []

    for global_len, agg_len in MIXES:
        mix = f"({global_len},{agg_len})"
        at = at_exact_budget_trace(base, global_len, agg_len)
        ## Same construction as the sweep, so the matched control here is the
        ## identical config rather than a second calibration of it.
        arms = arms_for(base, root, global_len, agg_len)

        for name in ARM_NAMES:
            for seed in seeds:
                doc = execute(
                    with_arm(at, arms[name], seed), root, trace_policy_loss=False
                )
                trace = doc["trace"]
                rows.append(
                    {
                        "mix": mix,
                        "arm": name,
                        "seed": seed,
                        "err_at": {
                            f"{b:g}": err_at_backups(trace, b) for b in BUDGETS
                        },
                        "err_final": doc["final"]["err_inf"],
                        "billed_final": doc["final"]["billed"],
                    }
                )

            print(f"  {mix:<8} {name:<16} done", flush=True)

    return rows


def err_of(row: dict[str, Any], budget: str) -> float:
    return row["err_final"] if budget == "final" else row["err_at"][budget]


def contrast_at(
    rows: list[dict[str, Any]], mix: str, budget: str
) -> list[dict[str, Any]]:
    at = [r for r in rows if r["mix"] == mix]
    out = []

    for treatment, control, isolates in CONTRASTS:
        by_arm = {
            name: {r["seed"]: err_of(r, budget) for r in at if r["arm"] == name}
            for name in (treatment, control)
        }
        seeds = sorted(set(by_arm[treatment]) & set(by_arm[control]))
        paired = [
            (by_arm[treatment][s], by_arm[control][s])
            for s in seeds
            if not (math.isnan(by_arm[treatment][s]) or math.isnan(by_arm[control][s]))
        ]
        if not paired:
            continue

        diffs = [t - c for t, c in paired]
        ## The relative difference is formed per seed and then aggregated, for
        ## the same reason the absolute one is: dividing the median difference
        ## by the median control would discard the pairing.
        rel = [(t - c) / c for t, c in paired if c > 0.0]

        lo, hi = median_ci(diffs)
        rel_lo, rel_hi = median_ci(rel) if rel else (math.nan, math.nan)

        out.append(
            {
                "mix": mix,
                "budget": budget,
                "treatment": treatment,
                "control": control,
                "isolates": isolates,
                "n": len(paired),
                "control_median": statistics.median([c for _, c in paired]),
                "median": statistics.median(diffs),
                "ci_lo": lo,
                "ci_hi": hi,
                "rel_median": statistics.median(rel) if rel else math.nan,
                "rel_ci_lo": rel_lo,
                "rel_ci_hi": rel_hi,
                ## Reported, not decisive: the margin was calibrated to
                ## final-iterate errors and is only interpretable on rungs
                ## where the arms are already near their floors.
                "within_null_region": abs(lo) < NULL_REGION and abs(hi) < NULL_REGION,
                "excludes_zero": lo > 0.0 or hi < 0.0,
                "diffs": diffs,
            }
        )

    return out


def global_sweeps_by(t: int, global_len: int, agg_len: int) -> int:
    ## The schedule is deterministic: cycles of `global_len` global iterations
    ## followed by `agg_len` aggregate ones. Counting the global sweeps in the
    ## first t iterations gives the exponent the contraction bound applies to.
    cycle = global_len + agg_len
    whole, rest = divmod(t, cycle)
    return whole * global_len + min(rest, global_len)


def schedule_path(base: RunCfg, root: Path, seed: int = 0) -> list[dict[str, Any]]:
    ## One residual run per mix at full trace resolution. The sweep traces every
    ## tenth iteration, which is too coarse to see a schedule that reaches its
    ## floor in about 14 aggregate entries.
    out = []

    for global_len, agg_len in MIXES:
        cfg = at_exact_budget_trace(base, global_len, agg_len)
        doc = execute(
            with_arm(cfg, ARMS["residual"], seed), root, trace_policy_loss=False
        )
        trace = doc["trace"]

        iters = list(trace["iteration"])
        eps = list(trace["eps"])
        span = list(trace["residual_span"])
        gamma = base.problem.gamma
        floor = ARMS["residual"].eps_min
        cycle = global_len + agg_len

        ## eps is 0 until the first partition is built, so the path starts at
        ## the first aggregate entry. It ends where the width clamps: past that
        ## point the rule has stopped reading the residual and there is no
        ## schedule left to describe.
        rows = []
        for t, e, s in zip(iters, eps, span, strict=True):
            if not (math.isfinite(s) and e > 0.0):
                continue
            rows.append(
                {
                    "iteration": int(t),
                    "global_sweeps": global_sweeps_by(t, global_len, agg_len),
                    "residual_span": float(s),
                    "eps": float(e),
                }
            )

        live = [r for r in rows if r["eps"] > floor * (1.0 + 1e-12)]

        ## The reference is anchored at the first width the rule sets and
        ## decays by gamma for every global sweep taken since. Contraction
        ## justifies this curve only within a consecutive block of global
        ## sweeps; aggregate phases can increase the residual. The comparison
        ## over the alternating path is therefore descriptive, not a theorem.
        ratios = []
        if live:
            span0, n0 = live[0]["residual_span"], live[0]["global_sweeps"]
            for r in live:
                r["envelope"] = float(span0 * gamma ** (r["global_sweeps"] - n0))
                ratios.append(r["residual_span"] / r["envelope"])

        ## Realized decay of the width per cycle, against gamma**global_len,
        ## which is what the contraction bound allows over one cycle's worth of
        ## global sweeps.
        realized = math.nan
        if len(live) >= 2:
            slope = np.polyfit(
                [r["iteration"] / cycle for r in live],
                np.log([r["eps"] for r in live]),
                1,
            )[0]
            realized = float(np.exp(slope))

        clamped_at = next(
            (r["iteration"] for r in rows if r["eps"] <= floor * (1.0 + 1e-12)), None
        )

        out.append(
            {
                "mix": f"({global_len},{agg_len})",
                "seed": seed,
                "eps_0": live[0]["eps"] if live else math.nan,
                "gamma_per_cycle": gamma**global_len,
                "realized_per_cycle": realized,
                "entries_to_floor": (
                    math.nan if clamped_at is None else clamped_at // cycle + 1
                ),
                ## Descriptive comparison with the geometric reference. A
                ## value <= 1 is observed here, not guaranteed by contraction
                ## across the intervening aggregate phases.
                "span_over_envelope_max": max(ratios, default=math.nan),
                "envelope_holds": all(r <= 1.0 + 1e-9 for r in ratios),
                "path": live,
            }
        )

    return out


def result_payload(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    budgets = [f"{b:g}" for b in BUDGETS] + ["final"]

    return {
        "config": config,
        "environment": environment(),
        "mixes": [list(m) for m in MIXES],
        "arms": list(ARM_NAMES),
        "seeds": list(seeds),
        "budgets": budgets,
        "budget_trace_stride": 1,
        "null_region": NULL_REGION,
        "preregistered": False,
        "contrasts": [
            c
            for g, a in MIXES
            for b in budgets
            for c in contrast_at(rows, f"({g},{a})", b)
        ],
        "schedule_paths": paths,
        "rows": rows,
    }


def report(payload: dict[str, Any]) -> None:
    print("\npaired differences by budget (median [95% CI]; rel = / control):")
    for c in payload["contrasts"]:
        tag = f"{c['treatment']}-{c['control']}"
        print(
            f"  {c['mix']:<8} {c['budget']:<7} {tag:<28} "
            f"ctrl {c['control_median']:>9.4g}  "
            f"abs {c['median']:+.4g} [{c['ci_lo']:+.4g},{c['ci_hi']:+.4g}]  "
            f"rel {100 * c['rel_median']:+7.2f}% "
            f"[{100 * c['rel_ci_lo']:+.2f}%,{100 * c['rel_ci_hi']:+.2f}%]"
        )

    print("\nresidual schedule path vs the empirical geometric reference:")
    for p in payload["schedule_paths"]:
        print(
            f"  {p['mix']:<8} eps_0 {p['eps_0']:.4f}  "
            f"realized decay/cycle {p['realized_per_cycle']:.4f}  "
            f"reference gamma^l_g {p['gamma_per_cycle']:.4f}  "
            f"entries to floor {p['entries_to_floor']:>3.0f}  "
            f"max span/reference {p['span_over_envelope_max']:.4f}  "
            f"below reference {p['envelope_holds']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/anytime.py")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args(argv)

    base: RunCfg = load(args.config)
    if base.problem.kind != "inventory":
        print(
            f"anytime is the inventory case study; {base.problem.kind!r} problems "
            "vary their instance per seed and are not paired this way",
            file=sys.stderr,
        )
        return 1

    seeds = tuple(range(args.seeds))

    try:
        rows = rows_at_budgets(base, args.root, seeds)
        paths = schedule_path(base, args.root)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    payload = result_payload(base.model_dump(mode="json"), rows, paths, seeds)

    out = args.out or RESULTS_ROOT / f"anytime_{args.config.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    report(payload)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
