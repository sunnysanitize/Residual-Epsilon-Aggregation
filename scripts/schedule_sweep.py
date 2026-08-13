"""Phase-mix robustness check for the residual-vs-timetable comparison.

`sweep.py --arms` answers the preregistered question at one global/aggregate
mix, (2, 5), and indexes its budget endpoints by wall-clock. Both choices are
load-bearing and both are weak: a single mix cannot say whether the conclusion
is a property of the epsilon rules or of that particular schedule, and a
wall-clock endpoint selects a different iterate on every rerun.

This script varies the mix and indexes by billed backups instead. The endpoint
is then a property of the run rather than of the machine it ran on, so the
compute-matched curves and the full-state VI reference are directly comparable
across arms that do very different amounts of work per iteration.

The epsilon arms are the preregistered six, unchanged, plus a recalibrated
`geometric_slow`. The frozen `cycles=1429` was derived for a cycle length of
seven; transplanted to a longer cycle it is still decaying at the horizon and
never reaches its floor, which would read as a scheduling result when it is an
artifact of the transplant.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sweep import ARMS, median_ci

from mdpagg.config import EpsilonCfg, GeometricEpsilonCfg, RunCfg, load
from mdpagg.run import RESULTS_ROOT, execute
from mdpagg.solve import CACHE_ROOT
from mdpagg.trace import environment

## (global_len, agg_len). The preregistered mix sits in the middle; the other
## two move the aggregate share from roughly a third to almost all of the
## iterations, which is the axis the conclusion has to survive.
MIXES = ((5, 2), (2, 5), (1, 20))

SEEDS = tuple(range(20))

## Carried from sweep.py so the recalibrated arm shares the endpoints of the
## frozen one and differs only in how long it takes to get there.
EPS_0 = 0.4985
EPS_MIN = 0.05

## The paper's contrast. The remaining arms are still run and still written to
## the result file; they are context, not the claim.
CONTRASTS = (
    ("residual", "geometric_fast", "null -- the real test"),
    ("residual", "fixed_0.05", "null or negative"),
    ("residual", "geometric_slow_recal", "null once the decay is recalibrated"),
)

NULL_REGION = 0.02

## Log-spaced so the early descent, where the arms actually differ, is not
## compressed into the first pixel of the figure.
CURVE_POINTS = 60


def recalibrated_cycles(iterations: int, global_len: int, agg_len: int) -> int:
    ## Round up: rounding down would leave the last cycles past the end of the
    ## schedule, floored at eps_min for a stretch the schedule never planned.
    return math.ceil(iterations / (global_len + agg_len))


def err_at_backups(trace: dict[str, Any], budget: float) -> float:
    ## A step function, not an interpolation: the error the budget actually
    ## delivered, not the one the run was on its way to. Mirrors
    ## `sweep.err_at_budget`, but indexed by work rather than by time.
    latest = math.nan
    for billed, err in zip(trace["billed"], trace["err_inf"], strict=True):
        if billed > budget:
            break
        latest = err

    return latest


def arms_for(iterations: int, global_len: int, agg_len: int) -> dict[str, EpsilonCfg]:
    return {
        **ARMS,
        "geometric_slow_recal": GeometricEpsilonCfg(
            eps_0=EPS_0,
            eps_min=EPS_MIN,
            cycles=recalibrated_cycles(iterations, global_len, agg_len),
        ),
    }


def at_mix(cfg: RunCfg, global_len: int, agg_len: int) -> RunCfg:
    schedule = cfg.algorithm.schedule.model_copy(
        update={"global_len": global_len, "agg_len": agg_len}
    )
    return cfg.model_copy(
        update={"algorithm": cfg.algorithm.model_copy(update={"schedule": schedule})}
    )


def with_arm(cfg: RunCfg, epsilon: EpsilonCfg, seed: int) -> RunCfg:
    ## The inventory MDP is deterministic in its parameters, so seed index i
    ## selects the sampling stream and nothing else. Every arm sees the same
    ## stream at the same index, which is what makes the differences paired.
    return cfg.model_copy(
        update={
            "algorithm": cfg.algorithm.model_copy(update={"epsilon": epsilon}),
            "master_seed": seed,
        }
    )


def curve_grid(max_billed: int) -> list[float]:
    return list(np.unique(np.geomspace(1.0, float(max_billed), CURVE_POINTS)))


def median_curve(traces: list[dict[str, Any]], grid: list[float]) -> list[float]:
    ## Median across seeds at each budget, rather than the curve of any one
    ## seed. Points before the first recorded row are nan in every seed and
    ## stay nan here.
    out = []
    for budget in grid:
        errs = [err_at_backups(t, budget) for t in traces]
        finite = [e for e in errs if not math.isnan(e)]
        out.append(statistics.median(finite) if finite else math.nan)

    return out


def run_vi(base: RunCfg, root: Path) -> dict[str, Any]:
    ## agg_len=0 never enters the aggregate phase, so the partition is never
    ## built and the sampling stream is never drawn from: the run is ordinary
    ## value iteration and is identical for every seed. It is the reference the
    ## adaptive arms have to justify themselves against, and omitting it was
    ## the most substantive gap in the original comparison.
    doc = execute(at_mix(base, 1, 0), root, trace_policy_loss=False)

    return {
        "final": doc["final"],
        "trace": {k: doc["trace"][k] for k in ("billed", "err_inf", "iteration")},
    }


def run_rows(
    base: RunCfg, root: Path, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for global_len, agg_len in MIXES:
        mix = f"({global_len},{agg_len})"
        at = at_mix(base, global_len, agg_len)
        arms = arms_for(base.algorithm.iterations, global_len, agg_len)

        for name, epsilon in arms.items():
            for seed in seeds:
                doc = execute(
                    with_arm(at, epsilon, seed), root, trace_policy_loss=False
                )
                final = doc["final"]
                rows.append(
                    {
                        "mix": mix,
                        "arm": name,
                        "seed": seed,
                        "eps_final": doc["trace"]["eps"][-1],
                        "wall_ns": doc["wall_ns"],
                        "trace": {
                            k: doc["trace"][k] for k in ("billed", "err_inf")
                        },
                        **final,
                    }
                )

            errs = [r["err_inf"] for r in rows if r["mix"] == mix and r["arm"] == name]
            print(
                f"  {mix:<8} {name:<22} err {statistics.median(errs):.6g}",
                flush=True,
            )

    return rows


def contrasts_at(rows: list[dict[str, Any]], mix: str) -> list[dict[str, Any]]:
    at = [r for r in rows if r["mix"] == mix]
    out = []

    for treatment, control, prediction in CONTRASTS:
        by_arm = {
            name: {r["seed"]: r["err_inf"] for r in at if r["arm"] == name}
            for name in (treatment, control)
        }
        seeds = sorted(set(by_arm[treatment]) & set(by_arm[control]))
        if not seeds:
            continue

        diffs = [by_arm[treatment][s] - by_arm[control][s] for s in seeds]
        lo, hi = median_ci(diffs)
        out.append(
            {
                "mix": mix,
                "treatment": treatment,
                "control": control,
                "prediction": prediction,
                "n": len(seeds),
                "median": statistics.median(diffs),
                "mean": statistics.fmean(diffs),
                "ci_lo": lo,
                "ci_hi": hi,
                ## The whole interval inside the region, not just the point.
                "within_null_region": abs(lo) < NULL_REGION
                and abs(hi) < NULL_REGION,
                ## Equivalence is not agreement: an interval can sit wholly
                ## inside the margin and still exclude zero, which is a real
                ## difference that happens to be small enough not to matter.
                "excludes_zero": lo > 0.0 or hi < 0.0,
                "diffs": diffs,
            }
        )

    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for global_len, agg_len in MIXES:
        mix = f"({global_len},{agg_len})"
        for name in dict.fromkeys(r["arm"] for r in rows if r["mix"] == mix):
            at = [r for r in rows if r["mix"] == mix and r["arm"] == name]
            out.append(
                {
                    "mix": mix,
                    "arm": name,
                    "n": len(at),
                    "err_median": statistics.median([r["err_inf"] for r in at]),
                    "err_min": min(r["err_inf"] for r in at),
                    "err_max": max(r["err_inf"] for r in at),
                    "policy_loss_median": statistics.median(
                        [r["policy_loss"] for r in at]
                    ),
                    "eps_final_median": statistics.median(
                        [r["eps_final"] for r in at]
                    ),
                    "billed_median": statistics.median([r["billed"] for r in at]),
                    "num_groups_mean": statistics.fmean(
                        [r["num_groups"] for r in at]
                    ),
                }
            )

    return out


def curves(rows: list[dict[str, Any]], vi: dict[str, Any]) -> dict[str, Any]:
    max_billed = max(
        max(r["billed"] for r in rows), vi["final"]["billed"]
    )
    grid = curve_grid(max_billed)
    out: dict[str, Any] = {"grid": grid, "vi": median_curve([vi["trace"]], grid)}

    for global_len, agg_len in MIXES:
        mix = f"({global_len},{agg_len})"
        for name in dict.fromkeys(r["arm"] for r in rows if r["mix"] == mix):
            traces = [
                r["trace"] for r in rows if r["mix"] == mix and r["arm"] == name
            ]
            out[f"{mix}|{name}"] = median_curve(traces, grid)

    return out


def result_payload(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    vi: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "config": config,
        "environment": environment(),
        "mixes": [list(m) for m in MIXES],
        "seeds": list(seeds),
        "null_region": NULL_REGION,
        "vi": vi,
        "summary": summarize(rows),
        "contrasts": [
            c for g, a in MIXES for c in contrasts_at(rows, f"({g},{a})")
        ],
        "curves": curves(rows, vi),
        ## The curves already carry the compute-matched medians, so the raw
        ## per-seed traces would multiply the file size to no purpose.
        "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows],
    }


def report(payload: dict[str, Any]) -> None:
    vi = payload["vi"]["final"]
    print(
        f"\nfull-state VI at the 10,000-sweep horizon: err {vi['err_inf']:.6g}  "
        f"policy_loss {vi['policy_loss']:.6g}  billed {vi['billed']:,}"
    )

    print("\nmedian final error by mix and arm:")
    for s in payload["summary"]:
        print(
            f"  {s['mix']:<8} {s['arm']:<22} err {s['err_median']:.6g}  "
            f"loss {s['policy_loss_median']:.4g}  "
            f"eps_final {s['eps_final_median']:.4g}  "
            f"billed {s['billed_median']:,.0f}"
        )

    print(f"\npaired contrasts, final iterate (null region +/-{NULL_REGION}):")
    for c in payload["contrasts"]:
        verdict = "NULL" if c["within_null_region"] else "OUTSIDE null"
        note = ", but excludes zero" if c["excludes_zero"] else ""
        print(
            f"  {c['mix']:<8} {c['treatment']} - {c['control']:<22} "
            f"median {c['median']:+.4g}  "
            f"CI [{c['ci_lo']:+.4g}, {c['ci_hi']:+.4g}]  {verdict}{note}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/schedule_sweep.py")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args(argv)

    base: RunCfg = load(args.config)
    if base.problem.kind != "inventory":
        print(
            f"schedule_sweep is the inventory case study; {base.problem.kind!r} "
            "problems vary their instance per seed and need the paired design "
            "in sweep.py instead",
            file=sys.stderr,
        )
        return 1

    seeds = tuple(range(args.seeds))

    try:
        vi = run_vi(base, args.root)
        rows = run_rows(base, args.root, seeds)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    payload = result_payload(base.model_dump(mode="json"), rows, vi, seeds)

    out = args.out or RESULTS_ROOT / f"schedule_{args.config.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    report(payload)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
