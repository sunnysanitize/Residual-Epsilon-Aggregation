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

The epsilon arms are the preregistered six, unchanged, plus two recalibrated
ones. The frozen `geometric_slow` used `cycles=1429`, derived for a cycle length
of seven; transplanted to a longer cycle it is still decaying at the horizon and
never reaches its floor, which would read as a scheduling result when it is an
artifact of the transplant.

`geometric_matched` fixes the same class of transplant error in the *fast*
control, which is the one the paper's claim rests on. `geometric_fast` is
anchored at eps_0 = 0.4985 and 14 aggregate entries, both calibrated at the
preregistered (2, 5). The residual rule's own opening width is `c` times the
residual span after `global_len` deterministic global sweeps, so it moves with
the mix: 0.400 at (5, 2) and 0.538 at (1, 20). Holding the control's anchor
fixed therefore leaves it 25% coarser than the treatment at (5, 2), which is
exactly the mix where the two arms separate -- the difference is confounded with
the mismatch. `geometric_matched` re-derives both endpoints per mix from the
residual arm's realized width path, so starting width, floor and entries to the
floor agree by construction and only the closed loop is left to differ.
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

from mdpagg.config import EpsilonCfg, GeometricEpsilonCfg, RunCfg, TraceCfg, load
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
    ("residual", "geometric_matched", "null -- the real test, mismatch removed"),
    ("residual", "geometric_fast", "null -- the preregistered anchor"),
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


def residual_anchor(
    base: RunCfg, root: Path, global_len: int, agg_len: int, seed: int = 0
) -> tuple[float, int]:
    ## The two endpoints an open-loop control has to reproduce: the first width
    ## the residual rule ever sets, and the aggregate entry at which it clamps.
    ##
    ## The first is available before the treatment is ever run. It is `c` times
    ## the residual span after `global_len` global sweeps, the MDP is
    ## deterministic, and no aggregation has happened yet, so those sweeps are
    ## arm-independent. The second is not: it depends on how far the aggregate
    ## phases move V, so it is read off one residual run. `geometric_matched` is
    ## therefore a matched control, not a preregistered one, which is why the
    ## frozen `geometric_fast` is carried alongside it rather than replaced.
    ##
    ## fine_stride=1 because a schedule that clamps after seven aggregate
    ## entries is invisible at the sweep's default stride of ten iterations.
    cfg = at_mix(base, global_len, agg_len).model_copy(
        update={"trace": TraceCfg(fine_stride=1)}
    )
    doc = execute(with_arm(cfg, ARMS["residual"], seed), root, trace_policy_loss=False)

    floor = ARMS["residual"].eps_min
    clamp = floor * (1.0 + 1e-12)
    ## eps is 0 until the first partition is built, so `set` starts at the first
    ## aggregate entry rather than at iteration 0.
    set_widths = [
        (int(t), float(e))
        for t, e in zip(doc["trace"]["iteration"], doc["trace"]["eps"], strict=True)
        if e > 0.0
    ]
    if not set_widths:
        raise ValueError(f"residual arm set no width at ({global_len},{agg_len})")

    eps_0 = set_widths[0][1]
    clamped_at = next((t for t, e in set_widths if e <= clamp), None)
    if clamped_at is None:
        raise ValueError(
            f"residual arm never reached its floor at ({global_len},{agg_len}); "
            "there is no time-to-floor for the control to match"
        )

    return eps_0, clamped_at // (global_len + agg_len) + 1


def arms_for(
    base: RunCfg, root: Path, global_len: int, agg_len: int
) -> dict[str, EpsilonCfg]:
    eps_0, entries = residual_anchor(base, root, global_len, agg_len)

    return {
        **ARMS,
        "geometric_slow_recal": GeometricEpsilonCfg(
            eps_0=EPS_0,
            eps_min=EPS_MIN,
            cycles=recalibrated_cycles(
                base.algorithm.iterations, global_len, agg_len
            ),
        ),
        ## Same floor as every other scheduled arm; the anchor and the entry
        ## count come from the treatment's own path at this mix. cycles-1 is the
        ## exponent's denominator, so cycles == entries puts the control at
        ## eps_min on the same aggregate entry the residual rule clamps on.
        "geometric_matched": GeometricEpsilonCfg(
            eps_0=eps_0, eps_min=EPS_MIN, cycles=max(2, entries)
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
        arms = arms_for(base, root, global_len, agg_len)

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
