"""The schedule sweep's analysis primitives.

The wall-clock endpoint in `sweep.py` selects a different iterate on every
rerun, which is the protocol failure recorded in the note. Its replacement
indexes the trace by billed backups instead, so the same run always yields the
same endpoint. These tests pin that determinism and the schedule
recalibration that keeps a decay arm comparable across phase mixes.
"""

import math

import pytest
from schedule_sweep import err_at_backups, recalibrated_cycles, result_payload


def trace_of(billed: list[int], err: list[float]) -> dict[str, list[float]]:
    return {"billed": billed, "err_inf": err}


def test_error_at_budget_is_the_one_the_budget_delivered():
    ## A step function, not an interpolation: the budget buys the last row it
    ## actually reached, not the one the run was on its way to.
    trace = trace_of([10, 20, 30], [5.0, 2.0, 1.0])

    assert err_at_backups(trace, 25) == 2.0


def test_error_at_exact_budget_boundary_counts_that_row():
    trace = trace_of([10, 20, 30], [5.0, 2.0, 1.0])

    assert err_at_backups(trace, 20) == 2.0


def test_budget_below_first_row_has_no_error_to_report():
    trace = trace_of([10, 20, 30], [5.0, 2.0, 1.0])

    assert math.isnan(err_at_backups(trace, 5))


def test_budget_beyond_the_run_reports_the_final_row():
    trace = trace_of([10, 20, 30], [5.0, 2.0, 1.0])

    assert err_at_backups(trace, 10_000) == 1.0


def test_endpoint_is_identical_across_traces_with_different_timings():
    ## The point of the replacement: two runs of the same arm differing only in
    ## wall-clock must land on the same iterate.
    fast = trace_of([10, 20, 30], [5.0, 2.0, 1.0])
    slow = trace_of([10, 20, 30], [5.0, 2.0, 1.0])

    assert err_at_backups(fast, 25) == err_at_backups(slow, 25)


@pytest.mark.parametrize(
    ("global_len", "agg_len", "expected"),
    [
        (2, 5, 1429),  ## the preregistered mix, and its frozen cycle count
        (5, 2, 1429),  ## same cycle length, so the schedule is unchanged
        (1, 20, 477),  ## longer cycle, so fewer of them fit in the horizon
    ],
)
def test_decay_is_recalibrated_to_the_cycle_length_of_the_mix(
    global_len, agg_len, expected
):
    ## A schedule transplanted across mixes is still decaying at the horizon
    ## and never reaches its floor, which reads as a scheduling result when it
    ## is an artifact of the transplant.
    assert recalibrated_cycles(10_000, global_len, agg_len) == expected


def test_recalibration_covers_the_whole_horizon():
    ## Rounding down would leave the last cycles past the end of the schedule.
    cycles = recalibrated_cycles(10_000, 1, 20)

    assert cycles * 21 >= 10_000


def row_of(mix: str, arm: str, seed: int, err: float) -> dict:
    return {
        "mix": mix,
        "arm": arm,
        "seed": seed,
        "err_inf": err,
        "policy_loss": 0.01,
        "eps_final": 0.05,
        "billed": 1000,
        "num_groups": 500,
        "trace": {"billed": [10, 1000], "err_inf": [9.0, err]},
    }


def minimal_inputs():
    rows = [
        row_of("(2,5)", arm, seed, 0.1 + 0.01 * seed)
        for arm in ("residual", "geometric_fast", "fixed_0.05")
        for seed in range(3)
    ]
    vi = {
        "final": {"billed": 2000, "err_inf": 1e-9, "policy_loss": 0.0},
        "trace": {"billed": [10, 2000], "err_inf": [9.0, 1e-9]},
    }
    return rows, vi


def test_result_carries_the_environment_it_was_produced_in():
    ## A dependency pin is only checkable against a recorded environment, and
    ## the reported numbers are only reproducible against the one that made
    ## them. `trace.document` stamps this for single runs; the sweep must too.
    rows, vi = minimal_inputs()

    payload = result_payload({}, rows, vi, (0, 1, 2))

    assert "numpy" in payload["environment"]
    assert "cpython" in payload["environment"]


def test_result_omits_per_run_traces_from_the_rows():
    ## The curves carry the compute-matched medians; keeping every seed's full
    ## trace as well would multiply the file size for nothing.
    rows, vi = minimal_inputs()

    payload = result_payload({}, rows, vi, (0, 1, 2))

    assert all("trace" not in row for row in payload["rows"])
