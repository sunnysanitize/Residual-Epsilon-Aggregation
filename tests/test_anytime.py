"""Regression tests for the exact-backup analysis."""

from anytime import at_exact_budget_trace, result_payload

from mdpagg.config import RunCfg


def base_config() -> RunCfg:
    return RunCfg.model_validate(
        {
            "problem": {
                "kind": "inventory",
                "num_assets": 1,
                "q_max": 2,
                "lam": 0.02,
                "rho": 0.0,
                "fill": [0.1, 0.2, 0.4, 0.6, 0.8],
                "spread": [1.0, 0.8, 0.5, 0.3, 0.15],
            },
            "algorithm": {
                "iterations": 100,
                "epsilon": {"kind": "fixed", "value": 0.05},
                "schedule": {"global_len": 2, "agg_len": 5},
            },
            "trace": {"fine_stride": 10},
        }
    )


def test_budget_analysis_records_every_completed_update():
    original = base_config()

    exact = at_exact_budget_trace(original, global_len=1, agg_len=20)

    assert original.trace.fine_stride == 10
    assert exact.trace.fine_stride == 1
    assert exact.algorithm.schedule.global_len == 1
    assert exact.algorithm.schedule.agg_len == 20


def test_result_declares_exact_budget_trace_stride():
    payload = result_payload({}, [], [], ())

    assert payload["budget_trace_stride"] == 1
