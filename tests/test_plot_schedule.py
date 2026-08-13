"""The schedule figure's series selection.

The sweep runs seven epsilon arms; the paper's figure carries three plus the
stopped-VI reference. The rest stay in the result file. These tests pin that
the figure draws the paper's series and not whatever the sweep happened to
produce, and that identity never rests on colour alone.
"""

import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from plot_schedule import PANEL_ARMS, figure  # noqa: E402


def payload_of(mixes: list[str], arms: list[str]) -> dict:
    grid = [1.0, 10.0, 100.0]
    curves = {"grid": grid, "vi": [9.0, 3.0, 1e-9]}
    for mix in mixes:
        for arm in arms:
            curves[f"{mix}|{arm}"] = [5.0, 2.0, 0.5]

    return {
        "config": {"problem": {"gamma": 0.95, "num_assets": 3, "q_max": 10}},
        "mixes": [[int(n) for n in m.strip("()").split(",")] for m in mixes],
        "curves": curves,
        "vi": {"final": {"billed": 100, "err_inf": 1e-9}},
    }


def test_figure_draws_one_panel_per_schedule_mix():
    data = payload_of(["(5,2)", "(2,5)", "(1,20)"], list(PANEL_ARMS))

    fig = figure(data, "light")

    assert len(fig.axes) == 3
    plt.close(fig)


def test_figure_omits_arms_the_paper_does_not_carry():
    ## geometric_slow and the fixed 0.1 / 0.5 arms are context in the result
    ## file, not claims in the figure.
    data = payload_of(["(2,5)"], [*PANEL_ARMS, "geometric_slow", "fixed_0.5"])

    fig = figure(data, "light")

    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "geometric_slow" not in labels
    assert "fixed_0.5" not in labels
    plt.close(fig)


def test_each_panel_carries_the_stopped_vi_reference():
    data = payload_of(["(5,2)", "(2,5)"], list(PANEL_ARMS))

    fig = figure(data, "light")

    for ax in fig.axes:
        labels = {line.get_label() for line in ax.get_lines()}
        assert "stopped VI" in labels
    plt.close(fig)


def test_series_are_distinguishable_without_colour():
    ## Light-mode aqua sits below 3:1 on the surface and dark-mode tritan
    ## separation is weak, so linestyle carries identity too -- and a paper
    ## figure may well be printed in greyscale.
    data = payload_of(["(2,5)"], list(PANEL_ARMS))

    fig = figure(data, "light")

    ## `get_linestyle` normalises every custom dash tuple to "--", so it cannot
    ## see the difference. `_dash_pattern` is the pattern actually applied.
    patterns = [
        (line._dash_pattern[0], tuple(line._dash_pattern[1] or ()))
        for line in fig.axes[0].get_lines()
    ]
    assert len(set(patterns)) == len(patterns)
    plt.close(fig)
