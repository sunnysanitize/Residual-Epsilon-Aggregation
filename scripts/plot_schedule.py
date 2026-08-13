"""The paper's main figure: error against billed backups, one panel per mix.

Indexing the x axis by work rather than by wall-clock is the point. The arms do
very different amounts of work per iteration -- an aggregate sweep costs one
backup per group where a global sweep costs one per state -- so an iteration
axis flatters the arms that spend most of their iterations in the cheap phase,
and a wall-clock axis is not reproducible across machines or reruns.

The figure carries three of the seven arms plus the full-state VI reference. The
rest are in the result file; they are context, not the claim.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

## dataviz reference palette, categorical slots 1-3. Validated all-pairs in both
## modes. Light-mode aqua sits below 3:1 on the surface and dark-mode tritan
## separation is thin, so linestyle carries identity as well as hue -- which a
## figure that may be printed in greyscale wants in any case.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}

FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

## arm key -> (label, palette slot, linestyle)
PANEL_ARMS: dict[str, tuple[str, int, Any]] = {
    "residual": ("residual", 0, "solid"),
    "geometric_fast": ("geometric (fast)", 1, (0, (5, 2))),
    "fixed_0.05": (r"fixed $\varepsilon=0.05$", 2, (0, (1, 1.6))),
}

VI_STYLE = (0, (6, 2, 1, 2))


def figure(data: dict[str, Any], mode: str) -> Any:
    c = THEME[mode]
    curves = data["curves"]
    grid = curves["grid"]
    mixes = [f"({g},{a})" for g, a in data["mixes"]]

    fig, axes = plt.subplots(
        1, len(mixes), figsize=(3.1 * len(mixes), 3.0), dpi=200, sharey=True
    )
    axes = list(axes) if len(mixes) > 1 else [axes]
    fig.patch.set_facecolor(c["surface"])

    for ax, mix in zip(axes, mixes, strict=True):
        ax.set_facecolor(c["surface"])

        ## The reference goes down first so the arms read on top of it.
        ax.plot(
            grid,
            curves["vi"],
            linestyle=VI_STYLE,
            linewidth=1.6,
            color=c["muted"],
            label="full-state VI",
            zorder=2,
        )

        for arm, (label, slot, style) in PANEL_ARMS.items():
            series = curves.get(f"{mix}|{arm}")
            if series is None:
                continue
            ax.plot(
                grid,
                series,
                linestyle=style,
                linewidth=2.0,
                color=c["series"][slot],
                label=label,
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"$(\\ell_g,\\ell_a) = {mix}$", fontsize=10, color=c["ink"], pad=8
        )
        ax.set_xlabel("billed backups", fontsize=9, color=c["ink2"])
        ax.grid(True, which="major", color=c["grid"], linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=c["muted"], labelsize=8, which="both")

    axes[0].set_ylabel(r"median $\ell_\infty$ error", fontsize=9, color=c["ink2"])

    legend = axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    problem = data["config"]["problem"]
    fig.suptitle(
        (
            "Rules reaching $\\varepsilon_{\\min}$ have similar error floors — "
            f"inventory $N={problem['num_assets']}$, "
            f"$Q={problem['q_max']}$, $\\gamma={problem['gamma']:g}$"
        ),
        fontsize=11,
        color=c["ink"],
        x=0.01,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/plot_schedule.py")
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = FONTS
    data = json.loads(args.sweep.read_text())
    if "curves" not in data:
        print(
            f"{args.sweep} has no compute-matched curves: it is probably a "
            "sweep.py result, not a schedule_sweep.py one",
            file=sys.stderr,
        )
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        fig = figure(data, mode)
        out = args.outdir / f"{args.sweep.stem}_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
