# Workshop draft

`residual_feedback_mlxor_2026.tex` is a four-page-main-text draft for the
NeurIPS 2026 MLxOR workshop. References start on the following page.

The official `neurips_2026.sty` used to build the PDF is included beside the
TeX source. The current 2026 MLxOR call for papers has not yet published its
own formatting details; the draft uses the NeurIPS 2026 single-blind workshop
option and the previous MLxOR limit of four main-body pages plus references.
Recheck the workshop call before submission.

From this directory, build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error residual_feedback_mlxor_2026.tex
```

Regenerate the manuscript figure from the repository root with:

```bash
.venv/bin/python scripts/plot_schedule.py results/schedule_inventory_n3.json
```

The author block is filled from the existing MLxOR manuscript in the local
workspace. Confirm the name, affiliation, and email before submission.
