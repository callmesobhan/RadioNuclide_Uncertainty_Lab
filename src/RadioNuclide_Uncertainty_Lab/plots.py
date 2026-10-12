"""Static figures (requires the optional ``plot`` extra: ``pip install .[plot]``).

Design rules followed: one scale per axis, thin marks, recessive grid, text in ink colours
(never the series colour), and a legend whenever a chart has two or more encodings.
The budget chart nests first-order inside total-order in one hue so the light remainder
reads directly as "interaction share"; the factor x output matrix is a single-hue sequential
heatmap with every visible cell labelled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .budget import BudgetMatrix, ErrorBudget
from .propagation import UncertaintyResult
from .sensitivity import RankSensitivity
from .workflow import Analysis

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_LIGHT = "#86b6ef"  # sequential step 250
BLUE_DARK = "#1c5cab"  # sequential step 550
BLUE = "#2a78d6"  # categorical slot 1
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def _figure(width: float, height: float):
    try:
        from matplotlib.figure import Figure
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib: pip install 'RadioNuclide_Uncertainty_Lab[plot]'"
        ) from err
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.add_subplot()
    _style(ax)
    return fig, ax


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.label.set_color(INK_2)


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.97, title, color=INK, fontsize=12, fontweight="medium", va="top")
    fig.text(0.02, 0.915, subtitle, color=INK_2, fontsize=9, va="top")


def budget_plot(
    budget: ErrorBudget, path: str | Path, max_rows: int = 12, min_share: float = 0.005
) -> Path:
    visible = [ln for ln in budget.lines if ln.st >= min_share] or list(budget.lines[:1])
    lines = visible[:max_rows][::-1]  # largest at the top
    hidden = len(budget.lines) - len(lines)
    fig, ax = _figure(7.5, 1.6 + 0.5 * len(lines))
    y = np.arange(len(lines))
    st = [ln.st for ln in lines]
    s1 = [ln.s1 for ln in lines]
    ax.barh(
        y,
        st,
        height=0.5,
        color=BLUE_LIGHT,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="Total-order (S1 + interactions)",
    )
    ax.barh(
        y,
        s1,
        height=0.5,
        color=BLUE_DARK,
        edgecolor=SURFACE,
        linewidth=1.5,
        label="First-order (S1, factor acting alone)",
    )
    for yi, ln in zip(y, lines, strict=True):
        label = "<1%" if ln.st < 0.005 else f"{ln.st:.0%}"
        ax.text(max(ln.st, 0) + 0.012, yi, label, va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [ln.factor for ln in lines])
    ax.set_xlim(0, max(max(st), 0.05) * 1.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of output variance")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    note = f"; {hidden} factors under {min_share:.1%} not shown" if hidden > 0 else ""
    _title(
        fig,
        f"Error budget: {budget.output}",
        f"Output SD = {budget.relative_sd:.1%} of mean; interactions = "
        f"{budget.interaction_share:.1%}{note}",
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.16)
    return _save(fig, path)


def budget_matrix_plot(
    matrix: BudgetMatrix, path: str | Path, min_share: float = 0.005, max_rows: int = 22
) -> Path:
    keep = np.flatnonzero(matrix.st.max(axis=1) >= min_share)[:max_rows]
    st = matrix.st[keep]
    factors = [matrix.factors[i] for i in keep]
    n_rows, n_cols = st.shape
    fig, ax = _figure(3.0 + 1.25 * n_cols, 1.8 + 0.34 * n_rows)
    ax.spines["bottom"].set_visible(False)
    top = float(st.max())
    for i in range(n_rows):
        for j in range(n_cols):
            v = max(float(st[i, j]), 0.0)
            level = int(round(v / top * (len(SEQ_BLUE) - 1))) if v >= min_share else -1
            color = SEQ_BLUE[level] if level >= 0 else SURFACE
            ax.add_patch(_rect(j, i, color))
            if v >= min_share:
                dark = level >= 7
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff" if dark else INK,
                )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, [o.replace("_to_", "\nto\n") for o in matrix.outputs])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows) + 0.5, factors)
    hidden = int((matrix.st.max(axis=1) >= min_share).sum()) - len(keep)
    note = f" ({hidden} more below the cut)" if hidden > 0 else ""
    _title(
        fig,
        "Where each output's variance comes from",
        f"Total-order share per factor; blank = under {min_share:.1%}{note}",
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.80, bottom=0.04)
    return _save(fig, path)


def _rect(col: int, row: int, color: str):
    from matplotlib.patches import Rectangle

    return Rectangle((col, row), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2)


def rank_plot(
    rank: RankSensitivity, path: str | Path, by: str = "prcc", max_rows: int = 12
) -> Path:
    items = rank.ranking(by)[:max_rows][::-1]
    fig, ax = _figure(7.5, 1.6 + 0.42 * len(items))
    y = np.arange(len(items))
    vals = [v for _, v in items]
    ax.barh(y, vals, height=0.45, color=BLUE)
    ax.axvline(0, color=AXIS, linewidth=1)
    for yi, v in zip(y, vals, strict=True):
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            yi,
            f"{v:+.2f}".replace("-", "−"),
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(y, [n for n, _ in items])
    ax.set_xlim(-1.1, 1.1)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"{by.upper()} with output (sign = direction of effect)")
    _title(
        fig,
        f"Parameter ranking (screening): {rank.output}",
        f"Rank-based, per parameter; linear R² = {rank.r2:.2f}",
    )
    fig.subplots_adjust(left=0.40, right=0.95, top=0.82, bottom=0.16)
    return _save(fig, path)


def distribution_plot(
    result: UncertaintyResult, output: str | None, path: str | Path, level: float = 0.95
) -> Path:
    s = result.summary(output, level)
    y = result[s.output]
    fig, ax = _figure(7.5, 3.6)
    lo, hi = np.quantile(y, [0.0005, 0.9995])  # keep rare tail samples off the axis
    ax.hist(y, bins=60, range=(lo, hi), color=BLUE, edgecolor=SURFACE, linewidth=0.8, alpha=0.9)
    ax.set_xlim(lo, hi)
    top = ax.get_ylim()[1]
    for x, label, ha in (
        (s.ci_low, f"{s.ci_low:.3g}", "right"),
        (s.ci_high, f"{s.ci_high:.3g}", "left"),
    ):
        ax.axvline(x, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x, top * 0.97, f" {label} ", color=INK, fontsize=9, ha=ha, va="top")
    ax.axvline(s.median, color=INK, linewidth=1.5)
    ax.set_yticks([])
    ax.set_xlabel(f"{s.output.replace('_', ' ')} ({s.unit})" if s.unit else s.output)
    _title(
        fig,
        f"{result.model_name}: {s.output}",
        f"Median {s.median:.3g} {s.unit}; {s.level:.0%} interval "
        f"[{s.ci_low:.3g}, {s.ci_high:.3g}] (dashed); n = {s.n:,}",
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.82, bottom=0.17)
    return _save(fig, path)


def intervals_plot(analysis: Analysis, path: str | Path) -> Path:
    """Median and central interval of every output, one panel per unit (never a shared axis
    for different units)."""
    try:
        from matplotlib.figure import Figure
    except ImportError as err:  # pragma: no cover
        raise ImportError("plotting needs matplotlib") from err
    by_unit: dict[str, list[str]] = {}
    for o in analysis.model.outputs:
        by_unit.setdefault(analysis.model.unit(o), []).append(o)
    rows = sum(len(v) for v in by_unit.values())
    fig = Figure(figsize=(7.5, 1.5 + 0.5 * rows + 0.5 * len(by_unit)), facecolor=SURFACE)
    axes = fig.subplots(
        len(by_unit),
        1,
        gridspec_kw={"height_ratios": [len(v) for v in by_unit.values()]},
        squeeze=False,
    )[:, 0]
    level = analysis.run.level
    for ax, (unit, names) in zip(axes, by_unit.items(), strict=True):
        _style(ax)
        y = np.arange(len(names))[::-1]
        for yi, o in zip(y, names, strict=True):
            s = analysis.summaries[o]
            ax.plot(
                [s.ci_low, s.ci_high], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round"
            )
            ax.plot(
                [s.median],
                [yi],
                "o",
                color=BLUE,
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
            )
            ax.text(
                s.ci_high,
                yi + 0.18,
                f"  {s.median:.3g} [{s.ci_low:.3g}, {s.ci_high:.3g}]",
                color=INK,
                fontsize=8.5,
                va="bottom",
                ha="left",
            )
        ax.set_yticks(y, names)
        ax.set_ylim(-0.6, len(names) - 0.2)
        ax.xaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
        xmax = max(analysis.summaries[o].ci_high for o in names)
        ax.set_xlim(0, xmax * 1.55)
        ax.set_xlabel(unit if unit else "ratio (dimensionless)")
    _title(fig, "Outputs with uncertainty", f"Median (dot) and {level:.0%} central interval (line)")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.84, bottom=0.12, hspace=0.9)
    return _save(fig, path)


def write_figures(analysis: Analysis, out_dir: str | Path) -> list[Path]:
    """Write all figures for an analysis; returns the paths written."""
    out = Path(out_dir)
    outputs = list(analysis.model.outputs)
    multi = len(outputs) > 1
    written: list[Path] = []
    if multi:
        written.append(intervals_plot(analysis, out / "intervals.png"))
        if analysis.matrix is not None:
            written.append(budget_matrix_plot(analysis.matrix, out / "budget_matrix.png"))
    for o in outputs:
        folder = out / "outputs" / o if multi else out
        written.append(
            distribution_plot(analysis.result, o, folder / "distribution.png", analysis.run.level)
        )
        written.append(rank_plot(analysis.ranks[o], folder / "rank.png"))
        if analysis.budgets is not None:
            written.append(budget_plot(analysis.budgets[o], folder / "budget.png"))
    return written


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return path
