# %% helper_functions.py — shared visualization utilities
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from pathlib import Path
import warnings
import logging

# Default to non-interactive (batch) unless a caller flips it.
DEV_MODE = False


# ---------------------------------------------------------------------
# Global chart style
# ---------------------------------------------------------------------
# Suppress matplotlib font warnings and set font with fallback
warnings.filterwarnings("ignore", message=".*findfont.*", category=UserWarning, module="matplotlib")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Try Montserrat, fallback to sans-serif if not available
try:
    # Check if Montserrat is available
    from matplotlib import font_manager
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    if "Montserrat" in available_fonts:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        plt.rcParams["font.family"] = "sans-serif"
except:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update(
    {
        # Bigger, Slides-friendly typography
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": True,
        # High contrast text (Slides-friendly)
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "legend.edgecolor": "#dddddd",
        # Slightly heavier defaults
        "font.weight": "regular",
        "axes.titleweight": "bold",
        # Make exported charts look crisp in Google Slides
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        # Cleaner look by default
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------
# Palettes and category orders
# ---------------------------------------------------------------------
default_quintile_colors = ["#808080", "#c5c5c5", "#78daf4", "#00baeb", "#0381a2"]

# ============================================================
# Shared Palette (Low → High)
# ============================================================
default_quartile_colors = ["#808080", "#c5c5c5", "#00baeb", "#0381a2"]

# ---------------------------------------------------------------------
# Color mappings
# ---------------------------------------------------------------------

CERS_LEVELS = [
    "Level 1 - Standard Not Met",
    "Level 2 - Standard Nearly Met",
    "Level 3 - Standard Met",
    "Level 4 - Standard Exceeded",
]

CERS_LEVEL_COLORS = {
    lvl: default_quartile_colors[i] for i, lvl in enumerate(CERS_LEVELS)
}

# NWEA quintile order (Low → High) maps to default_quintile_colors
NWEA_QUINTILE_ORDER = ["Low", "LoAvg", "Avg", "HiAvg", "High"]
NWEA_QUINTILE_COLORS = {
    q: default_quintile_colors[i] for i, q in enumerate(NWEA_QUINTILE_ORDER)
}

# STAR levels (1 → 4) map to default_quartile_colors
STAR_LEVELS = [
    "1 - Standard Not Met",
    "2 - Standard Nearly Met",
    "3 - Standard Met",
    "4 - Standard Exceeded",
]
STAR_LEVEL_COLORS = {
    lvl: default_quartile_colors[i] for i, lvl in enumerate(STAR_LEVELS)
}

# i-Ready placement / performance bands (Low → High)
IREADY_ORDER = [
    "3+ Below",
    "2 Below",
    "1 Below",
    "Early On",
    "Mid/Above",
]

# Color map for i-Ready bands using default_quintile_colors
IREADY_COLORS = {
    band: default_quintile_colors[i] for i, band in enumerate(IREADY_ORDER)
}

# ---------- i-Ready ----------
IREADY_CAT_COL = "relative_placement"
IREADY_SCORE_COL = "scale_score"
IREADY_TIME_COL_OPTIONS = ["academicyear", "testwindow"]

IREADY_HIGH_GROUP = ["Early On", "Mid/Above"]
IREADY_LOW_GROUP = ["3+ Below", "2 Below"]

# Mapping from original dataset labels to truncated labels for charts
IREADY_LABEL_MAP = {
    "3 or More Grade Levels Below": "3+ Below",
    "2 Grade Levels Below": "2 Below",
    "1 Grade Level Below": "1 Below",
    "Early On Grade Level": "Early On",
    "Mid or Above Grade Level": "Mid/Above",
}


# ---------------------------------------------------------------------
# i-Ready label mapper — safely handles Series or DataFrame
# ---------------------------------------------------------------------
def _map_iready_labels(df_or_series, col: str = "relative_placement"):
    """
    Convert i-Ready relative placement categories to their truncated chart labels.

    Works with either a DataFrame (in place) or a Series (returns mapped Series).

    Example:
        df = _map_iready_labels(df, "relative_placement")
        or
        df["placement_mapped"] = _map_iready_labels(df["relative_placement"])
    """
    if isinstance(df_or_series, pd.Series):
        return df_or_series.replace(IREADY_LABEL_MAP)
    elif isinstance(df_or_series, pd.DataFrame):
        if col in df_or_series.columns:
            df_or_series[col] = df_or_series[col].replace(IREADY_LABEL_MAP)
        return df_or_series
    else:
        raise TypeError("Expected a pandas DataFrame or Series for _map_iready_labels.")


# ============================================================
# Placement Groupings
# ============================================================
# (Canonical definitions kept in assessment sections below)


# ----------------------------------------------
# Resolve which tables to query from yaml settings
# ----------------------------------------------
def resolve_table(cfg: dict, key: str) -> str:
    partner = cfg["partner_name"]
    base_tpl = cfg["sources"][key]
    overrides = cfg.get("sources_overrides", {}) or {}
    suffixes = cfg.get("sources_suffix", {}) or {}

    if key in overrides and overrides[key]:
        return overrides[key].format(partner_name=partner)

    base = base_tpl.format(partner_name=partner)
    suff = suffixes.get(key, "")
    return f"{base}{suff}"


# ----------------------------------------------
# Save and Render
# ----------------------------------------------


def _save_and_render(
    fig: plt.Figure, out_path: Path | None = None, dev_mode: bool | None = None
):
    """
    Save the figure if a path is provided, then either show (DEV_MODE)
    or close to keep batch runs non-interactive.
    Precedence: explicit dev_mode arg > helper_functions.DEV_MODE
    """
    if out_path is not None:
        # Ensure consistent high-res export (Slides-friendly)
        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
            pad_inches=0.15,
        )

    effective_dev = DEV_MODE if dev_mode is None else bool(dev_mode)
    if effective_dev:
        plt.show(block=False)
        plt.pause(0.001)  # allow nonblocking preview in VSCode
    else:
        plt.close(fig)


# ----------------------------------------------
# Normalize Star Subject
# ----------------------------------------------
def normalize_star_subject(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a 'subject' column standardized to 'Reading' or 'Math'.
    Uses activity_type. Leaves other values as NaN.
    """
    df = df_in.copy()

    def _to_subject(val):
        s = str(val).lower()
        if "math" in s:
            return "Math"
        if "read" in s:
            return "Reading"
        return None

    if "subject" not in df.columns or df["subject"].isna().all():
        df["subject"] = df.get("activity_type", pd.Series(index=df.index)).apply(
            _to_subject
        )

    df["subject"] = df["subject"].where(df["subject"].isin(["Reading", "Math"]))
    return df


# ---------------------------------------------------------------------
# Legend builder (keeps legend order == plotting order)
# ---------------------------------------------------------------------
def _legend_for_levels(levels_order, color_map):
    return [
        Patch(facecolor=color_map.get(l, "#ccc"), edgecolor="white", label=l)
        for l in levels_order
    ]


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
def draw_nested_donut(
    ax,
    inner_vals,
    inner_cols,
    outer_vals,
    outer_cols,
    inner_width=0.4,
    outer_width=0.3,
    center_text=None,
):
    inner_vals = np.nan_to_num(inner_vals)
    outer_vals = np.nan_to_num(outer_vals)

    if np.sum(inner_vals) == 0 and np.sum(outer_vals) == 0:
        ax.text(
            0, 0, "No Data", ha="center", va="center", fontsize=12, fontweight="bold"
        )
        ax.axis("equal")
        return

    inner_radius = 1.0
    outer_radius = 1.3

    def pct_label(pct):
        return f"{pct:.1f}%" if pct >= 3 else ""

    ax.pie(
        inner_vals,
        radius=inner_radius,
        startangle=90,
        counterclock=False,
        colors=inner_cols,
        wedgeprops=dict(width=inner_width, edgecolor="white"),
        autopct=pct_label,
        pctdistance=(inner_radius - inner_width / 2) / inner_radius,
        textprops=dict(color="black", fontsize=10),
    )

    if np.sum(outer_vals) > 0:
        ax.pie(
            outer_vals,
            radius=outer_radius,
            startangle=90,
            counterclock=False,
            colors=outer_cols,
            wedgeprops=dict(width=outer_width, edgecolor="white"),
            autopct=pct_label,
            pctdistance=(outer_radius - outer_width / 2) / outer_radius,
            textprops=dict(color="black", fontsize=10),
        )

    ax.add_artist(plt.Circle((0, 0), inner_radius - inner_width, color="white"))
    if center_text:
        ax.text(
            0, 0, center_text, ha="center", va="center", fontsize=9, fontweight="bold"
        )
    ax.axis("equal")


def plot_100_stacked(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    weight_col: str | None = None,
    x_order: list | None = None,
    y_order: list | None = None,
    color_map: dict | None = None,
    orient: str = "v",
    show_pct_labels: bool = True,
    label_min_pct: float = 5.0,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_ncol: int = 1,
    bar_gap: float = 0.15,
    n_labels: bool = False,
    fmt_percent: str = "{:.0f}%",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    # NEW legend controls (defaults = district standard)
    legend_outside: bool = True,
    legend_outside_bbox: tuple = (0.9, 0.85),
    legend_reverse: bool = True,
    legend_title: str | None = None,
):
    agg = (
        df.groupby([x_col, y_col], dropna=False)[weight_col].sum()
        if weight_col
        else df.groupby([x_col, y_col], dropna=False).size()
    )
    agg = agg.rename("value").reset_index()

    if x_order is None:
        x_order = df[x_col].dropna().unique().tolist()
    if y_order is None:
        y_order = df[y_col].dropna().unique().tolist()

    idx = pd.MultiIndex.from_product([x_order, y_order], names=[x_col, y_col])
    agg = agg.set_index([x_col, y_col]).reindex(idx, fill_value=0).reset_index()

    pv = agg.pivot(index=x_col, columns=y_col, values="value").fillna(0)
    totals = pv.sum(axis=1).replace(0, np.nan)
    props = (pv.T / totals).T.fillna(0)
    n_by_x = pv.sum(axis=1).astype(int)

    color_map = color_map or {}
    colors = [color_map.get(cat, None) for cat in y_order]

    bottoms = np.zeros(len(x_order))
    idxs = np.arange(len(x_order))
    width = 0.8 * (1 - bar_gap)

    def pct_text(v):
        return fmt_percent.format(v * 100)

    for i, cat in enumerate(y_order):
        vals = props.reindex(x_order)[cat].to_numpy()
        if orient == "v":
            bars = ax.bar(
                idxs,
                vals,
                width=width,
                bottom=bottoms,
                color=colors[i],
                edgecolor="white",
                linewidth=0.5,
                label=str(cat),
            )
            if show_pct_labels:
                for j, b in enumerate(bars):
                    v = vals[j]
                    # Suppress labels below 5%
                    if v * 100 >= 5.0:
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            bottoms[j] + v / 2,
                            pct_text(v),
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
        else:
            bars = ax.barh(
                idxs,
                vals,
                height=width,
                left=bottoms,
                color=colors[i],
                edgecolor="white",
                linewidth=0.5,
                label=str(cat),
            )
            if show_pct_labels:
                for j, b in enumerate(bars):
                    v = vals[j]
                    # Suppress labels below 5%
                    if v * 100 >= 5.0:
                        ax.text(
                            bottoms[j] + v / 2,
                            b.get_y() + b.get_height() / 2,
                            pct_text(v),
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
        bottoms += vals

    if orient == "v":
        ax.set_xticks(idxs)
        ax.set_xticklabels([str(x) for x in x_order])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([pct_text(t) for t in [0, 0.25, 0.5, 0.75, 1.0]])
        if n_labels:
            for j, x in enumerate(idxs):
                ax.text(
                    x,
                    -0.045,
                    f"n={n_by_x.reindex(x_order).iloc[j]:,}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
            ax.margins(y=0.08)
    else:
        ax.set_yticks(idxs)
        ax.set_yticklabels([str(x) for x in x_order])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([pct_text(t) for t in [0, 0.25, 0.5, 0.75, 1.0]])
        if n_labels:
            for j, y in enumerate(idxs):
                ax.text(
                    1.02,
                    y,
                    f"n={n_by_x.reindex(x_order).iloc[j]:,}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    transform=ax.get_yaxis_transform(),
                )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if show_legend:
        order_for_legend = list(reversed(y_order)) if legend_reverse else y_order
        handles = _legend_for_levels(order_for_legend, color_map)
        ttl = legend_title or y_col
        if legend_outside:
            ax.figure.legend(
                handles=handles,
                title=ttl,
                loc="center left",
                bbox_to_anchor=legend_outside_bbox,
                frameon=False,
            )
        else:
            ax.legend(
                handles=handles,
                title=ttl,
                loc=legend_loc,
                ncol=legend_ncol,
                frameon=False,
            )
    ax.grid(False)
    return {"props": props, "n": n_by_x}


def plot_crosstab_heatmap(
    ax,
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    *,
    row_order: list | None = None,
    col_order: list | None = None,
    normalize: str | None = None,
    annot: bool = True,
    fmt: str = "d",
    cmap=None,
    cbar: bool = False,
    na_fill: int | float = 0,
):
    ct = (
        pd.crosstab(df[row_col], df[col_col])
        if normalize is None
        else pd.crosstab(df[row_col], df[col_col], normalize=normalize)
    )
    if row_order is not None:
        ct = ct.reindex(row_order, fill_value=na_fill)
    if col_order is not None:
        ct = ct.reindex(columns=col_order, fill_value=na_fill)

    im = ax.imshow(ct.values, aspect="auto", cmap=cmap)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_yticks(range(ct.shape[0]), labels=[str(i) for i in ct.index])
    ax.set_xticks(range(ct.shape[1]), labels=[str(c) for c in ct.columns], rotation=0)

    if annot:
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.iat[i, j]
                txt = (
                    f"{val:.0%}"
                    if (normalize is not None and fmt == ".0%")
                    else f"{int(val)}"
                )
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    return ct


def plot_corr_scatter(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color=None,
    s: float = 12,
    alpha: float = 0.6,
    add_line: bool = True,
    show_r: bool = True,
    r_loc: tuple = (0.05, 0.95),
):
    df2 = df[[x, y]].dropna()
    if df2.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return np.nan

    ax.scatter(df2[x], df2[y], s=s, alpha=alpha, color=color, edgecolors="none")
    r = np.corrcoef(df2[x], df2[y])[0, 1] if len(df2) >= 2 else np.nan

    if add_line and len(df2) >= 2:
        m, b = np.polyfit(df2[x], df2[y], 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, m * xs + b, linestyle="--", linewidth=1.2, color="black")

    if show_r and not np.isnan(r):
        ax.text(
            r_loc[0],
            r_loc[1],
            f"r = {r:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return r


def plot_grouped_bar(
    ax,
    df: pd.DataFrame,
    x: str,
    series: str,
    value: str,
    *,
    x_order: list | None = None,
    series_order: list | None = None,
    colors: dict | None = None,
    bar_width: float = 0.35,
    gap: float = 0.15,
    show_labels: bool = True,
    label_fmt: str = "{:.1f}",
    threshold: float | None = None,
):
    if x_order is None:
        x_order = df[x].dropna().unique().tolist()
    if series_order is None:
        series_order = df[series].dropna().unique().tolist()

    x_idx = np.arange(len(x_order))
    n_series = len(series_order)
    total_width = 1.0 - gap
    width = total_width / n_series

    for k, ser in enumerate(series_order):
        sdf = df[df[series] == ser].set_index(x).reindex(x_order)
        vals = sdf[value].to_numpy(dtype=float)
        offs = -total_width / 2 + width / 2 + k * width
        bars = ax.bar(
            x_idx + offs,
            vals,
            width=width,
            color=None if colors is None else colors.get(ser),
            edgecolor="white",
            linewidth=0.5,
            label=str(ser),
        )
        if show_labels:
            for b, v in zip(bars, vals):
                if pd.notna(v):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v + (1 if v >= 0 else -1),
                        label_fmt.format(v),
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=9,
                        fontweight="bold",
                    )

    ax.set_xticks(x_idx, labels=[str(i) for i in x_order])
    ax.set_xlim(x_idx[0] - 0.5, x_idx[-1] + 0.5)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="gray")
    ax.set_ylabel(value)
    ax.legend(frameon=False)
    return ax


def plot_line_by_group(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    *,
    groups_order: list | None = None,
    x_sort_numeric: bool = True,
    marker: str = "o",
    linewidth: float = 2.0,
    alpha: float = 0.9,
    show_n: bool = False,
):
    if groups_order is None:
        groups_order = df[group].dropna().unique().tolist()

    for g, gdf in df.groupby(group):
        if g not in groups_order:
            continue
        t = gdf.copy()
        if x_sort_numeric:
            t[x] = pd.to_numeric(t[x], errors="coerce")
        t = t.groupby(x, dropna=False).agg(y=(y, "mean"), n=(y, "count")).reset_index()
        t = t.sort_values(x)
        ax.plot(
            t[x], t[y], marker=marker, linewidth=linewidth, alpha=alpha, label=str(g)
        )
        if show_n:
            for xv, yv, nn in zip(t[x], t[y], t["n"]):
                ax.text(xv, yv, f"n={int(nn):,}", fontsize=8, ha="left", va="bottom")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(frameon=False)
    return ax


def plot_bar_line_dual(
    ax_bar,
    ax_line,
    x,
    bar_y,
    line_y,
    df,
    *,
    x_order=None,
    bar_color=None,
    line_color=None,
    line_marker="o",
    bar_label=None,
    line_label=None,
    bar_ylim=None,
    line_ylim=(0, 1),
    line_as_pct=True,
):
    t = df[[x, bar_y, line_y]].copy()
    if x_order is None:
        x_order = t[x].dropna().unique().tolist()
    t = t.set_index(x).reindex(x_order)

    bars = ax_bar.bar(
        x_order,
        t[bar_y].to_numpy(dtype=float),
        color=bar_color,
        label=bar_label,
        edgecolor="white",
        linewidth=0.5,
    )
    if bar_ylim:
        ax_bar.set_ylim(*bar_ylim)
    ax_bar.set_ylabel(bar_y)

    if ax_line is None:
        ax_line = ax_bar.twinx()
    y = t[line_y].to_numpy(dtype=float)
    ax_line.plot(x_order, y, marker=line_marker, color=line_color, label=line_label)
    if line_as_pct:
        ax_line.set_ylim(*line_ylim)
        ax_line.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_line.set_yticklabels(
            [f"{int(v*100)}%" for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        )
    elif line_ylim:
        ax_line.set_ylim(*line_ylim)
    ax_line.set_ylabel(line_y)

    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax_line.get_legend_handles_labels()
    if h1 or h2:
        ax_bar.legend(h1 + l2, l1 + l2, frameon=False, loc="upper left")
    return bars


def facet_grid(
    df,
    facet_col,
    plot_fn,
    facet_order=None,
    ncols=3,
    figsize=(12, 8),
    sharex=False,
    sharey=False,
    hspace=0.35,
    wspace=0.25,
    reserve_bottom=0.18,
    apply_tight_layout=True,
    **plot_kwargs,
):
    import math

    if facet_order is None:
        facet_order = df[facet_col].dropna().unique().tolist()
    n = len(facet_order)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    for i, val in enumerate(facet_order):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = df[df[facet_col] == val]
        plot_fn(ax=ax, df=sub, **plot_kwargs)
        ax.set_title(str(val))
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    if apply_tight_layout:
        fig.tight_layout()

    for ax in axes.ravel():
        if not ax.has_data():
            continue
        box = ax.get_position()
        dy = reserve_bottom * box.height
        ax.set_position([box.x0, box.y0 + dy, box.width, box.height - dy])
    return fig, axes


def adjust_facet_boxes(fig, axes, reserve_bottom=0.12):
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        if not ax.has_data():
            continue
        box = ax.get_position()
        dy = reserve_bottom * box.height
        ax.set_position([box.x0, box.y0 + dy, box.width, box.height - dy])


def plot_distribution_by_group(
    ax, df, x_col, y_col, kind="box", order=None, show_n=True, ref=None
):
    t = df[[x_col, y_col]].dropna()
    if order is None:
        order = t[x_col].dropna().unique().tolist()
    data = [t[t[x_col] == k][y_col].to_numpy(dtype=float) for k in order]
    if kind == "violin":
        parts = ax.violinplot(data, showmeans=False, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
    else:
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for b in bp["boxes"]:
            b.set_edgecolor("black")
            b.set_linewidth(0.5)
    ax.set_xticks(range(1, len(order) + 1), labels=[str(k) for k in order])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if ref is not None:
        ax.axhline(ref, linestyle="--", color="gray")
    if show_n:
        for i, arr in enumerate(data, start=1):
            ax.text(
                i,
                ax.get_ylim()[0],
                f"n={len(arr)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    return ax


# ---------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------
def _slugify(text: str) -> str:
    s = str(text).strip().lower().replace("&", "and")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace() or ch in ("/", "\\", ",", ".", "(", ")", "[", "]", "'", '"'):
            out.append("_")
    return "_".join(filter(None, "".join(out).split("_")))


def save_chart(
    fig,
    name: str,
    *,
    level: str = "district",
    school: str | None = None,
    charts_dir: str | None = None,
    dpi: int = 200,
    fmt: tuple[str, ...] = ("png",),
):
    from pathlib import Path
    import os
    import tempfile

    # Default to runner-provided temp charts dir when available; otherwise fall back to
    # a per-run system temp directory to avoid writing into the repo (../charts).
    if charts_dir is None:
        charts_dir = (
            os.getenv("NWEA_BOY_CHARTS_DIR")
            or os.getenv("NWEA_MOY_CHARTS_DIR")
            or os.getenv("NWEA_CHARTS_DIR")
            or tempfile.mkdtemp(prefix="parsec_nwea_charts_")
        )

    base = Path(charts_dir)
    if level.lower() == "district":
        outdir = base / "_district"
    else:
        if not school:
            raise ValueError("school is required when level='site'")
        outdir = base / _slugify(school)
    outdir.mkdir(parents=True, exist_ok=True)

    fname = _slugify(name)
    saved_first = None
    for ext in fmt:
        fpath = outdir / f"{fname}.{ext}"
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if saved_first is None:
            saved_first = fpath
    return saved_first


# ---------------------------------------------------------------------
# Band groups + Insights
# ---------------------------------------------------------------------
BAND_GROUPS = {
    "iready": {
        "high": ["Early On Grade Level", "Mid or Above Grade Level"],
        "low": ["2 Grade Levels Below", "3 or More Grade Levels Below"],
    },
    "nwea_quintile": {"high": ["HiAvg", "High"], "low": ["LoAvg", "Low"]},
}


def _band_share_by_x_group(df, x_col, y_col, levels):
    t = (
        df.assign(_one=1)
        .groupby([x_col, y_col], dropna=False)["_one"]
        .sum()
        .unstack(y_col)
        .fillna(0)
    )
    denom = t.sum(axis=1).replace(0, float("nan"))
    share = t.reindex(columns=levels, fill_value=0).sum(axis=1) / denom
    share = share.fillna(0.0)
    share.index = share.index.astype(str)
    return share


def compute_band_insights(df, x_col, y_col, high_levels, low_levels):
    xs = sorted(df[x_col].dropna().astype(str).unique())
    if len(xs) < 2:
        return None
    x_prev, x_curr = xs[-2], xs[-1]
    s_high = _band_share_by_x_group(df, x_col, y_col, high_levels)
    s_low = _band_share_by_x_group(df, x_col, y_col, low_levels)
    if x_prev not in s_high.index or x_curr not in s_high.index:
        return None
    return {
        "x_prev": x_prev,
        "x_curr": x_curr,
        "delta_high": float(s_high.loc[x_curr] - s_high.loc[x_prev]),
        "delta_low": float(s_low.loc[x_curr] - s_low.loc[x_prev]),
        "years_used": (x_prev, x_curr),
    }


def annotate_facet_insights(
    ax,
    insight,
    *,
    high_label="High",
    low_label="Low",
    y_offset=-0.1,
    line_gap=0.07,
    fontsize=9,
    add_last_two_note=True,
):
    if not insight:
        return

    def fmt(p):
        return f"{'+' if p>0 else ''}{p*100:.1f}%"

    ax.text(
        0.5,
        y_offset,
        "\u2022 " + f"{fmt(insight['delta_high'])} change in {high_label}",
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        0.5,
        y_offset - line_gap,
        "\u2022 " + f"{fmt(insight['delta_low'])} change in {low_label}",
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )
    if add_last_two_note:
        prev_y, curr_y = insight["years_used"]
        ax.text(
            0.5,
            y_offset - 2 * line_gap,
            f"(comparisons use last two years only: {prev_y}\u2192{curr_y})",
            ha="center",
            va="top",
            fontsize=fontsize - 1,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )


# ---------------------------------------------------------------------
# Matched cohort helpers
# ---------------------------------------------------------------------
def build_matched_cohort(
    df_prev: pd.DataFrame,
    df_next: pd.DataFrame,
    *,
    id_col: str = "student_id",
    subj_col: str = "subject",
    band_prev_col: str,
    band_next_col: str,
) -> pd.DataFrame:
    a = (
        df_prev[[id_col, subj_col, band_prev_col]]
        .dropna(subset=[id_col, subj_col, band_prev_col])
        .copy()
    )
    b = (
        df_next[[id_col, subj_col, band_next_col]]
        .dropna(subset=[id_col, subj_col, band_next_col])
        .copy()
    )
    m = a.merge(b, on=[id_col, subj_col], how="inner", suffixes=("_prev", "_next"))
    m = m.rename(
        columns={
            f"{band_prev_col}_prev": "prev_band",
            f"{band_next_col}_next": "next_band",
        }
    )
    return m[[subj_col, "prev_band", "next_band"]].copy()


def plot_matched_cohort_100(
    df_pairs: pd.DataFrame,
    *,
    prev_col: str = "prev_band",
    next_col: str = "next_band",
    facet_by: str | None = None,
    x_order: list[str],
    y_order: list[str],
    color_map: dict[str, str],
    figsize=(10, 6),
    n_labels: bool = False,
):
    def _one_panel(d: pd.DataFrame, title: str):
        ct = (
            d.assign(_one=1)
            .groupby([prev_col, next_col])["_one"]
            .sum()
            .unstack(next_col)
            .reindex(index=x_order, columns=y_order, fill_value=0)
        )
        denom = ct.sum(axis=1).replace(0, float("nan"))
        pct = ct.divide(denom, axis=0).fillna(0.0)

        fig, ax = plt.subplots(figsize=figsize)
        bottom = np.zeros(len(pct))
        for lvl in y_order:
            vals = pct[lvl].values
            ax.bar(
                pct.index.astype(str),
                vals,
                bottom=bottom,
                label=lvl,
                color=color_map.get(lvl, "#cccccc"),
                edgecolor="white",
                linewidth=0.6,
            )
            if n_labels:
                for i, v in enumerate(vals):
                    # Suppress labels below 5%
                    if v * 100 >= 5.0:
                        ax.text(
                            i,
                            bottom[i] + v / 2,
                            f"{v*100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            bottom += vals

        ax.set_ylabel("Percent")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
        ax.set_xlabel("Spring 2025 Placement")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.legend(
            handles=_legend_for_levels(y_order, color_map),
            title="Fall 2026 Placement",
            ncol=min(len(y_order), 5),
            loc="upper right",
            frameon=True,
        )
        return fig, ax

    if facet_by:
        out = {}
        for val in sorted(df_pairs[facet_by].dropna().unique()):
            d = df_pairs[df_pairs[facet_by] == val]
            if d.empty:
                continue
            out[val] = _one_panel(
                d, f"{val} — Matched Cohort (Spring 2025 → Fall 2026)"
            )
        return out
    else:
        return _one_panel(df_pairs, "Matched Cohort (Spring 2025 → Fall 2026)")


def build_grade_progression_pairs(
    df_prev: pd.DataFrame,
    df_next: pd.DataFrame,
    *,
    id_col: str = "student_id",
    subj_col: str = "subject",
    grade_col: str = "student_grade",
    band_prev_col: str,
    band_next_col: str,
    school_col: str = "school",
) -> pd.DataFrame:
    cols_prev = [id_col, subj_col, grade_col, band_prev_col, school_col]
    cols_next = [id_col, subj_col, grade_col, band_next_col, school_col]

    a = (
        df_prev[cols_prev]
        .dropna(subset=[id_col, subj_col, grade_col, band_prev_col])
        .copy()
    )
    b = (
        df_next[cols_next]
        .dropna(subset=[id_col, subj_col, grade_col, band_next_col])
        .copy()
    )

    a[grade_col] = pd.to_numeric(a[grade_col], errors="coerce").astype("Int64")
    b[grade_col] = pd.to_numeric(b[grade_col], errors="coerce").astype("Int64")

    a = a[a[grade_col].between(0, 7)]
    b = b[b[grade_col].between(1, 8)]

    m = a.merge(b, on=[id_col, subj_col], how="inner", suffixes=("_prev", "_next"))
    m = m[m[f"{grade_col}_next"] == (m[f"{grade_col}_prev"] + 1)].copy()

    def _glabel(g: pd.Series) -> pd.Series:
        gv = g.fillna(-1).astype(int)
        return gv.astype(str).replace({"0": "K"})

    cohort = _glabel(m[f"{grade_col}_prev"]) + "→" + _glabel(m[f"{grade_col}_next"])

    out = pd.DataFrame(
        {
            "subject": m[subj_col],
            "cohort": cohort,
            "prev_band": m[f"{band_prev_col}_prev"],
            "next_band": m[f"{band_next_col}_next"],
            "school": m.get(f"{school_col}_prev", m.get(f"{school_col}_next")),
            "prev_grade": m[f"{grade_col}_prev"],
            "next_grade": m[f"{grade_col}_next"],
        }
    )
    return out


def plot_cohort_prev_next_100(
    df_pairs: pd.DataFrame,
    *,
    cohort_order: list[str],
    levels_order: list[str],
    color_map: dict[str, str],
    title: str,
    xlabel: str = "Cohort (Spring → Fall)",
    ylabel: str = "Percent",
    bar_gap: float = 0.35,
    figsize=(14, 6),
    show_pct_labels: bool = True,
    pct_label_min: float = 0.05,
    label_fs: int = 9,
):
    n_by_cohort = (
        df_pairs.groupby("cohort")["prev_band"]
        .size()
        .reindex(cohort_order)
        .fillna(0)
        .astype(int)
    )
    xtick_labels = [f"{c}\n(n={n_by_cohort.get(c,0)})" for c in cohort_order]

    def _share(colname: str) -> pd.DataFrame:
        t = (
            df_pairs.assign(_one=1)
            .groupby(["cohort", colname])["_one"]
            .sum()
            .unstack(colname)
            .reindex(index=cohort_order, columns=levels_order, fill_value=0)
        )
        denom = t.sum(axis=1).replace(0, float("nan"))
        return t.divide(denom, axis=0).fillna(0.0)

    prev_pct = _share("prev_band")
    next_pct = _share("next_band")

    n = len(cohort_order)
    x = np.arange(n, dtype=float)
    dx = bar_gap / 2.0

    fig, ax = plt.subplots(figsize=figsize)

    def _stack_with_labels(xpos, pct_table):
        bottom = np.zeros(len(pct_table), dtype=float)
        for lvl in levels_order:
            vals = pct_table[lvl].values
            ax.bar(
                xpos,
                vals,
                bottom=bottom,
                width=bar_gap,
                color=color_map.get(lvl, "#cccccc"),
                edgecolor="white",
                linewidth=0.6,
                label=lvl,
            )
            if show_pct_labels:
                for i, v in enumerate(vals):
                    # Suppress labels below 5%
                    if v * 100 >= 5.0:
                        ax.text(
                            xpos[i],
                            bottom[i] + v / 2.0,
                            f"{v*100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=label_fs,
                        )
            bottom += vals

    _stack_with_labels(x - dx, prev_pct)
    _stack_with_labels(x + dx, next_pct)

    ax.set_xticks(x, xtick_labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v*100)}%"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    ax.legend(
        handles=_legend_for_levels(levels_order, color_map),
        title="Performance Band",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# Gradient shade under a line: dark at top (line) -> fade to white at axis
# ---------------------------------------------------------------------


def shade_under_line_gradient(ax, x, y, color, y0=0.0, alpha_top=0.9, steps=100):
    """
    Fill under a line with a vertical gradient — dark at the line (top) → light at the axis (bottom).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    y_min = y0

    # Compute evenly spaced horizontal slices
    for i in range(steps):
        # fraction from top (near line) to bottom (axis)
        frac_top = i / steps
        frac_bottom = (i + 1) / steps

        # alpha fades as you move down
        alpha = alpha_top * (1 - frac_top)

        # vertical bounds for this slice
        y_upper = y - (y - y_min) * frac_top
        y_lower = y - (y - y_min) * frac_bottom

        verts = np.concatenate(
            [
                np.column_stack([x, y_upper]),
                np.column_stack([x[::-1], y_lower[::-1]]),
            ]
        )

        patch = PathPatch(Path(verts), facecolor=color, alpha=alpha, edgecolor="none")
        ax.add_patch(patch)


# ----------------------------------------------------------------------
# Filter student groups of small n sizes
# ----------------------------------------------------------------------


def filter_small_groups(
    df: pd.DataFrame, group_col: str, min_n: int = 12
) -> pd.DataFrame:
    """
    Exclude student groups with fewer than `min_n` students.
    Operates at the chart level (not global filtering).
    """
    group_counts = df[group_col].value_counts(dropna=False)
    valid_groups = group_counts[group_counts >= min_n].index
    return df[df[group_col].isin(valid_groups)].copy()


# =========================================
# CATEGORY DEFINITIONS PER ASSESSMENT
# =========================================

# ---------- NWEA ----------
NWEA_CAT_COL = "achievementquintile"
NWEA_SCORE_COL = "testritscore"
NWEA_TIME_COL_OPTIONS = ["termname", "year", "testwindow"]  # used to build time_label

NWEA_ORDER = [
    "Low",
    "LoAvg",
    "Avg",
    "HiAvg",
    "High",
]

# use raw quintile names as-is for legend
NWEA_COLORS = {cat: default_quintile_colors[i] for i, cat in enumerate(NWEA_ORDER)}

# rollups for insights
NWEA_HIGH_GROUP = ["Avg", "HiAvg", "High"]
NWEA_LOW_GROUP = ["Low"]


# ---------- STAR ----------

# column names in your file
STAR_CAT_COL = "state_benchmark_achievement"
STAR_SCORE_COL = "unified_scale"
STAR_TIME_COL_OPTIONS = ["academicyear", "testwindow"]

# exact category text in the data
STAR_ORDER = [
    "1 - Standard Not Met",
    "2 - Standard Nearly Met",
    "3 - Standard Met",
    "4 - Standard Exceeded",
]

# short labels for legend (leader friendly)
STAR_LEVEL_LABELS = {
    "1 - Standard Not Met": "Standard Not Met",
    "2 - Standard Nearly Met": "Standard Nearly Met",
    "3 - Standard Met": "Standard Met",
    "4 - Standard Exceeded": "Standard Exceeded",
}

# palette mapping low → high using your default_quintile_colors
STAR_COLORS = {
    "1 - Standard Not Met": default_quintile_colors[0],  # darkest gray
    "2 - Standard Nearly Met": default_quintile_colors[1],  # light gray
    "3 - Standard Met": default_quintile_colors[3],  # bright teal
    "4 - Standard Exceeded": default_quintile_colors[4],  # deep teal
}

# group rollups (if you compute insights later)
STAR_HIGH_GROUP = [
    "3 - Standard Met",
    "4 - Standard Exceeded",
]
STAR_LOW_GROUP = ["1 - Standard Not Met"]

# ---------- STAR (District Benchmark) ----------

# exact category text for district_benchmark_achievement
STAR_DISTRICT_ORDER = [
    "1 - Urgent Intervention",
    "2 - Intervention",
    "3 - On Watch",
    "4 - At/Above Benchmark",
]

# rollups for insights
STAR_DISTRICT_LOW_GROUP = ["1 - Urgent Intervention"]
STAR_DISTRICT_MID_GROUP = ["2 - Intervention", "3 - On Watch"]
STAR_DISTRICT_HIGH_GROUP = ["4 - At/Above Benchmark"]
STAR_DISTRICT_HIGHER_GROUP = ["3 - On Watch", "4 - At/Above Benchmark"]

# friendly labels (optional)
STAR_DISTRICT_LEVEL_LABELS = {
    "1 - Urgent Intervention": "Urgent Intervention",
    "2 - Intervention": "Intervention",
    "3 - On Watch": "On Watch",
    "4 - At/Above Benchmark": "At/ or Above Benchmark",
}

# color palette (reuse default_quintile_colors)
STAR_DISTRICT_COLORS = {
    "1 - Urgent Intervention": default_quintile_colors[0],  # darkest gray
    "2 - Intervention": default_quintile_colors[1],  # light gray
    "3 - On Watch": default_quintile_colors[2],  # teal-ish mid
    "4 - At/Above Benchmark": default_quintile_colors[4],  # deep teal
}

# =========================================
# GENERIC PREP LOGIC FOR STACKED % + SCORE
# =========================================


def _build_time_label(df_sub: pd.DataFrame, time_col_options: list[str]) -> pd.Series:
    """
    Build a categorical 'time_label' to use on the x-axis.
    Priority:
      1. termname (if it exists and is not all null)
      2. "<year> <window>" if year + testwindow exists
      3. first available col in time_col_options
    """
    # Case 1: 'termname' present and non-null
    if "termname" in df_sub.columns and df_sub["termname"].notna().any():
        return df_sub["termname"].astype(str)

    # Case 2: year + testwindow pattern
    if "year" in df_sub.columns and "testwindow" in df_sub.columns:
        return (
            df_sub["year"].astype(str).str.strip()
            + " "
            + df_sub["testwindow"].astype(str).str.strip()
        )

    # Fallback: try to combine first two available from time_col_options
    existing = [c for c in time_col_options if c in df_sub.columns]
    if len(existing) >= 2:
        return (
            df_sub[existing[0]].astype(str).str.strip()
            + " "
            + df_sub[existing[1]].astype(str).str.strip()
        )
    elif len(existing) == 1:
        return df_sub[existing[0]].astype(str)

    # Worst case: single constant bucket
    return pd.Series(["Time"] * len(df_sub), index=df_sub.index)


def _prepare_assessment_agg(
    df: pd.DataFrame,
    *,
    subject_str: str,
    window_filter: str,
    subject_col: str,
    window_col: str,
    cat_col: str,
    score_col: str,
    time_col_options: list[str],
    ordered_levels: list[str],
    high_group: list[str],
    low_group: list[str],
):
    """
    Shared engine for:
      - stacked % by category (top panel)
      - mean score (middle panel)
      - insight deltas (bottom panel)

    Returns:
        pct_df: long pct by (time_label, category)
        score_df: mean score per time_label
        insight_metrics: dict with deltas between last 2 windows
        time_order: sorted list of time_label
    """
    d = df.copy()

    # Filter to subject and specific window/season (Fall etc)
    d = d[
        (d[subject_col] == subject_str)
        & (d[window_col].astype(str).str.upper() == window_filter.upper())
    ].copy()

    # Build time_label for x-axis
    d["time_label"] = _build_time_label(d, time_col_options)

    # Count per category
    counts = d.groupby(["time_label", cat_col]).size().rename("n").reset_index()
    totals = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = counts.merge(totals, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]

    # Ensure all levels exist for every time_label (even 0%)
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), ordered_levels], names=["time_label", cat_col]
    )
    pct_df = pct_df.set_index(["time_label", cat_col]).reindex(all_idx).reset_index()
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df["N_total"].ffill().bfill()

    # Chronological order for display
    time_order = sorted(pct_df["time_label"].unique().tolist())
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    pct_df.sort_values(["time_label", cat_col], inplace=True)

    # Mean score per time window
    score_df = (
        d.groupby("time_label")[score_col].mean().rename("avg_score").reset_index()
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    score_df.sort_values("time_label", inplace=True)

    # Insights: compare last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two[0], last_two[1]

        def pct_for(group_list, t):
            return pct_df[
                (pct_df["time_label"] == t) & (pct_df[cat_col].isin(group_list))
            ]["pct"].sum()

        hi_curr = pct_for(high_group, t_curr)
        hi_prev = pct_for(high_group, t_prev)
        lo_curr = pct_for(low_group, t_curr)
        lo_prev = pct_for(low_group, t_prev)

        score_curr = float(
            score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
        )
        score_prev = float(
            score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
        )

        insight_metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": score_curr,
            "score_delta": score_curr - score_prev,
        }
    else:
        insight_metrics = {
            "t_prev": None,
            "t_curr": time_order[-1] if time_order else None,
            "hi_now": None,
            "hi_delta": None,
            "lo_now": None,
            "lo_delta": None,
            "score_now": None,
            "score_delta": None,
        }

    return pct_df, score_df, insight_metrics, time_order


# ---------------------------------------------------------------------
# Local helper for safe school name normalization
# ---------------------------------------------------------------------
def _safe_normalize_school_name(raw_school, cfg_dict):
    """
    Return a cleaned display name using cfg['school_name_map'] when available.
    Accepts dict or list-like configs. Falls back to the original string.
    District (None) falls back to None.
    """
    if raw_school is None:
        return None

    s = str(raw_school)

    raw_map = (cfg_dict or {}).get("school_name_map", {})
    lower_map = {}

    # Normalize to a {lower_from: to} dict if possible
    if isinstance(raw_map, dict):
        lower_map = {str(k).lower(): v for k, v in raw_map.items()}
    elif isinstance(raw_map, (list, tuple)):
        # Support list of 2-tuples/lists or list of small dicts {'from':..., 'to':...}
        for item in raw_map:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                lower_map[str(k).lower()] = v
            elif isinstance(item, dict):
                if "from" in item and "to" in item:
                    lower_map[str(item["from"]).lower()] = item["to"]
                elif "raw" in item and "display" in item:
                    lower_map[str(item["raw"]).lower()] = item["display"]
                elif len(item) == 1:
                    # single-key dict: {'Old Name': 'New Name'}
                    k, v = next(iter(item.items()))
                    lower_map[str(k).lower()] = v
            else:
                # ignore unsupported shapes
                pass
    else:
        # Not a dict or list; ignore
        lower_map = {}

    return lower_map.get(s.lower(), s)


# =========================================
# WRAPPERS FOR EACH ASSESSMENT
# =========================================


def prepare_nwea_agg(df: pd.DataFrame, subject_str: str, window_filter: str = "Fall"):
    return _prepare_assessment_agg(
        df,
        subject_str=subject_str,
        window_filter=window_filter,
        subject_col="subject",  # NWEA "subject"
        window_col="testwindow",  # "Fall" / "Winter" / "Spring"
        cat_col=NWEA_CAT_COL,  # "achievementquintile"
        score_col=NWEA_SCORE_COL,  # "testritscore"
        time_col_options=NWEA_TIME_COL_OPTIONS,
        ordered_levels=NWEA_ORDER,
        high_group=NWEA_HIGH_GROUP,
        low_group=NWEA_LOW_GROUP,
    )


def prepare_iready_agg(df: pd.DataFrame, subject_str: str, window_filter: str = "Fall"):
    return _prepare_assessment_agg(
        df,
        subject_str=subject_str,
        window_filter=window_filter,
        subject_col="subject",  # i-Ready should also have "subject" like "ELA","Math"
        window_col="testwindow",  # usually "Fall","Winter","Spring"
        cat_col=IREADY_CAT_COL,  # "relative_placement"
        score_col=IREADY_SCORE_COL,  # "scale_score"
        time_col_options=IREADY_TIME_COL_OPTIONS,
        ordered_levels=IREADY_ORDER,
        high_group=IREADY_HIGH_GROUP,
        low_group=IREADY_LOW_GROUP,
    )


def prepare_star_agg(df: pd.DataFrame, subject_str: str, window_filter: str = "Fall"):
    return _prepare_assessment_agg(
        df,
        subject_str=subject_str,
        window_filter=window_filter,
        subject_col="subject",  # confirm STAR subject column is named "subject"
        window_col="testwindow",  # STAR seasonal window if present
        cat_col=STAR_CAT_COL,  # "State_benchmark_achievement"
        score_col=STAR_SCORE_COL,  # "unified_scale"
        time_col_options=STAR_TIME_COL_OPTIONS,
        ordered_levels=STAR_ORDER,
        high_group=STAR_HIGH_GROUP,
        low_group=STAR_LOW_GROUP,
    )


# %%
