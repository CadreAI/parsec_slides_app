"""
STAR Winter chart generation script - generates charts from ingested STAR data for Winter window
Based on star_moy.py structure but specifically for Winter filtering
"""

# Set matplotlib backend to non-interactive before any imports
import matplotlib
import matplotlib.transforms as mtransforms

matplotlib.use('Agg')

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

from .star_chart_utils import (
    FIGSIZE_WIDTH,
    LABEL_MIN_PCT,
    PADDING,
    calculate_bar_width,
    draw_insight_card,
    draw_score_bar,
    draw_stacked_bar,
)

# Import utility modules
from .star_data import (
    _short_year,
    filter_star_subject_rows,
    get_scopes,
    load_config_from_args,
    load_star_data,
    prep_star_for_charts,
)
from .star_filters import (
    apply_chart_filters,
    should_generate_grade,
    should_generate_student_group,
    should_generate_subject,
)

# Chart tracking for CSV generation
chart_links = []
_chart_tracking_set = set()

def _requested_star_subjects(chart_filters):
    """
    Return ordered STAR subject labels based on chart_filters.subjects.
    Supports: Reading, Reading (Spanish), Mathematics.
    """
    subjects_filter = (chart_filters or {}).get("subjects") or []
    if not isinstance(subjects_filter, list) or len(subjects_filter) == 0:
        return ["Reading", "Mathematics"]
    norm = [str(s).strip().lower() for s in subjects_filter if s is not None]
    out = []
    if any(("reading" in s and "spanish" in s) or ("spanish reading" in s) for s in norm):
        out.append("Reading (Spanish)")
    if any(("reading" in s) or (s == "ela") for s in norm):
        out.append("Reading")
    if any(("math" in s) for s in norm):
        out.append("Mathematics")
    seen = set()
    out2 = []
    for s in out:
        if s not in seen:
            seen.add(s)
            out2.append(s)
    return out2 or ["Reading", "Mathematics"]

def track_chart(chart_name, file_path, scope="district", section=None, chart_data=None):
    """Track chart for CSV generation and save chart data if provided"""
    global _chart_tracking_set
    
    chart_path = Path(file_path)
    normalized_path = str(chart_path.resolve())
    
    if normalized_path in _chart_tracking_set:
        print(f"  ⚠ Skipping duplicate chart: {chart_name}")
        return
    
    _chart_tracking_set.add(normalized_path)
    
    chart_info = {
        "chart_name": chart_name,
        "scope": scope,
        "section": section,
        "file_path": str(file_path),
        "file_link": f"file://{chart_path.absolute()}"
    }
    
    if chart_data is not None:
        data_path = chart_path.parent / f"{chart_path.stem}_data.json"
        try:
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif pd.isna(obj):
                    return None
                return obj
            
            serializable_data = convert_to_json_serializable(chart_data)
            with open(data_path, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            chart_info["data_path"] = str(data_path)
            print(f"  Chart data saved to: {data_path}")
        except Exception as e:
            print(f"  Warning: Failed to save chart data: {e}")
    
    chart_links.append(chart_info)

# ---------------------------------------------------------------------
# SECTION 0 — Winter Predicted vs Actual CAASPP (Spring)
# ---------------------------------------------------------------------

def _plot_section0_star_single_subject_winter(scope_label, folder, subject, proj_pct, act_pct, metrics, output_dir, preview=False):
    """Render Section 0 chart for single subject: STAR predicted vs actual CAASPP - Winter version"""
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35)
    
    # Calculate bar width for 2 side-by-side bars
    n_bars = 2
    bar_width = calculate_bar_width(n_bars, fig_width)
    
    title = "Reading" if subject == "Reading" else "Math"
    
    # Legend
    handles = [
        Patch(facecolor=hf.CERS_LEVEL_COLORS[l], edgecolor="none", label=l)
        for l in metrics["act_order"]
    ]
    fig.legend(
        handles=handles,
        labels=metrics["act_order"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(metrics["act_order"]),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    
    # Create axes
    bar_ax = fig.add_subplot(gs[0, 0])
    txt_ax = fig.add_subplot(gs[1, 0])
    insight_ax = fig.add_subplot(gs[2, 0])
    
    # Plot predicted vs actual bars
    cumulative = 0
    for level in metrics["proj_order"]:
        val = float(proj_pct.get(level, 0))
        idx = metrics["proj_order"].index(level)
        mapped_level = (
            metrics["act_order"][idx]
            if idx < len(metrics["act_order"])
            else metrics["act_order"][-1]
        )
        col = hf.CERS_LEVEL_COLORS.get(mapped_level, "#cccccc")
        bars = bar_ax.bar(
            -0.2,
            val,
            bottom=cumulative,
            width=bar_width,
            color=col,
            alpha=0.6,
            edgecolor="#434343",
            linewidth=1.2,
            linestyle="--",
        )
        rect = bars.patches[0]
        if val >= 5:
            bar_ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                cumulative + val / 2.0,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#434343",
            )
        cumulative += val
    
    cumulative = 0
    for level in metrics["act_order"]:
        val = float(act_pct.get(level, 0))
        col = hf.CERS_LEVEL_COLORS.get(level, "#cccccc")
        bars = bar_ax.bar(
            0.2, val, bottom=cumulative, width=bar_width, color=col,
            edgecolor="#434343", linewidth=1.2
        )
        rect = bars.patches[0]
        if val >= 5:
            bar_ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                cumulative + val / 2.0,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        cumulative += val
    
    bar_ax.set_ylim(0, 100)
    bar_ax.set_ylabel("% of Students")
    bar_ax.set_xticks([])
    bar_ax.set_xticklabels(["Predicted", "Actual"])
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.spines["bottom"].set_visible(False)
    bar_ax.set_title(title, fontsize=14, fontweight="bold", y=1.05)
    
    # Text panel
    txt_ax.axis("off")
    txt_ax.text(
        0.5, 0.5,
        r"Predicted vs Actual Met/Exceed:"
        + f"\n{metrics['proj_met']:.1f}% vs {metrics['act_met']:.1f}% "
        + f"({metrics['delta']:+.1f} ppts)",
        ha="center", va="center", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"),
    )
    
    # Insight panel
    insight_ax.axis("off")
    insight_txt = (
        f"STAR Winter {metrics['year']}-{metrics['year'] + 1} predicted "
        f"{metrics['proj_met']:.1f}% meeting/exceeding on CAASPP. Actual Spring result: "
        f"{metrics['act_met']:.1f}%."
    )
    insight_ax.text(
        0.5, 0.5, insight_txt, ha="center", va="center", fontsize=11,
        wrap=True, color="#434343",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"),
    )
    
    fig.suptitle(
        f"{scope_label} • {title} • Winter Prediction vs Spring Actual CAASPP",
        fontsize=20, fontweight="bold", y=1.00
    )
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    activity_type = 'reading' if subject == 'Reading' else 'math'
    out_name = f"{scope_label.replace(' ', '_')}_STAR_section0_winter_{activity_type}_predicted_vs_actual.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    chart_data = {
        "subject": subject,
        "predicted_vs_actual": {
            "predicted_met_exceed": metrics["proj_met"],
            "actual_met_exceed": metrics["act_met"],
            "delta": metrics["delta"],
        },
        "year": metrics["year"],
    }
    track_chart(f"Section 0: Predicted vs Actual {title} (Winter)", out_path, scope=scope_label, section=0, chart_data=chart_data)
    print(f"Saved Section 0 ({title}): {out_path}")
    
    return str(out_path)


def _prep_section0_star_winter(df, subject):
    """Prepare data for Section 0: STAR predicted vs actual CAASPP - Winter version"""
    d = df.copy()
    d = d[d["testwindow"].str.upper() == "WINTER"].copy()
    
    if d.empty or d["academicyear"].dropna().empty:
        return None, None, None, None
    
    d = filter_star_subject_rows(d, subject)
    
    if d.empty or d["academicyear"].dropna().empty:
        return None, None, None, None
    
    d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
    if d["academicyear"].dropna().empty:
        return None, None, None, None
    
    # Target year is the latest Winter test year present (no offset)
    target_year = int(d["academicyear"].max() - 1)
    
    # Keep only the latest Winter year slice
    d = d[d["academicyear"] == target_year].copy()
    
    if "activity_completed_date" in d.columns:
        d["activity_completed_date"] = pd.to_datetime(
            d["activity_completed_date"], errors="coerce"
        )
        d = d.sort_values("activity_completed_date").drop_duplicates(
            "student_state_id", keep="last"
        )
    
    d = d.dropna(subset=["state_benchmark_achievement", "cers_overall_performanceband"])
    if d.empty:
        return None, None, None, target_year
    
    proj_order = sorted(d["state_benchmark_achievement"].unique())
    act_order = hf.CERS_LEVELS
    
    def pct_table(col, order):
        return (
            d.groupby(col)
            .size()
            .reindex(order, fill_value=0)
            .pipe(lambda s: 100 * s / s.sum())
        )
    
    proj_pct = pct_table("state_benchmark_achievement", proj_order)
    act_pct = pct_table("cers_overall_performanceband", act_order)
    
    def pct_met_exceed(series, met_levels):
        return 100 * d[d[series].isin(met_levels)].shape[0] / d.shape[0]
    
    proj_met = pct_met_exceed(
        "state_benchmark_achievement", ["3 - Standard Met", "4 - Standard Exceeded"]
    )
    act_met = pct_met_exceed(
        "cers_overall_performanceband",
        ["Level 3 - Standard Met", "Level 4 - Standard Exceeded"],
    )
    
    delta = proj_met - act_met
    
    metrics = {
        "proj_met": proj_met,
        "act_met": act_met,
        "delta": delta,
        "year": target_year,
        "proj_order": proj_order,
        "act_order": act_order,
        "proj_pct": proj_pct,
        "act_pct": act_pct,
    }
    
    return proj_pct, act_pct, metrics, target_year

def _plot_section0_star_winter(scope_label, folder, subj_payload, output_dir, preview=False):
    """Render Section 0 chart: STAR predicted vs actual CAASPP - Winter version"""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Calculate bar width for 2 side-by-side bars per column
    n_bars = 2
    bar_width = calculate_bar_width(n_bars, 16 / 2)  # Half width per column
    
    subjects = [s for s in ["Reading", "Mathematics"] if s in subj_payload]
    titles = {"Reading": "Reading", "Mathematics": "Math"}
    
    first_metrics = next(iter(subj_payload.values()))["metrics"]
    handles = [
        Patch(facecolor=hf.CERS_LEVEL_COLORS[l], edgecolor="none", label=l)
        for l in first_metrics["act_order"]
    ]
    fig.legend(
        handles=handles,
        labels=first_metrics["act_order"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(first_metrics["act_order"]),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    
    for i, subject in enumerate(subjects):
        proj_pct = subj_payload[subject]["proj_pct"]
        act_pct = subj_payload[subject]["act_pct"]
        metrics = subj_payload[subject]["metrics"]
        
        bar_ax = fig.add_subplot(gs[0, i])
        cumulative = 0
        for level in metrics["proj_order"]:
            val = float(proj_pct.get(level, 0))
            idx = metrics["proj_order"].index(level)
            mapped_level = (
                metrics["act_order"][idx]
                if idx < len(metrics["act_order"])
                else metrics["act_order"][-1]
            )
            col = hf.CERS_LEVEL_COLORS.get(mapped_level, "#cccccc")
            bars = bar_ax.bar(
                -0.2,
                val,
                bottom=cumulative,
                width=bar_width,
                color=col,
                alpha=0.6,
                edgecolor="#434343",
                linewidth=1.2,
                linestyle="--",
            )
            rect = bars.patches[0]
            if val >= 5:
                bar_ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    cumulative + val / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#434343",
                )
            cumulative += val
        
        cumulative = 0
        for level in metrics["act_order"]:
            val = float(act_pct.get(level, 0))
            col = hf.CERS_LEVEL_COLORS.get(level, "#cccccc")
            bars = bar_ax.bar(
                0.2,
                val,
                bottom=cumulative,
                width=bar_width,
                color=col,
                edgecolor="white",
                linewidth=1.2,
            )
            rect = bars.patches[0]
            if val >= 5:
                txt_color = "#434343" if "Nearly" in level else "white"
                bar_ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    cumulative + val / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=txt_color,
                )
            cumulative += val
        
        bar_ax.set_xticks([-0.2, 0.2])
        bar_ax.set_xticklabels(["Predicted", "Actual"])
        bar_ax.set_ylim(0, 100)
        bar_ax.set_ylabel("% of Students")
        bar_ax.set_title(titles[subject], fontsize=14, fontweight="bold", pad=30)
        bar_ax.set_axisbelow(True)
        # bar_ax.grid(False)  # Gridlines disabled globally
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        
        pct_ax = fig.add_subplot(gs[1, i])
        pct_ax.bar(
            "Pred Met/Exc",
            metrics["proj_met"],
            color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            alpha=0.6,
            edgecolor="#434343",
            linewidth=1.2,
            linestyle="--",
        )
        pct_ax.bar(
            "Actual Met/Exc",
            metrics["act_met"],
            color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            alpha=1.0,
            edgecolor=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
            linewidth=1.2,
        )
        for x, v in zip([0, 1], [metrics["proj_met"], metrics["act_met"]]):
            pct_ax.text(
                x,
                v + 1,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#434343",
            )
        pct_ax.set_ylim(0, 100)
        pct_ax.set_ylabel("% Met/Exc")
        pct_ax.set_axisbelow(True)
        # pct_ax.grid(False)  # Gridlines disabled globally
        pct_ax.spines["top"].set_visible(False)
        pct_ax.spines["right"].set_visible(False)
        
        ax3 = fig.add_subplot(gs[2, i])
        ax3.axis("off")
        pred = float(metrics["proj_met"])
        act = float(metrics["act_met"])
        delta = pred - act
        insight_text = (
            r"Predicted vs Actual Met/Exceed:"
            + "\n"
            + rf"${pred:.1f}\% - {act:.1f}\% = \mathbf{{{delta:+.1f}}}$ pts"
        )
        ax3.text(
            0.5,
            0.5,
            insight_text,
            fontsize=12,
            ha="center",
            va="center",
            color="#434343",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#f5f5f5",
                edgecolor="#ccc",
                linewidth=1.0,
            ),
        )
    
    fig.suptitle(
        f"{scope_label} • Winter {first_metrics['year']} Prediction Accuracy",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{scope_label}_STAR_section0_pred_vs_actual_winter.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": "Winter",
        "subjects": list(subj_payload.keys()),
        "predicted_vs_actual": {
            subj: {
                "predicted_pct": {level: float(proj_pct.get(level, 0)) for level in subj_payload[subj]["metrics"]["proj_order"]},
                "actual_pct": {level: float(act_pct.get(level, 0)) for level in subj_payload[subj]["metrics"]["act_order"]},
                "predicted_met_exceed": float(subj_payload[subj]["metrics"]["proj_met"]),
                "actual_met_exceed": float(subj_payload[subj]["metrics"]["act_met"]),
                "delta": float(subj_payload[subj]["metrics"]["delta"])
            }
            for subj in subj_payload.keys()
            for proj_pct, act_pct in [(subj_payload[subj]["proj_pct"], subj_payload[subj]["act_pct"])]
        }
    }
    track_chart(f"Section 0: Predicted vs Actual (Winter)", out_path, scope=scope_label, section=0, chart_data=chart_data)
    print(f"Saved Section 0 (Winter): {out_path}")
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 1 - Winter Performance Trends (Single Subject Dashboard)
# ---------------------------------------------------------------------

def plot_star_single_subject_dashboard_winter(
    df, scope_label, folder, output_dir, subject_str, window_filter="Winter", preview=False
):
    """Single-subject dashboard for either Math or Reading - Winter version"""
    
    # Determine subject and title
    if subject_str.lower() in ['math', 'mathematics']:
        activity_type_filter = 'math'
        title = 'Math'
    else:
        activity_type_filter = 'reading'
        title = 'Reading'
    
    # Single-column layout
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3)
    
    legend_handles = [
        Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q)
        for q in hf.STAR_ORDER
    ]
    
    # Prepare data for this subject
    pct_df, score_df, metrics, time_order = prep_star_for_charts(
        df, subject_str=activity_type_filter, window_filter=window_filter
    )
    
    # Limit to most recent 4 timepoints
    if len(time_order) > 4:
        time_order = time_order[-4:]
        pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
        score_df = score_df[score_df["time_label"].isin(time_order)].copy()
    
    # Build n_map for x-axis labels
    if pct_df is not None and not pct_df.empty:
        n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
        n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
    else:
        n_map = {}
    
    # Calculate dynamic bar width
    n_bars = len(time_order)
    bar_width = calculate_bar_width(n_bars, fig_width)
    
    # Plot panels
    ax1 = fig.add_subplot(gs[0, 0])
    if pct_df is not None and not pct_df.empty:
        draw_stacked_bar(ax1, pct_df, score_df, hf.STAR_ORDER, bar_width=bar_width, fig_width=fig_width)
    else:
        ax1.text(0.5, 0.5, f"No {title} data", ha="center", va="center", fontsize=12)
        ax1.axis("off")
    ax1.set_title(f"{title}", fontsize=14, fontweight="bold", pad=30)
    
    ax2 = fig.add_subplot(gs[1, 0])
    if score_df is not None and not score_df.empty:
        draw_score_bar(ax2, score_df, hf.STAR_ORDER, n_map, bar_width=bar_width, fig_width=fig_width)
    else:
        ax2.text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
        ax2.axis("off")
    ax2.set_title("Avg Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
    
    ax3 = fig.add_subplot(gs[2, 0])
    draw_insight_card(ax3, metrics, title)
    
    fig.legend(
        handles=legend_handles,
        labels=hf.STAR_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(hf.STAR_ORDER),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    
    fig.suptitle(
        f"{scope_label} • {window_filter} Year-to-Year Trends\n{title}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    # Include subject in filename so they don't overwrite each other
    out_name = f"{scope_label}_STAR_section1_{window_filter.lower()}_{activity_type_filter}_trends.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    chart_data = {
        "metrics": metrics,
        "time_order": time_order
    }
    track_chart(f"Section 1: {window_filter} {title} Trends", out_path, 
                scope=scope_label, section=1, chart_data=chart_data)
    
    print(f"Saved Section 1 ({window_filter} {title}): {out_path}")
    return str(out_path)


# # COMMENTED OUT: Dual-subject version (kept for reference)
# def plot_star_dual_subject_dashboard_winter(
#     df, scope_label, folder, output_dir, window_filter="Winter", preview=False
# ):
#     """Faceted dashboard showing both Math and Reading for a given scope - Winter version"""
#     fig = plt.figure(figsize=(16, 9), dpi=300)
#     gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
#     fig.subplots_adjust(hspace=0.3, wspace=0.25)
#     
#     subjects = ["reading", "math"]
#     titles = ["Reading", "Math"]
#     
#     legend_handles = [
#         Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q)
#         for q in hf.STAR_ORDER
#     ]
#     
#     pct_dfs, score_dfs, metrics_list, time_orders, n_maps = [], [], [], [], []
#     
#     for i, (activity_type_filter, title) in enumerate(zip(subjects, titles)):
#         pct_df, score_df, metrics, time_order = prep_star_for_charts(
#             df, subject_str=activity_type_filter, window_filter=window_filter
#         )
#         
#         # Limit to most recent 4 timepoints
#         if len(time_order) > 4:
#             time_order = time_order[-4:]
#             pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
#             score_df = score_df[score_df["time_label"].isin(time_order)].copy()
#         
#         pct_dfs.append(pct_df)
#         score_dfs.append(score_df)
#         metrics_list.append(metrics)
#         time_orders.append(time_order)
#         
#         # Build n_map for x-axis labels
#         if pct_df is not None and not pct_df.empty:
#             n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
#             n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
#         else:
#             n_map = {}
#         n_maps.append(n_map)
#     
#     # Plot panels
#     for i, (pct_df, score_df, metrics, time_order, n_map) in enumerate(
#         zip(pct_dfs, score_dfs, metrics_list, time_orders, n_maps)
#     ):
#         ax1 = fig.add_subplot(gs[0, i])
#         if pct_df is not None and not pct_df.empty:
#             draw_stacked_bar(ax1, pct_df, score_df, hf.STAR_ORDER)
#         else:
#             ax1.text(0.5, 0.5, f"No {titles[i]} data", ha="center", va="center", fontsize=12)
#             ax1.axis("off")
#         ax1.set_title(f"{titles[i]}", fontsize=14, fontweight="bold", pad=30)
#         
#         ax2 = fig.add_subplot(gs[1, i])
#         if score_df is not None and not score_df.empty:
#             draw_score_bar(ax2, score_df, hf.STAR_ORDER, n_map)
#         else:
#             ax2.text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
#             ax2.axis("off")
#         ax2.set_title("Avg Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
#         
#         ax3 = fig.add_subplot(gs[2, i])
#         draw_insight_card(ax3, metrics, titles[i])
#     
#     fig.legend(
#         handles=legend_handles,
#         labels=hf.STAR_ORDER,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.93),
#         ncol=len(hf.STAR_ORDER),
#         frameon=False,
#         fontsize=10,
#         handlelength=1.8,
#         handletextpad=0.5,
#         columnspacing=1.1,
#     )
#     
#     fig.suptitle(
#         f"{scope_label} • {window_filter} Year-to-Year Trends",
#         fontsize=20,
#         fontweight="bold",
#         y=1,
#     )
#     
#     out_dir_path = Path(output_dir) / folder
#     out_dir_path.mkdir(parents=True, exist_ok=True)
#     out_name = f"{scope_label}_STAR_section1_{window_filter.lower()}_trends.png"
#     out_path = out_dir_path / out_name
#     hf._save_and_render(fig, out_path, dev_mode=preview)
#     
#     chart_data = {
#         "metrics": {titles[i]: metrics_list[i] for i in range(len(titles))},
#         "time_orders": {titles[i]: time_orders[i] for i in range(len(titles))}
#     }
#     track_chart(f"Section 1: {window_filter} Trends", out_path, scope=scope_label, section=1, chart_data=chart_data)
#     print(f"Saved Section 1 (Winter): {out_path}")
#     return str(out_path)

# ---------------------------------------------------------------------
# SECTION 1.1 — Fall → Winter Performance Progression (Reading + Math)
# ---------------------------------------------------------------------

def _prep_star_fall_winter(df, subj):
    """Filter to Fall/Winter 2026 for one subject. Deduplicate to latest test per student per window."""
    d = df.copy()
    
    # Get most recent academic year with Fall/Winter data
    d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
    d_window = d[d["testwindow"].astype(str).str.upper().isin(["FALL", "WINTER"])].copy()
    if d_window.empty or d_window["academicyear"].dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    target_year = int(d_window["academicyear"].max())
    # restrict to most recent academic year and Fall/Winter
    d = d[d["academicyear"] == target_year].copy()
    d = d[d["testwindow"].astype(str).str.upper().isin(["FALL", "WINTER"])].copy()
    
    # subject filtering (includes Spanish Reading preference)
    d = filter_star_subject_rows(d, subj)
    
    # valid benchmark levels only
    d = d[d["state_benchmark_achievement"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # build Fall/Winter labels (always 25–26)
    d["time_label"] = d["testwindow"].str.title() + " 25-26"
    
    # dedupe to latest attempt
    if "activity_completed_date" in d.columns:
        d["activity_completed_date"] = pd.to_datetime(d["activity_completed_date"], errors="coerce")
    else:
        d["activity_completed_date"] = pd.NaT
    
    d.sort_values(["student_state_id", "time_label", "activity_completed_date"], inplace=True)
    d = d.groupby(["student_state_id", "time_label"], as_index=False).tail(1)
    
    # percent by quintile
    quint = d.groupby(["time_label", "state_benchmark_achievement"]).size().rename("n").reset_index()
    totals = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = quint.merge(totals, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # ensure full STAR_ORDER exists
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), hf.STAR_ORDER],
        names=["time_label", "state_benchmark_achievement"],
    )
    pct_df = (
        pct_df.set_index(["time_label", "state_benchmark_achievement"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(lambda s: s.ffill().bfill())
    
    # avg unified scale score
    score_df = d.groupby("time_label")["unified_scale"].mean().rename("avg_score").reset_index()
    
    # ensure order: Fall, Winter
    time_order = ["Fall 25-26", "Winter 25-26"]
    pct_df["time_label"] = pd.Categorical(pct_df["time_label"], time_order, True)
    score_df["time_label"] = pd.Categorical(score_df["time_label"], time_order, True)
    pct_df.sort_values(["time_label", "state_benchmark_achievement"], inplace=True)
    score_df.sort_values("time_label", inplace=True)
    
    # insights: Fall → Winter
    if set(time_order).issubset(set(pct_df["time_label"].astype(str))):
        t_prev, t_curr = time_order
        def pct_for(buckets):
            return (
                pct_df[(pct_df["time_label"] == t_curr) & (pct_df["state_benchmark_achievement"].isin(buckets))]["pct"].sum()
                - pct_df[(pct_df["time_label"] == t_prev) & (pct_df["state_benchmark_achievement"].isin(buckets))]["pct"].sum()
            )
        delta_exceed = pct_for(["4 - Standard Exceeded"])
        delta_meet_exceed = pct_for(["4 - Standard Exceeded", "3 - Standard Met"])
        delta_not_met = pct_for(["1 - Standard Not Met"])
        score_curr = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0])
        score_prev = float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])
        metrics = {
            "pct_df": pct_df,
            "t_prev": t_prev,
            "t_curr": t_curr,
            "delta_exceed": delta_exceed,
            "delta_meet_exceed": delta_meet_exceed,
            "delta_not_met": delta_not_met,
            "delta_score": score_curr - score_prev,
        }
    else:
        metrics = {"pct_df": pct_df}
    
    return pct_df, score_df, metrics, time_order

def plot_section_1_1_single_subject(df, scope_label, folder, output_dir, subject_str, school_raw=None, preview=False):
    """Plot Section 1.1: Fall → Winter Performance Progression - Single Subject"""
    
    # Determine subject
    if subject_str.lower() in ['math', 'mathematics']:
        subj = 'math'
        title = 'Math'
    else:
        subj = 'reading'
        title = 'Reading'
    
    # Single-column layout
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3)
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    pct_df, score_df, metrics, time_order = _prep_star_fall_winter(df, subj)
    
    if pct_df.empty or score_df.empty or "time_label" not in pct_df.columns:
        print(f"[Section 1.1] No {title} data for {scope_label}")
        plt.close(fig)
        return None
    
    # Panel 1 — 100% stacked bars
    ax = fig.add_subplot(gs[0, 0])
    stack_df = (
        pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
        .reindex(columns=hf.STAR_ORDER)
        .fillna(0)
    )
    x = np.arange(len(stack_df))
    
    # Dynamic bar width for consistent physical appearance
    n_bars = len(stack_df)
    fig_width = FIGSIZE_WIDTH // 2  # Single column width
    bar_width = calculate_bar_width(n_bars, fig_width)
    padding = PADDING
    
    cumulative = np.zeros(len(stack_df))
    for cat in hf.STAR_ORDER:
        vals = stack_df[cat].to_numpy()
        bars = ax.bar(x, vals, width=bar_width, bottom=cumulative, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
        for j, rect in enumerate(bars):
            h = vals[j]
            if h >= 3:
                label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
                ax.text(rect.get_x() + rect.get_width() / 2, cumulative[j] + h / 2, f"{h:.1f}%",
                       ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
        cumulative += vals
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-padding, n_bars - 1 + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(stack_df.index.tolist())
    ax.set_ylabel("% of Students")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Panel 2 — Avg score bars
    ax2 = fig.add_subplot(gs[1, 0])
    x2 = np.arange(len(score_df))
    vals = score_df["avg_score"].to_numpy()
    bars = ax2.bar(x2, vals, width=bar_width, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
    for rect, v in zip(bars, vals):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="#434343")
    n_map = pct_df.groupby("time_label")["N_total"].max().to_dict()
    labels = [f"{tl}\n(n = {int(n_map.get(tl, 0))})" if n_map.get(tl) else tl 
             for tl in score_df["time_label"].astype(str).tolist()]
    ax2.set_xlim(-padding, n_bars - 1 + padding)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Avg Unified Scale Score")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # Panel 3 — Insights
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if metrics.get("t_prev"):
        t_curr = metrics.get("t_curr", "Current")
        pct_df_metrics = metrics.get("pct_df")
        if pct_df_metrics is not None and not pct_df_metrics.empty:
            def _bucket_pct(bucket, tlabel):
                return pct_df_metrics.loc[
                    (pct_df_metrics["time_label"] == tlabel) &
                    (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
            
            high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
            hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
            lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
            score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if not score_df.empty and len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
            
            delta_exceed = metrics.get("delta_exceed", 0)
            delta_meet_exceed = metrics.get("delta_meet_exceed", 0)
            delta_not_met = metrics.get("delta_not_met", 0)
            
            lines = [
                "Change",
                f"Met+Exceeded: {delta_meet_exceed:+.1f}%",
                f"Exceeded: {delta_exceed:+.1f}%",
                f"Not Met: {delta_not_met:+.1f}%",
            ]
        else:
            lines = ["Not enough data for insights"]
    else:
        lines = ["Not enough data for insights"]
    ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
            usetex=False, color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
    
    fig.suptitle(f"{scope_label} • Fall → Winter Performance Progression • {title}",
                fontsize=18, fontweight="bold", y=1)
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    activity_type = 'math' if subj == 'math' else 'reading'
    out_name = f"{safe_scope}_STAR_section1_1_fall_winter_{activity_type}_progression.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    m = dict(metrics) if isinstance(metrics, dict) else {}
    m.pop("pct_df", None)
    chart_data = {
        "chart_type": "star_winter_section1_1_fall_winter_progression_single_subject",
        "section": 1.1,
        "scope": scope_label,
        "window_filter": "Fall/Winter",
        "subject": title,
        "pct_data": pct_df.to_dict("records") if not pct_df.empty else [],
        "score_data": score_df.to_dict("records") if not score_df.empty else [],
        "metrics": m,
        "time_orders": [str(t) for t in time_order],
    }
    track_chart(f"Section 1.1: Fall → Winter {title}", out_path, scope=scope_label, section=1.1, chart_data=chart_data)
    print(f"Saved Section 1.1 ({title}): {out_path}")
    return str(out_path)


# COMMENTED OUT: Dual-subject version (kept for reference)
# def plot_section_1_1(df, scope_label, folder, output_dir, school_raw=None, preview=False):
#     """Plot Section 1.1: Fall → Winter Performance Progression"""
#     fig = plt.figure(figsize=(16, 9), dpi=300)
#     gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
#     fig.subplots_adjust(hspace=0.3, wspace=0.25)
#     
#     subjects = ["reading", "math"]
#     titles = ["Reading", "Math"]
    # # Sidecar JSON payloads for chart_analyzer.py
    # json_subjects: list[str] = []
    # pct_payload: list[dict] = []
    # score_payload: list[dict] = []
    # metrics_payload: list[dict] = []
    # time_orders: list[str] = []
    # axes = [
    #     [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
    #     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    #     [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    # ]
    
    # legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    # for i, subj in enumerate(subjects):
    #     pct_df, score_df, metrics, time_order = _prep_star_fall_winter(df, subj)
        
    #     if pct_df.empty or score_df.empty or "time_label" not in pct_df.columns:
    #         for ax in (axes[0][i], axes[1][i], axes[2][i]):
    #             ax.axis("off")
    #         axes[1][i].text(0.5, 0.5, f"No {titles[i]} data", transform=axes[1][i].transAxes,
    #                        ha="center", va="center", fontsize=12, fontweight="bold", color="#999999")
    #         continue

    #     # Prepare sidecar payloads (only for subjects that have data)
    #     json_subjects.append(titles[i])
    #     pct_payload.append({"subject": titles[i], "data": pct_df.to_dict("records")})
    #     score_payload.append({"subject": titles[i], "data": score_df.to_dict("records")})
    #     if isinstance(metrics, dict):
    #         m = dict(metrics)
    #         # Drop embedded df to keep JSON smaller; we already include pct_data/score_data.
    #         m.pop("pct_df", None)
    #     else:
    #         m = {}
    #     metrics_payload.append(m)
    #     if not time_orders and isinstance(time_order, list):
    #         time_orders = [str(t) for t in time_order]
        
    #     # Panel 1 — 100% stacked bars
    #     ax = axes[0][i]
    #     stack_df = (
    #         pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
    #         .reindex(columns=hf.STAR_ORDER)
    #         .fillna(0)
    #     )
    #     x = np.arange(len(stack_df))
    #     cumulative = np.zeros(len(stack_df))
    #     for cat in hf.STAR_ORDER:
    #         vals = stack_df[cat].to_numpy()
    #         bars = ax.bar(x, vals, bottom=cumulative, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
    #         for j, rect in enumerate(bars):
    #             h = vals[j]
    #             if h >= 3:
    #                 label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
    #                 ax.text(rect.get_x() + rect.get_width() / 2, cumulative[j] + h / 2, f"{h:.1f}%",
    #                        ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
    #         cumulative += vals
    #     ax.set_ylim(0, 100)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(stack_df.index.tolist())
    #     ax.set_ylabel("% of Students")
    #     # ax.grid(False)  # Gridlines disabled globally
    #     ax.set_title(titles[i], fontsize=14, fontweight="bold", pad=30)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
        
    #     # Panel 2 — Avg score bars
    #     ax2 = axes[1][i]
    #     x2 = np.arange(len(score_df))
    #     vals = score_df["avg_score"].to_numpy()
    #     bars = ax2.bar(x2, vals, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
    #     for rect, v in zip(bars, vals):
    #         ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
    #                 ha="center", va="bottom", fontsize=14, fontweight="bold", color="#434343")
    #     n_map = pct_df.groupby("time_label")["N_total"].max().to_dict()
    #     labels = [f"{tl}\n(n = {int(n_map.get(tl, 0))})" if n_map.get(tl) else tl 
    #              for tl in score_df["time_label"].astype(str).tolist()]
    #     ax2.set_xticks(x2)
    #     ax2.set_xticklabels(labels)
    #     ax2.set_ylabel("Avg Unified Scale Score")
    #     # ax2.grid(False)  # Gridlines disabled globally
    #     ax2.spines["top"].set_visible(False)
    #     ax2.spines["right"].set_visible(False)
        
    #     # Panel 3 — Insights
    #     ax3 = axes[2][i]
    #     ax3.axis("off")
    #     if metrics.get("t_prev"):
    #         t_curr = metrics.get("t_curr", "Current")
    #         pct_df_metrics = metrics.get("pct_df")
    #         # Show current values, not deltas (deltas still calculated in metrics)
    #         if pct_df_metrics is not None and not pct_df_metrics.empty:
    #             def _bucket_pct(bucket, tlabel):
    #                 return pct_df_metrics.loc[
    #                     (pct_df_metrics["time_label"] == tlabel) &
    #                     (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
    #                 ].sum()
                
    #             high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
    #             hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
    #             lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
    #             score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if not score_df.empty and len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
                
    #             lines = [
    #                 f"Current values ({t_curr}):",
    #                 f"Exceeded: {high_now:.1f} ppts",
    #                 f"Meet/Exceed: {hi_now:.1f} ppts",
    #                 f"Not Met: {lo_now:.1f} ppts",
    #                 f"Avg Score: {score_now:.1f} pts",
    #             ]
    #         else:
    #             lines = ["Not enough data for insights"]
    #     else:
    #         lines = ["Not enough data for insights"]
    #     ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
    #             usetex=False, color="#333333",
    #             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    # fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
    #           bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
    
    # fig.suptitle(f"{scope_label} • Fall → Winter Performance Progression",
    #             fontsize=20, fontweight="bold", y=1)
    
    # out_dir_path = Path(output_dir) / folder
    # out_dir_path.mkdir(parents=True, exist_ok=True)
    # safe_scope = scope_label.replace(" ", "_")
    # out_name = f"{safe_scope}_STAR_section1_1_fall_winter_progression.png"
    # out_path = out_dir_path / out_name
    # hf._save_and_render(fig, out_path, dev_mode=preview)
    
    # chart_data = {
    #     "chart_type": "star_winter_section1_1_fall_winter_progression",
    #     "section": 1.1,
    #     "scope": scope_label,
    #     "window_filter": "Fall/Winter",
    #     "subjects": json_subjects or titles,
    #     "pct_data": pct_payload,
    #     "score_data": score_payload,
    #     "metrics": metrics_payload,
    #     "time_orders": time_orders,
    # }
    # track_chart(f"Section 1.1: Fall → Winter Progression", out_path, scope=scope_label, section=1.1, chart_data=chart_data)
    # print(f"Saved Section 1.1: {out_path}")
    # return str(out_path)

# ---------------------------------------------------------------------
# SECTION 1.2 — Fall → Winter Performance Progression by Grade
# ---------------------------------------------------------------------


def plot_section_1_2_for_grade_single_subject(df, scope_label, folder, output_dir, grade, subject_str, school_raw=None, preview=False):
    """Plot Section 1.2 for a single grade and subject - Fall → Winter Progression"""
    
    # Determine subject
    if subject_str.lower() in ['math', 'mathematics']:
        subj = 'math'
        title = 'Math'
    else:
        subj = 'reading'
        title = 'Reading'
    
    # Single-column layout
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3)
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    pct_df, score_df, metrics, time_order = _prep_star_fall_winter(df, subj)
    
    if pct_df.empty or score_df.empty:
        print(f"Grade {grade} ({title}): No Fall/Winter data — skipping chart.")
        plt.close(fig)
        return None
    
    # Panel 1 — 100% stacked bars
    ax = fig.add_subplot(gs[0, 0])
    stack_df = (
        pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
        .reindex(columns=hf.STAR_ORDER)
        .fillna(0)
    )
    x = np.arange(len(stack_df))
    
    # Dynamic bar width for consistent physical appearance
    n_bars = len(stack_df)
    fig_width = FIGSIZE_WIDTH // 2  # Single column width
    bar_width = calculate_bar_width(n_bars, fig_width)
    padding = PADDING
    
    cum = np.zeros(len(stack_df))
    for cat in hf.STAR_ORDER:
        vals = stack_df[cat].to_numpy()
        bars = ax.bar(x, vals, width=bar_width, bottom=cum, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
        for j, rect in enumerate(bars):
            h = vals[j]
            if h >= 3:
                label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
                ax.text(rect.get_x() + rect.get_width() / 2, cum[j] + h / 2, f"{h:.1f}%",
                       ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
        cum += vals
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-padding, n_bars - 1 + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(stack_df.index.tolist())
    ax.set_ylabel("% of Students")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Panel 2 — Avg score bars
    ax2 = fig.add_subplot(gs[1, 0])
    x2 = np.arange(len(score_df))
    vals = score_df["avg_score"].to_numpy()
    bars2 = ax2.bar(x2, vals, width=bar_width, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
    for rect, v in zip(bars2, vals):
        ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#434343")
    n_map = pct_df.groupby("time_label")["N_total"].max().to_dict()
    labels = [f"{tl}\n(n = {int(n_map.get(tl, 0))})" if n_map.get(tl) else tl 
             for tl in score_df["time_label"].astype(str)]
    ax2.set_xlim(-padding, n_bars - 1 + padding)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Avg Unified Scale Score")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    # Panel 3 — Insights
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if metrics.get("t_prev"):
        t_curr = metrics.get("t_curr", "Current")
        pct_df_metrics = metrics.get("pct_df")
        if pct_df_metrics is not None and not pct_df_metrics.empty:
            def _bucket_pct(bucket, tlabel):
                return pct_df_metrics.loc[
                    (pct_df_metrics["time_label"] == tlabel) &
                    (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
            
            high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
            hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
            lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
            score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if not score_df.empty and len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
            
            delta_exceed = metrics.get("delta_exceed", 0)
            delta_meet_exceed = metrics.get("delta_meet_exceed", 0)
            delta_not_met = metrics.get("delta_not_met", 0)
            
            lines = [
                "Change",
                f"Met+Exceeded: {delta_meet_exceed:+.1f}%",
                f"Exceeded: {delta_exceed:+.1f}%",
                f"Not Met: {delta_not_met:+.1f}%",
            ]
        else:
            lines = ["Not enough data for insights"]
    else:
        lines = ["Not enough data for insights"]
    ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
            usetex=False, color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
    
    grade_label = f"Grade {grade}"
    fig.suptitle(f"{scope_label} • {grade_label} • Fall → Winter Performance Progression • {title}",
                fontsize=18, fontweight="bold", y=1)
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    activity_type = 'math' if subj == 'math' else 'reading'
    out_name = f"{safe_scope}_STAR_grade{grade}_section1_2_fall_winter_{activity_type}_progression.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    m = dict(metrics) if isinstance(metrics, dict) else {}
    m.pop("pct_df", None)
    chart_data = {
        "chart_type": "star_winter_section1_2_fall_winter_progression_by_grade_single_subject",
        "section": 1.2,
        "scope": scope_label,
        "window_filter": "Fall/Winter",
        "subject": title,
        "grade_data": {"grade": int(grade) if grade is not None else grade},
        "pct_data": pct_df.to_dict("records") if not pct_df.empty else [],
        "score_data": score_df.to_dict("records") if not score_df.empty else [],
        "metrics": m,
        "time_orders": [str(t) for t in time_order],
    }
    track_chart(f"Section 1.2: Grade {grade} Fall → Winter {title}", out_path, scope=scope_label, section=1.2, chart_data=chart_data)
    print(f"Saved Section 1.2 (Grade {grade} {title}): {out_path}")
    return str(out_path)

#def plot_section_1_2_for_grade(df, scope_label, folder, output_dir, grade, school_raw=None, preview=False):
#    """Plot Section 1.2 for a single grade"""
#    fig = plt.figure(figsize=(16, 9), dpi=300)
#    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
#    fig.subplots_adjust(hspace=0.3, wspace=0.25)
#
#    subjects = ["reading", "math"]
#    titles = ["Reading", "Math"]
    # Sidecar JSON payloads for chart_analyzer.py
#    json_subjects: list[str] = []
#    pct_payload: list[dict] = []
#    score_payload: list[dict] = []
#    metrics_payload: list[dict] = []
#    time_orders: list[str] = []
#    axes = [
#        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
#        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
#        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
#    ]
#
#    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
#    any_subject_plotted = False
#
#    for i, subj in enumerate(subjects):
#        pct_df, score_df, metrics, time_order = _prep_star_fall_winter(df, subj)
#
#        if pct_df.empty or score_df.empty:
#            for ax in (axes[0][i], axes[1][i], axes[2][i]):
#                ax.axis("off")
#            continue
#
#        any_subject_plotted = True
#
        # Sidecar payloads (only for subjects that have data)
#        json_subjects.append(titles[i])
#        pct_payload.append({"subject": titles[i], "data": pct_df.to_dict("records")})
#        score_payload.append({"subject": titles[i], "data": score_df.to_dict("records")})
#        if isinstance(metrics, dict):
#            m = dict(metrics)
#            m.pop("pct_df", None)
#        else:
#            m = {}
#        metrics_payload.append(m)
#        if not time_orders and isinstance(time_order, list):
#            time_orders = [str(t) for t in time_order]
#
        # Same plotting logic as Section 1.1
#        ax = axes[0][i]
#        stack_df = (
#            pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
#            .reindex(columns=hf.STAR_ORDER)
#            .fillna(0)
#        )
#        x = np.arange(len(stack_df))
#        cum = np.zeros(len(stack_df))
#        for cat in hf.STAR_ORDER:
#            vals = stack_df[cat].to_numpy()
#            bars = ax.bar(x, vals, bottom=cum, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
#            for j, rect in enumerate(bars):
#                h = vals[j]
#                if h >= 3:
#                    label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
#                    ax.text(rect.get_x() + rect.get_width() / 2, cum[j] + h / 2, f"{h:.1f}%",
#                           ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
#            cum += vals
#        ax.set_ylim(0, 100)
#        ax.set_xticks(x)
#        ax.set_xticklabels(stack_df.index.tolist())
#        ax.set_ylabel("% of Students")
        # ax.grid(False)  # Gridlines disabled globally
#        ax.set_title(titles[i], fontsize=14, fontweight="bold", pad=30)
#        ax.spines["top"].set_visible(False)
#        ax.spines["right"].set_visible(False)
#
#        ax2 = axes[1][i]
#        x2 = np.arange(len(score_df))
#        vals = score_df["avg_score"].to_numpy()
#        bars2 = ax2.bar(x2, vals, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
#        for rect, v in zip(bars2, vals):
#            ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
#                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#434343")
#        n_map = pct_df.groupby("time_label")["N_total"].max().to_dict()
#        labels = [f"{tl}\n(n = {int(n_map.get(tl, 0))})" if n_map.get(tl) else tl 
#                 for tl in score_df["time_label"].astype(str)]
#        ax2.set_xticks(x2)
#        ax2.set_xticklabels(labels)
#        ax2.set_ylabel("Avg Unified Scale Score")
        # ax2.grid(False)  # Gridlines disabled globally
#        ax2.spines["top"].set_visible(False)
#        ax2.spines["right"].set_visible(False)
#
#        ax3 = axes[2][i]
#        ax3.axis("off")
#        if metrics.get("t_prev"):
#            t_curr = metrics.get("t_curr", "Current")
#            pct_df_metrics = metrics.get("pct_df")
            # Show current values, not deltas (deltas still calculated in metrics)
#            if pct_df_metrics is not None and not pct_df_metrics.empty:
#                def _bucket_pct(bucket, tlabel):
#                    return pct_df_metrics.loc[
#                        (pct_df_metrics["time_label"] == tlabel) &
#                        (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
#                    ].sum()
#
#                high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
#                hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
#                lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
#                score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if not score_df.empty and len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
#
#                lines = [
#                    f"Current values ({t_curr}):",
#                    f"Exceeded: {high_now:.1f} ppts",
#                    f"Meet/Exceed: {hi_now:.1f} ppts",
#                    f"Not Met: {lo_now:.1f} ppts",
#                    f"Avg Score: {score_now:.1f} pts",
#                ]
#            else:
#                lines = ["Not enough data for insights"]
#        else:
#            lines = ["Not enough data for insights"]
#        ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
#                usetex=False, color="#333333",
#                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
#
#    if not any_subject_plotted:
#        print(f"Grade {grade}: No Fall/Winter data — skipping chart.")
#        plt.close(fig)
#        return None
#
#    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
#              bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
#
#    grade_label = f"Grade {grade}"
#    fig.suptitle(f"{scope_label} • {grade_label} • Fall → Winter Performance Progression",
#                fontsize=20, fontweight="bold", y=1)
#
#    out_dir_path = Path(output_dir) / folder
#    out_dir_path.mkdir(parents=True, exist_ok=True)
#    safe_scope = scope_label.replace(" ", "_")
#    out_name = f"{safe_scope}_STAR_grade{grade}_section1_2_fall_winter_progression.png"
#    out_path = out_dir_path / out_name
#    hf._save_and_render(fig, out_path, dev_mode=preview)
#
#    chart_data = {
#        "chart_type": "star_winter_section1_2_fall_winter_progression_by_grade",
#        "section": 1.2,
#        "scope": scope_label,
#        "window_filter": "Fall/Winter",
#        "subjects": json_subjects or titles,
#        "grade_data": {"grade": int(grade) if grade is not None else grade},
#        "pct_data": pct_payload,
#        "score_data": score_payload,
#        "metrics": metrics_payload,
#        "time_orders": time_orders,
#    }
#    track_chart(f"Section 1.2: Grade {grade} Fall → Winter", out_path, scope=scope_label, section=1.2, chart_data=chart_data)
#    print(f"Saved Section 1.2 (Grade {grade}): {out_path}")
#    return str(out_path)

def plot_section_1_2(df, scope_label, folder, output_dir, chart_filters=None, school_raw=None, preview=False):
    """Plot Section 1.2 for all grades and subjects"""
    grade_col = "grade" if "grade" in df.columns else ("gradelevelwhenassessed" if "gradelevelwhenassessed" in df.columns else "studentgrade")
    if grade_col not in df.columns:
        print("No grade column found.")
        return []
    
    df["__grade_int"] = pd.to_numeric(df[grade_col], errors="coerce")
    grades = df["__grade_int"].dropna().astype(int).sort_values().unique().tolist()
    
    chart_paths = []
    for grade in grades:
        df_grade = df[df["__grade_int"] == grade].copy()
        if df_grade.empty:
            continue
        # Generate separate charts for Reading and Math
        for subject in ["Reading", "Mathematics"]:
            if chart_filters and not should_generate_subject(subject, chart_filters):
                continue
            path = plot_section_1_2_for_grade_single_subject(
                df_grade, scope_label, folder, output_dir, grade, subject, school_raw, preview
            )
            if path:
                chart_paths.append(path)
    return chart_paths

# ---------------------------------------------------------------------
# SECTION 1.3 — Fall → Winter Performance Progression by Student Group
# ---------------------------------------------------------------------

def _apply_student_group_mask(df_in, group_name, group_def):
    """Returns boolean mask for df_in selecting the student group"""
    if group_def.get("type") == "all":
        return pd.Series(True, index=df_in.index)
    col = group_def["column"]
    allowed_vals = group_def["in"]
    vals = df_in[col].astype(str).str.strip().str.lower()
    allowed_norm = {str(v).strip().lower() for v in allowed_vals}
    return vals.isin(allowed_norm)


def plot_section_1_3_for_group_single_subject(df, scope_label, folder, output_dir, group_name, group_def, subject_str, school_raw=None, preview=False):
    """Plot Section 1.3 for a single student group and subject - Fall → Winter Progression"""
    mask = _apply_student_group_mask(df, group_name, group_def)
    d0 = df[mask].copy()
    
    if d0.empty:
        print(f"[1.3][{group_name}] skipped — no rows")
        return None
    
    if d0["student_state_id"].nunique() < 12:
        print(f"[1.3][{group_name}] skipped (<12 students)")
        return None
    
    # Determine subject
    if subject_str.lower() in ['math', 'mathematics']:
        subj = 'math'
        title = 'Math'
    else:
        subj = 'reading'
        title = 'Reading'
    
    pct_df, score_df, metrics, time_order = _prep_star_fall_winter(d0, subj)
    
    if pct_df.empty or "time_label" not in pct_df.columns:
        print(f"[1.3][{group_name}] No {title} data")
        return None
    
    # Single-column layout
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3)
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    # Panel 1 — 100% stacked bars
    ax = fig.add_subplot(gs[0, 0])
    stack_df = (
        pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
        .reindex(columns=hf.STAR_ORDER)
        .fillna(0)
    )
    x = np.arange(len(stack_df))
    
    # Dynamic bar width for consistent physical appearance
    n_bars = len(stack_df)
    fig_width = FIGSIZE_WIDTH // 2  # Single column width
    bar_width = calculate_bar_width(n_bars, fig_width)
    padding = PADDING
    
    cumulative = np.zeros(len(stack_df))
    for cat in hf.STAR_ORDER:
        vals = stack_df[cat].to_numpy()
        bars = ax.bar(x, vals, width=bar_width, bottom=cumulative, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
        for j, rect in enumerate(bars):
            h = vals[j]
            if h >= 3:
                label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
                ax.text(rect.get_x() + rect.get_width() / 2, cumulative[j] + h / 2, f"{h:.1f}%",
                       ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
        cumulative += vals
    
    ax.set_ylim(0, 100)
    ax.set_xlim(-padding, n_bars - 1 + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(stack_df.index.tolist())
    ax.set_ylabel("% of Students")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
    
    # Panel 2 — Avg score bars
    ax2 = fig.add_subplot(gs[1, 0])
    if score_df is not None and not score_df.empty and "time_label" in score_df.columns:
        x2 = np.arange(len(score_df))
        vals = score_df["avg_score"].to_numpy()
        bars = ax2.bar(x2, vals, width=bar_width, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
        for rect, v in zip(bars, vals):
            ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color="#434343")
        n_map = pct_df.groupby("time_label")["N_total"].max().dropna().astype(int).to_dict()
        labels = [f"{tl}\n(n = {n_map.get(tl, 0)})" for tl in score_df["time_label"].astype(str).tolist()]
        ax2.set_xlim(-padding, n_bars - 1 + padding)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Avg Unified Scale Score")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.axis("off")
    
    # Panel 3 — Insights
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    if metrics.get("t_prev"):
        t_curr = metrics.get("t_curr", "Current")
        pct_df_metrics = metrics.get("pct_df")
        if pct_df_metrics is not None and not pct_df_metrics.empty:
            def _bucket_pct(bucket, tlabel):
                return pct_df_metrics.loc[
                    (pct_df_metrics["time_label"] == tlabel) &
                    (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
            
            high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
            hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
            lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
            score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if score_df is not None and not score_df.empty and len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
            
            delta_exceed = metrics.get("delta_exceed", 0)
            delta_meet_exceed = metrics.get("delta_meet_exceed", 0)
            delta_not_met = metrics.get("delta_not_met", 0)
            
            lines = [
                "Change",
                f"Met+Exceeded: {delta_meet_exceed:+.1f}%",
                f"Exceeded: {delta_exceed:+.1f}%",
                f"Not Met: {delta_not_met:+.1f}%",
            ]
        else:
            lines = ["Not enough data for insights"]
    else:
        lines = ["Not enough data for insights"]
    ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.suptitle(f"{scope_label} • {group_name} • Fall → Winter Performance Progression • {title}",
                fontsize=18, fontweight="bold", y=1)
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    safe_group = group_name.replace(" ", "_").replace("/", "_")
    safe_scope = scope_label.replace(" ", "_")
    activity_type = 'math' if subj == 'math' else 'reading'
    out_name = f"{safe_scope}_STAR_section1_3_{safe_group}_fall_winter_{activity_type}_progression.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    m = dict(metrics) if isinstance(metrics, dict) else {}
    m.pop("pct_df", None)
    chart_data = {
        "chart_type": "star_winter_section1_3_fall_winter_progression_by_group_single_subject",
        "section": 1.3,
        "scope": scope_label,
        "window_filter": "Fall/Winter",
        "subject": title,
        "cohort_data": {"group_name": group_name},
        "pct_data": pct_df.to_dict("records") if not pct_df.empty else [],
        "score_data": score_df.to_dict("records") if score_df is not None and not score_df.empty else [],
        "metrics": m,
        "time_orders": [str(t) for t in time_order],
    }
    track_chart(f"Section 1.3: {group_name} Fall → Winter {title}", out_path, scope=scope_label, section=1.3, chart_data=chart_data)
    print(f"[1.3] Saved ({title}): {out_path}")
    return str(out_path)

#def plot_section_1_3_for_group(df, scope_label, folder, output_dir, group_name, group_def, school_raw=None, preview=False):
#    """Plot Section 1.3 for a single student group"""
#    mask = _apply_student_group_mask(df, group_name, group_def)
#    d0 = df[mask].copy()
#
#    if d0.empty:
#        print(f"[1.3][{group_name}] skipped — no rows")
#        return None
#
#    if d0["student_state_id"].nunique() < 12:
#        print(f"[1.3][{group_name}] skipped (<12 students)")
#        return None
#
#    subjects = ["reading", "math"]
#    titles = ["Reading", "Math"]
#    pct_dfs, score_dfs, metrics_list, time_orders_list = [], [], [], []
#
#    for subj in subjects:
#        pct_df, score_df, metrics, time_order = _prep_star_fall_winter(d0, subj)
#        pct_dfs.append(pct_df)
#        score_dfs.append(score_df)
#        metrics_list.append(metrics)
#        time_orders_list.append(time_order)
#
#    fig = plt.figure(figsize=(16, 9), dpi=300)
#    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
#    fig.subplots_adjust(hspace=0.3, wspace=0.25)
#    axes = [
#        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
#        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
#        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
#    ]
#
#    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
#
#    for i, subj in enumerate(subjects):
#        pct_df = pct_dfs[i]
#        metrics = metrics_list[i]
#        ax = axes[0][i]
#
#        if pct_df.empty or "time_label" not in pct_df.columns:
#            for _ax in (axes[0][i], axes[1][i], axes[2][i]):
#                _ax.axis("off")
#            axes[1][i].text(0.5, 0.5, f"No {titles[i]} data", transform=axes[1][i].transAxes,
#                           ha="center", va="center", fontsize=12, fontweight="bold", color="#999999")
#            continue
#
#        stack_df = (
#            pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
#            .reindex(columns=hf.STAR_ORDER)
#            .fillna(0)
#        )
#        x = np.arange(len(stack_df))
#        cumulative = np.zeros(len(stack_df))
#        for cat in hf.STAR_ORDER:
#            vals = stack_df[cat].to_numpy()
#            bars = ax.bar(x, vals, bottom=cumulative, color=hf.STAR_COLORS[cat], edgecolor="white", linewidth=1.0)
#            for j, rect in enumerate(bars):
#                h = vals[j]
#                if h >= 3:
#                    label_color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
#                    ax.text(rect.get_x() + rect.get_width() / 2, cumulative[j] + h / 2, f"{h:.1f}%",
#                           ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
#            cumulative += vals
#        ax.set_ylim(0, 100)
#        ax.set_xticks(x)
#        ax.set_xticklabels(stack_df.index.tolist())
#        ax.set_ylabel("% of Students")
        # ax.grid(False)  # Gridlines disabled globally
#        ax.set_title(titles[i], fontsize=14, fontweight="bold", pad=30)
#        ax.spines["top"].set_visible(False)
#        ax.spines["right"].set_visible(False)
#
#    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
#              bbox_to_anchor=(0.5, 0.93), ncol=len(hf.STAR_ORDER), frameon=False)
#
#    for i, subj in enumerate(subjects):
#        ax2 = axes[1][i]
#        score_df = score_dfs[i]
#        pct_df = pct_dfs[i]
#
#        if score_df.empty or "time_label" not in score_df.columns:
#            ax2.axis("off")
#            continue
#
#        x2 = np.arange(len(score_df))
#        vals = score_df["avg_score"].to_numpy()
#        bars = ax2.bar(x2, vals, color=hf.default_quartile_colors[3], edgecolor="white", linewidth=1.0)
#        for rect, v in zip(bars, vals):
#            ax2.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}",
#                    ha="center", va="bottom", fontsize=14, fontweight="bold", color="#434343")
#        n_map = pct_df.groupby("time_label")["N_total"].max().dropna().astype(int).to_dict()
#        labels = [f"{tl}\n(n = {n_map.get(tl, 0)})" for tl in score_df["time_label"].astype(str).tolist()]
#        ax2.set_xticks(x2)
#        ax2.set_xticklabels(labels)
#        ax2.set_ylabel("Avg Unified Scale Score")
        # ax2.grid(False)  # Gridlines disabled globally
#        ax2.spines["top"].set_visible(False)
#        ax2.spines["right"].set_visible(False)
#
#    for i, subj in enumerate(subjects):
#        ax3 = axes[2][i]
#        ax3.axis("off")
#        metrics = metrics_list[i]
#        if metrics.get("t_prev"):
#            t_curr = metrics.get("t_curr", "Current")
#            pct_df_metrics = metrics.get("pct_df")
#            score_df_metrics = score_dfs[i] if i < len(score_dfs) else None
            # Show current values, not deltas (deltas still calculated in metrics)
#            if pct_df_metrics is not None and not pct_df_metrics.empty:
#                def _bucket_pct(bucket, tlabel):
#                    return pct_df_metrics.loc[
#                        (pct_df_metrics["time_label"] == tlabel) &
#                        (pct_df_metrics["state_benchmark_achievement"] == bucket), "pct"
#                    ].sum()
#
#                high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
#                hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
#                lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
#                score_now = float(score_df_metrics.loc[score_df_metrics["time_label"] == t_curr, "avg_score"].iloc[0]) if score_df_metrics is not None and not score_df_metrics.empty and len(score_df_metrics[score_df_metrics["time_label"] == t_curr]) > 0 else 0.0
#
#                lines = [
#                    f"Current values ({t_curr}):",
#                    f"Exceeded: {high_now:.1f} ppts",
#                    f"Meet/Exceed: {hi_now:.1f} ppts",
#                    f"Not Met: {lo_now:.1f} ppts",
#                    f"Avg Score: {score_now:.1f} pts",
#                ]
#            else:
#                lines = ["Not enough data for insights"]
#        else:
#            lines = ["Not enough data for insights"]
#        ax3.text(0.5, 0.5, "\n".join(lines), fontsize=10, ha="center", va="center", wrap=True,
#                color="#333333",
#                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
#
#    fig.suptitle(f"{scope_label} • {group_name} • Fall → Winter Performance Progression",
#                fontsize=20, fontweight="bold", y=1)
#
#    out_dir_path = Path(output_dir) / folder
#    out_dir_path.mkdir(parents=True, exist_ok=True)
#    safe_group = group_name.replace(" ", "_").replace("/", "_")
#    safe_scope = scope_label.replace(" ", "_")
#    out_name = f"{safe_scope}_STAR_section1_3_{safe_group}_fall_winter_progression.png"
#    out_path = out_dir_path / out_name
#    hf._save_and_render(fig, out_path, dev_mode=preview)
#
#    json_subjects: list[str] = []
#    pct_payload: list[dict] = []
#    score_payload: list[dict] = []
#    metrics_payload: list[dict] = []
#    time_orders: list[str] = []
#
#    for i in range(len(subjects)):
#        if pct_dfs[i] is None or pct_dfs[i].empty:
#            continue
#        if score_dfs[i] is None or score_dfs[i].empty:
#            continue
#        json_subjects.append(titles[i])
#        pct_payload.append({"subject": titles[i], "data": pct_dfs[i].to_dict("records")})
#        score_payload.append({"subject": titles[i], "data": score_dfs[i].to_dict("records")})
#        met = metrics_list[i]
#        if isinstance(met, dict):
#            m = dict(met)
#            m.pop("pct_df", None)
#        else:
#            m = {}
#        metrics_payload.append(m)
#        if not time_orders and isinstance(time_orders_list[i], list):
#            time_orders = [str(t) for t in time_orders_list[i]]
#
#    chart_data = {
#        "chart_type": "star_winter_section1_3_fall_winter_progression_by_group",
#        "section": 1.3,
#        "scope": scope_label,
#        "window_filter": "Fall/Winter",
#        "subjects": json_subjects or titles,
#        "cohort_data": {"group_name": group_name},
#        "pct_data": pct_payload,
#        "score_data": score_payload,
#        "metrics": metrics_payload,
#        "time_orders": time_orders,
#    }
#    track_chart(f"Section 1.3: {group_name} Fall → Winter", out_path, scope=scope_label, section=1.3, chart_data=chart_data)
#    print(f"[1.3] Saved: {out_path}")
#    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 2 — Student Group Performance Trends (Winter)
# ---------------------------------------------------------------------

def plot_star_single_subject_dashboard_by_group_winter(
    df, scope_label, folder, output_dir, subject_str, window_filter="Winter",
    group_name=None, group_def=None, cfg=None, preview=False
):
    """Single-subject dashboard filtered to one student group - Winter version"""
    d0 = df.copy()
    
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    
    if d0.empty:
        print(f"[group {group_name}] no rows after group mask ({scope_label})")
        return None
    
    # Determine subject and filter
    if subject_str.lower() in ['math', 'mathematics']:
        subj_df = d0[d0["activity_type"].astype(str).str.contains("math", case=False, na=False)].copy()
        title = 'Mathematics'
        activity_type = 'math'
    else:
        subj_df = d0[d0["activity_type"].astype(str).str.contains("reading", case=False, na=False)].copy()
        title = 'Reading'
        activity_type = 'reading'
    
    subj_df["testwindow"] = subj_df["testwindow"].astype(str).str.strip().str.lower()
    win = str(window_filter).strip().lower()
    subj_df = subj_df[subj_df["testwindow"] == win]
    
    if subj_df.empty:
        print(f"[group {group_name}] no {subject_str} data for {scope_label}")
        return None
    
    pct_df, score_df, metrics, time_order = prep_star_for_charts(
        subj_df, subject_str=subject_str, window_filter=window_filter
    )
    
    if len(time_order) > 4:
        time_order = time_order[-4:]
        pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
        score_df = score_df[score_df["time_label"].isin(time_order)].copy()
    
    # Check minimum N
    if pct_df is not None and not pct_df.empty and time_order:
        latest_label = time_order[-1]
        latest_slice = pct_df[pct_df["time_label"] == latest_label]
        if "N_total" in latest_slice.columns:
            latest_n = latest_slice["N_total"].max()
        else:
            latest_n = latest_slice["n"].sum()
        if latest_n < 12:
            print(f"[group {group_name}] skipped (<12 students) in {scope_label}")
            return None
    else:
        print(f"[group {group_name}] skipped (no data) in {scope_label}")
        return None
    
    # Create figure with 3-row layout (insights at bottom) - single column
    fig_width = FIGSIZE_WIDTH // 2
    fig = plt.figure(figsize=(fig_width, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3)
    
    ax_perf = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_insight = fig.add_subplot(gs[2, 0])
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    # Plot performance percentages
    if pct_df is not None and not pct_df.empty:
        stack_df = (
            pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
            .reindex(columns=hf.STAR_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        
        # Dynamic bar width for consistent physical appearance
        n_bars = len(x_labels)
        fig_width = FIGSIZE_WIDTH // 2  # Single column width
        bar_width = calculate_bar_width(n_bars, fig_width)
        padding = PADDING
        
        cumulative = np.zeros(len(stack_df))
        
        for cat in hf.STAR_ORDER:
            vals = stack_df[cat].to_numpy()
            bars = ax_perf.bar(x, vals, width=bar_width, bottom=cumulative, color=hf.STAR_COLORS[cat],
                                 edgecolor="white", linewidth=1.2)
            for idx, rect in enumerate(bars):
                h = vals[idx]
                if h >= LABEL_MIN_PCT:
                    color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
                    ax_perf.text(rect.get_x() + rect.get_width() / 2, cumulative[idx] + h / 2,
                                   f"{h:.1f}%", ha="center", va="center",
                                   fontsize=8, fontweight="bold", color=color)
            cumulative += vals
        
        ax_perf.set_ylim(0, 100)
        ax_perf.set_xlim(-padding, n_bars - 1 + padding)
        ax_perf.set_ylabel("% of Students")
        ax_perf.set_xticks(x)
        ax_perf.set_xticklabels(x_labels)
        ax_perf.spines["top"].set_visible(False)
        ax_perf.spines["right"].set_visible(False)
    else:
        ax_perf.text(0.5, 0.5, f"No {title} data", ha="center", va="center", fontsize=12)
        ax_perf.axis("off")
    ax_perf.set_title(title, fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.92), ncol=len(hf.STAR_ORDER), frameon=False, fontsize=9,
              handlelength=1.8, handletextpad=0.5, columnspacing=1.1)
    
    # Plot score bar
    if pct_df is not None and not pct_df.empty:
        n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
        n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
    else:
        n_map = {}
    
    if score_df is not None and not score_df.empty:
        draw_score_bar(ax_score, score_df, hf.STAR_ORDER, n_map, bar_width=bar_width, fig_width=fig_width)
    else:
        ax_score.text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
        ax_score.axis("off")
    ax_score.set_title("Average Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
    
    # Plot insights
    ax_insight.axis("off")
    if metrics and metrics.get("t_prev"):
        t_prev = metrics["t_prev"]
        t_curr = metrics["t_curr"]
        
        if pct_df is not None and not pct_df.empty:
            def _bucket_pct(bucket, tlabel):
                return pct_df.loc[
                    (pct_df["time_label"] == tlabel) &
                    (pct_df["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
            
            high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
            hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
            lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
            score_now = metrics.get("score_now", 0)
            
            high_delta = metrics.get("high_delta", 0)
            hi_delta = metrics.get("hi_delta", 0)
            lo_delta = metrics.get("lo_delta", 0)
            
            insight_lines = [
                "Change",
                f"Met+Exceeded: {hi_delta:+.1f}%",
                f"Exceeded: {high_delta:+.1f}%",
                f"Not Met: {lo_delta:+.1f}%",
            ]
        else:
            insight_lines = []
    else:
        insight_lines = ["Not enough history for insights"]
    
    ax_insight.text(0.5, 0.5, "\n".join(insight_lines), fontsize=11, fontweight="normal", color="#434343",
                  ha="center", va="center", wrap=True, usetex=False,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.suptitle(f"{scope_label} • {group_name} • {window_filter} {title} Year-to-Year Trends",
                fontsize=20, fontweight="bold", y=1)
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    order_map = cfg.get("student_group_order", {}) if cfg else {}
    group_order_val = order_map.get(group_name, 99)
    safe_group = group_name.replace(" ", "_").replace("/", "_")
    out_name = f"{scope_label.replace(' ', '_')}_STAR_section2_{group_order_val:02d}_{safe_group}_{window_filter.lower()}_{activity_type}_trends.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    # Sidecar JSON
    chart_data = {
        "chart_type": "star_winter_section2_student_group_single_subject",
        "section": 2,
        "scope": scope_label,
        "window_filter": window_filter,
        "subject": title,
        "cohort_data": {"group_name": group_name},
        "pct_data": pct_df.to_dict("records") if pct_df is not None else [],
        "score_data": score_df.to_dict("records") if score_df is not None else [],
        "metrics": metrics if isinstance(metrics, dict) else {},
        "time_order": time_order,
    }
    track_chart(f"Section 2: {group_name} {title}", out_path, scope=scope_label, section=2, chart_data=chart_data)
    print(f"Saved Section 2 ({title}): {out_path}")
    return str(out_path)


# COMMENTED OUT: Dual-subject student group dashboard
# def plot_star_subject_dashboard_by_group_winter(
#     df, scope_label, folder, output_dir, window_filter="Winter",
#     group_name=None, group_def=None, cfg=None, preview=False
# ):
#     """Same layout as main dashboard but filtered to one student group - Winter version"""
#     d0 = df.copy()
#     
#     mask = _apply_student_group_mask(d0, group_name, group_def)
#     d0 = d0[mask].copy()
#     
#     if d0.empty:
#         print(f"[group {group_name}] no rows after group mask ({scope_label})")
#         return None
#     
#     subjects = ["Reading", "Mathematics"]
#     subject_titles = ["Reading", "Mathematics"]
#     
#     pct_dfs, score_dfs, metrics_list, time_orders, min_ns, n_maps = [], [], [], [], [], []
#     
#     for subj in subjects:
#         if subj == "Reading":
#             subj_df = d0[d0["activity_type"].astype(str).str.contains("reading", case=False, na=False)].copy()
#         elif subj == "Mathematics":
#             subj_df = d0[d0["activity_type"].astype(str).str.contains("math", case=False, na=False)].copy()
#         else:
#             subj_df = d0.copy()
#         
#         subj_df["testwindow"] = subj_df["testwindow"].astype(str).str.strip().str.lower()
#         win = str(window_filter).strip().lower()
#         subj_df = subj_df[subj_df["testwindow"] == win]
#         
#         if subj_df.empty:
#             pct_dfs.append(None)
#             score_dfs.append(None)
#             metrics_list.append(None)
#             time_orders.append([])
#             min_ns.append(0)
#             n_maps.append({})
#             continue
#         
#         pct_df, score_df, metrics, time_order = prep_star_for_charts(
#             subj_df, subject_str=subj, window_filter=window_filter
#         )
#         
#         if len(time_order) > 4:
#             time_order = time_order[-4:]
#             pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
#             score_df = score_df[score_df["time_label"].isin(time_order)].copy()
#         
#         pct_dfs.append(pct_df)
#         score_dfs.append(score_df)
#         metrics_list.append(metrics)
#         time_orders.append(time_order)
#         
#         if pct_df is not None and not pct_df.empty and time_order:
#             latest_label = time_order[-1]
#             latest_slice = pct_df[pct_df["time_label"] == latest_label]
#             if "N_total" in latest_slice.columns:
#                 latest_n = latest_slice["N_total"].max()
#             else:
#                 latest_n = latest_slice["n"].sum()
#             min_ns.append(latest_n if not pd.isna(latest_n) else 0)
#         else:
#             min_ns.append(0)
#         
#         if pct_df is not None and not pct_df.empty:
#             n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
#             n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
#         else:
#             n_map = {}
#         n_maps.append(n_map)
#     
#     if any((n is None or n < 12) for n in min_ns):
#         print(f"[group {group_name}] skipped (<12 students) in {scope_label}")
#         return None
#     
#     if all((df is None or df.empty) for df in pct_dfs):
#         print(f"[group {group_name}] skipped (no data) in {scope_label}")
#         return None
#     
#     fig = plt.figure(figsize=(16, 9), dpi=300)
#     gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
#     fig.subplots_adjust(hspace=0.3, wspace=0.25)
#     axes = [
#         [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
#         [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
#         [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
#     ]
#     
#     legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
#     
#     for i, subj in enumerate(subjects):
#         pct_df = pct_dfs[i]
#         score_df = score_dfs[i]
#         metrics = metrics_list[i]
#         time_order = time_orders[i]
#         n_map = n_maps[i]
#         
#         if pct_df is not None and not pct_df.empty:
#             stack_df = (
#                 pct_df.pivot(index="time_label", columns="state_benchmark_achievement", values="pct")
#                 .reindex(columns=hf.STAR_ORDER)
#                 .fillna(0)
#             )
#             x_labels = stack_df.index.tolist()
#             x = np.arange(len(x_labels))
#             cumulative = np.zeros(len(stack_df))
#             
#             for cat in hf.STAR_ORDER:
#                 vals = stack_df[cat].to_numpy()
#                 bars = axes[0][i].bar(x, vals, bottom=cumulative, color=hf.STAR_COLORS[cat],
#                                      edgecolor="white", linewidth=1.2)
#                 for idx, rect in enumerate(bars):
#                     h = vals[idx]
#                     if h >= LABEL_MIN_PCT:
#                         color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
#                         axes[0][i].text(rect.get_x() + rect.get_width() / 2, cumulative[idx] + h / 2,
#                                        f"{h:.1f}%", ha="center", va="center",
#                                        fontsize=8, fontweight="bold", color=color)
#                 cumulative += vals
#             
#             axes[0][i].set_ylim(0, 100)
#             axes[0][i].set_ylabel("% of Students")
#             axes[0][i].set_xticks(x)
#             axes[0][i].set_xticklabels(x_labels)
#             # axes[0][i].grid(False)  # Gridlines disabled globally
#             axes[0][i].spines["top"].set_visible(False)
#             axes[0][i].spines["right"].set_visible(False)
#         else:
#             axes[0][i].text(0.5, 0.5, f"No {subj} data", ha="center", va="center", fontsize=12)
#             axes[0][i].axis("off")
#         axes[0][i].set_title(subject_titles[i], fontsize=14, fontweight="bold", y=1.1)
#     
#     fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center",
#               bbox_to_anchor=(0.5, 0.92), ncol=len(hf.STAR_ORDER), frameon=False, fontsize=9,
#               handlelength=1.8, handletextpad=0.5, columnspacing=1.1)
#     
#     for i in range(2):
#         score_df = score_dfs[i]
#         n_map = n_maps[i] if i < len(n_maps) else {}
#         if score_df is not None and not score_df.empty:
#             draw_score_bar(axes[1][i], score_df, hf.STAR_ORDER, n_map)
#         else:
#             axes[1][i].text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
#             axes[1][i].axis("off")
#         axes[1][i].set_title("Average Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
#     
#     for i in range(2):
#         metrics = metrics_list[i]
#         pct_df = pct_dfs[i]
#         axes[2][i].axis("off")
#         if metrics and metrics.get("t_prev"):
#             t_prev = metrics["t_prev"]
#             t_curr = metrics["t_curr"]
#             
#             def _bucket_delta(bucket, pct_df):
#                 curr = pct_df.loc[(pct_df["time_label"] == t_curr) & (pct_df["state_benchmark_achievement"] == bucket), "pct"].sum()
#                 prev = pct_df.loc[(pct_df["time_label"] == t_prev) & (pct_df["state_benchmark_achievement"] == bucket), "pct"].sum()
#                 return curr - prev
#             
#             if pct_df is not None and not pct_df.empty:
#                 # Show current values, not deltas (deltas still calculated in metrics)
#                 def _bucket_pct(bucket, tlabel):
#                     return pct_df.loc[
#                         (pct_df["time_label"] == tlabel) &
#                         (pct_df["state_benchmark_achievement"] == bucket), "pct"
#                     ].sum()
#                 
#                 high_now = _bucket_pct("4 - Standard Exceeded", t_curr)
#                 hi_now = sum(_bucket_pct(b, t_curr) for b in ["4 - Standard Exceeded", "3 - Standard Met"])
#                 lo_now = _bucket_pct("1 - Standard Not Met", t_curr)
#                 score_now = metrics.get("score_now", 0)
#                 
#                 insight_lines = [
#                     f"Current values ({t_curr}):",
#                     f"Exceed: {high_now:.1f} ppts",
#                     f"Meet or Exceed: {hi_now:.1f} ppts",
#                     f"Not Met: {lo_now:.1f} ppts",
#                     f"Avg Unified Scale Score: {score_now:.1f} pts",
#                 ]
#             else:
#                 insight_lines = []
#         else:
#             insight_lines = ["Not enough history for insights"]
#         
#         axes[2][i].text(0.5, 0.5, "\n".join(insight_lines), fontsize=11, fontweight="normal", color="#434343",
#                       ha="center", va="center", wrap=True, usetex=False,
#                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
#     
#     fig.suptitle(f"{scope_label} • {group_name} • {window_filter} Year-to-Year Trends",
#                 fontsize=20, fontweight="bold", y=1)
#     
#     out_dir_path = Path(output_dir) / folder
#     out_dir_path.mkdir(parents=True, exist_ok=True)
#     order_map = cfg.get("student_group_order", {}) if cfg else {}
#     group_order_val = order_map.get(group_name, 99)
#     safe_group = group_name.replace(" ", "_").replace("/", "_")
#     out_name = f"{scope_label.replace(' ', '_')}_STAR_section2_{group_order_val:02d}_{safe_group}_{window_filter.lower()}_trends.png"
#     out_path = out_dir_path / out_name
#     hf._save_and_render(fig, out_path, dev_mode=preview)
#     
#     # Sidecar JSON for chart_analyzer.py
#     json_subjects: list[str] = []
#     pct_payload: list[dict] = []
#     score_payload: list[dict] = []
#     metrics_payload: list[dict] = []
#     time_order_top: list[str] = []
# 
#     for i in range(len(subjects)):
#         if pct_dfs[i] is None or score_dfs[i] is None:
#             continue
#         if pct_dfs[i].empty or score_dfs[i].empty:
#             continue
#         json_subjects.append(subject_titles[i])
#         pct_payload.append({"subject": subject_titles[i], "data": pct_dfs[i].to_dict("records")})
#         score_payload.append({"subject": subject_titles[i], "data": score_dfs[i].to_dict("records")})
#         met = metrics_list[i]
#         if isinstance(met, dict):
#             m = dict(met)
#             m.pop("pct_df", None)
#         else:
#             m = {}
#         metrics_payload.append(m)
#         if not time_order_top and isinstance(time_orders[i], list):
#             time_order_top = [str(t) for t in time_orders[i]]
# 
#     chart_data = {
#         "chart_type": "star_winter_section2_student_group_trends",
#         "section": 2,
#         "scope": scope_label,
#         "window_filter": window_filter,
#         "subjects": json_subjects or subject_titles,
#         "cohort_data": {"group_name": group_name},
#         "pct_data": pct_payload,
#         "score_data": score_payload,
#         "metrics": metrics_payload,
#         "time_orders": time_order_top,
#     }
#     track_chart(f"Section 2: {group_name}", out_path, scope=scope_label, section=2, chart_data=chart_data)
#     print(f"Saved Section 2: {out_path}")
#     return str(out_path)

# ---------------------------------------------------------------------
# SECTION 3 — Overall + Cohort Trends (Winter)
# ---------------------------------------------------------------------

def _prep_star_matched_cohort_by_grade_winter(df, subject_str, current_grade, window_filter, cohort_year):
    """Prepare matched cohort data for Section 3 - tracks same students across grades - Winter version"""
    base = df.copy()
    base["academicyear"] = pd.to_numeric(base.get("academicyear"), errors="coerce")
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade", "gradelevelwhenassessed"]:
        if col in base.columns:
            grade_col = col
            break
    
    if grade_col is None:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    base[grade_col] = pd.to_numeric(base[grade_col], errors="coerce")
    
    anchor_year = int(cohort_year or base["academicyear"].max())
    cohort_rows, ordered_labels = [], []
    
    for offset in range(0, 4):
        yr = anchor_year - 3 + offset
        gr = current_grade - 3 + offset
        if gr < 0:
            continue
        
        tmp = base[
            (base["testwindow"].astype(str).str.upper() == window_filter.upper())
            & (base[grade_col] == gr)
            & (base["academicyear"] == yr)
        ].copy()
        
        tmp = filter_star_subject_rows(tmp, subject_str)
        
        tmp = tmp[tmp["state_benchmark_achievement"].notna()]
        
        if tmp.empty:
            continue
        
        # Dedupe to latest completion per student
        if "activity_completed_date" in tmp.columns:
            tmp["activity_completed_date"] = pd.to_datetime(tmp["activity_completed_date"], errors="coerce")
            tmp.sort_values(["student_state_id", "activity_completed_date"], inplace=True)
            tmp = tmp.groupby("student_state_id", as_index=False).tail(1)
        elif "student_state_id" in tmp.columns:
            tmp = tmp.groupby("student_state_id", as_index=False).tail(1)
        
        y_prev, y_curr = str(yr - 1)[-2:], str(yr)[-2:]
        label = f"Gr {int(gr)} • {window_filter} {y_prev}-{y_curr}"
        tmp["cohort_label"] = label
        cohort_rows.append(tmp)
        ordered_labels.append(label)
    
    if not cohort_rows:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    cohort_df = pd.concat(cohort_rows, ignore_index=True)
    cohort_df["label"] = cohort_df["cohort_label"]
    
    # Percent by benchmark achievement
    counts = cohort_df.groupby(["label", "state_benchmark_achievement"]).size().rename("n").reset_index()
    totals = cohort_df.groupby("label").size().rename("N_total").reset_index()
    pct_df = counts.merge(totals, on="label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Ensure all combinations exist
    all_idx = pd.MultiIndex.from_product(
        [pct_df["label"].unique(), hf.STAR_ORDER],
        names=["label", "state_benchmark_achievement"],
    )
    pct_df = pct_df.set_index(["label", "state_benchmark_achievement"]).reindex(all_idx).reset_index()
    pct_df[["pct", "n"]] = pct_df[["pct", "n"]].fillna(0)
    pct_df["N_total"] = pct_df.groupby("label")["N_total"].transform(lambda s: s.ffill().bfill())
    
    # Score averages
    score_df = (
        cohort_df[["label", "unified_scale"]]
        .dropna(subset=["unified_scale"])
        .groupby("label")["unified_scale"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )
    
    # Set categorical order
    pct_df["label"] = pd.Categorical(pct_df["label"], categories=ordered_labels, ordered=True)
    score_df["label"] = pd.Categorical(score_df["label"], categories=ordered_labels, ordered=True)
    pct_df = pct_df.rename(columns={"label": "time_label"}).sort_values("time_label").reset_index(drop=True)
    score_df = score_df.rename(columns={"label": "time_label"}).sort_values("time_label").reset_index(drop=True)
    
    # Metrics
    labels_order = ordered_labels
    last_two = labels_order[-2:] if len(labels_order) >= 2 else labels_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        
        def pct_for(buckets, tlabel):
            tlabel_str = str(tlabel)
            mask = (pct_df["time_label"].astype(str) == tlabel_str) & (pct_df["state_benchmark_achievement"].isin(buckets))
            return pct_df[mask]["pct"].sum()
        
        hi_now = pct_for(hf.STAR_HIGH_GROUP, t_curr)
        lo_now = pct_for(hf.STAR_LOW_GROUP, t_curr)
        hi_delta = hi_now - pct_for(hf.STAR_HIGH_GROUP, t_prev)
        lo_delta = lo_now - pct_for(hf.STAR_LOW_GROUP, t_prev)
        
        high_now = pct_for(["4 - Standard Exceeded"], t_curr)
        high_prev = pct_for(["4 - Standard Exceeded"], t_prev)
        high_delta = high_now - high_prev
        
        score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0])
        score_prev = float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])
        
        metrics = dict(
            t_prev=t_prev,
            t_curr=t_curr,
            hi_now=hi_now,
            hi_delta=hi_delta,
            high_now=high_now,
            high_delta=high_delta,
            lo_now=lo_now,
            lo_delta=lo_delta,
            score_now=score_now,
            score_delta=score_now - score_prev,
        )
    else:
        metrics = {k: None for k in ["t_prev", "t_curr", "hi_now", "hi_delta", "high_now", "high_delta", "lo_now", "lo_delta", "score_now", "score_delta"]}
    
    return pct_df, score_df, metrics, ordered_labels


def plot_star_blended_dashboard_winter(
    df, scope_label, folder, output_dir, subject_str, current_grade,
    window_filter="Winter", cohort_year=None, cfg=None, preview=False
):
    """Dual-facet dashboard showing Overall vs Cohort Trends - Winter version"""
    d = df.copy()
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade", "gradelevelwhenassessed"]:
        if col in d.columns:
            grade_col = col
            break
    
    if grade_col is None:
        print(f"[Section 3] No grade column found for {scope_label}")
        return None
    
    d[grade_col] = pd.to_numeric(d[grade_col], errors="coerce")
    d = d[d[grade_col] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    d = filter_star_subject_rows(d, subject_str)
    
    d = d[d["state_benchmark_achievement"].notna()]
    
    pct_df_left, score_df_left, metrics_left, time_order_left = prep_star_for_charts(
        d, subject_str=subject_str, window_filter=window_filter
    )
    
    pct_df_right, score_df_right, metrics_right, cohort_labels = _prep_star_matched_cohort_by_grade_winter(
        df, subject_str, current_grade, window_filter, cohort_year
    )
    
    # Check if we have data
    if (pct_df_left.empty or score_df_left.empty) and (pct_df_right.empty or score_df_right.empty):
        print(f"[Section 3] No data for {scope_label} - Grade {current_grade} - {subject_str}")
        return None
    
    # Create figure with 2 columns (Overall left, Cohort right, insights at bottom)
    fig = plt.figure(figsize=(FIGSIZE_WIDTH, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Left side: Overall trends
    axes_left = [
        fig.add_subplot(gs[0, 0]),  # Stacked bars (top left)
        fig.add_subplot(gs[1, 0]),  # Score bars (bottom left)
        fig.add_subplot(gs[2, 0])   # Insights (bottom left)
    ]
    
    # Right side: Cohort trends
    axes_right = [
        fig.add_subplot(gs[0, 1]),  # Stacked bars (top right)
        fig.add_subplot(gs[1, 1]),  # Score bars (bottom right)
        fig.add_subplot(gs[2, 1])   # Insights (bottom right)
    ]
    
    # Left side: Overall trends
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    if not pct_df_left.empty and not score_df_left.empty:
        draw_stacked_bar(axes_left[0], pct_df_left, score_df_left, hf.STAR_ORDER)
        n_map_left = None
        if "N_total" in pct_df_left.columns:
            n_map_df = pct_df_left.groupby("time_label")["N_total"].max().reset_index()
            n_map_left = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(axes_left[1], score_df_left, hf.STAR_ORDER, n_map_left)
        draw_insight_card(axes_left[2], metrics_left, subject_str)
    else:
        for ax in axes_left:
            ax.text(0.5, 0.5, "No overall data", ha="center", va="center", fontsize=12)
            ax.axis("off")
    
    axes_left[0].set_title("Overall Trends", fontsize=14, fontweight="bold", y=1.1)
    
    # Right side: Cohort trends
    if not pct_df_right.empty and not score_df_right.empty:
        draw_stacked_bar(axes_right[0], pct_df_right, score_df_right, hf.STAR_ORDER)
        n_map_right = None
        if "N_total" in pct_df_right.columns:
            n_map_df = pct_df_right.groupby("time_label")["N_total"].max().reset_index()
            n_map_right = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(axes_right[1], score_df_right, hf.STAR_ORDER, n_map_right)
        draw_insight_card(axes_right[2], metrics_right, subject_str)
    else:
        for ax in axes_right:
            ax.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=12)
            ax.axis("off")
    
    axes_right[0].set_title("Cohort Trends", fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.STAR_ORDER), frameon=False, fontsize=10)
    
    fig.suptitle(f"{scope_label} • Grade {current_grade} • {subject_str} • {window_filter} Trends",
                fontsize=20, fontweight="bold", y=1)
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else ""
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_STAR_section3_grade{current_grade}_{safe_subj}_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 3: {out_path}")
    
    m_left = dict(metrics_left) if isinstance(metrics_left, dict) else {}
    m_left.pop("pct_df", None)
    m_right = dict(metrics_right) if isinstance(metrics_right, dict) else {}
    m_right.pop("pct_df", None)

    chart_data = {
        "chart_type": "star_winter_section3_overall_and_cohort_trends",
        "section": 3,
        "scope": scope_label,
        "window_filter": window_filter,
        # This chart has two panels (Overall vs Cohort) for a single subject
        "subjects": ["Overall", "Cohort"],
        "grade_data": {"grade": int(current_grade), "subject": subject_str},
        "pct_data": [
            {"subject": "Overall", "data": pct_df_left.to_dict("records") if not pct_df_left.empty else []},
            {"subject": "Cohort", "data": pct_df_right.to_dict("records") if not pct_df_right.empty else []},
        ],
        "score_data": [
            {"subject": "Overall", "data": score_df_left.to_dict("records") if not score_df_left.empty else []},
            {"subject": "Cohort", "data": score_df_right.to_dict("records") if not score_df_right.empty else []},
        ],
        "metrics": [m_left, m_right],
        "time_orders": [str(t) for t in (time_order_left or [])],
        "cohort_data": {"cohort_labels": [str(c) for c in (cohort_labels or [])]},
    }
    track_chart(f"Section 3: Grade {current_grade} {subject_str}", out_path, scope=scope_label, section=3, chart_data=chart_data)
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 4 — Overall Growth Trends by Site (Winter)
# ---------------------------------------------------------------------

def plot_star_growth_by_site_winter(
    df, scope_label, folder, output_dir, subject_str, window_filter="Winter", cfg=None, preview=False
):
    """Show growth trends broken down by school/site - Winter version"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    d = filter_star_subject_rows(d, subject_str)
    
    d = d[d["state_benchmark_achievement"].notna()]
    
    # Get school column
    school_col = "school_name" if "school_name" in d.columns else "schoolname"
    if school_col not in d.columns:
        print(f"[Section 4] No school column found for {scope_label}")
        return None
    
    schools = sorted(d[school_col].dropna().unique())
    if not schools:
        print(f"[Section 4] No schools found for {scope_label}")
        return None
    
    # Prepare data for each school
    school_data = {}
    for school in schools:
        school_df = d[d[school_col] == school].copy()
        pct_df, score_df, metrics, time_order = prep_star_for_charts(
            school_df, subject_str=subject_str, window_filter=window_filter
        )
        if not pct_df.empty and not score_df.empty:
            m = dict(metrics) if isinstance(metrics, dict) else {}
            m.pop("pct_df", None)
            school_data[school] = {
                "metrics": m,
                "time_order": [str(t) for t in (time_order or [])],
                "pct_data": pct_df,
                "score_data": score_df,
            }
    
    if not school_data:
        print(f"[Section 4] No valid school data for {scope_label}")
        return None
    
    # Create figure - one row per school
    n_schools = len(school_data)
    fig = plt.figure(figsize=(FIGSIZE_WIDTH, 4 * n_schools), dpi=300)
    gs = fig.add_gridspec(nrows=n_schools, ncols=2, height_ratios=[1] * n_schools)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    for idx, (school, data) in enumerate(sorted(school_data.items())):
        pct_df = data["pct_data"]
        score_df = data["score_data"]
        
        # Stacked bar chart
        ax1 = fig.add_subplot(gs[idx, 0])
        draw_stacked_bar(ax1, pct_df, score_df, hf.STAR_ORDER)
        ax1.set_title(f"{school}", fontsize=12, fontweight="bold")
        
        # Score bar chart
        ax2 = fig.add_subplot(gs[idx, 1])
        n_map = None
        if "N_total" in pct_df.columns:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(ax2, score_df, hf.STAR_ORDER, n_map)
        ax2.set_title("Average Unified Scale Score", fontsize=10, fontweight="bold")
    
    fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.98),
              ncol=len(hf.STAR_ORDER), frameon=False, fontsize=9)
    
    fig.suptitle(f"{scope_label} • {subject_str} • {window_filter} Growth Trends by Site",
                fontsize=18, fontweight="bold", y=0.995)
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else ""
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_STAR_section4_{safe_subj}_{window_filter.lower()}_growth_by_site.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 4: {out_path}")
    
    chart_data = {
        "chart_type": "star_winter_section4_growth_by_site",
        "section": 4,
        "scope": scope_label,
        "window_filter": window_filter,
        "subjects": [subject_str],
        "school_data": {
            school: {
                "pct_data": data["pct_df"].to_dict("records") if not data["pct_data"].empty else [],
                "score_data": data["score_df"].to_dict("records") if not data["score_data"].empty else [],
                "metrics": data["metrics"],
                "time_order": data["time_order"]
            }
            for school, data in school_data.items()
        },
    }
    track_chart(f"Section 4: {subject_str} Growth by Site", out_path, scope=scope_label, section=4, chart_data=chart_data)
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 4.1 — District-Level SGP Overview Helper Functions
# ---------------------------------------------------------------------

def _prep_star_sgp_trend_district_overview(df, subject_str):
    """
    Prepare Winter SGP trend data aggregated across all grades for district overview.
    Returns: (DataFrame, vector_used: str)
    - DataFrame has columns: time_label, median_sgp, n, subject
    - vector_used is the SGP vector string (e.g., "FALL_WINTER") or None if no data
    Limited to the most recent 4 time_labels.
    Uses any available SGP data (prioritizes FALL_WINTER if available).
    """
    d = df.copy()
    
    # Check for SGP columns
    if "current_sgp_vector" not in d.columns:
        print(f"[Section 4 Overview] No current_sgp_vector column found for {subject_str}")
        return pd.DataFrame(columns=["time_label", "median_sgp", "n", "subject"]), None
    
    if "current_sgp" not in d.columns:
        print(f"[Section 4 Overview] No current_sgp column found for {subject_str}")
        return pd.DataFrame(columns=["time_label", "median_sgp", "n", "subject"]), None
    
    # Winter term only
    d = d[d["testwindow"].astype(str).str.upper() == "WINTER"].copy()
    
    # Subject filtering using existing filter_star_subject_rows
    d = filter_star_subject_rows(d, subject_str)
    
    # Check for rows with any SGP data
    d_with_sgp = d[d["current_sgp_vector"].notna() & d["current_sgp"].notna()].copy()
    
    if d_with_sgp.empty:
        print(f"[Section 4 Overview] No SGP data found for {subject_str}")
        return pd.DataFrame(columns=["time_label", "median_sgp", "n", "subject"]), None
    
    # Check what SGP vectors are available
    available_vectors = d_with_sgp["current_sgp_vector"].unique()
    vector_used = None  # Track which vector we use
    
    # Prefer FALL_WINTER, but use any available SGP data
    if "FALL_WINTER" in available_vectors:
        d = d_with_sgp[d_with_sgp["current_sgp_vector"] == "FALL_WINTER"].copy()
        vector_used = "FALL_WINTER"
        print(f"[Section 4 Overview] Using FALL_WINTER SGP data for {subject_str}")
    else:
        most_common_vector = d_with_sgp["current_sgp_vector"].mode().iloc[0] if not d_with_sgp["current_sgp_vector"].mode().empty else available_vectors[0]
        d = d_with_sgp[d_with_sgp["current_sgp_vector"] == most_common_vector].copy()
        vector_used = most_common_vector
        print(f"[Section 4 Overview] No FALL_WINTER data, using {most_common_vector} SGP data for {subject_str}")
    
    if d.empty:
        return pd.DataFrame(columns=["time_label", "median_sgp", "n", "subject"]), None
    
    # Build time labels using existing _short_year function
    d["academicyear_short"] = d["academicyear"].apply(_short_year)
    d["time_label"] = "Winter " + d["academicyear_short"]
    
    # Deduplicate to latest test per student per time_label
    if "activity_completed_date" in d.columns:
        d["activity_completed_date"] = pd.to_datetime(d["activity_completed_date"], errors="coerce")
        d.sort_values(["student_state_id", "time_label", "activity_completed_date"], inplace=True)
        d = d.groupby(["student_state_id", "time_label"], as_index=False).tail(1)
    
    # Aggregate median SGP across ALL grades
    out = (
        d.groupby("time_label", dropna=False)
        .agg(
            median_sgp=("current_sgp", "median"),
            n=("current_sgp", "size"),
        )
        .reset_index()
    )
    
    if out.empty:
        return pd.DataFrame(columns=["time_label", "median_sgp", "n", "subject"]), None
    
    # Keep last 4 Winter windows
    order = sorted(out["time_label"].astype(str).unique())
    out["time_label"] = pd.Categorical(out["time_label"], order, ordered=True)
    out = out.sort_values("time_label").tail(4).reset_index(drop=True)
    out["subject"] = subject_str
    
    return out, vector_used


def plot_district_sgp_overview_single_subject_winter(
    df, scope_label, folder, output_dir, subject_str, window_filter="Winter", preview=False
):
    """
    District-level SGP overview: Single subject version.
    Aggregates across all grades to show overall district growth for one subject.
    """
    # Determine subject title
    if subject_str.lower() in ['math', 'mathematics']:
        title = 'Math'
    else:
        title = 'Reading'
    
    sgp_color = "#0381a2"
    band_color = "#eab308"
    band_line_color = "#ffa800"
    
    # Prepare SGP data for this subject
    trend_df, vector_used = _prep_star_sgp_trend_district_overview(df, subject_str=subject_str)
    
    # Check if we have any data
    if trend_df.empty:
        print(f"[Section 4 Overview] No SGP data for {scope_label} - {title}")
        return None
    
    # Get the SGP vector label from the actual data used
    sgp_vector_label = "SGP"  # Default
    if vector_used:
        if vector_used == "FALL_WINTER":
            sgp_vector_label = "Fall→Winter SGP"
        elif vector_used == "SPRING_FALL":
            sgp_vector_label = "Spring→Fall SGP"
        else:
            sgp_vector_label = f"{vector_used} SGP"
    
    # Create single-subject plot (single column)
    fig_width = FIGSIZE_WIDTH // 2
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 9), dpi=300)
    
    if not {"time_label", "median_sgp", "n"}.issubset(trend_df.columns):
        ax.axis("off")
        ax.text(0.5, 0.5, f"No {title} data", ha="center", va="center",
               fontsize=16, fontweight="bold", color="#434343")
        plt.close(fig)
        return None
    
    sub = trend_df.copy()
    x = np.arange(len(sub))
    y = sub["median_sgp"].to_numpy(float)
    
    # Calculate dynamic bar width for consistent appearance
    n_bars = len(sub)
    bar_width = calculate_bar_width(n_bars, fig_width)
    padding = PADDING
    
    # Growth band (35-65 typical growth range)
    ax.axhspan(35, 65, facecolor=band_color, alpha=0.25, zorder=0)
    for yref in [35, 50, 65]:
        ax.axhline(yref, ls="--", color=band_line_color, lw=1.2, zorder=0)
    
    # Bars with consistent width
    bars = ax.bar(x, y, width=bar_width, color=sgp_color, edgecolor="white", linewidth=1.2, zorder=2)
    for rect, val in zip(bars, y):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
               f"{val:.1f}", ha="center", va="center", fontsize=9,
               fontweight="bold", color="white")
    
    # Add n-counts under x-axis
    n_map = sub.set_index("time_label")["n"].astype(int).to_dict()
    formatted_labels = [f"{tl}\n(n = {n_map.get(tl, 0)})" 
                       for tl in sub["time_label"].astype(str).tolist()]
    
    ax.set_xlim(-padding, n_bars - 1 + padding)
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_labels)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"Median {sgp_vector_label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend
    fig.legend(
        handles=[Patch(facecolor=sgp_color, label="Median SGP")],
        loc="upper center", bbox_to_anchor=(0.5, 0.95),
        frameon=False, fontsize=10
    )
    
    # Title
    fig.suptitle(
        f"{scope_label} • {sgp_vector_label} Trends (All Grades) • {title}",
        fontsize=18, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"DISTRICT_{safe_scope}_STAR_section4_sgp_overview_{safe_subj}_{window_filter.lower()}.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 4 SGP Overview ({title}): {out_path}")
    
    # Track chart with data
    chart_data = {
        "chart_type": "star_winter_section4_sgp_overview_single_subject",
        "section": 4,
        "scope": scope_label,
        "window_filter": window_filter,
        "subject": title,
        "sgp_data": trend_df.to_dict("records") if not trend_df.empty else []
    }
    track_chart(f"Section 4: District SGP Overview {title}", out_path, scope=scope_label, section=4, chart_data=chart_data)
    plt.close(fig)
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 5 — STAR SGP Growth: Grade Trend + Backward Cohort (Winter)
# ---------------------------------------------------------------------

def _prep_star_sgp_data_winter(df, subject_str, current_grade, window_filter):
    """Prepare SGP (Student Growth Percentile) data for Section 5 - Winter version"""
    d = df.copy()
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade", "gradelevelwhenassessed"]:
        if col in d.columns:
            grade_col = col
            break
    
    if grade_col is None:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    d[grade_col] = pd.to_numeric(d[grade_col], errors="coerce")
    d = d[d[grade_col] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    d = filter_star_subject_rows(d, subject_str)
    
    # Check for SGP columns - Winter uses Fall→Winter SGP vector
    if "current_sgp_vector" in d.columns:
        d = d[d["current_sgp_vector"] == "FALL_WINTER"].copy()
    
    sgp_col = None
    for col in ["current_sgp", "sgp", "student_growth_percentile"]:
        if col in d.columns:
            sgp_col = col
            break
    
    if sgp_col is None:
        print(f"[Section 5] No SGP column found - Grade {current_grade} - {subject_str}")
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    d = d[d[sgp_col].notna()]
    
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # Build time label
    d["academicyear_short"] = d["academicyear"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["academicyear_short"]
    
    # Dedupe
    if "activity_completed_date" in d.columns:
        d["activity_completed_date"] = pd.to_datetime(d["activity_completed_date"], errors="coerce")
        d.sort_values(["student_state_id", "activity_completed_date"], inplace=True)
        d = d.groupby(["student_state_id", "time_label"], as_index=False).tail(1)
    elif "student_state_id" in d.columns:
        d = d.groupby(["student_state_id", "time_label"], as_index=False).tail(1)
    
    # Average SGP per time window
    sgp_df = (
        d.groupby("time_label")[sgp_col]
        .median()
        .rename("avg_sgp")
        .reset_index()
    )
    
    # Count per time window
    counts = d.groupby("time_label").size().rename("N_total").reset_index()
    sgp_df = sgp_df.merge(counts, on="time_label", how="left")
    
    # Chronological order
    time_order = sorted(sgp_df["time_label"].unique().tolist())
    sgp_df["time_label"] = pd.Categorical(sgp_df["time_label"], categories=time_order, ordered=True)
    sgp_df.sort_values("time_label", inplace=True)
    
    # Limit to most recent 4
    if len(time_order) > 4:
        time_order = time_order[-4:]
        sgp_df = sgp_df[sgp_df["time_label"].isin(time_order)].copy()
    
    # Metrics
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        sgp_curr = float(sgp_df.loc[sgp_df["time_label"] == t_curr, "avg_sgp"].iloc[0])
        sgp_prev = float(sgp_df.loc[sgp_df["time_label"] == t_prev, "avg_sgp"].iloc[0])
        
        metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "sgp_now": sgp_curr,
            "sgp_delta": sgp_curr - sgp_prev,
        }
    else:
        metrics = {k: None for k in ["t_prev", "t_curr", "sgp_now", "sgp_delta"]}
    
    return sgp_df, pd.DataFrame(), metrics, time_order


def _prep_star_sgp_cohort_winter(df, subject_str, current_grade, window_filter):
    """Prepare backward cohort SGP data - Winter version"""
    d0 = df.copy()
    d0["studentgrade"] = pd.to_numeric(d0.get("studentgrade", d0.get("gradelevelwhenassessed", pd.Series())), errors="coerce")
    d0["academicyear"] = pd.to_numeric(d0["academicyear"], errors="coerce")
    
    if d0[d0["studentgrade"] == current_grade]["academicyear"].notna().any():
        anchor_year = int(d0[d0["studentgrade"] == current_grade]["academicyear"].max())
    else:
        return pd.DataFrame()
    
    cohort_rows = []
    for offset in range(3, -1, -1):
        yr = anchor_year - offset
        gr = current_grade - offset
        if gr < 0:
            continue
        
        d = df.copy()
        d["studentgrade"] = pd.to_numeric(d.get("studentgrade", d.get("gradelevelwhenassessed", pd.Series())), errors="coerce")
        d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
        
        if "current_sgp_vector" not in d.columns or "testwindow" not in d.columns:
            continue
        
        d = d[
            (d["academicyear"] == yr)
            & (d["studentgrade"] == gr)
            & (d["testwindow"].astype(str).str.upper() == window_filter.upper())
            & (d["current_sgp_vector"] == "FALL_WINTER")
        ].copy()
        
        d = filter_star_subject_rows(d, subject_str)
        
        if d.empty:
            continue
        
        d["activity_completed_date"] = pd.to_datetime(d.get("activity_completed_date"), errors="coerce")
        d = d.dropna(subset=["activity_completed_date"])
        
        # Dedupe most recent Winter test per student
        for id_col in ["student_state_id", "ssid", "studentid"]:
            if id_col in d.columns:
                d = d.sort_values("activity_completed_date").drop_duplicates(id_col, keep="last")
                break
        
        if "current_sgp" not in d.columns:
            continue
        
        d = d.dropna(subset=["current_sgp"])
        if d.empty:
            continue
        
        # Build time_label in "Gr {gr} • Winter YY-YY" format
        yy_prev = str(int(yr) - 1)[-2:]
        yy = str(int(yr))[-2:]
        label = f"Gr {int(gr)} • Winter {yy_prev}-{yy}"
        
        cohort_rows.append({
            "time_label": label,
            "median_sgp": d["current_sgp"].median(),
            "n": len(d),
            "gr": gr,
            "yr": yr,
        })
    
    if not cohort_rows:
        return pd.DataFrame()
    
    out = pd.DataFrame(cohort_rows)
    
    # Ensure sorting by grade numeric order
    def _extract_grade_num(label):
        import re
        m = re.search(r"Gr\s*(\d+)", str(label))
        return int(m.group(1)) if m else 0
    
    out["grade_num"] = out["time_label"].apply(_extract_grade_num)
    out = out.sort_values(["grade_num", "time_label"])
    out["time_label"] = pd.Categorical(out["time_label"], categories=out["time_label"], ordered=True)
    
    return out


def plot_star_sgp_growth_winter(
    df, scope_label, folder, output_dir, subject_str, current_grade,
    window_filter="Winter", cfg=None, preview=False
):
    """Show SGP growth trends by grade and backward cohort - Winter version"""
    # Grade trend (current grade over time)
    sgp_df_grade, _, metrics_grade, time_order = _prep_star_sgp_data_winter(
        df, subject_str, current_grade, window_filter
    )
    
    # Backward cohort (same students tracked backward)
    cohort_df = _prep_star_sgp_cohort_winter(df, subject_str, current_grade, window_filter)
    
    if sgp_df_grade.empty and cohort_df.empty:
        print(f"[Section 5] No SGP data for {scope_label} - Grade {current_grade} - {subject_str}")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(FIGSIZE_WIDTH, 9), dpi=300)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    sgp_color = "#0381a2"
    band_color = "#eab308"
    band_line_color = "#ffa800"
    
    # Top left: Grade trend (SGP over time)
    ax1 = fig.add_subplot(gs[0, 0])
    if not sgp_df_grade.empty:
        x_labels = sgp_df_grade["time_label"].tolist()
        x = np.arange(len(x_labels))
        y = sgp_df_grade["avg_sgp"].tolist()
        
        # Dynamic bar width for consistent physical appearance
        n_bars = len(x_labels)
        fig_width = FIGSIZE_WIDTH // 2  # Single column width
        bar_width = calculate_bar_width(n_bars, fig_width)
        padding = PADDING
        
        # Growth band
        ax1.axhspan(35, 65, facecolor=band_color, alpha=0.25, zorder=0)
        for yref in [35, 50, 65]:
            ax1.axhline(yref, ls="--", color=band_line_color, lw=1.2, zorder=0)
        
        bars = ax1.bar(x, y, width=bar_width, color=sgp_color, edgecolor="white", linewidth=1.2, zorder=2)
        for rect, v in zip(bars, y):
            ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
                    f"{v:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        
        # Add n-counts
        if "N_total" in sgp_df_grade.columns:
            n_map = dict(zip(sgp_df_grade["time_label"].astype(str), sgp_df_grade["N_total"]))
            formatted_labels = []
            for tl in x_labels:
                n = n_map.get(str(tl), 0)
                formatted_labels.append(f"{tl}\n(n = {n})")
            ax1.set_xticklabels(formatted_labels)
        else:
            ax1.set_xticklabels(x_labels)
        
        ax1.set_xlim(-padding, n_bars - 1 + padding)
        ax1.set_xticks(x)
        ax1.set_ylabel("Median Fall→Winter SGP", fontsize=11, fontweight="bold")
        ax1.set_title("Overall Growth Trends", fontsize=14, fontweight="bold")
        ax1.set_ylim(0, 100)
        # ax1.grid(False)  # Gridlines disabled globally
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
    else:
        ax1.text(0.5, 0.5, "No grade trend data", ha="center", va="center", fontsize=12)
        ax1.axis("off")
    
    # Top right: Backward cohort
    ax2 = fig.add_subplot(gs[0, 1])
    if not cohort_df.empty:
        x_labels_cohort = cohort_df["time_label"].tolist()
        x_cohort = np.arange(len(x_labels_cohort))
        y_cohort = cohort_df["median_sgp"].tolist()
        
        # Dynamic bar width for consistent physical appearance
        n_bars_cohort = len(x_labels_cohort)
        fig_width = FIGSIZE_WIDTH // 2  # Single column width
        bar_width = calculate_bar_width(n_bars_cohort, fig_width)
        padding_cohort = PADDING
        
        # Growth band
        ax2.axhspan(35, 65, facecolor=band_color, alpha=0.25, zorder=0)
        for yref in [35, 50, 65]:
            ax2.axhline(yref, ls="--", color=band_line_color, lw=1.2, zorder=0)
        
        bars_cohort = ax2.bar(x_cohort, y_cohort, width=bar_width, color=sgp_color, edgecolor="white", linewidth=1.2, zorder=2)
        for rect, v in zip(bars_cohort, y_cohort):
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2,
                    f"{v:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        
        # Add n-counts
        if "n" in cohort_df.columns:
            n_map_cohort = dict(zip(cohort_df["time_label"].astype(str), cohort_df["n"]))
            formatted_labels_cohort = []
            for tl in x_labels_cohort:
                n = n_map_cohort.get(str(tl), 0)
                formatted_labels_cohort.append(f"{tl}\n(n = {n})")
            ax2.set_xticklabels(formatted_labels_cohort)
        else:
            ax2.set_xticklabels(x_labels_cohort)
        
        ax2.set_xlim(-padding_cohort, n_bars_cohort - 1 + padding_cohort)
        ax2.set_xticks(x_cohort)
        ax2.set_ylabel("Median Fall→Winter SGP", fontsize=11, fontweight="bold")
        ax2.set_title("Cohort Growth Trends", fontsize=14, fontweight="bold")
        ax2.set_ylim(0, 100)
        # ax2.grid(False)  # Gridlines disabled globally
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=12)
        ax2.axis("off")
    
    # Unified legend
    fig.legend(
        handles=[Patch(facecolor=sgp_color, label="Median SGP")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=10,
    )
    
    # Main title
    fig.suptitle(
        f"{scope_label} • {subject_str} • Grade {current_grade} • Fall→Winter SGP",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else ""
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_STAR_section5_grade{current_grade}_{safe_subj}_{window_filter.lower()}_sgp_growth.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 5: {out_path}")
    
    m = dict(metrics_grade) if isinstance(metrics_grade, dict) else {}
    m.pop("pct_df", None)
    chart_data = {
        "chart_type": "star_winter_section5_sgp_growth",
        "section": 5,
        "scope": scope_label,
        "window_filter": window_filter,
        "subjects": [subject_str],
        "grade_data": {"grade": int(current_grade), "subject": subject_str},
        "metrics": [m],
        "time_orders": [str(t) for t in (time_order or [])],
        "sgp_data": {
            "grade_trend": sgp_df_grade.to_dict("records") if not sgp_df_grade.empty else [],
            "cohort_trend": cohort_df.to_dict("records") if not cohort_df.empty else [],
        },
    }
    track_chart(f"Section 5: Grade {current_grade} {subject_str} SGP", out_path, scope=scope_label, section=5, chart_data=chart_data)
    
    return str(out_path)

# ---------------------------------------------------------------------
# HELPER FUNCTIONS FOR SECTIONS 6-11
# ---------------------------------------------------------------------

def _latest_academicyear(df: pd.DataFrame) -> int | None:
    """Get the latest academic year from dataframe"""
    if "academicyear" not in df.columns:
        return None
    ay = pd.to_numeric(df["academicyear"], errors="coerce")
    if ay.notna().any():
        return int(ay.max())
    return None


def _dedupe_latest_attempt(
    d: pd.DataFrame,
    id_col: str,
    keys: list[str],
    date_col: str = "activity_completed_date",
) -> pd.DataFrame:
    """Keep latest attempt per (id + keys) by activity_completed_date if available."""
    out = d.copy()
    
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out[date_col] = pd.NaT
    
    sort_cols = [id_col] + keys + [date_col]
    sort_cols = [c for c in sort_cols if c in out.columns]
    out = out.sort_values(sort_cols)
    
    grp_cols = [id_col] + [c for c in keys if c in out.columns]
    out = out.groupby(grp_cols, as_index=False).tail(1)
    return out


def _prep_star_perf_fall_winter_by(
    df: pd.DataFrame,
    subject: str,
    by_col: str,
) -> tuple[pd.DataFrame, dict]:
    """Prepare percent-by-band for Fall vs Winter for the latest academicyear."""
    d = df.copy()
    
    # Latest year only
    latest_ay = _latest_academicyear(d)
    if latest_ay is None:
        return pd.DataFrame(), {}
    d = d[pd.to_numeric(d["academicyear"], errors="coerce") == latest_ay].copy()
    
    # Fall/Winter only
    if "testwindow" not in d.columns:
        return pd.DataFrame(), {}
    d = d[d["testwindow"].astype(str).str.upper().isin(["FALL", "WINTER"])].copy()
    
    # Subject filtering
    d = filter_star_subject_rows(d, subject)
    if d.empty:
        return pd.DataFrame(), {}
    
    # Must have benchmark
    if "state_benchmark_achievement" not in d.columns:
        return pd.DataFrame(), {}
    d = d[d["state_benchmark_achievement"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), {}
    
    # Time labels
    yy_prev = str(int(latest_ay) - 1)[-2:]
    yy = str(int(latest_ay))[-2:]
    d["time_label"] = d["testwindow"].astype(str).str.title() + f" {yy_prev}-{yy}"
    
    # Ensure by_col exists
    if by_col not in d.columns:
        return pd.DataFrame(), {}
    
    # Dedupe to latest attempt per student + (by_col, time_label)
    if "student_state_id" not in d.columns:
        return pd.DataFrame(), {}
    d = _dedupe_latest_attempt(d, "student_state_id", keys=[by_col, "time_label"])
    
    # Aggregate pct within each (by, time_label)
    counts = (
        d.groupby([by_col, "time_label", "state_benchmark_achievement"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = d.groupby([by_col, "time_label"]).size().rename("N_total").reset_index()
    out = counts.merge(totals, on=[by_col, "time_label"], how="left")
    out["pct"] = 100 * out["n"] / out["N_total"]
    
    # Ensure full STAR_ORDER grid for each (by, time)
    idx = pd.MultiIndex.from_product(
        [out[by_col].dropna().unique(), out["time_label"].unique(), hf.STAR_ORDER],
        names=[by_col, "time_label", "state_benchmark_achievement"],
    )
    out = (
        out.set_index([by_col, "time_label", "state_benchmark_achievement"])
        .reindex(idx)
        .reset_index()
    )
    out["pct"] = out["pct"].fillna(0)
    out["n"] = out["n"].fillna(0)
    out["N_total"] = out.groupby([by_col, "time_label"])["N_total"].transform(
        lambda s: s.ffill().bfill()
    )
    
    meta = {
        "latest_ay": latest_ay,
        "time_order": [f"Fall {yy_prev}-{yy}", f"Winter {yy_prev}-{yy}"],
        "yy_label": f"{yy_prev}-{yy}",
    }
    return out, meta


def _prep_star_sgp_latest_by(
    df: pd.DataFrame,
    subject: str,
    by_col: str,
    sgp_vector: str = "FALL_WINTER",
) -> pd.DataFrame:
    """Prepare latest-year Winter SGP by a category column."""
    d = df.copy()
    
    # Latest year only
    latest_ay = _latest_academicyear(d)
    if latest_ay is None:
        return pd.DataFrame()
    d = d[pd.to_numeric(d["academicyear"], errors="coerce") == latest_ay].copy()
    
    # Winter only
    if "testwindow" not in d.columns:
        return pd.DataFrame()
    d = d[d["testwindow"].astype(str).str.upper() == "WINTER"].copy()
    
    # Subject
    d = filter_star_subject_rows(d, subject)
    if d.empty:
        return pd.DataFrame()
    
    # SGP window
    if "current_sgp_vector" not in d.columns:
        return pd.DataFrame()
    d = d[d["current_sgp_vector"] == sgp_vector].copy()
    
    # Must have SGP values
    if "current_sgp" not in d.columns:
        return pd.DataFrame()
    d = d[d["current_sgp"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    
    # Ensure by_col exists
    if by_col not in d.columns:
        return pd.DataFrame()
    
    # Dedupe latest attempt per student + by_col
    if "student_state_id" not in d.columns:
        return pd.DataFrame()
    d = _dedupe_latest_attempt(
        d, "student_state_id", keys=[by_col, "current_sgp_vector"]
    )
    
    out = (
        d.groupby(by_col)
        .agg(median_sgp=("current_sgp", "median"), n=("current_sgp", "size"))
        .reset_index()
    )
    
    # Sort
    if by_col == "studentgrade":
        out[by_col] = pd.to_numeric(out[by_col], errors="coerce")
        out = out.sort_values(by_col)
    else:
        out[by_col] = out[by_col].astype(str)
        out = out.sort_values(by_col)
    
    return out


def _resolve_group_key(desired: str, student_groups_cfg: dict) -> str | None:
    """Resolve group key by exact match, casefold match, or common aliases."""
    if not student_groups_cfg:
        return None
    
    # Direct
    if desired in student_groups_cfg:
        return desired
    
    # Case-insensitive match
    desired_norm = str(desired).strip().casefold()
    for k in student_groups_cfg.keys():
        if str(k).strip().casefold() == desired_norm:
            return k
    
    # Aliases
    alias_map = {
        "swd": "Students with Disabilities",
        "sed": "Socioeconomically Disadvantaged",
        "el": "English Learners",
    }
    if desired_norm in alias_map:
        return _resolve_group_key(alias_map[desired_norm], student_groups_cfg)
    
    return None

# ---------------------------------------------------------------------
# PLOTTING FUNCTIONS FOR SECTIONS 6-11
# ---------------------------------------------------------------------

def _plot_star_perf_fall_winter_single_subject(
    pct_df: pd.DataFrame,
    by_col: str,
    subject: str,
    scope_label: str,
    folder: str,
    output_dir: str,
    section_num: int,
    out_stub: str,
    preview: bool = False,
) -> str | None:
    """100% stacked bars: two bars per category (Fall vs Winter) with Winter higher opacity."""
    if pct_df is None or pct_df.empty:
        print(f"[Section {section_num}] Skipped {subject} (no data)")
        return None
    
    # Build time order from labels found
    time_vals = sorted(pct_df["time_label"].astype(str).unique().tolist())
    fall_label = [t for t in time_vals if t.lower().startswith("fall")]
    winter_label = [t for t in time_vals if t.lower().startswith("winter")]
    time_order = (
        (fall_label + winter_label) if (fall_label and winter_label) else time_vals
    )
    
    if len(time_order) < 2:
        print(f"[Section {section_num}] Skipped {subject} (need both Fall and Winter data)")
        return None
    
    # Keep grade as numeric to match pivot index; other categories use strings
    if by_col == "studentgrade":
        cats = pct_df[by_col].dropna().astype(int).unique().tolist()
    else:
        cats = pct_df[by_col].dropna().astype(str).unique().tolist()
    
    # Sort categories
    if by_col == "studentgrade":
        cats = sorted(cats)
    else:
        cats = sorted(cats)
    
    if len(cats) == 0:
        print(f"[Section {section_num}] Skipped {subject} (no categories)")
        return None
    
    # Create a pivot table per time
    def _stack_for(tlabel: str) -> pd.DataFrame:
        sub = pct_df[pct_df["time_label"].astype(str) == str(tlabel)].copy()
        stack = (
            sub.pivot(index=by_col, columns="state_benchmark_achievement", values="pct")
            .reindex(index=cats, columns=hf.STAR_ORDER)
            .fillna(0)
        )
        return stack
    
    # Handle both single and dual window scenarios
    stack_f = _stack_for(time_order[0]) if len(time_order) > 0 else pd.DataFrame()
    stack_w = _stack_for(time_order[1]) if len(time_order) > 1 else pd.DataFrame()
    has_both = len(time_order) >= 2
    
    # n maps
    n_map = (
        pct_df.groupby([by_col, "time_label"])["N_total"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )
    
    # Single column layout
    fig_width = FIGSIZE_WIDTH
    fig, ax = plt.subplots(figsize=(fig_width, 9), dpi=300)
    
    x = np.arange(len(cats))
    
    # Adjust bar positioning based on whether we have one or two windows
    # Let matplotlib use default bar width (0.8) instead of custom calculation
    if has_both:
        bar_w = 0.35  # Default width for side-by-side bars
        x_f = x - bar_w / 2
        x_w = x + bar_w / 2
    else:
        bar_w = 0.8  # Default matplotlib bar width for single bars
        x_f = x
        x_w = x  # Won't be used
    
    # Match source: Fall slightly lighter when comparing, full color when alone
    fall_alpha = 0.60 if has_both else 1.00
    winter_alpha = 1.00
    
    # Plot Fall bars (or the only window if single)
    cum = np.zeros(len(cats))
    for band in hf.STAR_ORDER:
        vals = stack_f[band].to_numpy() if not stack_f.empty else np.zeros(len(cats))
        ax.bar(
            x_f,
            vals,
            width=bar_w,
            bottom=cum,
            color=hf.STAR_COLORS[band],
            alpha=fall_alpha,
            edgecolor="white",
            linewidth=1.0,
        )
        # labels
        for i, v in enumerate(vals):
            if v >= 3:
                label_color = (
                    "#434343" if band == "2 - Standard Nearly Met" else "white"
                )
                ax.text(
                    x_f[i],
                    cum[i] + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=label_color,
                )
        cum += vals
    
    # Plot Winter bars (only if we have Winter data)
    if has_both and not stack_w.empty:
        cum = np.zeros(len(cats))
        for band in hf.STAR_ORDER:
            vals = stack_w[band].to_numpy()
            ax.bar(
                x_w,
                vals,
                width=bar_w,
                bottom=cum,
                color=hf.STAR_COLORS[band],
                alpha=winter_alpha,
                edgecolor="white",
                linewidth=1.0,
            )
            # labels
            for i, v in enumerate(vals):
                if v >= 3:
                    label_color = (
                        "#434343" if band == "2 - Standard Nearly Met" else "white"
                    )
                    ax.text(
                        x_w[i],
                        cum[i] + v / 2,
                        f"{v:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=label_color,
                    )
            cum += vals
    
    # X labels include n sizes
    labels = []
    for c in cats:
        nf = n_map.get((c, str(time_order[0])), 0)
        
        # Display K for grade 0
        if by_col == "studentgrade":
            disp = "K" if int(c) == 0 else str(int(c))
        else:
            disp = str(c)
        
        if has_both:
            nw = n_map.get((c, str(time_order[1])), 0)
            labels.append(f"{disp}\n(F n={nf} | W n={nw})")
        else:
            # Single window - just show that window's label and count
            window_label = "F" if "fall" in str(time_order[0]).lower() else "W"
            labels.append(f"{disp}\n({window_label} n={nf})")
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Students")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Rotate school + student group labels
    if by_col in ["school_name", "schoolname", "student_group"]:
        for t in ax.get_xticklabels():
            t.set_rotation(35)
            t.set_ha("right")
    
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add small "Fall" / "Winter" tags ABOVE each category group (only if both windows exist)
    if has_both:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i in range(len(cats)):
            ax.text(
                x_f[i],
                1.01,
                "Fall",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#434343",
                transform=trans,
                clip_on=False,
            )
            ax.text(
                x_w[i],
                1.01,
                "Winter",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#434343",
                transform=trans,
                clip_on=False,
            )
    
    # Legend
    legend_handles = [
        Patch(facecolor=hf.STAR_COLORS[b], edgecolor="none", label=b)
        for b in hf.STAR_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        labels=hf.STAR_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(hf.STAR_ORDER),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    
    # Title
    title_by = {
        "school_name": "School",
        "studentgrade": "Grade",
        "student_group": "Student Group",
    }.get(by_col, by_col)
    
    # Adjust title based on available windows
    if has_both:
        title_text = f"{scope_label} • {subject}\nFall → Winter Performance by {title_by}"
    else:
        window_name = time_order[0].split()[0]  # Extract "Fall" or "Winter"
        title_text = f"{scope_label} • {subject}\n{window_name} Performance by {title_by}"
    
    fig.suptitle(
        title_text,
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    safe_subject = subject.lower().replace(" ", "_")
    out_name = f"{safe_scope}_STAR_section{section_num}_{out_stub}_{safe_subject}.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    # Track chart
    chart_data = {
        "chart_type": f"star_winter_section{section_num}_{out_stub}",
        "section": section_num,
        "scope": scope_label,
        "window_filter": "Fall/Winter",
        "subject": subject,
        "pct_data": pct_df.to_dict("records") if not pct_df.empty else [],
        "time_orders": [str(t) for t in time_order],
    }
    track_chart(f"Section {section_num}: {title_by} Fall→Winter {subject}", out_path, scope=scope_label, section=section_num, chart_data=chart_data)
    
    print(f"[Section {section_num}] Saved: {out_path}")
    if preview:
        plt.show()
    plt.close(fig)
    
    return str(out_path)


def _plot_star_sgp_by_single_subject(
    sgp_df: pd.DataFrame,
    by_col: str,
    subject: str,
    scope_label: str,
    folder: str,
    output_dir: str,
    section_num: int,
    out_stub: str,
    window_label: str = "Fall→Winter",
    preview: bool = False,
) -> str | None:
    """Bar chart of median SGP by category with 35–65 reference band."""
    if sgp_df is None or sgp_df.empty:
        print(f"[Section {section_num}] Skipped {subject} ({window_label}) (no data)")
        return None
    
    d = sgp_df.copy()
    
    # Enforce min-n on student group charts
    if by_col == "student_group":
        d = d[d["n"].fillna(0).astype(int) >= 12].copy()
        if d.empty:
            print(
                f"[Section {section_num}] Skipped {subject} ({window_label}) (<12 students for all groups)"
            )
            return None
    
    # X labels
    if by_col == "studentgrade":
        x_labels = d[by_col].fillna(-999).astype(int).apply(lambda g: "K" if g == 0 else str(g)).tolist()
        title_by = "Grade"
    elif by_col == "school_name":
        x_labels = d[by_col].astype(str).tolist()
        title_by = "School"
    else:
        x_labels = d[by_col].astype(str).tolist()
        title_by = "Student Group"
    
    x = np.arange(len(d))
    y = d["median_sgp"].to_numpy(float)
    
    # Single column layout - full width to match sections 6-8
    fig_width = FIGSIZE_WIDTH
    fig, ax = plt.subplots(figsize=(fig_width, 9), dpi=300)
    
    # Reference band
    band_color = "#eab308"
    band_line_color = "#ffa800"
    ax.axhspan(35, 65, facecolor=band_color, alpha=0.25, zorder=0)
    for yref in [35, 50, 65]:
        ax.axhline(yref, ls="--", color=band_line_color, lw=1.2, zorder=0)
    
    sgp_color = "#0381a2"
    
    # Let matplotlib decide bar width automatically
    n_bars = len(d)
    
    bars = ax.bar(x, y, color=sgp_color, edgecolor="white", linewidth=1.2, zorder=2)
    
    for rect, val in zip(bars, y):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() / 2,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
    
    # X tick labels with n
    labels = []
    for lbl, n in zip(x_labels, d["n"].fillna(0).astype(int).tolist()):
        labels.append(f"{lbl}\n(n = {n})")
    
    # Let matplotlib decide axis limits
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Rotate school + student group labels
    if by_col in ["school_name", "schoolname", "student_group"]:
        for t in ax.get_xticklabels():
            t.set_rotation(35)
            t.set_ha("right")
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Median SGP")
    ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle(
        f"{scope_label} • {subject}\n{window_label} Median SGP by {title_by}",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    safe_subject = subject.lower().replace(" ", "_")
    safe_win = window_label.lower().replace("→", "to").replace(" ", "_")
    out_name = (
        f"{safe_scope}_STAR_section{section_num}_{out_stub}_{safe_subject}_{safe_win}.png"
    )
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    # Track chart
    chart_data = {
        "chart_type": f"star_winter_section{section_num}_{out_stub}",
        "section": section_num,
        "scope": scope_label,
        "window_filter": window_label,
        "subject": subject,
        "sgp_data": sgp_df.to_dict("records") if not sgp_df.empty else [],
    }
    track_chart(f"Section {section_num}: {title_by} {window_label} SGP {subject}", out_path, scope=scope_label, section=section_num, chart_data=chart_data)
    
    print(f"[Section {section_num}] Saved: {out_path}")
    if preview:
        plt.show()
    plt.close(fig)
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 6 — Performance by School (Fall vs Winter)
# ---------------------------------------------------------------------

def plot_section_6_performance_by_school_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: Fall vs Winter performance by school for one subject"""
    school_col = "school_name" if "school_name" in df.columns else "schoolname"
    
    pct_df, meta = _prep_star_perf_fall_winter_by(df, subject_str, by_col=school_col)
    
    return _plot_star_perf_fall_winter_single_subject(
        pct_df,
        by_col=school_col,
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=6,
        out_stub="by_school_fall_winter_perf",
        preview=preview,
    )

# ---------------------------------------------------------------------
# SECTION 7 — Performance by Grade (Fall vs Winter)
# ---------------------------------------------------------------------

def plot_section_7_performance_by_grade_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: Fall vs Winter performance by grade for one subject"""
    pct_df, meta = _prep_star_perf_fall_winter_by(df, subject_str, by_col="studentgrade")
    
    return _plot_star_perf_fall_winter_single_subject(
        pct_df,
        by_col="studentgrade",
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=7,
        out_stub="by_grade_fall_winter_perf",
        preview=preview,
    )

# ---------------------------------------------------------------------
# SECTION 8 — Performance by Student Group (Fall vs Winter)
# ---------------------------------------------------------------------

def plot_section_8_performance_by_group_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: Fall vs Winter performance by student group"""
    student_groups_cfg = cfg.get("student_groups", {}) if cfg else {}
    
    # Default group list
    STAR_GROUPS_DEFAULT = [
        "All Students",
        "Students with Disabilities",
        "Socioeconomically Disadvantaged",
        "English Learners",
        "Hispanic",
        "White",
    ]
    
    # Build working group list
    star_group_keys = []
    for g in STAR_GROUPS_DEFAULT:
        k = _resolve_group_key(g, student_groups_cfg)
        if k is not None:
            star_group_keys.append(k)
    
    if not star_group_keys:
        print(f"[Section 8] Skipped {subject_str} (student_groups not configured)")
        return None
    
    # Materialize synthetic column `student_group`
    rows = []
    for gk in star_group_keys:
        gdef = student_groups_cfg.get(gk, {})
        if not gdef:
            continue
        mask = _apply_student_group_mask(df, gk, gdef)
        d_g = df[mask].copy()
        if d_g.empty:
            continue
        if d_g["student_state_id"].nunique() < 12:
            continue
        d_g["student_group"] = gk
        rows.append(d_g)
    
    if not rows:
        print(f"[Section 8] Skipped {subject_str} (no groups with n>=12)")
        return None
    
    d_all = pd.concat(rows, ignore_index=True)
    pct_df, meta = _prep_star_perf_fall_winter_by(
        d_all, subject_str, by_col="student_group"
    )
    
    return _plot_star_perf_fall_winter_single_subject(
        pct_df,
        by_col="student_group",
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=8,
        out_stub="by_group_fall_winter_perf",
        preview=preview,
    )

# ---------------------------------------------------------------------
# SECTION 9 — SGP by School (Fall→Winter)
# ---------------------------------------------------------------------

def plot_section_9_sgp_by_school_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    sgp_vector: str = "FALL_WINTER",
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: SGP by school for one subject and one vector"""
    school_col = "school_name" if "school_name" in df.columns else "schoolname"
    
    sgp_df = _prep_star_sgp_latest_by(
        df, subject_str, by_col=school_col, sgp_vector=sgp_vector
    )
    
    return _plot_star_sgp_by_single_subject(
        sgp_df,
        by_col=school_col,
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=9,
        out_stub="by_school_sgp",
        window_label="Fall→Winter",
        preview=preview,
    )

# ---------------------------------------------------------------------
# SECTION 10 — SGP by Grade (Fall→Winter)
# ---------------------------------------------------------------------

def plot_section_10_sgp_by_grade_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    sgp_vector: str = "FALL_WINTER",
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: SGP by grade for one subject and one vector"""
    sgp_df = _prep_star_sgp_latest_by(
        df, subject_str, by_col="studentgrade", sgp_vector=sgp_vector
    )
    
    return _plot_star_sgp_by_single_subject(
        sgp_df,
        by_col="studentgrade",
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=10,
        out_stub="by_grade_sgp",
        window_label="Fall→Winter",
        preview=preview,
    )

# ---------------------------------------------------------------------
# SECTION 11 — SGP by Student Group (Fall→Winter)
# ---------------------------------------------------------------------

def plot_section_11_sgp_by_group_winter(
    df: pd.DataFrame,
    scope_label: str,
    folder: str,
    output_dir: str,
    subject_str: str,
    sgp_vector: str = "FALL_WINTER",
    cfg: dict = None,
    preview: bool = False,
) -> str | None:
    """District-level: SGP by student group"""
    student_groups_cfg = cfg.get("student_groups", {}) if cfg else {}
    
    # Default group list
    STAR_GROUPS_DEFAULT = [
        "All Students",
        "Students with Disabilities",
        "Socioeconomically Disadvantaged",
        "English Learners",
        "Hispanic",
        "White",
    ]
    
    # Build working group list
    star_group_keys = []
    for g in STAR_GROUPS_DEFAULT:
        k = _resolve_group_key(g, student_groups_cfg)
        if k is not None:
            star_group_keys.append(k)
    
    if not star_group_keys:
        print(f"[Section 11] Skipped {subject_str} (student_groups not configured)")
        return None
    
    # Materialize synthetic column `student_group`
    rows = []
    for gk in star_group_keys:
        gdef = student_groups_cfg.get(gk, {})
        if not gdef:
            continue
        mask = _apply_student_group_mask(df, gk, gdef)
        d_g = df[mask].copy()
        if d_g.empty:
            continue
        if d_g["student_state_id"].nunique() < 12:
            continue
        d_g["student_group"] = gk
        rows.append(d_g)
    
    if not rows:
        print(
            f"[Section 11] Skipped {subject_str} (Fall→Winter) (no groups with n>=12)"
        )
        return None
    
    d_all = pd.concat(rows, ignore_index=True)
    sgp_df = _prep_star_sgp_latest_by(
        d_all, subject_str, by_col="student_group", sgp_vector=sgp_vector
    )
    
    return _plot_star_sgp_by_single_subject(
        sgp_df,
        by_col="student_group",
        subject=subject_str,
        scope_label=scope_label,
        folder=folder,
        output_dir=output_dir,
        section_num=11,
        out_stub="by_group_sgp",
        window_label="Fall→Winter",
        preview=preview,
    )

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main(star_data=None):
    """
    Main function to generate STAR Winter charts
    
    Args:
        star_data: Optional list of dicts or DataFrame with STAR data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate STAR Winter charts')
    parser.add_argument('--partner', required=True, help='Partner name')
    parser.add_argument('--data-dir', required=False, help='Data directory path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--dev-mode', default='false', help='Development mode')
    parser.add_argument('--config', default='{}', help='Config JSON string')
    
    args = parser.parse_args()
    
    cfg = load_config_from_args(args.config)
    hf.DEV_MODE = args.dev_mode.lower() in ('true', '1', 'yes', 'on')
    
    chart_filters = cfg.get("chart_filters", {})
    
    # Load data
    if star_data is not None:
        star_base = load_star_data(star_data=star_data, cfg=cfg)
    else:
        if not args.data_dir:
            raise ValueError("Either star_data must be provided or --data-dir must be specified")
        star_base = load_star_data(data_dir=args.data_dir, cfg=cfg)
    
    # Always use Winter for this module
    selected_quarters = ["Winter"]
    
    # Get scopes
    scopes = get_scopes(star_base, cfg)
    
    chart_paths = []
    
    # Section 0: Predicted vs Actual CAASPP (Winter)
    print("\n[Section 0] Generating Winter Predicted vs Actual CAASPP...")
    for scope_df, scope_label, folder in scopes:
        for subj in ["Reading", "Mathematics"]:
            if not should_generate_subject(subj, chart_filters):
                continue
            try:
                proj, act, metrics, year = _prep_section0_star_winter(scope_df, subj)
                if proj is None:
                    continue
                chart_path = _plot_section0_star_single_subject_winter(
                    scope_label, folder, subj, proj, act, metrics,
                    args.output_dir, preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error generating Section 0 {subj} chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 1: Winter Performance Trends
    print("\n[Section 1] Generating Winter Performance Trends...")
    for quarter in selected_quarters:
        for scope_df, scope_label, folder in scopes:
            # Generate separate charts for Reading and Math
            for subject in ["Reading", "Mathematics"]:
                if not should_generate_subject(subject, chart_filters):
                    continue
                try:
                    chart_path = plot_star_single_subject_dashboard_winter(
                        scope_df,
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subject,
                        window_filter=quarter,
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating {subject} chart for {scope_label} ({quarter}): {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 1.1: Fall → Winter Performance Progression
    print("\n[Section 1.1] Generating Fall → Winter Performance Progression...")
    for scope_df, scope_label, folder in scopes:
        # Generate separate charts for Reading and Math
        for subject in ["Reading", "Mathematics"]:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_1_1_single_subject(
                    scope_df,
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    school_raw=None if folder == "_district" else scope_label,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error generating Section 1.1 {subject} chart for {scope_label}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Section 1.2: Fall → Winter Performance Progression by Grade
    print("\n[Section 1.2] Generating Fall → Winter Performance Progression by Grade...")
    for scope_df, scope_label, folder in scopes:
        try:
            grade_paths = plot_section_1_2(
                scope_df,
                scope_label,
                folder,
                args.output_dir,
                chart_filters=chart_filters,
                school_raw=None if folder == "_district" else scope_label,
                preview=hf.DEV_MODE
            )
            chart_paths.extend(grade_paths)
        except Exception as e:
            print(f"Error generating Section 1.2 charts for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 1.3: Fall → Winter Performance Progression by Student Group
    print("\n[Section 1.3] Generating Fall → Winter Performance Progression by Student Group...")
    student_groups_cfg = cfg.get("student_groups", {})
    group_order = cfg.get("student_group_order", {})
    
    # Limit number of groups if max_student_groups filter is set
    max_groups = chart_filters.get("max_student_groups", 10)  # Default limit
    sorted_groups = sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 999))
    groups_to_plot = sorted_groups[:max_groups]
    
    for scope_df, scope_label, folder in scopes:
        for group_name, group_def in groups_to_plot:
            if group_def.get("type") == "all":
                continue
            # Check if this student group should be generated based on filters
            if not should_generate_student_group(group_name, chart_filters):
                continue
            # Generate separate charts for Reading and Math
            for subject in ["Reading", "Mathematics"]:
                if not should_generate_subject(subject, chart_filters):
                    continue
                try:
                    chart_path = plot_section_1_3_for_group_single_subject(
                        scope_df,
                        scope_label,
                        folder,
                        args.output_dir,
                        group_name,
                        group_def,
                        subject_str=subject,
                        school_raw=None if folder == "_district" else scope_label,
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 1.3 {subject} chart for {scope_label} ({group_name}): {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 2: Student Group Performance Trends (Winter)
    print("\n[Section 2] Generating Student Group Performance Trends (Winter)...")
    
    # Limit number of groups if max_student_groups filter is set
    max_groups = chart_filters.get("max_student_groups", 10)  # Default limit
    sorted_groups = sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99))
    groups_to_plot = sorted_groups[:max_groups]
    
    for scope_df, scope_label, folder in scopes:
        for group_name, group_def in groups_to_plot:
            if group_def.get("type") == "all":
                continue
            # Check if this student group should be generated based on filters
            if not should_generate_student_group(group_name, chart_filters):
                continue
            # Generate separate charts for Reading and Math
            for subject in ["Reading", "Mathematics"]:
                if not should_generate_subject(subject, chart_filters):
                    continue
                try:
                    chart_path = plot_star_single_subject_dashboard_by_group_winter(
                        scope_df,
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subject,
                        window_filter="Winter",
                        group_name=group_name,
                        group_def=group_def,
                        cfg=cfg,
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 2 {subject} chart for {scope_label} ({group_name}): {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Section 3: Overall + Cohort Trends (Winter)
    print("\n[Section 3] Generating Overall + Cohort Trends (Winter)...")
    selected_grades = chart_filters.get("grades", [])
    if not selected_grades:
        # Query all available grades from data (no hardcoded limit)
        grade_col = "grade" if "grade" in star_base.columns else ("gradelevelwhenassessed" if "gradelevelwhenassessed" in star_base.columns else "studentgrade")
        if grade_col in star_base.columns:
            star_base["__grade_int"] = pd.to_numeric(star_base[grade_col], errors="coerce")
            available_grades = sorted([int(g) for g in star_base["__grade_int"].dropna().unique() if not pd.isna(g)])
            selected_grades = available_grades
            star_base = star_base.drop(columns=["__grade_int"], errors="ignore")
        else:
            # Fallback: use all grades from Pre-K to 12
            selected_grades = list(range(-1, 13))
    
    anchor_year = int(star_base["academicyear"].max()) if "academicyear" in star_base.columns else None
    
    subjects_to_plot = _requested_star_subjects(chart_filters)
    for scope_df, scope_label, folder in scopes:
        for subj in subjects_to_plot:
            if not should_generate_subject(subj, chart_filters):
                continue
            for grade in selected_grades:
                if not should_generate_grade(grade, chart_filters):
                    continue
                try:
                    chart_path = plot_star_blended_dashboard_winter(
                        scope_df.copy(),
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subj,
                        current_grade=grade,
                        window_filter="Winter",
                        cohort_year=anchor_year,
                        cfg=cfg,
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 3 chart for {scope_label} - Grade {grade} - {subj}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 4: Overall Growth Trends by Site (Winter)
    print("\n[Section 4] Generating Overall Growth Trends by Site (Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder == "_district":
            # District-level SGP overview (separate charts for Reading and Math)
            for subj in subjects_to_plot:
                if not should_generate_subject(subj, chart_filters):
                    continue
                try:
                    chart_path = plot_district_sgp_overview_single_subject_winter(
                        scope_df.copy(),
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subj,
                        window_filter="Winter",
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 4 SGP Overview for {scope_label} - {subj}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
        else:
            # School-level SGP overview (separate charts for Reading and Math)
            for subj in subjects_to_plot:
                if not should_generate_subject(subj, chart_filters):
                    continue
                try:
                    chart_path = plot_district_sgp_overview_single_subject_winter(
                        scope_df.copy(),
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subj,
                        window_filter="Winter",
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 4 SGP Overview for {scope_label} - {subj}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 5: STAR SGP Growth - Grade Trend + Backward Cohort (Winter)
    print("\n[Section 5] Generating STAR SGP Growth (Winter)...")
    for scope_df, scope_label, folder in scopes:
        for subj in subjects_to_plot:
            if not should_generate_subject(subj, chart_filters):
                continue
            for grade in selected_grades:
                if not should_generate_grade(grade, chart_filters):
                    continue
                try:
                    chart_path = plot_star_sgp_growth_winter(
                        scope_df.copy(),
                        scope_label,
                        folder,
                        args.output_dir,
                        subject_str=subj,
                        current_grade=grade,
                        window_filter="Winter",
                        cfg=cfg,
                        preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"Error generating Section 5 chart for {scope_label} - Grade {grade} - {subj}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 6: Performance by School (Fall vs Winter)
    print("\n[Section 6] Generating Performance by School (Fall vs Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_6_performance_by_school_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 6 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    # Section 7: Performance by Grade (Fall vs Winter)
    print("\n[Section 7] Generating Performance by Grade (Fall vs Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_7_performance_by_grade_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 7 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    # Section 8: Performance by Student Group (Fall vs Winter)
    print("\n[Section 8] Generating Performance by Student Group (Fall vs Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_8_performance_by_group_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 8 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    # Section 9: SGP by School (Fall→Winter)
    print("\n[Section 9] Generating SGP by School (Fall→Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_9_sgp_by_school_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    sgp_vector="FALL_WINTER",
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 9 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    # Section 10: SGP by Grade (Fall→Winter)
    print("\n[Section 10] Generating SGP by Grade (Fall→Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_10_sgp_by_grade_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    sgp_vector="FALL_WINTER",
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 10 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    # Section 11: SGP by Student Group (Fall→Winter)
    print("\n[Section 11] Generating SGP by Student Group (Fall→Winter)...")
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":  # District-level only
            continue
        for subject in subjects_to_plot:
            if not should_generate_subject(subject, chart_filters):
                continue
            try:
                chart_path = plot_section_11_sgp_by_group_winter(
                    scope_df.copy(),
                    scope_label,
                    folder,
                    args.output_dir,
                    subject_str=subject,
                    sgp_vector="FALL_WINTER",
                    cfg=cfg,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error Section 11 {subject}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
    
    return chart_paths


def generate_star_winter_charts(
    star_data=None,
    config=None,
    partner_name="default",
    data_dir=None,
    output_dir="./charts",
    chart_filters=None,
    dev_mode=False
):
    """
    Flask wrapper function to generate STAR Winter charts
    
    Args:
        star_data: List of dicts or DataFrame with STAR data
        config: Config dict
        partner_name: Partner name
        data_dir: Data directory path (if star_data not provided)
        output_dir: Output directory for charts
        chart_filters: Chart filters dict
        dev_mode: Development mode flag
    
    Returns:
        List of chart file paths
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    cfg = config or {}
    if chart_filters:
        cfg['chart_filters'] = chart_filters
    
    hf.DEV_MODE = cfg.get('dev_mode', dev_mode)
    
    class Args:
        def __init__(self):
            self.partner = partner_name
            self.data_dir = data_dir if star_data is None else None
            self.output_dir = output_dir
            self.dev_mode = 'true' if hf.DEV_MODE else 'false'
            self.config = json.dumps(cfg) if cfg else '{}'
    
    args = Args()
    
    old_argv = sys.argv
    try:
        sys.argv = [
            'star_winter.py',
            '--partner', args.partner,
            '--output-dir', args.output_dir,
            '--dev-mode', args.dev_mode,
            '--config', args.config
        ]
        if args.data_dir:
            sys.argv.extend(['--data-dir', args.data_dir])
        
        chart_paths = main(star_data=star_data)
    finally:
        sys.argv = old_argv
    
    return chart_paths


if __name__ == "__main__":
    try:
        chart_paths = main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
