"""
iReady chart generation script - generates charts from ingested iReady data
"""

# Set matplotlib backend to non-interactive before any imports
# This is required when running in Flask/threaded environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI required)

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

# Import utility modules
from .iready_data import (
    load_config_from_args,
    load_iready_data,
    get_scopes,
    prep_iready_for_charts,
    _short_year
)
from .iready_filters import (
    apply_chart_filters,
    should_generate_subject,
    should_generate_student_group,
    should_generate_grade
)
from .iready_chart_utils import (
    draw_stacked_bar,
    draw_score_bar,
    draw_insight_card,
    LABEL_MIN_PCT
)

# Chart tracking for CSV generation
chart_links = []
_chart_tracking_set = set()  # Track by file path to prevent duplicates

def track_chart(chart_name, file_path, scope="district", section=None, chart_data=None):
    """
    Track chart for CSV generation and save chart data if provided
    
    Args:
        chart_name: Name of the chart
        file_path: Path to the chart image file
        scope: Scope of the chart (district, school, etc.)
        section: Section number
        chart_data: Optional dictionary containing chart metrics/data to save as JSON
    """
    global _chart_tracking_set
    
    chart_path = Path(file_path)
    normalized_path = str(chart_path.resolve())
    
    # Check if this chart was already tracked
    if normalized_path in _chart_tracking_set:
        print(f"  ⚠ Skipping duplicate chart: {chart_name}")
        return
    
    # Add to tracking set
    _chart_tracking_set.add(normalized_path)
    
    chart_info = {
        "chart_name": chart_name,
        "scope": scope,
        "section": section,
        "file_path": str(file_path),
        "file_link": f"file://{chart_path.absolute()}"
    }
    
    # Save chart data as JSON if provided
    if chart_data is not None:
        data_path = chart_path.parent / f"{chart_path.stem}_data.json"
        try:
            # Convert any numpy/pandas types to native Python types for JSON serialization
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
# SECTION 0 — i-Ready vs CERS (Predicted vs Actual)
# ---------------------------------------------------------------------

def _prep_section0_iready(df, subject):
    """Prepare data for Section 0: i-Ready vs CERS comparison"""
    d = df.copy()
    
    # Normalize i-Ready placement labels
    if hasattr(hf, "IREADY_LABEL_MAP"):
        d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)
    
    subj = subject.upper()
    
    # Filter for most recent academic year with Spring + CERS data
    valid_years = (
        d.loc[
            (d["testwindow"].astype(str).str.upper() == "SPRING")
            & (d["cers_overall_performanceband"].notna())
        ]["academicyear"]
        .dropna()
        .unique()
    )
    if len(valid_years) == 0:
        return None, None, None
    
    last_year = max(valid_years)
    d = d[
        (d["academicyear"] == last_year)
        & (d["testwindow"].astype(str).str.upper() == "SPRING")
        & (d["subject"].astype(str).str.upper() == subj)
        & (d["cers_overall_performanceband"].notna())
        & (d["domain"] == "Overall")
        & (d["relative_placement"].notna())
        & (d["enrolled"] == "Enrolled")
    ].copy()
    
    if d.empty:
        return None, None, None
    
    placement_col = "relative_placement"
    cers_col = "cers_overall_performanceband"
    
    # Build cross-tab of (placement x CERS band)
    cross = d.groupby([placement_col, cers_col]).size().reset_index(name="n")
    total = cross.groupby(placement_col)["n"].sum().reset_index(name="N_total")
    cross = cross.merge(total, on=placement_col, how="left")
    cross["pct"] = 100 * cross["n"] / cross["N_total"]
    cross_dict = {
        (r[placement_col], r[cers_col]): r["pct"] for _, r in cross.iterrows()
    }
    
    # Compute summary metrics
    iready_mid_above = (
        d[placement_col]
        .eq(hf.IREADY_LABEL_MAP.get("Mid or Above Grade Level", "Mid/Above"))
        .mean()
        * 100
    )
    cers_met_exceed = (
        d[cers_col]
        .isin(["Level 3 - Standard Met", "Level 4 - Standard Exceeded"])
        .mean()
        * 100
    )
    
    metrics = {
        "iready_mid_above": iready_mid_above,
        "cers_met_exceed": cers_met_exceed,
        "delta": iready_mid_above - cers_met_exceed,
        "year": int(last_year),
    }
    
    return cross_dict, metrics, last_year


def _plot_section0_iready(scope_label, folder, output_dir, data_dict, preview=False):
    """Plot Section 0: i-Ready vs CERS comparison"""
    cers_levels = hf.CERS_LEVELS
    placements = hf.IREADY_ORDER
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.8, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Legend for CERS bands
    handles = [
        Patch(facecolor=hf.CERS_LEVEL_COLORS.get(l, "#ccc"), label=l)
        for l in cers_levels
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    
    for i, (subj, (cross_pct, metrics)) in enumerate(data_dict.items()):
        # Stacked bar: % of students in each CERS band by i-Ready placement
        ax_top = fig.add_subplot(gs[0, i])
        bottom = np.zeros(len(placements))
        for lvl in cers_levels:
            vals = [cross_pct.get((p, lvl), 0) for p in placements]
            color = hf.CERS_LEVEL_COLORS.get(lvl, "#ccc")
            ax_top.bar(
                placements,
                vals,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=1,
            )
            for j, v in enumerate(vals):
                if v >= LABEL_MIN_PCT:
                    ax_top.text(
                        j,
                        bottom[j] + v / 2,
                        f"{v:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )
            bottom += np.array(vals)
        ax_top.set_ylim(0, 100)
        ax_top.set_ylabel("% of Students")
        ax_top.set_title(subj, fontsize=14, fontweight="bold")
        ax_top.grid(axis="y", alpha=0.2)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        
        # Bar panel: i-Ready Mid/Above vs CERS Met/Exceed
        ax_mid = fig.add_subplot(gs[1, i])
        bars = ax_mid.bar(
            ["i-Ready Mid/Above", "CERS Met/Exceed"],
            [metrics["iready_mid_above"], metrics["cers_met_exceed"]],
            color=["#00baeb", "#0381a2"],
            edgecolor="white",
            width=0.6,
        )
        for rect, val in zip(
            bars, [metrics["iready_mid_above"], metrics["cers_met_exceed"]]
        ):
            ax_mid.text(
                rect.get_x() + rect.get_width() / 2,
                val + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#434343",
            )
        ax_mid.set_ylim(0, 100)
        ax_mid.set_ylabel("% of Students")
        ax_mid.grid(axis="y", alpha=0.2)
        ax_mid.spines["top"].set_visible(False)
        ax_mid.spines["right"].set_visible(False)
        
        # Insight panel
        ax_bot = fig.add_subplot(gs[2, i])
        ax_bot.axis("off")
        insight_text = (
            f"i-Ready Mid/Above vs CERS Met/Exceed:\n"
            rf"${metrics['iready_mid_above']:.1f}\% - {metrics['cers_met_exceed']:.1f}\% = "
            rf"\mathbf{{{metrics['delta']:+.1f}}}$ pts"
        )
        ax_bot.text(
            0.5,
            0.5,
            insight_text,
            ha="center",
            va="center",
            fontsize=12,
            color="#333",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"
            ),
        )
    
    year = next(iter(data_dict.values()))[1].get("year", "")
    fig.suptitle(
        f"{scope_label} • Spring {year} • i-Ready Placement vs CERS Performance",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    out_path = out_dir / f"{prefix}{scope_label.replace(' ', '_')}_section0_iready_vs_cers.png"
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 0: {out_path}")
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": "Spring",
        "year": year,
        "subjects": list(data_dict.keys()),
        "iready_vs_cers": {
            subj: {
                "iready_mid_above": float(metrics["iready_mid_above"]),
                "cers_met_exceed": float(metrics["cers_met_exceed"]),
                "delta": float(metrics["delta"]),
                "iready_distribution": {level: float(pct) for level, pct in metrics["iready_pct"].items()},
                "cers_distribution": {level: float(pct) for level, pct in metrics["cers_pct"].items()}
            }
            for subj, (_, metrics) in data_dict.items()
        }
    }
    track_chart(f"{prefix}{scope_label.replace(' ', '_')}_section0_iready_vs_cers", out_path, scope=folder, section=0, chart_data=chart_data)
    if preview:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# SECTION 1 — Dual Subject Dashboard
# ---------------------------------------------------------------------

def plot_dual_subject_dashboard(df, scope_label, folder, output_dir, window_filter="Fall", preview=False):
    """Unified function to plot dual subject dashboard"""
    if df.empty or len(df) == 0:
        raise ValueError(f"No data available for {scope_label}")
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    subjects = ["ELA", "Math"]
    pct_dfs, score_dfs, metrics_list, time_orders = [], [], [], []
    
    for subj in subjects:
        pct_df, score_df, metrics, time_order = prep_iready_for_charts(df, subj, window_filter)
        
        if len(time_order) == 0 or len(pct_df) == 0:
            pct_df = pd.DataFrame(columns=["time_label", "relative_placement", "pct", "n", "N_total"])
            score_df = pd.DataFrame(columns=["time_label", "avg_score"])
            time_order = []
            metrics = {}
        
        if len(time_order) > 4:
            time_order = time_order[-4:]
            pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
            score_df = score_df[score_df["time_label"].isin(time_order)].copy()
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
        time_orders.append(time_order)
    
    if all(len(to) == 0 for to in time_orders):
        raise ValueError(f"No time periods found for {scope_label} with window filter '{window_filter}'")
    
    axes = [[fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i]), fig.add_subplot(gs[2, i])] for i in range(2)]
    
    # Panel 1: Stacked bars
    legend_handles = [Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q) for q in hf.IREADY_ORDER]
    for i, subj in enumerate(subjects):
        if len(pct_dfs[i]) > 0 and len(time_orders[i]) > 0:
            draw_stacked_bar(axes[i][0], pct_dfs[i], score_dfs[i], hf.IREADY_ORDER)
        else:
            axes[i][0].text(0.5, 0.5, f"No {subj} data available", ha="center", va="center", fontsize=12)
            axes[i][0].axis("off")
        axes[i][0].set_title(subj, fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.IREADY_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.IREADY_ORDER), frameon=False, fontsize=10)
    
    # Panel 2: Scale scores
    for i in range(2):
        if len(score_dfs[i]) > 0 and len(time_orders[i]) > 0:
            # Build n-map for labels
            n_map = None
            if not pct_dfs[i].empty and "N_total" in pct_dfs[i].columns:
                n_map_df = pct_dfs[i].groupby("time_label")["N_total"].max().reset_index()
                n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
            draw_score_bar(axes[i][1], score_dfs[i], hf.IREADY_ORDER, n_map)
        else:
            axes[i][1].text(0.5, 0.5, "No score data available", ha="center", va="center", fontsize=12)
            axes[i][1].axis("off")
    
    # Panel 3: Insights
    for i in range(2):
        draw_insight_card(axes[i][2], metrics_list[i], subjects[i])
    
    fig.suptitle(f"{scope_label} • {window_filter} Year-to-Year Trends", fontsize=20, fontweight="bold", y=1)
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    out_path = out_dir / f"{prefix}{scope_label.replace(' ', '_')}_section1_{window_filter.lower()}_trends.png"
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Chart saved to: {out_path}")
    
    chart_data = {
        "scope": scope_label,
        "window_filter": window_filter,
        "subjects": subjects,
        "metrics": metrics_list,
        "time_orders": time_orders,
    }
    
    track_chart(f"{prefix}{scope_label.replace(' ', '_')}_section1_{window_filter.lower()}_trends", 
                str(out_path), scope=scope_label, section=1, chart_data=chart_data)
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 2 — Student Group Performance Trends
# ---------------------------------------------------------------------

def _apply_student_group_mask(df_in, group_name, group_def):
    """
    Returns boolean mask for df_in selecting the student group.
    Uses cfg['student_groups'] spec:
      type: "all"                  -> everyone True
      or {column: <col>, in: [...]} -> membership by value match (case-insensitive str compare)
    """
    if group_def.get("type") == "all":
        return pd.Series(True, index=df_in.index)
    
    col = group_def["column"]
    allowed_vals = group_def["in"]
    
    # normalize both sides as lowercase strings
    vals = df_in[col].astype(str).str.strip().str.lower()
    allowed_norm = {str(v).strip().lower() for v in allowed_vals}
    return vals.isin(allowed_norm)


def plot_iready_subject_dashboard_by_group(
    df, scope_label, folder, output_dir, window_filter="Fall",
    group_name=None, group_def=None, cfg=None, preview=False
):
    """
    Same visual layout as the main dashboard but filtered to one student group.
    We also enforce min n >= 12 unique students in the current scope.
    """
    d0 = df.copy()
    
    # Apply group mask first
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    
    if d0.empty:
        print(f"[group {group_name}] no rows after group mask ({scope_label})")
        return None
    
    subjects = ["ELA", "Math"]
    subject_titles = ["ELA", "Math"]
    
    # Aggregate for each subject
    pct_dfs, score_dfs, metrics_list, time_orders, min_ns, n_maps = [], [], [], [], [], []
    
    for subj in subjects:
        # Filter for subject
        if subj == "ELA":
            subj_df = d0[d0["subject"].astype(str).str.contains("ela", case=False, na=False)].copy()
        elif subj == "Math":
            subj_df = d0[d0["subject"].astype(str).str.contains("math", case=False, na=False)].copy()
        else:
            subj_df = d0.copy()
        
        if subj_df.empty:
            pct_dfs.append(None)
            score_dfs.append(None)
            metrics_list.append(None)
            time_orders.append([])
            min_ns.append(0)
            n_maps.append({})
            continue
        
        pct_df, score_df, metrics, time_order = prep_iready_for_charts(
            subj_df, subject_str=subj, window_filter=window_filter
        )
        
        # Restrict to most recent 4 timepoints
        if len(time_order) > 4:
            time_order = time_order[-4:]
            pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
            score_df = score_df[score_df["time_label"].isin(time_order)].copy()
        
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
        time_orders.append(time_order)
        
        # Minimum n >= 12 check
        if pct_df is not None and not pct_df.empty and time_order:
            latest_label = time_order[-1]
            latest_slice = pct_df[pct_df["time_label"] == latest_label]
            if "N_total" in latest_slice.columns:
                latest_n = latest_slice["N_total"].max()
            else:
                latest_n = latest_slice["n"].sum()
            min_ns.append(latest_n if not pd.isna(latest_n) else 0)
        else:
            min_ns.append(0)
        
        # n_map for xticklabels in score panel
        if pct_df is not None and not pct_df.empty:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        else:
            n_map = {}
        n_maps.append(n_map)
    
    # If either panel fails min_n, skip
    if any((n is None or n < 12) for n in min_ns):
        print(f"[group {group_name}] skipped (<12 students in one or both subjects) in {scope_label}")
        return None
    
    # If either panel has no data, skip
    if any((df is None or df.empty) for df in pct_dfs) or any((df is None or df.empty) for df in score_dfs):
        print(f"[group {group_name}] skipped (empty data in one or both subjects) in {scope_label}")
        return None
    
    # Setup subplots: 3 rows x 2 columns (ELA left, Math right)
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    
    # Panel 1: Stacked bar for each subject
    legend_handles = [Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q) for q in hf.IREADY_ORDER]
    
    for i, subj in enumerate(subjects):
        pct_df = pct_dfs[i]
        score_df = score_dfs[i]
        metrics = metrics_list[i]
        time_order = time_orders[i]
        
        if pct_df is not None and not pct_df.empty and len(time_order) > 0:
            draw_stacked_bar(axes[0][i], pct_df, score_df, hf.IREADY_ORDER)
        else:
            axes[0][i].text(0.5, 0.5, f"No {subj} data available", ha="center", va="center", fontsize=12)
            axes[0][i].axis("off")
        axes[0][i].set_title(f"{subject_titles[i]}", fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.IREADY_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.IREADY_ORDER), frameon=False, fontsize=10, handlelength=1.8, handletextpad=0.5, columnspacing=1.1)
    
    # Panel 2: Avg score by subject
    for i in range(2):
        score_df = score_dfs[i]
        n_map = n_maps[i] if i < len(n_maps) else {}
        if score_df is not None and not score_df.empty and len(time_orders[i]) > 0:
            draw_score_bar(axes[1][i], score_df, hf.IREADY_ORDER, n_map)
        else:
            axes[1][i].text(0.5, 0.5, "No score data available", ha="center", va="center", fontsize=12)
            axes[1][i].axis("off")
    
    # Panel 3: Insights by subject
    for i in range(2):
        metrics = metrics_list[i]
        draw_insight_card(axes[2][i], metrics, subjects[i])
    
    # Main title
    fig.suptitle(f"{scope_label} • {group_name} • {window_filter} Year-to-Year Trends",
                fontsize=20, fontweight="bold", y=1)
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    
    safe_group = group_name.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
    order_map = cfg.get("student_group_order", {}) if cfg else {}
    group_order_val = order_map.get(group_name, 99)
    
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_section2_{group_order_val:02d}_{safe_group}_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 2: {out_path}")
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": window_filter,
        "group_name": group_name,
        "subjects": subjects,
        "metrics": metrics_list,
        "time_orders": time_orders,
        "pct_data": [
            {
                "subject": subj,
                "data": pct_df.to_dict('records') if pct_df is not None and not pct_df.empty else []
            }
            for subj, pct_df in zip(subjects, pct_dfs)
        ],
        "score_data": [
            {
                "subject": subj,
                "data": score_df.to_dict('records') if score_df is not None and not score_df.empty else []
            }
            for subj, score_df in zip(subjects, score_dfs)
        ]
    }
    track_chart(out_name, str(out_path), scope=scope_label, section=2, chart_data=chart_data)
    
    return str(out_path)


# ---------------------------------------------------------------------
# SECTION 3 — Overall + Cohort Trends
# ---------------------------------------------------------------------

def _prep_iready_matched_cohort_by_grade(df, subject_str, current_grade, window_filter, cohort_year):
    """Prepare matched cohort data for Section 3"""
    base = df.copy()
    base["academicyear"] = pd.to_numeric(base.get("academicyear"), errors="coerce")
    base["student_grade"] = pd.to_numeric(base.get("student_grade"), errors="coerce")
    
    anchor_year = int(cohort_year or base["academicyear"].max())
    cohort_rows, ordered_labels = [], []
    
    for offset in range(0, 4):
        yr = anchor_year - 3 + offset
        gr = current_grade - 3 + offset
        if gr < 0:
            continue
        
        tmp = base[
            (base["testwindow"].astype(str).str.upper() == window_filter.upper())
            & (base["student_grade"] == gr)
            & (base["academicyear"] == yr)
        ].copy()
        
        subj_norm = subject_str.strip().lower()
        if "math" in subj_norm:
            tmp = tmp[tmp["subject"].astype(str).str.contains("math", case=False, na=False)]
        elif "ela" in subj_norm:
            tmp = tmp[tmp["subject"].astype(str).str.contains("ela", case=False, na=False)]
        
        tmp = tmp[tmp["domain"] == "Overall"]
        tmp = tmp[tmp["relative_placement"].notna()]
        tmp = tmp[tmp["enrolled"] == "Enrolled"]
        
        if hasattr(hf, "IREADY_LABEL_MAP"):
            tmp["relative_placement"] = tmp["relative_placement"].replace(hf.IREADY_LABEL_MAP)
        
        if tmp.empty:
            continue
        
        # Dedupe to latest completion per student
        if "completion_date" in tmp.columns:
            tmp["completion_date"] = pd.to_datetime(tmp["completion_date"], errors="coerce")
            tmp.sort_values(["uniqueidentifier", "completion_date"], inplace=True)
            tmp = tmp.groupby("uniqueidentifier", as_index=False).tail(1)
        
        y_prev, y_curr = str(yr - 1)[-2:], str(yr)[-2:]
        label = f"Gr {int(gr)} • {window_filter} {y_prev}-{y_curr}"
        tmp["cohort_label"] = label
        cohort_rows.append(tmp)
        ordered_labels.append(label)
    
    if not cohort_rows:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    cohort_df = pd.concat(cohort_rows, ignore_index=True)
    cohort_df["label"] = cohort_df["cohort_label"]
    
    # Percent by placement
    counts = cohort_df.groupby(["label", "relative_placement"]).size().rename("n").reset_index()
    totals = cohort_df.groupby("label").size().rename("N_total").reset_index()
    pct_df = counts.merge(totals, on="label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Ensure all combinations exist
    all_idx = pd.MultiIndex.from_product(
        [pct_df["label"].unique(), hf.IREADY_ORDER],
        names=["label", "relative_placement"],
    )
    pct_df = pct_df.set_index(["label", "relative_placement"]).reindex(all_idx).reset_index()
    pct_df[["pct", "n"]] = pct_df[["pct", "n"]].fillna(0)
    pct_df["N_total"] = pct_df.groupby("label")["N_total"].transform(lambda s: s.ffill().bfill())
    
    # Score averages
    score_df = (
        cohort_df[["label", "scale_score"]]
        .dropna(subset=["scale_score"])
        .groupby("label")["scale_score"]
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
            # Convert Categorical to string for reliable comparison
            tlabel_str = str(tlabel)
            mask = (pct_df["time_label"].astype(str) == tlabel_str) & (pct_df["relative_placement"].isin(buckets))
            return pct_df[mask]["pct"].sum()
        
        hi_now = pct_for(hf.IREADY_HIGH_GROUP, t_curr)
        lo_now = pct_for(hf.IREADY_LOW_GROUP, t_curr)
        hi_delta = hi_now - pct_for(hf.IREADY_HIGH_GROUP, t_prev)
        lo_delta = lo_now - pct_for(hf.IREADY_LOW_GROUP, t_prev)
        
        # Calculate Mid/Above delta separately (for insight card display)
        high_now = pct_for(["Mid/Above"], t_curr)
        high_prev = pct_for(["Mid/Above"], t_prev)
        high_delta = high_now - high_prev
        
        score_now = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0])
        score_prev = float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])
        
        metrics = dict(
            t_prev=t_prev,
            t_curr=t_curr,
            hi_now=hi_now,
            hi_delta=hi_delta,  # Delta for IREADY_HIGH_GROUP (Early On + Mid/Above)
            high_now=high_now,
            high_delta=high_delta,  # Delta for Mid/Above only
            lo_now=lo_now,
            lo_delta=lo_delta,
            score_now=score_now,
            score_delta=score_now - score_prev,
        )
    else:
        metrics = {k: None for k in ["t_prev", "t_curr", "hi_now", "hi_delta", "high_now", "high_delta", "lo_now", "lo_delta", "score_now", "score_delta"]}
    
    return pct_df, score_df, metrics, ordered_labels


def plot_iready_blended_dashboard(
    df, scope_label, folder, output_dir, subject_str, current_grade,
    window_filter="Fall", cohort_year=None, cfg=None, preview=False
):
    """Dual-facet dashboard showing Overall vs Cohort Trends"""
    # Prep left (overall) and right (cohort)
    d = df.copy()
    d["student_grade"] = pd.to_numeric(d["student_grade"], errors="coerce")
    d = d[d["student_grade"] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    subj_norm = subject_str.strip().lower()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)]
    elif "ela" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("ela", case=False, na=False)]
    
    d = d[d["domain"] == "Overall"]
    d = d[d["relative_placement"].notna()]
    d = d[d["enrolled"] == "Enrolled"]
    
    pct_df_left, score_df_left, metrics_left, _ = prep_iready_for_charts(
        d, subject_str=subject_str, window_filter=window_filter
    )
    
    pct_df_right, score_df_right, metrics_right, cohort_labels = _prep_iready_matched_cohort_by_grade(
        df, subject_str, current_grade, window_filter, cohort_year
    )
    
    # Check if we have data
    if (pct_df_left.empty or score_df_left.empty) and (pct_df_right.empty or score_df_right.empty):
        print(f"[Section 3] No data for {scope_label} - Grade {current_grade} - {subject_str}")
        return None
    
    # Create figure with 2 columns (Overall left, Cohort right)
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    axes_left = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
    axes_right = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])]
    
    # Left side: Overall trends
    legend_handles = [Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q) for q in hf.IREADY_ORDER]
    
    if not pct_df_left.empty and not score_df_left.empty:
        draw_stacked_bar(axes_left[0], pct_df_left, score_df_left, hf.IREADY_ORDER)
        n_map_left = None
        if "N_total" in pct_df_left.columns:
            n_map_df = pct_df_left.groupby("time_label")["N_total"].max().reset_index()
            n_map_left = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(axes_left[1], score_df_left, hf.IREADY_ORDER, n_map_left)
        draw_insight_card(axes_left[2], metrics_left, subject_str)
    else:
        for ax in axes_left:
            ax.text(0.5, 0.5, "No overall data", ha="center", va="center", fontsize=12)
            ax.axis("off")
    
    axes_left[0].set_title("Overall Trends", fontsize=14, fontweight="bold", y=1.1)
    
    # Right side: Cohort trends
    if not pct_df_right.empty and not score_df_right.empty:
        draw_stacked_bar(axes_right[0], pct_df_right, score_df_right, hf.IREADY_ORDER)
        n_map_right = None
        if "N_total" in pct_df_right.columns:
            n_map_df = pct_df_right.groupby("time_label")["N_total"].max().reset_index()
            n_map_right = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(axes_right[1], score_df_right, hf.IREADY_ORDER, n_map_right)
        draw_insight_card(axes_right[2], metrics_right, subject_str)
    else:
        for ax in axes_right:
            ax.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=12)
            ax.axis("off")
    
    axes_right[0].set_title("Cohort Trends", fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.IREADY_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.IREADY_ORDER), frameon=False, fontsize=10)
    
    fig.suptitle(f"{scope_label} • Grade {current_grade} • {subject_str} • {window_filter} Trends",
                fontsize=20, fontweight="bold", y=1)
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_section3_grade{current_grade}_{safe_subj}_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 3: {out_path}")
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": window_filter,
        "grade": current_grade,
        "subject": subject_str,
        "metrics": metrics_left,
        "pct_data": {
            "overall": pct_df_left.to_dict('records') if not pct_df_left.empty else []
        },
        "score_data": {
            "overall": score_df_left.to_dict('records') if not score_df_left.empty else []
        }
    }
    
    if not pct_df_right.empty and not score_df_right.empty:
        chart_data["cohort_metrics"] = metrics_right
        chart_data["pct_data"]["cohort"] = pct_df_right.to_dict('records')
        chart_data["score_data"]["cohort"] = score_df_right.to_dict('records')
    
    track_chart(out_name, str(out_path), scope=scope_label, section=3, chart_data=chart_data)
    
    return str(out_path)


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main(iready_data=None):
    """
    Main function to generate iReady charts
    
    Args:
        iready_data: Optional list of dicts or DataFrame with iReady data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate iReady charts')
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
    if iready_data is not None:
        iready_base = load_iready_data(iready_data=iready_data)
    else:
        if not args.data_dir:
            raise ValueError("Either iready_data must be provided or --data-dir must be specified")
        iready_base = load_iready_data(data_dir=args.data_dir)
    
    # Match old flow: Each section receives unfiltered scope data and filters internally
    # Only scope filtering (district vs school) happens before sections
    # This matches iready2.py behavior where filtering is progressive within each section
    
    # Get selected quarters (for determining which charts to generate)
    selected_quarters = ["Fall"]
    if chart_filters and chart_filters.get("quarters") and len(chart_filters["quarters"]) > 0:
        selected_quarters = chart_filters["quarters"]
    
    # Get scopes from unfiltered data (all sections filter internally)
    scopes = get_scopes(iready_base, cfg)
    
    chart_paths = []
    
    # Section 0: i-Ready vs CERS
    print("\n[Section 0] Generating i-Ready vs CERS...")
    for scope_df, scope_label, folder in scopes:
        try:
            payload = {}
            for subj in ["ELA", "Math"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                cross, metrics, year = _prep_section0_iready(scope_df, subj)
                if cross is None:
                    continue
                payload[subj] = (cross, metrics)
            if payload:
                _plot_section0_iready(scope_label, folder, args.output_dir, payload, preview=hf.DEV_MODE)
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            continue
    
    # Section 1: Fall Performance Trends
    print("\n[Section 1] Generating Fall Performance Trends...")
    for quarter in selected_quarters:
        for scope_df, scope_label, folder in scopes:
            try:
                chart_path = plot_dual_subject_dashboard(
                    scope_df,
                    scope_label,
                    folder,
                    args.output_dir,
                    window_filter=quarter,
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error generating chart for {scope_label} ({quarter}): {e}")
                continue
    
    # Section 2: Student Group Performance Trends
    print("\n[Section 2] Generating Student Group Performance Trends...")
    student_groups_cfg = cfg.get("student_groups", {})
    race_ethnicity_cfg = cfg.get("race_ethnicity", {})
    group_order = cfg.get("student_group_order", {})
    
    for scope_df, scope_label, folder in scopes:
        # Process regular student groups
        for group_name, group_def in sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)):
            # Skip "All Students" - it's handled in Section 1
            if group_def.get("type") == "all":
                continue
            # Only generate charts for selected student groups
            should_gen = should_generate_student_group(group_name, chart_filters)
            if not should_gen:
                continue
            print(f"  [Generate] {group_name}")
            for quarter in selected_quarters:
                try:
                    chart_path = plot_iready_subject_dashboard_by_group(
                        scope_df.copy(), scope_label, folder, args.output_dir,
                        window_filter=quarter, group_name=group_name, group_def=group_def,
                        cfg=cfg, preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"  Error generating Section 2 chart for {scope_label} - {group_name} ({quarter}): {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
        
        # Process race/ethnicity groups if race filter is specified
        race_filters = chart_filters.get("race", []) if chart_filters else []
        if race_filters and race_ethnicity_cfg:
            for race_name in race_filters:
                race_def = race_ethnicity_cfg.get(race_name)
                if not race_def:
                    print(f"  [Skip] Race group '{race_name}' not found in race_ethnicity config")
                    continue
                
                # Create a combined group_def for race
                combined_group_def = {
                    "column": race_def.get("column"),
                    "in": race_def.get("values", race_def.get("in", [])),
                    "type": "race"
                }
                
                print(f"  [Generate] Race group: {race_name}")
                for quarter in selected_quarters:
                    try:
                        chart_path = plot_iready_subject_dashboard_by_group(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            window_filter=quarter, group_name=race_name, group_def=combined_group_def,
                            cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                    except Exception as e:
                        print(f"  Error generating Section 2 chart for {scope_label} - {race_name} ({quarter}): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Section 3: Overall + Cohort Trends
    print("\n[Section 3] Generating Overall + Cohort Trends...")
    
    def _run_scope_section3(scope_df, scope_label, folder):
        # Match old flow: Receive unfiltered scope data, filter internally
        scope_df = scope_df.copy()
        scope_df["academicyear"] = pd.to_numeric(scope_df["academicyear"], errors="coerce")
        scope_df["student_grade"] = pd.to_numeric(scope_df["student_grade"], errors="coerce")
        
        # Normalize grades (K -> 0)
        def normalize_grade_val(grade_val):
            if pd.isna(grade_val):
                return None
            grade_str = str(grade_val).strip().upper()
            if grade_str == "K" or grade_str == "KINDERGARTEN":
                return 0
            try:
                return int(float(grade_str))
            except:
                return None
        
        scope_df["grade_normalized"] = scope_df["student_grade"].apply(normalize_grade_val)
        
        if scope_df["academicyear"].notna().any():
            anchor_year = int(scope_df["academicyear"].max())
        else:
            anchor_year = None
        
        # Get unique grades from scope data (for determining which charts to generate)
        # Apply grade filter here if specified (for chart generation decision only)
        unique_grades = sorted([g for g in scope_df["grade_normalized"].dropna().unique() if g is not None])
        
        # Filter grades if chart_filters specifies grades (for determining which charts to generate)
        if chart_filters and chart_filters.get("grades") and len(chart_filters["grades"]) > 0:
            unique_grades = [g for g in unique_grades if g in chart_filters["grades"]]
        
        print(f"  [Section 3] Found {len(unique_grades)} grade(s) in filtered data: {unique_grades}")
        
        # Use consolidated charts if more than 3 grades
        use_consolidated = len(unique_grades) > 3
        
        if use_consolidated:
            # Generate consolidated chart with all grades arranged horizontally
            print(f"  [Section 3] Using consolidated horizontal layout for {len(unique_grades)} grades")
            # For now, generate individual charts but we'll add consolidated function later
            # TODO: Create plot_iready_consolidated_blended_dashboard function
            pass
        
        for g in unique_grades:
            # Check if grade exists in data (for chart generation decision)
            grade_check = scope_df[scope_df["grade_normalized"] == g].copy()
            if grade_check.empty:
                continue
            
            subjects_in_data = set(grade_check["subject"].dropna().astype(str).str.lower())
            for subject_str in ["ELA", "Math"]:
                # Map subject string to filter check
                subject_filter_name = "ELA" if subject_str == "ELA" else "Math"
                if not should_generate_subject(subject_filter_name, chart_filters):
                    continue
                
                # Check if subject exists in data
                subject_match = False
                if subject_str == "ELA":
                    subject_match = any("ela" in s for s in subjects_in_data)
                elif subject_str == "Math":
                    subject_match = any("math" in s for s in subjects_in_data)
                
                if subject_match:
                    for quarter in selected_quarters:
                        try:
                            print(f"  [Section 3] Generating chart for {scope_label} - Grade {g} - {subject_str} - {quarter}")
                            # Pass unfiltered scope_df - function filters internally (matches old flow)
                            chart_path = plot_iready_blended_dashboard(
                                scope_df.copy(), scope_label, folder, args.output_dir,
                                subject_str=subject_str, current_grade=int(g),
                                window_filter=quarter, cohort_year=anchor_year,
                                cfg=cfg, preview=hf.DEV_MODE
                            )
                            if chart_path:
                                chart_paths.append(chart_path)
                        except Exception as e:
                            print(f"  [Section 3] Error generating chart for {scope_label} - Grade {g} - {subject_str} ({quarter}): {e}")
                            if hf.DEV_MODE:
                                import traceback
                                traceback.print_exc()
                            continue
    
    # Use unfiltered scopes for Section 3 (matches old flow - each section filters internally)
    for scope_df, scope_label, folder in scopes:
        _run_scope_section3(scope_df.copy(), scope_label, folder)
    
    # Build chart_paths from tracked charts if chart_paths is incomplete
    if chart_links and len(chart_links) > len(chart_paths):
        print(f"[Chart Tracking] Found {len(chart_links)} tracked charts but only {len(chart_paths)} in chart_paths")
        print(f"[Chart Tracking] Building chart_paths from tracked charts...")
        chart_paths = [str(Path(chart['file_path']).absolute()) for chart in chart_links]
        print(f"[Chart Tracking] Built {len(chart_paths)} chart paths from tracked charts")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chart_paths = []
    for chart_path in chart_paths:
        normalized_path = str(Path(chart_path).resolve())
        if normalized_path not in seen:
            seen.add(normalized_path)
            unique_chart_paths.append(chart_path)
    
    if len(chart_paths) != len(unique_chart_paths):
        print(f"[Deduplication] Removed {len(chart_paths) - len(unique_chart_paths)} duplicate chart(s)")
        chart_paths = unique_chart_paths
    
    print(f"\n✅ Generated {len(chart_paths)} iReady charts")
    return chart_paths


def generate_iready_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    iready_data: list = None
) -> list:
    """
    Generate iReady charts (wrapper function for Flask backend)
    
    Args:
        partner_name: Partner name
        output_dir: Output directory for charts
        config: Partner configuration dict
        chart_filters: Chart filters dict
        data_dir: Data directory path (used only if iready_data is None)
        iready_data: Optional list of dicts with iReady data (preferred over CSV loading)
    
    Returns:
        List of chart file paths
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    cfg = config or {}
    if chart_filters:
        cfg['chart_filters'] = chart_filters
    
    hf.DEV_MODE = cfg.get('dev_mode', False)
    
    class Args:
        def __init__(self):
            self.partner = partner_name
            self.data_dir = data_dir if iready_data is None else None
            self.output_dir = output_dir
            self.dev_mode = 'true' if hf.DEV_MODE else 'false'
            self.config = json.dumps(cfg) if cfg else '{}'
    
    args = Args()
    
    old_argv = sys.argv
    try:
        sys.argv = [
            'iready_charts.py',
            '--partner', args.partner,
            '--output-dir', args.output_dir,
            '--dev-mode', args.dev_mode,
            '--config', args.config
        ]
        if args.data_dir:
            sys.argv.extend(['--data-dir', args.data_dir])
        
        chart_paths = main(iready_data=iready_data)
    finally:
        sys.argv = old_argv
    
    return chart_paths


if __name__ == "__main__":
    try:
        chart_paths = main()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
