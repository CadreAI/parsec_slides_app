"""
STAR Fall chart generation script - generates charts from ingested STAR data for Fall window
"""

# Set matplotlib backend to non-interactive before any imports
import matplotlib
matplotlib.use('Agg')

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

# Import utility modules
from .star_data import (
    load_config_from_args,
    load_star_data,
    get_scopes,
    prep_star_for_charts,
    _short_year
)
from .star_filters import (
    apply_chart_filters,
    should_generate_subject,
    should_generate_student_group,
    should_generate_grade
)
from .star_chart_utils import (
    draw_stacked_bar,
    draw_score_bar,
    draw_insight_card,
    LABEL_MIN_PCT
)

# Chart tracking for CSV generation
chart_links = []
_chart_tracking_set = set()

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
# SECTION 0 — Predicted vs Actual CAASPP (Spring)
# ---------------------------------------------------------------------

def _prep_section0_star(df, subject):
    """Prepare data for Section 0: STAR predicted vs actual CAASPP"""
    d = df.copy()
    d = d[d["testwindow"].str.upper() == "SPRING"].copy()
    
    if d.empty or d["academicyear"].dropna().empty:
        return None, None, None, None
    
    subj = subject.lower()
    if "math" in subj:
        d = d[d["activity_type"].str.contains("math", case=False, na=False)]
    else:
        d = d[d["activity_type"].str.contains("read", case=False, na=False)]
    
    if d.empty or d["academicyear"].dropna().empty:
        return None, None, None, None
    
    d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
    if d["academicyear"].dropna().empty:
        return None, None, None, None
    
    target_year = int(d["academicyear"].max())
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

def _plot_section0_star(scope_label, folder, subj_payload, output_dir, preview=False):
    """Render Section 0 chart: STAR predicted vs actual CAASPP"""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
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
                width=0.35,
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
                width=0.35,
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
        bar_ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
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
        pct_ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
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
        f"{scope_label} • Spring {first_metrics['year']} Prediction Accuracy",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{scope_label}_section0_pred_vs_actual.png"
    out_path = out_dir / out_name
    
    try:
        hf._save_and_render(fig, out_path, dev_mode=preview)
        
        # Verify file was actually saved
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise IOError(f"Chart file was not created or is empty: {out_path}")
        
        # Prepare chart data for saving
        chart_data = {
            "scope": scope_label,
            "window_filter": "Fall",
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
        track_chart(f"Section 0: Predicted vs Actual", out_path, scope=scope_label, section=0, chart_data=chart_data)
        print(f"Saved Section 0: {out_path}")
    except Exception as e:
        print(f"ERROR: Failed to save Section 0 chart for {scope_label}: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to be caught by outer try-except

# ---------------------------------------------------------------------
# SECTION 1 - Fall Performance Trends (Dual Subject Dashboard)
# ---------------------------------------------------------------------

def plot_star_dual_subject_dashboard(
    df, scope_label, folder, output_dir, window_filter="Fall", preview=False
):
    """Faceted dashboard showing both Math and Reading for a given scope"""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    subjects = ["reading", "math"]
    titles = ["Reading", "Math"]
    
    legend_handles = [
        Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q)
        for q in hf.STAR_ORDER
    ]
    
    pct_dfs, score_dfs, metrics_list, time_orders, n_maps = [], [], [], [], []
    
    for i, (activity_type_filter, title) in enumerate(zip(subjects, titles)):
        pct_df, score_df, metrics, time_order = prep_star_for_charts(
            df, subject_str=activity_type_filter, window_filter=window_filter
        )
        
        # Limit to most recent 4 timepoints
        if len(time_order) > 4:
            time_order = time_order[-4:]
            pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
            score_df = score_df[score_df["time_label"].isin(time_order)].copy()
        
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
        time_orders.append(time_order)
        
        # Build n_map for x-axis labels
        if pct_df is not None and not pct_df.empty:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        else:
            n_map = {}
        n_maps.append(n_map)
    
    # Plot panels
    for i, (pct_df, score_df, metrics, time_order, n_map) in enumerate(
        zip(pct_dfs, score_dfs, metrics_list, time_orders, n_maps)
    ):
        ax1 = fig.add_subplot(gs[0, i])
        if pct_df is not None and not pct_df.empty:
            draw_stacked_bar(ax1, pct_df, score_df, hf.STAR_ORDER)
        else:
            ax1.text(0.5, 0.5, f"No {titles[i]} data", ha="center", va="center", fontsize=12)
            ax1.axis("off")
        ax1.set_title(f"{titles[i]}", fontsize=14, fontweight="bold", pad=30)
        
        ax2 = fig.add_subplot(gs[1, i])
        if score_df is not None and not score_df.empty:
            draw_score_bar(ax2, score_df, hf.STAR_ORDER, n_map)
        else:
            ax2.text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
            ax2.axis("off")
        ax2.set_title("Avg Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
        
        ax3 = fig.add_subplot(gs[2, i])
        draw_insight_card(ax3, metrics, titles[i])
    
    fig.legend(
        handles=legend_handles,
        labels=hf.STAR_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(hf.STAR_ORDER),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    
    fig.suptitle(
        f"{scope_label} • {window_filter} Year-to-Year Trends",
        fontsize=20,
        fontweight="bold",
        y=1,
    )
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_name = f"{scope_label}_section1_star_{window_filter.lower()}_trends.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
    chart_data = {
        "metrics": {titles[i]: metrics_list[i] for i in range(len(titles))},
        "time_orders": {titles[i]: time_orders[i] for i in range(len(titles))}
    }
    track_chart(f"Section 1: {window_filter} Trends", out_path, scope=scope_label, section=1, chart_data=chart_data)
    print(f"Saved Section 1: {out_path}")
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 2 — Student Group Performance Trends
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

def plot_star_subject_dashboard_by_group(
    df, scope_label, folder, output_dir, window_filter="Fall",
    group_name=None, group_def=None, cfg=None, preview=False
):
    """Same layout as main dashboard but filtered to one student group"""
    d0 = df.copy()
    
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    
    if d0.empty:
        print(f"[group {group_name}] no rows after group mask ({scope_label})")
        return None
    
    subjects = ["Reading", "Mathematics"]
    subject_titles = ["Reading", "Mathematics"]
    
    pct_dfs, score_dfs, metrics_list, time_orders, min_ns, n_maps = [], [], [], [], [], []
    
    for subj in subjects:
        if subj == "Reading":
            subj_df = d0[
                d0["activity_type"].astype(str).str.contains("reading", case=False, na=False)
            ].copy()
        elif subj == "Mathematics":
            subj_df = d0[
                d0["activity_type"].astype(str).str.contains("math", case=False, na=False)
            ].copy()
        else:
            subj_df = d0.copy()
        
        subj_df["testwindow"] = subj_df["testwindow"].astype(str).str.strip().str.lower()
        win = str(window_filter).strip().lower()
        subj_df = subj_df[subj_df["testwindow"] == win]
        
        if subj_df.empty:
            pct_dfs.append(None)
            score_dfs.append(None)
            metrics_list.append(None)
            time_orders.append([])
            min_ns.append(0)
            n_maps.append({})
            continue
        
        pct_df, score_df, metrics, time_order = prep_star_for_charts(
            subj_df, subject_str=subj, window_filter=window_filter
        )
        
        if len(time_order) > 4:
            time_order = time_order[-4:]
            pct_df = pct_df[pct_df["time_label"].isin(time_order)].copy()
            score_df = score_df[score_df["time_label"].isin(time_order)].copy()
        
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
        time_orders.append(time_order)
        
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
        
        if pct_df is not None and not pct_df.empty:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        else:
            n_map = {}
        n_maps.append(n_map)
    
    if any((n is None or n < 12) for n in min_ns):
        print(f"[group {group_name}] skipped (<12 students) in {scope_label}")
        return None
    
    if any((df is None or df.empty) for df in pct_dfs) or any((df is None or df.empty) for df in score_dfs):
        print(f"[group {group_name}] skipped (empty data) in {scope_label}")
        return None
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    
    legend_handles = [
        Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q)
        for q in hf.STAR_ORDER
    ]
    
    for i, subj in enumerate(subjects):
        pct_df = pct_dfs[i]
        score_df = score_dfs[i]
        metrics = metrics_list[i]
        time_order = time_orders[i]
        n_map = n_maps[i]
        
        if pct_df is not None and not pct_df.empty:
            stack_df = (
                pct_df.pivot(
                    index="time_label", columns="state_benchmark_achievement", values="pct"
                )
                .reindex(columns=hf.STAR_ORDER)
                .fillna(0)
            )
            x_labels = stack_df.index.tolist()
            x = np.arange(len(x_labels))
            cumulative = np.zeros(len(stack_df))
            
            for cat in hf.STAR_ORDER:
                vals = stack_df[cat].to_numpy()
                bars = axes[0][i].bar(
                    x, vals, bottom=cumulative,
                    color=hf.STAR_COLORS[cat],
                    edgecolor="white", linewidth=1.2
                )
                for idx, rect in enumerate(bars):
                    h = vals[idx]
                    if h >= LABEL_MIN_PCT:
                        color = "#434343" if cat == "2 - Standard Nearly Met" else "white"
                        axes[0][i].text(
                            rect.get_x() + rect.get_width() / 2,
                            cumulative[idx] + h / 2,
                            f"{h:.1f}%",
                            ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color
                        )
                cumulative += vals
            
            axes[0][i].set_ylim(0, 100)
            axes[0][i].set_ylabel("% of Students")
            axes[0][i].set_xticks(x)
            axes[0][i].set_xticklabels(x_labels)
            axes[0][i].grid(axis="y", alpha=0.2)
            axes[0][i].spines["top"].set_visible(False)
            axes[0][i].spines["right"].set_visible(False)
        else:
            axes[0][i].text(0.5, 0.5, f"No {subj} data", ha="center", va="center", fontsize=12)
            axes[0][i].axis("off")
        axes[0][i].set_title(subject_titles[i], fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(
        handles=legend_handles, labels=hf.STAR_ORDER,
        loc="upper center", bbox_to_anchor=(0.5, 0.92),
        ncol=len(hf.STAR_ORDER), frameon=False, fontsize=9,
        handlelength=1.8, handletextpad=0.5, columnspacing=1.1
    )
    
    for i in range(2):
        score_df = score_dfs[i]
        n_map = n_maps[i] if i < len(n_maps) else {}
        if score_df is not None and not score_df.empty:
            draw_score_bar(axes[1][i], score_df, hf.STAR_ORDER, n_map)
        else:
            axes[1][i].text(0.5, 0.5, "No score data", ha="center", va="center", fontsize=12)
            axes[1][i].axis("off")
        axes[1][i].set_title("Average Unified Scale Score", fontsize=8, fontweight="bold", pad=10)
    
    for i in range(2):
        metrics = metrics_list[i]
        pct_df = pct_dfs[i]
        axes[2][i].axis("off")
        if metrics and metrics.get("t_prev"):
            t_prev = metrics["t_prev"]
            t_curr = metrics["t_curr"]
            
            def _bucket_delta(bucket, pct_df):
                curr = pct_df.loc[
                    (pct_df["time_label"] == t_curr) &
                    (pct_df["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
                prev = pct_df.loc[
                    (pct_df["time_label"] == t_prev) &
                    (pct_df["state_benchmark_achievement"] == bucket), "pct"
                ].sum()
                return curr - prev
            
            if pct_df is not None and not pct_df.empty:
                high_delta = _bucket_delta("4 - Standard Exceeded", pct_df)
                hi_delta = sum(
                    _bucket_delta(b, pct_df)
                    for b in ["4 - Standard Exceeded", "3 - Standard Met"]
                )
                lo_delta = _bucket_delta("1 - Standard Not Met", pct_df)
                score_delta = metrics["score_delta"]
                
                insight_lines = [
                    "Comparison of current and prior year",
                    rf"$\Delta$ Exceed: $\mathbf{{{high_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ Meet or Exceed: $\mathbf{{{hi_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ Not Met: $\mathbf{{{lo_delta:+.1f}}}$ ppts",
                    rf"$\Delta$ Avg Unified Scale Score: $\mathbf{{{score_delta:+.1f}}}$ pts",
                ]
            else:
                insight_lines = ["(No pct_df for insight calculation)"]
        else:
            insight_lines = ["Not enough history for change insights"]
        
        axes[2][i].text(
            0.5, 0.5, "\n".join(insight_lines),
            fontsize=11, fontweight="normal", color="#434343",
            ha="center", va="center", wrap=True, usetex=False,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8)
        )
    
    fig.suptitle(
        f"{scope_label} • {group_name} • {window_filter} Year-to-Year Trends",
        fontsize=20, fontweight="bold", y=1
    )
    
    out_dir_path = Path(output_dir) / folder
    out_dir_path.mkdir(parents=True, exist_ok=True)
    order_map = cfg.get("student_group_order", {}) if cfg else {}
    group_order_val = order_map.get(group_name, 99)
    safe_group = group_name.replace(" ", "_").replace("/", "_")
    out_name = f"{scope_label.replace(' ', '_')}_section2_{group_order_val:02d}_{safe_group}_{window_filter.lower()}_trends.png"
    out_path = out_dir_path / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    
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
    track_chart(f"Section 2: {group_name}", out_path, scope=scope_label, section=2, chart_data=chart_data)
    print(f"Saved Section 2: {out_path}")
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 3 — Overall + Cohort Trends
# ---------------------------------------------------------------------

def _prep_star_matched_cohort_by_grade(df, subject_str, current_grade, window_filter, cohort_year):
    """Prepare matched cohort data for Section 3 - tracks same students across grades"""
    base = df.copy()
    base["academicyear"] = pd.to_numeric(base.get("academicyear"), errors="coerce")
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade"]:
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
        
        subj_norm = subject_str.strip().lower()
        if "math" in subj_norm:
            tmp = tmp[tmp["subject"].astype(str).str.contains("math", case=False, na=False)]
        elif "read" in subj_norm:
            tmp = tmp[tmp["subject"].astype(str).str.contains("read", case=False, na=False)]
        
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


def plot_star_blended_dashboard(
    df, scope_label, folder, output_dir, subject_str, current_grade,
    window_filter="Fall", cohort_year=None, cfg=None, preview=False
):
    """Dual-facet dashboard showing Overall vs Cohort Trends"""
    # Prep left (overall) and right (cohort)
    d = df.copy()
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade"]:
        if col in d.columns:
            grade_col = col
            break
    
    if grade_col is None:
        print(f"[Section 3] No grade column found for {scope_label}")
        return None
    
    d[grade_col] = pd.to_numeric(d[grade_col], errors="coerce")
    d = d[d[grade_col] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    subj_norm = subject_str.strip().lower()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)]
    elif "read" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("read", case=False, na=False)]
    
    d = d[d["state_benchmark_achievement"].notna()]
    
    pct_df_left, score_df_left, metrics_left, _ = prep_star_for_charts(
        d, subject_str=subject_str, window_filter=window_filter
    )
    
    pct_df_right, score_df_right, metrics_right, cohort_labels = _prep_star_matched_cohort_by_grade(
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
# SECTION 4 — Overall Growth Trends by Site
# ---------------------------------------------------------------------

def plot_star_growth_by_site(
    df, scope_label, folder, output_dir, subject_str, window_filter="Fall", cfg=None, preview=False
):
    """Show growth trends broken down by school/site"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    subj_norm = subject_str.strip().lower()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)]
    elif "read" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("read", case=False, na=False)]
    
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
            school_data[school] = {
                "pct_df": pct_df,
                "score_df": score_df,
                "metrics": metrics,
                "time_order": time_order
            }
    
    if not school_data:
        print(f"[Section 4] No valid school data for {scope_label}")
        return None
    
    # Create figure - one row per school
    n_schools = len(school_data)
    fig = plt.figure(figsize=(16, 4 * n_schools), dpi=300)
    gs = fig.add_gridspec(nrows=n_schools, ncols=2, height_ratios=[1] * n_schools)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
    
    for idx, (school, data) in enumerate(sorted(school_data.items())):
        pct_df = data["pct_df"]
        score_df = data["score_df"]
        metrics = data["metrics"]
        
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
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_section4_{safe_subj}_{window_filter.lower()}_growth_by_site.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 4: {out_path}")
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": window_filter,
        "subject": subject_str,
        "school_data": {
            school: {
                "pct_data": data["pct_df"].to_dict('records') if not data["pct_df"].empty else [],
                "score_data": data["score_df"].to_dict('records') if not data["score_df"].empty else [],
                "metrics": data["metrics"],
                "time_order": data["time_order"]
            }
            for school, data in school_data.items()
        }
    }
    track_chart(out_name, str(out_path), scope=scope_label, section=4, chart_data=chart_data)
    
    return str(out_path)


# ---------------------------------------------------------------------
# SECTION 5 — STAR SGP Growth: Grade Trend + Backward Cohort
# ---------------------------------------------------------------------

def _prep_star_sgp_data(df, subject_str, current_grade, window_filter):
    """Prepare SGP (Student Growth Percentile) data for Section 5"""
    d = df.copy()
    
    # Try different grade column names
    grade_col = None
    for col in ["studentgrade", "student_grade", "grade"]:
        if col in d.columns:
            grade_col = col
            break
    
    if grade_col is None:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    d[grade_col] = pd.to_numeric(d[grade_col], errors="coerce")
    d = d[d[grade_col] == current_grade]
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()]
    
    subj_norm = subject_str.strip().lower()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)]
    elif "read" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("read", case=False, na=False)]
    
    # Check for SGP columns
    sgp_col = None
    for col in ["current_sgp", "sgp", "student_growth_percentile", "current_sgp_vector"]:
        if col in d.columns:
            sgp_col = col
            break
    
    if sgp_col is None:
        print(f"[Section 5] No SGP column found, using unified_scale as proxy")
        sgp_col = "unified_scale"
    
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
        .mean()
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


def plot_star_sgp_growth(
    df, scope_label, folder, output_dir, subject_str, current_grade,
    window_filter="Fall", cfg=None, preview=False
):
    """Show SGP growth trends by grade and backward cohort"""
    # Grade trend (current grade over time)
    sgp_df_grade, _, metrics_grade, time_order = _prep_star_sgp_data(
        df, subject_str, current_grade, window_filter
    )
    
    # Backward cohort (same students tracked backward)
    cohort_df, _, metrics_cohort, cohort_labels = _prep_star_matched_cohort_by_grade(
        df, subject_str, current_grade, window_filter, None
    )
    
    if sgp_df_grade.empty and cohort_df.empty:
        print(f"[Section 5] No SGP data for {scope_label} - Grade {current_grade} - {subject_str}")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Top left: Grade trend (SGP over time)
    ax1 = fig.add_subplot(gs[0, 0])
    if not sgp_df_grade.empty:
        x_labels = sgp_df_grade["time_label"].tolist()
        x = np.arange(len(x_labels))
        y = sgp_df_grade["avg_sgp"].tolist()
        
        ax1.plot(x, y, marker="o", linewidth=2, markersize=8, color="#2E86AB")
        ax1.fill_between(x, y, alpha=0.2, color="#2E86AB")
        
        # Add value labels
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax1.text(xi, yi + 2, f"{yi:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, rotation=45, ha="right")
        ax1.set_ylabel("Average SGP", fontsize=11, fontweight="bold")
        ax1.set_title("Grade Trend: SGP Over Time", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        
        # Add n-counts
        if "N_total" in sgp_df_grade.columns:
            n_map = dict(zip(sgp_df_grade["time_label"].astype(str), sgp_df_grade["N_total"]))
            for i, label in enumerate(x_labels):
                n_val = n_map.get(str(label), 0)
                ax1.text(i, ax1.get_ylim()[0] + 2, f"n={n_val}", ha="center", fontsize=8, style="italic")
    else:
        ax1.text(0.5, 0.5, "No grade trend data", ha="center", va="center", fontsize=12)
        ax1.axis("off")
    
    # Top right: Backward cohort (same students tracked backward)
    ax2 = fig.add_subplot(gs[0, 1])
    if not cohort_df.empty:
        draw_stacked_bar(ax2, cohort_df, pd.DataFrame(), hf.STAR_ORDER)
        ax2.set_title("Backward Cohort: Same Students Over Time", fontsize=14, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=12)
        ax2.axis("off")
    
    # Bottom: Insight cards
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    if metrics_grade and metrics_grade.get("sgp_delta") is not None:
        lines = [
            "Grade Trend Insights:",
            rf"$\Delta$ Avg SGP: $\mathbf{{{metrics_grade['sgp_delta']:+.1f}}}$ pts",
        ]
        ax3.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))
    else:
        ax3.text(0.5, 0.5, "Not enough history for insights", ha="center", va="center", fontsize=11)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    if metrics_cohort and metrics_cohort.get("hi_delta") is not None:
        lines = [
            "Cohort Insights:",
            rf"$\Delta$ Meet/Exceed: $\mathbf{{{metrics_cohort['hi_delta']:+.1f}}}$ ppts",
            rf"$\Delta$ Avg Score: $\mathbf{{{metrics_cohort['score_delta']:+.1f}}}$ pts",
        ]
        ax4.text(0.5, 0.5, "\n".join(lines), ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))
    else:
        ax4.text(0.5, 0.5, "Not enough cohort history", ha="center", va="center", fontsize=11)
    
    fig.suptitle(f"{scope_label} • Grade {current_grade} • {subject_str} • {window_filter} SGP Growth",
                fontsize=20, fontweight="bold", y=0.98)
    
    # Save
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
    
    safe_subj = subject_str.replace(" ", "_").lower()
    out_name = f"{prefix}{scope_label.replace(' ', '_')}_section5_grade{current_grade}_{safe_subj}_{window_filter.lower()}_sgp_growth.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 5: {out_path}")
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "window_filter": window_filter,
        "grade": current_grade,
        "subject": subject_str,
        "grade_metrics": metrics_grade,
        "cohort_metrics": metrics_cohort,
        "sgp_data": {
            "grade_trend": sgp_df_grade.to_dict('records') if not sgp_df_grade.empty else [],
            "cohort_trend": cohort_df.to_dict('records') if not cohort_df.empty else []
        },
        "time_order": time_order,
        "cohort_labels": cohort_labels
    }
    track_chart(out_name, str(out_path), scope=scope_label, section=5, chart_data=chart_data)
    
    return str(out_path)


# ---------------------------------------------------------------------
# CONSOLIDATED CHARTS - Reduce chart count by combining dimensions
# ---------------------------------------------------------------------

def plot_star_consolidated_cohort_all_grades(
    df, scope_label, folder, output_dir, subject_str, window_filter="Fall",
    cohort_year=None, selected_grades=None, cfg=None, preview=False
):
    """Consolidated cohort chart showing 3 grades per chart"""
    if selected_grades is None:
        selected_grades = list(range(3, 9))
    
    # Prepare data for each grade
    grade_data = {}
    for grade in selected_grades:
        pct_df, score_df, metrics, labels = _prep_star_matched_cohort_by_grade(
            df, subject_str, grade, window_filter, cohort_year
        )
        if not pct_df.empty:
            pct_df["grade"] = grade
            score_df["grade"] = grade
            grade_data[grade] = {
                "pct_df": pct_df,
                "score_df": score_df,
                "metrics": metrics
            }
    
    if not grade_data:
        print(f"[Consolidated Section 3] No cohort data for {scope_label} - {subject_str}")
        return None
    
    # Generate one chart per grade (no grouping)
    sorted_grades = sorted(grade_data.keys())
    chart_paths = []
    
    # Create one chart per grade
    for grade in sorted_grades:
        # Single grade chart - use standard layout
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 0.6])
        fig.subplots_adjust(wspace=0.3)
        
        legend_handles = [Patch(facecolor=hf.STAR_COLORS[q], edgecolor="none", label=q) for q in hf.STAR_ORDER]
        
        # Process single grade
        data = grade_data[grade]
        pct_df = data["pct_df"]
        score_df = data["score_df"]
        metrics = data["metrics"]
        
        # Standard layout: stacked bar on left, score bar on right
        ax1 = fig.add_subplot(gs[0, 0])
        draw_stacked_bar(ax1, pct_df, score_df, hf.STAR_ORDER)
        ax1.set_title(f"Grade {grade} - Cohort Trends", fontsize=12, fontweight="bold")
        
        ax2 = fig.add_subplot(gs[0, 1])
        n_map = None
        if "N_total" in pct_df.columns:
            n_map_df = pct_df.groupby("time_label")["N_total"].max().reset_index()
            n_map = dict(zip(n_map_df["time_label"].astype(str), n_map_df["N_total"]))
        draw_score_bar(ax2, score_df, hf.STAR_ORDER, n_map)
        ax2.set_title(f"Grade {grade} Avg Score", fontsize=10, fontweight="bold")
        
        fig.legend(handles=legend_handles, labels=hf.STAR_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                  ncol=len(hf.STAR_ORDER), frameon=False, fontsize=9)
        
        # Create title for single grade
        fig.suptitle(f"{scope_label} • {subject_str} • {window_filter} Cohort Trends (Grade {grade})",
                    fontsize=18, fontweight="bold", y=0.995)
        
        # Save
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        
        safe_subj = subject_str.replace(" ", "_").lower()
        out_name = f"{prefix}{scope_label.replace(' ', '_')}_section3_grade_{grade}_{safe_subj}_{window_filter.lower()}_cohort.png"
        out_path = out_dir / out_name
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"Saved Section 3 Grade {grade}: {out_path}")
        
        # Prepare chart_data for consolidated chart
        chart_data = {
            "scope": scope_label,
            "window_filter": window_filter,
            "subject": subject_str,
            "grades": [grade],
            "grade_data": {
                grade: {
                    "metrics": grade_data[grade]["metrics"],
                    "pct_data": grade_data[grade]["pct_df"].to_dict('records') if not grade_data[grade]["pct_df"].empty else [],
                    "score_data": grade_data[grade]["score_df"].to_dict('records') if not grade_data[grade]["score_df"].empty else []
                }
            }
        }
        
        track_chart(out_name, str(out_path), scope=scope_label, section=3, chart_data=chart_data)
        chart_paths.append(str(out_path))
    
    return chart_paths[0] if len(chart_paths) == 1 else chart_paths


def plot_star_consolidated_sgp_all_grades(
    df, scope_label, folder, output_dir, subject_str, window_filter="Fall",
    selected_grades=None, cfg=None, preview=False
):
    """Consolidated SGP chart showing 3 grades per chart"""
    if selected_grades is None:
        selected_grades = list(range(3, 9))
    
    # Prepare SGP data for each grade
    grade_data = {}
    for grade in selected_grades:
        sgp_df, _, metrics, _ = _prep_star_sgp_data(df, subject_str, grade, window_filter)
        if not sgp_df.empty:
            sgp_df["grade"] = grade
            grade_data[grade] = {
                "sgp_df": sgp_df,
                "metrics": metrics
            }
    
    if not grade_data:
        print(f"[Consolidated Section 5] No SGP data for {scope_label} - {subject_str}")
        return None
    
    # Generate one chart per grade (no grouping)
    sorted_grades = sorted(grade_data.keys())
    chart_paths = []
    
    # Create one chart per grade
    for grade in sorted_grades:
        # Single grade chart - use standard layout
        fig = plt.figure(figsize=(16, 6), dpi=300)
        gs = fig.add_gridspec(nrows=1, ncols=1)
        fig.subplots_adjust()
        
        # Process single grade
        data = grade_data[grade]
        sgp_df = data["sgp_df"]
        metrics = data["metrics"]
        
        ax = fig.add_subplot(gs[0, 0])
        
        x_labels = sgp_df["time_label"].tolist()
        x = np.arange(len(x_labels))
        y = sgp_df["avg_sgp"].tolist()
        
        ax.plot(x, y, marker="o", linewidth=2, markersize=8, color="#2E86AB", label=f"Grade {grade}")
        ax.fill_between(x, y, alpha=0.2, color="#2E86AB")
        
        # Add value labels with clearer formatting
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi + 2, f"{yi:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Average SGP", fontsize=11, fontweight="bold")
        ax.set_title(f"Grade {grade} SGP", fontsize=11, fontweight="bold", pad=8)
        
        # Add latest SGP value prominently
        if len(y) > 0:
            latest_sgp = y[-1]
            ax.text(0.5, 0.95, f"Latest: {latest_sgp:.1f}", 
                   transform=ax.transAxes, ha="center", va="top",
                   fontsize=10, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add n-counts
        if "N_total" in sgp_df.columns:
            n_map = dict(zip(sgp_df["time_label"].astype(str), sgp_df["N_total"]))
            for i, label in enumerate(x_labels):
                n_val = n_map.get(str(label), 0)
                ax.text(i, ax.get_ylim()[0] + 2, f"n={n_val}", ha="center", fontsize=8, style="italic")
        
        # Create title for single grade
        fig.suptitle(f"{scope_label} • {subject_str} • {window_filter} SGP Growth (Grade {grade})",
                    fontsize=18, fontweight="bold", y=0.995)
        
        # Save
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        
        safe_subj = subject_str.replace(" ", "_").lower()
        out_name = f"{prefix}{scope_label.replace(' ', '_')}_section5_grade_{grade}_{safe_subj}_{window_filter.lower()}_sgp.png"
        out_path = out_dir / out_name
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"Saved Section 5 Grade {grade}: {out_path}")
        
        # Prepare chart_data for consolidated SGP chart
        chart_data = {
            "scope": scope_label,
            "window_filter": window_filter,
            "subject": subject_str,
            "grades": [grade],
            "sgp_data": {
                grade: {
                    "metrics": data["metrics"],
                    "sgp_trend": sgp_df.to_dict('records') if not sgp_df.empty else []
                }
            }
        }
        
        track_chart(out_name, str(out_path), scope=scope_label, section=5, chart_data=chart_data)
        chart_paths.append(str(out_path))
    
    return chart_paths[0] if len(chart_paths) == 1 else chart_paths


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main(star_data=None):
    """
    Main function to generate STAR Fall charts
    
    Args:
        star_data: Optional list of dicts or DataFrame with STAR data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate STAR charts')
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
    
    # Always use Fall for this module
    selected_quarters = ["Fall"]
    
    # Get scopes
    scopes = get_scopes(star_base, cfg)
    
    chart_paths = []
    
    # Section 0: Predicted vs Actual CAASPP
    print("\n[Section 0] Generating Predicted vs Actual CAASPP...")
    for scope_df, scope_label, folder in scopes:
        try:
            payload = {}
            for subj in ["Reading", "Mathematics"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                proj, act, metrics, year = _prep_section0_star(scope_df, subj)
                if proj is None:
                    continue
                payload[subj] = {"proj_pct": proj, "act_pct": act, "metrics": metrics}
            if payload:
                _plot_section0_star(scope_label, folder, payload, args.output_dir, preview=hf.DEV_MODE)
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 1: Fall Performance Trends
    print("\n[Section 1] Generating Fall Performance Trends...")
    # If multiple quarters selected, generate one chart per scope (shows all quarters if possible)
    # Otherwise generate per quarter
    if len(selected_quarters) > 1:
        # Generate one chart per scope (will show latest quarter or combine)
        for scope_df, scope_label, folder in scopes:
            try:
                # Use the first quarter as primary, but chart will show trends across time
                chart_path = plot_star_dual_subject_dashboard(
                    scope_df,
                    scope_label,
                    folder,
                    args.output_dir,
                    window_filter=selected_quarters[0],  # Use first quarter
                    preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"Error generating chart for {scope_label}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
                continue
    else:
        # Single quarter - generate normally
        for quarter in selected_quarters:
            for scope_df, scope_label, folder in scopes:
                try:
                    chart_path = plot_star_dual_subject_dashboard(
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
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 2: Student Group Performance Trends
    print("\n[Section 2] Generating Student Group Performance Trends...")
    student_groups_cfg = cfg.get("student_groups", {})
    group_order = cfg.get("student_group_order", {})
    
    # Limit to most important student groups if too many
    max_groups = chart_filters.get("max_student_groups", 10)  # Default limit
    sorted_groups = sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99))
    groups_to_plot = sorted_groups[:max_groups]
    
    for scope_df, scope_label, folder in scopes:
        for group_name, group_def in groups_to_plot:
            if group_def.get("type") == "all":
                continue
            if not should_generate_student_group(group_name, chart_filters):
                continue
            print(f"  [Generate] {group_name}")
            # Use first quarter only if multiple quarters selected (reduces charts)
            quarters_to_use = selected_quarters[:1] if len(selected_quarters) > 1 else selected_quarters
            for quarter in quarters_to_use:
                try:
                    chart_path = plot_star_subject_dashboard_by_group(
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
    
    # Section 3: Overall + Cohort Trends
    print("\n[Section 3] Generating Overall + Cohort Trends...")
    selected_grades = chart_filters.get("grades", [])
    if not selected_grades:
        # Default: grades 3-8
        selected_grades = list(range(3, 9))
    
    anchor_year = int(star_base["academicyear"].max()) if "academicyear" in star_base.columns else None
    
    # Always generate individual charts per grade (no consolidated charts)
    
    for scope_df, scope_label, folder in scopes:
        for subj in ["Reading", "Mathematics"]:
            if not should_generate_subject(subj, chart_filters):
                continue
            # Use first quarter only if multiple quarters selected (reduces charts significantly)
            quarters_to_use = selected_quarters[:1] if len(selected_quarters) > 1 else selected_quarters
            for quarter in quarters_to_use:
                # Always generate individual charts per grade (no consolidated charts for cohort trends)
                for grade in selected_grades:
                    if not should_generate_grade(grade, chart_filters):
                        continue
                    try:
                        chart_path = plot_star_blended_dashboard(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            subject_str=subj, current_grade=grade,
                            window_filter=quarter, cohort_year=anchor_year,
                            cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                    except Exception as e:
                        print(f"  Error generating Section 3 chart for {scope_label} - Grade {grade} - {subj} ({quarter}): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Section 4: Overall Growth Trends by Site
    print("\n[Section 4] Generating Overall Growth Trends by Site...")
    for scope_df, scope_label, folder in scopes:
        # Only generate for district scope (shows all schools)
        if folder == "_district":
            for subj in ["Reading", "Mathematics"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                # Use first quarter only if multiple quarters selected
                quarters_to_use = selected_quarters[:1] if len(selected_quarters) > 1 else selected_quarters
                for quarter in quarters_to_use:
                    try:
                        chart_path = plot_star_growth_by_site(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            subject_str=subj, window_filter=quarter,
                            cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                    except Exception as e:
                        print(f"  Error generating Section 4 chart for {scope_label} - {subj} ({quarter}): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Section 5: STAR SGP Growth - Grade Trend + Backward Cohort
    print("\n[Section 5] Generating STAR SGP Growth...")
    # Always generate individual charts per grade (no consolidated charts for SGP growth)
    
    for scope_df, scope_label, folder in scopes:
        for subj in ["Reading", "Mathematics"]:
            if not should_generate_subject(subj, chart_filters):
                continue
            # Use first quarter only if multiple quarters selected
            quarters_to_use = selected_quarters[:1] if len(selected_quarters) > 1 else selected_quarters
            for quarter in quarters_to_use:
                # Always generate individual charts per grade
                for grade in selected_grades:
                    if not should_generate_grade(grade, chart_filters):
                        continue
                    try:
                        chart_path = plot_star_sgp_growth(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            subject_str=subj, current_grade=grade,
                            window_filter=quarter, cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                    except Exception as e:
                        print(f"  Error generating Section 5 chart for {scope_label} - Grade {grade} - {subj} ({quarter}): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Build chart_paths from tracked charts (merge both sources)
    # Some sections manually append to chart_paths, others use track_chart() which adds to chart_links
    chart_paths_from_links = [str(Path(chart['file_path']).absolute()) for chart in chart_links]
    
    print(f"\n[Chart Tracking] Manual chart_paths: {len(chart_paths)}, track_chart() chart_links: {len(chart_links)}")
    
    # Merge both sources
    all_chart_paths = chart_paths + chart_paths_from_links
    
    # Remove duplicates
    seen = set()
    unique_chart_paths = []
    for chart_path in all_chart_paths:
        normalized_path = str(Path(chart_path).resolve())
        if normalized_path not in seen:
            seen.add(normalized_path)
            unique_chart_paths.append(chart_path)
    
    print(f"\n✅ Generated {len(unique_chart_paths)} unique STAR Fall charts")
    if len(unique_chart_paths) == 0:
        print(f"⚠️  WARNING: No charts were generated!")
        print(f"   - Data rows after filtering: {star_base.shape[0]:,}")
        print(f"   - Scopes found: {len(scopes)}")
        print(f"   - Selected quarters: {selected_quarters}")
        print(f"   - Chart filters: {chart_filters}")
    
    return unique_chart_paths


def generate_star_fall_charts(
    star_data=None,
    config=None,
    partner_name="default",
    data_dir=None,
    output_dir="./charts",
    chart_filters=None,
    dev_mode=False
):
    """
    Flask wrapper function to generate STAR Fall charts
    
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
            'star_fall.py',
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

