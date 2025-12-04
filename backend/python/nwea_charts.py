"""
NWEA chart generation script - generates charts from ingested NWEA data
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
import matplotlib as mpl
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines
import helper_functions as hf

# Import utility modules
from nwea_data import (
    load_config_from_args,
    load_nwea_data,
    get_scopes,
    prep_nwea_for_charts,
    _short_year
)
from nwea_filters import (
    apply_chart_filters,
    should_generate_subject,
    should_generate_student_group,
    should_generate_grade
)
from nwea_chart_utils import (
    draw_stacked_bar,
    draw_score_bar,
    draw_insight_card,
    LABEL_MIN_PCT
)

# Chart tracking for CSV generation
chart_links = []

def track_chart(chart_name, file_path, scope="district", section=None):
    """Track chart for CSV generation"""
    chart_links.append({
        "chart_name": chart_name,
        "scope": scope,
        "section": section,
        "file_path": str(file_path),
        "file_link": f"file://{Path(file_path).absolute()}"
    })

# ---------------------------------------------------------------------
# Unified Dashboard Plotter
# ---------------------------------------------------------------------

def plot_dual_subject_dashboard(df, scope_label, folder, output_dir, window_filter="Fall", preview=False):
    """Unified function to plot dual subject dashboard"""
    # Check if dataframe is empty
    if df.empty or len(df) == 0:
        raise ValueError(f"No data available for {scope_label}")
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    subjects = ["Reading", "Mathematics"]
    pct_dfs, score_dfs, metrics_list, time_orders = [], [], [], []
    
    for subj in subjects:
        pct_df, score_df, metrics, time_order = prep_nwea_for_charts(df, subj, window_filter)
        
        # Check if we have any data for this subject
        if len(time_order) == 0 or len(pct_df) == 0:
            # Create empty dataframes with proper structure
            pct_df = pd.DataFrame(columns=["time_label", "achievementquintile", "pct", "n", "N_total"])
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
    
    # Check if we have any data at all
    if all(len(to) == 0 for to in time_orders):
        raise ValueError(f"No time periods found for {scope_label} with window filter '{window_filter}'")
    
    # axes structure: axes[column][row] where column=0,1 (Reading/Math) and row=0,1,2 (bars/scores/insights)
    axes = [[fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i]), fig.add_subplot(gs[2, i])] for i in range(2)]
    
    # Panel 1: Stacked bars (row 0)
    legend_handles = [Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q) for q in hf.NWEA_ORDER]
    for i, subj in enumerate(subjects):
        if len(pct_dfs[i]) > 0 and len(time_orders[i]) > 0:
            draw_stacked_bar(axes[i][0], pct_dfs[i], score_dfs[i], hf.NWEA_ORDER)
        else:
            axes[i][0].text(0.5, 0.5, f"No {subj} data available", ha="center", va="center", fontsize=12)
            axes[i][0].axis("off")
        axes[i][0].set_title(subj, fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.NWEA_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.NWEA_ORDER), frameon=False, fontsize=10)
    
    # Panel 2: RIT scores (row 1)
    for i in range(2):
        if len(score_dfs[i]) > 0 and len(time_orders[i]) > 0:
            draw_score_bar(axes[i][1], score_dfs[i], hf.NWEA_ORDER)
        else:
            axes[i][1].text(0.5, 0.5, "No score data available", ha="center", va="center", fontsize=12)
            axes[i][1].axis("off")
    
    # Panel 3: Insights (row 2)
    for i in range(2):
        draw_insight_card(axes[i][2], metrics_list[i], subjects[i])
    
    fig.suptitle(f"{scope_label} • {window_filter} Year-to-Year Trends", fontsize=20, fontweight="bold", y=1)
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scope_label.replace(' ', '_')}_section1_{window_filter.lower()}_trends.png"
    
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Chart saved to: {out_path}")
    
    return str(out_path)

# ---------------------------------------------------------------------
# SECTION 0 — Predicted vs Actual CAASPP (Spring)
# ---------------------------------------------------------------------

def _prep_section0(df, subject):
    """Prepare data for Section 0: Predicted vs Actual CAASPP"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == "SPRING"].copy()
    
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None
    
    subj = subject.lower()
    if "math" in subj:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)]
    else:
        d = d[d["course"].astype(str).str.contains("read", case=False, na=False)]
    
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None
    
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    if d["year"].dropna().empty:
        return None, None, None, None
    
    target_year = int(d["year"].max())
    d = d[d["year"] == target_year].copy()
    
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        d = d.sort_values("teststartdate").drop_duplicates("uniqueidentifier", keep="last")
    
    d = d.dropna(subset=["projectedproficiencylevel2", "cers_overall_performanceband"])
    if d.empty:
        return None, None, None, target_year
    
    proj_order = sorted(d["projectedproficiencylevel2"].unique())
    act_order = hf.CERS_LEVELS
    
    def pct_table(col, order):
        return (d.groupby(col).size().reindex(order, fill_value=0).pipe(lambda s: 100 * s / s.sum()))
    
    proj_pct = pct_table("projectedproficiencylevel2", proj_order)
    act_pct = pct_table("cers_overall_performanceband", act_order)
    
    def pct_met_exceed(series, met_levels):
        return 100 * d[d[series].isin(met_levels)].shape[0] / d.shape[0]
    
    proj_met = pct_met_exceed("projectedproficiencylevel2", ["Level 3 - Standard Met", "Level 4 - Standard Exceeded"])
    act_met = pct_met_exceed("cers_overall_performanceband", ["Level 3 - Standard Met", "Level 4 - Standard Exceeded"])
    delta = proj_met - act_met
    
    metrics = {
        "proj_met": proj_met, "act_met": act_met, "delta": delta, "year": target_year,
        "proj_order": proj_order, "act_order": act_order, "proj_pct": proj_pct, "act_pct": act_pct,
    }
    return proj_pct, act_pct, metrics, target_year

def _plot_section0_dual(scope_label, folder, output_dir, subj_payload, preview=False):
    """Render Section 0: Predicted vs Actual CAASPP dual-panel chart"""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
    subjects = [s for s in ["Reading", "Mathematics"] if s in subj_payload]
    titles = {"Reading": "Reading", "Mathematics": "Math"}
    
    first_metrics = next(iter(subj_payload.values()))["metrics"]
    handles = [Patch(facecolor=hf.CERS_LEVEL_COLORS[l], edgecolor="none", label=l) for l in first_metrics["act_order"]]
    
    fig.legend(handles=handles, labels=first_metrics["act_order"], loc="upper center",
              bbox_to_anchor=(0.5, 0.93), ncol=len(first_metrics["act_order"]), frameon=False,
              fontsize=9, handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    
    for i, subject in enumerate(subjects):
        proj_pct = subj_payload[subject]["proj_pct"]
        act_pct = subj_payload[subject]["act_pct"]
        metrics = subj_payload[subject]["metrics"]
        
        bar_ax = fig.add_subplot(gs[0, i])
        cumulative = 0
        for level in metrics["proj_order"]:
            val = float(proj_pct.get(level, 0))
            idx = metrics["proj_order"].index(level)
            mapped_level = metrics["act_order"][idx] if idx < len(metrics["act_order"]) else metrics["act_order"][-1]
            col = hf.CERS_LEVEL_COLORS.get(mapped_level, "#cccccc")
            bars = bar_ax.bar(-0.2, val, bottom=cumulative, width=0.35, color=col, alpha=0.6,
                             edgecolor="#434343", linewidth=1.2, linestyle="--")
            rect = bars.patches[0]
            if val >= LABEL_MIN_PCT:
                bar_ax.text(rect.get_x() + rect.get_width() / 2.0, cumulative + val / 2.0, f"{val:.1f}%",
                           ha="center", va="center", fontsize=8, fontweight="bold", color="#434343")
            cumulative += val
        
        cumulative = 0
        for level in metrics["act_order"]:
            val = float(act_pct.get(level, 0))
            col = hf.CERS_LEVEL_COLORS.get(level, "#cccccc")
            bars = bar_ax.bar(0.2, val, bottom=cumulative, width=0.35, color=col,
                             edgecolor="white", linewidth=1.2)
            rect = bars.patches[0]
            if val >= LABEL_MIN_PCT:
                txt_color = "#434343" if "Nearly" in level else "white"
                bar_ax.text(rect.get_x() + rect.get_width() / 2.0, cumulative + val / 2.0, f"{val:.1f}%",
                           ha="center", va="center", fontsize=8, fontweight="bold", color=txt_color)
            cumulative += val
        
        bar_ax.set_xticks([-0.2, 0.2])
        bar_ax.set_xticklabels(["Predicted", "Actual"])
        bar_ax.set_ylim(0, 100)
        bar_ax.set_ylabel("% of Students")
        bar_ax.set_title(titles[subject], fontsize=14, fontweight="bold", pad=30)
        bar_ax.grid(axis="y", alpha=0.5)
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        
        pct_ax = fig.add_subplot(gs[1, i])
        pct_ax.bar("Pred Met/Exc", metrics["proj_met"], color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
                   alpha=0.6, edgecolor="#434343", linewidth=1.2, linestyle="--")
        pct_ax.bar("Actual Met/Exc", metrics["act_met"], color=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"],
                   alpha=1.0, edgecolor=hf.CERS_LEVEL_COLORS["Level 4 - Standard Exceeded"], linewidth=1.2)
        for x, v in zip([0, 1], [metrics["proj_met"], metrics["act_met"]]):
            pct_ax.text(x, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#434343")
        pct_ax.set_ylim(0, 100)
        pct_ax.set_ylabel("% Met/Exc")
        pct_ax.grid(axis="y", alpha=0.2)
        pct_ax.spines["top"].set_visible(False)
        pct_ax.spines["right"].set_visible(False)
        
        ax3 = fig.add_subplot(gs[2, i])
        ax3.axis("off")
        pred = float(metrics["proj_met"])
        act = float(metrics["act_met"])
        delta = pred - act
        insight_text = (r"Predicted vs Actual Met/Exceed:" + "\n" +
                       rf"${pred:.1f}\% - {act:.1f}\% = \mathbf{{{pred - act:+.1f}}}$ pts")
        ax3.text(0.5, 0.5, insight_text, fontsize=12, ha="center", va="center", color="#434343",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=1.0))
    
    fig.suptitle(f"{scope_label} • Spring {first_metrics['year']} Prediction Accuracy",
                fontsize=20, fontweight="bold", y=1.02)
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{scope_label.replace(' ', '_')}-section0_pred_vs_actual_{folder}.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 0: {str(out_path.absolute())}")
    track_chart(out_name, out_path, scope=folder, section=0)
    if preview:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# SECTION 2 — Student Group Performance Trends
# ---------------------------------------------------------------------

def _apply_student_group_mask(df_in, group_name, group_def):
    """Returns boolean mask for student group selection"""
    if group_def.get("type") == "all":
        return pd.Series(True, index=df_in.index)
    col = group_def["column"]
    allowed_vals = group_def["in"]
    vals = df_in[col].astype(str).str.strip().str.lower()
    allowed_norm = {str(v).strip().lower() for v in allowed_vals}
    return vals.isin(allowed_norm)

def plot_nwea_subject_dashboard_by_group(df, subject_str, window_filter, group_name, group_def,
                                         output_dir, cfg, figsize=(16, 9), school_raw=None, scope_label=None, preview=False):
    """Plot dashboard filtered to one student group"""
    d0 = df.copy()
    school_display = hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    title_label = cfg.get("district_name", ["District (All Students)"])[0] if not school_display else school_display
    
    subjects = ["Reading", "Mathematics"]
    subject_titles = ["Reading", "Mathematics"]
    
    mask = _apply_student_group_mask(d0, group_name, group_def)
    d0 = d0[mask].copy()
    
    if d0.empty:
        print(f"[group {group_name}] no rows after group mask ({school_raw or 'district'})")
        return
    
    pct_dfs, score_dfs, metrics_list, time_orders, min_ns = [], [], [], [], []
    
    for subj in subjects:
        if subj == "Reading":
            subj_df = d0[d0["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
        elif subj == "Mathematics":
            subj_df = d0[d0["course"].astype(str).str.contains("math", case=False, na=False)].copy()
        else:
            subj_df = d0.copy()
        
        if subj_df.empty:
            pct_dfs.append(None)
            score_dfs.append(None)
            metrics_list.append(None)
            time_orders.append([])
            min_ns.append(0)
            continue
        
        pct_df, score_df, metrics, time_order = prep_nwea_for_charts(subj_df, subject_str=subj, window_filter=window_filter)
        
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
    
    if any((n is None or n < 12) for n in min_ns):
        print(f"[group {group_name}] skipped (<12 students in one or both subjects) in {title_label}")
        return
    
    if any((df is None or df.empty) for df in pct_dfs) or any((df is None or df.empty) for df in score_dfs):
        print(f"[group {group_name}] skipped (empty data in one or both subjects) in {title_label}")
        return
    
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axes = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]]
    
    legend_handles = [Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q) for q in hf.NWEA_ORDER]
    
    for i, subj in enumerate(subjects):
        pct_df = pct_dfs[i]
        score_df = score_dfs[i]
        metrics = metrics_list[i]
        time_order = time_orders[i]
        
        stack_df = (pct_df.pivot(index="time_label", columns="achievementquintile", values="pct")
                   .reindex(columns=hf.NWEA_ORDER).fillna(0))
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        ax1 = axes[0][i]
        cumulative = np.zeros(len(stack_df))
        
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax1.bar(x, band_vals, bottom=cumulative, color=hf.NWEA_COLORS[cat],
                          edgecolor="white", linewidth=1.2)
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    label_color = "white" if cat in ["High", "HiAvg", "Low"] else "#434343"
                    ax1.text(rect.get_x() + rect.get_width() / 2, bottom_before + h / 2, f"{h:.2f}%",
                            ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
            cumulative += band_vals
        
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("% of Students")
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels)
        ax1.grid(axis="y", alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_title(f"{subject_titles[i]}", fontsize=14, fontweight="bold", y=1.1)
    
    fig.legend(handles=legend_handles, labels=hf.NWEA_ORDER, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=len(hf.NWEA_ORDER), frameon=False, fontsize=10, handlelength=1.8, handletextpad=0.5, columnspacing=1.1)
    
    for i, subj in enumerate(subjects):
        score_df = score_dfs[i]
        pct_df = pct_dfs[i]
        ax2 = axes[1][i]
        rit_x = np.arange(len(score_df["time_label"]))
        rit_vals = score_df["avg_score"].to_numpy()
        rit_bars = ax2.bar(rit_x, rit_vals, color=hf.default_quintile_colors[4], edgecolor="white", linewidth=1.2)
        for rect, v in zip(rit_bars, rit_vals):
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#434343")
        
        if "N_total" in pct_df.columns:
            n_map = pct_df.groupby("time_label", observed=False)["N_total"].max().reset_index().rename(columns={"N_total": "n"})
        else:
            n_map = pd.DataFrame(columns=["time_label", "n"])
        
        if not n_map.empty:
            label_map = {row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                        for _, row in n_map.iterrows() if not pd.isna(row["n"])}
            x_labels = [label_map.get(lbl, str(lbl)) for lbl in score_df["time_label"]]
        else:
            x_labels = score_df["time_label"].astype(str).tolist()
        
        ax2.set_ylabel("Avg RIT")
        ax2.set_xticks(rit_x)
        ax2.set_xticklabels(x_labels)
        ax2.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
        ax2.grid(axis="y", alpha=0.2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
    
    for i, subj in enumerate(subjects):
        metrics = metrics_list[i]
        pct_df = pct_dfs[i]
        time_order = time_orders[i]
        ax3 = axes[2][i]
        ax3.axis("off")
        
        if metrics and metrics.get("t_prev"):
            t_prev = metrics["t_prev"]
            t_curr = metrics["t_curr"]
            
            def _pct_for_bucket(bucket_name, tlabel):
                return pct_df[(pct_df["time_label"] == tlabel) & (pct_df["achievementquintile"] == bucket_name)]["pct"].sum()
            
            high_now = _pct_for_bucket("High", t_curr)
            high_prev = _pct_for_bucket("High", t_prev)
            high_delta = high_now - high_prev
            hi_delta = metrics["hi_delta"]
            lo_delta = metrics["lo_delta"]
            score_delta = metrics["score_delta"]
            
            title_line = "Change calculations from " + f"{t_prev} to {t_curr}:\n"
            line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
            line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
            line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
            line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
            insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
        else:
            insight_lines = ["Not enough history for change insights"]
        
        ax3.text(0.5, 0.5, "\n".join(insight_lines), fontsize=11, fontweight="medium", color="#333333",
                ha="center", va="center", wrap=True, usetex=False,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.suptitle(f"{title_label} • {group_name} • {window_filter} Year-to-Year Trends",
                fontsize=20, fontweight="bold", y=1)
    
    charts_dir = Path(output_dir)
    folder_name = "_district" if school_raw is None else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    order_map = cfg.get("student_group_order", {})
    group_order_val = order_map.get(group_name, 99)
    safe_group = group_name.replace(" ", "_").replace("/", "_")
    out_name = f"{scope_label.replace(' ', '_')}_section2_{group_order_val:02d}_{safe_group}_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved Section 2: {str(out_path.absolute())}")
    track_chart(out_name, out_path, scope=folder_name, section=2)
    if preview:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# Model Functions (Section 6)
# ---------------------------------------------------------------------

_CASP_BAND_ORDER = ["Level 1 - Standard Not Met", "Level 2 - Standard Nearly Met", 
                     "Level 3 - Standard Met", "Level 4 - Standard Exceeded"]

def filter_fall_course_grades(df, subject):
    """Filter for fall tests, specific grades, and valid CAASPP data"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == "FALL"]
    if "math" in subject.lower():
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)]
    else:
        d = d[d["course"].astype(str).str.contains("read", case=False, na=False)]
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    d = d[d["grade"].isin([3, 4, 5, 6, 7, 8, 11])]
    d = d[d["cers_overall_performanceband"].notna()]
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    return d

def train_model(df, subject):
    """Train RandomForest model for CAASPP prediction"""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("Warning: sklearn not available, skipping model training")
        return None
    
    d0 = filter_fall_course_grades(df, subject)
    train = d0[d0["year"] <= 2024].copy()
    if train.empty:
        return None
    
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)
    clf.fit(train[["testritscore"]], train["cers_overall_performanceband"].astype(str))
    return clf

def predict_2025(df, subject, clf):
    """Predict 2025 CAASPP performance"""
    d0 = filter_fall_course_grades(df, subject)
    val = d0[d0["year"] == 2025].copy()
    if val.empty:
        return None, None, None
    
    y_true = val["cers_overall_performanceband"].astype(str)
    y_pred = clf.predict(val[["testritscore"]])
    labs = [l for l in _CASP_BAND_ORDER if l in y_true.values or l in y_pred]
    return y_true, y_pred, labs

def predict_2026(df, subject, clf):
    """Predict 2026 CAASPP performance"""
    d0 = filter_fall_course_grades(df, subject)
    fut = d0[d0["year"] == 2026].copy()
    if fut.empty:
        return None, None
    
    y_pred = clf.predict(fut[["testritscore"]])
    labs = [l for l in _CASP_BAND_ORDER if l in y_pred]
    return y_pred, labs

def _pct(arr, labels):
    """Calculate percentage distribution"""
    arr = np.asarray(arr).astype(str)
    raw = np.array([(arr == lab).mean() * 100 if len(arr) else 0 for lab in labels])
    total = raw.sum()
    if total > 0:
        raw = raw * (100 / total)
    return raw

# ---------------------------------------------------------------------
# SECTION 3 — Overall + Cohort Trends (Blended Dashboard)
# ---------------------------------------------------------------------

def _prep_nwea_matched_cohort_by_grade(df, course_str, current_grade, window_filter, cohort_year):
    """Prepare matched cohort data for Section 3"""
    base = df.copy()
    base["year"] = pd.to_numeric(base.get("year"), errors="coerce")
    base["grade"] = pd.to_numeric(base.get("grade"), errors="coerce")
    if "teststartdate" in base.columns:
        base["teststartdate"] = pd.to_datetime(base["teststartdate"], errors="coerce")
    else:
        base["teststartdate"] = pd.NaT
    
    if cohort_year is None:
        anchor_year = int(base["year"].max()) if base["year"].notna().any() else None
        if anchor_year is None:
            return pd.DataFrame(), pd.DataFrame(), {}, []
    else:
        anchor_year = int(cohort_year)
    
    cohort_grades = list(range(0, current_grade + 1))
    cohort_rows = []
    ordered_labels = []
    
    for grade in cohort_grades:
        offset = current_grade - grade
        year = anchor_year - offset
        if pd.isna(year):
            continue
        
        cohort_slice = base.copy()
        cohort_slice = cohort_slice[
            (cohort_slice["testwindow"].astype(str).str.upper() == window_filter.upper())
            & (cohort_slice["grade"] == grade)
            & (cohort_slice["year"] == year)
        ].copy()
        
        if "course" in cohort_slice.columns:
            if course_str == "Math K-12":
                cohort_slice = cohort_slice[cohort_slice["course"] == "Math K-12"]
            elif course_str == "Reading":
                cohort_slice = cohort_slice[cohort_slice["course"].str.startswith("Reading")]
            else:
                continue
        
        if "teststartdate" in cohort_slice.columns:
            cohort_slice["teststartdate"] = pd.to_datetime(cohort_slice["teststartdate"], errors="coerce")
        else:
            cohort_slice["teststartdate"] = pd.NaT
        
        cohort_slice.sort_values(["uniqueidentifier", "teststartdate"], inplace=True)
        cohort_slice = cohort_slice.groupby("uniqueidentifier", as_index=False).tail(1)
        cohort_slice = cohort_slice[cohort_slice["achievementquintile"].notna()].copy()
        
        if cohort_slice.empty:
            continue
        
        year_str_prev = str(year - 1)[-2:]
        year_str_curr = str(year)[-2:]
        label_full = f"Gr {int(grade)} • Fall {year_str_prev}-{year_str_curr}"
        cohort_slice["cohort_label"] = label_full
        cohort_rows.append(cohort_slice)
        ordered_labels.append(label_full)
    
    if not cohort_rows:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    cohort_df = pd.concat(cohort_rows, ignore_index=True)
    cohort_df["label"] = cohort_df["cohort_label"]
    
    def _extract_grade(label):
        try:
            return int(label.split()[1])
        except Exception:
            return 999
    
    cohort_df["cohort_label"] = pd.Categorical(cohort_df["cohort_label"],
                                                categories=sorted(cohort_df["cohort_label"].unique(), key=_extract_grade),
                                                ordered=True)
    
    quint_counts = cohort_df.groupby(["label", "achievementquintile"]).size().rename("n").reset_index()
    total_counts = cohort_df.groupby("label").size().rename("N_total").reset_index()
    pct_df = quint_counts.merge(total_counts, on="label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    all_idx = pd.MultiIndex.from_product([pct_df["label"].unique(), hf.NWEA_ORDER],
                                        names=["label", "achievementquintile"])
    pct_df = pct_df.set_index(["label", "achievementquintile"]).reindex(all_idx).reset_index()
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("label")["N_total"].transform(lambda s: s.ffill().bfill())
    
    score_df = (cohort_df[["label", "testritscore"]].dropna(subset=["testritscore"])
               .groupby("label")["testritscore"].mean().rename("avg_score").reset_index())
    
    pct_df["label"] = pd.Categorical(pct_df["label"], categories=ordered_labels, ordered=True)
    pct_df = pct_df.sort_values("label").reset_index(drop=True)
    score_df["label"] = pd.Categorical(score_df["label"], categories=ordered_labels, ordered=True)
    score_df = score_df.sort_values("label").reset_index(drop=True)
    
    pct_df = pct_df.rename(columns={"label": "time_label"})
    score_df = score_df.rename(columns={"label": "time_label"})
    
    labels_order = ordered_labels
    last_two = labels_order[-2:] if len(labels_order) >= 2 else labels_order
    
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        
        def pct_for(bucket_list, tlabel):
            return pct_df[(pct_df["time_label"] == tlabel) & (pct_df["achievementquintile"].isin(bucket_list))]["pct"].sum()
        
        metrics = {
            "t_prev": t_prev, "t_curr": t_curr,
            "hi_now": pct_for(hf.NWEA_HIGH_GROUP, t_curr),
            "hi_delta": pct_for(hf.NWEA_HIGH_GROUP, t_curr) - pct_for(hf.NWEA_HIGH_GROUP, t_prev),
            "lo_now": pct_for(hf.NWEA_LOW_GROUP, t_curr),
            "lo_delta": pct_for(hf.NWEA_LOW_GROUP, t_curr) - pct_for(hf.NWEA_LOW_GROUP, t_prev),
            "score_now": float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]),
            "score_delta": (float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) -
                          float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])),
        }
    else:
        metrics = {"t_prev": None, "t_curr": labels_order[-1] if labels_order else None,
                  "hi_now": None, "hi_delta": None, "lo_now": None, "lo_delta": None,
                  "score_now": None, "score_delta": None}
    
    return pct_df, score_df, metrics, labels_order

def plot_nwea_blended_dashboard(df, course_str, current_grade, window_filter, cohort_year,
                                output_dir, cfg, figsize=(16, 9), school_raw=None, scope_label=None, preview=False):
    """Produce blended dashboard: Overall Trends (left) + Cohort Trends (right)"""
    school_display = hf._safe_normalize_school_name(school_raw, cfg) if school_raw else None
    folder_name = "_district" if school_display is None else school_display.replace(" ", "_")
    district_label = cfg.get("district_name", ["District (All Students)"])[0] if not school_display else school_display
    
    course_str_for_title = "Math" if course_str == "Math K-12" else course_str
    
    df_left = df.copy()
    df_left["grade"] = pd.to_numeric(df_left["grade"], errors="coerce")
    df_left = df_left[df_left["grade"] == current_grade].copy()
    
    if "course" in df_left.columns:
        if course_str == "Math K-12":
            df_left = df_left[df_left["course"] == "Math K-12"]
        elif course_str == "Reading":
            df_left = df_left[df_left["course"].str.startswith("Reading")]
    
    pct_df_left, score_df_left, metrics_left, time_order_left = prep_nwea_for_charts(df_left, subject_str=course_str, window_filter=window_filter)
    
    if len(time_order_left) > 4:
        time_order_left = time_order_left[-4:]
        pct_df_left = pct_df_left[pct_df_left["time_label"].isin(time_order_left)].copy()
        score_df_left = score_df_left[score_df_left["time_label"].isin(time_order_left)].copy()
    
    cohort_df = df.copy()
    if school_raw:
        cohort_df = cohort_df[cohort_df["schoolname"] == school_raw]
    
    pct_df_right, score_df_right, metrics_right, time_order_right = _prep_nwea_matched_cohort_by_grade(
        cohort_df, course_str=course_str, current_grade=current_grade,
        window_filter=window_filter, cohort_year=cohort_year)
    
    if len(time_order_right) > 4:
        time_order_right = time_order_right[-4:]
        pct_df_right = pct_df_right[pct_df_right["time_label"].isin(time_order_right)].copy()
        score_df_right = score_df_right[score_df_right["time_label"].isin(time_order_right)].copy()
    
    # Check if we have enough data for left side (overall trends) - this is required
    if pct_df_left.empty or score_df_left.empty:
        print(f"[blended] no left-side data for Grade {current_grade} {course_str} ({school_display or 'district'})")
        return
    
    # Check if we have enough time periods for left side metrics
    if len(time_order_left) < 2:
        print(f"[blended] insufficient time periods ({len(time_order_left)}) for left side - need at least 2")
        return
    
    # Right side (cohort trends) is optional - if insufficient, we'll still generate chart with just overall trends
    has_cohort_data = not (pct_df_right.empty or score_df_right.empty) and len(time_order_right) >= 2
    if not has_cohort_data:
        print(f"[blended] insufficient cohort data for Grade {current_grade} {course_str} - generating overall trends only")
        # Create empty cohort dataframes with proper structure
        pct_df_right = pd.DataFrame(columns=["time_label", "achievementquintile", "pct", "n", "N_total"])
        score_df_right = pd.DataFrame(columns=["time_label", "avg_score"])
        time_order_right = []
        metrics_right = {"t_prev": None, "t_curr": None, "hi_now": None, "hi_delta": None, 
                        "lo_now": None, "lo_delta": None, "score_now": None, "score_delta": None}
    
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    def draw_stacked_bar(ax, stack_df, pct_df, time_labels):
        x = np.arange(len(stack_df))
        cumulative = np.zeros(len(stack_df))
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax.bar(x, band_vals, bottom=cumulative, color=hf.NWEA_COLORS[cat],
                         edgecolor="white", linewidth=1.2)
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    label_color = "white" if cat in ["High", "HiAvg", "Low"] else "#434343"
                    ax.text(rect.get_x() + rect.get_width() / 2, bottom_before + h / 2, f"{h:.2f}%",
                           ha="center", va="center", fontsize=8, fontweight="bold", color=label_color)
            cumulative += band_vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(stack_df.index.tolist())
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    stack_df_left = (pct_df_left.pivot(index="time_label", columns="achievementquintile", values="pct")
                    .reindex(columns=hf.NWEA_ORDER).fillna(0))
    ax1 = fig.add_subplot(gs[0, 0])
    draw_stacked_bar(ax1, stack_df_left, pct_df_left, time_order_left)
    ax1.set_title("Overall Trends", fontsize=14, fontweight="bold", pad=30)
    
    legend_handles = [Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q) for q in hf.NWEA_ORDER]
    fig.legend(handles=legend_handles, labels=hf.NWEA_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.95), ncol=len(hf.NWEA_ORDER), frameon=False,
              fontsize=9, handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    
    ax2 = fig.add_subplot(gs[0, 1])
    if has_cohort_data and not pct_df_right.empty:
        cohort_df_for_pivot = pct_df_right.copy()
        cohort_df_for_pivot = cohort_df_for_pivot.groupby(["time_label", "achievementquintile"], as_index=False).agg({"pct": "mean", "n": "sum", "N_total": "max"})
        stack_df_right = (cohort_df_for_pivot.pivot(index="time_label", columns="achievementquintile", values="pct")
                         .reindex(columns=hf.NWEA_ORDER).fillna(0))
        x_labels_cohort = stack_df_right.index.tolist()
        draw_stacked_bar(ax2, stack_df_right, cohort_df_for_pivot, x_labels_cohort)
        ax2.set_title("Cohort Trends", fontsize=14, fontweight="bold", pad=30)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Insufficient cohort data\n(need multiple years)", 
                ha="center", va="center", fontsize=12, color="#999999",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))
        ax2.set_title("Cohort Trends", fontsize=14, fontweight="bold", pad=30)
    
    ax3 = fig.add_subplot(gs[1, 0])
    rit_x = np.arange(len(score_df_left["time_label"]))
    rit_vals = score_df_left["avg_score"].to_numpy()
    rit_bars = ax3.bar(rit_x, rit_vals, color=hf.default_quintile_colors[4], edgecolor="white", linewidth=1.2)
    for rect, v in zip(rit_bars, rit_vals):
        ax3.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#434343")
    
    if "N_total" in pct_df_left.columns:
        n_map_left = pct_df_left.groupby("time_label", observed=False)["N_total"].max().reset_index().rename(columns={"N_total": "n"})
    else:
        n_map_left = pd.DataFrame(columns=["time_label", "n"])
    
    if not n_map_left.empty:
        label_map_left = {row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                         for _, row in n_map_left.iterrows() if not pd.isna(row["n"])}
        x_labels_left = [label_map_left.get(lbl, str(lbl)) for lbl in score_df_left["time_label"]]
    else:
        x_labels_left = score_df_left["time_label"].astype(str).tolist()
    
    ax3.set_ylabel("Avg RIT", labelpad=10)
    ax3.set_xticks(rit_x)
    ax3.set_xticklabels(x_labels_left, ha="center")
    for label in ax3.get_xticklabels():
        label.set_y(-0.09)
    ax3.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
    ax3.grid(axis="y", alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    
    ax4 = fig.add_subplot(gs[1, 1])
    if has_cohort_data and not score_df_right.empty and len(score_df_right) > 0:
        rit_xr = np.arange(len(score_df_right["time_label"]))
        rit_valsr = score_df_right["avg_score"].to_numpy()
        rit_barsr = ax4.bar(rit_xr, rit_valsr, color=hf.default_quintile_colors[4], edgecolor="white", linewidth=1.2)
        for rect, v in zip(rit_barsr, rit_valsr):
            ax4.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#434343")
        
        if has_cohort_data and not pct_df_right.empty:
            cohort_df_for_pivot = pct_df_right.copy()
            cohort_df_for_pivot = cohort_df_for_pivot.groupby(["time_label", "achievementquintile"], as_index=False).agg({"pct": "mean", "n": "sum", "N_total": "max"})
            if "N_total" in cohort_df_for_pivot.columns:
                n_map_right = cohort_df_for_pivot.groupby("time_label")["N_total"].max().reset_index().rename(columns={"N_total": "n"})
            else:
                n_map_right = pd.DataFrame(columns=["time_label", "n"])
            
            if not n_map_right.empty:
                label_map_right = {row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                                  for _, row in n_map_right.iterrows() if not pd.isna(row["n"])}
                x_labels_right = [label_map_right.get(lbl, str(lbl)) for lbl in score_df_right["time_label"]]
            else:
                x_labels_right = score_df_right["time_label"].astype(str).tolist()
        else:
            x_labels_right = score_df_right["time_label"].astype(str).tolist()
        
        ax4.set_ylabel("Avg RIT", labelpad=10)
        ax4.set_xticks(rit_xr)
        ax4.set_xticklabels(x_labels_right, ha="center")
        for label in ax4.get_xticklabels():
            label.set_y(-0.09)
        ax4.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
        ax4.grid(axis="y", alpha=0.2)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "Insufficient cohort data", 
                ha="center", va="center", fontsize=10, color="#999999")
        ax4.set_title("Average RIT", fontsize=8, fontweight="bold", pad=10)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")
    if metrics_left.get("t_prev"):
        t_prev = metrics_left["t_prev"]
        t_curr = metrics_left["t_curr"]
        
        def _pct_for_bucket_left(bucket_name, tlabel):
            return pct_df_left[(pct_df_left["time_label"] == tlabel) & (pct_df_left["achievementquintile"] == bucket_name)]["pct"].sum()
        
        high_now = _pct_for_bucket_left("High", t_curr)
        high_prev = _pct_for_bucket_left("High", t_prev)
        high_delta = high_now - high_prev
        hi_delta = metrics_left["hi_delta"]
        lo_delta = metrics_left["lo_delta"]
        score_delta = metrics_left["score_delta"]
        
        title_line = "Change calculations from previous to current year\n"
        line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
        insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
    else:
        insight_lines = ["Not enough history for change insights"]
    
    ax5.text(0.5, 0.5, "\n".join(insight_lines), fontsize=9, fontweight="normal", color="#434343",
            ha="center", va="center", wrap=True, usetex=False,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    if has_cohort_data and metrics_right.get("t_prev"):
        t_prev = metrics_right["t_prev"]
        t_curr = metrics_right["t_curr"]
        
        def _pct_for_bucket_right(bucket_name, tlabel):
            return pct_df_right[(pct_df_right["time_label"] == tlabel) & (pct_df_right["achievementquintile"] == bucket_name)]["pct"].sum()
        
        high_now = _pct_for_bucket_right("High", t_curr)
        high_prev = _pct_for_bucket_right("High", t_prev)
        high_delta = high_now - high_prev
        hi_delta = metrics_right["hi_delta"]
        lo_delta = metrics_right["lo_delta"]
        score_delta = metrics_right["score_delta"]
        
        title_line = "Change calculations from previous to current year\n"
        line_high = rf"$\Delta$ High: $\mathbf{{{high_delta:+.1f}}}$ ppts"
        line_hiavg = rf"$\Delta$ Avg+HiAvg+High: $\mathbf{{{hi_delta:+.1f}}}$ ppts"
        line_low = rf"$\Delta$ Low: $\mathbf{{{lo_delta:+.1f}}}$ ppts"
        line_rit = rf"$\Delta$ Avg RIT: $\mathbf{{{score_delta:+.1f}}}$ pts"
        insight_lines = [title_line, line_high, line_hiavg, line_low, line_rit]
    else:
        insight_lines = ["Insufficient cohort data\n(need multiple years)"]
    
    ax6.text(0.5, 0.5, "\n".join(insight_lines), fontsize=9, fontweight="normal", color="#434343",
            ha="center", va="center", wrap=True, usetex=False,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    fig.suptitle(f"{district_label} • Grade {int(current_grade)} • {course_str_for_title}",
                fontsize=20, fontweight="bold", y=1)
    
    charts_dir = Path(output_dir)
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = scope_label or (cfg.get("district_name", ["Districtwide"])[0] if school_raw is None else hf._safe_normalize_school_name(school_raw, cfg))
    out_name = f"{scope.replace(' ', '_')}_section3_grade{int(current_grade)}_{course_str.lower().replace(' ', '_')}_{window_filter.lower()}_trends.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    # Print absolute path as string for reliable parsing
    print(f"Saved: {str(out_path.absolute())}")
    track_chart(out_name, out_path, scope=folder_name, section=3)
    if preview:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# SECTION 4 — Overall Growth Trends by Site (CGP + CGI)
# ---------------------------------------------------------------------

def _prep_cgp_trend(df, subject_str, cfg):
    """Prepare CGP trend data"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == "FALL"].copy()
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    elif "reading" in subj_norm:
        d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
    
    if "falltofallconditionalgrowthpercentile" not in d.columns:
        return pd.DataFrame(columns=["scope_label", "time_label", "median_cgp", "mean_cgi"])
    
    d = d[d["falltofallconditionalgrowthpercentile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["scope_label", "time_label", "median_cgp", "mean_cgi"])
    
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
    d["subject"] = subject_str
    d["site_display"] = d["schoolname"].apply(lambda x: hf._safe_normalize_school_name(x, cfg))
    
    dist_rows = d.copy()
    dist_rows["site_display"] = cfg.get("district_name", ["District (All Students)"])[0]
    both = pd.concat([d, dist_rows], ignore_index=True)
    
    has_cgi = "falltofallconditionalgrowthindex" in both.columns
    grp_cols = ["site_display", "time_label"]
    
    if has_cgi:
        out = both.groupby(grp_cols, dropna=False).agg(
            median_cgp=("falltofallconditionalgrowthpercentile", "median"),
            mean_cgi=("falltofallconditionalgrowthindex", "mean")).reset_index()
    else:
        out = both.groupby(grp_cols, dropna=False).agg(
            median_cgp=("falltofallconditionalgrowthpercentile", "median")).reset_index()
        out["mean_cgi"] = np.nan
    
    time_order = sorted(out["time_label"].astype(str).unique().tolist())
    out["time_label"] = pd.Categorical(out["time_label"], categories=time_order, ordered=True)
    out.sort_values(["site_display", "time_label"], inplace=True)
    
    keep_list = []
    for scope_val, chunk in out.groupby("site_display", dropna=False):
        ordered = chunk["time_label"].cat.categories.tolist()
        present = chunk["time_label"].astype(str).unique().tolist()
        recent = set([t for t in ordered if t in present][-4:])
        keep_list.append(chunk[chunk["time_label"].astype(str).isin(recent)])
    
    out_recent = pd.concat(keep_list, ignore_index=True)
    out_recent["subject"] = subject_str
    return out_recent.rename(columns={"site_display": "scope_label"})

def _plot_cgp_trend(df, subject_str, scope_label, ax=None):
    """Plot CGP trend bars with CGI line"""
    sub = df[df["scope_label"] == scope_label].copy()
    if sub.empty:
        return
    
    sub["time_label"] = pd.Categorical(sub["time_label"], categories=sorted(sub["time_label"].astype(str).unique()), ordered=True)
    sub.sort_values("time_label", inplace=True)
    
    x_vals = np.arange(len(sub))
    y_cgp = sub["median_cgp"].to_numpy(float)
    y_cgi = sub["mean_cgi"].to_numpy(float)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.axhspan(0, 20, facecolor="#808080", alpha=0.5, zorder=0)
    ax.axhspan(20, 40, facecolor="#c5c5c5", alpha=0.5, zorder=0)
    ax.axhspan(40, 60, facecolor="#78daf4", alpha=0.5, zorder=0)
    ax.axhspan(60, 80, facecolor="#00baeb", alpha=0.5, zorder=0)
    ax.axhspan(80, 100, facecolor="#0381a2", alpha=0.5, zorder=0)
    
    for y in [42, 50, 58]:
        ax.axhline(y, linestyle="--", color="#6B7280", linewidth=1.2)
    
    bars = ax.bar(x_vals, y_cgp, color="#0381a2", edgecolor="white", linewidth=1.2, zorder=3)
    for rect, yv in zip(bars, y_cgp):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2, f"{yv:.1f}",
               ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    
    ax.set_ylabel("Median Fall→Fall CGP")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(sub["time_label"].astype(str).tolist())
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax2 = ax.twinx()
    ax2.set_ylim(-2.5, 2.5)
    ax2.patch.set_alpha(0)
    blend = mtransforms.BlendedGenericTransform(ax.transData, ax2.transData)
    x0, x1 = ax.get_xlim()
    
    band = mpl.patches.Rectangle((x0, -0.2), x1 - x0, 0.4, transform=blend,
                                 facecolor="#facc15", alpha=0.35, zorder=1)
    ax.add_patch(band)
    
    for yb in [-0.2, 0.2]:
        ax.add_line(mlines.Line2D([x0, x1], [yb, yb], transform=blend,
                                 linestyle="--", color="#eab308", linewidth=1.2))
    
    cgi_line = mlines.Line2D(x_vals, y_cgi, transform=blend, marker="o", linewidth=2,
                            markersize=6, color="#ffa800", zorder=3)
    ax.add_line(cgi_line)
    
    for xv, yv in zip(x_vals, y_cgi):
        if pd.notna(yv):
            ax.text(xv, yv + (0.12 if yv >= 0 else -0.12), f"{yv:.2f}", transform=blend,
                   ha="center", va="bottom" if yv >= 0 else "top", fontsize=8,
                   fontweight="bold", color="#ffa800")
    
    ax2.set_ylabel("Avg Fall→Fall CGI")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title(f"{subject_str}", fontweight="bold", fontsize=14, pad=10)

def _run_cgp_dual_trend(scope_df, scope_label, output_dir, cfg, preview=False, subjects=None):
    """Run CGP dual trend chart generation"""
    if subjects is None:
        subjects = ["Reading", "Mathematics"]
    
    cgp_trend = pd.concat([_prep_cgp_trend(scope_df, subj, cfg) for subj in subjects], ignore_index=True)
    if cgp_trend.empty:
        print(f"[skip] No CGP data for {scope_label}")
        return
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle(f"{scope_label} • Fall→Fall Growth (All Students)", fontsize=20, fontweight="bold", y=0.99)
    
    axes = []
    for i, subject_str in enumerate(subjects):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        sub_df = cgp_trend[(cgp_trend["scope_label"] == scope_label) & (cgp_trend["subject"] == subject_str)]
        if not sub_df.empty:
            _plot_cgp_trend(sub_df, subject_str, scope_label, ax=ax)
        
        subj_norm = subject_str.strip().casefold()
        d = scope_df.copy()
        d = d[d["testwindow"].astype(str).str.upper() == "FALL"].copy()
        if "math" in subj_norm:
            d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
        elif "reading" in subj_norm:
            d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
        
        d["year_short"] = d["year"].apply(_short_year)
        d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
        d["site_display"] = d["schoolname"].apply(lambda x: hf._safe_normalize_school_name(x, cfg))
        if scope_label == cfg.get("district_name", ["District (All Students)"])[0]:
            d["site_display"] = cfg.get("district_name", ["District (All Students)"])[0]
        d = d[d["site_display"] == scope_label]
        
        n_map = d.groupby("time_label")["uniqueidentifier"].nunique().reset_index().rename(columns={"uniqueidentifier": "n"})
        n_map_dict = dict(zip(n_map["time_label"], n_map["n"]))
        ticklabels = [str(lbl) for lbl in sub_df["time_label"]]
        labels_with_n = [f"{lbl}\n(n = {int(n_map_dict.get(lbl, 0))})" for lbl in ticklabels]
        ax.set_xticklabels(labels_with_n)
        ax.tick_params(axis="x", pad=10)
    
    legend_handles = [Patch(facecolor="#0381a2", edgecolor="white", label="Median CGP"),
                     Line2D([0], [0], color="#ffa800", marker="o", linewidth=2, markersize=6, label="Mean CGI")]
    fig.legend(handles=legend_handles, labels=["Median CGP", "Mean CGI"], loc="upper center",
              bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, handlelength=2, handletextpad=0.5, columnspacing=1.2)
    
    charts_dir = Path(output_dir)
    folder_name = "_district" if scope_label == cfg.get("district_name", ["District (All Students)"])[0] else scope_label.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section4_cgp_fall_to_fall_dualpanel.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved: {str(out_path.absolute())}")
    track_chart(out_name, out_path, scope=folder_name, section=4)
    if preview:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# SECTION 6 — Plotting Functions for Model Predictions
# ---------------------------------------------------------------------

def _plot_pred_vs_actual(scope_label, folder, output_dir, results, preview=False):
    """Plot 2025 predicted vs actual CAASPP"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(wspace=0.25)
    
    for ax, subject in zip(axs, ["Reading", "Mathematics"]):
        r = results.get(subject)
        if not r:
            ax.set_axis_off()
            continue
        
        y_true, y_pred, labs = r
        pct_true = _pct(y_true, labs)
        pct_pred = _pct(y_pred, labs)
        x_pred, x_act = -0.2, 0.2
        w = 0.35
        
        bottom = 0
        for lab, val in zip(labs, pct_pred):
            ax.bar(x_pred, val, bottom=bottom, width=w, color=hf.CERS_LEVEL_COLORS[lab],
                  edgecolor="black", linestyle="--", alpha=0.5)
            if val > 3:
                ax.text(x_pred, bottom + val / 2, f"{val:.1f}%", ha="center", va="center",
                       color="black", fontsize=8, fontweight="bold")
            bottom += val
        
        bottom = 0
        for lab, val in zip(labs, pct_true):
            ax.bar(x_act, val, bottom=bottom, width=w, color=hf.CERS_LEVEL_COLORS[lab],
                  edgecolor="white")
            if val > 3:
                ax.text(x_act, bottom + val / 2, f"{val:.1f}%", ha="center", va="center",
                       color="white", fontsize=8, fontweight="bold")
            bottom += val
        
        ax.set_xticks([x_pred, x_act])
        ax.set_xticklabels(["2025\nPredicted", "2025\nActual"])
        ax.set_ylim(0, 100)
        ax.set_title(subject, fontweight="bold")
        ax.set_ylabel("% of Students")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"{scope_label} \n 2025 Predicted vs Actual CAASPP (Fall NWEA)",
                fontsize=15, fontweight="bold")
    
    handles = [plt.Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=hf.CERS_LEVEL_COLORS[l],
                          markeredgecolor="white", markersize=10, label=l) for l in _CASP_BAND_ORDER]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.03))
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"section6A_2025_pred_vs_actual_{folder}.png"
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved: {str(out_path.absolute())}")
    track_chart(f"section6A_2025_pred_vs_actual_{folder}.png", out_path, scope=folder, section=6)
    if preview:
        plt.show()
    plt.close(fig)

def _plot_projection_2026(scope_label, folder, output_dir, results, preview=False):
    """Plot 2026 projected CAASPP"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(wspace=0.25)
    
    for ax, subject in zip(axs, ["Reading", "Mathematics"]):
        r = results.get(subject)
        if not r:
            ax.set_axis_off()
            continue
        
        y_pred, labs = r
        pct = _pct(y_pred, labs)
        bottom = 0
        for lab, val in zip(labs, pct):
            ax.bar(0, val, bottom=bottom, width=0.55, color=hf.CERS_LEVEL_COLORS[lab],
                  edgecolor="white", alpha=0.9)
            if val > 3:
                ax.text(0, bottom + val / 2, f"{val:.1f}%", ha="center", va="center",
                       color="white", fontsize=9, fontweight="bold")
            bottom += val
        
        ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.set_title(f"{subject} — 2026 Projected", fontweight="bold")
        ax.set_ylabel("% of Students")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"{scope_label} \n Projected 2026 CAASPP (Fall 2026 NWEA)", fontsize=15, fontweight="bold")
    
    handles = [plt.Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=hf.CERS_LEVEL_COLORS[l],
                          markeredgecolor="white", markersize=10, label=l) for l in _CASP_BAND_ORDER]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.03))
    
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"section6B_2026_projection_{folder}.png"
    hf._save_and_render(fig, out_path, dev_mode=preview)
    print(f"Saved: {str(out_path.absolute())}")
    track_chart(f"section6B_2026_projection_{folder}.png", out_path, scope=folder, section=6)
    if preview:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate NWEA charts')
    parser.add_argument('--partner', required=True, help='Partner name')
    parser.add_argument('--data-dir', required=True, help='Data directory path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--dev-mode', default='false', help='Development mode')
    parser.add_argument('--config', default='{}', help='Config JSON string')
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config_from_args(args.config)
    
    # Set dev mode
    hf.DEV_MODE = args.dev_mode.lower() in ('true', '1', 'yes', 'on')
    
    # Extract chart filters
    chart_filters = cfg.get("chart_filters", {})
    if chart_filters:
        print(f"\n[Filters] Applying chart generation filters:")
        if chart_filters.get("grades"):
            print(f"  - Grades: {chart_filters['grades']}")
        if chart_filters.get("years"):
            print(f"  - Years: {chart_filters['years']}")
        if chart_filters.get("quarters"):
            print(f"  - Quarters: {chart_filters['quarters']}")
        if chart_filters.get("subjects"):
            print(f"  - Subjects: {chart_filters['subjects']}")
        if chart_filters.get("student_groups"):
            print(f"  - Student Groups: {chart_filters['student_groups']}")
        if chart_filters.get("race"):
            print(f"  - Race/Ethnicity: {chart_filters['race']}")
    
    # Check if NWEA charts should be generated based on subject filter
    # NWEA only supports Math and Reading, so skip if only other subjects are selected
    should_generate_nwea = True
    if chart_filters and chart_filters.get("subjects") and len(chart_filters["subjects"]) > 0:
        selected_subjects = [str(s).strip().lower() for s in chart_filters["subjects"]]
        nwea_subjects = ["math", "mathematics", "reading"]
        # Only generate NWEA charts if at least one NWEA-compatible subject is selected
        should_generate_nwea = any(s in nwea_subjects for s in selected_subjects)
        if not should_generate_nwea:
            print("\n[Skip] NWEA charts skipped - selected subjects don't include Math or Reading")
            print(f"  Selected subjects: {chart_filters['subjects']}")
            print(f"  NWEA supports: Math, Reading")
            # Still generate chart index for any existing charts
            charts_dir = Path(args.output_dir).resolve()
            if charts_dir.exists() and chart_links:
                df_charts = pd.DataFrame(chart_links)
                df_charts = df_charts.sort_values(["scope", "section", "chart_name"])
                index_path = charts_dir / "chart_index.csv"
                df_charts.to_csv(index_path, index=False)
                print(f"✅ Chart index saved: {len(chart_links)} tracked charts -> {index_path}")
            return []
    
    # Load data
    nwea_base = load_nwea_data(args.data_dir)
    
    # Apply filters to base data
    if chart_filters:
        nwea_base = apply_chart_filters(nwea_base, chart_filters)
        print(f"Data after filtering: {nwea_base.shape[0]:,} rows")
    
    # Get selected quarters (default to Fall if not specified)
    selected_quarters = ["Fall"]
    if chart_filters and chart_filters.get("quarters") and len(chart_filters["quarters"]) > 0:
        selected_quarters = chart_filters["quarters"]
        print(f"Generating charts for quarters: {selected_quarters}")
    else:
        print("No quarters specified, defaulting to Fall")
    
    # Get scopes
    scopes = get_scopes(nwea_base, cfg)
    
    # Generate charts
    chart_paths = []
    
    # Section 0: Predicted vs Actual CAASPP (Spring)
    print("\n[Section 0] Generating Predicted vs Actual CAASPP...")
    for scope_df, scope_label, folder in scopes:
        try:
            payload = {}
            for subj in ["Reading", "Mathematics"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                proj, act, metrics, _ = _prep_section0(scope_df, subj)
                if proj is None:
                    continue
                payload[subj] = {"proj_pct": proj, "act_pct": act, "metrics": metrics}
            if payload:
                _plot_section0_dual(scope_label, folder, args.output_dir, payload, preview=hf.DEV_MODE)
                chart_paths.append(str(Path(args.output_dir) / folder / f"{scope_label.replace(' ', '_')}-section0_pred_vs_actual_{folder}.png"))
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"Error generating Section 0 chart for {scope_label}: {error_msg}")
            if hf.DEV_MODE:
                print(f"Traceback:\n{tb_str}")
            continue
    
    # Section 1: Fall Performance Trends (All Students only)
    print("\n[Section 1] Generating Fall Performance Trends...")
    # Section 1 generates dual-panel charts for "All Students"
    # Only generate if "All Students" is selected OR no student group filters are specified
    should_generate_section1 = True
    if chart_filters and chart_filters.get("subjects") and len(chart_filters["subjects"]) > 0:
        # Only generate if at least one of Reading or Mathematics is selected
        selected_subjects = [str(s).strip().lower() for s in chart_filters["subjects"]]
        should_generate_section1 = any(s in ["reading", "math", "mathematics"] for s in selected_subjects)
    
    # Check if "All Students" is selected in student groups
    if chart_filters and chart_filters.get("student_groups") and len(chart_filters["student_groups"]) > 0:
        # Only generate Section 1 if "All Students" is explicitly selected
        if "All Students" not in chart_filters["student_groups"]:
            should_generate_section1 = False
            print("  Skipping Section 1 (only specific student groups selected, not 'All Students')")
    
    if should_generate_section1:
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
                    chart_paths.append(chart_path)
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    tb_str = traceback.format_exc()
                    print(f"Error generating chart for {scope_label} ({quarter}): {error_msg}")
                    if hf.DEV_MODE:
                        print(f"Traceback:\n{tb_str}")
                    continue
    else:
        print("  Skipping Section 1 (no matching subjects selected)")
    
    # Section 2: Student Group Performance Trends
    print("\n[Section 2] Generating Student Group Performance Trends...")
    student_groups_cfg = cfg.get("student_groups", {})
    group_order = cfg.get("student_group_order", {})
    
    # Debug: Print filter info
    if chart_filters:
        print(f"  [Debug] Student groups filter: {chart_filters.get('student_groups')}")
        print(f"  [Debug] Race filter: {chart_filters.get('race')}")
    
    for scope_df, scope_label, folder in scopes:
        for group_name, group_def in sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)):
            # Skip "All Students" - it's handled in Section 1
            if group_def.get("type") == "all":
                continue
            # Only generate charts for selected student groups
            should_gen = should_generate_student_group(group_name, chart_filters)
            if not should_gen:
                if hf.DEV_MODE:
                    print(f"  [Skip] {group_name} - not in selected filters")
                continue
            if hf.DEV_MODE:
                print(f"  [Generate] {group_name}")
            for quarter in selected_quarters:
                try:
                    plot_nwea_subject_dashboard_by_group(
                        scope_df.copy(), None, quarter, group_name, group_def,
                        args.output_dir, cfg, figsize=(16, 9),
                        school_raw=None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None,
                        scope_label=scope_label, preview=hf.DEV_MODE)
                except Exception as e:
                    if hf.DEV_MODE:
                        print(f"  Error generating Section 2 chart for {scope_label} - {group_name} ({quarter}): {e}")
                    continue
    
    # Section 3: Overall + Cohort Trends
    print("\n[Section 3] Generating Overall + Cohort Trends...")
    
    def _run_scope_section3(scope_df, scope_label, school_raw):
        # Use filtered data but generate charts for all grades present in the filtered data
        scope_df = scope_df.copy()
        scope_df["year"] = pd.to_numeric(scope_df["year"], errors="coerce")
        
        # Normalize grades (K -> 0) for chart generation
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
        
        scope_df["grade_normalized"] = scope_df["grade"].apply(normalize_grade_val)
        
        if scope_df["year"].notna().any():
            anchor_year = int(scope_df["year"].max())
        else:
            anchor_year = None
        
        # Get ALL unique normalized grades from the filtered data
        # Generate charts for all grades that exist in the filtered dataset
        unique_grades = sorted([g for g in scope_df["grade_normalized"].dropna().unique() if g is not None])
        
        print(f"  [Section 3] Found {len(unique_grades)} grade(s) in filtered data: {unique_grades}")
        
        for g in unique_grades:
            # Filter scope_df to this specific grade for chart generation
            grade_df = scope_df[scope_df["grade_normalized"] == g].copy()
            if grade_df.empty:
                continue
            
            courses_in_data = set(grade_df["course"].dropna().unique())
            for course_str in ["Math K-12", "Reading"]:
                # Map course string to subject name for filter check
                subject_name = "Mathematics" if course_str == "Math K-12" else "Reading"
                if not should_generate_subject(subject_name, chart_filters):
                    continue
                
                if (course_str == "Math K-12" and "Math K-12" in courses_in_data) or \
                   (course_str == "Reading" and any(str(c).startswith("Reading") for c in courses_in_data)):
                    for quarter in selected_quarters:
                        try:
                            print(f"  [Section 3] Generating chart for {scope_label} - Grade {g} - {course_str} - {quarter}")
                            plot_nwea_blended_dashboard(
                                grade_df.copy(), course_str=course_str, current_grade=int(g),
                                window_filter=quarter, cohort_year=anchor_year, output_dir=args.output_dir, cfg=cfg,
                                figsize=(16, 9), school_raw=school_raw, preview=hf.DEV_MODE, scope_label=scope_label)
                        except Exception as e:
                            print(f"  [Section 3] Error generating chart for {scope_label} - Grade {g} - {course_str} ({quarter}): {e}")
                            if hf.DEV_MODE:
                                import traceback
                                traceback.print_exc()
                            continue
    
    # Use filtered scopes (data is already filtered by chart_filters)
    for scope_df, scope_label, folder in scopes:
        _run_scope_section3(scope_df.copy(), scope_label, None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None)
    
    # Section 4: CGP/CGI Growth Trends (Fall→Fall only, requires Fall quarter)
    if "Fall" in selected_quarters:
        print("\n[Section 4] Generating CGP/CGI Growth Trends (Fall→Fall)...")
        for scope_df, scope_label, folder in scopes:
            try:
                # Filter subjects for Section 4
                subjects_to_generate = ["Reading", "Mathematics"]
                if chart_filters and chart_filters.get("subjects"):
                    subjects_to_generate = [s for s in subjects_to_generate if should_generate_subject(s, chart_filters)]
                
                if subjects_to_generate:
                    _run_cgp_dual_trend(scope_df.copy(), scope_label, args.output_dir, cfg, preview=hf.DEV_MODE, subjects=subjects_to_generate)
            except Exception as e:
                import traceback
                error_msg = str(e)
                tb_str = traceback.format_exc()
                print(f"Error generating Section 4 chart for {scope_label}: {error_msg}")
                if hf.DEV_MODE:
                    print(f"Traceback:\n{tb_str}")
                continue
    
    # Section 6: Model Predictions
    print("\n[Section 6] Generating Model Predictions...")
    for scope_df, scope_label, folder in scopes:
        results_2025, results_2026 = {}, {}
        for subj in ["Reading", "Mathematics"]:
            if not should_generate_subject(subj, chart_filters):
                continue
            try:
                clf = train_model(scope_df, subj)
                if clf is None:
                    continue
                y_true25, y_pred25, labs25 = predict_2025(scope_df, subj, clf)
                if y_true25 is not None:
                    results_2025[subj] = (y_true25, y_pred25, labs25)
                y_pred26, labs26 = predict_2026(scope_df, subj, clf)
                if y_pred26 is not None:
                    results_2026[subj] = (y_pred26, labs26)
            except Exception as e:
                if hf.DEV_MODE:
                    print(f"  Error generating predictions for {scope_label} - {subj}: {e}")
                continue
        
        if results_2025:
            try:
                _plot_pred_vs_actual(scope_label, folder, args.output_dir, results_2025, preview=hf.DEV_MODE)
            except Exception as e:
                if hf.DEV_MODE:
                    print(f"  Error plotting 2025 predictions for {scope_label}: {e}")
        
        if results_2026:
            try:
                _plot_projection_2026(scope_label, folder, args.output_dir, results_2026, preview=hf.DEV_MODE)
            except Exception as e:
                if hf.DEV_MODE:
                    print(f"  Error plotting 2026 projections for {scope_label}: {e}")
    
    # Generate chart index
    print("\n" + "="*80)
    print("Generating chart index CSV...")
    charts_dir = Path(args.output_dir).resolve()
    
    # Use tracked charts if available, otherwise scan directory
    if chart_links:
        df_charts = pd.DataFrame(chart_links)
        df_charts = df_charts.sort_values(["scope", "section", "chart_name"])
        index_path = charts_dir / "chart_index.csv"
        df_charts.to_csv(index_path, index=False)
        print(f"✅ Chart index saved: {len(chart_links)} tracked charts -> {index_path}")
    elif charts_dir.exists():
        chart_files = []
        for f in charts_dir.rglob("*.png"):
            chart_files.append({
                "chart_name": f.stem,
                "folder": f.parent.name,
                "file_path": str(f.relative_to(charts_dir)),
                "file_link": f"file://{f.absolute()}"
            })
        
        if chart_files:
            df_charts = pd.DataFrame(chart_files).sort_values(["folder", "chart_name"])
            index_path = charts_dir / "chart_index.csv"
            df_charts.to_csv(index_path, index=False)
            print(f"✅ Chart index saved: {len(chart_files)} charts -> {index_path}")
        else:
            print("⚠️  No chart files found to index")
    else:
        print(f"⚠️  Charts directory does not exist: {charts_dir}")
    print("="*80)
    
    # Build chart_paths from tracked charts if chart_paths is incomplete
    # This ensures all tracked charts are returned, not just those from sections 0 and 1
    if chart_links and len(chart_links) > len(chart_paths):
        print(f"[Chart Tracking] Found {len(chart_links)} tracked charts but only {len(chart_paths)} in chart_paths")
        print(f"[Chart Tracking] Building chart_paths from tracked charts...")
        # Use absolute paths from chart_links to ensure consistency
        chart_paths = [str(Path(chart['file_path']).absolute()) for chart in chart_links]
        print(f"[Chart Tracking] Built {len(chart_paths)} chart paths from tracked charts")
    
    print(f"\n✅ Generated {len(chart_paths)} NWEA charts")
    return chart_paths

def generate_nwea_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data'
) -> list:
    """
    Generate NWEA charts (wrapper function for Flask backend)
    
    Args:
        partner_name: Partner name
        output_dir: Output directory for charts
        config: Partner configuration dict
        chart_filters: Chart filters dict
        data_dir: Data directory path
    
    Returns:
        List of chart file paths
    """
    # Set up config
    cfg = config or {}
    if chart_filters:
        cfg['chart_filters'] = chart_filters
    
    # Set dev mode from config
    hf.DEV_MODE = cfg.get('dev_mode', False)
    
    # Create mock args object to pass to main logic
    class Args:
        def __init__(self):
            self.partner = partner_name
            self.data_dir = data_dir
            self.output_dir = output_dir
            self.dev_mode = 'true' if hf.DEV_MODE else 'false'
            self.config = json.dumps(cfg) if cfg else '{}'
    
    args = Args()
    
    # Temporarily override sys.argv to call main()
    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            'nwea_charts.py',
            '--partner', args.partner,
            '--data-dir', args.data_dir,
            '--output-dir', args.output_dir,
            '--dev-mode', args.dev_mode,
            '--config', args.config
        ]
        chart_paths = main()
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

