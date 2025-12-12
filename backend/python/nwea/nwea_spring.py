"""
NWEA Spring chart generation module
Generates charts specifically for Spring (EOY) test window
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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib import transforms as mtransforms
from matplotlib import lines as mlines

# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

# Import utility modules
from .nwea_data import (
    load_config_from_args,
    load_nwea_data,
    get_scopes,
    prep_nwea_for_charts,
    _short_year
)
from .nwea_filters import (
    apply_chart_filters,
    should_generate_subject,
    should_generate_student_group,
    should_generate_grade
)
from .nwea_chart_utils import (
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
    
    # Save chart data as JSON if provided
    if chart_data is not None:
        data_path = chart_path.parent / f"{chart_path.stem}_data.json"
        try:
            import json
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

# Import chart generation functions from main module
# We'll use these but filter for Winter
from .nwea_charts import (
    plot_dual_subject_dashboard,
    plot_nwea_subject_dashboard_by_group,
    plot_nwea_blended_dashboard,
    _plot_section0_dual,
    track_chart as track_chart_base
)

def _get_school_column(df):
    """Get the appropriate school column name, checking learning_center first"""
    if "learning_center" in df.columns:
        return "learning_center"
    elif "schoolname" in df.columns:
        return "schoolname"
    elif "school" in df.columns:
        return "school"
    return None

# ---------------------------------------------------------------------
# Spring-specific Section 0: Predicted vs Actual CAASPP (Spring)
# ---------------------------------------------------------------------

def _prep_section0_spring(df, subject):
    """Prepare data for Section 0: Predicted vs Actual CAASPP - Spring version"""
    d = df.copy()
    # Use Spring test window
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper() == "SPRING"].copy()
    
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
    
    # Only consider rows that actually have both projected and actual CAASPP values
    d = d.dropna(subset=["projectedproficiencylevel2", "cers_overall_performanceband"]).copy()
    if d.empty or d["year"].dropna().empty:
        return None, None, None, None
    
    # Target year is the latest test year with valid projected + CAASPP join
    target_year = int(d["year"].max())
    
    # Keep only that target year slice
    d = d[d["year"] == target_year].copy()
    
    # Dedupe — most recent test per student within the target year slice
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        d = d.sort_values("teststartdate").drop_duplicates("uniqueidentifier", keep="last")
    
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

# ---------------------------------------------------------------------
# Section 0.1: Winter vs Spring Performance Snapshot (Spring-specific)
# ---------------------------------------------------------------------

def _prep_nwea_winter_spring_snapshot(df: pd.DataFrame, subject_str: str):
    """Prep frame for Winter vs Spring comparison in the latest year."""
    d = df.copy()
    
    # 1. Restrict to Winter and Spring only
    d["testwindow"] = d["testwindow"].astype(str)
    mask_window = d["testwindow"].str.upper().isin(["WINTER", "SPRING"])
    d = d[mask_window].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # 2. Course-based filtering
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math k-12", case=False, na=False)].copy()
    elif "reading" in subj_norm:
        d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
    
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # 3. Require valid quintile bucket
    d = d[d["achievementquintile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # 4. Normalize year and choose latest
    d["year_num"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d = d[d["year_num"].notna()].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    target_year = int(d["year_num"].max())
    d = d[d["year_num"] == target_year].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # 5. Build time_label like "Winter 25-26" / "Spring 25-26"
    d["year_short"] = d["year_num"].apply(_short_year)
    d["time_label"] = d["testwindow"].str.title() + " " + d["year_short"]
    
    # 6. Dedupe to latest attempt per student per time_label
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT
    
    d.sort_values(["uniqueidentifier", "time_label", "teststartdate"], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)
    
    # 7. Percent by quintile
    quint_counts = d.groupby(["time_label", "achievementquintile"]).size().rename("n").reset_index()
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()
    
    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Ensure all quintiles exist for Winter/Spring
    time_labels = pct_df["time_label"].unique().tolist()
    all_idx = pd.MultiIndex.from_product(
        [time_labels, hf.NWEA_ORDER],
        names=["time_label", "achievementquintile"],
    )
    pct_df = (
        pct_df.set_index(["time_label", "achievementquintile"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(
        lambda s: s.ffill().bfill()
    )
    
    # 8. Avg RIT per time_label
    score_df = (
        d[["time_label", "testritscore"]]
        .dropna(subset=["testritscore"])
        .groupby("time_label")["testritscore"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )
    
    # 9. Enforce Winter→Spring ordering
    def _sort_key(lbl: str) -> tuple:
        if lbl.startswith("Winter"):
            season_order = 0
        elif lbl.startswith("Spring"):
            season_order = 1
        else:
            season_order = 99
        return (season_order, lbl)
    
    time_order = sorted(
        pct_df["time_label"].dropna().astype(str).unique().tolist(), key=_sort_key
    )
    
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"], categories=time_order, ordered=True
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"], categories=time_order, ordered=True
    )
    
    pct_df.sort_values(["time_label", "achievementquintile"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)
    
    # 10. Insight metrics from Winter → Spring
    if len(time_order) >= 2:
        t_prev, t_curr = time_order[0], time_order[-1]
        
        def pct_for(bucket_list, tlabel):
            return pct_df[
                (pct_df["time_label"] == tlabel)
                & (pct_df["achievementquintile"].isin(bucket_list))
            ]["pct"].sum()
        
        hi_curr = pct_for(hf.NWEA_HIGH_GROUP, t_curr)
        hi_prev = pct_for(hf.NWEA_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.NWEA_LOW_GROUP, t_curr)
        lo_prev = pct_for(hf.NWEA_LOW_GROUP, t_prev)
        high_curr = pct_for(["High"], t_curr)
        high_prev = pct_for(["High"], t_prev)
        
        score_curr = float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0])
        score_prev = float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])
        
        metrics = {
            "t_prev": t_prev, "t_curr": t_curr,
            "hi_now": hi_curr, "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr, "lo_delta": lo_curr - lo_prev,
            "score_now": score_curr, "score_delta": score_curr - score_prev,
            "high_now": high_curr, "high_delta": high_curr - high_prev,
        }
    else:
        metrics = {
            "t_prev": None, "t_curr": time_order[-1] if time_order else None,
            "hi_now": None, "hi_delta": None,
            "lo_now": None, "lo_delta": None,
            "score_now": None, "score_delta": None,
            "high_now": None, "high_delta": None,
        }
    
    return pct_df, score_df, metrics, time_order


def plot_nwea_dual_subject_winter_spring_dashboard(
    df: pd.DataFrame, cfg: dict, output_dir: str, figsize: tuple = (16, 9),
    school_raw: str = None, scope_label: str = None, preview: bool = False
):
    """Faceted dashboard showing Reading and Math Winter vs Spring snapshot."""
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    
    subjects = ["Reading", "Math K-12"]
    titles = ["Reading", "Math"]
    
    def draw_stacked_bar(ax, pct_df):
        stack_df = (
            pct_df.pivot(index="time_label", columns="achievementquintile", values="pct")
            .reindex(columns=hf.NWEA_ORDER)
            .fillna(0)
        )
        x_labels = stack_df.index.tolist()
        x = np.arange(len(x_labels))
        cumulative = np.zeros(len(stack_df))
        for cat in hf.NWEA_ORDER:
            band_vals = stack_df[cat].to_numpy()
            bars = ax.bar(x, band_vals, bottom=cumulative, color=hf.NWEA_COLORS[cat],
                         edgecolor="white", linewidth=1.2)
            for idx, rect in enumerate(bars):
                h = band_vals[idx]
                if h >= LABEL_MIN_PCT:
                    bottom_before = cumulative[idx]
                    label_color = "white" if cat in ("High", "HiAvg", "Low") else "#434343"
                    ax.text(rect.get_x() + rect.get_width() / 2, bottom_before + h / 2,
                           f"{h:.1f}%", ha="center", va="center", fontsize=8,
                           fontweight="bold", color=label_color)
            cumulative += band_vals
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of Students")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    def draw_score_bar(ax, pct_df, score_df):
        x = np.arange(len(score_df["time_label"]))
        vals = score_df["avg_score"].to_numpy()
        bars = ax.bar(x, vals, color=hf.default_quintile_colors[4], edgecolor="white", linewidth=1.2)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.1f}",
                   ha="center", va="bottom", fontsize=9, fontweight="bold", color="#434343")
        
        if "N_total" in pct_df.columns:
            n_map = pct_df.groupby("time_label")["N_total"].max().reset_index().rename(columns={"N_total": "n"})
            label_map = {row["time_label"]: f"{row['time_label']}\n(n = {int(row['n'])})"
                        for _, row in n_map.iterrows() if not pd.isna(row["n"])}
            x_labels = [label_map.get(lbl, str(lbl)) for lbl in score_df["time_label"]]
        else:
            x_labels = score_df["time_label"].astype(str).tolist()
        
        ax.set_ylabel("Avg RIT")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    def draw_insight_card(ax, metrics):
        ax.axis("off")
        if metrics.get("t_prev") and metrics.get("t_curr"):
            t_prev, t_curr = metrics["t_prev"], metrics["t_curr"]
            def _fmt_delta(val):
                return "N/A" if val is None or pd.isna(val) else f"{val:+.1f}"
            insight_lines = [
                f"Change from {t_prev} to {t_curr}:",
                f"Δ High: {_fmt_delta(metrics.get('high_delta'))} ppts",
                f"Δ Avg+HiAvg+High: {_fmt_delta(metrics.get('hi_delta'))} ppts",
                f"Δ Low: {_fmt_delta(metrics.get('lo_delta'))} ppts",
                f"Δ Avg RIT: {_fmt_delta(metrics.get('score_delta'))} pts"
            ]
        else:
            insight_lines = ["Not enough data for Winter→Spring insights"]
        
        ax.text(0.5, 0.5, "\n".join(insight_lines), fontsize=10, fontweight="normal",
               color="#434343", ha="center", va="center", wrap=True, usetex=False,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc", linewidth=0.8))
    
    pct_dfs, score_dfs, metrics_list = [], [], []
    for subj in subjects:
        pct_df, score_df, metrics, _ = _prep_nwea_winter_spring_snapshot(df, subj)
        pct_dfs.append(pct_df)
        score_dfs.append(score_df)
        metrics_list.append(metrics)
    
    if any((pct_df.empty or score_df.empty) for pct_df, score_df in zip(pct_dfs, score_dfs)):
        label = scope_label or cfg.get("district_name", ["Districtwide"])[0]
        print(f"[Section 0.1] Skipped Winter/Spring snapshot for {label} (missing data)")
        plt.close(fig)
        return
    
    for i, (pct_df, score_df, metrics, title) in enumerate(zip(pct_dfs, score_dfs, metrics_list, titles)):
        ax1 = fig.add_subplot(gs[0, i])
        draw_stacked_bar(ax1, pct_df)
        ax1.set_title(title, fontsize=14, fontweight="bold", pad=30)
        
        ax2 = fig.add_subplot(gs[1, i])
        draw_score_bar(ax2, pct_df, score_df)
        ax2.set_title("Avg RIT (Winter vs Spring)", fontsize=8, fontweight="bold", pad=10)
        
        ax3 = fig.add_subplot(gs[2, i])
        draw_insight_card(ax3, metrics)
    
    legend_handles = [Patch(facecolor=hf.NWEA_COLORS[q], edgecolor="none", label=q) for q in hf.NWEA_ORDER]
    fig.legend(handles=legend_handles, labels=hf.NWEA_ORDER, loc="upper center",
              bbox_to_anchor=(0.5, 0.925), ncol=len(hf.NWEA_ORDER), frameon=False,
              fontsize=9, handlelength=1.5, handletextpad=0.4, columnspacing=1.0)
    
    if school_raw:
        school_display = hf._safe_normalize_school_name(school_raw, cfg)
    else:
        school_display = cfg.get("district_name", ["Districtwide"])[0]
    
    main_label = school_display
    fig.suptitle(f"{main_label} • Winter to Spring Performance Snapshot", fontsize=20, fontweight="bold", y=1)
    
    charts_dir = Path(output_dir) / "charts"
    folder_name = "_district" if school_raw is None else school_display.replace(" ", "_")
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scope = scope_label or main_label
    out_name = f"{scope.replace(' ', '_')}_section0_1_dual_subject_winter_spring_snapshot.png"
    out_path = out_dir / out_name
    
    hf._save_and_render(fig, out_path)
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope,
        "subjects": subjects,
        "metrics": metrics_list,
        "pct_data": [
            {
                "subject": subj,
                "data": pct_df.to_dict('records') if not pct_df.empty else []
            }
            for subj, pct_df in zip(subjects, pct_dfs)
        ],
        "score_data": [
            {
                "subject": subj,
                "data": score_df.to_dict('records') if not score_df.empty else []
            }
            for subj, score_df in zip(subjects, score_dfs)
        ]
    }
    
    track_chart(out_name, str(out_path), scope=scope, section=0.1, chart_data=chart_data)
    print(f"Saved Section 0.1: {out_path}")
    
    if preview:
        plt.show()
    plt.close()
    
    return str(out_path)


def _run_section0_1_spring(nwea_base, cfg, output_dir):
    """Run Section 0.1 for all scopes"""
    chart_paths = []
    # District-level snapshot
    scope_label_01 = cfg.get("district_name", ["Districtwide"])[0]
    path = plot_nwea_dual_subject_winter_spring_dashboard(
        nwea_base.copy(), cfg, output_dir, figsize=(16, 9), scope_label=scope_label_01
    )
    if path:
        chart_paths.append(path)
    
    # School-level snapshots
    school_col = _get_school_column(nwea_base)
    if school_col:
        for raw_school in sorted(nwea_base[school_col].dropna().unique()):
            scope_df = nwea_base[nwea_base[school_col] == raw_school].copy()
        scope_label = hf._safe_normalize_school_name(raw_school, cfg)
        path = plot_nwea_dual_subject_winter_spring_dashboard(
            scope_df.copy(), cfg, output_dir, figsize=(16, 9),
            school_raw=raw_school, scope_label=scope_label
        )
        if path:
            chart_paths.append(path)
    
    return chart_paths


# ---------------------------------------------------------------------
# Section 4: Overall Growth Trends by Site (CGP + CGI) - Fall→Winter
# ---------------------------------------------------------------------

def _prep_cgp_trend(df: pd.DataFrame, subject_str: str, cfg: dict) -> pd.DataFrame:
    """Return tidy frame with columns: scope_label, time_label, median_cgp, mean_cgi"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == "SPRING"].copy()
    
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
    elif "reading" in subj_norm:
        d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
    
    if "falltowinterconditionalgrowthpercentile" not in d.columns:
        return pd.DataFrame(columns=["scope_label", "time_label", "median_cgp", "mean_cgi"])
    
    d = d[d["falltowinterconditionalgrowthpercentile"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["scope_label", "time_label", "median_cgp", "mean_cgi"])
    
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
    d["subject"] = subject_str
    
    school_col = _get_school_column(d)
    if school_col:
        d["site_display"] = d[school_col].apply(lambda x: hf._safe_normalize_school_name(x, cfg))
    else:
        d["site_display"] = cfg.get("district_name", ["District (All Students)"])[0]
    dist_rows = d.copy()
    dist_rows["site_display"] = cfg.get("district_name", ["District (All Students)"])[0]
    
    both = pd.concat([d, dist_rows], ignore_index=True)
    has_cgi = "falltowinterconditionalgrowthindex" in both.columns
    grp_cols = ["site_display", "time_label"]
    
    if has_cgi:
        out = both.groupby(grp_cols, dropna=False).agg(
            median_cgp=("falltowinterconditionalgrowthpercentile", "median"),
            mean_cgi=("falltowinterconditionalgrowthindex", "mean"),
        ).reset_index()
    else:
        out = both.groupby(grp_cols, dropna=False).agg(
            median_cgp=("falltowinterconditionalgrowthpercentile", "median")
        ).reset_index()
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


def _plot_cgp_trend(df, subject_str, scope_label, ax=None, cfg=None):
    """Bars = median CGP, line = mean CGI with blended transform."""
    sub = df[df["scope_label"] == scope_label].copy()
    if sub.empty:
        return
    
    sub["time_label"] = pd.Categorical(
        sub["time_label"], categories=sorted(sub["time_label"].astype(str).unique()), ordered=True
    )
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
    
    ax.set_ylabel("Median Fall→Winter CGP")
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
        ax.add_line(mlines.Line2D([x0, x1], [yb, yb], transform=blend, linestyle="--",
                                  color="#eab308", linewidth=1.2))
    
    cgi_line = mlines.Line2D(x_vals, y_cgi, transform=blend, marker="o", linewidth=2,
                            markersize=6, color="#ffa800", zorder=3)
    ax.add_line(cgi_line)
    for xv, yv in zip(x_vals, y_cgi):
        if pd.notna(yv):
            ax.text(xv, yv + (0.12 if yv >= 0 else -0.12), f"{yv:.2f}", transform=blend,
                   ha="center", va="bottom" if yv >= 0 else "top", fontsize=8,
                   fontweight="bold", color="#ffa800")
    
    ax2.set_ylabel("Avg Fall→Winter CGI")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax.set_title(f"{subject_str}", fontweight="bold", fontsize=14, pad=10)


def _save_cgp_chart(fig, scope_label, output_dir, cfg, section_num=4, suffix="cgp_winter_to_spring_dualpanel", chart_data=None):
    charts_dir = Path(output_dir) / "charts"
    folder_name = (
        "_district" if scope_label == cfg.get("district_name", ["District (All Students)"])[0]
        else scope_label.replace(" ", "_")
    )
    out_dir = charts_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_scope = scope_label.replace(" ", "_")
    out_name = f"{safe_scope}_section{section_num}_{suffix}.png"
    out_path = out_dir / out_name
    hf._save_and_render(fig, out_path)
    # Pass chart_data if provided, otherwise pass empty dict to ensure JSON file is created
    track_chart(out_name, str(out_path), scope=scope_label, section=section_num, chart_data=chart_data or {})
    print(f"Saved: {out_path}")
    return str(out_path)


def _run_cgp_dual_trend(scope_df, scope_label, cfg, output_dir):
    cgp_trend = pd.concat(
        [_prep_cgp_trend(scope_df, subj, cfg) for subj in ["Reading", "Mathematics"]],
        ignore_index=True,
    )
    if cgp_trend.empty:
        print(f"[skip] No CGP data for {scope_label}")
        return None
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle(f"{scope_label} • Fall→Winter Growth (All Students)", fontsize=20, fontweight="bold", y=0.99)
    
    axes, n_labels_axes = [], []
    for i, subject_str in enumerate(["Reading", "Mathematics"]):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        sub_df = cgp_trend[(cgp_trend["scope_label"] == scope_label) & (cgp_trend["subject"] == subject_str)]
        if not sub_df.empty:
            _plot_cgp_trend(sub_df, subject_str, scope_label, ax=ax, cfg=cfg)
        
        # Add n-count labels
        subj_norm = subject_str.strip().casefold()
        d = scope_df.copy()
        d = d[d["testwindow"].astype(str).str.upper() == "SPRING"].copy()
        if "math" in subj_norm:
            d = d[d["course"].astype(str).str.contains("math", case=False, na=False)].copy()
        elif "reading" in subj_norm:
            d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
        
        d["year_short"] = d["year"].apply(_short_year)
        d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
        school_col = _get_school_column(d)
        if school_col:
            d["site_display"] = d[school_col].apply(lambda x: hf._safe_normalize_school_name(x, cfg))
        else:
            d["site_display"] = scope_label
        if scope_label == cfg.get("district_name", ["District (All Students)"])[0]:
            d["site_display"] = cfg.get("district_name", ["District (All Students)"])[0]
        d = d[d["site_display"] == scope_label]
        
        n_map = d.groupby("time_label")["uniqueidentifier"].nunique().reset_index().rename(columns={"uniqueidentifier": "n"})
        n_map_dict = dict(zip(n_map["time_label"], n_map["n"]))
        ticklabels = [str(lbl) for lbl in sub_df["time_label"]]
        labels_with_n = [f"{lbl}\n(n = {int(n_map_dict.get(lbl, 0))})" for lbl in ticklabels]
        ax.set_xticklabels(labels_with_n)
        ax.tick_params(axis="x", pad=10)
        n_labels_axes.append(ax)
    
    legend_handles = [
        Patch(facecolor="#0381a2", edgecolor="white", label="Median CGP"),
        Line2D([0], [0], color="#ffa800", marker="o", linewidth=2, markersize=6, label="Mean CGI"),
    ]
    fig.legend(handles=legend_handles, labels=["Median CGP", "Mean CGI"], loc="upper center",
              bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, handlelength=2, handletextpad=0.5, columnspacing=1.2)
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "subjects": ["Reading", "Mathematics"],
        "cgp_data": cgp_trend.to_dict('records') if not cgp_trend.empty else []
    }
    
    return _save_cgp_chart(fig, scope_label, output_dir, cfg, chart_data=chart_data)


def _run_section4_spring(nwea_base, cfg, output_dir, scopes):
    """Run Section 4 for all scopes"""
    chart_paths = []
    district_label = cfg.get("district_name", ["Districtwide"])[0]
    path = _run_cgp_dual_trend(nwea_base.copy(), district_label, cfg, output_dir)
    if path:
        chart_paths.append(path)
    
    for scope_df, scope_label, folder in scopes:
        if folder != "_district":
            path = _run_cgp_dual_trend(scope_df.copy(), scope_label, cfg, output_dir)
            if path:
                chart_paths.append(path)
    
    return chart_paths


# ---------------------------------------------------------------------
# Section 5: CGP/CGI Growth: Grade Trend + Backward Cohort - Fall→Winter
# ---------------------------------------------------------------------

def _prep_cgp_by_grade(df, subject, grade):
    d = df.copy()
    d["testwindow"] = d["testwindow"].astype(str)
    d = d[d["testwindow"].str.upper() == "WINTER"]
    d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
    d = d[d["grade"] == grade]
    
    subj = subject.lower()
    d["course"] = d["course"].astype(str)
    if "math" in subj:
        d = d[d["course"].str.contains("math", case=False, na=False)]
    else:
        d = d[d["course"].str.contains("read", case=False, na=False)]
    
    d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    d = d.dropna(subset=["teststartdate"])
    d = d.sort_values("teststartdate").drop_duplicates("uniqueidentifier", keep="last")
    
    d = d.dropna(subset=["falltowinterconditionalgrowthpercentile", "falltowinterconditionalgrowthindex"])
    
    if d.empty:
        return pd.DataFrame(columns=["time_label", "median_cgp", "mean_cgi"])
    
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = "Gr " + d["grade"].astype(int).astype(str) + " • Winter " + d["year_short"]
    
    out = d.groupby("time_label").agg(
        median_cgp=("falltowinterconditionalgrowthpercentile", "median"),
        mean_cgi=("falltowinterconditionalgrowthindex", "mean"),
    ).reset_index()
    
    def _extract_year_short(label):
        try:
            return label.split("Winter")[-1].strip()
        except:
            return ""
    
    out["year_short"] = out["time_label"].apply(_extract_year_short)
    out = out.sort_values("year_short").tail(4)
    out["time_label"] = pd.Categorical(out["time_label"], categories=out["time_label"], ordered=True)
    return out


def _plot_cgp_dual_facet(overall_df, cohort_df, grade, subject_str, scope_label, output_dir, cfg, preview=False):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.subplots_adjust(wspace=0.28)
    
    def draw_panel(df, ax, title):
        if df.empty:
            return
        df = df.copy().sort_values("time_label")
        x_vals = np.arange(len(df))
        y_cgp = df["median_cgp"].to_numpy(dtype=float)
        y_cgi = df["mean_cgi"].to_numpy(dtype=float)
        
        for y_start, y_end, color in [(0, 20, "#808080"), (20, 40, "#c5c5c5"), (40, 60, "#78daf4"),
                                      (60, 80, "#00baeb"), (80, 100, "#0381a2")]:
            ax.axhspan(y_start, y_end, facecolor=color, alpha=0.5, zorder=0)
        for yref in [42, 50, 58]:
            ax.axhline(yref, linestyle="--", color="#6B7280", linewidth=1.2, zorder=0)
        
        bars = ax.bar(x_vals, y_cgp, color="#0381a2", edgecolor="white", linewidth=1.2, zorder=3)
        for rect, yv in zip(bars, y_cgp):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() / 2, f"{yv:.1f}",
                   ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        
        labels_with_n = df["time_label"].astype(str).tolist()
        ax.set_ylabel("Median Fall→Winter CGP")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels_with_n, ha="center", fontsize=8)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, 100)
        
        ax2 = ax.twinx()
        ax2.set_ylim(-2.5, 2.5)
        ax2.set_ylabel("Avg Fall→Winter CGI")
        ax2.set_yticks([-2, -1, 0, 1, 2])
        ax2.set_zorder(ax.get_zorder() - 1)
        ax2.patch.set_alpha(0)
        
        ax.set_xlim(-0.5, len(x_vals) - 0.5)
        x0, x1 = ax.get_xlim()
        blend = mtransforms.BlendedGenericTransform(ax.transData, ax2.transData)
        
        band = mpl.patches.Rectangle((x0, -0.2), x1 - x0, 0.4, transform=blend,
                                     facecolor="#facc15", alpha=0.35, zorder=1.5)
        ax.add_patch(band)
        for yref in [-0.2, 0.2]:
            ax.add_line(mlines.Line2D([x0, x1], [yref, yref], transform=blend, linestyle="--",
                                     color="#eab308", linewidth=1.2, zorder=1.6))
        
        cgi_line = mlines.Line2D(x_vals, y_cgi, transform=blend, marker="o", linewidth=2,
                                markersize=6, color="#ffa800", zorder=3)
        ax.add_line(cgi_line)
        
        for xv, yv in zip(x_vals, y_cgi):
            if pd.isna(yv):
                continue
            ax.text(xv, yv + (0.12 if yv >= 0 else -0.12), f"{yv:.2f}", transform=blend,
                   ha="center", va="bottom" if yv >= 0 else "top", fontsize=8,
                   fontweight="bold", color="#ffa800", zorder=3.1)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
    
    draw_panel(overall_df, axs[0], "Overall Growth Trends")
    draw_panel(cohort_df, axs[1], "Cohort Growth Trends")
    
    legend_handles = [
        Patch(facecolor="#0381a2", edgecolor="white", label="Median CGP"),
        mlines.Line2D([0], [0], color="#ffa800", marker="o", linewidth=2, markersize=6, label="Mean CGI"),
    ]
    fig.legend(handles=legend_handles, labels=["Median CGP", "Mean CGI"], loc="upper center",
              bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False, handlelength=2, handletextpad=0.5, columnspacing=1.2)
    
    fig.suptitle(f"{scope_label} • {subject_str} • Grade {grade} • Fall→Winter Growth",
                fontsize=20, fontweight="bold", y=1)
    
    # Prepare chart data for saving
    chart_data = {
        "scope": scope_label,
        "subject": subject_str,
        "grade": grade,
        "overall_data": overall_df.to_dict('records') if not overall_df.empty else [],
        "cohort_data": cohort_df.to_dict('records') if not cohort_df.empty else []
    }
    
    path = _save_cgp_chart(fig, scope_label, output_dir, cfg, section_num=5,
                   suffix=f"cgp_cgi_grade_trends_grade{grade}_{subject_str.lower().replace(' ', '_')}",
                   chart_data=chart_data)
    if preview:
        plt.show()
    plt.close()
    return path


def _run_section5_spring(nwea_base, cfg, output_dir, scopes):
    """Run Section 5 for all scopes"""
    chart_paths = []
    d0 = nwea_base.copy()
    d0["year"] = pd.to_numeric(d0["year"], errors="coerce")
    d0["grade"] = pd.to_numeric(d0["grade"], errors="coerce")
    grades = sorted(d0["grade"].dropna().unique())
    subjects = ["Reading", "Mathematics"]
    
    # District-level
    district_display = cfg.get("district_name", ["Districtwide"])[0]
    for grade in grades:
        for subject in subjects:
            overall_df = _prep_cgp_by_grade(d0, subject, grade)
            if overall_df.empty:
                continue
            anchor_year = int(d0[d0["grade"] == grade]["year"].max())
            cohort_rows = []
            for offset in range(3, -1, -1):
                yr = anchor_year - offset
                gr = grade - offset
                if gr < 0:
                    continue
                d = d0.copy()
                d["testwindow"] = d["testwindow"].astype(str)
                d = d[(d["year"] == yr) & (d["grade"] == gr) & (d["testwindow"].str.upper() == "WINTER")]
                if "teststartdate" in d.columns:
                    d = d.sort_values("teststartdate").drop_duplicates(
                        subset=["uniqueidentifier", "year", "grade", "course", "subject"], keep="last")
                if subject.lower() == "mathematics":
                    d = d[d["course"] == "Math K-12"]
                else:
                    d = d[d["course"].str.contains("read", case=False, na=False)]
                d = d.dropna(subset=["falltowinterconditionalgrowthpercentile", "falltowinterconditionalgrowthindex"])
                if d.empty:
                    continue
                cohort_rows.append({
                    "gr": gr, "yr": yr,
                    "time_label": f"Gr {int(gr)} • Winter {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                    "median_cgp": d["falltowinterconditionalgrowthpercentile"].median(),
                    "mean_cgi": d["falltowinterconditionalgrowthindex"].mean(),
                })
            if not cohort_rows:
                continue
            cohort_df = pd.DataFrame(cohort_rows)
            cohort_df = cohort_df.sort_values(["gr", "yr"])
            ordered_labels = cohort_df["time_label"].tolist()
            cohort_df["time_label"] = pd.Categorical(cohort_df["time_label"], categories=ordered_labels, ordered=True)
            path = _plot_cgp_dual_facet(overall_df, cohort_df, grade, subject, district_display, output_dir, cfg)
            if path:
                chart_paths.append(path)
    
    # School-level
    for scope_df, scope_label, folder in scopes:
        if folder == "_district":
            continue
        d0_school = scope_df.copy()
        d0_school["year"] = pd.to_numeric(d0_school["year"], errors="coerce")
        d0_school["grade"] = pd.to_numeric(d0_school["grade"], errors="coerce")
        for grade in grades:
            for subject in subjects:
                overall_df = _prep_cgp_by_grade(d0_school, subject, grade)
                if overall_df.empty:
                    continue
                anchor_year = d0_school.loc[d0_school["grade"] == grade, "year"].max()
                if pd.isna(anchor_year):
                    continue
                anchor_year = int(anchor_year)
                cohort_rows = []
                for offset in range(3, -1, -1):
                    yr = anchor_year - offset
                    gr = grade - offset
                    if gr < 0:
                        continue
                    d = d0_school.copy()
                    d["testwindow"] = d["testwindow"].astype(str)
                    d = d[(d["year"] == yr) & (d["grade"] == gr) & (d["testwindow"].str.upper() == "WINTER")]
                    if "teststartdate" in d.columns:
                        d = d.sort_values("teststartdate").drop_duplicates(
                            subset=["uniqueidentifier", "year", "grade", "course", "subject"], keep="last")
                    if subject.lower() == "mathematics":
                        d = d[d["course"] == "Math K-12"]
                    else:
                        d = d[d["course"].str.contains("read", case=False, na=False)]
                    d = d.dropna(subset=["falltowinterconditionalgrowthpercentile", "falltowinterconditionalgrowthindex"])
                    if d.empty:
                        continue
                    cohort_rows.append({
                        "gr": gr, "yr": yr,
                        "time_label": f"Gr {int(gr)} • Winter {str(yr - 1)[-2:]}-{str(yr)[-2:]}",
                        "median_cgp": d["falltowinterconditionalgrowthpercentile"].median(),
                        "mean_cgi": d["falltowinterconditionalgrowthindex"].mean(),
                    })
                if not cohort_rows:
                    continue
                cohort_df = pd.DataFrame(cohort_rows)
                cohort_df = cohort_df.sort_values(["gr", "yr"])
                ordered_labels = cohort_df["time_label"].tolist()
                cohort_df["time_label"] = pd.Categorical(cohort_df["time_label"], categories=ordered_labels, ordered=True)
                path = _plot_cgp_dual_facet(overall_df, cohort_df, grade, subject, scope_label, output_dir, cfg)
                if path:
                    chart_paths.append(path)
    
    return chart_paths


# ---------------------------------------------------------------------
# Main Execution for Winter Charts
# ---------------------------------------------------------------------

def main(nwea_data=None):
    """
    Main function to generate NWEA Winter charts
    
    Args:
        nwea_data: Optional list of dicts or DataFrame with NWEA data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate NWEA Winter charts')
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
    if nwea_data is not None:
        nwea_base = load_nwea_data(nwea_data=nwea_data)
    else:
        if not args.data_dir:
            raise ValueError("Either nwea_data must be provided or --data-dir must be specified")
        nwea_base = load_nwea_data(data_dir=args.data_dir)
    
    # Always use Spring for this module
    selected_quarters = ["Spring"]
    
    # Apply filters to base data
    if chart_filters:
        nwea_base = apply_chart_filters(nwea_base, chart_filters)
        print(f"Data after filtering: {nwea_base.shape[0]:,} rows")
    
    # Get scopes
    scopes = get_scopes(nwea_base, cfg)
    
    chart_paths = []
    
    # Section 0: Predicted vs Actual CAASPP (Spring)
    print("\n[Section 0] Generating Spring Predicted vs Actual CAASPP...")
    for scope_df, scope_label, folder in scopes:
        try:
            payload = {}
            for subj in ["Reading", "Mathematics"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                proj, act, metrics, _ = _prep_section0_spring(scope_df, subj)
                if proj is None:
                    continue
                payload[subj] = {"proj_pct": proj, "act_pct": act, "metrics": metrics}
            if payload:
                _plot_section0_dual(scope_label, folder, args.output_dir, payload, preview=hf.DEV_MODE)
                chart_paths.append(str(Path(args.output_dir) / folder / f"{scope_label.replace(' ', '_')}_section0_pred_vs_actual_{folder}.png"))
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 1: Spring Performance Trends
    print("\n[Section 1] Generating Spring Performance Trends...")
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
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Section 2: Student Group Performance Trends (Spring)
    print("\n[Section 2] Generating Student Group Performance Trends (Spring)...")
    student_groups_cfg = cfg.get("student_groups", {})
    group_order = cfg.get("student_group_order", {})
    max_groups = chart_filters.get("max_student_groups", 10)  # Default limit
    sorted_groups = sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99))
    groups_to_plot = sorted_groups[:max_groups]
    
    for scope_df, scope_label, folder in scopes:
        for group_name, group_def in groups_to_plot:
            if group_def.get("type") == "all":
                continue
            if not should_generate_student_group(group_name, chart_filters):
                continue
            for quarter in selected_quarters:
                try:
                    plot_nwea_subject_dashboard_by_group(
                        scope_df.copy(), None, quarter, group_name, group_def,
                        args.output_dir, cfg, figsize=(16, 9),
                        school_raw=None if folder == "_district" else (scope_df[_get_school_column(scope_df)].iloc[0] if _get_school_column(scope_df) else None),
                        scope_label=scope_label, preview=hf.DEV_MODE)
                except Exception as e:
                    print(f"Error generating Section 2 chart for {scope_label} - {group_name} ({quarter}): {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 3: Overall + Cohort Trends (Winter)
    print("\n[Section 3] Generating Overall + Cohort Trends (Winter)...")
    selected_grades = chart_filters.get("grades", [])
    if not selected_grades:
        selected_grades = list(range(0, 13))  # Default to all grades
    
    def _run_scope_section3_spring(scope_df, scope_label, school_raw):
        scope_df = scope_df.copy()
        scope_df["year"] = pd.to_numeric(scope_df["year"], errors="coerce")
        
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
        
        unique_grades = sorted([g for g in scope_df["grade_normalized"].dropna().unique() if g is not None and g in selected_grades])
        
        for g in unique_grades:
            grade_df = scope_df[scope_df["grade_normalized"] == g].copy()
            if grade_df.empty:
                continue
            
            courses_in_data = set(grade_df["course"].dropna().unique())
            for course_str in ["Math K-12", "Reading"]:
                subject_name = "Mathematics" if course_str == "Math K-12" else "Reading"
                if not should_generate_subject(subject_name, chart_filters):
                    continue
                
                if (course_str == "Math K-12" and "Math K-12" in courses_in_data) or \
                   (course_str == "Reading" and any(str(c).startswith("Reading") for c in courses_in_data)):
                    for quarter in selected_quarters:
                        try:
                            plot_nwea_blended_dashboard(
                                grade_df.copy(), course_str=course_str, current_grade=int(g),
                                window_filter=quarter, cohort_year=anchor_year, output_dir=args.output_dir, cfg=cfg,
                                figsize=(16, 9), school_raw=school_raw, preview=hf.DEV_MODE, scope_label=scope_label)
                        except Exception as e:
                            print(f"Error generating Section 3 chart for {scope_label} - Grade {g} - {course_str} ({quarter}): {e}")
                            if hf.DEV_MODE:
                                import traceback
                                traceback.print_exc()
                            continue
    
    for scope_df, scope_label, folder in scopes:
        _run_scope_section3_spring(scope_df.copy(), scope_label, None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None)
    
    # Section 0.1: Winter vs Spring Performance Snapshot (Spring-specific)
    print("\n[Section 0.1] Generating Winter vs Spring Performance Snapshot...")
    section0_1_paths = _run_section0_1_spring(nwea_base.copy(), cfg, args.output_dir)
    chart_paths.extend(section0_1_paths)
    
    # Section 4: Overall Growth Trends by Site (CGP + CGI) - Winter→Spring (Spring-specific)
    print("\n[Section 4] Generating Overall Growth Trends by Site (Winter→Spring CGP/CGI)...")
    section4_paths = _run_section4_spring(nwea_base.copy(), cfg, args.output_dir, scopes)
    chart_paths.extend(section4_paths)
    
    # Section 5: CGP/CGI Growth: Grade Trend + Backward Cohort - Winter→Spring (Spring-specific)
    print("\n[Section 5] Generating CGP/CGI Growth: Grade Trend + Backward Cohort (Winter→Spring)...")
    section5_paths = _run_section5_spring(nwea_base.copy(), cfg, args.output_dir, scopes)
    chart_paths.extend(section5_paths)
    
    return chart_paths

def generate_nwea_spring_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    nwea_data: list = None
) -> list:
    """
    Generate NWEA Spring charts (wrapper function for Flask backend)
    
    Args:
        partner_name: Partner name
        output_dir: Output directory for charts
        config: Partner configuration dict
        chart_filters: Chart filters dict
        data_dir: Data directory path (used only if nwea_data is None)
        nwea_data: Optional list of dicts with NWEA data (preferred over CSV loading)
    
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
            self.data_dir = data_dir if nwea_data is None else None
            self.output_dir = output_dir
            self.dev_mode = 'true' if hf.DEV_MODE else 'false'
            self.config = json.dumps(cfg) if cfg else '{}'
    
    args = Args()
    
    old_argv = sys.argv
    try:
        sys.argv = [
            'nwea_spring.py',
            '--partner', args.partner,
            '--output-dir', args.output_dir,
            '--dev-mode', args.dev_mode,
            '--config', args.config
        ]
        if args.data_dir:
            sys.argv.extend(['--data-dir', args.data_dir])
        
        chart_paths = main(nwea_data=nwea_data)
    finally:
        sys.argv = old_argv
    
    return chart_paths

