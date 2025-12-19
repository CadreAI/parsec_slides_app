"""
iReady Winter chart generation module
Generates charts specifically for Winter (MOY) test window
Note: This is a simplified version - full implementation can be expanded from iready_moy.py
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

# Use iReady-specific helper utilities + styling
from . import helper_functions_iready as hf

# Import utility modules
from .iready_data import (
    load_config_from_args,
    load_iready_data,
    get_scopes,
    prep_iready_for_charts
)
from .iready_filters import (
    apply_chart_filters,
    should_generate_subject,
    should_generate_student_group
)

# Import shared plotting functions from iready_charts
from .iready_charts import (
    plot_dual_subject_dashboard,
    plot_iready_subject_dashboard_by_group,
    plot_iready_blended_dashboard,
    _apply_student_group_mask,
    _find_column_fuzzy
)

# Chart tracking for CSV generation
chart_links = []
_chart_tracking_set = set()

# LABEL_MIN_PCT for inline labels
LABEL_MIN_PCT = 5.0

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
            import json
            import numpy as np
            import pandas as pd
            
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

def main(iready_data=None):
    """
    Main function to generate iReady Winter charts
    
    Args:
        iready_data: Optional list of dicts or DataFrame with iReady data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate iReady Winter charts')
    parser.add_argument('--partner', required=True, help='Partner name')
    parser.add_argument('--data-dir', required=False, help='Data directory path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--dev-mode', default='false', help='Development mode')
    parser.add_argument('--config', default='{}', help='Config JSON string')
    
    args = parser.parse_args()
    
    cfg = load_config_from_args(args.config)
    hf.DEV_MODE = args.dev_mode.lower() in ('true', '1', 'yes', 'on')
    
    chart_filters = cfg.get("chart_filters", {})
    # Ensure chart_filters is a dict, not a string
    if isinstance(chart_filters, str):
        try:
            chart_filters = json.loads(chart_filters)
        except:
            print(f"[Warning] Could not parse chart_filters from config as JSON: {chart_filters}")
            chart_filters = {}
    
    # Load data
    if iready_data is not None:
        iready_base = load_iready_data(iready_data=iready_data)
    else:
        if not args.data_dir:
            raise ValueError("Either iready_data must be provided or --data-dir must be specified")
        iready_base = load_iready_data(data_dir=args.data_dir)
    
    # Apply filters to base data
    if chart_filters:
        iready_base = apply_chart_filters(iready_base, chart_filters)
        print(f"Data after filtering: {iready_base.shape[0]:,} rows")
    
    # Get scopes from unfiltered data (all sections filter internally)
    scopes = get_scopes(iready_base, cfg)
    
    chart_paths = []
    
    # Section 1: Winter Performance Trends
    print("\n[Section 1] Generating Winter Performance Trends...")
    for scope_df, scope_label, folder in scopes:
        try:
            chart_path = plot_dual_subject_dashboard(
                scope_df,
                scope_label,
                folder,
                args.output_dir,
                window_filter="Winter",
                preview=hf.DEV_MODE
            )
            if chart_path:
                chart_paths.append(chart_path)
        except Exception as e:
            print(f"Error generating chart for {scope_label} (Winter): {e}")
            continue
    
    # Section 2: Student Group Performance Trends
    print("\n" + "="*80)
    print("[Section 2] Generating Student Group Performance Trends...")
    print("="*80)
    student_groups_cfg = cfg.get("student_groups", {})
    race_ethnicity_cfg = cfg.get("race_ethnicity", {})
    group_order = cfg.get("student_group_order", {})
    
    print(f"[Section 2] Found {len(student_groups_cfg)} student groups in config")
    print(f"[Section 2] Found {len(race_ethnicity_cfg)} race/ethnicity groups in config")
    print(f"[Section 2] Processing {len(scopes)} scope(s)")
    
    for scope_idx, (scope_df, scope_label, folder) in enumerate(scopes, 1):
        print(f"\n[Section 2] Processing scope {scope_idx}/{len(scopes)}: {scope_label} (folder: {folder})")
        print(f"[Section 2]   Initial data rows: {len(scope_df):,}")
        
        # Check for Winter data in scope
        if "testwindow" in scope_df.columns:
            winter_rows = scope_df[scope_df["testwindow"].astype(str).str.upper() == "WINTER"]
            print(f"[Section 2]   Rows with Winter testwindow: {len(winter_rows):,}")
            if len(winter_rows) == 0:
                available_windows = scope_df["testwindow"].astype(str).str.upper().unique()
                print(f"[Section 2]   ⚠️  WARNING: No Winter data found in scope!")
                print(f"[Section 2]   Available testwindow values: {sorted(available_windows)}")
        
        # Process regular student groups
        print(f"[Section 2]   Processing {len(student_groups_cfg)} student group(s)...")
        for group_idx, (group_name, group_def) in enumerate(sorted(student_groups_cfg.items(), key=lambda kv: group_order.get(kv[0], 99)), 1):
            # Skip "All Students" - it's handled in Section 1
            if group_def.get("type") == "all":
                print(f"[Section 2]     [{group_idx}] Skipping '{group_name}' (type='all' - handled in Section 1)")
                continue
            
            # Only generate charts for selected student groups
            should_gen = should_generate_student_group(group_name, chart_filters)
            if not should_gen:
                print(f"[Section 2]     [{group_idx}] Skipping '{group_name}' (not selected in chart_filters)")
                continue
            
            print(f"[Section 2]     [{group_idx}] Processing '{group_name}'...")
            print(f"[Section 2]       Group definition: {group_def}")
            
            try:
                # Log data before group mask
                scope_df_copy = scope_df.copy()
                print(f"[Section 2]       Data rows before group mask: {len(scope_df_copy):,}")
                
                # Check if group column exists (with fuzzy matching)
                if group_def.get("type") != "all":
                    group_col = group_def.get("column")
                    if group_col:
                        # Try exact match first
                        if group_col not in scope_df_copy.columns:
                            # Try fuzzy matching
                            group_col_found = _find_column_fuzzy(scope_df_copy, group_col)
                            if group_col_found:
                                print(f"[Section 2]       Found column '{group_col_found}' using fuzzy match for '{group_col}'")
                                # Update group_def to use found column
                                group_def = group_def.copy()
                                group_def["column"] = group_col_found
                                group_col = group_col_found
                            else:
                                print(f"[Section 2]       ❌ ERROR: Group column '{group_col}' not found in data (even with fuzzy matching)!")
                                print(f"[Section 2]       Available columns: {sorted(scope_df_copy.columns.tolist())[:30]}...")
                                # Show similar column names
                                target_norm = str(group_col).lower().replace("_", "").replace("-", "").replace(" ", "")
                                similar = [c for c in scope_df_copy.columns 
                                          if target_norm in str(c).lower().replace("_", "").replace("-", "").replace(" ", "")]
                                if similar:
                                    print(f"[Section 2]       Similar columns found: {similar}")
                                continue
                        
                        # Log unique values in group column
                        unique_vals = scope_df_copy[group_col].dropna().unique()
                        print(f"[Section 2]       Unique values in '{group_col}': {len(unique_vals)} values")
                        if len(unique_vals) <= 10:
                            print(f"[Section 2]       Values: {sorted([str(v) for v in unique_vals])}")
                        
                        # Log expected values
                        expected_vals = group_def.get("in", [])
                        print(f"[Section 2]       Expected values for group: {expected_vals}")
                
                chart_path = plot_iready_subject_dashboard_by_group(
                    scope_df_copy, scope_label, folder, args.output_dir,
                    window_filter="Winter", group_name=group_name, group_def=group_def,
                    cfg=cfg, preview=hf.DEV_MODE
                )
                
                if chart_path:
                    chart_paths.append(chart_path)
                    print(f"[Section 2]       ✅ Successfully generated chart: {chart_path}")
                else:
                    print(f"[Section 2]       ⚠️  Chart generation returned None (likely skipped due to data constraints)")
                    
            except Exception as e:
                print(f"[Section 2]       ❌ ERROR generating chart for '{group_name}': {e}")
                print(f"[Section 2]       Error type: {type(e).__name__}")
                if hf.DEV_MODE:
                    import traceback
                    print(f"[Section 2]       Full traceback:")
                    traceback.print_exc()
                else:
                    print(f"[Section 2]       Enable DEV_MODE for full traceback")
                continue
        
        # Process race/ethnicity groups if race filter is specified
        race_filters = chart_filters.get("race", []) if chart_filters else []
        print(f"\n[Section 2]   Race filters specified: {race_filters}")
        
        if race_filters and race_ethnicity_cfg:
            print(f"[Section 2]   Processing {len(race_filters)} race/ethnicity group(s)...")
            for race_idx, race_name in enumerate(race_filters, 1):
                print(f"[Section 2]     [{race_idx}] Processing race group: '{race_name}'...")
                
                race_def = race_ethnicity_cfg.get(race_name)
                if not race_def:
                    print(f"[Section 2]       ❌ ERROR: Race group '{race_name}' not found in race_ethnicity config")
                    print(f"[Section 2]       Available race groups: {list(race_ethnicity_cfg.keys())}")
                    continue
                
                print(f"[Section 2]       Race definition: {race_def}")
                
                # Create a combined group_def for race
                combined_group_def = {
                    "column": race_def.get("column"),
                    "in": race_def.get("values", race_def.get("in", [])),
                    "type": "race"
                }
                
                try:
                    # Log data before race mask
                    scope_df_copy = scope_df.copy()
                    print(f"[Section 2]       Data rows before race mask: {len(scope_df_copy):,}")
                    
                    # Check if race column exists (with fuzzy matching)
                    race_col = combined_group_def.get("column")
                    if race_col:
                        # Try exact match first
                        if race_col not in scope_df_copy.columns:
                            # Try fuzzy matching
                            race_col_found = _find_column_fuzzy(scope_df_copy, race_col)
                            if race_col_found:
                                print(f"[Section 2]       Found column '{race_col_found}' using fuzzy match for '{race_col}'")
                                # Update combined_group_def to use found column
                                combined_group_def = combined_group_def.copy()
                                combined_group_def["column"] = race_col_found
                                race_col = race_col_found
                            else:
                                print(f"[Section 2]       ❌ ERROR: Race column '{race_col}' not found in data (even with fuzzy matching)!")
                                print(f"[Section 2]       Available columns: {sorted(scope_df_copy.columns.tolist())[:30]}...")
                                # Show similar column names
                                target_norm = str(race_col).lower().replace("_", "").replace("-", "").replace(" ", "")
                                similar = [c for c in scope_df_copy.columns 
                                          if target_norm in str(c).lower().replace("_", "").replace("-", "").replace(" ", "")]
                                if similar:
                                    print(f"[Section 2]       Similar columns found: {similar}")
                                continue
                        
                        # Log unique values in race column
                        unique_vals = scope_df_copy[race_col].dropna().unique()
                        print(f"[Section 2]       Unique values in '{race_col}': {len(unique_vals)} values")
                        if len(unique_vals) <= 10:
                            print(f"[Section 2]       Values: {sorted([str(v) for v in unique_vals])}")
                        
                        # Log expected values
                        expected_vals = combined_group_def.get("in", [])
                        print(f"[Section 2]       Expected values for race group: {expected_vals}")
                    
                    chart_path = plot_iready_subject_dashboard_by_group(
                        scope_df_copy, scope_label, folder, args.output_dir,
                        window_filter="Winter", group_name=race_name, group_def=combined_group_def,
                        cfg=cfg, preview=hf.DEV_MODE
                    )
                    
                    if chart_path:
                        chart_paths.append(chart_path)
                        print(f"[Section 2]       ✅ Successfully generated chart: {chart_path}")
                    else:
                        print(f"[Section 2]       ⚠️  Chart generation returned None (likely skipped due to data constraints)")
                        
                except Exception as e:
                    print(f"[Section 2]       ❌ ERROR generating chart for race group '{race_name}': {e}")
                    print(f"[Section 2]       Error type: {type(e).__name__}")
                    if hf.DEV_MODE:
                        import traceback
                        print(f"[Section 2]       Full traceback:")
                        traceback.print_exc()
                    else:
                        print(f"[Section 2]       Enable DEV_MODE for full traceback")
                    continue
        else:
            if not race_filters:
                print(f"[Section 2]   No race filters specified - skipping race/ethnicity groups")
            elif not race_ethnicity_cfg:
                print(f"[Section 2]   ⚠️  Race filters specified but no race_ethnicity config found")
    
    print(f"\n[Section 2] Completed. Generated {len([p for p in chart_paths if 'section2' in str(p).lower()])} Section 2 chart(s)")
    print("="*80)
    
    # Section 3: Overall + Cohort Trends
    print("\n[Section 3] Generating Overall + Cohort Trends...")
    
    def _run_scope_section3(scope_df, scope_label, folder):
        scope_df = scope_df.copy()
        scope_df["academicyear"] = pd.to_numeric(scope_df["academicyear"], errors="coerce")
        scope_df["student_grade"] = pd.to_numeric(scope_df["student_grade"], errors="coerce")
        
        # Normalize grades (K -> 0, -1 -> pre-k)
        def normalize_grade_val(grade_val):
            if pd.isna(grade_val):
                return None
            grade_str = str(grade_val).strip().upper()
            if grade_str == "K" or grade_str == "KINDERGARTEN":
                return 0
            try:
                grade_num = int(float(grade_str))
                return grade_num
            except:
                return None
        
        scope_df["grade_normalized"] = scope_df["student_grade"].apply(normalize_grade_val)
        
        if scope_df["academicyear"].notna().any():
            anchor_year = int(scope_df["academicyear"].max())
        else:
            anchor_year = None
        
        unique_grades = sorted([g for g in scope_df["grade_normalized"].dropna().unique() if g is not None])
        
        if chart_filters and chart_filters.get("grades") and len(chart_filters["grades"]) > 0:
            unique_grades = [g for g in unique_grades if g in chart_filters["grades"]]
        
        print(f"  [Section 3] Found {len(unique_grades)} grade(s) in filtered data: {unique_grades}")
        
        for g in unique_grades:
            grade_check = scope_df[scope_df["grade_normalized"] == g].copy()
            if grade_check.empty:
                continue
            
            subjects_in_data = set(grade_check["subject"].dropna().astype(str).str.lower())
            for subject_str in ["ELA", "Math"]:
                subject_filter_name = "ELA" if subject_str == "ELA" else "Math"
                if not should_generate_subject(subject_filter_name, chart_filters):
                    continue
                
                subject_match = False
                if subject_str == "ELA":
                    subject_match = any("ela" in s for s in subjects_in_data)
                elif subject_str == "Math":
                    subject_match = any("math" in s for s in subjects_in_data)
                
                if subject_match:
                    try:
                        print(f"  [Section 3] Generating chart for {scope_label} - Grade {g} - {subject_str} - Winter")
                        chart_path = plot_iready_blended_dashboard(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            subject_str=subject_str, current_grade=int(g),
                            window_filter="Winter", cohort_year=anchor_year,
                            cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                        else:
                            print(f"  [Section 3] SKIPPED (no output) for {scope_label} - Grade {g} - {subject_str} - Winter (likely no data after window/subject filters or below min-N).")
                    except Exception as e:
                        print(f"  [Section 3] Error generating chart for {scope_label} - Grade {g} - {subject_str} (Winter): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    for scope_df, scope_label, folder in scopes:
        _run_scope_section3(scope_df.copy(), scope_label, folder)
    
    # Section 0: Winter i-Ready vs CERS
    print("\n[Section 0] Generating Winter i-Ready vs CERS...")
    
    def _prep_section0_iready_winter(df, subject):
        """Prepare data for Section 0: Winter i-Ready vs CERS comparison"""
        d = df.copy()
        
        # Normalize i-Ready placement labels
        if hasattr(hf, "IREADY_LABEL_MAP"):
            d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)
        
        subj = subject.upper()
        
        # Filter for most recent academic year with Winter + CERS data
        valid_years = (
            d.loc[
                (d["testwindow"].astype(str).str.upper() == "WINTER")
                & (d["cers_overall_performanceband"].notna())
            ]["academicyear"]
            .dropna()
            .unique()
        )
        if len(valid_years) == 0:
            print(f"[WARN] No Winter rows with valid CERS data for {subj}")
            return None, None, None
        
        last_year = max(valid_years)
        d = d[
            (d["academicyear"] == last_year)
            & (d["testwindow"].astype(str).str.upper() == "WINTER")
            & (d["subject"].astype(str).str.upper() == subj)
            & (d["cers_overall_performanceband"].notna())
            & (d["domain"] == "Overall")
            & (d["relative_placement"].notna())
            & (d["enrolled"] == "Enrolled")
        ].copy()
        
        if d.empty:
            print(f"[WARN] No Winter {last_year} data for {subj}")
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
    
    def _plot_section0_iready_winter(scope_label, folder, output_dir, data_dict, preview=False):
        """Plot Section 0: Winter i-Ready vs CERS comparison"""
        cers_levels = [
            "Level 1 - Standard Not Met",
            "Level 2 - Standard Nearly Met",
            "Level 3 - Standard Met",
            "Level 4 - Standard Exceeded",
        ]
        placements = [
            hf.IREADY_LABEL_MAP.get("3 or More Grade Levels Below", "3+ Below"),
            hf.IREADY_LABEL_MAP.get("2 Grade Levels Below", "2 Below"),
            hf.IREADY_LABEL_MAP.get("1 Grade Level Below", "1 Below"),
            hf.IREADY_LABEL_MAP.get("Early On Grade Level", "Early On"),
            hf.IREADY_LABEL_MAP.get("Mid or Above Grade Level", "Mid/Above"),
        ]
        
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
            # ax_top.grid(False)  # Gridlines disabled globally
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
            for rect, val in zip(bars, [metrics["iready_mid_above"], metrics["cers_met_exceed"]]):
                ax_mid.text(
                    rect.get_x() + rect.get_width() / 2,
                    val + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                    color="#434343",
                )
            ax_mid.set_ylim(0, 100)
            ax_mid.set_ylabel("% of Students")
            # ax_mid.grid(False)  # Gridlines disabled globally
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)
            
            # Insight panel
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            insight_text = (
                f"Winter i-Ready Mid/Above vs CAASPP Met/Exceed:\n"
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
            f"{scope_label} • Winter {year} • i-Ready Placement vs CERS Performance",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )
        
        fig.text(
            0.5,
            0.975,
            "The charts below reflect data for students in Gr 3-8 and 11 with matched CAASPP scores. "
            "The 'CERS Met/Exceed' may not align to official results.",
            ha="center",
            fontsize=10,
            style="italic",
        )
        
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        safe_scope = scope_label.replace(" ", "_")
        out_path = out_dir / f"{prefix}{safe_scope}_IREADY_section0_iready_vs_cers.png"
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"[SAVE] Section 0 → {out_path}")
        
        chart_data = {
            "scope": scope_label,
            "window_filter": "Winter",
            "year": year,
            "subjects": list(data_dict.keys()),
        }
        track_chart(f"{prefix}{safe_scope}_section0_iready_vs_cers", str(out_path), scope=folder, section=0, chart_data=chart_data)
        
        if preview:
            plt.show()
        plt.close(fig)
        
        return str(out_path)
    
    for scope_df, scope_label, folder in scopes:
        try:
            payload = {}
            for subj in ["ELA", "Math"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                cross, metrics, year = _prep_section0_iready_winter(scope_df, subj)
                if cross is None:
                    continue
                payload[subj] = (cross, metrics)
            if payload:
                path = _plot_section0_iready_winter(scope_label, folder, args.output_dir, payload, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 0.1: Fall → Winter Comparison
    print("\n[Section 0.1] Generating Fall → Winter Comparison...")
    
    def _prep_section0_1(df, subject):
        d = df.copy()
        
        if hasattr(hf, "IREADY_LABEL_MAP"):
            d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)
        
        subj = str(subject).upper()
        
        d["academicyear"] = pd.to_numeric(d.get("academicyear"), errors="coerce")
        year = int(d["academicyear"].max())
        
        d = d[
            (d["academicyear"] == year)
            & (d["testwindow"].astype(str).str.upper().isin(["FALL", "WINTER"]))
            & (d["subject"].astype(str).str.upper() == subj)
            & (d["domain"].astype(str) == "Overall")
            & (d["enrolled"].astype(str) == "Enrolled")
            & (d["relative_placement"].notna())
        ].copy()
        
        if d.empty:
            return None, None, None
        
        d["scale_score"] = pd.to_numeric(d.get("scale_score"), errors="coerce")
        
        win_order = ["Fall", "Winter"]
        counts = (
            d.groupby(["testwindow", "relative_placement"])
            .size()
            .rename("n")
            .reset_index()
        )
        totals = d.groupby("testwindow").size().rename("N_total").reset_index()
        pct_df = counts.merge(totals, on="testwindow", how="left")
        pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
        
        pct_df["testwindow"] = pct_df["testwindow"].astype(str).str.title()
        pct_df = pct_df[pct_df["testwindow"].isin(win_order)].copy()
        
        all_idx = pd.MultiIndex.from_product(
            [win_order, hf.IREADY_ORDER], names=["testwindow", "relative_placement"]
        )
        pct_df = (
            pct_df.set_index(["testwindow", "relative_placement"])
            .reindex(all_idx)
            .reset_index()
        )
        pct_df["pct"] = pct_df["pct"].fillna(0)
        pct_df["n"] = pct_df["n"].fillna(0)
        pct_df["N_total"] = pct_df.groupby("testwindow")["N_total"].transform(
            lambda s: s.ffill().bfill()
        )
        
        score_df = (
            d.dropna(subset=["scale_score"])
            .groupby(d["testwindow"].astype(str).str.title())["scale_score"]
            .mean()
            .reindex(win_order)
            .rename("avg_score")
            .reset_index()
            .rename(columns={"testwindow": "window"})
        )
        
        fall_val = score_df.loc[score_df["window"] == "Fall", "avg_score"]
        winter_val = score_df.loc[score_df["window"] == "Winter", "avg_score"]
        if len(fall_val) == 0 or len(winter_val) == 0:
            diff = np.nan
        else:
            diff = float(winter_val.iloc[0]) - float(fall_val.iloc[0])
        
        metrics = {"year": year, "diff": diff}
        return pct_df, score_df, metrics
    
    def _plot_section0_1(scope_label, folder, output_dir, data_dict, preview=False):
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.5])
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        
        legend_handles = [
            Patch(facecolor=hf.IREADY_COLORS[q], edgecolor="none", label=q)
            for q in hf.IREADY_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            labels=hf.IREADY_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(hf.IREADY_ORDER),
            frameon=False,
            fontsize=9,
        )
        
        win_order = ["Fall", "Winter"]
        
        for i, (subj, (pct_df, score_df, metrics)) in enumerate(data_dict.items()):
            # Row 1: 100% stacked bar
            ax_top = fig.add_subplot(gs[0, i])
            pivot = (
                pct_df.pivot(
                    index="testwindow", columns="relative_placement", values="pct"
                )
                .reindex(index=win_order)
                .reindex(columns=hf.IREADY_ORDER)
                .fillna(0)
            )
            x = np.arange(len(win_order))
            bottom = np.zeros(len(win_order))
            for cat in hf.IREADY_ORDER:
                vals = pivot[cat].to_numpy()
                bars = ax_top.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=hf.IREADY_COLORS[cat],
                    edgecolor="white",
                    linewidth=1.2,
                    width=0.7,
                )
                for j, v in enumerate(vals):
                    if v >= LABEL_MIN_PCT:
                        ax_top.text(
                            x[j],
                            bottom[j] + v / 2,
                            f"{v:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            color=(
                                "white"
                                if cat in ["3+ Below", "Mid/Above", "Early On"]
                                else "#333"
                            ),
                        )
                bottom += vals
            
            ax_top.set_ylim(0, 100)
            ax_top.set_ylabel("% of Students")
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(win_order)
            ax_top.set_title(subj, fontsize=14, fontweight="bold")
            # ax_top.grid(False)  # Gridlines disabled globally
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            
            # Row 2: Avg Scale Score
            ax_mid = fig.add_subplot(gs[1, i])
            xx = np.arange(len(win_order))
            yvals = score_df["avg_score"].to_numpy()
            bars = ax_mid.bar(
                xx,
                yvals,
                color=hf.default_quintile_colors[4],
                edgecolor="white",
                linewidth=1.2,
                width=0.7,
            )
            for rect, val in zip(bars, yvals):
                if pd.notna(val):
                    ax_mid.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="bold",
                        color="#333",
                    )
            
            n_map = (
                pct_df.groupby("testwindow")["N_total"]
                .max()
                .reindex(win_order)
                .to_dict()
            )
            labels_with_n = [
                f"{w}\n(n = {0 if pd.isna(n_map.get(w, np.nan)) else int(n_map.get(w))})"
                for w in win_order
            ]
            ax_mid.set_ylabel("Avg Scale Score")
            ax_mid.set_xticks(xx)
            ax_mid.set_xticklabels(labels_with_n)
            # ax_mid.grid(False)  # Gridlines disabled globally
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)
            
            # Row 3: Insight box
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            diff = metrics.get("diff", np.nan)
            diff_str = "NA" if pd.isna(diff) else f"{diff:+.1f}"
            insight_text = "Change in Avg Scale Score Fall to Winter:\n" f"{diff_str}"
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
        
        year = next(iter(data_dict.values()))[2].get("year", "")
        fig.suptitle(
            f"{scope_label} • {year} • i-Ready Fall vs Winter Trends",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )
        
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        safe_scope = scope_label.replace(" ", "_")
        out_path = out_dir / f"{prefix}{safe_scope}_IREADY_section0_1_fall_to_winter.png"
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"[SAVE] Section 0.1 → {out_path}")
        
        chart_data = {
            "scope": scope_label,
            "year": year,
            "subjects": list(data_dict.keys()),
        }
        track_chart(f"{prefix}{safe_scope}_section0_1_fall_to_winter", str(out_path), scope=folder, section=0.1, chart_data=chart_data)
        
        if preview:
            plt.show()
        plt.close(fig)
        
        return str(out_path)
    
    for scope_df, scope_label, folder in scopes:
        try:
            data_dict = {}
            for subj in ["ELA", "Math"]:
                if not should_generate_subject(subj, chart_filters):
                    continue
                prep = _prep_section0_1(scope_df, subj)
                if prep and prep[0] is not None:
                    pct_df, score_df, metrics = prep
                    data_dict[subj] = (pct_df, score_df, metrics)
            
            if data_dict:
                path = _plot_section0_1(scope_label, folder, args.output_dir, data_dict, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
        except Exception as e:
            print(f"Error generating Section 0.1 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 0.1: Grade-level Fall → Winter Comparison
    print("\n[Section 0.1] Generating Grade-level Fall → Winter Comparison...")
    
    # Use current year only (matches Section 0.1 logic)
    _base0_1 = iready_base.copy()
    _base0_1["academicyear"] = pd.to_numeric(_base0_1.get("academicyear"), errors="coerce")
    if _base0_1["academicyear"].notna().any():
        _curr_year0_1 = int(_base0_1["academicyear"].max())
        _base0_1 = _base0_1[_base0_1["academicyear"] == _curr_year0_1].copy()
    else:
        _curr_year0_1 = None
        _base0_1 = pd.DataFrame()
    
    if not _base0_1.empty:
        _base0_1["student_grade"] = pd.to_numeric(_base0_1.get("student_grade"), errors="coerce")
        
        # Get unique grades from filtered data
        unique_grades = sorted([g for g in _base0_1["student_grade"].dropna().unique() if g is not None])
        
        # Filter grades if chart_filters specifies grades
        if chart_filters and chart_filters.get("grades") and len(chart_filters["grades"]) > 0:
            unique_grades = [g for g in unique_grades if g in chart_filters["grades"]]
        
        # District-level (by grade)
        # Be defensive: some configs provide district_name=[] which would raise IndexError.
        _dn = cfg.get("district_name", ["Districtwide"])
        if isinstance(_dn, list) and len(_dn) > 0:
            scope_label_district = _dn[0]
        elif isinstance(_dn, str) and _dn.strip():
            scope_label_district = _dn
        else:
            scope_label_district = "Districtwide"
        for g in unique_grades:
            df_g = _base0_1[_base0_1["student_grade"] == g].copy()
            if df_g.empty:
                continue
            try:
                data_dict = {}
                for subj in ["ELA", "Math"]:
                    if not should_generate_subject(subj, chart_filters):
                        continue
                    prep = _prep_section0_1(df_g, subj)
                    if prep and prep[0] is not None:
                        pct_df, score_df, metrics = prep
                        data_dict[subj] = (pct_df, score_df, metrics)
                
                if data_dict:
                    path = _plot_section0_1(
                        f"{scope_label_district} • Grade {int(g)}",
                        "_district",
                        args.output_dir,
                        data_dict,
                        preview=hf.DEV_MODE
                    )
                    if path:
                        chart_paths.append(path)
            except Exception as e:
                print(f"Error generating Section 0.1 grade-level chart for Grade {g}: {e}")
                if hf.DEV_MODE:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Site-level (by grade)
        for scope_df, scope_label, folder in scopes:
            if folder == "_district":
                continue  # Skip district, already handled above
            
            site_df = _base0_1.copy()
            # Filter to this site's data
            if "school" in site_df.columns:
                raw_school = scope_df["school"].iloc[0] if not scope_df.empty else None
                if raw_school:
                    site_df = site_df[site_df["school"] == raw_school].copy()
            
            site_grades = sorted([g for g in site_df["student_grade"].dropna().unique() if g is not None])
            if chart_filters and chart_filters.get("grades") and len(chart_filters["grades"]) > 0:
                site_grades = [g for g in site_grades if g in chart_filters["grades"]]
            
            for g in site_grades:
                df_g = site_df[site_df["student_grade"] == g].copy()
                if df_g.empty:
                    continue
                try:
                    data_dict = {}
                    for subj in ["ELA", "Math"]:
                        if not should_generate_subject(subj, chart_filters):
                            continue
                        prep = _prep_section0_1(df_g, subj)
                        if prep and prep[0] is not None:
                            pct_df, score_df, metrics = prep
                            data_dict[subj] = (pct_df, score_df, metrics)
                    
                    if data_dict:
                        path = _plot_section0_1(
                            f"{scope_label} • Grade {int(g)}",
                            folder,
                            args.output_dir,
                            data_dict,
                            preview=hf.DEV_MODE
                        )
                        if path:
                            chart_paths.append(path)
                except Exception as e:
                    print(f"Error generating Section 0.1 grade-level chart for {scope_label} - Grade {g}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Section 4: Winter i-Ready Mid/Above → % CERS Met/Exceeded (≤2025)
    print("\n[Section 4] Generating Winter i-Ready Mid/Above → % CAASPP Met/Exceeded...")
    
    _ME_LABELS = {"Level 3 - Standard Met", "Level 4 - Standard Exceeded"}
    _SUBJECT_COLORS = {"ELA": "#0381a2", "Math": "#0381a2"}
    
    def _prep_mid_above_to_cers_winter(df_in, subject):
        d = df_in.copy()
        placement_col = (
            "relative_placement" if "relative_placement" in d.columns else "placement"
        )
        
        d = d[
            (d["domain"].astype(str).str.lower() == "overall")
            & (d["testwindow"].astype(str).str.lower() == "winter")
            & (d["cers_overall_performanceband"].notna())
            & (d[placement_col].notna())
        ].copy()
        
        d = d[d["subject"].astype(str).str.lower().str.contains(subject.lower())]
        d["academicyear"] = pd.to_numeric(d["academicyear"], errors="coerce")
        d = d[d["academicyear"] <= 2025]
        
        mid_vals = {"mid/above", "mid or above", "mid or above grade level"}
        d = d[d[placement_col].astype(str).str.strip().str.lower().isin(mid_vals)].copy()
        if d.empty:
            return pd.DataFrame()
        
        id_col = "student_id" if "student_id" in d.columns else "uniqueidentifier"
        denom = d.groupby("academicyear")[id_col].nunique().rename("n")
        numer = (
            d[d["cers_overall_performanceband"].isin(_ME_LABELS)]
            .groupby("academicyear")[id_col]
            .nunique()
            .rename("me")
        )
        trend = denom.to_frame().join(numer, how="left").fillna(0).reset_index()
        trend["pct_me"] = (trend["me"] / trend["n"]) * 100
        return trend.sort_values("academicyear")
    
    def _plot_mid_above_to_cers_faceted_winter(scope_df, scope_label, folder, output_dir, preview=False):
        subjects = ["ELA", "Math"]
        trends = {s: _prep_mid_above_to_cers_winter(scope_df, s) for s in subjects}
        
        if all(tr.empty for tr in trends.values()):
            print(f"[Section 4] No qualifying data for {scope_label}")
            return None
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 9), height_ratios=[2, 0.6])
        fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88, bottom=0.1)
        
        for j, subj in enumerate(subjects):
            ax_bar = axs[0, j]
            ax_box = axs[1, j]
            
            tr = trends[subj]
            if tr.empty:
                ax_bar.axis("off")
                ax_box.axis("off")
                continue
            
            x = np.arange(len(tr))
            color = _SUBJECT_COLORS[subj]
            bars = ax_bar.bar(x, tr["pct_me"], color=color, edgecolor="white", width=0.55)
            
            for rect, yv, n in zip(bars, tr["pct_me"], tr["n"]):
                ax_bar.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() / 2,
                    f"{yv:.0f}%\n(n={int(n)})",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )
            
            ax_bar.set_ylim(0, 100)
            ax_bar.set_xlim(-0.5, len(tr) - 0.5)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(tr["academicyear"].astype(int))
            ax_bar.set_yticks(range(0, 101, 20))
            ax_bar.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
            # ax_bar.grid(False)  # Gridlines disabled globally
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)
            ax_bar.set_ylabel("% Met or Exceeded")
            ax_bar.set_xlabel("Academic Year")
            ax_bar.set_title(subj, fontsize=14, fontweight="bold", pad=20)
            ax_bar.margins(x=0.15)
            
            ax_box.axis("off")
            overall_pct = 100 * tr["me"].sum() / tr["n"].sum()
            
            lines = [
                rf"Historically, $\mathbf{{{overall_pct:.1f}\%}}$ of students that meet ",
                r"$\mathbf{Mid\ or\ Above}$ Grade Level in Winter i-Ready tend to ",
                r"$\mathbf{Meet\ or\ Exceed\ Standard}$ on CAASPP for " + subj + ".",
            ]
            
            ax_box.text(
                0.5,
                0.5,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=13,
                color="#333",
                wrap=True,
                usetex=False,
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="#f5f5f5",
                    edgecolor="#bbb",
                    linewidth=0.8,
                ),
            )
        
        fig.suptitle(
            f"{scope_label} \n Winter i-Ready Mid/Above → % CAASPP Met/Exceeded",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )
        
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        safe_scope = scope_label.replace(" ", "_")
        out_path = out_dir / f"{prefix}{safe_scope}_IREADY_section4_mid_plus_to_3plus.png"
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"[Section 4] Saved: {out_path}")
        
        chart_data = {
            "scope": scope_label,
            "section": 4,
            "trends": {
                subj: tr.to_dict('records') if not tr.empty else []
                for subj, tr in trends.items()
            }
        }
        track_chart(f"{prefix}{safe_scope}_section4_mid_plus_to_3plus", str(out_path), scope=folder, section=4, chart_data=chart_data)
        
        if preview:
            plt.show()
        plt.close(fig)
        
        return str(out_path)
    
    for scope_df, scope_label, folder in scopes:
        try:
            subjects_to_generate = ["ELA", "Math"]
            if chart_filters and chart_filters.get("subjects"):
                subjects_to_generate = [s for s in subjects_to_generate if should_generate_subject(s, chart_filters)]
            
            if subjects_to_generate:
                path = _plot_mid_above_to_cers_faceted_winter(scope_df.copy(), scope_label, folder, args.output_dir, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
        except Exception as e:
            print(f"Error generating Section 4 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 5: MOY Growth Progress (Median % Progress + On-Track + BOY-Anchored Insights)
    print("\n" + "="*80)
    print("[Section 5] Generating MOY Growth Progress...")
    print("="*80)
    
    def _safe_id_col(df):
        return "student_id" if "student_id" in df.columns else "uniqueidentifier"
    
    def _dedupe_latest(df, id_col, sort_col_candidates):
        d = df.copy()
        sort_col = None
        for c in sort_col_candidates:
            if c in d.columns:
                sort_col = c
                break
        if sort_col is None:
            d = d.sort_values([id_col])
            return d.groupby([id_col], as_index=False).tail(1)
        
        d[sort_col] = pd.to_datetime(d[sort_col], errors="coerce")
        d = d.sort_values([id_col, sort_col])
        return d.groupby([id_col], as_index=False).tail(1)
    
    def _normalize_placement(d):
        if hasattr(hf, "IREADY_LABEL_MAP") and "relative_placement" in d.columns:
            d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)
        return d
    
    def _prep_section5_subject(df, subject, scope_label=""):
        print(f"[Section 5]   Preparing data for subject: {subject} (scope: {scope_label})")
        d0 = df.copy()
        print(f"[Section 5]     Initial data rows: {len(d0):,}")
        
        try:
            d0["academicyear"] = pd.to_numeric(d0.get("academicyear"), errors="coerce")
            if d0["academicyear"].isna().all():
                print(f"[Section 5]     ❌ ERROR: No valid academicyear values found")
                return None
            
            year = int(d0["academicyear"].max())
            print(f"[Section 5]     Using academic year: {year}")
            d0 = d0[d0["academicyear"] == year].copy()
            print(f"[Section 5]     Rows for year {year}: {len(d0):,}")
            
            if "student_grade" in d0.columns:
                d0["student_grade"] = pd.to_numeric(d0.get("student_grade"), errors="coerce")
                rows_before_grade_filter = len(d0)
                d0 = d0[d0["student_grade"] <= 8].copy()
                print(f"[Section 5]     Rows after grade <= 8 filter: {len(d0):,} (removed {rows_before_grade_filter - len(d0):,})")
            
            subj = str(subject).strip().upper()
            
            base_mask = (d0["enrolled"].astype(str) == "Enrolled") & (
                d0["domain"].astype(str).str.lower() == "overall"
            )
            rows_before_subj_filter = len(d0)
            
            if subj == "ELA":
                subj_mask = (
                    d0["subject"]
                    .astype(str)
                    .str.contains("ela|reading", case=False, na=False)
                )
            elif subj == "MATH":
                subj_mask = (
                    d0["subject"].astype(str).str.contains("math", case=False, na=False)
                )
            else:
                subj_mask = (
                    d0["subject"].astype(str).str.contains(subj, case=False, na=False)
                )
            
            d0 = d0[base_mask & subj_mask].copy()
            print(f"[Section 5]     Rows after subject + enrolled + overall filter: {len(d0):,} (removed {rows_before_subj_filter - len(d0):,})")
            
            if d0.empty:
                print(f"[Section 5]     ❌ SKIPPED: No data after filtering for {subject}")
                print(f"[Section 5]       Check: enrolled='Enrolled', domain='Overall', subject contains '{subj}'")
                return None
            
            id_col = _safe_id_col(d0)
            print(f"[Section 5]     Using ID column: {id_col}")
            d0[id_col] = d0[id_col].astype(str).str.strip()
            d0.loc[d0[id_col].isin(["nan", "None", "<NA>"]), id_col] = np.nan
            
            # Check for Fall baseline data
            print(f"[Section 5]     Checking for Fall baseline data...")
            if "testwindow" not in d0.columns:
                print(f"[Section 5]     ❌ ERROR: 'testwindow' column not found in data")
                return None
            
            if "baseline_diagnostic" not in d0.columns:
                print(f"[Section 5]     ⚠️  WARNING: 'baseline_diagnostic' column not found - will use all Fall data")
                fall = d0[(d0["testwindow"].astype(str).str.lower() == "fall")].copy()
            else:
                fall = d0[
                    (d0["testwindow"].astype(str).str.lower() == "fall")
                    & (d0.get("baseline_diagnostic", "").astype(str).str.lower() == "yes")
                ].copy()
            
            print(f"[Section 5]     Fall baseline rows: {len(fall):,}")
            
            if fall.empty:
                print(f"[Section 5]     ❌ SKIPPED: No Fall baseline data found for {subject}")
                print(f"[Section 5]       Available testwindow values: {sorted(d0['testwindow'].astype(str).str.lower().unique())}")
                if "baseline_diagnostic" in d0.columns:
                    baseline_vals = d0["baseline_diagnostic"].astype(str).str.lower().unique()
                    print(f"[Section 5]       Available baseline_diagnostic values: {sorted(baseline_vals)}")
                return None
            
            fall = _normalize_placement(fall)
            fall = _dedupe_latest(fall, id_col, ["completion_date", "teststartdate"])
            print(f"[Section 5]     Fall rows after deduplication: {len(fall):,}")
            
            # Check for required columns
            required_cols = ["scale_score", "annual_typical_growth_measure", "annual_stretch_growth_measure", "mid_on_grade_level_scale_score"]
            missing_cols = [col for col in required_cols if col not in fall.columns]
            if missing_cols:
                print(f"[Section 5]     ⚠️  WARNING: Missing columns: {missing_cols}")
            
            ss = pd.to_numeric(fall.get("scale_score"), errors="coerce")
            typ = pd.to_numeric(fall.get("annual_typical_growth_measure"), errors="coerce")
            strg = pd.to_numeric(fall.get("annual_stretch_growth_measure"), errors="coerce")
            mid = pd.to_numeric(fall.get("mid_on_grade_level_scale_score"), errors="coerce")
            
            print(f"[Section 5]     Valid scale_score values: {ss.notna().sum()}/{len(fall)}")
            print(f"[Section 5]     Valid typical_growth values: {typ.notna().sum()}/{len(fall)}")
            print(f"[Section 5]     Valid stretch_growth values: {strg.notna().sum()}/{len(fall)}")
            print(f"[Section 5]     Valid mid_on_grade_level values: {mid.notna().sum()}/{len(fall)}")
            
            already_mid = (
                fall["relative_placement"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["mid/above", "mid or above grade level", "mid or above", "mid/above grade level"])
            )
            typ_reach = (ss + typ) >= mid
            str_reach = (ss + strg) >= mid
            
            out = np.full(len(fall), np.nan, dtype=object)
            out[already_mid.to_numpy()] = "Already Mid+"
            m2 = (~already_mid) & typ_reach
            out[m2.to_numpy()] = "Mid with Typical"
            m3 = (~already_mid) & (~typ_reach) & str_reach
            out[m3.to_numpy()] = "Mid with Stretch"
            fall["mid_flag"] = pd.Series(out, index=fall.index).fillna("Mid Beyond Stretch")
            
            fall_counts = (
                fall["mid_flag"]
                .value_counts(dropna=False)
                .reindex(["Already Mid+", "Mid with Typical", "Mid with Stretch", "Mid Beyond Stretch"])
                .fillna(0)
                .astype(int)
                .to_dict()
            )
            print(f"[Section 5]     Fall cohort breakdown: {fall_counts}")
            
            # Winter rows for same Fall cohort
            print(f"[Section 5]     Checking for Winter data for Fall cohort...")
            winter = d0[(d0["testwindow"].astype(str).str.lower() == "winter")].copy()
            print(f"[Section 5]     Total Winter rows: {len(winter):,}")
            
            winter = _normalize_placement(winter)
            winter = _dedupe_latest(winter, id_col, ["completion_date", "teststartdate"])
            print(f"[Section 5]     Winter rows after deduplication: {len(winter):,}")
            
            cohort_ids = set(fall[id_col].dropna().unique().tolist())
            winter_ids = set(winter[id_col].dropna().unique().tolist())
            intersection = cohort_ids.intersection(winter_ids)
            print(f"[Section 5]     Fall cohort IDs: {len(cohort_ids):,}")
            print(f"[Section 5]     Winter IDs: {len(winter_ids):,}")
            print(f"[Section 5]     Intersection (IDs in both Fall and Winter): {len(intersection):,}")
            
            winter = winter[winter[id_col].isin(cohort_ids)].copy()
            print(f"[Section 5]     Winter rows matching Fall cohort: {len(winter):,}")
            
            if winter.empty:
                print(f"[Section 5]     ❌ SKIPPED: No Winter data found for Fall cohort in {subject}")
                print(f"[Section 5]       This means students with Fall baseline data don't have matching Winter data")
                return None
            
            # Check for progress columns
            progress_cols = ["percent_progress_to_annual_typical_growth_", "percent_progress_to_annual_stretch_growth_"]
            missing_progress = [col for col in progress_cols if col not in winter.columns]
            if missing_progress:
                print(f"[Section 5]     ⚠️  WARNING: Missing progress columns: {missing_progress}")
                print(f"[Section 5]       Available columns containing 'progress': {[c for c in winter.columns if 'progress' in c.lower()]}")
            
        except Exception as e:
            print(f"[Section 5]     ❌ ERROR in _prep_section5_subject for {subject}: {e}")
            print(f"[Section 5]       Error type: {type(e).__name__}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            return None
        
        winter = winter.merge(
            fall[[id_col, "mid_flag"]].drop_duplicates(subset=[id_col]),
            on=id_col,
            how="left",
        )
        
        winter["pct_typ"] = pd.to_numeric(
            winter.get("percent_progress_to_annual_typical_growth_"), errors="coerce"
        )
        winter["pct_str"] = pd.to_numeric(
            winter.get("percent_progress_to_annual_stretch_growth_"), errors="coerce"
        )
        
        valid_typ = winter["pct_typ"].notna().sum()
        valid_str = winter["pct_str"].notna().sum()
        print(f"[Section 5]     Valid typical progress values: {valid_typ}/{len(winter)}")
        print(f"[Section 5]     Valid stretch progress values: {valid_str}/{len(winter)}")
        
        med_typ = float(np.nanmedian(winter["pct_typ"].to_numpy()))
        med_str = float(np.nanmedian(winter["pct_str"].to_numpy()))
        pct50_typ = float((winter["pct_typ"] >= 50).mean() * 100)
        pct50_str = float((winter["pct_str"] >= 50).mean() * 100)
        
        print(f"[Section 5]     Median typical progress: {med_typ:.1f}%")
        print(f"[Section 5]     Median stretch progress: {med_str:.1f}%")
        print(f"[Section 5]     % >=50% typical: {pct50_typ:.1f}%")
        print(f"[Section 5]     % >=50% stretch: {pct50_str:.1f}%")
        
        winter_mid = (
            winter["relative_placement"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["mid/above", "mid or above grade level", "mid or above", "mid/above grade level"])
        )
        winter_mid_count = int(winter_mid.sum())
        print(f"[Section 5]     Winter Mid/Above count: {winter_mid_count}")
        
        def _on_track(group_name, col):
            base_ids = set(
                fall.loc[fall["mid_flag"] == group_name, id_col].dropna().tolist()
            )
            denom = len(base_ids)
            if denom == 0:
                return {"denom": 0, "num": 0, "pct": np.nan}
            w = winter[winter[id_col].isin(base_ids)].copy()
            num = int((w[col] >= 50).sum())
            pct = 100 * num / denom
            return {"denom": denom, "num": num, "pct": pct}
        
        on_typ = _on_track("Mid with Typical", "pct_typ")
        on_str = _on_track("Mid with Stretch", "pct_str")
        
        print(f"[Section 5]     ✅ Successfully prepared metrics for {subject}")
        
        metrics = dict(
            year=year,
            med_typ=med_typ,
            med_str=med_str,
            pct50_typ=pct50_typ,
            pct50_str=pct50_str,
            fall_counts=fall_counts,
            winter_mid_count=winter_mid_count,
            on_typ=on_typ,
            on_str=on_str,
            n_winter=int(winter[id_col].nunique()),
        )
        
        return metrics
    
    def _plot_section5(scope_label, folder, output_dir, metrics_by_subject, preview=False):
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.85, 0.65, 0.6])
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        
        c_typ = hf.IREADY_COLORS.get("Mid/Above", "#0381a2")
        c_str = hf.IREADY_COLORS.get("Early On", "#00baeb")
        
        for i, subj in enumerate(["ELA", "Math"]):
            m = metrics_by_subject.get(subj)
            if not m:
                ax0 = fig.add_subplot(gs[0, i])
                ax0.axis("off")
                ax1 = fig.add_subplot(gs[1, i])
                ax1.axis("off")
                ax2 = fig.add_subplot(gs[2, i])
                ax2.axis("off")
                continue
            
            # Top: Median % progress
            ax_top = fig.add_subplot(gs[0, i])
            x = np.arange(2)
            vals = [m["med_typ"], m["med_str"]]
            bars = ax_top.bar(
                x,
                vals,
                color=[c_typ, c_str],
                edgecolor="white",
                linewidth=1.2,
                width=0.6,
            )
            for rect, v in zip(bars, vals):
                if pd.notna(v):
                    ax_top.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1,
                        f"{v:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="bold",
                        color="#333",
                    )
            ax_top.axhline(50, linestyle="--", linewidth=1.2, color="#666", alpha=0.8)
            ax_top.set_ylim(0, 100)
            ax_top.set_yticks(range(0, 101, 20))
            ax_top.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(["Median % Typical", "Median % Stretch"])
            ax_top.set_ylabel("% Progress")
            ax_top.set_title(subj, fontsize=14, fontweight="bold")
            # ax_top.grid(False)  # Gridlines disabled globally
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            
            # Middle: % of students >= 50% progress
            ax_mid = fig.add_subplot(gs[1, i])
            x2 = np.arange(2)
            vals2 = [m["pct50_typ"], m["pct50_str"]]
            bars2 = ax_mid.bar(
                x2,
                vals2,
                color=[c_typ, c_str],
                edgecolor="white",
                linewidth=1.2,
                width=0.6,
            )
            for rect, v in zip(bars2, vals2):
                if pd.notna(v):
                    ax_mid.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1,
                        f"{v:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="bold",
                        color="#333",
                    )
            ax_mid.axhline(50, linestyle="--", linewidth=1.0, color="#666", alpha=0.8)
            ax_mid.set_ylim(0, 100)
            ax_mid.set_yticks(range(0, 101, 20))
            ax_mid.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
            ax_mid.set_xticks(x2)
            ax_mid.set_xticklabels([">=50% Typical", ">=50% Stretch"])
            ax_mid.set_ylabel("% of Students")
            # ax_mid.grid(False)  # Gridlines disabled globally
            ax_mid.spines["top"].set_visible(False)
            ax_mid.spines["right"].set_visible(False)
            
            # Bottom: BOY-anchored insight summary
            ax_bot = fig.add_subplot(gs[2, i])
            ax_bot.axis("off")
            
            fc = m["fall_counts"]
            on_typ = m["on_typ"]
            on_str = m["on_str"]
            
            typ_pct_str = (
                "NA"
                if (on_typ["denom"] == 0 or pd.isna(on_typ["pct"]))
                else f"{on_typ['pct']:.0f}%"
            )
            str_pct_str = (
                "NA"
                if (on_str["denom"] == 0 or pd.isna(on_str["pct"]))
                else f"{on_str['pct']:.0f}%"
            )
            
            fall_mid_ct = fc.get("Already Mid+", 0)
            winter_mid_ct = int(m.get("winter_mid_count", 0))
            
            lines = [
                rf"Fall Already Mid+ = $\mathbf{{{fall_mid_ct:,}}}$",
                rf"Winter Already Mid+ = $\mathbf{{{winter_mid_ct:,}}}$",
                "",
                "Fall to Winter Mid On Level Update:",
                rf"Mid With Typical: $\mathbf{{{on_typ['num']:,}}}$ out of $\mathbf{{{on_typ['denom']:,}}}$ "
                rf"({typ_pct_str}) are at least 50% to typical growth",
                rf"Mid With Stretch: $\mathbf{{{on_str['num']:,}}}$ out of $\mathbf{{{on_str['denom']:,}}}$ "
                rf"({str_pct_str}) are at least 50% to stretch growth",
            ]
            
            ax_bot.text(
                0.5,
                0.5,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=11,
                color="#333",
                bbox=dict(
                    boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#bbb"
                ),
            )
        
        year = next(iter(metrics_by_subject.values())).get("year", "")
        fig.suptitle(
            f"{scope_label} • Winter {year} • Growth Progress Toward Annual Goals",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )
        
        out_dir = Path(output_dir) / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "DISTRICT_" if folder == "_district" else "SCHOOL_"
        safe_scope = scope_label.replace(" ", "_")
        out_path = out_dir / f"{prefix}{safe_scope}_IREADY_section5_growth_progress_moy.png"
        
        hf._save_and_render(fig, out_path, dev_mode=preview)
        print(f"[SAVE] Section 5 → {out_path}")
        
        chart_data = {
            "scope": scope_label,
            "year": year,
            "subjects": list(metrics_by_subject.keys()),
        }
        track_chart(f"{prefix}{safe_scope}_section5_growth_progress_moy", str(out_path), scope=folder, section=5, chart_data=chart_data)
        
        if preview:
            plt.show()
        plt.close(fig)
        
        return str(out_path)
    
    print(f"[Section 5] Processing {len(scopes)} scope(s)...")
    
    for scope_idx, (scope_df, scope_label, folder) in enumerate(scopes, 1):
        print(f"\n[Section 5] Processing scope {scope_idx}/{len(scopes)}: {scope_label} (folder: {folder})")
        print(f"[Section 5]   Initial data rows: {len(scope_df):,}")
        
        try:
            metrics_by_subject = {}
            for subj in ["ELA", "Math"]:
                if not should_generate_subject(subj, chart_filters):
                    print(f"[Section 5]   Skipping {subj} (not selected in chart_filters)")
                    continue
                
                m = _prep_section5_subject(scope_df.copy(), subj, scope_label)
                if m is not None:
                    metrics_by_subject[subj] = m
                    print(f"[Section 5]   ✅ Added metrics for {subj}")
                else:
                    print(f"[Section 5]   ⚠️  No metrics returned for {subj}")
            
            print(f"[Section 5]   Total subjects with metrics: {len(metrics_by_subject)}/2")
            
            if metrics_by_subject:
                print(f"[Section 5]   Generating chart with metrics for: {list(metrics_by_subject.keys())}")
                path = _plot_section5(scope_label, folder, args.output_dir, metrics_by_subject, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
                    print(f"[Section 5]   ✅ Successfully generated chart: {path}")
                else:
                    print(f"[Section 5]   ⚠️  Chart generation returned None")
            else:
                print(f"[Section 5]   ❌ SKIPPED: No metrics collected for any subject")
                print(f"[Section 5]   This means either:")
                print(f"[Section 5]     - No Fall baseline data found")
                print(f"[Section 5]     - No Winter data for Fall cohort")
                print(f"[Section 5]     - Data filtering removed all rows")
                
        except Exception as e:
            print(f"[Section 5]   ❌ ERROR generating Section 5 chart for {scope_label}: {e}")
            print(f"[Section 5]   Error type: {type(e).__name__}")
            if hf.DEV_MODE:
                import traceback
                print(f"[Section 5]   Full traceback:")
                traceback.print_exc()
            else:
                print(f"[Section 5]   Enable DEV_MODE for full traceback")
            continue
    
    print(f"\n[Section 5] Completed. Generated {len([p for p in chart_paths if 'section5' in str(p).lower()])} Section 5 chart(s)")
    print("="*80)
    
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
    
    print(f"\n✅ Generated {len(chart_paths)} iReady Winter charts")
    if len(chart_paths) == 0:
        print(f"⚠️  WARNING: No charts were generated!")
        print(f"   - Data rows after filtering: {iready_base.shape[0]:,}")
        print(f"   - Scopes found: {len(scopes)}")
        print(f"   - Chart filters: {chart_filters}")
    
    return chart_paths

def generate_iready_winter_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    iready_data: list = None
) -> list:
    """
    Generate iReady Winter charts (wrapper function for Flask backend)
    
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
        # Ensure chart_filters is a dict, not a string
        if isinstance(chart_filters, str):
            try:
                chart_filters = json.loads(chart_filters)
            except:
                print(f"[Warning] Could not parse chart_filters as JSON: {chart_filters}")
                chart_filters = {}
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
            'iready_winter.py',
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

