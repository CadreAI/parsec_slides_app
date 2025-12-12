"""
NWEA Fall chart generation module
Generates charts specifically for Fall (BOY) test window
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
    """Track chart for CSV generation"""
    global _chart_tracking_set
    
    chart_path = Path(file_path)
    normalized_path = str(chart_path.resolve())
    
    if normalized_path in _chart_tracking_set:
        return
    
    _chart_tracking_set.add(normalized_path)
    
    chart_info = {
        "chart_name": chart_name,
        "scope": scope,
        "section": section,
        "file_path": str(file_path),
        "file_link": f"file://{chart_path.absolute()}"
    }
    
    chart_links.append(chart_info)

# Import chart generation functions from main module
from .nwea_charts import (
    plot_dual_subject_dashboard,
    plot_nwea_subject_dashboard_by_group,
    plot_nwea_blended_dashboard,
    _prep_section0,
    _plot_section0_dual,
    _run_cgp_dual_trend,
    train_model,
    predict_2025,
    predict_2026,
    _plot_pred_vs_actual,
    _plot_projection_2026
)

# ---------------------------------------------------------------------
# Main Execution for Fall Charts
# ---------------------------------------------------------------------

def main(nwea_data=None):
    """
    Main function to generate NWEA Fall charts
    
    Args:
        nwea_data: Optional list of dicts or DataFrame with NWEA data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate NWEA Fall charts')
    parser.add_argument('--partner', required=True, help='Partner name')
    parser.add_argument('--data-dir', required=False, help='Data directory path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--dev-mode', default='false', help='Development mode')
    parser.add_argument('--config', default='{}', help='Config JSON string')
    
    args = parser.parse_args()
    
    cfg = load_config_from_args(args.config)
    hf.DEV_MODE = args.dev_mode.lower() in ('true', '1', 'yes', 'on')
    
    chart_filters = cfg.get("chart_filters", {})
    
    if chart_filters:
        print(f"\n[Filters] Applying chart generation filters:")
        if chart_filters.get("grades"):
            print(f"  - Grades: {chart_filters['grades']}")
        if chart_filters.get("years"):
            print(f"  - Years: {chart_filters['years']}")
        if chart_filters.get("subjects"):
            print(f"  - Subjects: {chart_filters['subjects']}")
        if chart_filters.get("student_groups"):
            print(f"  - Student Groups: {chart_filters['student_groups']}")
        if chart_filters.get("race"):
            print(f"  - Race/Ethnicity: {chart_filters['race']}")
    
    # Check if NWEA charts should be generated based on subject filter
    should_generate_nwea = True
    if chart_filters and chart_filters.get("subjects") and len(chart_filters["subjects"]) > 0:
        selected_subjects = [str(s).strip().lower() for s in chart_filters["subjects"]]
        nwea_subjects = ["math", "mathematics", "reading"]
        should_generate_nwea = any(s in nwea_subjects for s in selected_subjects)
        if not should_generate_nwea:
            print("\n[Skip] NWEA charts skipped - selected subjects don't include Math or Reading")
            return []
    
    # Load data
    if nwea_data is not None:
        nwea_base = load_nwea_data(nwea_data=nwea_data)
    else:
        if not args.data_dir:
            raise ValueError("Either nwea_data must be provided or --data-dir must be specified")
        nwea_base = load_nwea_data(data_dir=args.data_dir)
    
    # Apply filters to base data
    if chart_filters:
        nwea_base = apply_chart_filters(nwea_base, chart_filters)
        print(f"Data after filtering: {nwea_base.shape[0]:,} rows")
    
    # Always use Fall for this module
    selected_quarters = ["Fall"]
    
    # Get scopes
    scopes = get_scopes(nwea_base, cfg)
    
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
                chart_paths.append(str(Path(args.output_dir) / folder / f"{scope_label.replace(' ', '_')}_section0_pred_vs_actual_{folder}.png"))
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
    # Section 1: Fall Performance Trends
    print("\n[Section 1] Generating Fall Performance Trends...")
    should_generate_section1 = True
    if chart_filters and chart_filters.get("subjects") and len(chart_filters["subjects"]) > 0:
        selected_subjects = [str(s).strip().lower() for s in chart_filters["subjects"]]
        should_generate_section1 = any(s in ["reading", "math", "mathematics"] for s in selected_subjects)
    
    if chart_filters and chart_filters.get("student_groups") and len(chart_filters["student_groups"]) > 0:
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
                    print(f"Error generating chart for {scope_label} ({quarter}): {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
    else:
        print("  Skipping Section 1 (no matching subjects selected)")
    
    # Section 2: Student Group Performance Trends
    print("\n[Section 2] Generating Student Group Performance Trends...")
    student_groups_cfg = cfg.get("student_groups", {})
    race_ethnicity_cfg = cfg.get("race_ethnicity", {})
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
                        school_raw=None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None,
                        scope_label=scope_label, preview=hf.DEV_MODE)
                except Exception as e:
                    print(f"Error generating Section 2 chart for {scope_label} - {group_name} ({quarter}): {e}")
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
                    continue
                
                combined_group_def = {
                    "column": race_def.get("column"),
                    "values": race_def.get("values"),
                    "type": "race"
                }
                
                for quarter in selected_quarters:
                    try:
                        plot_nwea_subject_dashboard_by_group(
                            scope_df.copy(), None, quarter, race_name, combined_group_def,
                            args.output_dir, cfg, figsize=(16, 9),
                            school_raw=None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None,
                            scope_label=scope_label, preview=hf.DEV_MODE)
                    except Exception as e:
                        print(f"Error generating Section 2 chart for {scope_label} - {race_name} ({quarter}): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Section 3: Overall + Cohort Trends
    print("\n[Section 3] Generating Overall + Cohort Trends...")
    
    def _run_scope_section3_fall(scope_df, scope_label, school_raw):
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
        
        selected_grades = chart_filters.get("grades", [])
        unique_grades = sorted([g for g in scope_df["grade_normalized"].dropna().unique() if g is not None])
        if selected_grades:
            unique_grades = [g for g in unique_grades if g in selected_grades]
        
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
        _run_scope_section3_fall(scope_df.copy(), scope_label, None if folder == "_district" else scope_df["schoolname"].iloc[0] if "schoolname" in scope_df.columns else None)
    
    # Section 4: CGP/CGI Growth Trends (Fall→Fall only, requires Fall quarter)
    print("\n[Section 4] Generating CGP/CGI Growth Trends (Fall→Fall)...")
    for scope_df, scope_label, folder in scopes:
        try:
            subjects_to_generate = ["Reading", "Mathematics"]
            if chart_filters and chart_filters.get("subjects"):
                subjects_to_generate = [s for s in subjects_to_generate if should_generate_subject(s, chart_filters)]
            
            if subjects_to_generate:
                _run_cgp_dual_trend(scope_df.copy(), scope_label, args.output_dir, cfg, preview=hf.DEV_MODE, subjects=subjects_to_generate)
        except Exception as e:
            print(f"Error generating Section 4 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
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
    
    # Build chart_paths from tracked charts
    if chart_links and len(chart_links) > len(chart_paths):
        chart_paths = [str(Path(chart['file_path']).absolute()) for chart in chart_links]
    
    # Remove duplicates
    seen = set()
    unique_chart_paths = []
    for chart_path in chart_paths:
        normalized_path = str(Path(chart_path).resolve())
        if normalized_path not in seen:
            seen.add(normalized_path)
            unique_chart_paths.append(chart_path)
    
    print(f"\n✅ Generated {len(unique_chart_paths)} unique NWEA Fall charts")
    return unique_chart_paths

def generate_nwea_fall_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    nwea_data: list = None
) -> list:
    """
    Generate NWEA Fall charts (wrapper function for Flask backend)
    
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
            'nwea_fall.py',
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

