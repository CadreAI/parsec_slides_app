"""
iReady Fall chart generation module
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

# Import shared plotting functions from iready_charts
from .iready_charts import (
    plot_dual_subject_dashboard,
    plot_iready_subject_dashboard_by_group,
    plot_iready_blended_dashboard,
    _prep_section5_dqc,
    _print_section5_dqc,
    _prep_growth_path_data,
    _plot_mid_flag_stacked
)

# Chart tracking for CSV generation
chart_links = []
_chart_tracking_set = set()

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
    Main function to generate iReady Fall charts
    
    Args:
        iready_data: Optional list of dicts or DataFrame with iReady data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate iReady Fall charts')
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
    
    # Always use Fall for this module
    selected_quarters = ["Fall"]
    
    # Apply filters to base data
    if chart_filters:
        iready_base = apply_chart_filters(iready_base, chart_filters)
        print(f"Data after filtering: {iready_base.shape[0]:,} rows")
    
    # Get scopes from unfiltered data (all sections filter internally)
    scopes = get_scopes(iready_base, cfg)
    
    chart_paths = []
    
    # Section 1: Fall Performance Trends
    print("\n[Section 1] Generating Fall Performance Trends...")
    for scope_df, scope_label, folder in scopes:
        try:
            chart_path = plot_dual_subject_dashboard(
                scope_df,
                scope_label,
                folder,
                args.output_dir,
                window_filter="Fall",
                preview=hf.DEV_MODE
            )
            if chart_path:
                chart_paths.append(chart_path)
        except Exception as e:
            print(f"Error generating chart for {scope_label} (Fall): {e}")
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
            try:
                chart_path = plot_iready_subject_dashboard_by_group(
                    scope_df.copy(), scope_label, folder, args.output_dir,
                    window_filter="Fall", group_name=group_name, group_def=group_def,
                    cfg=cfg, preview=hf.DEV_MODE
                )
                if chart_path:
                    chart_paths.append(chart_path)
            except Exception as e:
                print(f"  Error generating Section 2 chart for {scope_label} - {group_name} (Fall): {e}")
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
                try:
                    chart_path = plot_iready_subject_dashboard_by_group(
                        scope_df.copy(), scope_label, folder, args.output_dir,
                        window_filter="Fall", group_name=race_name, group_def=combined_group_def,
                        cfg=cfg, preview=hf.DEV_MODE
                    )
                    if chart_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"  Error generating Section 2 chart for {scope_label} - {race_name} (Fall): {e}")
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
        
        # Normalize grades (K -> 0, -1 -> pre-k)
        def normalize_grade_val(grade_val):
            if pd.isna(grade_val):
                return None
            grade_str = str(grade_val).strip().upper()
            if grade_str == "K" or grade_str == "KINDERGARTEN":
                return 0
            try:
                grade_num = int(float(grade_str))
                # -1 represents pre-k
                return grade_num
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
                    try:
                        print(f"  [Section 3] Generating chart for {scope_label} - Grade {g} - {subject_str} - Fall")
                        # Pass unfiltered scope_df - function filters internally (matches old flow)
                        chart_path = plot_iready_blended_dashboard(
                            scope_df.copy(), scope_label, folder, args.output_dir,
                            subject_str=subject_str, current_grade=int(g),
                            window_filter="Fall", cohort_year=anchor_year,
                            cfg=cfg, preview=hf.DEV_MODE
                        )
                        if chart_path:
                            chart_paths.append(chart_path)
                    except Exception as e:
                        print(f"  [Section 3] Error generating chart for {scope_label} - Grade {g} - {subject_str} (Fall): {e}")
                        if hf.DEV_MODE:
                            import traceback
                            traceback.print_exc()
                        continue
    
    # Use unfiltered scopes for Section 3 (matches old flow - each section filters internally)
    for scope_df, scope_label, folder in scopes:
        _run_scope_section3(scope_df.copy(), scope_label, folder)
    
    # Section 5: Growth Path Counts and Charts (DQC/Validation)
    print("\n" + "="*80)
    print("[Section 5] Running Growth Path DQC/Validation...")
    print("="*80)
    try:
        dqc_df = _prep_section5_dqc(iready_base.copy(), academicyear=2026)
        if not dqc_df.empty:
            print(f"[Section 5] Successfully prepared DQC data: {dqc_df.shape}")
            _print_section5_dqc(dqc_df)
        else:
            print("[Section 5] WARNING: No Fall 2026 data available for DQC")
            print("[Section 5] This may be expected if:")
            print("  - No data exists for academic year 2026")
            print("  - All data was filtered out by enrolled/most_recent_diagnostic filters")
            print("  - No data available after filters")
    except Exception as e:
        print(f"[Section 5] ERROR during DQC preparation: {e}")
        import traceback
        traceback.print_exc()
    print("="*80)
    
    # Section 5a: Fall 2026 Mid+ Progression Flags (Growth Path by Grade)
    print("\n[Section 5a] Generating Fall 2026 Mid+ Progression Flags...")
    for scope_df, scope_label, folder in scopes:
        # Prepare growth path data for this scope (scope_df is already filtered)
        scope_growth_df = _prep_growth_path_data(scope_df.copy(), academicyear=2026)
        
        if not scope_growth_df.empty:
            for subj in ["ELA", "Math"]:
                # Check if subject should be generated
                if not should_generate_subject(subj, chart_filters):
                    continue
                
                try:
                    dsub = scope_growth_df[scope_growth_df["subject"].astype(str).str.lower() == subj.lower()].copy()
                    if not dsub.empty:
                        path = _plot_mid_flag_stacked(dsub, subj, scope_label, folder, args.output_dir, cfg, preview=hf.DEV_MODE)
                        if path:
                            chart_paths.append(path)
                except Exception as e:
                    print(f"Error generating Section 5a chart for {scope_label} - {subj}: {e}")
                    if hf.DEV_MODE:
                        import traceback
                        traceback.print_exc()
                    continue
        else:
            print(f"[Section 5a] No Fall 2026 data available for {scope_label}")
    
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
    
    print(f"\n✅ Generated {len(chart_paths)} iReady Fall charts")
    if len(chart_paths) == 0:
        print(f"⚠️  WARNING: No charts were generated!")
        print(f"   - Data rows after filtering: {iready_base.shape[0]:,}")
        print(f"   - Scopes found: {len(scopes)}")
        print(f"   - Selected quarters: {selected_quarters}")
        print(f"   - Chart filters: {chart_filters}")
    
    return chart_paths

def generate_iready_fall_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    iready_data: list = None
) -> list:
    """
    Generate iReady Fall charts (wrapper function for Flask backend)
    
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
            'iready_fall.py',
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

