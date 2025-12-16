"""
iReady Spring chart generation module
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
import matplotlib.pyplot as plt

# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf

# Import utility modules
from .iready_data import (
    load_config_from_args,
    load_iready_data,
    get_scopes
)
from .iready_filters import (
    apply_chart_filters,
    should_generate_subject
)

# Import shared plotting functions from iready_charts
from .iready_charts import (
    _prep_section0_iready,
    _plot_section0_iready,
    _prep_mid_above_to_cers,
    _plot_mid_above_to_cers_faceted
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
    Main function to generate iReady Spring charts
    
    Args:
        iready_data: Optional list of dicts or DataFrame with iReady data.
                   If None, will load from CSV using args.data_dir
    """
    global chart_links, _chart_tracking_set
    chart_links = []
    _chart_tracking_set = set()
    
    parser = argparse.ArgumentParser(description='Generate iReady Spring charts')
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
    
    # Section 0: i-Ready vs CERS (Spring)
    print("\n[Section 0] Generating Spring i-Ready vs CERS...")
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
                path = _plot_section0_iready(scope_label, folder, args.output_dir, payload, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
        except Exception as e:
            print(f"Error generating Section 0 chart for {scope_label}: {e}")
            continue
    
    # Section 4: Spring i-Ready Mid/Above → % CERS Met/Exceeded (≤2025)
    print("\n[Section 4] Generating Spring i-Ready Mid/Above → % CERS Met/Exceeded...")
    for scope_df, scope_label, folder in scopes:
        try:
            # Check if any subjects should be generated
            subjects_to_generate = ["ELA", "Math"]
            if chart_filters and chart_filters.get("subjects"):
                subjects_to_generate = [s for s in subjects_to_generate if should_generate_subject(s, chart_filters)]
            
            if subjects_to_generate:
                path = _plot_mid_above_to_cers_faceted(scope_df.copy(), scope_label, folder, args.output_dir, cfg, preview=hf.DEV_MODE)
                if path:
                    chart_paths.append(path)
        except Exception as e:
            print(f"Error generating Section 4 chart for {scope_label}: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            continue
    
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
    
    print(f"\n✅ Generated {len(chart_paths)} iReady Spring charts")
    if len(chart_paths) == 0:
        print(f"⚠️  WARNING: No charts were generated!")
        print(f"   - Data rows after filtering: {iready_base.shape[0]:,}")
        print(f"   - Scopes found: {len(scopes)}")
        print(f"   - Chart filters: {chart_filters}")
    
    return chart_paths

def generate_iready_spring_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    iready_data: list = None
) -> list:
    """
    Generate iReady Spring charts (wrapper function for Flask backend)
    
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
            'iready_spring.py',
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

