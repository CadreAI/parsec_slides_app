"""
iReady chart generation router - directs to Fall/Winter/Spring modules based on quarter selection
"""

import json
from . import helper_functions_iready as hf

# Chart tracking for CSV generation (used by legacy modules)
chart_links = []
_chart_tracking_set = set()


def generate_iready_charts(
    partner_name: str,
    output_dir: str,
    config: dict = None,
    chart_filters: dict = None,
    data_dir: str = './data',
    iready_data: list = None
) -> list:
    """
    Generate iReady charts (router function - directs to Fall, Winter, or Spring modules based on quarter selection)
    
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
    
    chart_filters_check = cfg.get('chart_filters', {})
    selected_quarters = chart_filters_check.get("quarters", [])
    all_chart_paths = []
    
    # Normalize selected_quarters to handle both list and single value
    if isinstance(selected_quarters, str):
        selected_quarters = [selected_quarters]
    elif not isinstance(selected_quarters, list):
        selected_quarters = []
    
    # If no quarters are explicitly selected, default to Fall for backward compatibility
    if not selected_quarters:
        selected_quarters = ["Fall"]
        print("\n[iReady Router] No quarters specified in chart_filters - defaulting to Fall")
    
    normalized_quarters = [str(q).lower() for q in selected_quarters]
    has_winter = "winter" in normalized_quarters
    has_fall = "fall" in normalized_quarters
    has_spring = "spring" in normalized_quarters
    
    print(f"\n[iReady Router] Selected quarters from chart_filters: {selected_quarters}")
    print(f"\n[iReady Router] Normalized quarters: {normalized_quarters}")
    print(f"[iReady Router] has_winter={has_winter}, has_fall={has_fall}, has_spring={has_spring}")
    
    # Route to Winter module if Winter is selected
    if has_winter:
        # Legacy MOY script runner (subprocess)
        from .iready_moy_runner import generate_iready_winter_charts
        print("\n[iReady Router] Winter detected - routing to iready_moy.py (runner)...")
        try:
            winter_charts = generate_iready_winter_charts(
                partner_name=partner_name,
                output_dir=output_dir,
                config=cfg,
                chart_filters=chart_filters_check,
                data_dir=data_dir,
                iready_data=iready_data
            )
            if winter_charts:
                all_chart_paths.extend(winter_charts)
                print(f"[iReady Router] Generated {len(winter_charts)} Winter charts")
        except Exception as e:
            print(f"[iReady Router] Error generating Winter charts: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            # If ONLY Winter is selected, raise so the task surfaces the real root cause
            if has_winter and not has_fall and not has_spring:
                raise
        
        # If ONLY Winter is selected (no Fall, no Spring), return early
        if has_winter and not has_fall and not has_spring:
            print("\n[iReady Router] Only Winter selected - returning early, skipping Fall/Spring chart generation.")
            return all_chart_paths
    
    # Route to EOY runner if Spring is selected (covers Spring-only and Winter+Spring)
    if has_spring:
        from .iready_eoy_runner import generate_iready_eoy_charts
        print("\n[iReady Router] Spring detected - routing to iready_eoy.py (runner)...")
        try:
            eoy_charts = generate_iready_eoy_charts(
                partner_name=partner_name,
                output_dir=output_dir,
                config=cfg,
                chart_filters=chart_filters_check,
                data_dir=data_dir,
                iready_data=iready_data
            )
            if eoy_charts:
                all_chart_paths.extend(eoy_charts)
                print(f"[iReady Router] Generated {len(eoy_charts)} EOY charts")
        except Exception as e:
            print(f"[iReady Router] Error generating EOY charts: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
            # If ONLY Spring is selected, raise so the task surfaces the real root cause
            if has_spring and not has_fall and not has_winter:
                raise
        
        # If ONLY Spring is selected (no Fall, no Winter), return early
        if has_spring and not has_fall and not has_winter:
            print("\n[iReady Router] Only Spring selected - returning early, skipping Fall chart generation.")
            return all_chart_paths
    
    # Route to Fall module if Fall is selected
    if has_fall:
        # Legacy BOY script runner (subprocess)
        from .iready_boy_runner import generate_iready_fall_charts
        print("\n[iReady Router] Fall detected - routing to iready_boy.py (runner)...")
        try:
            fall_charts = generate_iready_fall_charts(
                partner_name=partner_name,
                output_dir=output_dir,
                config=cfg,
                chart_filters=chart_filters_check,
                data_dir=data_dir,
                iready_data=iready_data
            )
            if fall_charts:
                all_chart_paths.extend(fall_charts)
                print(f"[iReady Router] Generated {len(fall_charts)} Fall charts")
        except Exception as e:
            print(f"[iReady Router] Error generating Fall charts: {e}")
            if hf.DEV_MODE:
                import traceback
                traceback.print_exc()
    else:
        if not has_winter and not has_spring:
            print("\n[iReady Router] No Fall, Spring, or Winter selected - skipping chart generation.")
    
    return all_chart_paths

