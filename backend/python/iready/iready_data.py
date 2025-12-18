"""
iReady data loading and preparation utilities
"""

import json
import sys
from pathlib import Path
import pandas as pd
# Use iReady-specific helper utilities + styling
from . import helper_functions_iready as hf


def load_config_from_args(config_json_str):
    """Load config from JSON string passed via command line"""
    if not config_json_str or config_json_str == '{}':
        return {}
    try:
        return json.loads(config_json_str)
    except:
        return {}


def normalize_iready_dataframe(df):
    """Normalize iReady DataFrame (column names, mappings, etc.)"""
    iready_base = df.copy()
    iready_base.columns = iready_base.columns.str.strip().str.lower()
    
    # Normalize i-Ready placement labels
    if hasattr(hf, "IREADY_LABEL_MAP") and "relative_placement" in iready_base.columns:
        iready_base["relative_placement"] = iready_base["relative_placement"].replace(
            hf.IREADY_LABEL_MAP
        )
    
    # Map school column if needed
    if "school" in iready_base.columns and "schoolname" not in iready_base.columns:
        iready_base = iready_base.rename(columns={"school": "schoolname"})
    
    return iready_base


def load_iready_data(data_dir=None, iready_data=None):
    """
    Load and normalize iReady data from CSV file or use provided data
    
    Args:
        data_dir: Directory containing iready_data.csv (optional if iready_data provided)
        iready_data: List of dicts or DataFrame with iReady data (optional if data_dir provided)
    
    Returns:
        Normalized DataFrame
    """
    if iready_data is not None:
        # Convert list of dicts to DataFrame if needed
        if isinstance(iready_data, list):
            iready_base = pd.DataFrame(iready_data)
        elif isinstance(iready_data, pd.DataFrame):
            iready_base = iready_data.copy()
        else:
            raise ValueError(f"iready_data must be list of dicts or DataFrame, got {type(iready_data)}")
        
        iready_base = normalize_iready_dataframe(iready_base)
        print(f"iReady data loaded from memory: {iready_base.shape[0]:,} rows, {iready_base.shape[1]} columns")
        return iready_base
    
    # Fallback to CSV loading for backward compatibility
    if data_dir is None:
        raise ValueError("Either data_dir or iready_data must be provided")
    
    csv_path = Path(data_dir) / "iready_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}. Please run data ingestion first.")
    
    iready_base = pd.read_csv(csv_path)
    iready_base = normalize_iready_dataframe(iready_base)
    print(f"iReady data loaded from CSV: {iready_base.shape[0]:,} rows, {iready_base.shape[1]} columns")
    return iready_base


def get_scopes(iready_base, cfg):
    """Generate list of (scope_df, scope_label, folder_name) tuples"""
    district_name = cfg.get("district_name", [])
    # Safely get district label - handle empty list or non-list types
    if isinstance(district_name, list) and len(district_name) > 0:
        district_label = district_name[0]
    elif isinstance(district_name, str):
        district_label = district_name
    else:
        district_label = "Districtwide"
    
    scopes = []
    
    # Check if district scope should be included
    selected_schools = cfg.get("selected_schools", [])
    include_district = cfg.get("include_district_scope", True)  # Default to True for backward compatibility
    
    # For iReady: If only one school is selected, skip district charts
    if selected_schools and len(selected_schools) == 1:
        include_district = False
        print(f"[Scope Filter] Only one school selected - skipping district charts for iReady")
    
    if include_district:
        scopes.append((iready_base.copy(), district_label, "_district"))
    
    # Only add school scopes if a school column exists and has data.
    # iReady datasets vary: prefer schoolname, but fall back to other common columns.
    school_col = None
    for col in ["learning_center", "schoolname", "school_name", "school"]:
        if col in iready_base.columns:
            school_col = col
            break

    if school_col:
        available_schools = sorted(iready_base[school_col].dropna().unique())

        # Filter schools based on selection if provided
        if selected_schools and len(selected_schools) > 0:
            print(f"[Scope Filter] Selected schools from config: {selected_schools}")
            selected_schools_normalized = [hf._safe_normalize_school_name(s, cfg) for s in selected_schools]
            print(f"[Scope Filter] Normalized selected schools: {selected_schools_normalized}")

            # Build mapping of display name -> raw values (avoids subtle mismatches)
            display_to_raw = {}
            for raw in available_schools:
                disp = hf._safe_normalize_school_name(raw, cfg)
                display_to_raw.setdefault(disp, []).append(raw)

            # Match by display name OR raw case-insensitive / partial contains.
            selected_display_set = set(selected_schools_normalized)
            matched_displays = set()
            for disp, raws in display_to_raw.items():
                if disp in selected_display_set:
                    matched_displays.add(disp)
                    continue
                for raw in raws:
                    if any(
                        str(raw).lower() == str(sel).lower()
                        or str(sel).lower() in str(raw).lower()
                        or str(raw).lower() in str(sel).lower()
                        for sel in selected_schools
                    ):
                        matched_displays.add(disp)
                        break

            schools_to_process = []
            for disp in sorted(matched_displays):
                schools_to_process.extend(display_to_raw.get(disp, []))

            # De-dupe while preserving order
            seen = set()
            schools_to_process = [s for s in schools_to_process if not (s in seen or seen.add(s))]

            print(f"[Scope Filter] Matched {len(schools_to_process)} schools: {schools_to_process}")
            if len(schools_to_process) == 0:
                print(f"[Scope Filter] ⚠️  No matching schools found for selected schools: {selected_schools}")
                print(f"[Scope Filter]   Available schools in '{school_col}': {available_schools[:25]}{'...' if len(available_schools) > 25 else ''}")
        else:
            # If no schools selected, don't generate school-level charts
            schools_to_process = []

        for raw_school in schools_to_process:
            scope_df = iready_base[iready_base[school_col] == raw_school].copy()
            if len(scope_df) > 0:  # Only add if there's data
                scope_label = hf._safe_normalize_school_name(raw_school, cfg)
                scopes.append((scope_df, scope_label, scope_label.replace(" ", "_")))
    
    return scopes


def _short_year(y):
    """Convert year to YY-YY format"""
    ys = str(y)
    if "-" in ys:
        a, b = ys.split("-", 1)
        return f"{a[-2:]}-{b[-2:]}"
    try:
        yi = int(float(ys))
        return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"
    except:
        return str(ys)


def prep_iready_for_charts(df, subject_str, window_filter="Fall"):
    """Filters and aggregates iReady data for dashboard plotting"""
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()].copy()
    
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("math", case=False, na=False)].copy()
    elif "ela" in subj_norm:
        d = d[d["subject"].astype(str).str.contains("ela", case=False, na=False)].copy()
    
    d = d[d["domain"] == "Overall"].copy()
    d = d[d["enrolled"] == "Enrolled"].copy()
    d = d[d["relative_placement"].notna()].copy()
    
    # Normalize i-Ready placement labels
    if hasattr(hf, "IREADY_LABEL_MAP"):
        d["relative_placement"] = d["relative_placement"].replace(hf.IREADY_LABEL_MAP)
    
    # Ensure academic year is numeric
    d["year"] = pd.to_numeric(d["academicyear"], errors="coerce")
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
    
    # Dedupe to latest completion per student/year
    if "completion_date" in d.columns:
        d["completion_date"] = pd.to_datetime(d["completion_date"], errors="coerce")
        sort_col = "completion_date"
    elif "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
        sort_col = "teststartdate"
    else:
        d["completion_date"] = pd.NaT
        sort_col = "completion_date"
    
    d.sort_values(["uniqueidentifier", "time_label", sort_col], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)
    
    # Check if we have any data after filtering
    if d.empty or len(d) == 0:
        # Return empty dataframes with proper structure
        pct_df = pd.DataFrame(columns=["time_label", "relative_placement", "pct", "n", "N_total"])
        score_df = pd.DataFrame(columns=["time_label", "avg_score"])
        time_order = []
        metrics = {}
        return pct_df, score_df, metrics, time_order
    
    # Percent by placement
    quint_counts = (
        d.groupby(["time_label", "relative_placement"]).size().rename("n").reset_index()
    )
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Ensure all quintiles exist for stacking
    unique_time_labels = pct_df["time_label"].unique()
    if len(unique_time_labels) > 0:
        all_idx = pd.MultiIndex.from_product(
            [unique_time_labels, hf.IREADY_ORDER],
            names=["time_label", "relative_placement"],
        )
        pct_df = (
            pct_df.set_index(["time_label", "relative_placement"])
            .reindex(all_idx)
            .reset_index()
        )
        pct_df["pct"] = pct_df["pct"].fillna(0)
        pct_df["n"] = pct_df["n"].fillna(0)
        pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(
            lambda s: s.ffill().bfill()
        )
    else:
        pct_df = pd.DataFrame(columns=["time_label", "relative_placement", "pct", "n", "N_total"])
    
    # Average scale score
    score_df = (
        d[["time_label", "scale_score"]]
        .dropna(subset=["scale_score"])
        .groupby("time_label")["scale_score"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )
    
    # Chronological order
    time_order = sorted(unique_time_labels.tolist()) if len(unique_time_labels) > 0 else []
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"], categories=time_order, ordered=True
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"], categories=time_order, ordered=True
    )
    pct_df.sort_values(["time_label", "relative_placement"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)
    
    # Year map for filtering
    year_map = d.drop_duplicates("time_label")[["time_label", "year"]]
    pct_df = pct_df.merge(year_map, on="time_label", how="left")
    score_df = score_df.merge(year_map, on="time_label", how="left")
    
    # Insight metrics from last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    metrics = {}
    
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        
        def pct_for(buckets, tlabel):
            # Convert Categorical to string for reliable comparison
            tlabel_str = str(tlabel)
            mask = (pct_df["time_label"].astype(str) == tlabel_str) & (pct_df["relative_placement"].isin(buckets))
            return pct_df[mask]["pct"].sum()
        
        hi_curr = pct_for(hf.IREADY_HIGH_GROUP, t_curr)
        hi_prev = pct_for(hf.IREADY_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.IREADY_LOW_GROUP, t_curr)
        lo_prev = pct_for(hf.IREADY_LOW_GROUP, t_prev)
        high_curr = pct_for(["Mid/Above"], t_curr)
        high_prev = pct_for(["Mid/Above"], t_prev)
        
        metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": float(
                score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
            ) if len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0,
            "score_delta": (
                float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0])
                if len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0
            ) - (
                float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0])
                if len(score_df[score_df["time_label"] == t_prev]) > 0 else 0.0
            ),
            "high_now": high_curr,
            "high_delta": high_curr - high_prev,
        }
    
    return pct_df, score_df, metrics, time_order

