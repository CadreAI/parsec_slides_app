"""
NWEA data loading and preparation utilities
"""

import json
import sys
from pathlib import Path
import pandas as pd
# Add parent directory to path to import helper_functions
sys.path.insert(0, str(Path(__file__).parent.parent))
import helper_functions as hf


def load_config_from_args(config_json_str):
    """Load config from JSON string passed via command line"""
    if not config_json_str or config_json_str == '{}':
        return {}
    try:
        return json.loads(config_json_str)
    except:
        return {}


def normalize_nwea_dataframe(df):
    """Normalize NWEA DataFrame (column names, mappings, etc.)"""
    nwea_base = df.copy()
    nwea_base.columns = nwea_base.columns.str.strip().str.lower()
    
    if "school" in nwea_base.columns and "schoolname" not in nwea_base.columns:
        nwea_base = nwea_base.rename(columns={"school": "schoolname"})
    
    prof_prof_map = {
        "Not Met": "Level 1 - Standard Not Met",
        "Nearly Met": "Level 2 - Standard Nearly Met",
        "Met": "Level 3 - Standard Met",
        "Exceeded": "Level 4 - Standard Exceeded",
        "Level 1": "Level 1 - Standard Not Met",
        "Level 2": "Level 2 - Standard Nearly Met",
        "Level 3": "Level 3 - Standard Met",
        "Level 4": "Level 4 - Standard Exceeded",
    }
    
    if "projectedproficiencylevel2" in nwea_base.columns:
        nwea_base["projectedproficiencylevel2"] = nwea_base["projectedproficiencylevel2"].replace(prof_prof_map)
    
    return nwea_base


def load_nwea_data(data_dir=None, nwea_data=None):
    """
    Load and normalize NWEA data from CSV file or use provided data
    
    Args:
        data_dir: Directory containing nwea_data.csv (optional if nwea_data provided)
        nwea_data: List of dicts or DataFrame with NWEA data (optional if data_dir provided)
    
    Returns:
        Normalized DataFrame
    """
    if nwea_data is not None:
        # Convert list of dicts to DataFrame if needed
        if isinstance(nwea_data, list):
            nwea_base = pd.DataFrame(nwea_data)
        elif isinstance(nwea_data, pd.DataFrame):
            nwea_base = nwea_data.copy()
        else:
            raise ValueError(f"nwea_data must be list of dicts or DataFrame, got {type(nwea_data)}")
        
        nwea_base = normalize_nwea_dataframe(nwea_base)
        print(f"NWEA data loaded from memory: {nwea_base.shape[0]:,} rows, {nwea_base.shape[1]} columns")
        return nwea_base
    
    # Fallback to CSV loading for backward compatibility
    if data_dir is None:
        raise ValueError("Either data_dir or nwea_data must be provided")
    
    csv_path = Path(data_dir) / "nwea_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}. Please run data ingestion first.")
    
    nwea_base = pd.read_csv(csv_path)
    nwea_base = normalize_nwea_dataframe(nwea_base)
    print(f"NWEA data loaded from CSV: {nwea_base.shape[0]:,} rows, {nwea_base.shape[1]} columns")
    return nwea_base


def get_scopes(nwea_base, cfg):
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
    # Include district if no schools are selected, or if explicitly requested
    selected_schools = cfg.get("selected_schools", [])
    include_district = cfg.get("include_district_scope", True)  # Default to True for backward compatibility
    
    if include_district:
        scopes.append((nwea_base.copy(), district_label, "_district"))
    
    # Only add school scopes if schoolname column exists and has data
    if "schoolname" in nwea_base.columns:
        available_schools = sorted(nwea_base["schoolname"].dropna().unique())
        
        # Filter schools based on selection if provided
        if selected_schools and len(selected_schools) > 0:
            # Normalize selected school names for matching
            selected_schools_normalized = [hf._safe_normalize_school_name(s, cfg) for s in selected_schools]
            schools_to_process = [
                school for school in available_schools
                if hf._safe_normalize_school_name(school, cfg) in selected_schools_normalized
            ]
            if len(schools_to_process) == 0:
                print(f"[Scope Filter] No matching schools found for selected schools: {selected_schools}")
        else:
            # If no schools selected, don't generate school-level charts
            schools_to_process = []
        
        for raw_school in schools_to_process:
            scope_df = nwea_base[nwea_base["schoolname"] == raw_school].copy()
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


def prep_nwea_for_charts(df, subject_str, window_filter="Fall"):
    """Filters and aggregates NWEA data for dashboard plotting"""
    # helper_functions already imported at top
    
    d = df.copy()
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()].copy()
    
    subj_norm = subject_str.strip().casefold()
    if "math" in subj_norm:
        d = d[d["course"].astype(str).str.contains("math k-12", case=False, na=False)].copy()
    elif "reading" in subj_norm:
        d = d[d["course"].astype(str).str.contains("reading", case=False, na=False)].copy()
    
    d = d[d["achievementquintile"].notna()].copy()
    d["year"] = pd.to_numeric(d["year"].astype(str).str[:4], errors="coerce")
    d["year_short"] = d["year"].apply(_short_year)
    d["time_label"] = d["testwindow"].astype(str).str.title() + " " + d["year_short"]
    
    if "teststartdate" in d.columns:
        d["teststartdate"] = pd.to_datetime(d["teststartdate"], errors="coerce")
    else:
        d["teststartdate"] = pd.NaT
    
    d.sort_values(["uniqueidentifier", "time_label", "teststartdate"], inplace=True)
    d = d.groupby(["uniqueidentifier", "time_label"], as_index=False).tail(1)
    
    # Check if we have any data after filtering
    if d.empty or len(d) == 0:
        # Return empty dataframes with proper structure
        pct_df = pd.DataFrame(columns=["time_label", "achievementquintile", "pct", "n", "N_total"])
        score_df = pd.DataFrame(columns=["time_label", "avg_score"])
        time_order = []
        metrics = {}
        return pct_df, score_df, metrics, time_order
    
    quint_counts = d.groupby(["time_label", "achievementquintile"]).size().rename("n").reset_index()
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Only create MultiIndex if we have time labels
    unique_time_labels = pct_df["time_label"].unique()
    if len(unique_time_labels) > 0:
        all_idx = pd.MultiIndex.from_product([unique_time_labels, hf.NWEA_ORDER], 
                                             names=["time_label", "achievementquintile"])
        pct_df = pct_df.set_index(["time_label", "achievementquintile"]).reindex(all_idx).reset_index()
        pct_df["pct"] = pct_df["pct"].fillna(0)
        pct_df["n"] = pct_df["n"].fillna(0)
        pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(lambda s: s.ffill().bfill())
    else:
        # No time labels found, return empty dataframe
        pct_df = pd.DataFrame(columns=["time_label", "achievementquintile", "pct", "n", "N_total"])
    
    score_df = d[["time_label", "testritscore"]].dropna(subset=["testritscore"]).groupby("time_label")["testritscore"].mean().rename("avg_score").reset_index()
    
    time_order = sorted(unique_time_labels.tolist()) if len(unique_time_labels) > 0 else []
    pct_df["time_label"] = pd.Categorical(pct_df["time_label"], categories=time_order, ordered=True)
    score_df["time_label"] = pd.Categorical(score_df["time_label"], categories=time_order, ordered=True)
    pct_df.sort_values(["time_label", "achievementquintile"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)
    
    # Metrics from last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    metrics = {}
    
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        
        def pct_for(bucket_list, tlabel):
            return pct_df[(pct_df["time_label"] == tlabel) & (pct_df["achievementquintile"].isin(bucket_list))]["pct"].sum()
        
        metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": pct_for(hf.NWEA_HIGH_GROUP, t_curr),
            "hi_delta": pct_for(hf.NWEA_HIGH_GROUP, t_curr) - pct_for(hf.NWEA_HIGH_GROUP, t_prev),
            "lo_now": pct_for(hf.NWEA_LOW_GROUP, t_curr),
            "lo_delta": pct_for(hf.NWEA_LOW_GROUP, t_curr) - pct_for(hf.NWEA_LOW_GROUP, t_prev),
            "score_now": float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0,
            "score_delta": (float(score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]) if len(score_df[score_df["time_label"] == t_curr]) > 0 else 0.0) - (float(score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]) if len(score_df[score_df["time_label"] == t_prev]) > 0 else 0.0),
        }
    
    return pct_df, score_df, metrics, time_order

