"""
STAR data loading and preparation utilities
"""

import json
import sys
from pathlib import Path
from typing import Optional
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


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name present in df (case-insensitive)."""
    if df is None or df.empty:
        # Still allow column detection on empty df
        pass
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def filter_star_subject_rows(df: pd.DataFrame, subject_str: str) -> pd.DataFrame:
    """
    STAR subject filter that also handles Spanish Reading.

    Behavior:
      - Math: keep rows where activity_type contains "math"
      - Reading:
          - base reading rows: activity_type contains "read"
          - exclude Language Usage: activity_type contains "language"
          - If subject_str indicates Spanish (contains "spanish"/"españ"/"espanol"), keep ONLY Spanish reading rows
          - Otherwise (plain Reading), keep ONLY non-Spanish reading rows
    """
    d = df.copy()

    activity_col = _pick_first_col(d, ["activity_type", "activitytype", "activity"])
    if not activity_col:
        return d.iloc[0:0].copy()

    activity = d[activity_col].astype(str)
    subj_norm = str(subject_str).strip().casefold()

    if "math" in subj_norm:
        return d[activity.str.contains("math", case=False, na=False)].copy()

    if "read" in subj_norm:
        base = d[activity.str.contains("read", case=False, na=False)].copy()
        # Exclude Language Usage from Reading
        base = base[~base[activity_col].astype(str).str.contains("language", case=False, na=False)].copy()

        if base.empty:
            return base

        # Identify Spanish reading rows
        spanish_mask = (
            base[activity_col].astype(str).str.contains("spanish", case=False, na=False)
            | base[activity_col].astype(str).str.contains("españ", case=False, na=False)
            | base[activity_col].astype(str).str.contains("espanol", case=False, na=False)
        )
        wants_spanish = ("spanish" in subj_norm) or ("españ" in subj_norm) or ("espanol" in subj_norm)
        if wants_spanish:
            return base[spanish_mask].copy()
        # Plain Reading: exclude Spanish reading so English + Spanish can be charted separately
        return base[~spanish_mask].copy()

    return d.iloc[0:0].copy()


def normalize_star_dataframe(df, cfg=None):
    """Normalize STAR DataFrame (column names, mappings, etc.)"""
    star_base = df.copy()
    star_base.columns = star_base.columns.str.strip().str.lower()
    
    # Map school column if needed
    if "school" in star_base.columns and "schoolname" not in star_base.columns:
        star_base = star_base.rename(columns={"school": "schoolname"})
    
    # Normalize subject column from activity_type if needed
    from .star_helper_functions import normalize_star_subject
    star_base = normalize_star_subject(star_base)
    
    # Kairos-specific school_name normalization (if needed)
    if cfg and "kairos" in str(cfg.get("partner_name", "")).lower():
        print("[INFO] Applying Kairos-specific school_name normalization")
        def _kairos_school_name(row):
            school = str(row.get("school_name", "")).strip()
            grade = row.get("studentgrade")
            # Match Innovative Scholars
            if "innovative scholars" in school.lower():
                return "Innovative Scholars"
            # Match Foundations → split by grade
            if "foundations" in school.lower():
                try:
                    g = float(grade)
                    if g < 6:
                        return "Foundations Academy"
                    elif g >= 6:
                        return "Leadership Academy"
                except Exception:
                    return "Foundations Academy"
            # Match Luminary
            if "luminary" in school.lower():
                return "Luminary Academy"
            return school
        
        star_base["school_name"] = star_base.apply(_kairos_school_name, axis=1)
        print(
            "[INFO] Kairos-specific school_name normalization complete — unique values:",
            star_base["school_name"].dropna().unique().tolist(),
        )
    
    return star_base


def load_star_data(data_dir=None, star_data=None, cfg=None):
    """
    Load and normalize STAR data from CSV file or use provided data
    
    Args:
        data_dir: Directory containing star_data.csv (optional if star_data provided)
        star_data: List of dicts or DataFrame with STAR data (optional if data_dir provided)
        cfg: Config dict for partner-specific normalization (optional)
    
    Returns:
        Normalized DataFrame
    """
    if star_data is not None:
        # Convert list of dicts to DataFrame if needed
        if isinstance(star_data, list):
            star_base = pd.DataFrame(star_data)
        elif isinstance(star_data, pd.DataFrame):
            star_base = star_data.copy()
        else:
            raise ValueError(f"star_data must be list of dicts or DataFrame, got {type(star_data)}")
        
        star_base = normalize_star_dataframe(star_base, cfg)
        print(f"STAR data loaded from memory: {star_base.shape[0]:,} rows, {star_base.shape[1]} columns")
        return star_base
    
    # Fallback to CSV loading for backward compatibility
    if data_dir is None:
        raise ValueError("Either data_dir or star_data must be provided")
    
    csv_path = Path(data_dir) / "star_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}. Please run data ingestion first.")
    
    star_base = pd.read_csv(csv_path)
    star_base = normalize_star_dataframe(star_base, cfg)
    print(f"STAR data loaded from CSV: {star_base.shape[0]:,} rows, {star_base.shape[1]} columns")
    return star_base


def get_scopes(star_base, cfg):
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
    include_district = cfg.get("include_district_scope", True)
    
    # For STAR: If only one school is selected, skip district charts (similar to iReady)
    if selected_schools and len(selected_schools) == 1:
        include_district = False
        print(f"[Scope Filter] Only one school selected - skipping district charts for STAR")
    
    # District scope
    if include_district:
        scopes.append((star_base.copy(), district_label, "_district"))
    
    # School scopes
    school_col = "school_name" if "school_name" in star_base.columns else "schoolname"
    
    if selected_schools and len(selected_schools) > 0:
        # Filter to selected schools only
        for raw_school in selected_schools:
            if raw_school not in star_base[school_col].values:
                print(f"[Warning] School '{raw_school}' not found in data, skipping")
                continue
            scope_df = star_base[star_base[school_col] == raw_school].copy()
            scope_label = hf._safe_normalize_school_name(raw_school, cfg)
            folder = scope_label.replace(" ", "_").replace("/", "_").replace("&", "and")
            scopes.append((scope_df, scope_label, folder))
    else:
        # If no schools selected (empty array), don't generate school-level charts
        # This happens when "District Only" mode is enabled
        print(f"[Scope Filter] No schools selected - skipping school-level charts (district only mode)")
    
    return scopes


def _short_year(y):
    """
    Return a YY-YY pair for an integer school academicyear (e.g., 2026 -> '25-26').
    If already a 'YYYY-YYYY' string, keep the last-two/last-two.
    """
    ys = str(y)
    if "-" in ys:
        a, b = ys.split("-", 1)
        return f"{a[-2:]}-{b[-2:]}"
    yi = int(float(ys))
    return f"{str(yi-1)[-2:]}-{str(yi)[-2:]}"


def prep_star_for_charts(df, subject_str, window_filter="Fall"):
    """
    Filters and aggregates STAR data for dashboard plotting.
    
    Rules:
      - Keep only the requested test window (ex: "Fall").
      - For Mathematics: keep rows where activity_type contains "math".
      - For Reading: keep rows where activity_type is "reading" or "reading (spanish)".
        exclude "language usage".
      - Drop rows with missing state_benchmark_achievement.
      - Keep the latest test per student per time_label using the most recent
        activity_completed_date.
      - Build time_label like "22-23 Fall".
      - Return:
            pct_df   = % by quintile per time window
            score_df = avg unified_scale per time window
            metrics  = delta metrics between last two windows
            time_order = ordered list of time labels
    """
    d = df.copy()
    
    # 1. restrict to requested test window
    d = d[d["testwindow"].astype(str).str.upper() == window_filter.upper()].copy()
    
    # 2. subject filtering (includes Spanish Reading preference)
    d = filter_star_subject_rows(d, subject_str)
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, []
    
    # 3. require valid benchmark achievement bucket
    d = d[d["state_benchmark_achievement"].notna()].copy()
    
    # 4. build "22-23 Fall" style label
    d["academicyear_short"] = d["academicyear"].apply(_short_year)
    d["time_label"] = (
        d["testwindow"].astype(str).str.title() + " " + d["academicyear_short"]
    )
    
    # 5. dedupe to latest attempt per student per time_label
    if "activity_completed_date" in d.columns:
        d["activity_completed_date"] = pd.to_datetime(
            d["activity_completed_date"], errors="coerce"
        )
    else:
        d["activity_completed_date"] = pd.NaT
    
    d.sort_values(
        ["student_state_id", "time_label", "activity_completed_date"], inplace=True
    )
    d = d.groupby(["student_state_id", "time_label"], as_index=False).tail(1)
    
    # 6. percent by benchmark achievement level
    quint_counts = (
        d.groupby(["time_label", "state_benchmark_achievement"])
        .size()
        .rename("n")
        .reset_index()
    )
    total_counts = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = quint_counts.merge(total_counts, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # ensure all benchmark levels exist for stacking in Low->High order
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), hf.STAR_ORDER],
        names=["time_label", "state_benchmark_achievement"],
    )
    pct_df = (
        pct_df.set_index(["time_label", "state_benchmark_achievement"])
        .reindex(all_idx)
        .reset_index()
    )
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df.groupby("time_label")["N_total"].transform(
        lambda s: s.ffill().bfill()
    )
    
    # 7. avg unified_scale per time_label
    score_df = (
        d[["time_label", "unified_scale"]]
        .dropna(subset=["unified_scale"])
        .groupby("time_label")["unified_scale"]
        .mean()
        .rename("avg_score")
        .reset_index()
    )
    
    # 8. enforce chronological order
    time_order = sorted(pct_df["time_label"].unique().tolist())
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    pct_df.sort_values(["time_label", "state_benchmark_achievement"], inplace=True)
    score_df.sort_values(["time_label"], inplace=True)
    
    # Ensure academicyear column is present in both pct_df and score_df
    if "academicyear" not in pct_df.columns:
        time_label_to_academicyear = d.drop_duplicates("time_label")[
            ["time_label", "academicyear"]
        ]
        pct_df = pct_df.merge(time_label_to_academicyear, on="time_label", how="left")
    
    if "academicyear" not in score_df.columns:
        time_label_to_academicyear = d.drop_duplicates("time_label")[
            ["time_label", "academicyear"]
        ]
        score_df = score_df.merge(
            time_label_to_academicyear, on="time_label", how="left"
        )
    
    # 9. insight metrics from last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two
        
        def pct_for(bucket_list, tlabel):
            tlabel_str = str(tlabel)
            return pct_df[
                (pct_df["time_label"].astype(str) == tlabel_str)
                & (pct_df["state_benchmark_achievement"].isin(bucket_list))
            ]["pct"].sum()
        
        hi_curr = pct_for(hf.STAR_HIGH_GROUP, t_curr)
        hi_prev = pct_for(hf.STAR_HIGH_GROUP, t_prev)
        lo_curr = pct_for(hf.STAR_LOW_GROUP, t_curr)
        lo_prev = pct_for(hf.STAR_LOW_GROUP, t_prev)
        
        # Add High only group (Standard Exceeded)
        high_curr = pct_for(["4 - Standard Exceeded"], t_curr)
        high_prev = pct_for(["4 - Standard Exceeded"], t_prev)
        
        metrics = {
            "high_delta": high_curr - high_prev,
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": float(
                score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
            ),
            "score_delta": float(
                score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
            )
            - float(
                score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
            ),
            "high_now": high_curr,
        }
    else:
        metrics = {
            "t_prev": None,
            "t_curr": time_order[-1] if time_order else None,
            "hi_now": None,
            "hi_delta": None,
            "lo_now": None,
            "lo_delta": None,
            "score_now": None,
            "score_delta": None,
            "high_now": None,
            "high_delta": None,
        }
    
    return pct_df, score_df, metrics, time_order

