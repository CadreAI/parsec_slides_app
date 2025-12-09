"""
STAR-specific helper functions and utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import shared constants from parent helper_functions
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))
import helper_functions as hf

# ---------------------------------------------------------------------
# STAR Column Definitions
# ---------------------------------------------------------------------
STAR_CAT_COL = "state_benchmark_achievement"
STAR_SCORE_COL = "unified_scale"
STAR_TIME_COL_OPTIONS = ["academicyear", "testwindow"]

# Short labels for legend (leader friendly)
STAR_LEVEL_LABELS = {
    "1 - Standard Not Met": "Standard Not Met",
    "2 - Standard Nearly Met": "Standard Nearly Met",
    "3 - Standard Met": "Standard Met",
    "4 - Standard Exceeded": "Standard Exceeded",
}

# ---------------------------------------------------------------------
# STAR-specific helper functions
# ---------------------------------------------------------------------

def normalize_star_subject(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with a 'subject' column standardized to 'Reading' or 'Math'.
    Uses activity_type. Leaves other values as pd.NA.
    """
    df = df_in.copy()
    
    def _to_subject(s):
        s = str(s).lower()
        if "math" in s:
            return "Math"
        if "read" in s:
            return "Reading"
        return pd.NA
    
    if "subject" not in df.columns or df["subject"].isna().all():
        df["subject"] = df.get("activity_type", pd.Series(index=df.index)).apply(_to_subject)
    
    df["subject"] = df["subject"].where(df["subject"].isin(["Reading", "Math"]))
    return df

def _build_time_label(df_sub: pd.DataFrame, time_col_options: list) -> pd.Series:
    """
    Build a categorical 'time_label' to use on the x-axis.
    Priority:
      1. termname (if it exists and is not all null)
      2. "<year> <window>" if year + testwindow exists
      3. first available col in time_col_options
    """
    # Case 1: 'termname' present and non-null
    if "termname" in df_sub.columns and df_sub["termname"].notna().any():
        return df_sub["termname"].astype(str)
    
    # Case 2: year + testwindow pattern
    if "year" in df_sub.columns and "testwindow" in df_sub.columns:
        return (
            df_sub["year"].astype(str).str.strip()
            + " "
            + df_sub["testwindow"].astype(str).str.strip()
        )
    
    # Fallback: try to combine first two available from time_col_options
    existing = [c for c in time_col_options if c in df_sub.columns]
    if len(existing) >= 2:
        return (
            df_sub[existing[0]].astype(str).str.strip()
            + " "
            + df_sub[existing[1]].astype(str).str.strip()
        )
    elif len(existing) == 1:
        return df_sub[existing[0]].astype(str)
    
    # Worst case: single constant bucket
    return pd.Series(["Time"] * len(df_sub), index=df_sub.index)

def _prepare_assessment_agg(
    df: pd.DataFrame,
    *,
    subject_str: str,
    window_filter: str,
    subject_col: str,
    window_col: str,
    cat_col: str,
    score_col: str,
    time_col_options: list,
    ordered_levels: list,
    high_group: list,
    low_group: list,
):
    """
    Shared engine for:
      - stacked % by category (top panel)
      - mean score (middle panel)
      - insight deltas (bottom panel)
    
    Returns:
        pct_df: long pct by (time_label, category)
        score_df: mean score per time_label
        insight_metrics: dict with deltas between last 2 windows
        time_order: sorted list of time_label
    """
    d = df.copy()
    
    # Filter to subject and specific window/season (Fall etc)
    d = d[
        (d[subject_col] == subject_str)
        & (d[window_col].astype(str).str.upper() == window_filter.upper())
    ].copy()
    
    # Build time_label for x-axis
    d["time_label"] = _build_time_label(d, time_col_options)
    
    # Count per category
    counts = d.groupby(["time_label", cat_col]).size().rename("n").reset_index()
    totals = d.groupby("time_label").size().rename("N_total").reset_index()
    pct_df = counts.merge(totals, on="time_label", how="left")
    pct_df["pct"] = 100 * pct_df["n"] / pct_df["N_total"]
    
    # Ensure all levels exist for every time_label (even 0%)
    all_idx = pd.MultiIndex.from_product(
        [pct_df["time_label"].unique(), ordered_levels], names=["time_label", cat_col]
    )
    pct_df = pct_df.set_index(["time_label", cat_col]).reindex(all_idx).reset_index()
    pct_df["pct"] = pct_df["pct"].fillna(0)
    pct_df["n"] = pct_df["n"].fillna(0)
    pct_df["N_total"] = pct_df["N_total"].ffill().bfill()
    
    # Chronological order for display
    time_order = sorted(pct_df["time_label"].unique().tolist())
    pct_df["time_label"] = pd.Categorical(
        pct_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    pct_df.sort_values(["time_label", cat_col], inplace=True)
    
    # Mean score per time window
    score_df = (
        d.groupby("time_label")[score_col].mean().rename("avg_score").reset_index()
    )
    score_df["time_label"] = pd.Categorical(
        score_df["time_label"],
        categories=time_order,
        ordered=True,
    )
    score_df.sort_values("time_label", inplace=True)
    
    # Insights: compare last two windows
    last_two = time_order[-2:] if len(time_order) >= 2 else time_order
    if len(last_two) == 2:
        t_prev, t_curr = last_two[0], last_two[1]
        
        def pct_for(group_list, t):
            return pct_df[
                (pct_df["time_label"] == t) & (pct_df[cat_col].isin(group_list))
            ]["pct"].sum()
        
        hi_curr = pct_for(high_group, t_curr)
        hi_prev = pct_for(high_group, t_prev)
        lo_curr = pct_for(low_group, t_curr)
        lo_prev = pct_for(low_group, t_prev)
        score_curr = float(
            score_df.loc[score_df["time_label"] == t_curr, "avg_score"].iloc[0]
        )
        score_prev = float(
            score_df.loc[score_df["time_label"] == t_prev, "avg_score"].iloc[0]
        )
        
        insight_metrics = {
            "t_prev": t_prev,
            "t_curr": t_curr,
            "hi_now": hi_curr,
            "hi_delta": hi_curr - hi_prev,
            "lo_now": lo_curr,
            "lo_delta": lo_curr - lo_prev,
            "score_now": score_curr,
            "score_delta": score_curr - score_prev,
        }
    else:
        insight_metrics = {
            "t_prev": None,
            "t_curr": time_order[-1] if time_order else None,
            "hi_now": None,
            "hi_delta": None,
            "lo_now": None,
            "lo_delta": None,
            "score_now": None,
            "score_delta": None,
        }
    
    return pct_df, score_df, insight_metrics, time_order

def prepare_star_agg(df: pd.DataFrame, subject_str: str, window_filter: str = "Fall"):
    """
    Wrapper for STAR assessment aggregation using _prepare_assessment_agg.
    """
    return _prepare_assessment_agg(
        df,
        subject_str=subject_str,
        window_filter=window_filter,
        subject_col="subject",  # STAR subject column
        window_col="testwindow",  # STAR seasonal window
        cat_col=STAR_CAT_COL,  # "state_benchmark_achievement"
        score_col=STAR_SCORE_COL,  # "unified_scale"
        time_col_options=STAR_TIME_COL_OPTIONS,
        ordered_levels=hf.STAR_ORDER,
        high_group=hf.STAR_HIGH_GROUP,
        low_group=hf.STAR_LOW_GROUP,
    )

def filter_small_groups(df: pd.DataFrame, group_col: str, min_n: int = 12) -> pd.DataFrame:
    """
    Exclude student groups with fewer than `min_n` students.
    Operates at the chart level (not global filtering).
    """
    group_counts = df[group_col].value_counts(dropna=False)
    valid_groups = group_counts[group_counts >= min_n].index
    return df[df[group_col].isin(valid_groups)].copy()

