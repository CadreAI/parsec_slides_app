"""
SQL query builders for BigQuery - STAR

Policy:
- Always filter by years in SQL.
- If schools are selected, also filter by schools in SQL to reduce result size.
"""
from typing import List, Optional, Dict, Any


# Default exclude columns for STAR
DEFAULT_STAR_EXCLUDES = [
    "Teacher_State_ID",
    "Teacher_Identifier",
    "Lexile_Score",
    "Lexile_Range",
    "Quantile_Measure",
    "Quantile_Range",
    "WindowOrder",
    "Match",
    "MatchYear",
    "MatchWindow",
    "MatchOrder",
]


def sql_star(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    year_column: Optional[str] = None
) -> str:
    """
    Build SQL query for STAR data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of additional columns to exclude
        filters: Optional filters dict (currently only years supported in SQL)
    
    Returns:
        SQL query string
    """
    filters = filters or {}

    def _sql_escape(s: str) -> str:
        return str(s).replace("'", "\\'")

    def _sql_like_any(lower_expr: str, needles: list[str]) -> Optional[str]:
        pats = [n.strip().lower() for n in needles if n is not None and str(n).strip()]
        if not pats:
            return None
        ors = [f"{lower_expr} LIKE '%{_sql_escape(p)}%'" for p in pats]
        return "(" + " OR ".join(ors) + ")"

    available_cols = [str(c).lower() for c in (filters.get("available_columns") or [])]
    def _pick_col(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in available_cols:
                return c
        return None
    
    # Base excludes
    base_excludes = DEFAULT_STAR_EXCLUDES.copy()
    
    # Add dynamic excludes if provided
    if exclude_cols:
        base_excludes.extend(exclude_cols)
    
    # Convert to SQL list with backticks
    excludes_sql = ",\n        ".join([f"`{col}`" for col in base_excludes])
    
    # Build WHERE conditions
    where_conditions = []
    
    # Year filter (STAR might use AcademicYear or Year column)
    # Use detected column or COALESCE as fallback
    if year_column:
        # Use detected column name
        year_col_name = year_column
    else:
        # Fallback to COALESCE (will fail if neither column exists, but that's expected)
        year_col_name = "COALESCE(AcademicYear, Year)"
    
    if filters.get('years') and len(filters['years']) > 0:
        year_list = ', '.join(map(str, filters['years']))
        where_conditions.append(f"{year_col_name} IN ({year_list})")
    else:
        # Default: last 3 years (matching the provided SQL pattern)
        where_conditions.append(f"""{year_col_name} >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")

    # Optional school filter pushdown
    schools = filters.get("schools") or []
    if not isinstance(schools, list):
        schools = []
    if schools:
        school_col = _pick_col(["School_Name", "SchoolName", "School", "learning_center", "Learning_Center"])
        if school_col:
            school_expr = f"LOWER(CAST({school_col} AS STRING))"
            like_clause = _sql_like_any(school_expr, schools)
            if like_clause:
                where_conditions.append(like_clause)
    
    where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
    
    return f"""
    SELECT DISTINCT * 
      EXCEPT(
        -- Comment any line below that you want to keep in your analysis
        {excludes_sql}
      )
    FROM
      `{table_id}`
    {where_clause}
    """

