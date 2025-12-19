"""
SQL query builders for BigQuery (iReady)

Policy:
- Always filter by years in SQL.
- If schools are selected, also filter by schools in SQL to reduce result size.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime


def sql_iready(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    year_column: Optional[str] = None
) -> str:
    """
    Build SQL query for iReady data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of columns to exclude from the query
        filters: Optional filters dict with years (other filters applied in Python)
        year_column: Optional year column name ('Year' or 'AcademicYear'). If None, uses COALESCE.
    
    Returns:
        SQL query string with year filtering in SQL
    """
    filters = filters or {}

    def _sql_escape(s: str) -> str:
        return str(s).replace("'", "\\'")

    def _sql_like_tokens_any(lower_expr: str, phrases: list[str]) -> Optional[str]:
        """
        Token-based fuzzy LIKE matcher.
        - For each phrase (e.g. "ADDAMS ELEMENTARY SCHOOL"), build an AND of meaningful tokens.
        - Combine phrases with OR.
        This avoids brittle exact-substring matching when table values omit suffixes like "School".
        """
        import re

        stop = {
            "school",
            "elementary",
            "middle",
            "high",
            "academy",
            "charter",
            "k8",
            "k-8",
            "k12",
            "k-12",
            "of",
            "the",
        }

        clauses: list[str] = []
        for p in phrases or []:
            if p is None:
                continue
            raw = str(p).strip().lower()
            if not raw:
                continue
            # Normalize punctuation/whitespace
            raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            tokens = [t for t in raw.split() if t and t not in stop and len(t) >= 3]
            if not tokens:
                tokens = [raw]
            ands = [f"{lower_expr} LIKE '%{_sql_escape(t)}%'" for t in tokens]
            clauses.append("(" + " AND ".join(ands) + ")")

        if not clauses:
            return None
        return "(" + " OR ".join(clauses) + ")"

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
    
    # Base excludes from the i-Ready ingestion logic
    base_excludes = [
        # Demographics (less robust than CALPADS)
        "Hispanic_or_Latino",
        "Race",
        "English_Language_Learner",
        "Special_Education",
        "Economically_Disadvantaged",
        "Migrant",
        
        # Lexile / Quantile
        "Measure",
        "Range",
        
        # Redundant columns (iReady + Parsec)
        "Annual_Typical_Growth_Percent",
        "Annual_Stretch_Growth_Percent",
        "Relative_Tier_Placement",
        "Relative_5Tier_Placement",
        
        # Duplicate creators
        "TestView",
        "Match",
        "MatchYear",
        "MatchWindow",
        "WindowOrder",
    ]
    
    # Add dynamic excludes if provided
    if exclude_cols:
        base_excludes.extend(exclude_cols)
    
    # Convert to SQL list with backticks (required for reserved keywords like "Range")
    # Backtick all column names to handle reserved keywords
    excludes_sql = ",\n        ".join([f"`{col}`" for col in base_excludes])
    
    # Build WHERE conditions
    where_conditions = ["Enrolled = 'Enrolled'"]
    
    # Year filter (iReady might use AcademicYear or Year column)
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
        # Default: last 3 years (matching the data ingestion script pattern)
        where_conditions.append(f"""{year_col_name} >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")

    # Optional school filter pushdown.
    # If districtwide is included, do NOT push down school-name filtering:
    # district aggregate needs all schools; school selection happens later in chart generation.
    include_districtwide = bool(filters.get("include_districtwide"))
    schools = [] if include_districtwide else (filters.get("schools") or [])
    if not isinstance(schools, list):
        schools = []
    if schools:
        school_col = _pick_col(["SchoolName", "School_Name", "School", "learning_center", "Learning_Center"])
        if school_col:
            school_expr = f"LOWER(CAST({school_col} AS STRING))"
            like_clause = _sql_like_tokens_any(school_expr, schools) or _sql_like_any(school_expr, schools)
            if like_clause:
                where_conditions.append(like_clause)
    
    where_clause = " AND ".join(where_conditions)
    
    return f"""
    SELECT DISTINCT * 
      EXCEPT(
        {excludes_sql}
      )
    FROM `{table_id}`
    WHERE {where_clause}
    """

