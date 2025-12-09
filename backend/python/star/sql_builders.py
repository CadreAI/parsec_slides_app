"""
SQL query builders for BigQuery - STAR
"""
from typing import List, Optional, Dict


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
    filters: Optional[Dict] = None
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
    
    # Base excludes
    base_excludes = DEFAULT_STAR_EXCLUDES.copy()
    
    # Add dynamic excludes if provided
    if exclude_cols:
        base_excludes.extend(exclude_cols)
    
    # Convert to SQL list with backticks
    excludes_sql = ",\n        ".join([f"`{col}`" for col in base_excludes])
    
    # Build WHERE conditions
    where_conditions = []
    
    # Year filter (STAR uses AcademicYear column)
    if filters.get('years') and len(filters['years']) > 0:
        year_list = ', '.join(map(str, filters['years']))
        where_conditions.append(f"AcademicYear IN ({year_list})")
    else:
        # Default: last 3 years (matching the provided SQL pattern)
        where_conditions.append("""AcademicYear >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")
    
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

