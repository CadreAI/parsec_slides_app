"""
SQL query builders for BigQuery
Note: iReady queries include year filtering in SQL, but other filters (grades, schools) are done in Python
"""
from typing import List, Optional, Dict
from datetime import datetime


def sql_iready(
    table_id: str,
    exclude_cols: Optional[List[str]] = None,
    filters: Optional[Dict] = None
) -> str:
    """
    Build SQL query for iReady data
    
    Args:
        table_id: BigQuery table ID (project.dataset.table)
        exclude_cols: Optional list of columns to exclude from the query
        filters: Optional filters dict with years (other filters applied in Python)
    
    Returns:
        SQL query string with year filtering in SQL
    """
    filters = filters or {}
    
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
    # Use COALESCE to handle both column names
    year_column = "COALESCE(AcademicYear, Year)"
    if filters.get('years') and len(filters['years']) > 0:
        year_list = ', '.join(map(str, filters['years']))
        where_conditions.append(f"{year_column} IN ({year_list})")
    else:
        # Default: last 3 years (matching the data ingestion script pattern)
        where_conditions.append(f"""{year_column} >= (
            CASE
                WHEN EXTRACT(MONTH FROM CURRENT_DATE()) >= 7
                    THEN EXTRACT(YEAR FROM CURRENT_DATE()) + 1
                ELSE EXTRACT(YEAR FROM CURRENT_DATE())
            END
        ) - 3""")
    
    where_clause = " AND ".join(where_conditions)
    
    return f"""
    SELECT DISTINCT * 
      EXCEPT(
        {excludes_sql}
      )
    FROM `{table_id}`
    WHERE {where_clause}
    """

